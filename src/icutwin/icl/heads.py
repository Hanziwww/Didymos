"""Heads, losses and scores of every label type (plan Phase G, Architecture 6), and the base-rate baseline the
learning confirmation compares against (the context's own label distribution, on the same rows).

* binary and in-hospital death: a Bernoulli logit;
* ordinal: logits over at most 10 classes (the episode's k + 1 masked in);
* regression and forecasts: a bar distribution (TabPFN's Riemann head) over the standardized, squashed value
  ``u = symlog((y - median) / scale)`` of ``batch.standardize``, on a fixed grid of ``bins`` over [-8, 8];
* time to event and multi-horizon: a discrete-time competing-risk hazard over ``GRID_HOURS`` (bins of
  (e_{j-1}, e_j], the last one open), with the outcomes none / event / death / discharge per bin; a censored row
  contributes its survival through the bins before its censoring bin; a multi-horizon label is the event's
  cumulative incidence at the horizon (uniform within a bin), so one head serves every horizon consistently.

Every loss is a negative log-likelihood (a proper score). An episode's loss is the mean over its valid targets (a
forecast row: the mean over its observed cells) and the batch's the mean over its episodes.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

GRID_HOURS = tuple(float(x) for x in np.geomspace(0.5, 168.0, 24))
N_BINS = len(GRID_HOURS) + 1                 # the last bin is open (beyond 7 days)
OUTCOMES = 4                                 # none, event, death, discharge
BAR_LO, BAR_HI = -8.0, 8.0
EPS = 1e-6


def bar_edges(bins: int, device=None) -> torch.Tensor:
    return torch.linspace(BAR_LO, BAR_HI, bins + 1, device=device)


def bar_index(u: torch.Tensor, bins: int) -> torch.Tensor:
    w = (BAR_HI - BAR_LO) / bins
    return ((u - BAR_LO) / w).floor().clamp(0, bins - 1).long()


def bar_nll(logits: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """-log density of ``u`` under the bar distribution (per row)."""
    bins = logits.shape[-1]
    w = (BAR_HI - BAR_LO) / bins
    lp = torch.log_softmax(logits.float(), -1)
    return -lp.gather(-1, bar_index(u, bins).unsqueeze(-1)).squeeze(-1) + math.log(w)


def bar_crps(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """CRPS of the bar distribution at ``u`` (per row): E|X - u| - E|X - X'| / 2 on the bins' centres (exact up to
    the within-bin spread, at most a bin's width / 6)."""
    bins = probs.shape[-1]
    e = bar_edges(bins, probs.device)
    c = (e[:-1] + e[1:]) / 2
    a = (probs * (c - u.unsqueeze(-1)).abs()).sum(-1)
    pair = (c[:, None] - c[None, :]).abs()
    b = torch.einsum("nk,kl,nl->n", probs, pair, probs)
    return a - b / 2


def hazard_bins(time_h: torch.Tensor) -> torch.Tensor:
    edges = torch.tensor(GRID_HOURS, device=time_h.device, dtype=time_h.dtype)
    return torch.bucketize(time_h.clamp_min(0), edges, right=False)


def hazard_nll(logits: torch.Tensor, time_h: torch.Tensor, cause: torch.Tensor) -> torch.Tensor:
    """NLL of (time, cause) under a discrete competing-risk hazard ``logits`` [n, N_BINS, 4] (per row); ``cause``
    0 event, 1 death, 2 discharge, -1 censored at ``time_h``."""
    lp = torch.log_softmax(logits.float(), -1)
    j = hazard_bins(time_h)
    surv = torch.cumsum(lp[..., 0], -1)
    before = torch.cat([torch.zeros_like(surv[:, :1]), surv[:, :-1]], 1).gather(1, j[:, None]).squeeze(1)
    c = cause.clamp_min(0) + 1
    at = lp[torch.arange(len(j), device=j.device), j, c]
    return -(before + torch.where(cause >= 0, at, torch.zeros_like(at)))


def cumulative_incidence(logits: torch.Tensor, hours: torch.Tensor) -> torch.Tensor:
    """P(event by h) [n, H] for horizons ``hours`` [n, H] (uniform within a bin; the open bin spans 168-336 h)."""
    p = torch.softmax(logits.float(), -1)
    surv_after = torch.cumprod(p[..., 0], -1)
    surv_before = torch.cat([torch.ones_like(surv_after[:, :1]), surv_after[:, :-1]], 1)
    mass = surv_before * p[..., 1]                                       # [n, B] event mass per bin
    cum = torch.cumsum(mass, -1)
    lo = torch.tensor((0.0,) + GRID_HOURS, device=logits.device)
    hi = torch.tensor(GRID_HOURS + (2 * GRID_HOURS[-1],), device=logits.device)
    h = torch.nan_to_num(hours.float(), nan=0.0)
    j = hazard_bins(h).clamp_max(N_BINS - 1)
    frac = ((h - lo[j]) / (hi[j] - lo[j])).clamp(0, 1)
    prev = torch.where(j > 0, cum.gather(1, (j - 1).clamp_min(0)), torch.zeros_like(h))
    return (prev + mass.gather(1, j) * frac).clamp(EPS, 1 - EPS)


class Heads(nn.Module):
    """The output heads on the test rows' final states ``z`` [n, d]."""

    def __init__(self, d: int, bins: int, d_chan: int, max_classes: int = 10):
        super().__init__()
        self.bins = bins
        self.binary = nn.Linear(d, 1)
        self.ordinal = nn.Linear(d, max_classes)
        self.regression = nn.Linear(d, bins)
        self.hazard = nn.Linear(d, N_BINS * OUTCOMES)
        self.fc_step = nn.Embedding(256, d)
        self.fc_chan = nn.Linear(d_chan, d)
        self.forecast = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, bins))

    def hazard_logits(self, z):
        return self.hazard(z).view(len(z), N_BINS, OUTCOMES)

    def forecast_logits(self, z, chan, n_steps: int):
        """[n, H, C, bins] from row states ``z`` [n, d] and the forecast channels' embeddings ``chan`` [n, C, d_chan]."""
        n, c = chan.shape[:2]
        step = self.fc_step(torch.arange(n_steps, device=z.device))                      # [H, d]
        q = step[None, :, None, :] + self.fc_chan(chan)[:, None, :, :]                      # [n, H, C, d]
        x = torch.cat([z[:, None, None, :].expand(n, n_steps, c, z.shape[-1]), q], -1)
        return self.forecast(x)


def row_losses(heads: Heads, z: torch.Tensor, b: dict, fc_chan: torch.Tensor | None) -> tuple[torch.Tensor, dict]:
    """Per test row: (loss [n] with NaN where no target applies, outputs by kind for scoring)."""
    from icutwin.icl.batch import LABELS

    n = len(z)
    ep = b["te_ep"]
    kind = b["ep_label"][ep]
    loss = torch.full((n,), float("nan"), device=z.device)
    out: dict = {}
    valid = b["te_valid"]
    y = b["te_y"]
    for name in LABELS:
        m = kind == LABELS.index(name)
        if not m.any():
            continue
        zi = z[m]
        if name in ("binary", "hosp_death"):
            logit = heads.binary(zi).squeeze(-1).float()
            li = nn.functional.binary_cross_entropy_with_logits(logit, torch.nan_to_num(y[m]), reduction="none")
            out[name] = logit
        elif name == "ordinal":
            logits = heads.ordinal(zi).float()
            k = b["ep_ncls"][ep[m]]
            cls = torch.arange(logits.shape[-1], device=z.device)
            logits = logits.masked_fill(cls[None, :] >= k[:, None], -1e4)
            target = torch.nan_to_num(y[m]).long().clamp(0, logits.shape[-1] - 1)
            li = nn.functional.cross_entropy(logits, target, reduction="none")
            out[name] = logits
        elif name == "regression":
            logits = heads.regression(zi)
            li = bar_nll(logits, torch.nan_to_num(y[m]))
            out[name] = logits
        elif name == "tte":
            logits = heads.hazard_logits(zi)
            li = hazard_nll(logits, b["te_time"][m], b["te_cause"][m])
            out[name] = logits
        elif name == "multihorizon":
            logits = heads.hazard_logits(zi)
            cif = cumulative_incidence(logits, b["ep_hz"][ep[m]])
            mh = b["te_mh"][m]
            bce = nn.functional.binary_cross_entropy(cif, b["te_yh"][m].clamp(0, 1), reduction="none")
            li = (bce * mh).sum(1) / mh.sum(1).clamp_min(1)
            valid = valid.clone()
            valid[m] = valid[m] & mh.any(1)
            out[name] = cif
        else:                                                            # forecast
            yf, mf = b["te_yf"][m], b["te_mf"][m]
            logits = heads.forecast_logits(zi, fc_chan[ep[m]], yf.shape[1])
            nll = bar_nll(logits, yf)
            li = (nll * mf).sum((1, 2)) / mf.sum((1, 2)).clamp_min(1)
            valid = valid.clone()
            valid[m] = valid[m] & mf.flatten(1).any(1)
            out[name] = logits
        loss[m] = torch.where(valid[m], li, torch.full_like(li, float("nan")))
    return loss, out


def episode_mean(loss: torch.Tensor, ep: torch.Tensor, n_ep: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(per-episode mean of the rows' losses over finite ones [B], whether an episode has any [B])."""
    ok = torch.isfinite(loss)
    s = torch.zeros(n_ep, device=loss.device).index_add(0, ep, torch.where(ok, loss, torch.zeros_like(loss)))
    c = torch.zeros(n_ep, device=loss.device).index_add(0, ep, ok.float())
    return s / c.clamp_min(1), c > 0


# -- the base-rate baseline: the context's own label distribution ------------------------------------------------
def baseline_losses(b: dict, bins: int) -> torch.Tensor:
    """Per test row: the loss of the context's own label distribution (the same rows and scores as the model's):
    the prevalence or class shares (add-half smoothing), the histogram of the context's values on the bar grid
    (half a count per bin), the discrete-time competing-risk hazards estimated from the context's rows."""
    from icutwin.icl.batch import LABELS

    dev = b["te_y"].device
    n, n_ep = len(b["te_y"]), int(b["n_episodes"])
    loss = torch.full((n,), float("nan"), device=dev)
    tr_ep, lab = b["tr_ep"], b["tr_lab"]
    tr_valid = lab[:, 0] > 0
    for e in range(n_ep):
        rows = b["te_ep"] == e
        if not rows.any():
            continue
        kind = LABELS[int(b["ep_label"][e])]
        ctx = (tr_ep == e) & tr_valid
        y, valid = b["te_y"][rows], b["te_valid"][rows]
        if kind in ("binary", "hosp_death"):
            p = (lab[ctx, 8].sum() + 0.5) / (ctx.sum() + 1.0)
            li = -(torch.nan_to_num(y) * torch.log(p) + (1 - torch.nan_to_num(y)) * torch.log(1 - p))
        elif kind == "ordinal":
            k = int(b["ep_ncls"][e])
            cnt = lab[ctx, 9:9 + k].sum(0) + 0.5
            lp = torch.log(cnt / cnt.sum())
            li = -lp[torch.nan_to_num(y).long().clamp(0, k - 1)]
        elif kind == "regression":
            li = bar_nll(_histogram_logits(lab[ctx, 19], bins).expand(int(rows.sum()), bins), torch.nan_to_num(y))
        elif kind in ("tte", "multihorizon"):
            logits = _hazard_from_rows(lab[ctx]).expand(int(rows.sum()), N_BINS, OUTCOMES)
            if kind == "tte":
                li = hazard_nll(logits, b["te_time"][rows], b["te_cause"][rows])
            else:
                cif = cumulative_incidence(logits, b["ep_hz"][e].expand(int(rows.sum()), -1))
                mh = b["te_mh"][rows]
                bce = nn.functional.binary_cross_entropy(cif, b["te_yh"][rows].clamp(0, 1), reduction="none")
                li = (bce * mh).sum(1) / mh.sum(1).clamp_min(1)
                valid = valid & mh.any(1)
        else:                                                            # forecast: per step and channel
            tm = (tr_ep == e)
            yf, mf = b["te_yf"][rows], b["te_mf"][rows]
            h, c = yf.shape[1:]
            logits = torch.zeros(h, c, bins, device=dev)
            for i in range(h):
                for j in range(c):
                    logits[i, j] = _histogram_logits(b["tr_yf"][tm, i, j][b["tr_mf"][tm, i, j]], bins)
            nll = bar_nll(logits.expand(len(yf), h, c, bins), yf)
            li = (nll * mf).sum((1, 2)) / mf.sum((1, 2)).clamp_min(1)
            valid = valid & mf.flatten(1).any(1)
        loss[rows] = torch.where(valid, li, torch.full_like(li, float("nan")))
    return loss


def _histogram_logits(u: torch.Tensor, bins: int) -> torch.Tensor:
    cnt = torch.zeros(bins, device=u.device)
    if len(u):
        cnt = cnt.index_add(0, bar_index(u, bins), torch.ones_like(u))
    return torch.log((cnt + 0.5) / (cnt.sum() + 0.5 * bins))


def _hazard_from_rows(lab: torch.Tensor) -> torch.Tensor:
    """Log-probabilities [N_BINS, 4] of the discrete competing-risk hazards from context rows' event processes
    (``batch.label_features`` columns 20-25): per bin, the rows at risk and their outcomes, add-half smoothed."""
    time = torch.expm1(lab[:, 21])
    cause = torch.full((len(lab),), -1, dtype=torch.long, device=lab.device)
    for c in range(3):
        cause = torch.where(lab[:, 22 + c] > 0, torch.full_like(cause, c), cause)
    j = hazard_bins(time)
    at_risk = torch.zeros(N_BINS, device=lab.device)
    events = torch.zeros(N_BINS, OUTCOMES, device=lab.device)
    for b_ in range(N_BINS):
        at_risk[b_] = ((j > b_) | ((j == b_) & (cause >= 0))).sum()
        for c in range(3):
            events[b_, 1 + c] = ((j == b_) & (cause == c)).sum()
    events[:, 0] = (at_risk - events[:, 1:].sum(1)).clamp_min(0)
    p = (events + 0.5) / (events.sum(1, keepdim=True) + 0.5 * OUTCOMES)
    return torch.log(p)
