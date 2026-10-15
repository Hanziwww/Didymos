"""A reference consumer of factory episodes (plan Phase F, S9): a tiny model, randomly initialized and never trained
here, that reads every task type of "Task formats" end to end, to prove the format is consumable and its targets
usable by the usual heads. It is not the architecture (the next round designs it).

Per episode:
* tokens: each event (its channel, global or episode-local when anonymized; its standardized value and presence; its
  time; its flags), each state interval (channel, value, start, duration), each static (key, value) and, for context
  patients, each label of their timeline (time, label, validity);
* a context patient: its tokens through a small transformer encoder, mean-pooled (a complete trajectory: no causal
  mask is needed);
* a query at each prediction time t_k (its start vector shifted by the context's declared selection: log case odds and
  the cases it guarantees): an attention pool over its own tokens available by t_k (events by their time,
  intervals started by then, statics documented by then, its own labels resolved by then), with no attention among
  the query's tokens, so nothing after t_k reaches the state at t_k; then attention over the context patients;
* heads: a Bernoulli logit (binary, in-hospital death), class logits (ordinal), a Gaussian (regression), a
  discrete-time hazard on a log-spaced grid (time to event, censored likelihood), a logit per horizon (multi-horizon),
  a value per (horizon, channel) (forecasts).

``loss`` returns each episode's masked loss (labels where they apply only), so a training step could follow.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

GRID_HOURS = tuple(float(x) for x in np.geomspace(0.5, 168.0, 24))     # the hazard grid's bin edges (hours)
KINDS = {"binary": 1, "hosp_death": 1, "ordinal": 10, "regression": 2, "tte": len(GRID_HOURS) + 1,
         "multihorizon": 8, "forecast": 1}


def _symlog(x: torch.Tensor) -> torch.Tensor:
    """Raw magnitudes (infusion rates, statics, label values in channel units) squashed: sign(x) log(1 + |x|)."""
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sign(x) * torch.log1p(x.abs())


def _time(t_min: torch.Tensor, d: int) -> torch.Tensor:
    """Sinusoidal features of times (minutes since ICU admission; negative before it)."""
    h = t_min.float().unsqueeze(-1) / 60.0
    k = torch.arange(d // 2, device=t_min.device, dtype=torch.float32)
    f = torch.exp(-k * math.log(1000.0) / max(d // 2, 1))
    return torch.cat([torch.sin(h * f), torch.cos(h * f)], -1)


class Consumer(nn.Module):
    def __init__(self, n_channels: int, n_statics: int = 4096, d: int = 32, heads: int = 2,
                 max_tokens: int = 256, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.d, self.max_tokens = d, max_tokens
        self.ch = nn.Embedding(n_channels + 1, d)
        self.stat = nn.Embedding(n_statics + 1, d)
        self.val = nn.Linear(2, d)
        self.flag = nn.Embedding(64, d)
        self.kind = nn.Embedding(4, d)                    # event, interval, static, label
        self.dur = nn.Linear(1, d)
        self.lab = nn.Linear(3, d)
        self.time = nn.Linear(d, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True),
                                         num_layers=1)
        self.tok = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.q0 = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.own = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.heads = nn.ModuleDict({k: nn.Linear(2 * d, n) for k, n in KINDS.items()})
        self.fc = nn.Linear(2 * d + d + d, 1)             # a forecast's (state, horizon, channel) -> value
        self.sel = nn.Linear(2, d)                        # the context's selection: log case odds, cases guaranteed
        self.n_channels = n_channels

    # -- tokens ------------------------------------------------------------------------------------------------
    def _tokens(self, part: dict, k: int, labels: str | None) -> tuple[torch.Tensor, torch.Tensor]:
        """(tokens [L, d], their availability times [L] in minutes) of patient ``k`` of a part; ``labels``: the label
        fields' prefix to add as tokens (``lab_`` for a context timeline, ``own_lab_`` for a query's own labels)."""
        def sl(name, off):
            o = part[off]
            return np.asarray(part[name])[int(o[k]):int(o[k + 1])]

        dev = self.q0.device
        ch = torch.as_tensor(np.clip(sl("ev_c", "ev_off"), 0, self.n_channels).astype(np.int64), device=dev)
        z = torch.as_tensor(sl("ev_z", "ev_off").astype(np.float32), device=dev)
        has = torch.isfinite(z)
        z = torch.clamp(torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -10, 10)
        t_ev = torch.as_tensor(sl("ev_t", "ev_off").astype(np.int64), device=dev)
        f = torch.as_tensor((sl("ev_f", "ev_off") & 63).astype(np.int64), device=dev)
        ev = (self.ch(ch) + self.val(torch.stack([torch.nan_to_num(z), has.float()], -1)) + self.flag(f)
              + self.time(_time(t_ev, self.d)) + self.kind.weight[0])
        ic = torch.as_tensor(np.clip(sl("iv_c", "iv_off"), 0, self.n_channels).astype(np.int64), device=dev)
        s = torch.as_tensor(sl("iv_s", "iv_off").astype(np.int64), device=dev)
        e = torch.as_tensor(sl("iv_e", "iv_off").astype(np.int64), device=dev)
        iv_v = torch.as_tensor(sl("iv_v", "iv_off").astype(np.float32), device=dev)
        iv = (self.ch(ic) + self.val(torch.stack([_symlog(iv_v), torch.isfinite(iv_v).float()], -1))
              + self.dur(torch.log1p((e - s).clamp(min=0).float()).unsqueeze(-1)) + self.time(_time(s, self.d))
              + self.kind.weight[1])
        sk = torch.as_tensor(np.clip(sl("st_k", "st_off").astype(np.int64), 0, self.stat.num_embeddings - 1),
                             device=dev)
        sv = torch.as_tensor(sl("st_v", "st_off").astype(np.float32), device=dev)
        sa = torch.as_tensor(sl("st_avail", "st_off").astype(np.int64), device=dev)
        st = self.stat(sk) + self.val(torch.stack([_symlog(sv), torch.ones_like(sv)], -1)) + self.kind.weight[2]
        toks, times = [ev, iv, st], [t_ev, s, sa]
        if labels is not None and f"{labels}off" in part:
            lt = torch.as_tensor(sl(f"{labels}t", f"{labels}off").astype(np.int64), device=dev)
            ly = torch.as_tensor(sl(f"{labels}y", f"{labels}off").astype(np.float32), device=dev)
            lv = torch.as_tensor(sl(f"{labels}valid", f"{labels}off").astype(np.float32), device=dev)
            res = torch.as_tensor(sl(f"{labels}t_resolved", f"{labels}off").astype(np.int64), device=dev)
            lab = (self.lab(torch.stack([_symlog(ly), lv, torch.isfinite(ly).float()], -1))
                   + self.time(_time(lt, self.d)) + self.kind.weight[3])
            toks.append(lab)
            times.append(res if labels.startswith("own") else lt)     # a query sees its own label once resolved
        return torch.cat(toks, 0), torch.cat(times, 0)

    def _context(self, ctx: dict) -> torch.Tensor:
        n = len(ctx["ev_off"]) - 1
        if n == 0:
            return torch.zeros(0, self.d, device=self.q0.device)
        out = []
        for k in range(n):
            x, _t = self._tokens(ctx, k, "lab_")
            if len(x) > self.max_tokens:                      # a reference consumer: an even subsample
                x = x[torch.linspace(0, len(x) - 1, self.max_tokens, device=x.device).long()]
            out.append(self.enc(x.unsqueeze(0)).mean(1).squeeze(0) if len(x) else torch.zeros(self.d, device=x.device))
        return torch.stack(out, 0)

    # -- forward -----------------------------------------------------------------------------------------------
    def forward(self, ep: dict) -> list[dict]:
        """Per query patient: its prediction times and the head's outputs there (``out`` [K, n] or, for a forecast,
        [K, H, C])."""
        ctx_vec = self._context(ep["context"])
        q = ep["query"]
        label = ep["task"]["label"]
        sel = ep["task"].get("selection") or {}
        s0 = self.q0 + self.sel(torch.tensor([math.log(sel.get("case_odds") or 1.0),
                                              float(sel.get("min_cases") or 0) / 10.0], device=self.q0.device))
        lo = np.asarray(q["lab_off"])
        res = []
        for k in range(len(q["ev_off"]) - 1):
            x, tav = self._tokens(q, k, "own_lab_")
            x = self.tok(x) if len(x) else x
            tk = torch.as_tensor(np.asarray(q["lab_t"])[lo[k]:lo[k + 1]].astype(np.int64), device=self.q0.device)
            qs = s0 + self.time(_time(tk, self.d))                             # [K, d]
            if len(x):
                mask = tav.unsqueeze(0) > tk.unsqueeze(1)                       # True: not available by t_k
                empty = mask.all(1, keepdim=True)
                pooled, _w = self.own(qs.unsqueeze(1), x.unsqueeze(0).expand(len(tk), -1, -1),
                                      x.unsqueeze(0).expand(len(tk), -1, -1),
                                      key_padding_mask=mask & ~empty)
                h = torch.where(empty, torch.zeros_like(qs), pooled.squeeze(1))
            else:
                h = torch.zeros_like(qs)
            if len(ctx_vec):
                c, _w = self.cross(h.unsqueeze(1), ctx_vec.unsqueeze(0).expand(len(tk), -1, -1),
                                   ctx_vec.unsqueeze(0).expand(len(tk), -1, -1))
                c = c.squeeze(1)
            else:
                c = torch.zeros_like(h)
            state = torch.cat([h, c], -1)
            if label == "forecast":
                yf = np.asarray(q["lab_yf"])[lo[k]:lo[k + 1]]
                n_h, n_c = yf.shape[1:]
                he = _time(torch.arange(n_h, device=h.device) * 60, self.d)
                ce = self.ch.weight[:n_c]
                grid = torch.cat([state[:, None, None, :].expand(-1, n_h, n_c, -1),
                                  he[None, :, None, :].expand(len(tk), -1, n_c, -1),
                                  ce[None, None, :, :].expand(len(tk), n_h, -1, -1)], -1)
                out = self.fc(grid).squeeze(-1)
            else:
                out = self.heads[label](state)
            res.append({"t": tk, "out": out})
        return res

    # -- the targets -------------------------------------------------------------------------------------------
    def loss(self, ep: dict, outs: list[dict]) -> torch.Tensor:
        """The episode's mean loss over the query labels that apply (``lab_valid``; a forecast's observed cells)."""
        q, task = ep["query"], ep["task"]
        label = task["label"]
        lo = np.asarray(q["lab_off"])
        total, n = torch.zeros((), device=self.q0.device), 0
        for k, o in enumerate(outs):
            a, b = int(lo[k]), int(lo[k + 1])
            dev = o["out"].device
            valid = torch.as_tensor(np.asarray(q["lab_valid"])[a:b].astype(bool), device=dev)
            y = torch.as_tensor(np.nan_to_num(np.asarray(q["lab_y"])[a:b].astype(np.float32)), device=dev)
            if label == "regression":
                y = _symlog(y)                                           # a value in channel units, squashed
            if label in ("binary", "hosp_death"):
                lv = nn.functional.binary_cross_entropy_with_logits(o["out"][:, 0], y, reduction="none")
            elif label == "ordinal":
                lv = nn.functional.cross_entropy(o["out"], y.long().clamp(0, KINDS["ordinal"] - 1), reduction="none")
            elif label == "regression":
                mu, ls = o["out"][:, 0], o["out"][:, 1].clamp(-5, 5)
                lv = 0.5 * ((y - mu) / ls.exp()) ** 2 + ls
            elif label == "tte":
                te = np.asarray(q["lab_t_event"])[a:b].astype(np.float64)
                tc = np.asarray(q["lab_t_censor"])[a:b].astype(np.float64)
                ev = np.isfinite(te)
                when = np.where(ev, te, tc)
                bins = torch.as_tensor(np.searchsorted(GRID_HOURS, np.nan_to_num(when, nan=0.0)), device=dev)
                logh = nn.functional.logsigmoid(o["out"])                # hazard per bin
                log1m = nn.functional.logsigmoid(-o["out"])
                cum = torch.cumsum(log1m, 1)
                surv_before = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], 1).gather(1, bins[:, None])[:, 0]
                ll_event = surv_before + logh.gather(1, bins[:, None])[:, 0]
                ll_cens = cum.gather(1, bins[:, None])[:, 0]
                lv = -torch.where(torch.as_tensor(ev, device=dev), ll_event, ll_cens)
            elif label == "multihorizon":
                yh = torch.as_tensor(np.nan_to_num(np.asarray(q["lab_yh"])[a:b].astype(np.float32)), device=dev)
                mh = torch.as_tensor(np.asarray(q["lab_mh"])[a:b].astype(bool), device=dev)
                hh = yh.shape[1]
                lm = nn.functional.binary_cross_entropy_with_logits(o["out"][:, :hh], yh, reduction="none")
                lv = (lm * mh).sum(1) / mh.sum(1).clamp(min=1)
                valid = valid & mh.any(1)
            else:                                                        # forecast
                yf = np.asarray(q["lab_yf"])[a:b].astype(np.float32)
                mf = torch.as_tensor(np.asarray(q["lab_mf"])[a:b].astype(bool), device=dev)
                yt = _symlog(torch.as_tensor(np.nan_to_num(yf), device=dev))
                lv = (((o["out"] - yt) ** 2) * mf).flatten(1).sum(1) / mf.flatten(1).sum(1).clamp(min=1)
                valid = mf.flatten(1).any(1)
            if valid.any():
                total = total + lv[valid].sum()
                n += int(valid.sum())
        return total / max(n, 1)
