"""Risks from the model's step hazards (Paper P amendment 7, F2-F3; ``docs/p1_paper_protocol.md``).

Drawing a rollout samples, at every step, the next token from the model's distribution. The probability that this
distribution gives an event's tokens at a step is the event's hazard there; summed along a path up to the path's
first event or end, the hazards make the event's compensator, whose expectation equals the expected number of events
(the Doob-Meyer decomposition). So, within rollouts drawn from the same model, the summed hazards must match the
sampled events (the self-check), and replacing each rollout's sampled events by its hazard increments estimates the
same cumulative incidence with less variance (a Rao-Blackwellization). Three estimators of an event's risk by a horizon:

* ``e0``: A1's share of rollouts with the event among those whose outcome by the horizon is known (complete case);
* ``e1``: the Aalen-Johansen cumulative incidence over the rollouts' sampled events (competing causes; a rollout that
  reaches its token cap is censored there);
* ``e2`` (the hybrid): the same with each rollout's events replaced by its hazard increments while at risk.

The hazards read the grammar as the sampler does: an event token can come next only if it does not precede the
current time group's last event in canonical order (``group_first``), and nothing but the clock block can follow an
interval that opens it. Times are the labels' decode clock (``factory.twins.decode_labels``: the gap draws of the
anchor's clock), so hazards and labels share their minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAUSES = ("death", "alive", "imv", "vasopressor", "rrt")
TREATMENTS = ("imv", "vasopressor", "rrt")
END_ALIVE, END_DEAD, END_CENSORED = 0, 1, 2


@dataclass
class Grammar:
    """What the hazards need of a vocabulary (``grammar(v)``; a toy one in the tests)."""
    interval: np.ndarray            # bool [V]
    event: np.ndarray               # bool [V]: event-first tokens (the automaton's group order applies)
    clock: np.ndarray               # bool [V]: the clock block's tokens (written by the sampler)
    adm: np.ndarray                 # bool [V]: the ICU admission marker
    tl: int
    end: int
    sets: dict                      # cause -> sorted token ids (death, alive, imv, vasopressor, rrt)


def grammar(v) -> Grammar:
    from icutwin.eval.forecast import Reader
    from icutwin.tokenization.fsm import compile_tables

    t = compile_tables(v)
    rd = Reader(v)
    n = len(v.tokens)
    vec = lambda ids: np.isin(np.arange(n), np.fromiter(ids, dtype=np.int64, count=len(ids)))  # noqa: E731
    dead = sorted(i for i in rd.dis if (v.meta[i].get("code") or "").endswith("//dead"))
    alive = sorted(set(rd.dis) - set(dead))
    clock = {i for i, m in enumerate(v.meta) if m.get("cls") == "clock"}
    return Grammar(vec(t.interval), vec(t.event_first) & ~vec(clock), vec(clock), vec(rd.adm), v.id("[TL]"),
                   v.id("[END]"), {"death": np.array(dead), "alive": np.array(alive),
                                   **{k: np.array(sorted(rd.sets[k])) for k in TREATMENTS}})


@dataclass
class Normalizer:
    """The sampler's denominator at a timeline position: the model's probability of the tokens it may draw there
    (``generate.constrain.generation_mask``), from the automaton's timeline rules (an event not below the group's
    last one; no interval right after an interval; no [END] right after an interval or a week; after a system event,
    its next sub-token) and the dataset's support. The infusion rule (a rate change only while that infusion runs) is
    not modelled: those tokens count as allowed. ``normalizer(v)``."""
    e_ids: np.ndarray               # event-first tokens, sorted
    i_ids: np.ndarray               # interval tokens
    week: int
    end: int
    sub_ids: tuple                  # tokens of sub-token levels 1, 2, 3
    system: np.ndarray              # bool [V]
    level: np.ndarray               # int [V]: sub-token level, 0 elsewhere
    nr_ids: np.ndarray              # infusion events allowed only while their key runs (dose change, stop), sorted
    nr_key: np.ndarray              # their keys (0 .. n_keys - 1)
    n_keys: int
    inf_key: np.ndarray             # int [V]: an infusion event's key, -1 elsewhere
    inf_state: np.ndarray           # int [V]: 1 after a 'begins' action, 0 after an 'ends' one, -1 (no change) else


def normalizer(v) -> Normalizer:
    from icutwin.tokenization.fsm import compile_tables

    t = compile_tables(v)
    n = len(v.tokens)
    level = np.zeros(n, dtype=np.int64)
    for i, lv in t.chunk_level.items():
        level[i] = lv
    system = np.zeros(n, dtype=bool)
    system[np.fromiter(t.system, dtype=np.int64, count=len(t.system))] = True
    keys = sorted({k for k, _a in t.key.values()})
    kidx = {k: j for j, k in enumerate(keys)}
    inf_key = np.full(n, -1, dtype=np.int64)
    inf_state = np.full(n, -1, dtype=np.int64)
    nr = []
    for tok, (k, action) in t.key.items():
        inf_key[tok] = kidx[k]
        if action in t.begins:
            inf_state[tok] = 1
        elif action in t.ends:
            inf_state[tok] = 0
        if action in t.needs_running and action not in t.begins:
            nr.append(tok)
    nr_ids = np.array(sorted(nr), dtype=np.int64)
    return Normalizer(np.array(sorted(t.event_first)), np.array(sorted(t.interval)), int(t.week),
                      int(t.special["[END]"]), tuple(np.flatnonzero(level == k) for k in (1, 2, 3)), system, level,
                      nr_ids, inf_key[nr_ids], len(keys), inf_key, inf_state)


def running_state(seq, pos: np.ndarray, nz: Normalizer) -> np.ndarray:
    """bool [positions, keys]: the infusions running after each position's token (the automaton's ``running``: a
    'begins' action starts one, an 'ends' action stops it; one context window from the stay's start)."""
    seq = np.asarray(seq, dtype=np.int64)
    kk, st = nz.inf_key[seq], nz.inf_state[seq]
    ev = np.flatnonzero((kk >= 0) & (st >= 0))
    out = np.zeros((len(pos), nz.n_keys), dtype=bool)
    for k in np.unique(kk[ev]):
        e = ev[kk[ev] == k]
        j = np.searchsorted(e, pos, side="right") - 1
        out[:, k] = np.where(j >= 0, st[e[np.maximum(j, 0)]] == 1, False)
    return out


def masked_parts(lg, nz: Normalizer, sets: dict, support, gf):
    """On the model's logits at a chunk of positions (torch [c, V]): per position the supported probability of the
    events not below the group order (``gf``, torch [c]), of the intervals, the week, [END] and each sub-token level,
    and per cause of its allowed supported members. ``sets``: cause -> sorted token ids (torch); ``support``: torch
    float [V] (ones without a mask). Returns a dict of torch [c] tensors."""
    import torch

    p = torch.softmax(lg.float(), -1) * support
    dev = lg.device
    out = {}

    def suffix_at(ids, thr):
        cols = p[:, ids]
        suf = torch.flip(torch.cumsum(torch.flip(cols, [1]), 1), [1])
        suf = torch.cat([suf, torch.zeros((len(cols), 1), device=dev)], 1)
        j = torch.searchsorted(ids, thr.contiguous(), right=False)
        return suf.gather(1, j[:, None])[:, 0]

    e_ids = torch.as_tensor(nz.e_ids, device=dev)
    out["events"] = suffix_at(e_ids, gf)
    if len(nz.nr_ids):                              # infusion events per key (not below the group order)
        nr = torch.as_tensor(nz.nr_ids, device=dev)
        onehot = torch.zeros((len(nr), nz.n_keys), device=dev)
        onehot[torch.arange(len(nr), device=dev), torch.as_tensor(nz.nr_key, device=dev)] = 1.0
        pn = p[:, nr] * (nr[None, :] >= gf[:, None]).float()
        out["nr_by_key"] = pn @ onehot
        for c, ids in sets.items():
            member = torch.isin(nr, ids).float()
            if member.sum() > 0:
                out[f"{c}_nr_by_key"] = (pn * member[None, :]) @ onehot
    out["intervals"] = p[:, torch.as_tensor(nz.i_ids, device=dev)].sum(1)
    out["week"] = p[:, nz.week]
    out["end"] = p[:, nz.end]
    for k, ids in enumerate(nz.sub_ids, start=1):
        out[f"sub{k}"] = p[:, torch.as_tensor(ids, device=dev)].sum(1) if len(ids) else torch.zeros(len(p), device=dev)
    for c, ids in sets.items():
        out[c] = suffix_at(ids, gf) if len(ids) else torch.zeros(len(p), device=dev)
    return out


def normalized(parts: dict, prev: np.ndarray, nz: Normalizer, causes=CAUSES, running: np.ndarray | None = None
               ) -> dict:
    """Per cause the sampler's probability at the positions (``masked_parts`` moved to numpy; ``prev`` the token at
    each position, the last before the predicted one; ``running`` its infusions, ``running_state``): the cause's
    allowed mass over the allowed mass. Without ``running`` the infusion rule is not applied."""
    is_int = np.isin(prev, nz.i_ids) & (prev != nz.week)
    is_week = prev == nz.week
    hier = np.where(nz.system[prev], 0, np.where(nz.level[prev] > 0, nz.level[prev], -1))
    allowed = parts["events"] + np.where(is_int, 0.0, parts["intervals"]) \
        + np.where(is_int | is_week, 0.0, parts["end"])
    for k in (1, 2, 3):
        allowed = allowed + np.where(hier == k - 1, parts[f"sub{k}"], 0.0)
    num = {c: np.asarray(parts[c], dtype=np.float64).copy() for c in causes}
    if running is not None and "nr_by_key" in parts:    # a rate change or stop of an infusion that is not running
        off = ~running
        allowed = allowed - np.sum(parts["nr_by_key"] * off, axis=1)
        for c in causes:
            if f"{c}_nr_by_key" in parts:
                num[c] = num[c] - np.sum(parts[f"{c}_nr_by_key"] * off, axis=1)
    allowed = np.maximum(allowed, 1e-12)
    return {c: np.maximum(num[c], 0.0) / allowed for c in causes}


def group_first(seq: np.ndarray, g: Grammar) -> np.ndarray:
    """After each token: the last event token of the current time group (-1 after an interval or before any event):
    an event token below it cannot come next (``tokenization.fsm.Automaton``)."""
    seq = np.asarray(seq, dtype=np.int64)
    n = len(seq)
    is_int, is_ev = g.interval[seq], g.event[seq]
    mark = np.where(is_int | is_ev, np.arange(n), -1)
    last = np.maximum.accumulate(mark) if n else mark
    lc = np.maximum(last, 0)
    return np.where((last < 0) | is_int[lc], -1, seq[lc])


def set_hazard(p: np.ndarray, toks: np.ndarray, gf: np.ndarray) -> np.ndarray:
    """Per position, the probability of a token set's members the grammar allows next: ``p`` [positions, len(toks)]
    the probabilities of ``toks`` (sorted ids), ``gf`` the group order after each position."""
    if not len(toks):
        return np.zeros(len(p))
    suf = np.concatenate([np.cumsum(p[:, ::-1], axis=1)[:, ::-1], np.zeros((len(p), 1))], axis=1)
    j = np.searchsorted(toks, gf, side="left")              # members at or above the group's last event
    return suf[np.arange(len(p)), j]


class Clock:
    """The labels' decode clock of one anchor (``tokenization.decode.to_stay`` with ``factory.twins.clock_rng``): one
    draw per interval token that has a sketch, in order; ``state`` after the prompt is shared by its rollouts."""

    def __init__(self, v, dataset: str):
        from icutwin.tokenization.decode import load_spec

        self.v, self.dataset = v, dataset
        own, star = v.interval_sketches.get(dataset), v.interval_sketches["*"]
        edges = load_spec()["timeline"]["interval_edges"]
        self.sk, self.fixed = {}, {}
        n = len(v.meta)
        self.has_sk = np.zeros(n, dtype=bool)
        self.fixed_arr = np.full(n, np.nan)
        for i, m in enumerate(v.meta):
            if m.get("cls") != "interval":
                continue
            k = m["interval"]
            s = own[k] if own and own[k] is not None else star[k]
            if s is None:
                self.fixed[i] = float(edges[k])
                self.fixed_arr[i] = float(edges[k])
            else:
                self.sk[i] = np.asarray(s, dtype=np.float64)
                self.has_sk[i] = True

    def minutes(self, seq: np.ndarray, rng, timeline_from: int = 0) -> np.ndarray:
        """Minutes after each token (cumulative gaps of the interval tokens from ``timeline_from``); ``rng`` (a
        ``random.Random``) advances as the decode's: one draw per sketched interval token, in order."""
        seq = np.asarray(seq, dtype=np.int64)
        gaps = np.zeros(len(seq))
        k = np.arange(timeline_from, len(seq))
        toks = seq[timeline_from:]
        idx = k[self.has_sk[toks]]
        if len(idx):
            u = np.array([rng.random() for _ in range(len(idx))])
            ti = seq[idx]
            for tok in np.unique(ti):
                m = ti == tok
                s = self.sk[int(tok)]
                pos = u[m] * (len(s) - 1)
                i = np.minimum(pos.astype(np.int64), len(s) - 2)
                gaps[idx[m]] = s[i] + (pos - i) * (s[i + 1] - s[i])
        fx = self.fixed_arr[toks]
        fm = ~np.isnan(fx)
        gaps[k[fm]] = fx[fm]
        return np.cumsum(gaps)

    def minutes_loop(self, seq: np.ndarray, rng, timeline_from: int = 0) -> np.ndarray:
        """The reference (token by token) of ``minutes``."""
        seq = np.asarray(seq, dtype=np.int64)
        gaps = np.zeros(len(seq))
        for k in range(timeline_from, len(seq)):
            tok = int(seq[k])
            if tok in self.sk:
                s = self.sk[tok]
                pos = rng.random() * (len(s) - 1)
                i = min(int(pos), len(s) - 2)
                gaps[k] = s[i] + (pos - i) * (s[i + 1] - s[i])
            elif tok in self.fixed:
                gaps[k] = self.fixed[tok]
        return np.cumsum(gaps)


@dataclass
class Path:
    """One sequence's summary (a rollout, or an anchor's real future): per cause, its first event (hours after the
    anchor; inf without one), the follow-up's end and its kind, and the hazard increments per time bin while at
    risk: ``inc`` per cause (death and alive discharge until the ICU discharge; a treatment until its first event or
    the discharge), ``comp`` per treatment (death plus alive discharge while the treatment is at risk)."""
    first: dict                     # cause -> hours (inf: none)
    end_hour: float
    end_kind: int
    inc: dict                       # cause -> float32 [bins]
    comp: dict                      # treatment -> float32 [bins]
    cut_hour: float = np.inf        # amendment 7 D3: where A1's cap would have censored it
    inc_cut: dict | None = None     # death and alive increments up to cut_hour


def summarize(seq, start: int, haz: dict, hours: np.ndarray, edges: np.ndarray, g: Grammar,
              stop_pos: int | None = None, cut_pos: int | None = None) -> Path:
    """``seq``: prompt and continuation; ``start``: the first position whose prediction counts (the prompt's last);
    ``haz[cause][s]`` the hazard of position s (predicting token s + 1; zero where the sampler could not draw it);
    ``hours[s]`` hours after the anchor at position s. A cause's first event is the first of its tokens after the
    prompt; the follow-up ends at the ICU discharge (dead or alive), else at ``stop_pos`` (the last position whose
    prediction the sampler drew freely: the cap's reserve, the context, or [END]) as censored."""
    seq = np.asarray(seq, dtype=np.int64)
    n = len(seq)
    stop_pos = n - 1 if stop_pos is None else min(stop_pos, n - 1)
    tail = np.arange(start + 1, n)
    is_set = {c: np.isin(seq[tail], g.sets[c]) for c in CAUSES}
    pos_of = {c: (int(tail[np.argmax(is_set[c])]) if is_set[c].any() else None) for c in CAUSES}
    dis = [p for p in (pos_of["death"], pos_of["alive"]) if p is not None]
    if dis:
        d = min(dis)
        end_kind, end_pos = (END_DEAD if d == pos_of["death"] else END_ALIVE), d
        end_hour = float(hours[d - 1])                      # the token's time: its group's, set before it
    else:
        end_kind, end_pos = END_CENSORED, stop_pos + 1
        end_hour = float(hours[stop_pos])
    first = {}
    for c in CAUSES:
        p = pos_of[c]
        first[c] = float(hours[p - 1]) if p is not None and p <= end_pos else np.inf

    def accumulate(last_pos: int) -> np.ndarray:
        s = np.arange(start, max(start, last_pos))         # positions predicting tokens start+1 .. last_pos
        h = hours[s]
        b = np.searchsorted(edges, h, side="left") - 1    # bin of (e_b, e_b+1]; hour 0 (the anchor's group): none
        ok = (h > 0) & (b >= 0) & (b < len(edges) - 1)
        return s[ok], b[ok]

    nb = len(edges) - 1
    inc, comp = {}, {}
    for c in CAUSES:
        last = end_pos if c in ("death", "alive") else min(end_pos, pos_of[c] if pos_of[c] is not None else n)
        s, b = accumulate(last)
        inc[c] = np.bincount(b, weights=haz[c][s], minlength=nb).astype(np.float32)
        if c in TREATMENTS:
            comp[c] = np.bincount(b, weights=haz["death"][s] + haz["alive"][s], minlength=nb).astype(np.float32)
    out = Path(first, end_hour, end_kind, inc, comp)
    if cut_pos is not None and cut_pos < end_pos:
        out.cut_hour = float(hours[cut_pos])
        out.inc_cut = {}
        for c in ("death", "alive"):
            s, b = accumulate(cut_pos + 1)
            out.inc_cut[c] = np.bincount(b, weights=haz[c][s], minlength=len(edges) - 1).astype(np.float32)
    return out


# -- estimators ------------------------------------------------------------------------------------------------------
def e0(first: np.ndarray, end_hour: np.ndarray, end_kind: np.ndarray, horizon: float) -> float:
    """A1's estimator: the share with the event by ``horizon`` among rollouts whose outcome there is known (the event
    by then, follow-up beyond it, or an ICU discharge before it). NaN when none is known."""
    pos = first <= horizon
    known = pos | (end_hour > horizon) | (end_kind != END_CENSORED)
    return float(pos[known].mean()) if known.any() else float("nan")


def aj(edges: np.ndarray, stop: np.ndarray, d_event: np.ndarray, d_other: np.ndarray, horizon: float) -> float:
    """Aalen-Johansen cumulative incidence of the event by ``horizon``: ``stop`` [n] each member's stop (hours),
    ``d_event``/``d_other`` [n, bins] its event and competing-cause counts (or hazard increments) per bin. At risk in a
    bin: members stopping after its start. NaN when nobody is at risk at the first bin."""
    lo = edges[:-1]
    keep = edges[1:] <= horizon + 1e-9
    at = (stop[:, None] > lo[None, :]).sum(0)[keep].astype(float)
    if not len(at) or at[0] == 0:
        return float("nan")
    de, do = d_event.sum(0)[keep], d_other.sum(0)[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_e = np.where(at > 0, de / at, 0.0)
        lam_o = np.where(at > 0, do / at, 0.0)
    surv = np.cumprod(np.r_[1.0, 1.0 - np.clip(lam_e + lam_o, 0, 1)])[:-1]
    return float(np.sum(surv * lam_e))


def event_counts(first: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """[n, bins] one-hot of each member's event time in the bins (none: zeros)."""
    out = np.zeros((len(first), len(edges) - 1))
    b = np.searchsorted(edges, first, side="left") - 1
    ok = np.isfinite(first) & (b >= 0) & (b < len(edges) - 1)
    out[np.flatnonzero(ok), b[ok]] = 1.0
    return out


def estimate(cause: str, horizon: float, edges: np.ndarray, first: np.ndarray, end_hour: np.ndarray,
             end_kind: np.ndarray, inc_event: np.ndarray, inc_other: np.ndarray) -> dict:
    """E0, E1 and E2 of one anchor's rollouts for ``cause`` (death or a treatment) by ``horizon`` (np.inf: until the
    ICU discharge, the last bin). Arrays over the rollouts: the cause's first event, the follow-up's end and kind,
    and the hazard increments [rollouts, bins] of the cause and of its competitors while it is at risk."""
    first = np.where(first <= end_hour, first, np.inf)
    stop = np.minimum(first, end_hour)
    if cause == "death":                               # an alive discharge competes with death
        oth_first = np.where(end_kind == END_ALIVE, end_hour, np.inf)
    else:                                              # death and alive discharge compete with a treatment's start
        oth_first = np.where((end_kind != END_CENSORED) & ~np.isfinite(first), end_hour, np.inf)
    h = horizon if np.isfinite(horizon) else float(edges[-1])
    return {"e0": e0(first, end_hour, end_kind, h),
            "e1": aj(edges, stop, event_counts(first, edges), event_counts(oth_first, edges), h),
            "e2": aj(edges, stop, inc_event, inc_other, h),
            "censored_before": float(np.mean((end_kind == END_CENSORED) & (end_hour < h) & ~(first <= h)))}


def risks(paths: list, cause: str, horizon: float, edges: np.ndarray, cut: bool = False) -> dict:
    """``estimate`` from ``Path`` summaries; ``cut``: as if each rollout had stopped at A1's cap (death only)."""
    first = np.array([p.first[cause] for p in paths])
    end_hour = np.array([p.end_hour for p in paths])
    end_kind = np.array([p.end_kind for p in paths])
    if cause == "death":
        ev = np.stack([p.inc["death"] for p in paths])
        ot = np.stack([p.inc["alive"] for p in paths])
    else:
        ev = np.stack([p.inc[cause] for p in paths])
        ot = np.stack([p.comp[cause] for p in paths])
    if cut:
        ch = np.array([p.cut_hour for p in paths])
        hit = ch < end_hour
        end_hour = np.where(hit, ch, end_hour)
        end_kind = np.where(hit, END_CENSORED, end_kind)
        ev = np.stack([p.inc_cut["death"] if p.inc_cut is not None else p.inc["death"] for p in paths])
        ot = np.stack([p.inc_cut["alive"] if p.inc_cut is not None else p.inc["alive"] for p in paths])
    return estimate(cause, horizon, edges, first, end_hour, end_kind, ev, ot)
