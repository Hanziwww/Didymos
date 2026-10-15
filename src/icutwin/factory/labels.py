"""The label engine (plan Phase F, S2): label programs read event processes (``factory.processes``) and channel events
at many (stay, time) rows at once, vectorized by searches over (stay, time) keys.

Rows: prediction times in minutes since ICU admission (``cadence``: a rolling grid, a landmark, or the times a trigger
process fires). A ``LabelBlock`` holds, per row:
* ``y``: the label (a binary event, a class, a value; NaN where it does not apply);
* ``valid`` and ``reason`` (a bitmask, ``REASONS``: not in the ICU at t, already in the event, after a first event,
  censored before the horizon, no value in the window, treated, a competing death before the horizon);
* ``t_event`` (hours from t to the event, NaN without one), ``cause`` (``CAUSES``), ``t_censor`` (hours from t to the
  end of follow-up), ``t_resolved`` (minutes at which the label is known: the query's own label history shows a label
  only from then on).

Time to event is the primitive: one block serves every horizon (``at_horizon``, censoring-aware: an event by the
horizon is 1; no event and followed past it, or the stay ended by death or discharge, is 0, since these are ICU
events; administratively censored before it, the label is unknown). Death competes; follow-up ends at the ICU
discharge.

Values (regression) come from the stored values (drawn within their bins, as features are); thresholds read bins
(exact under both decodes).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from icutwin.factory.processes import Process, comp

REASONS = {"not_in_icu": 1, "in_event": 2, "after_event": 4, "censored": 8, "no_value": 16, "treated": 32,
           "competing": 64}
CAUSES = {"none": -1, "event": 0, "death": 1, "discharge": 2, "administrative": 3}


@dataclass
class Rows:
    stay: np.ndarray        # int64 global stay index
    t: np.ndarray           # int64 minutes since ICU admission

    def __len__(self) -> int:
        return len(self.stay)

    def take(self, idx) -> Rows:
        return Rows(self.stay[idx], self.t[idx])


@dataclass
class LabelBlock:
    y: np.ndarray
    valid: np.ndarray
    reason: np.ndarray
    t_event: np.ndarray
    cause: np.ndarray
    t_censor: np.ndarray
    t_resolved: np.ndarray

    def take(self, idx) -> LabelBlock:
        return LabelBlock(*(getattr(self, f.name)[idx] for f in fields(self)))


def rolling(store, step: int, stays=None, start: int | None = None) -> Rows:
    """Every ``step`` minutes of each stay while in the ICU (from ``start``, default ``step``, to before the
    discharge)."""
    stays = np.arange(len(store)) if stays is None else np.asarray(stays, dtype=np.int64)
    first = step if start is None else start
    n = np.maximum((store.los[stays] - 1 - first) // step + 1, 0)
    st = np.repeat(stays, n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    return Rows(st.astype(np.int64), (first + k * step).astype(np.int64))


def landmark(store, at: int, stays=None) -> Rows:
    """The time ``at`` minutes of each stay still in the ICU then."""
    stays = np.arange(len(store)) if stays is None else np.asarray(stays, dtype=np.int64)
    keep = stays[store.los[stays] > at]
    return Rows(keep.astype(np.int64), np.full(len(keep), at, dtype=np.int64))


def triggers(store, proc: Process, which: str = "onsets") -> Rows:
    """The times a process fires while in the ICU: its onsets, or its interval ends (e.g. extubation: the end of an
    IMV episode)."""
    stay, t = (proc.on_stay, proc.on_t) if which == "onsets" else (proc.stay, proc.e)
    keep = (t >= 0) & (t < store.los[stay])
    return Rows(stay[keep].astype(np.int64), t[keep].astype(np.int64))


def _in_icu(store, rows: Rows) -> np.ndarray:
    return (rows.t >= 0) & (rows.t < store.los[rows.stay])


def tte(store, proc: Process, rows: Rows, eligibility: tuple = ("in_icu", "not_in_event")) -> LabelBlock:
    """Time to the next onset of ``proc`` after t (``y``: 1 when it is observed before the ICU discharge). The
    eligibility: ``in_icu``, ``not_in_event`` (not in the state at t), ``never_before`` (no onset at or before t)."""
    n = len(rows)
    key = comp(rows.stay, rows.t)
    los = store.los[rows.stay]
    onk = comp(proc.on_stay, proc.on_t)
    j = np.searchsorted(onk, key, "right")
    jc = np.minimum(j, max(len(onk) - 1, 0))
    has = (j < len(onk)) & (proc.on_stay[jc] == rows.stay) if len(onk) else np.zeros(n, dtype=bool)
    t_next = np.where(has, proc.on_t[jc] if len(onk) else 0, np.iinfo(np.int64).max)
    event = has & (t_next <= los)
    ivk = comp(proc.stay, proc.s)
    k = np.searchsorted(ivk, key, "right") - 1
    kc = np.maximum(k, 0)
    in_ev = (k >= 0) & (proc.stay[kc] == rows.stay) & (proc.e[kc] > rows.t) if len(ivk) else np.zeros(n, dtype=bool)
    before = proc.first_onset(len(store))[rows.stay] <= rows.t
    end = store.end_kind[rows.stay]
    cause = np.where(event, 0, np.where(end == "dead", 1, np.where(np.isin(end, ("unknown", "censored")), 3, 2)))
    reason = np.zeros(n, dtype=np.uint8)
    valid = np.ones(n, dtype=bool)
    if "in_icu" in eligibility:
        bad = ~_in_icu(store, rows)
        reason |= np.where(bad, REASONS["not_in_icu"], 0).astype(np.uint8)
        valid &= ~bad
    if "not_in_event" in eligibility:
        reason |= np.where(in_ev, REASONS["in_event"], 0).astype(np.uint8)
        valid &= ~in_ev
    if "never_before" in eligibility:
        reason |= np.where(before, REASONS["after_event"], 0).astype(np.uint8)
        valid &= ~before
    t_event = np.where(event, (t_next - rows.t) / 60.0, np.nan).astype(np.float32)
    return LabelBlock(event.astype(np.float32), valid, reason, t_event, cause.astype(np.int8),
                      ((los - rows.t) / 60.0).astype(np.float32), np.where(event, t_next, los).astype(np.int64))


def at_horizon(b: LabelBlock, rows: Rows, h: int) -> LabelBlock:
    """The binary label 'event within ``h`` minutes' from a time-to-event block (module docstring)."""
    pos = (b.cause == 0) & (b.t_event * 60 <= h + 1e-6)
    ended = (b.cause == 1) | (b.cause == 2)
    known = pos | (b.t_censor * 60 > h) | ended
    reason = b.reason | np.where(~known, REASONS["censored"], 0).astype(np.uint8)
    reason |= np.where(~pos & (b.cause == 1) & (b.t_censor * 60 <= h), REASONS["competing"], 0).astype(np.uint8)
    resolved = np.where(pos, rows.t + np.nan_to_num(b.t_event) * 60, np.minimum(rows.t + h, rows.t + b.t_censor * 60))
    return LabelBlock(pos.astype(np.float32), b.valid & known, reason, b.t_event, b.cause, b.t_censor,
                      np.round(resolved).astype(np.int64))


def multi_horizon(b: LabelBlock, rows: Rows, hs: tuple) -> tuple[np.ndarray, np.ndarray]:
    """(y [n, H], valid [n, H]) of ``at_horizon`` at every horizon in ``hs``."""
    blocks = [at_horizon(b, rows, h) for h in hs]
    return np.stack([x.y for x in blocks], 1), np.stack([x.valid for x in blocks], 1)


def keyed(ev: dict, what: str = "key") -> np.ndarray:
    """An events dict's (stay, time) keys (``what="key"``), values as float64 (``"x"``) or their running sum (``"cs"``),
    computed once and kept in the dict (the evaluator caches a channel's dict, so a window over it costs a search per
    row, not a pass over the channel)."""
    k = f"_{what}"
    if k not in ev:
        if what == "key":
            ev[k] = comp(ev["stay"], ev["t"])
        elif what == "x":
            ev[k] = np.asarray(ev["v"], dtype=np.float64)
        else:
            ev[k] = np.r_[0.0, np.cumsum(np.nan_to_num(keyed(ev, "x")))]
    return ev[k]


def finite(ev: dict) -> dict:
    """The events with a numeric value (kept in the dict, as ``keyed``)."""
    if "_finite" not in ev:
        has = np.isfinite(keyed(ev, "x"))
        ev["_finite"] = ev if has.all() else {k: np.asarray(ev[k])[has] for k in ("stay", "t", "v", "bin", "f")
                                               if k in ev}
    return ev["_finite"]


def window_values(ev: dict, rows: Rows, a: int, b: int, agg: str) -> tuple[np.ndarray, np.ndarray]:
    """(value, count) of a channel's events (``store.channel_events``) in (t + a, t + b] per row: ``last``, ``first``,
    ``min``, ``max``, ``mean``, ``sum`` or ``count``."""
    keys = keyed(ev)
    lo = np.searchsorted(keys, comp(rows.stay, rows.t + a), "right")
    hi = np.searchsorted(keys, comp(rows.stay, rows.t + b), "right")
    cnt = hi - lo
    out = np.full(len(rows), np.nan)
    has = cnt > 0
    if agg == "count":
        return cnt.astype(np.float64), cnt
    x = keyed(ev, "x")
    if agg == "last":
        out[has] = x[hi[has] - 1]
    elif agg == "first":
        out[has] = x[lo[has]]
    elif has.any():
        starts = lo[has]
        if agg in ("sum", "mean"):
            cs = keyed(ev, "cs")
            s = cs[hi[has]] - cs[starts]
            out[has] = s if agg == "sum" else s / cnt[has]
        else:                                        # min / max over ragged windows, reduced at once
            red = np.minimum if agg == "min" else np.maximum
            lens = cnt[has]
            offs = np.r_[0, np.cumsum(lens)[:-1]]
            idx = np.repeat(starts - offs, lens) + np.arange(int(lens.sum()))
            out[has] = red.reduceat(x[idx], offs)
    return out, cnt


def value_target(store, ev: dict, rows: Rows, a: int, b: int, agg: str = "last",
                 baseline: int | None = None) -> LabelBlock:
    """A regression label: the channel's ``agg`` in (t + a, t + b] (``b`` < 0: to the end of the ICU stay, for stays
    that end with a discharge or a death); with ``baseline`` (minutes of lookback) its change from the last value in
    (t - baseline, t]. Valid while in the ICU through t + b with a value there (events without a numeric value do not
    count)."""
    ev = finite(ev)
    los = store.los[rows.stay]
    to_end = b < 0
    b = np.maximum(los - rows.t, 0) if to_end else b
    y, cnt = window_values(ev, rows, a, b, agg)
    ended = np.isin(store.end_kind[rows.stay], ("alive", "dead")) if to_end else np.ones(len(rows), dtype=bool)
    ok = _in_icu(store, rows) & (rows.t + b <= los) & ended
    reason = np.where(~_in_icu(store, rows), REASONS["not_in_icu"], 0).astype(np.uint8)
    reason |= np.where(_in_icu(store, rows) & ((rows.t + b > los) | ~ended), REASONS["censored"], 0).astype(np.uint8)
    if baseline is not None:
        y0, c0 = window_values(ev, rows, -baseline, 0, "last")
        y = y - y0
        cnt = np.minimum(cnt, c0)
    reason |= np.where(cnt == 0, REASONS["no_value"], 0).astype(np.uint8)
    ok &= cnt > 0
    n = len(rows)
    return LabelBlock(np.where(ok, y, np.nan).astype(np.float32), ok, reason, np.full(n, np.nan, dtype=np.float32),
                      np.full(n, -1, dtype=np.int8), ((los - rows.t) / 60.0).astype(np.float32),
                      np.asarray(rows.t + b).astype(np.int64))


def remaining_los(store, rows: Rows) -> LabelBlock:
    """Hours from t to the ICU discharge (valid in the ICU for stays that end with a discharge or a death)."""
    los = store.los[rows.stay]
    end = store.end_kind[rows.stay]
    inside = _in_icu(store, rows)
    known = np.isin(end, ("alive", "dead"))
    reason = np.where(~inside, REASONS["not_in_icu"], 0).astype(np.uint8)
    reason |= np.where(inside & ~known, REASONS["censored"], 0).astype(np.uint8)
    y = ((los - rows.t) / 60.0).astype(np.float32)
    n = len(rows)
    return LabelBlock(np.where(inside & known, y, np.nan).astype(np.float32), inside & known, reason,
                      np.full(n, np.nan, dtype=np.float32), np.full(n, -1, dtype=np.int8), y, los.astype(np.int64))


def hospital_death(store, rows: Rows) -> LabelBlock:
    """In-hospital death (the stay's end block; no time: known at the stay's end), for rows in the ICU."""
    hd = store.stays["hosp_death"].to_numpy(dtype=object)[rows.stay]
    known = np.array([x is not None and x == x for x in hd], dtype=bool)
    inside = _in_icu(store, rows)
    los = store.los[rows.stay]
    reason = np.where(~inside, REASONS["not_in_icu"], 0).astype(np.uint8)
    reason |= np.where(inside & ~known, REASONS["censored"], 0).astype(np.uint8)
    y = np.array([float(bool(x)) if k else np.nan for x, k in zip(hd, known)], dtype=np.float32)
    n = len(rows)
    return LabelBlock(y, inside & known, reason, np.full(n, np.nan, dtype=np.float32), np.full(n, -1, dtype=np.int8),
                      ((los - rows.t) / 60.0).astype(np.float32), los.astype(np.int64))


def band(b: LabelBlock, edges) -> LabelBlock:
    """An ordinal label: a value block's band (``np.digitize`` over ``edges``: class 0 below the first edge)."""
    y = np.where(b.valid, np.digitize(np.nan_to_num(b.y), np.asarray(edges, dtype=np.float64)), np.nan)
    return LabelBlock(y.astype(np.float32), b.valid, b.reason, b.t_event, b.cause, b.t_censor, b.t_resolved)


def stage_max(store, dv: dict, rows: Rows, a: int, b: int) -> LabelBlock:
    """The highest value of a derived point series (e.g. the KDIGO stage) in (t + a, t + b] (0 without one), valid
    while in the ICU through t + b."""
    ev = {"stay": dv["stay"], "t": dv["s"], "v": dv["v"]}
    y, cnt = window_values(ev, rows, a, b, "max")
    los = store.los[rows.stay]
    inside = _in_icu(store, rows)
    ok = inside & (rows.t + b <= los)
    reason = np.where(~inside, REASONS["not_in_icu"], 0).astype(np.uint8)
    reason |= np.where(inside & (rows.t + b > los), REASONS["censored"], 0).astype(np.uint8)
    n = len(rows)
    return LabelBlock(np.where(ok, np.nan_to_num(y, nan=0.0), np.nan).astype(np.float32), ok, reason,
                      np.full(n, np.nan, dtype=np.float32), np.full(n, -1, dtype=np.int8),
                      ((los - rows.t) / 60.0).astype(np.float32), (rows.t + b).astype(np.int64))
