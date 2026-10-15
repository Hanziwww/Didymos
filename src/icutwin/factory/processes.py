"""Event processes, the label engine's intermediate (plan Phase F, S2): for every stay of a store, the intervals in
which a state holds (merged, ordered by stay and start) and the onsets of the event (ordered by stay and time). A
label program reads one process at many (stay, time) rows at once (``factory.labels``).

Sources:
* ``episodes``: a derived concept's intervals (IMV, vasopressor, RRT, ... episodes from the labels' decode);
* ``points``: a state read off measurements: each on-measurement holds until the next measurement of the channel or
  for ``hold`` minutes, whichever comes first; consecutive on-measurements at most ``hold`` apart form a run. A run is
  an event once it has lasted ``sustain`` minutes, at its first measurement that far from the run's start (``sustain``
  0: at its start), and in the state from then on; shorter runs are neither events nor in the state;
* ``threshold``: ``points`` of a channel against one of its decile edges (bins are exact, so median and drawn decodes
  agree: below edge j is bin <= j, at or above it is bin >= j + 1);
* ``staged``: a derived point series (e.g. KDIGO stages) at or above a level;
* ``death`` / ``discharge``: the stay's end;
* ``intersect`` / ``union``: AND and OR of two processes (half-open intervals; onsets are the starts of the result).

Times are minutes since ICU admission. ``MemStore`` is a store held in memory, for hand-built tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T_OFF = 1 << 31


def comp(stay, t) -> np.ndarray:
    """A composite key ordering (stay, time) pairs: stay in the high 32 bits, time (offset to be non-negative) below."""
    return (np.asarray(stay, dtype=np.int64) << 32) | (np.asarray(t, dtype=np.int64) + T_OFF)


@dataclass
class Process:
    stay: np.ndarray        # intervals: stay, start, end (half-open, merged, ordered)
    s: np.ndarray
    e: np.ndarray
    on_stay: np.ndarray     # onsets: stay, time (ordered)
    on_t: np.ndarray

    @classmethod
    def empty(cls) -> Process:
        z = np.zeros(0, dtype=np.int64)
        return cls(z, z, z, z, z)

    def first_onset(self, n: int) -> np.ndarray:
        """Per stay (``n`` stays), the time of its first onset (inf without one)."""
        out = np.full(n, np.inf)
        if len(self.on_t):
            u, i = np.unique(self.on_stay, return_index=True)
            out[u] = self.on_t[i]
        return out


def merge(stay, s, e) -> tuple:
    """Merge overlapping or touching intervals per stay; returns (stay, start, end) ordered."""
    stay, s, e = (np.asarray(x, dtype=np.int64) for x in (stay, s, e))
    keep = e > s
    stay, s, e = stay[keep], s[keep], e[keep]
    if not len(s):
        return stay, s, e
    order = np.lexsort((s, stay))
    stay, s, e = stay[order], s[order], e[order]
    ks, ke = comp(stay, s), comp(stay, e)
    cm = np.maximum.accumulate(ke)
    new = np.r_[True, ks[1:] > cm[:-1]]
    first = np.flatnonzero(new)
    ends = np.maximum.reduceat(ke, first)
    return stay[first], s[first], (ends & 0xFFFFFFFF) - T_OFF


def from_intervals(stay, s, e) -> Process:
    st, a, b = merge(stay, s, e)
    return Process(st, a, b, st.copy(), a.copy())


def points(stay, t, on, hold: int, sustain: int = 0) -> Process:
    """The process of a state read off measurements (module docstring); inputs ordered by stay and time."""
    stay, t = np.asarray(stay, dtype=np.int64), np.asarray(t, dtype=np.int64)
    on = np.asarray(on, dtype=bool)
    if not on.any():
        return Process.empty()
    same = np.r_[False, stay[1:] == stay[:-1]]
    near = np.r_[False, (t[1:] - t[:-1]) <= hold]
    prev_on = np.r_[False, on[:-1]]
    start = on & ~(prev_on & same & near)
    nxt_same, nxt_near, nxt_on = np.r_[same[1:], False], np.r_[near[1:], False], np.r_[on[1:], False]
    end = on & ~(nxt_on & nxt_same & nxt_near)
    i0, i1 = np.flatnonzero(start), np.flatnonzero(end)
    nxt_t = np.r_[t[1:], np.iinfo(np.int64).max]
    stop = np.where(nxt_same[i1] & ((nxt_t[i1] - t[i1]) <= hold), nxt_t[i1], t[i1] + hold)
    if sustain > 0:
        keys = comp(stay, t)
        j = np.searchsorted(keys, comp(stay[i0], t[i0] + sustain), "left")
        ok = (j <= i1)
        j = np.minimum(j, len(t) - 1)
        i0, i1, stop, onset = i0[ok], i1[ok], stop[ok], t[j[ok]]
    else:
        onset = t[i0]
    st, a, b = merge(stay[i0], onset, stop)           # in the state from the event on (a sustained run's onset)
    return Process(st, a, b, stay[i0], onset)


def threshold(ev: dict, edge: int, direction: str, hold: int, sustain: int = 0) -> Process:
    """``points`` of one channel's events (``store.channel_events``) against its edge ``edge`` (0-based):
    ``below`` is bin <= edge (value under the edge), ``above`` is bin >= edge + 1 (value at or over it)."""
    b = np.asarray(ev["bin"]).astype(np.int64)
    has = b != 255
    on = (b <= edge) if direction == "below" else (b >= edge + 1)
    return points(ev["stay"][has], ev["t"][has], on[has], hold, sustain)


def staged(dv: dict, level: float, hold: int) -> Process:
    """A derived point series (``store.derived_events``; values in ``v``) at or above ``level``."""
    v = np.asarray(dv["v"])
    ok = ~np.isnan(v)
    return points(dv["stay"][ok], dv["s"][ok], v[ok] >= level, hold)


def episodes(dv: dict) -> Process:
    return from_intervals(dv["stay"], dv["s"], np.maximum(dv["e"], dv["s"] + 1))


def ends(store, kind: str) -> Process:
    """``death``: the ICU death at the stay's end; ``discharge``: the ICU discharge alive."""
    want = "dead" if kind == "death" else "alive"
    idx = np.flatnonzero(store.end_kind == want)
    t = store.los[idx]
    return Process(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                   idx.astype(np.int64), t.astype(np.int64))


def _combine(a: Process, b: Process, op) -> Process:
    if not len(a.s) and not len(b.s):
        return Process.empty()
    stay = np.concatenate([a.stay, a.stay, b.stay, b.stay])
    t = np.concatenate([a.s, a.e, b.s, b.e])
    da = np.concatenate([np.ones_like(a.s), -np.ones_like(a.e), np.zeros_like(b.s), np.zeros_like(b.e)])
    db = np.concatenate([np.zeros_like(a.s), np.zeros_like(a.e), np.ones_like(b.s), -np.ones_like(b.e)])
    first_end = (da + db) < 0                         # at equal times ends come first (half-open intervals)
    order = np.lexsort((~first_end, t, stay))
    stay, t, ca, cb = stay[order], t[order], np.cumsum(da[order]), np.cumsum(db[order])
    inside = op(ca > 0, cb > 0)
    nxt_same = np.r_[stay[1:] == stay[:-1], False]
    nxt_t = np.r_[t[1:], 0]
    k = np.flatnonzero(inside & nxt_same & (nxt_t > t))
    return from_intervals(stay[k], t[k], nxt_t[k])


def intersect(a: Process, b: Process) -> Process:
    return _combine(a, b, np.logical_and)


def union(a: Process, b: Process) -> Process:
    return _combine(a, b, np.logical_or)


class MemStore:
    """A store held in memory with the label engine's interface (hand-built tests): ``stays`` are dicts with
    ``los`` (minutes), ``end`` (alive / dead / unknown / censored), ``dataset``, ``hosp_death``, ``events``
    [(t, channel name, value)] and ``derived`` [(kind, start, end, value)]; ``channels`` a ``factory.channels``
    table (bins from its edges)."""

    def __init__(self, channels, stays: list[dict]):
        import pandas as pd

        from icutwin.tokenization.vocab import bin_of

        self.channels = channels
        self.los = np.array([s["los"] for s in stays], dtype=np.int64)
        self.end_kind = np.array([s.get("end", "alive") for s in stays], dtype=object)
        self.stays = pd.DataFrame({"dataset": [s.get("dataset", "x") for s in stays],
                                   "hosp_death": [s.get("hosp_death") for s in stays],
                                   "los_min": self.los, "end_kind": self.end_kind})
        self._ev, self._dv = {}, {}
        for i, s in enumerate(stays):
            for t, name, val in sorted(s.get("events", []), key=lambda x: x[0]):
                c = channels.index[name]
                edges = channels.edges[c]
                b = 255 if val is None else bin_of(val, edges)
                self._ev.setdefault(c, []).append((i, t, np.nan if val is None else val, b, 0))
            for kind, a, e, val in sorted(s.get("derived", []), key=lambda x: x[1]):
                self._dv.setdefault(kind, []).append((i, a, e, np.nan if val is None else val))

    def __len__(self) -> int:
        return len(self.los)

    def channel_events(self, cid: int) -> dict:
        rows = self._ev.get(cid, [])
        cols = list(zip(*rows)) if rows else [[], [], [], [], []]
        return {"stay": np.asarray(cols[0], dtype=np.int64), "t": np.asarray(cols[1], dtype=np.int64),
                "v": np.asarray(cols[2], dtype=np.float32), "bin": np.asarray(cols[3], dtype=np.uint8),
                "f": np.asarray(cols[4], dtype=np.uint8)}

    def derived_events(self, kind: str) -> dict:
        rows = self._dv.get(kind, [])
        cols = list(zip(*rows)) if rows else [[], [], [], []]
        return {"stay": np.asarray(cols[0], dtype=np.int64), "s": np.asarray(cols[1], dtype=np.int64),
                "e": np.asarray(cols[2], dtype=np.int64), "v": np.asarray(cols[3], dtype=np.float32)}
