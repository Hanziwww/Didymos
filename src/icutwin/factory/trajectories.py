"""Trajectories as episodes carry them (plan Phase F, Factory design 2; "Task formats"): a stay's events, state
intervals, statics and end, read from the stay store.

* ``trajectory(store, i, until=None, ...)``: the complete stay (a context patient), or the stay as known at ``until``
  minutes (a query patient: events and intervals available by then, intervals clipped there, no end). Times are
  minutes since ICU admission; events are available at their time (the store's times are availability times).
  Outcome channels never appear (``channels.is_input``).
* statics (``STATICS``): the prefix and header fields as (key, value, available-from minutes); a categorical field is
  the key ``field=value`` with value 1, a numeric one the key ``field`` with its value; a header field documented
  later than ICU admission (``spec.yaml`` ``available_within_hours``, per dataset) becomes available then.
* budgets: a trajectory over ``budget`` events is thinned deterministically: the frequent measurement groups
  (``THINNABLE``) lose events, every treatment, state, lab, procedure and boundary event stays; a context patient's
  are thinned evenly per channel, a query's oldest first. Thinned channels' kept events carry the flag ``THINNED``.
* values: ``ev_v`` as stored (canonical units) and ``ev_z`` standardized with the training twins' statistics (median,
  (p95 - p05) / 3.29); anonymized episodes keep only ``ev_z`` (``anonymize``).
"""

from __future__ import annotations

import numpy as np

from icutwin.factory.store import HEADER, PREFIX

THINNABLE = ("vitals", "assessments", "fluids", "respiratory", "hemodynamics")
THINNED = 32                       # ev_f bit: the event's channel was thinned in this trajectory
NUMERIC_STATICS = ("age", "prior_icu_stays", "pre_icu_los", "admission_hour")
STATICS = tuple(f for f in (*PREFIX, *HEADER))


def static_vocab(v) -> list[str]:
    """Static keys: ``field=value`` for every categorical prefix or header value of the vocabulary, ``field`` for the
    numeric fields (sorted: a fixed order)."""
    keys = set(NUMERIC_STATICS)
    for m in (v.meta if v is not None else ()):          # a generic store has no vocabulary: numeric keys only
        if m.get("cls") in ("prefix", "header") and m.get("field") in STATICS and m.get("field") not in NUMERIC_STATICS:
            if m.get("value") not in (None, "UNK", "unknown"):
                keys.add(f"{m['field']}={m['value']}")
    return sorted(keys)


def availability() -> dict:
    """field -> dataset -> minutes after ICU admission at which the field is documented (``spec.yaml``)."""
    from icutwin.tokenization.spec import load_spec

    return {f["field"]: {d: int(h * 60) for d, h in (f.get("available_within_hours") or {}).items()}
            for f in load_spec()["header"]}


def statics(row, keys: dict, avail: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(key ids, values, available-from minutes) of one stay's statics (``row``: its stays-table row)."""
    k, val, at = [], [], []
    ds = row["dataset"]
    for f in STATICS:
        x = row.get(f)
        if x is None or (isinstance(x, float) and np.isnan(x)) or x in ("UNK", "unknown"):
            continue
        key = f if f in NUMERIC_STATICS else f"{f}={x}"
        if key not in keys:
            continue
        k.append(keys[key])
        val.append(float(x) if f in NUMERIC_STATICS else 1.0)
        at.append(avail.get(f, {}).get(ds, 0))
    return np.asarray(k, dtype=np.int16), np.asarray(val, dtype=np.float32), np.asarray(at, dtype=np.int32)


class Reader:
    """Trajectories of one store (``stats``: the training twins' channel statistics for ``ev_z``)."""

    def __init__(self, store, v, stats: dict | None = None):
        ch = store.channels
        self.store = store
        self.input = np.array([ch.is_input(i) for i in range(len(ch))], dtype=bool)
        self.thinnable = np.array([g in THINNABLE for g in ch.group], dtype=bool)
        stats = stats if stats is not None else ch.stats
        self.median = np.full(len(ch), np.nan, dtype=np.float32)
        self.scale = np.full(len(ch), np.nan, dtype=np.float32)
        for name, s in (stats or {}).items():
            if name in ch.index:
                spread = (s["p95"] - s["p05"]) / 3.29
                self.median[ch.index[name]] = s["median"]
                self.scale[ch.index[name]] = spread if spread > 0 else 1.0
        self.keys = {k: j for j, k in enumerate(static_vocab(v))}
        self.avail = availability()
        self.rows = store.stays.to_dict("records")

    def trajectory(self, i: int, until: int | None = None, budget: int | None = None) -> dict:
        ev, iv = self.store.events(i), self.store.intervals(i)
        t, c = np.asarray(ev["t"], dtype=np.int64), np.asarray(ev["c"], dtype=np.int64)
        keep = self.input[c]
        cut = int(self.store.los[i]) if until is None else until   # a whole stay ends at its ICU discharge or death
        keep &= t <= cut
        idx = np.flatnonzero(keep)
        f = np.asarray(ev["f"], dtype=np.uint8)[idx].copy()
        if budget is not None and len(idx) > budget:
            idx, f = self._thin(idx, t, c, f, budget, query=until is not None)
        v = np.asarray(ev["v"], dtype=np.float32)[idx]
        cc = c[idx]
        z = (v - self.median[cc]) / self.scale[cc]
        s, e, ic = (np.asarray(iv[k], dtype=np.int64) for k in ("s", "e", "c"))
        ik = s < cut                                     # a state starting at the cut: its event
        row = self.rows[i]
        sk, sv, sa = statics(row, self.keys, self.avail)
        if until is not None:
            ok = sa <= until
            sk, sv, sa = sk[ok], sv[ok], sa[ok]
        out = {"ev_t": t[idx].astype(np.int32), "ev_c": cc.astype(np.int32), "ev_v": v, "ev_z": z.astype(np.float32),
               "ev_bin": np.asarray(ev["bin"], dtype=np.uint8)[idx], "ev_f": f,
               "iv_c": ic[ik].astype(np.int32), "iv_s": s[ik].astype(np.int32),
               "iv_e": np.minimum(e[ik], cut).astype(np.int32),
               "iv_v": np.asarray(iv["v"], dtype=np.float32)[ik],
               "st_k": sk, "st_v": sv, "st_avail": sa}
        if until is None:
            out["end_t"] = np.int32(self.store.los[i])
            out["end_kind"] = str(self.store.end_kind[i])
        return out

    def _thin(self, idx, t, c, f, budget: int, query: bool):
        """Thin the frequent measurement groups to fit ``budget`` (module docstring)."""
        thin = self.thinnable[c[idx]]
        n_keep = budget - int((~thin).sum())
        if n_keep <= 0:                                  # the protected events alone exceed the budget: the latest
            sel = np.sort(idx[~thin][-budget:]) if query else np.sort(idx[~thin][:budget])
            return sel, np.asarray(f)[np.searchsorted(idx, sel)]
        cand = idx[thin]
        if query:                                        # a query keeps its most recent measurements
            chosen = cand[-n_keep:]
        else:                                            # a context patient keeps every k-th of each channel
            k = int(np.ceil(len(cand) / n_keep))
            pos = np.zeros(len(cand), dtype=np.int64)
            cc = c[cand]
            for ch in np.unique(cc):
                m = cc == ch
                pos[m] = np.arange(m.sum())
            chosen = cand[pos % k == 0]
            if len(chosen) > n_keep:
                chosen = np.sort(np.random.default_rng(0).choice(chosen, size=n_keep, replace=False))
        sel = np.sort(np.concatenate([idx[~thin], chosen]))
        flags = np.asarray(f)[np.searchsorted(idx, sel)].copy()
        thinned_ch = np.unique(c[cand][~np.isin(cand, chosen)])
        flags[np.isin(c[sel], thinned_ch)] |= THINNED
        return sel, flags


def anonymize(trajs: list[dict], rng: np.random.Generator) -> tuple[list[dict], np.ndarray]:
    """Episode-local channel ids (a random permutation of the channels the episode's trajectories use) and values
    standardized only (``ev_v`` replaced by ``ev_z``); returns the trajectories and the map local -> global id."""
    used = np.unique(np.concatenate([np.concatenate([t["ev_c"], t["iv_c"]]) for t in trajs]))
    perm = rng.permutation(len(used))
    local = dict(zip(used.tolist(), perm.tolist()))
    out = []
    for t in trajs:
        u = dict(t)
        u["ev_c"] = np.array([local[x] for x in t["ev_c"]], dtype=np.int32)
        u["iv_c"] = np.array([local[x] for x in t["iv_c"]], dtype=np.int32)
        u["ev_v"] = t["ev_z"].copy()
        out.append(u)
    inverse = np.empty(len(used), dtype=np.int64)
    inverse[perm] = used
    return out, inverse
