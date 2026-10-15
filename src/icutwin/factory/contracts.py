"""Verification layer (d) of the structured-output guarantee (plan *Structured-output guarantee* 4): the contract of
every factory batch, checked before it leaves the factory (the production gate is no failure).

TabICL's ``get_batch`` layout ``X[B, T, H]``, ``y[B, T]``, ``d[B]``, ``seq_lens[B]``, ``train_sizes[B]``:
finite float32 values; 1 <= d <= H <= ``max_features``; seq_lens <= T; context rows within [0.1, 0.9] of the rows;
zero padding outside each episode; no feature constant over the context rows; classification labels numbered
0..k-1 with k <= ``max_classes``, every class in the context and in the query; regression labels finite.
"""

from __future__ import annotations

import numpy as np


class ContractError(ValueError):
    pass


def batch_failures(x, y, d, seq_lens, train_sizes, families: list, max_features: int = 100,
                   max_classes: int = 10) -> list:
    """Every way a batch breaks the contract (an empty list: none)."""
    x, y = np.asarray(x), np.asarray(y)
    d, seq, tr = (np.asarray(a) for a in (d, seq_lens, train_sizes))
    out = []
    if x.ndim != 3 or y.shape != x.shape[:2] or not (len(d) == len(seq) == len(tr) == len(families) == x.shape[0]):
        return [f"shapes: X {x.shape}, y {y.shape}, d {d.shape}, seq_lens {seq.shape}, train_sizes {tr.shape}"]
    if x.dtype != np.float32 or y.dtype != np.float32:
        out.append(f"dtypes {x.dtype}, {y.dtype}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        out.append("non-finite values")
    b, t, h = x.shape
    if h > max_features:
        out.append(f"H {h} > {max_features}")
    for i in range(b):
        n, k, m = int(seq[i]), int(d[i]), int(tr[i])
        where = f"episode {i}"
        if not 1 <= k <= h or not 1 <= n <= t:
            out.append(f"{where}: d {k}, rows {n} outside the batch's H {h}, T {t}")
            continue
        if not 0.1 * n - 1 <= m <= 0.9 * n + 1:
            out.append(f"{where}: {m} context rows of {n}")
        if (x[i, n:] != 0).any() or (x[i, :, k:] != 0).any() or (y[i, n:] != 0).any():
            out.append(f"{where}: padding not zero")
        xi, yi = x[i, :n, :k], y[i, :n]
        if m > 0 and (np.ptp(xi[:m], axis=0) == 0).any():
            out.append(f"{where}: a feature constant over the context")
        if families[i] != "regression":
            cls = np.unique(yi)
            if len(cls) > max_classes or not np.array_equal(cls, np.arange(len(cls))):
                out.append(f"{where}: classes {cls.tolist()[:12]}")
            elif set(np.unique(yi[:m])) != set(cls) or set(np.unique(yi[m:])) != set(cls):
                out.append(f"{where}: a class missing from the context or the query")
    return out


def check_batch(*batch, families: list, **kw) -> None:
    fails = batch_failures(*batch, families=families, **kw)
    if fails:
        raise ContractError("; ".join(fails[:10]))


# -- trajectory episodes (plan Phase F, "Task formats") -----------------------------------------------------------
EPISODE_LABELS = ("binary", "tte", "ordinal", "regression", "hosp_death", "multihorizon", "forecast")
_GROUPS = (("ev", ("ev_t", "ev_c", "ev_v", "ev_z", "ev_bin", "ev_f")), ("iv", ("iv_c", "iv_s", "iv_e", "iv_v")),
           ("st", ("st_k", "st_v", "st_avail")))
_LABEL_FIELDS = ("lab_t", "lab_y", "lab_valid", "lab_reason", "lab_t_event", "lab_cause", "lab_t_censor",
                 "lab_t_resolved")


def _ragged(d: dict, off_key: str, keys: tuple, where: str, out: list) -> int:
    off = np.asarray(d[off_key])
    if off.ndim != 1 or len(off) == 0 or off[0] != 0 or (np.diff(off) < 0).any():
        out.append(f"{where}: offsets {off_key} not 0-based and non-decreasing")
        return 0
    for k in keys:
        if len(d[k]) != off[-1]:
            out.append(f"{where}: {k} has {len(d[k])} rows, offsets say {off[-1]}")
    return len(off) - 1


def _labels(d: dict, prefix: str, label: str, bands: list, where: str, out: list) -> None:
    y, ok = np.asarray(d[f"{prefix}y"], dtype=np.float64), np.asarray(d[f"{prefix}valid"], dtype=bool)
    if np.isfinite(y[~ok]).any():
        out.append(f"{where}: a label where it does not apply")
    if label == "forecast":                               # the targets are the [n, H, C] values and their mask
        yf, mf = d.get(f"{prefix}yf"), d.get(f"{prefix}mf")
        if len(y) == 0:
            return
        if yf is None or mf is None or len(yf) != len(y) or np.asarray(yf).shape != np.asarray(mf).shape:
            out.append(f"{where}: forecast targets missing or misshapen")
        elif (not np.isfinite(np.asarray(yf)[np.asarray(mf)]).all()
              or (np.asarray(mf).reshape(len(y), -1).any(1) < ok).any()):
            out.append(f"{where}: forecast values missing where observed, or a valid row without any")
        return
    if label == "multihorizon":
        yh, mh = d.get(f"{prefix}yh"), d.get(f"{prefix}mh")
        if yh is None or mh is None or len(yh) != len(y) or np.asarray(yh).shape[1:] != (len(bands),):
            out.append(f"{where}: multi-horizon targets missing or misshapen")
        elif not np.isin(np.asarray(yh)[np.asarray(mh)], (0.0, 1.0)).all():
            out.append(f"{where}: multi-horizon labels outside 0/1")
    if not np.isfinite(y[ok]).all():
        out.append(f"{where}: a missing label where it applies")
    if label in ("binary", "hosp_death", "tte", "multihorizon") and not np.isin(y[ok], (0.0, 1.0)).all():
        out.append(f"{where}: binary labels outside 0/1")
    if label == "ordinal" and (not np.isin(y[ok], np.arange(len(bands) + 1)).all()):
        out.append(f"{where}: classes outside 0..{len(bands)}")
    t, tr = np.asarray(d[f"{prefix}t"]), np.asarray(d[f"{prefix}t_resolved"])
    if (tr[ok] <= t[ok]).any():
        out.append(f"{where}: a label resolved at or before its time (it would show as its own history)")


def episode_failures(ep: dict, n_channels: int | None = None) -> list:
    """Every way a trajectory episode breaks its contract (an empty list: none): ragged arrays consistent with their
    offsets; events in time order, known channels, finite values where flagged, bins 0-9 or 255; intervals that end
    after they start; labels present exactly where they apply, in their type's range, never resolved before their
    time; a query's events, intervals and statics available by its last prediction time, its targets at the
    recipe's times, its own labels at or before that time; context patients' labels inside their stay; context and
    query patients disjoint; no context without a shown descriptor (or a forecast); a context's selection declared
    (``task.selection.case_odds``, at least 1)."""
    out: list = []
    task, r = ep["task"], ep["recipe"]
    label = task["label"]
    if label not in EPISODE_LABELS:
        return [f"label type {label!r}"]
    n_ch = n_channels if n_channels is not None else ep["channels"]["n"]
    sizes = {}
    for part in ("context", "query"):
        d = ep[part]
        n = {g: _ragged(d, f"{g}_off", keys, part, out) for g, keys in _GROUPS}
        nl = _ragged(d, "lab_off", _LABEL_FIELDS, part, out)
        if len(set(n.values()) | {nl}) > 1:
            out.append(f"{part}: patients per group differ {n} / labels {nl}")
        sizes[part] = nl
        off, t = np.asarray(d["ev_off"]), np.asarray(d["ev_t"])
        for a, b in zip(off[:-1], off[1:]):
            if (np.diff(t[a:b]) < 0).any():
                out.append(f"{part}: events out of time order")
                break
        c = np.asarray(d["ev_c"])
        if len(c) and (c.min() < 0 or c.max() >= n_ch):
            out.append(f"{part}: channel ids outside 0..{n_ch - 1}")
        f, v = np.asarray(d["ev_f"]), np.asarray(d["ev_v"])
        if not np.isfinite(v[(f & 1) == 1]).all():
            out.append(f"{part}: a flagged value is not finite")
        bins = np.asarray(d["ev_bin"])
        if not ((bins <= 9) | (bins == 255)).all():
            out.append(f"{part}: bins outside 0-9 / 255")
        if (np.asarray(d["iv_e"]) <= np.asarray(d["iv_s"])).any():
            out.append(f"{part}: an interval that does not end after it starts")
        _labels(d, "lab_", label, task["bands"], part, out)
    ctx, q = ep["context"], ep["query"]
    end = np.asarray(ctx["end_t"])
    if len(end) != sizes["context"]:
        out.append("context: end times per patient")
    else:
        lo = np.asarray(ctx["lab_off"])
        for k in range(len(end)):
            lt = np.asarray(ctx["lab_t"])[lo[k]:lo[k + 1]]
            if len(lt) and (lt.min() < 0 or lt.max() >= end[k]):
                out.append("context: a label time outside its stay")
                break
            et = np.asarray(ctx["ev_t"])[ctx["ev_off"][k]:ctx["ev_off"][k + 1]]
            if len(et) and et.max() > end[k]:
                out.append("context: an event after the stay's end")
                break
    _ragged(q, "own_lab_off", tuple(f"own_{k}" for k in _LABEL_FIELDS), "query own", out)
    _labels(q, "own_lab_", label, task["bands"], "query own", out)
    qo, lo, oo = np.asarray(q["ev_off"]), np.asarray(q["lab_off"]), np.asarray(q["own_lab_off"])
    for k, rq in enumerate(r["queries"]):
        times = np.asarray(rq["times"])
        tk = times.max() if len(times) else -1
        if not np.array_equal(np.asarray(q["lab_t"])[lo[k]:lo[k + 1]], times):
            out.append(f"query {k}: targets not at the recipe's times")
        if (np.asarray(q["ev_t"])[qo[k]:qo[k + 1]] > tk).any():
            out.append(f"query {k}: an event after its last prediction time")
        so = np.asarray(q["st_off"])
        if (np.asarray(q["st_avail"])[so[k]:so[k + 1]] > tk).any():
            out.append(f"query {k}: a static not yet available")
        io = np.asarray(q["iv_off"])
        if (np.asarray(q["iv_s"])[io[k]:io[k + 1]] > tk).any() or (np.asarray(q["iv_e"])[io[k]:io[k + 1]] > tk).any():
            out.append(f"query {k}: an interval past its last prediction time")
        if (np.asarray(q["own_lab_t"])[oo[k]:oo[k + 1]] > tk).any():
            out.append(f"query {k}: an own label after its last prediction time")
    if set(r["context"]) & {x["stay"] for x in r["queries"]}:
        out.append("a patient both in the context and in the query")
    if not r["context"] and not r["descriptor"]:
        out.append("no context and no descriptor")
    odds = (task.get("selection") or {}).get("case_odds")
    if r["context"] and not (isinstance(odds, (int, float)) and odds >= 1):
        out.append("a context whose selection is not declared (case_odds)")
    return out


def check_episode(ep: dict, n_channels: int | None = None) -> None:
    fails = episode_failures(ep, n_channels)
    if fails:
        raise ContractError("; ".join(fails[:10]))
