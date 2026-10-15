"""Inference helpers (plan Phase G, Architecture 8; minimal in this round): a query as known at one of its prediction
times, which the tests use as the definition of causality (the model's output at t_k must equal its output on the
episode cut at t_k), and a batch prediction for evaluation.

The bedside path proper (the context bank's cached keys and values; a query re-encoded patch by patch) follows the
same contract and comes with the main run.
"""

from __future__ import annotations

import copy

import numpy as np

from icutwin.factory.contracts import _GROUPS, _LABEL_FIELDS


def _slice_part(part: dict, k: int, keep: dict) -> dict:
    """``part`` with patient ``k``'s rows of each ragged group restricted to the boolean masks in ``keep``."""
    out = dict(part)
    groups = dict(_GROUPS)
    groups["lab"] = _LABEL_FIELDS + tuple(x for x in ("lab_yh", "lab_mh", "lab_yf", "lab_mf") if x in part)
    groups["own_lab"] = tuple(f"own_{x}" for x in _LABEL_FIELDS) + tuple(
        f"own_{x}" for x in ("lab_yh", "lab_mh", "lab_yf", "lab_mf") if f"own_{x}" in part)
    for g, mask in keep.items():
        off = np.asarray(part[f"{g}_off"])
        a, b = int(off[k]), int(off[k + 1])
        sel = np.concatenate([np.arange(a), a + np.flatnonzero(mask), np.arange(b, int(off[-1]))])
        for key in groups[g]:
            out[key] = np.asarray(part[key])[sel]
        new = off.copy()
        new[k + 1:] -= (b - a) - int(mask.sum())
        out[f"{g}_off"] = new
    return out


def truncate_query(ep: dict, j: int, t: int) -> dict:
    """The episode with query ``j`` as the factory would give it with ``t`` as its last prediction time: events by
    t, states started before t (clipped there), statics documented by t, own labels at times up to t, and only its
    targets at times up to t. Other queries and the context are unchanged."""
    ep = copy.deepcopy(ep)
    q = ep["query"]
    off = {g: np.asarray(q[f"{g}_off"]) for g in ("ev", "iv", "st", "lab", "own_lab")}
    sl = {g: slice(int(o[j]), int(o[j + 1])) for g, o in off.items()}
    keep = {"ev": np.asarray(q["ev_t"][sl["ev"]]) <= t,
            "iv": np.asarray(q["iv_s"][sl["iv"]]) < t,
            "st": np.asarray(q["st_avail"][sl["st"]]) <= t,
            "lab": np.asarray(q["lab_t"][sl["lab"]]) <= t,
            "own_lab": np.asarray(q["own_lab_t"][sl["own_lab"]]) <= t}
    iv_e = np.asarray(q["iv_e"]).copy()
    iv_e[sl["iv"]] = np.minimum(iv_e[sl["iv"]], t)
    q = dict(q, iv_e=iv_e)
    ep["query"] = _slice_part(q, j, keep)
    times = [int(x) for x in ep["recipe"]["queries"][j]["times"] if x <= t]
    ep["recipe"]["queries"][j] = dict(ep["recipe"]["queries"][j], times=times)
    return ep


def select_context(ep: dict, keep: list) -> dict:
    """The episode with only the context patients ``keep`` (in that order): a subset for context-size curves, or a
    permutation."""
    ep = copy.deepcopy(ep)
    c = ep["context"]
    groups = dict(_GROUPS)
    groups["lab"] = _LABEL_FIELDS + tuple(x for x in ("lab_yh", "lab_mh", "lab_yf", "lab_mf") if x in c)
    for g, keys in groups.items():
        off = np.asarray(c[f"{g}_off"])
        idx = (np.concatenate([np.arange(off[k], off[k + 1]) for k in keep]) if len(keep)
               else np.zeros(0)).astype(np.int64)
        for key in keys:
            c[key] = np.asarray(c[key])[idx]
        c[f"{g}_off"] = np.concatenate([[0], np.cumsum([off[k + 1] - off[k] for k in keep])]).astype(np.int64)
    c["end_t"] = np.asarray(c["end_t"])[list(keep)] if len(keep) else np.zeros(0, dtype=np.int32)
    c["end_kind"] = [c["end_kind"][k] for k in keep]
    ep["recipe"]["context"] = [ep["recipe"]["context"][k] for k in keep]
    return ep


def predict(model, builder, eps: list, device=None) -> dict:
    """The model's outputs on ``eps`` (no gradient): the test rows' states, per-row losses and the batch."""
    import torch

    from icutwin.icl.batch import to_torch

    b = to_torch(builder(eps), device)
    model.eval()
    with torch.no_grad():
        res = model(b)
    return {"batch": b, **res}
