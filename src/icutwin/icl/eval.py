"""The learning confirmation's checks (plan Phase G, "Learning confirmation", pre-registered before the pilot runs).
Every comparison is on the same episodes and rows, per episode (an episode's mean loss over its valid targets):

* L1 it learns: twin-val episodes, by label type, the model against the base-rate baseline (the context's own
  label distribution: ``heads.baseline_losses``): the mean per-episode loss difference with a bootstrap 95% interval,
  which must lie below 0 for every label type with at least 30 episodes;
* L2 it learns from context: twin-val episodes with at least 16 context patients, rerun with uniform random
  subsets of 0 (descriptor shown only), 2, 4, 8 and 16 of them, same queries: the mean loss must not rise from one
  size to the next (1% tolerance) and must fall from the smallest to the largest (paired interval below 0);
* L3 it reads base rates: binary twin-val episodes with at least two positive and two negative context patients,
  rerun with half the positive (lower rate) or half the negative (higher rate) context patients removed: within each
  episode, the mean predicted probability against the context's prevalence; the pooled within-episode slope must be
  above 0 (bootstrap interval excluding 0);
* L4 it reads tasks from examples: hidden-descriptor and anonymized twin-val episodes, each stratum against the
  base rate (all label types pooled; interval below 0);
* L5 it transfers: real dev episodes with real contexts against the base rate (pooled, interval below 0; by label
  type reported), and the gap to twin-val;
* L6 memorization: the model's mean loss on twin-train episodes (the training twins, unseen episodes) against
  twin-val episodes (unseen twins), reported.

Reported, not gated: AUROC of binary labels against the baseline's (which ranks nothing within an episode, so it
is 0.5 by construction; the GBDT and tabular foundation-model baselines are separate, ``tools/icl_baselines.py``).
"""

from __future__ import annotations

import numpy as np
import torch

from icutwin.icl import batch as B
from icutwin.icl import heads as H
from icutwin.icl import infer


def episode_results(model, builder, eps: list, device, autocast, batch_episodes: int = 8) -> list[dict]:
    """Per episode, in the order of ``eps``: label type, the model's and the baseline's mean losses, flags, and for
    binary labels the predicted probabilities and targets of its valid rows."""
    from icutwin.icl.batch import to_torch
    from icutwin.icl.train import to_device

    model.eval()
    out = []
    for k in range(0, len(eps), batch_episodes):
        chunk = eps[k:k + batch_episodes]
        b = to_device(to_torch(builder(chunk)), device)
        with torch.no_grad(), autocast():
            res = model(b)
        n_ep = int(b["n_episodes"])
        mine, has = H.episode_mean(res["row_loss"].float(), b["te_ep"], n_ep)
        base, _ = H.episode_mean(H.baseline_losses(b, model.cfg.bins), b["te_ep"], n_ep)
        probs = torch.full((len(b["te_ep"]),), float("nan"), device=device)
        for name in ("binary", "hosp_death"):
            if name in res["out"]:
                m = b["ep_label"][b["te_ep"]] == B.LABELS.index(name)
                probs[m] = torch.sigmoid(res["out"][name].float())
        yhat = torch.full((len(b["te_ep"]),), float("nan"), device=device, dtype=torch.float64)
        if "regression" in res["out"]:                       # the bar distribution's median, in channel units
            m = b["ep_label"][b["te_ep"]] == B.LABELS.index("regression")
            cdf = torch.softmax(res["out"]["regression"].float(), -1).cumsum(-1)
            e_ = H.bar_edges(cdf.shape[-1], device)
            u = ((e_[:-1] + e_[1:]) / 2)[(cdf < 0.5).sum(-1).clamp_max(cdf.shape[-1] - 1)].double()
            st = b["ep_reg"][b["te_ep"][m]].double()
            yhat[m] = st[:, 0] + st[:, 1] * torch.sign(u) * torch.expm1(u.abs())
        for e, ep in enumerate(chunk):
            rows = (b["te_ep"] == e) & b["te_valid"]
            rec = {"index": int(ep["recipe"]["index"]), "label": B.LABELS[int(b["ep_label"][e])],
                   "has": bool(has[e]), "model": float(mine[e]), "base": float(base[e]),
                   "shown": ep["task"]["descriptor"] is not None, "anonymized": bool(ep["task"]["anonymized"]),
                   "n_context": len(ep["recipe"]["context"])}
            if rec["label"] in ("binary", "hosp_death"):
                rec["p"] = probs[rows].float().cpu().numpy()
                rec["y"] = b["te_y"][rows].float().cpu().numpy()
            elif rec["label"] == "regression":
                q = ep["query"]
                rec["yhat"] = yhat[rows].cpu().numpy()
                rec["y"] = np.asarray(q["lab_y"], dtype=np.float64)[np.asarray(q["lab_valid"], dtype=bool)]
            out.append(rec)
    model.train()
    return out


def bootstrap_mean(x, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _gain(rows: list) -> dict:
    d = [r["model"] - r["base"] for r in rows if r["has"] and np.isfinite(r["base"])]
    m, lo, hi = bootstrap_mean(d)
    return {"episodes": len(d), "model": float(np.mean([r["model"] for r in rows if r["has"]])) if d else None,
            "base_rate": float(np.mean([r["base"] for r in rows if r["has"] and np.isfinite(r["base"])])) if d else None,
            "diff": m, "ci": [lo, hi], "better_share": float(np.mean(np.asarray(d) < 0)) if d else None}


def auroc(p, y) -> float | None:
    from scipy.stats import rankdata

    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    pos = y > 0.5
    if pos.all() or not pos.any():
        return None
    r = rankdata(p)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def check_l1(rows: list) -> dict:
    out = {}
    for kind in sorted({r["label"] for r in rows}):
        out[kind] = _gain([r for r in rows if r["label"] == kind])
        aucs = [auroc(r["p"], r["y"]) for r in rows if r["label"] == kind and "p" in r]
        aucs = [a for a in aucs if a is not None]
        if aucs:
            out[kind]["auroc_median_within_episode"] = float(np.median(aucs))
    scored = {k: v for k, v in out.items() if v["episodes"] >= 30}
    out["pass"] = bool(scored) and all(v["ci"][1] < 0 for v in scored.values())
    return out


def check_l2(model, builder, eps: list, device, autocast, sizes=(0, 2, 4, 8, 16), seed: int = 0) -> dict:
    pool = [e for e in eps if len(e["recipe"]["context"]) >= max(sizes)]
    rng = np.random.default_rng(seed)
    variants, keys = [], []
    for e in pool:
        n = len(e["recipe"]["context"])
        order = list(rng.permutation(n))
        for s in sizes:
            if s == 0 and e["task"]["descriptor"] is None:
                continue
            variants.append(infer.select_context(e, sorted(order[:s])))   # same episode index: same channel draws
            keys.append((int(e["recipe"]["index"]), s))
    res = episode_results(model, builder, variants, device, autocast)
    by: dict = {}
    for (idx, s), r in zip(keys, res, strict=True):
        if r["has"]:
            by.setdefault(idx, {})[s] = r["model"]
    curve = {}
    nonzero = [s for s in sizes if s > 0]
    for s in sizes:
        xs = [d[s] for d in by.values() if s in d and all(t in d for t in nonzero)]
        curve[s] = {"episodes": len(xs), "loss": float(np.mean(xs)) if xs else None}
    lo_s, hi_s = nonzero[0], nonzero[-1]
    diffs = [d[hi_s] - d[lo_s] for d in by.values() if lo_s in d and hi_s in d]
    m, lo, hi = bootstrap_mean(diffs)
    seq = [curve[s]["loss"] for s in sizes if curve[s]["loss"] is not None]
    mono = all(b <= a * 1.01 + 1e-9 for a, b in zip(seq, seq[1:], strict=False))
    return {"curve": curve, "largest_minus_smallest": {"diff": m, "ci": [lo, hi], "episodes": len(diffs)},
            "monotone": mono, "pass": bool(mono and hi < 0)}


def check_l3(model, builder, eps: list, device, autocast, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    variants, meta = [], []
    for e in eps:
        if e["task"]["label"] not in ("binary", "hosp_death"):
            continue
        c = e["context"]
        lo = np.asarray(c["lab_off"])
        y = np.asarray(c["lab_y"], dtype=np.float64)
        ok = np.asarray(c["lab_valid"], dtype=bool)
        pos = [k for k in range(len(lo) - 1) if (np.nan_to_num(y[lo[k]:lo[k + 1]]) * ok[lo[k]:lo[k + 1]]).any()]
        neg = [k for k in range(len(lo) - 1) if k not in pos and ok[lo[k]:lo[k + 1]].any()]
        if len(pos) < 2 or len(neg) < 2:
            continue
        n_ctx = len(lo) - 1
        drop_pos = set(rng.choice(pos, size=len(pos) // 2, replace=False).tolist())
        drop_neg = set(rng.choice(neg, size=len(neg) // 2, replace=False).tolist())
        for tag, drop in (("low", drop_pos), ("mid", set()), ("high", drop_neg)):
            keep = [k for k in range(n_ctx) if k not in drop]
            v = infer.select_context(e, keep)
            cy, cv = np.asarray(v["context"]["lab_y"], dtype=np.float64), np.asarray(v["context"]["lab_valid"], bool)
            variants.append(v)
            meta.append((int(e["recipe"]["index"]), tag, float(np.nan_to_num(cy[cv]).mean()) if cv.any() else np.nan))
    res = episode_results(model, builder, variants, device, autocast)
    pts: dict = {}
    for (idx, _tag, prev), r in zip(meta, res, strict=True):
        if "p" in r and len(r["p"]):
            pts.setdefault(idx, []).append((prev, float(np.mean(r["p"]))))
    slopes = []
    for xs in pts.values():
        a = np.asarray(xs)
        if len(a) == 3 and np.ptp(a[:, 0]) > 0:
            slopes.append(float(np.polyfit(a[:, 0], a[:, 1], 1)[0]))
    m, lo, hi = bootstrap_mean(slopes)
    return {"episodes": len(slopes), "slope": m, "ci": [lo, hi], "pass": bool(len(slopes) >= 20 and lo > 0)}


def check_l4(rows: list) -> dict:
    out = {"hidden_descriptor": _gain([r for r in rows if not r["shown"]]),
           "anonymized": _gain([r for r in rows if r["anonymized"]])}
    out["pass"] = all(v["episodes"] >= 30 and v["ci"][1] < 0 for v in out.values())
    return out


def check_l5(real: list, twin: list) -> dict:
    pooled = _gain(real)
    by = {k: _gain([r for r in real if r["label"] == k]) for k in sorted({r["label"] for r in real})}
    tw = _gain(twin)
    return {"pooled": pooled, "by_label": by,
            "gap_model_loss_real_minus_twin": (pooled["model"] - tw["model"]) if pooled["model"] is not None
            and tw["model"] is not None else None,
            "pass": bool(pooled["episodes"] >= 30 and pooled["ci"][1] < 0)}


def check_l6(train_rows: list, val_rows: list) -> dict:
    a = [r["model"] for r in train_rows if r["has"]]
    b = [r["model"] for r in val_rows if r["has"]]
    return {"twin_train_loss": float(np.mean(a)) if a else None, "twin_val_loss": float(np.mean(b)) if b else None,
            "gap_val_minus_train": float(np.mean(b) - np.mean(a)) if a and b else None}
