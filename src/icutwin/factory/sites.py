"""Sites and realism (plan Phase F, Factory design 6; the study lead's Q&A 13: "make sure it isn't too difficult").

At most one site-level variant per episode (``draw_site``):
* ``plain``: one dataset as it is;
* ``virtual``: one transform of a dataset: ``case_mix`` (patients drawn with a tilt toward a header stratum),
  ``thinning`` (a share of the frequent measurements never charted), ``rounding`` (values charted coarser) or
  ``label_noise`` (a share of binary labels flipped, as charting errors would);
* ``label_shift``: positive trajectories enriched 2-10x in context and query alike (a case-control site);
* ``cross_site``: context patients from one dataset, queries from another (a new hospital).

Realism transforms (``draw_realism``, half of the episodes, at most two, mild): measurements missing completely at
random, multiplicative value noise, reporting delays (a lab visible only some minutes after its time), rare
outliers. Statistics they need come from the context only. A curriculum (``ramp``) halves the shares and keeps the
strengths at their low end early in a stage.

The difficulty control: ``snapshot_loss`` measures what a perturbation costs a probe on the episode's tabular
snapshot (rank AUROC, or Spearman for values, of the best feature, with and without the perturbation); a virtual site
or realism transform over its cap is drawn again at a lower strength (``Sampler`` in ``factory.episodes``).
"""

from __future__ import annotations

import numpy as np

from icutwin.factory.trajectories import THINNABLE


def _pick(rng, shares: dict) -> str:
    keys = list(shares)
    p = np.asarray([shares[k] for k in keys], dtype=np.float64)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def draw_site(rng, cfg: dict, datasets: list, ramp: float = 1.0, scale: float = 1.0) -> dict:
    """A site-level variant (module docstring); ``ramp`` < 1 halves the non-plain shares early in a stage; ``scale``
    (<= 1) lowers the strengths (a retry after the difficulty cap)."""
    shares = dict(cfg["sites"])
    if ramp < 1:
        for k in shares:
            if k != "plain":
                shares[k] *= 0.5
        shares["plain"] = 1 - sum(v for k, v in shares.items() if k != "plain")
    kind = _pick(rng, shares)
    if kind == "cross_site" and len(datasets) < 2:
        kind = "plain"
    out = {"kind": kind}
    if kind == "virtual":
        t = _pick(rng, cfg["virtual"])
        out["transform"] = t
        if t == "case_mix":
            out["stratum"] = str(rng.choice(["age_old", "age_young", "surgical", "medical"]))
            out["tilt"] = float(scale * rng.uniform(0.5, 2.0))
        elif t == "thinning":
            out["drop"] = float(scale * rng.uniform(0.2, 0.6))
        elif t == "rounding":
            out["digits"] = int(rng.choice([0, 1]))
        elif t == "label_noise":
            out["flip"] = float(scale * rng.uniform(0.01, 0.05))
    elif kind == "label_shift":
        out["enrich"] = float(1 + scale * (rng.uniform(2, 10) - 1))
    elif kind == "cross_site":
        a, b = rng.choice(datasets, size=2, replace=False)
        out["context_site"], out["query_site"] = str(a), str(b)
    return out


def draw_realism(rng, cfg: dict, ramp: float = 1.0, scale: float = 1.0) -> dict:
    """Realism transforms (module docstring): none, or one or two of ``mcar``, ``noise``, ``delay``, ``outliers``."""
    r = cfg["realism"]
    share = r["share"] * (0.5 if ramp < 1 else 1.0)
    if rng.random() >= share:
        return {}
    k = int(rng.integers(1, r["max_transforms"] + 1))
    out = {}
    low = 0.0 if ramp < 1 else 1.0
    for t in rng.choice(["mcar", "noise", "delay", "outliers"], size=k, replace=False):
        lo, hi = r[t]
        hi = lo + (hi - lo) * (0.5 if low == 0 else 1.0)
        out[str(t)] = float(lo + scale * rng.uniform(0, 1) * (hi - lo))
    return out


def apply(trajs: list[dict], site: dict, realism: dict, thinnable: np.ndarray, rng, query_cut: list | None = None
          ) -> list[dict]:
    """The measurement-level transforms of a site and of realism on materialized trajectories (copies): thinning,
    rounding, MCAR, noise, outliers on numeric values of the thinnable groups (labs included for MCAR and noise);
    delays hide a query's labs charted less than ``delay`` minutes before its cut (``query_cut``: the cut per
    trajectory, None for a context patient). ``ev_z`` follows ``ev_v`` by the same affine map."""
    out = []
    for k, t in enumerate(trajs):
        u = dict(t)
        n = len(t["ev_t"])
        keep = np.ones(n, dtype=bool)
        c = t["ev_c"]
        thin = thinnable[c] if n else np.zeros(0, dtype=bool)
        has = (t["ev_f"] & 1) == 1
        v, z = t["ev_v"].astype(np.float64).copy(), t["ev_z"].astype(np.float64).copy()
        a = np.where(np.abs(v) > 0, z / np.where(v == 0, 1, v), 0.0)      # z = a v + b, per event
        bb = z - a * v
        if site.get("transform") == "thinning":
            keep &= ~(thin & (rng.random(n) < site["drop"]))
        if site.get("transform") == "rounding":
            v = np.where(has & thin, np.round(v, site["digits"]), v)
        if "mcar" in realism:
            keep &= ~(has & (rng.random(n) < realism["mcar"]))
        if "noise" in realism:
            v = np.where(has, v * (1 + rng.normal(0, realism["noise"], n)), v)
        if "outliers" in realism:
            v = np.where(has & (rng.random(n) < realism["outliers"]), v * rng.choice([0.1, 10.0], n), v)
        if "delay" in realism and query_cut is not None and query_cut[k] is not None:
            lab = ~thin & has
            keep &= ~(lab & (t["ev_t"] > query_cut[k] - realism["delay"]))
        z = np.where(has, a * v + bb, z)
        for key in ("ev_t", "ev_c", "ev_bin", "ev_f"):
            u[key] = t[key][keep]
        u["ev_v"] = v[keep].astype(np.float32)
        u["ev_z"] = z[keep].astype(np.float32)
        out.append(u)
    return out


def flip_labels(y: np.ndarray, valid: np.ndarray, p: float, rng) -> np.ndarray:
    """Binary labels with a share ``p`` of the valid ones flipped (a site's label noise)."""
    y = y.copy()
    f = valid & (rng.random(len(y)) < p) & np.isfinite(y)
    y[f] = 1.0 - y[f]
    return y


def snapshot_loss(x: np.ndarray, y: np.ndarray, binary: bool, site: dict, realism: dict, rng) -> float:
    """What a site's or realism's measurement perturbation costs a probe: the best single feature's rank AUROC (or
    |Spearman|) on the snapshot, clean against perturbed (thinning and MCAR as missing values, noise, rounding,
    outliers on the values; label noise on the labels)."""
    from scipy.stats import rankdata

    def best(xx, yy):
        out = 0.0
        ry = rankdata(yy)
        pos = yy > 0
        for j in range(xx.shape[1]):
            rx = rankdata(xx[:, j])
            if binary:
                if pos.all() or not pos.any():
                    return 0.0
                auc = (rx[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
                out = max(out, abs(auc - 0.5))
            else:
                cc = np.corrcoef(rx, ry)[0, 1]
                out = max(out, abs(cc) if np.isfinite(cc) else 0.0)
        return out

    xp, yp = x.copy(), y.copy()
    n = x.shape[0]
    drop = site.get("drop", 0.0) + realism.get("mcar", 0.0)
    if drop:
        xp[:, 1:] = np.where(rng.random((n, x.shape[1] - 1)) < drop, 0.0, xp[:, 1:])
    if "noise" in realism:
        xp[:, 1:] = xp[:, 1:] * (1 + rng.normal(0, realism["noise"], (n, x.shape[1] - 1)))
    if "outliers" in realism:
        xp[:, 1:] = np.where(rng.random((n, x.shape[1] - 1)) < realism["outliers"], xp[:, 1:] * 10, xp[:, 1:])
    if site.get("transform") == "label_noise" and binary:
        yp = flip_labels(y, np.ones(n, dtype=bool), site["flip"], rng)
    return float(best(x, y) - best(xp, yp))


def tilt(store, stays: np.ndarray, site: dict, rng) -> np.ndarray:
    """Selection weights of a case-mix site over ``stays`` (higher for its stratum)."""
    st = store.stays
    s = site.get("stratum")
    if s in ("age_old", "age_young"):
        age = st["age"].to_numpy(dtype=np.float64)[stays]
        zz = np.nan_to_num((age - 60) / 15)
        zz = zz if s == "age_old" else -zz
    else:
        surg = st["surgical_status"].astype(str).to_numpy()[stays]
        zz = np.where(np.char.find(surg.astype(str), "surg") >= 0, 1.0, -1.0)
        zz = zz if s == "surgical" else -zz
    w = np.exp(site["tilt"] * zz)
    return w / w.sum()


__all__ = ["THINNABLE", "apply", "draw_realism", "draw_site", "flip_labels", "snapshot_loss", "tilt"]
