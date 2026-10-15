"""Trial designer (toolkit tool 6, ``docs/toolkit.md``; CPU only): what the twin's prognosis is worth to an ICU trial
enrolling at a landmark, from the paper's kept per-anchor predictions on real held-out stays (Paper P amendment 11,
V4).

The population is a landmark design's held-out anchors (``L24``, ``L48``: A1's 32-rollout predictions; ``L24x``:
F1's 128-rollout subset) meeting the eligibility filters: datasets, ranges of the bedside values and clinical scores
at the anchor (amendment 10's ``scores.parquet``: sofa, lactate, map, creatinine, kdigo, shock_index, ...), and organ
support at the anchor (``on_imv`` / ``on_vasopressor`` / ``on_rrt``: the onset program has no label there, so the
support was running). Labels are A1's (the real stores'). Prognostic scores: the Twin (its risk of the endpoint),
clinical scores (SOFA, SAPS II, OASIS, NEWS; APACHE IVa where it is native, eICU) and LightGBM (A1's pooled model).

* **Event rate and sample size**: the endpoint's rate, and the two-proportion sample size for a relative risk ``rr``
  (two-sided ``alpha``, ``power``).
* **Prognostic enrichment**: enrolling the top q of the population by a score (by rank, ties broken at random): its
  control event rate, the sample size, and the patients screened per patient enrolled (1 / q).
* **Prognostic covariate adjustment** (PROCOVA-style): simulated trials (``simulations`` per cell) drawn from the
  population, randomized 1:1, the effect imposed as a constant relative risk (a treated patient's event kept with
  probability rr), analysed unadjusted (two-proportion z test) and adjusted for the score (logistic regression on
  treatment and the score: the logit of a probability, a raw score standardized; Wald test); the power at the
  unadjusted analytic size, the size for the target power under each analysis (from the simulated power), and the
  type I error with rr = 1.

Simulated effects are constant relative risks on held-out ICU stays standing in for a trial population: the tool
compares designs, it makes no claim about any treatment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SCORES = {   # name: (source, column or model, kind)
    "twin": ("twin", None, "prob"), "sofa": ("clinical", "sofa", "raw"), "sapsii": ("clinical", "sapsii", "raw"),
    "oasis": ("clinical", "oasis", "raw"), "news": ("clinical", "news", "raw"),
    "apache_iva": ("clinical", "apache_iva", "raw"), "lgbm_pooled": ("baseline", "lgbm_pooled", "prob"),
}
SIGNED = {"map": ("neg_map", -1.0), "rox": ("neg_rox", -1.0), "platelets": ("neg_platelets", -1.0)}
SUPPORT = {"on_imv": "imv_onset@1440", "on_vasopressor": "vasopressor_onset@1440", "on_rrt": "rrt_onset@1440"}


def z_of(p: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(p))


def n_two_proportions(pc: float, rr: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Patients per arm to detect pc against rr x pc (two-sided, normal approximation with the pooled null)."""
    pt = rr * pc
    if not 0 < pc < 1 or pt == pc:
        return float("inf")
    pb = (pc + pt) / 2
    za, zb = z_of(1 - alpha / 2), z_of(power)
    return float((za * np.sqrt(2 * pb * (1 - pb)) + zb * np.sqrt(pc * (1 - pc) + pt * (1 - pt))) ** 2 / (pc - pt) ** 2)


def population(paper: Path, design: str, endpoint: str, filters: dict | None = None):
    """The eligible anchors with the endpoint's label and every available score (a DataFrame)."""
    import pandas as pd

    base = "L24" if design == "L24x" else design
    pred_f = paper / ("f1/L24x" if design == "L24x" else f"a1/test_{design}") / "predictions.parquet"
    pr = pd.read_parquet(pred_f)
    a1 = pd.read_parquet(paper / "a1" / f"test_{base}" / "predictions.parquet")
    lab = pr[pr["program"] == endpoint].drop_duplicates("anchor").set_index("anchor")
    if lab.empty:
        raise KeyError(f"{endpoint} is not scored in {design}")
    df = pd.DataFrame({"anchor": lab.index, "stay": lab["stay"].astype(str).to_numpy(),
                       "dataset": lab["dataset"].to_numpy(), "y": lab["label"].to_numpy(float),
                       "twin": lab["p"].to_numpy(float)})
    sc = pd.read_parquet(paper / "f10" / "clinical" / "scores.parquet")
    sc = sc[sc["design"] == base].drop_duplicates("anchor").set_index("anchor")
    for col in sc.columns:
        if col not in ("key", "design", "dataset", "stay_id") and col != "anchor":
            df[col] = sc[col].reindex(df["anchor"]).to_numpy(dtype=float)
    for name, (col, sign) in SIGNED.items():
        if col in df:
            df[name] = sign * df[col]
    bl_f = paper / "a1" / f"test_{base}" / "baseline_predictions.parquet"
    if bl_f.exists():
        bl = pd.read_parquet(bl_f)
        bl = bl[bl["program"] == endpoint]
        for m, g in bl.groupby("model"):
            df[m] = g.drop_duplicates("anchor").set_index("anchor")["p"].reindex(df["anchor"]).to_numpy(float)
    have = {p: set(g["anchor"]) for p, g in a1.groupby("program") if p in SUPPORT.values()}
    for flag, prog in SUPPORT.items():
        df[flag] = ~df["anchor"].isin(have.get(prog, set()))
    m = np.isfinite(df["y"].to_numpy())
    for k, want in (filters or {}).items():
        if k == "datasets":
            m &= df["dataset"].isin(list(want)).to_numpy()
        elif k in SUPPORT:
            m &= df[k].to_numpy() == bool(want)
        else:
            lo, hi = want
            x = df[k].to_numpy(dtype=float)
            m &= np.isfinite(x) & (x >= (-np.inf if lo is None else lo)) & (x <= (np.inf if hi is None else hi))
    return df[m].reset_index(drop=True)


def covariate(s: np.ndarray, kind: str) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    if kind == "prob":             # a share of 32 rollouts is often 0: clipped at 0.5%, not driven to -inf
        p = np.clip(s, 0.005, 0.995)
        return np.log(p / (1 - p))
    return (s - np.nanmean(s)) / (np.nanstd(s) or 1.0)


def logistic_wald(y: np.ndarray, t: np.ndarray, x: np.ndarray, iters: int = 25) -> np.ndarray:
    """Per replicate (rows), the Wald z of the treatment in logit P(y) = b0 + b1 t + b2 x (Newton-Raphson)."""
    R, N = y.shape
    X = np.stack([np.ones((R, N)), t, x], axis=2)                    # [R, N, 3]
    b = np.zeros((R, 3))
    for _ in range(iters):
        eta = np.einsum("rnk,rk->rn", X, b)
        p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        w = np.maximum(p * (1 - p), 1e-10)
        g = np.einsum("rnk,rn->rk", X, y - p)
        H = np.einsum("rnk,rn,rnl->rkl", X, w, X) + 1e-8 * np.eye(3)
        step = np.linalg.solve(H, g[..., None])[..., 0]
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = np.einsum("rnk,rk->rn", X, b)
    p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
    H = np.einsum("rnk,rn,rnl->rkl", X, np.maximum(p * (1 - p), 1e-10), X) + 1e-8 * np.eye(3)
    var = np.linalg.inv(H)[:, 1, 1]
    return b[:, 1] / np.sqrt(np.maximum(var, 1e-12))


def simulate(y: np.ndarray, x: np.ndarray, n_total: int, rr: float, sims: int, alpha: float, rng,
             chunk: int = 200) -> dict:
    """Rejection rates of the unadjusted and the score-adjusted analyses over simulated trials."""
    za = z_of(1 - alpha / 2)
    n_total = int(max(4, 2 * (n_total // 2)))
    rej_u, rej_a = 0, 0
    done = 0
    while done < sims:
        r = min(chunk, sims - done)
        idx = rng.integers(0, len(y), size=(r, n_total))
        yy, xx = y[idx].astype(float), x[idx]
        t = np.zeros((r, n_total))
        t[np.arange(r)[:, None], np.argsort(rng.random((r, n_total)), axis=1)[:, : n_total // 2]] = 1.0
        keep = rng.random((r, n_total)) < rr
        yy = np.where(t == 1, yy * keep, yy)
        n1 = t.sum(1)
        p1, p0 = (yy * t).sum(1) / n1, (yy * (1 - t)).sum(1) / (n_total - n1)
        pb = yy.mean(1)
        se = np.sqrt(np.maximum(pb * (1 - pb) * (1 / n1 + 1 / (n_total - n1)), 1e-12))
        rej_u += int(np.sum(np.abs((p1 - p0) / se) > za))
        rej_a += int(np.sum(np.abs(logistic_wald(yy, t, xx)) > za))
        done += r
    return {"unadjusted": rej_u / sims, "adjusted": rej_a / sims, "simulations": sims, "n_total": n_total}


def n_for_power(power_at_n: float, n: int, alpha: float, target: float) -> float:
    """The size giving ``target`` power, from the power at size n (Z ~ N(mu sqrt(N), 1))."""
    za, zt = z_of(1 - alpha / 2), z_of(target)
    pw = float(np.clip(power_at_n, 1e-4, 1 - 1e-4))
    mu = (z_of(pw) + za) / np.sqrt(n)
    return float(((za + zt) / mu) ** 2) if mu > 0 else float("inf")


def run(session, spec: dict) -> dict:
    from icutwin.eval import bench
    from icutwin.toolkit import schema

    spec = dict(spec)
    design, endpoint = spec.get("design", "L24"), spec["endpoint"]
    rr = float((spec.get("effect") or {}).get("rr", 0.8))
    alpha, target = float(spec.get("alpha", 0.05)), float(spec.get("power", 0.8))
    sims, seed = int(spec.get("simulations", 2000)), int(spec.get("seed", 0))
    fractions = [float(q) for q in spec.get("enrichment", (1.0, 0.75, 0.5, 0.25))]
    names = spec.get("scores", list(SCORES))
    rng = np.random.default_rng(seed)
    df = population(session.paths.paper, design, endpoint, spec.get("population"))
    y = df["y"].to_numpy(float)
    pc = float(y.mean()) if len(y) else float("nan")
    n_arm = n_two_proportions(pc, rr, alpha, target)
    out = {"design": design, "endpoint": endpoint, "rr": rr, "alpha": alpha, "power": target,
           "population": {"anchors": len(df), "stays": int(df["stay"].nunique()), "event_rate": pc,
                          "datasets": {d: int(c) for d, c in df["dataset"].value_counts().sort_index().items()}},
           "unenriched": {"n_per_arm": n_arm, "n_total": 2 * n_arm}, "scores": {}}
    for name in names:
        src, col, kind = SCORES.get(name, ("baseline", name, "prob"))
        key = "twin" if src == "twin" else col
        if key not in df:
            continue
        s = df[key].to_numpy(float)
        ok = np.isfinite(s)
        if ok.sum() < 50 or not 0 < y[ok].sum() < ok.sum():
            continue
        ys, ss = y[ok], s[ok]
        r = {"anchors": int(ok.sum()), "auroc": bench.auroc_or_none(ys, ss), "enrichment": []}
        # the top q by rank, ties broken at random (a share of 32 rollouts is often exactly 0)
        order = np.lexsort((np.random.default_rng(seed).random(len(ss)), -ss))
        for q in fractions:
            k = max(1, int(np.ceil(q * len(ss))))
            m = np.zeros(len(ss), dtype=bool)
            m[order[:k]] = True
            pq = float(ys[m].mean())
            n = n_two_proportions(pq, rr, alpha, target)
            r["enrichment"].append({"fraction": q, "threshold": None if q >= 1 else float(ss[order[k - 1]]),
                                    "event_rate": pq, "n_per_arm": n, "n_total": 2 * n, "screened": 2 * n / q,
                                    "n_vs_unenriched": (2 * n) / (2 * n_arm) if np.isfinite(n_arm) else None})
        if np.isfinite(n_arm) and sims > 0:
            n0 = int(np.ceil(2 * n_arm))
            x = covariate(ss, kind)
            alt = simulate(ys, x, n0, rr, sims, alpha, rng)
            null = simulate(ys, x, n0, 1.0, sims, alpha, rng)
            n_u = n_for_power(alt["unadjusted"], alt["n_total"], alpha, target)
            n_a = n_for_power(alt["adjusted"], alt["n_total"], alpha, target)
            r["adjustment"] = {"n_simulated": alt["n_total"], "power_unadjusted": alt["unadjusted"],
                               "power_adjusted": alt["adjusted"], "n_total_unadjusted": n_u,
                               "n_total_adjusted": n_a, "n_ratio": n_a / n_u if n_u else None,
                               "type1_unadjusted": null["unadjusted"], "type1_adjusted": null["adjusted"],
                               "unadjusted_vs_analytic": n_u / alt["n_total"]}
        out["scores"][name] = r
    res = schema.result("trial", session, settings={"spec": spec, "simulations": sims, "seed": seed}, outputs=out,
                        evidence_keys=("refs",))
    return res
