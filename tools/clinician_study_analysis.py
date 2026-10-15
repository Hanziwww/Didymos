"""Paper P's clinician study: the pre-specified analysis (``docs/paper/clinician_study_protocol.md`` section 8). The
code is fixed before the first rating and run once on the locked data.

    python tools/clinician_study_analysis.py tables       # responses.json + key.json -> analysis/*.csv (host)
    Rscript tools/clinician_study_models.R <dir>/analysis  # the primary mixed models (lme4) -> primary_lme4.json
    python tools/clinician_study_analysis.py secondary    # bootstrap cross-checks, secondary endpoints -> secondary.json
    python tools/clinician_study_analysis.py fake --dir D  # a simulated study (no data) through the same chain

``--dir`` defaults to ``runs/toolkit/clinician`` (git-ignored). The tables are per-estimate rows under opaque ids;
they never leave the host. The materials take aggregates from ``primary_lme4.json`` and ``secondary.json``.

Rules (protocol section 8):
* The pilot packets (P01, P02) and the practice items are not analysed.
* A rater with less than 80% of a module answered is left out of that module's primary analysis (``complete`` 0),
  and included in the sensitivity analysis.
* Intervals: a two-way bootstrap that resamples raters and items (cases in C) independently, 2,000 draws, seed
  20261010; percentile intervals, 90% for A's equivalence and 95% otherwise.
* B's contrasts are within item and within rater. There the two-way bootstrap overstates the residual variance
  (Owen's pigeonhole bootstrap is conservative; on a simulated study its intervals were about 1.7 times lme4's).
  B's cross-check is therefore the per-item contrast with an item-level t interval, the analysis the sizing
  simulated. The two-way bootstrap of the contrast (rater and item fixed effects) is reported beside it as a
  conservative bound.

Definitions fixed here:
* A: ``score`` is the 10-point confidence-weighted call, synthetic +1..+5 and real -1..-5; its AUC for a twin is
  the mean of per-rater AUCs.
* C: estimates are probabilities (the rater's percent / 100). The card's risk is the value shown (whole percent).
  The case-cohort weight of a case is its stratum's population over its sample count (dataset x ICU death), used
  for the natural-mix Brier score and for calibration.
* Weight of advice: (aided - unaided) / (card - unaided) where |card - unaided| >= 0.01, clipped to [0, 1].
  An estimate whose card equals its unaided value (|difference| < 0.01) takes the rater's median weight.
* Counterfactual blend: each estimate's own weight replayed toward the twin's card and toward a comparator
  (LightGBM, A1's pooled model; SAPS II's probability for ICU death); the twin-specific gain is the paired Brier
  difference of the two replays (negative favours the twin).
* Over-reliance: among estimates where moving toward the card would increase the error (the card on the wrong side
  of the unaided estimate given the outcome), the share that moved toward it by at least 1 point. Under-reliance:
  among estimates where moving toward the card would reduce the error, the share that moved less than 1 point.
* SUS: the standard 0-100 score. Trust: the mean of the six items on 1-7, the two distrust items reversed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261010
B = 2000
OUTCOMES = ("icu_death", "vasopressor", "aki")          # clinician_study.C_OUTCOMES; asked: C_ASKED


# -- tables ---------------------------------------------------------------------------------------------------------
def tables(rows: list, key: dict) -> dict:
    """Collected rows (``clinician_study.join``) and the key -> one tidy frame per module."""
    raters = sorted({r["rater"] for r in rows if not r["rater"].startswith("P")})
    prof = {r["rater"]: r for r in rows if r["module"] == "profile"}
    n_items = {r: {m: len(ids) for m, ids in key["raters"][r]["order"].items()} for r in raters}
    by = {m: [r for r in rows if r["module"] == m and r["rater"] in raters] for m in ("A", "B1", "B2", "C", "Q")}

    def secs(r, a="shown", b="answered"):
        return (r[b] - r[a]) / 1000.0 if r.get(a) and r.get(b) else np.nan

    a = pd.DataFrame([{"rater": r["rater"], "item": r["item"], "kind": r["truth"]["kind"],
                       "dataset": r["truth"]["dataset"], "band": r["truth"]["band"], "call": r["call"],
                       "correct": int((r["call"] == "synthetic") == (r["truth"]["kind"] == "twin")),
                       "confidence": r["confidence"],
                       "score": r["confidence"] * (1 if r["call"] == "synthetic" else -1),
                       "order": r["order"], "seconds": secs(r)} for r in by["A"]])
    cues = pd.DataFrame([{"rater": r["rater"], "item": r["item"], "kind": r["truth"]["kind"],
                          "correct": int((r["call"] == "synthetic") == (r["truth"]["kind"] == "twin")),
                          "cue": r["cue"]} for r in by["A"] if (r.get("cue") or "").strip()])
    b1 = pd.DataFrame([{"rater": r["rater"], "item": r["item"], "treatment": r["truth"]["treatment"],
                        "dataset": r["truth"]["dataset"], "version": r["version"], "twin": int(r["version"] == "twin"),
                        "plausibility": r["plausibility"], "direction": r["direction"], "order": r["order"],
                        "seconds": secs(r)} for r in by["B1"]])
    b2 = pd.DataFrame([{"rater": r["rater"], "item": r["item"], "program": r["truth"]["program"],
                        "dataset": r["truth"]["dataset"], "method": r["method"], "relevance": r["relevance"],
                        "n_irrelevant": int(sum(r["irrelevant"])), "order": r["order"], "seconds": secs(r)}
                       for r in by["B2"]])
    # C: one row per estimate (rater x case x outcome asked); case-cohort weights from the key's population
    cases = {i: x for i, x in key["items"].items() if x["module"] == "C"}
    sample: dict = {}
    for x in cases.values():
        k = f"{x['dataset']}|{int(x['labels']['icu_death'])}"
        sample[k] = sample.get(k, 0) + 1
    pop = key.get("c_population") or sample                     # without the population: unweighted
    c = []
    for r in by["C"]:
        t = r["truth"]
        w = pop[f"{t['dataset']}|{int(t['labels']['icu_death'])}"] / sample[f"{t['dataset']}|{int(t['labels']['icu_death'])}"]
        for o in OUTCOMES:
            if o not in r["unaided"] or o not in t["labels"]:
                continue
            p = (t.get("twin") or {}).get(o)
            c.append({"rater": r["rater"], "case": r["item"], "dataset": t["dataset"], "outcome": o,
                      "label": t["labels"][o], "unaided": r["unaided"][o] / 100.0, "aided": r["aided"][o] / 100.0,
                      "card": np.nan if p is None else float(f"{100 * p:.0f}") / 100.0,
                      "lgbm": (t.get("lgbm") or {}).get(o, np.nan),
                      "sapsii": t.get("sapsii_prob", np.nan) if o == "icu_death" else np.nan,
                      "weight": w, "changed": r["changed"], "order": r["order"],
                      "seconds_unaided": secs(r, "shown", "locked"), "seconds_aided": secs(r, "locked", "answered")})
    c = pd.DataFrame(c)
    q = []
    for r in by["Q"]:
        s, t = r["sus"], r["trust"]
        q.append({"rater": r["rater"], "sus": 2.5 * (sum(s[k] - 1 for k in range(0, 10, 2)) +
                                                     sum(5 - s[k] for k in range(1, 10, 2))),
                  "trust": float(np.mean([*t[:4], 8 - t[4], 8 - t[5]])),
                  **{f"sus_{k + 1}": x for k, x in enumerate(s)}, **{f"trust_{k + 1}": x for k, x in enumerate(t)}})
    q = pd.DataFrame(q)
    profile = pd.DataFrame([{"rater": r, "role": (prof.get(r) or {}).get("role"),
                             "years": (prof.get(r) or {}).get("years"), "final": (prof.get(r) or {}).get("final"),
                             **{f"answered_{m}": len({x["item"] for x in by[m] if x["rater"] == r}) / n_items[r][m]
                                for m in ("A", "B1", "B2", "C")}} for r in raters])
    out = {"a": a, "cues": cues, "b1": b1, "b2": b2, "c": c, "q": q, "profile": profile}
    for name, m in (("a", "A"), ("b1", "B1"), ("b2", "B2"), ("c", "C")):
        df = out[name]
        if len(df):
            ok = set(profile.loc[profile[f"answered_{m}"] >= 0.8, "rater"])
            df["complete"] = df["rater"].isin(ok).astype(int)
            df = df.merge(profile[["rater", "role", "years"]], on="rater", how="left")
            out[name] = df
    return out


# -- the two-way bootstrap ------------------------------------------------------------------------------------------
class TwoWay:
    """Rater x item resampling weights: each draw resamples raters and items with replacement, independently, and a
    row's weight is its rater's multiplicity times its item's. ``rw`` holds the current draw's rater multiplicity per
    row (ones for the estimate), for statistics averaged over raters."""

    def __init__(self, df: pd.DataFrame, item: str, draws: int | None = None, seed: int = SEED):
        self.r = pd.factorize(df["rater"])[0]
        self.i = pd.factorize(df[item])[0]
        rng = np.random.default_rng(seed)
        draws = draws or B
        nr, ni = self.r.max() + 1, self.i.max() + 1
        self.mr = rng.multinomial(nr, np.full(nr, 1 / nr), size=draws)
        self.mi = rng.multinomial(ni, np.full(ni, 1 / ni), size=draws)
        self.rw = np.ones(len(self.r))

    def ci(self, stat, level: float = 0.95, names: tuple | None = None):
        """``stat(w)`` -> a float, or a vector named by ``names`` (``w`` all ones is the estimate)."""
        self.rw = np.ones(len(self.r))
        est = np.atleast_1d(np.asarray(stat(np.ones(len(self.r))), float))
        bs = []
        for mr, mi in zip(self.mr, self.mi, strict=True):
            self.rw = mr[self.r].astype(float)
            bs.append(np.atleast_1d(np.asarray(stat((mr[self.r] * mi[self.i]).astype(float)), float)))
        self.rw = np.ones(len(self.r))
        bs = np.array(bs)
        out = []
        for k in range(len(est)):
            b = bs[:, k][np.isfinite(bs[:, k])]
            lo, hi = np.quantile(b, [(1 - level) / 2, (1 + level) / 2]) if len(b) else (np.nan, np.nan)
            out.append({"est": float(est[k]), "lo": float(lo), "hi": float(hi), "level": level, "draws": int(len(b))})
        return dict(zip(names, out, strict=True)) if names else out[0]


def wmean(x, w) -> float:
    """The weighted mean over rows of positive weight (a row of weight 0 may hold NaN)."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = w > 0
    return float(np.sum(w[m] * x[m]) / np.sum(w[m])) if m.any() else np.nan


def wauc(y, s, w=None) -> float:
    """The weighted AUC (ties count half)."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ok = np.isfinite(s) & np.isfinite(y) & (w > 0)
    y, s, w = y[ok], s[ok], w[ok]
    wp, wn = np.sum(w * y), np.sum(w * (1 - y))
    if wp <= 0 or wn <= 0:
        return np.nan
    u, inv = np.unique(s, return_inverse=True)
    pos, neg = np.bincount(inv, w * y, len(u)), np.bincount(inv, w * (1 - y), len(u))
    return float(np.sum(pos * (np.cumsum(neg) - neg + 0.5 * neg)) / (wp * wn))


def rater_auc(df: pd.DataFrame, y: str, s, w, rw) -> float:
    """The mean over raters of the per-rater AUC (the Obuchowski-Rockette reader average); items weighted by ``w``,
    raters by their multiplicity ``rw``. ``s`` is a column name or an array."""
    sv = df[s].to_numpy(float) if isinstance(s, str) else np.asarray(s, float)
    yv = df[y].to_numpy(float)
    out, ws = [], []
    for _r, idx in df.groupby("rater").indices.items():
        if w[idx].sum() == 0:
            continue
        a = wauc(yv[idx], sv[idx], w[idx])
        if np.isfinite(a):
            out.append(a)
            ws.append(rw[idx][0])
    return float(np.average(out, weights=ws)) if out else np.nan


def logit(p):
    p = np.clip(np.asarray(p, float), 0.005, 0.995)
    return np.log(p / (1 - p))


def calibration(y, p, w) -> tuple[float, float]:
    """The weighted logistic recalibration: slope of y on logit(p), and calibration-in-the-large (the intercept with
    logit(p) as an offset). Newton-Raphson."""
    x, y, w = logit(p), np.asarray(y, float), np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 10 or y.min() == y.max():
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(x)), x])
    b = np.zeros(2)
    for _ in range(50):
        m = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (w * (y - m))
        h = (X * (w * m * (1 - m))[:, None]).T @ X
        step = np.linalg.solve(h + 1e-9 * np.eye(2), g)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    a = 0.0
    for _ in range(50):
        m = 1 / (1 + np.exp(-np.clip(a + x, -30, 30)))
        step = np.sum(w * (y - m)) / max(np.sum(w * m * (1 - m)), 1e-12)
        a += step
        if abs(step) < 1e-10:
            break
    return float(b[1]), float(a)


def holm(p: list) -> list:
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, k in enumerate(o):
        run = max(run, min(1.0, (len(p) - rank) * p[k]))
        adj[k] = run
    return adj.tolist()


def fleiss(calls: pd.DataFrame) -> float:
    """Fleiss' kappa over items rated by every rater (``calls``: item x rater, values real/synthetic)."""
    m = calls.dropna()
    if m.empty:
        return np.nan
    n = m.shape[1]
    k1 = (m == "synthetic").sum(axis=1).to_numpy(float)
    counts = np.column_stack([k1, n - k1])
    p_i = (np.sum(counts ** 2, axis=1) - n) / (n * (n - 1))
    pj = counts.sum(axis=0) / counts.sum()
    pe = np.sum(pj ** 2)
    return float((p_i.mean() - pe) / (1 - pe)) if pe < 1 else np.nan


# -- secondary ------------------------------------------------------------------------------------------------------
def module_a(a: pd.DataFrame) -> dict:
    from scipy.stats import binomtest

    out = {}
    for name, df in (("primary", a[a["complete"] == 1]), ("sensitivity_all_raters", a)):
        df = df.reset_index(drop=True)
        tw = TwoWay(df, "item")
        y = df["correct"].to_numpy(float)
        out[name] = {"raters": int(df["rater"].nunique()), "items": int(df["item"].nunique()), "calls": len(df),
                     "accuracy_90": tw.ci(lambda w: wmean(y, w), 0.90),
                     "equivalent_bootstrap": None}
        ci = out[name]["accuracy_90"]
        out[name]["equivalent_bootstrap"] = bool(ci["lo"] > 0.40 and ci["hi"] < 0.60)
    df = a[a["complete"] == 1].reset_index(drop=True)
    tw = TwoWay(df, "item")
    y = df["correct"].to_numpy(float)
    df["is_twin"] = (df["kind"] == "twin").astype(float)
    out["auc_confidence_weighted"] = tw.ci(lambda w: rater_auc(df, "is_twin", "score", w, tw.rw))
    out["share_called_real"] = tw.ci(lambda w: wmean((df["call"] == "real").to_numpy(float), w))
    out["accuracy_on_real"] = tw.ci(lambda w: wmean(y * (df["kind"] == "real"), w * (df["kind"] == "real")))
    out["accuracy_on_twins"] = tw.ci(lambda w: wmean(y * (df["kind"] == "twin"), w * (df["kind"] == "twin")))
    for col in ("dataset", "band", "role", "years"):
        out[f"accuracy_by_{col}"] = {str(g): tw.ci(lambda w, g=g: wmean(y, w * (df[col] == g).to_numpy(float)))
                                     for g in sorted(df[col].dropna().unique(), key=str)}
    q = pd.cut(df["order"], [-1, 19, 39, 59, 10_000], labels=["1-20", "21-40", "41-60", "61-80"])
    out["accuracy_by_order"] = {str(g): tw.ci(lambda w, g=g: wmean(y, w * (q == g).to_numpy(float)))
                                for g in q.cat.categories}
    per = []
    for r, g in df.groupby("rater"):
        k, n = int(g["correct"].sum()), len(g)
        per.append({"rater": r, "n": n, "correct": k, "accuracy": k / n, "p": binomtest(k, n, 0.5).pvalue})
    for x, p in zip(per, holm([x["p"] for x in per]), strict=True):
        x["p_holm"] = p
    out["per_rater"] = {"above_chance_holm": sum(x["p_holm"] < 0.05 and x["accuracy"] > 0.5 for x in per),
                        "raters": len(per), "accuracy_range": [min(x["accuracy"] for x in per),
                                                               max(x["accuracy"] for x in per)]}
    out["fleiss_kappa"] = fleiss(df.pivot_table(index="item", columns="rater", values="call", aggfunc="first"))
    det = df.groupby(["item", "kind"])["correct"].mean().rename("detected").reset_index()
    out["item_detection"] = {k: {"median": float(g["detected"].median()), "q90": float(g["detected"].quantile(0.9)),
                                 "max": float(g["detected"].max()), "items_over_0.75": int((g["detected"] > 0.75).sum())}
                             for k, g in det.groupby("kind")}
    return out, per, det


def fe(df: pd.DataFrame, y: str, terms: list):
    """``stat(w)`` for the bootstrap: the weighted least-squares coefficients of ``terms`` (0/1 columns) with rater
    and item fixed effects, the fixed-effects analogue of the mixed models (rater and item effects removed)."""
    X = np.column_stack([df[terms].to_numpy(float), np.ones(len(df)),
                         pd.get_dummies(df["rater"], drop_first=True).to_numpy(float),
                         pd.get_dummies(df["item"], drop_first=True).to_numpy(float)])
    yv = df[y].to_numpy(float)

    def stat(w):
        m = w > 0
        sw = np.sqrt(w[m])[:, None]
        b = np.linalg.lstsq(X[m] * sw, yv[m] * sw[:, 0], rcond=None)[0]
        return b[: len(terms)]

    return stat


def per_item(df: pd.DataFrame, y: str, col: str, a, b) -> dict:
    """The per-item contrast (the mean rating in version ``a`` minus in ``b``, per item, averaged over items) with
    its item-level t interval: the analysis the protocol's sizing simulated."""
    from scipy.stats import t as tdist

    g = df.groupby(["item", col])[y].mean().unstack()
    d = (g[a] - g[b]).dropna().to_numpy(float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    q = tdist.ppf(0.975, len(d) - 1)
    return {"est": float(d.mean()), "se": float(se), "lo": float(d.mean() - q * se), "hi": float(d.mean() + q * se),
            "items": len(d)}


def module_b1(b1: pd.DataFrame) -> dict:
    out = {}
    for name, df in (("primary", b1[(b1["complete"] == 1) & (b1["treatment"] != "insulin")]),
                     ("sensitivity_all_raters", b1[b1["treatment"] != "insulin"])):
        df = df.reset_index(drop=True)
        pi = per_item(df, "plausibility", "version", "twin", "real")
        out[name] = {"raters": int(df["rater"].nunique()), "items": int(df["item"].nunique()),
                     "twin_minus_real": pi, "non_inferior_per_item": bool(pi["lo"] > -0.5),
                     "twin_minus_real_two_way_bootstrap": TwoWay(df, "item").ci(fe(df, "plausibility", ["twin"]))}
    df = b1[b1["complete"] == 1].reset_index(drop=True)
    tw = TwoWay(df, "item")
    t = df["twin"].to_numpy(float)
    yes = (df["direction"] == "yes").to_numpy(float)
    out["by_treatment"] = {}
    for g in sorted(df["treatment"].unique()):
        m = (df["treatment"] == g).to_numpy(float)
        out["by_treatment"][g] = {
            "twin_minus_real": per_item(df[df["treatment"] == g], "plausibility", "version", "twin", "real"),
            "expected_direction_twin": tw.ci(lambda w, m=m: wmean(yes, w * m * t)),
            "expected_direction_real": tw.ci(lambda w, m=m: wmean(yes, w * m * (1 - t)))}
    m = (df["treatment"] != "insulin").to_numpy(float)
    out["expected_direction"] = {"twin": tw.ci(lambda w: wmean(yes, w * m * t)),
                                 "real": tw.ci(lambda w: wmean(yes, w * m * (1 - t)))}
    return out


def module_b2(b2: pd.DataFrame) -> dict:
    out = {}
    for name, df in (("primary", b2[b2["complete"] == 1]), ("sensitivity_all_raters", b2)):
        df = df.reset_index(drop=True)
        tw = TwoWay(df, "item")
        x = df["relevance"].to_numpy(float)
        m = {k: (df["method"] == k).to_numpy(float) for k in ("edit", "knockout", "random")}
        d = df.assign(edit=m["edit"], knockout=m["knockout"])
        stat = fe(d, "relevance", ["edit", "knockout"])
        pairs = (("edit", "random"), ("knockout", "random"), ("edit", "knockout"))
        out[name] = {"raters": int(df["rater"].nunique()), "items": int(df["item"].nunique()),
                     **{f"{a}_minus_{b}": per_item(df, "relevance", "method", a, b) for a, b in pairs},
                     "two_way_bootstrap": tw.ci(lambda w: (lambda b: [b[0], b[1], b[0] - b[1]])(stat(w)),
                                                names=tuple(f"{a}_minus_{b}" for a, b in pairs)),
                     "mean_relevance": {k: tw.ci(lambda w, k=k: wmean(x, w * m[k])) for k in m},
                     "irrelevant_rate": {k: tw.ci(lambda w, k=k: wmean(df["n_irrelevant"].to_numpy(float) / 3,
                                                                       w * m[k])) for k in m}}
    return out


def blend(df: pd.DataFrame, comp: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights of advice and the two replays (toward the twin's card and toward ``comp``)."""
    u, a, card = df["unaided"].to_numpy(float), df["aided"].to_numpy(float), df["card"].to_numpy(float)
    d = card - u
    ok = np.abs(d) >= 0.01
    woa = np.full(len(df), np.nan)
    woa[ok] = np.clip((a[ok] - u[ok]) / d[ok], 0.0, 1.0)
    med = pd.Series(woa).groupby(df["rater"].to_numpy()).transform("median").to_numpy()
    w = np.where(ok, woa, med)
    w = np.where(np.isfinite(w), w, 0.0)
    return woa, u + w * (card - u), u + w * (df[comp].to_numpy(float) - u)


def module_c(c: pd.DataFrame) -> dict:
    out = {}
    for name, df0 in (("primary", c[c["complete"] == 1]), ("sensitivity_all_raters", c)):
        df = df0[df0["outcome"] == "icu_death"].reset_index(drop=True)
        tw = TwoWay(df, "case")
        y, u, a = df["label"].to_numpy(float), df["unaided"].to_numpy(float), df["aided"].to_numpy(float)
        d = (a - y) ** 2 - (u - y) ** 2
        out[name] = {"raters": int(df["rater"].nunique()), "cases": int(df["case"].nunique()),
                     "brier_aided_minus_unaided": tw.ci(lambda w: wmean(d, w))}
    df_all = c[c["complete"] == 1]
    for o in OUTCOMES:
        df = df_all[df_all["outcome"] == o].reset_index(drop=True)
        if df.empty:
            continue
        tw = TwoWay(df, "case")
        y = df["label"].to_numpy(float)
        cw = df["weight"].to_numpy(float)
        u, a, card = (df[k].to_numpy(float) for k in ("unaided", "aided", "card"))
        r = {"estimates": len(df), "cases": int(df["case"].nunique()), "events": int(df.groupby("case")["label"].first().sum())}
        r["brier_aided_minus_unaided"] = tw.ci(lambda w: wmean((a - y) ** 2 - (u - y) ** 2, w))
        sources = {"unaided": u, "aided": a, "twin_card": card, "lgbm": df["lgbm"].to_numpy(float)}
        if o == "icu_death":
            sources["sapsii"] = df["sapsii"].to_numpy(float)
        r["brier"] = {k: tw.ci(lambda w, s=s: wmean((s - y) ** 2, w * np.isfinite(s))) for k, s in sources.items()}
        r["brier_natural_mix"] = {k: tw.ci(lambda w, s=s: wmean((s - y) ** 2, w * cw * np.isfinite(s)))
                                  for k, s in sources.items()}
        r["auc"] = {k: tw.ci(lambda w, s=s: rater_auc(df, "label", s, w, tw.rw)) for k, s in sources.items()}
        r["auc_aided_minus_unaided"] = tw.ci(lambda w: rater_auc(df, "label", a, w, tw.rw) -
                                             rater_auc(df, "label", u, w, tw.rw))
        r["auc_aided_minus_twin_card"] = tw.ci(lambda w: rater_auc(df, "label", a, w, tw.rw) -
                                               rater_auc(df, "label", card, w, tw.rw))
        r["calibration"] = {k: tw.ci(lambda w, s=s: calibration(y, s, w * cw), names=("slope", "intercept"))
                            for k, s in sources.items()}
        woa, twin_rep, _ = blend(df, "lgbm")
        r["weight_of_advice"] = {"defined": int(np.isfinite(woa).sum()),
                                 "mean": tw.ci(lambda w: wmean(np.nan_to_num(woa), w * np.isfinite(woa))),
                                 "median": float(np.nanmedian(woa)) if np.isfinite(woa).any() else np.nan,
                                 "share_zero": float(np.mean(woa[np.isfinite(woa)] == 0))}
        toward = np.sign(a - u) == np.sign(card - u)
        moved = np.abs(a - u) >= 0.01
        harmful = np.where(y == 1, card < u, card > u) & (np.abs(card - u) >= 0.01)
        helpful = np.where(y == 1, card > u, card < u) & (np.abs(card - u) >= 0.01)
        r["over_reliance"] = tw.ci(lambda w: wmean((toward & moved).astype(float), w * harmful))
        r["under_reliance"] = tw.ci(lambda w: wmean((~moved).astype(float), w * helpful))
        r["counterfactual_blend"] = {}
        for comp in ("lgbm", "sapsii") if o == "icu_death" else ("lgbm",):
            m = np.isfinite(df[comp].to_numpy(float)) & np.isfinite(card)
            _w, tr, cr = blend(df, comp)
            r["counterfactual_blend"][comp] = {
                "twin_replay_minus_unaided": tw.ci(lambda w, m=m, tr=tr: wmean((tr - y) ** 2 - (u - y) ** 2, w * m)),
                "comparator_replay_minus_unaided": tw.ci(lambda w, m=m, cr=cr: wmean((cr - y) ** 2 - (u - y) ** 2, w * m)),
                "twin_specific_gain": tw.ci(lambda w, m=m, tr=tr, cr=cr: wmean((tr - y) ** 2 - (cr - y) ** 2, w * m)),
                "estimates": int(m.sum())}
        if o == "icu_death":
            first = df.drop_duplicates(["rater", "case"])
            r["changed_yes"] = float((first["changed"] == "yes").mean())
            r["seconds_median"] = {"unaided": float(first["seconds_unaided"].median()),
                                   "aided": float(first["seconds_aided"].median())}
        out[o] = r
    return out


def secondary(t: dict) -> dict:
    a, per, det = module_a(t["a"])
    q = t["q"]
    prof = t["profile"]
    return {
        "raters": {"n": len(prof), "role": prof["role"].value_counts(dropna=False).to_dict(),
                   "years": prof["years"].value_counts(dropna=False).to_dict(),
                   "complete": {m: int((prof[f"answered_{m}"] >= 0.8).sum()) for m in ("A", "B1", "B2", "C")}},
        "A": a, "B1": module_b1(t["b1"]), "B2": module_b2(t["b2"]), "C": module_c(t["c"]),
        "questionnaire": {"raters": len(q),
                          **({k: {"mean": float(q[k].mean()), "sd": float(q[k].std()), "median": float(q[k].median()),
                                  "iqr": [float(q[k].quantile(0.25)), float(q[k].quantile(0.75))]}
                              for k in ("sus", "trust")} if len(q) else {})},
        "_internal": {"a_per_rater": per, "a_item_detection": det.to_dict("records")},
    }


# -- fake: a simulated study (no data) ------------------------------------------------------------------------------
def fake(d: Path, raters: int = 16, seed: int = 7, acc: float = 0.55) -> None:
    """A synthetic key and synthetic exports in the page's format, joined by ``clinician_study.join``: the chain runs
    end to end without data. Effects: A accuracy ~``acc``; B1 twin 0.2 points below real, insulin twin 1 point;
    B2 editing +0.6, knockout +0.4 over random; C as the power simulation's base scenario. One rater stops C early."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import clinician_study as cs

    from icutwin.toolkit import rating

    rng = np.random.default_rng(seed)
    items: dict = {}
    for k in range(80):
        items[f"a{k:02d}"] = {"module": "A", "kind": ("real", "twin")[k % 2], "dataset": ("mimic", "eicu")[k // 40],
                              "band": int(k % 4), "u": rng.normal(0, 0.8)}
    for k in range(24):
        items[f"b1{k:02d}"] = {"module": "B1", "treatment": cs.TREATS[k // 4], "dataset": ("mimic", "eicu")[k % 2],
                               "u": rng.normal(0, 0.6)}
    for k in range(20):
        items[f"b2{k:02d}"] = {"module": "B2", "program": ("decompensation@1440", "vasopressor_onset@1440")[k // 10],
                               "dataset": ("mimic", "eicu")[k % 2], "u": rng.normal(0, 0.6)}
    for k in range(48):
        dead = k < 12
        true = rng.normal(0.4, 1.2) if dead else rng.normal(-2.6, 1.2)
        labels = {"icu_death": float(dead)}
        if k % 4:
            labels["aki"] = float(rng.random() < 0.3)
        items[f"c{k:02d}"] = {"module": "C", "dataset": ("mimic", "eicu")[k % 2], "labels": labels, "true": true,
                              "twin": {o: float(1 / (1 + np.exp(-(true + rng.normal(0, 0.8))))) if o in labels else None
                                       for o in OUTCOMES},
                              "lgbm": {o: float(1 / (1 + np.exp(-(true + rng.normal(0, 1.2))))) for o in OUTCOMES},
                              "sapsii_prob": float(1 / (1 + np.exp(-(true * 0.7 + rng.normal(0, 1.4)))))}
    key = {"build": "fakebuild", "salt": "x", "items": items, "raters": {},
           "c_population": {"mimic|1": 600, "mimic|0": 5400, "eicu|1": 500, "eicu|0": 6500}}
    roles, years = ("attending", "fellow"), ("<2", "2-5", "6-10", ">10")
    exports = {}
    for k in range(raters):
        name = f"R{k + 1:02d}"
        ru = rng.normal(0, 0.3)
        order = {m: [i for i, x in items.items() if x["module"] == m] for m in ("A", "B1", "B2", "C")}
        for m in order:
            order[m] = [order[m][j] for j in rng.permutation(len(order[m]))]
        ver = {i: ("twin", "real")[(k + j) % 2] for j, i in enumerate(sorted(order["B1"]))}
        meth = {i: cs.METHODS[(k + j) % 3] for j, i in enumerate(sorted(order["B2"]))}
        key["raters"][name] = {"index": k, "b1_version": ver, "b2_method": meth, "order": order}
        t = 1_790_000_000_000 + k * 10_000_000
        resp: dict = {"A": [], "B1": [], "B2": [], "C": []}
        mu = np.log(acc / (1 - acc)) * 1.15
        for j, i in enumerate(order["A"]):
            x = items[i]
            right = rng.random() < 1 / (1 + np.exp(-(mu + ru + x["u"])))
            call = ("synthetic" if x["kind"] == "twin" else "real") if right else (
                "real" if x["kind"] == "twin" else "synthetic")
            resp["A"].append({"item": i, "order": j, "call": call, "confidence": int(rng.integers(1, 6)),
                              "cue": "charting rhythm" if rng.random() < 0.1 else "", "shown": t, "answered": t + 30_000})
            t += 31_000
        for j, i in enumerate(order["B1"]):
            x = items[i]
            eff = (-1.0 if x["treatment"] == "insulin" else -0.2) if ver[i] == "twin" else 0.0
            v = int(np.clip(np.rint(3.2 + x["u"] + ru + eff + rng.normal(0, 0.9)), 1, 5))
            resp["B1"].append({"item": i, "order": j, "plausibility": v,
                               "direction": ("yes", "no", "unsure")[int(rng.integers(3)) if v < 3 else 0],
                               "shown": t, "answered": t + 40_000})
            t += 41_000
        for j, i in enumerate(order["B2"]):
            eff = {"edit": 0.6, "knockout": 0.4, "random": 0.0}[meth[i]]
            v = int(np.clip(np.rint(2.8 + items[i]["u"] + ru + eff + rng.normal(0, 0.9)), 1, 5))
            resp["B2"].append({"item": i, "order": j, "relevance": v, "irrelevant": [bool(rng.random() < 0.3)
                                                                                      for _ in range(3)],
                               "shown": t, "answered": t + 45_000})
            t += 46_000
        bias = rng.normal(0, 0.4)
        stop = 30 if k == raters - 1 else 48                    # the last rater answers 30 of 48 cases
        for j, i in enumerate(order["C"][:stop]):
            x = items[i]
            un, ai = {}, {}
            for o, y in x["labels"].items():
                lt = x["true"] if o == "icu_death" else -1.5 + 0.8 * (2 * y - 1)
                pu = 1 / (1 + np.exp(-(lt + bias + rng.normal(0, 1.2))))
                card = round(100 * x["twin"][o]) / 100
                w = rng.beta(1.0, 3.0)
                un[o] = int(round(100 * pu))
                ai[o] = int(round(100 * ((1 - w) * un[o] / 100 + w * card)))
            resp["C"].append({"item": i, "order": j, "unaided": un, "aided": ai,
                              "changed": "yes" if ai != un else "no", "shown": t, "locked": t + 60_000,
                              "answered": t + 90_000})
            t += 91_000
        s = [int(rng.integers(3, 6)) if j % 2 == 0 else int(rng.integers(1, 3)) for j in range(10)]
        qn = {"sus": s, "trust": [int(rng.integers(3, 8)) for _ in range(6)], "shown": t, "answered": t + 1}
        o = {"study": rating.STUDY, "packet": name, "build": "fakebuild",
             "consent": {"participation": True, "confidentiality": True, "role": roles[k % 2], "years": years[k % 4]},
             "responses": resp, "questionnaire": qn, "exported": "2026-11-01T00:00:00Z", "final": stop == 48}
        o["check"] = rating.check(o)
        exports[f"{name}.json"] = o
        if k == 0:                                               # a progress export the final one supersedes
            p = json.loads(json.dumps(o))
            p["responses"]["C"] = p["responses"]["C"][:10]
            p["final"], p["questionnaire"] = False, None
            p["check"] = rating.check(p)
            exports[f"{name}_progress.json"] = p
    exports["tampered.json"] = dict(exports["R02.json"], check="00000000")
    rows, problems = cs.join(key, exports)
    d.mkdir(parents=True, exist_ok=True)
    (d / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8", newline="\n")
    (d / "responses.json").write_text(json.dumps({"rows": rows, "problems": problems}, indent=1), encoding="utf-8",
                                      newline="\n")
    print(json.dumps({"rows": len(rows), "problems": problems}))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("tables", "secondary", "fake"))
    ap.add_argument("--dir", type=Path, default=Path("runs/toolkit/clinician"))
    ap.add_argument("--raters", type=int, default=16)
    a = ap.parse_args(argv)
    an = a.dir / "analysis"
    if a.step == "fake":
        fake(a.dir, a.raters)
    elif a.step == "tables":
        key = json.loads((a.dir / "key.json").read_text(encoding="utf-8"))
        rows = json.loads((a.dir / "responses.json").read_text(encoding="utf-8"))["rows"]
        an.mkdir(parents=True, exist_ok=True)
        t = tables(rows, key)
        for name, df in t.items():
            df.to_csv(an / f"{name}.csv", index=False, lineterminator="\n")
        print(json.dumps({name: len(df) for name, df in t.items()}))
    else:
        t = {name: pd.read_csv(an / f"{name}.csv", keep_default_na=True)
             for name in ("a", "b1", "b2", "c", "q", "profile")}
        res = secondary(t)
        (an / "secondary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8", newline="\n")
        print(json.dumps({"A": res["A"]["primary"], "B1": res["B1"]["primary"],
                          "C": res["C"]["primary"]}, default=float))


if __name__ == "__main__":
    main()
