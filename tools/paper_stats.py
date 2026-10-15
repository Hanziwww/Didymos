"""Paper P amendment 13 (``docs/p1_paper_protocol.md``): the manuscript's reporting statistics.

P values for the figures' key comparisons, and Monte Carlo intervals for the trial simulations, on CPU, from the
per-anchor predictions already kept or the intervals already stored. Nothing reported changes. Every stored interval
this script can recompute is reproduced first, and the script stops before any P value is computed if one differs.

Families (Holm within each):
1. Didymos minus LightGBM; 2. minus TabPFN-3.5; 3. the follow-up minus the pre-specified 32-rollout share. All three
   on the 13 primary endpoints (rolling anchors R128, 24-hour landmark L24x): the hazard estimate's P from its stored
   interval, the sampled estimate's by the exact paired bootstrap.
4. The sampled estimate minus the best clinical score per endpoint (amendment 10's resampling).
5. Calibration on V1's draws: O/E = 1, raw and recalibrated, and the changes in O/E and ECE.
6. What-if agreement; 7. the null arm; 8. membership signal against fit; 9. augmentation; 10. attribution.
Also the trial simulations: Clopper-Pearson intervals for the type I error, Monte Carlo intervals for the
sample-size ratios.

Amendment 14 (``--amendment14``, added to an existing ``stats.json``):
11. Every program A1 compared with LightGBM on its three anchor sets (165): exact P after the stored intervals are
    reproduced, Benjamini-Hochberg q values (Holm beside them), and joint-bootstrap summaries (exploratory, post hoc).
12. Label efficiency: the training size at which LightGBM per dataset matches the Twin, per primary endpoint.

Amendment 15 (``--amendment15``): ranks across endpoints. 13, every predictor on the 12 primary endpoints where all
learned comparators were fitted; 14, the five predictors scored on all 165 programs. Mean ranks with bootstrap
intervals over tasks, a Friedman test, and Didymos against each predictor (Wilcoxon signed-rank, Holm); Table 2.

Writes ``<out>/stats.json`` and, with ``--tables``, one CSV per family (cells as displayed).

    python tools/paper_stats.py --root runs/paper --toolkit runs/toolkit --out runs/paper/stats \\
        --tables docs/paper/tables/manuscript
    python tools/paper_stats.py --amendment14 --out runs/paper/stats --tables docs/paper/tables/manuscript
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import PRIMARY, _sha  # noqa: E402
from paper_results import NAMES  # noqa: E402

NAMES = {**NAMES, "decompensation@1440": "Death within 24 h"}
E2_PROGRAMS = {"R128": ("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880"),
               "L24x": ("icu_death@-1", "decompensation@1440")}
HEADLINE = [("R128", p) for p in PRIMARY["R"]] + [("L24x", p) for p in (*PRIMARY["L24"], "decompensation@1440")]
SUBSETS = {"R128": ("f1/R128", "a1/test_R", "R"), "L24x": ("f1/L24x", "a1/test_L24", "L24")}
VERSUS = {1: ("lgbm_pooled", "LightGBM"), 2: ("tabpfn_v3_5", "TabPFN-3.5"), 3: ("twin_a1_32", "pre-specified")}
CLINICAL = ("sofa", "news", "mews", "qsofa", "sirs", "sapsii_prob", "oasis_prob")
SCORE_NAMES = {"sofa": "SOFA", "news": "NEWS", "mews": "MEWS", "qsofa": "qSOFA", "sirs": "SIRS",
               "sapsii_prob": "SAPS II", "oasis_prob": "OASIS"}
B_CI, B_P, SEED_P = 1000, 10_000, 1
TOL = 1e-12


class NotReproduced(SystemExit):
    pass


# ---- the tests -------------------------------------------------------------------------------------------------

def p_boot(draws, est: float, null: float = 0.0) -> tuple[float | None, int]:
    """Two-sided bootstrap P: min(1, 2(1 + k)/(B + 1)), k the finite draws on the other side of ``null`` from the
    estimate."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)] - null
    e = est - null
    if not len(d) or not np.isfinite(e):
        return None, int(len(d))
    if e == 0:
        return 1.0, int(len(d))
    k = int(np.sum(d <= 0)) if e > 0 else int(np.sum(d >= 0))
    return float(min(1.0, 2 * (1 + k) / (len(d) + 1))), int(len(d))


def p_ci(est: float, lo: float, hi: float, null: float = 0.0) -> float | None:
    """P from a stored 95% interval (normal approximation; the half-width on the side of the null)."""
    from scipy.stats import norm

    if None in (est, lo, hi):
        return None
    e = est - null
    se = (e - (lo - null)) / 1.959963984540054 if e >= 0 else ((hi - null) - e) / 1.959963984540054
    if not np.isfinite(se) or se <= 0:
        return None
    return float(2 * norm.sf(abs(e) / se))


def holm(ps: list) -> list:
    from statsmodels.stats.multitest import multipletests

    idx = [i for i, p in enumerate(ps) if p is not None]
    out = [None] * len(ps)
    if idx:
        adj = multipletests([ps[i] for i in idx], method="holm")[1]
        for i, a in zip(idx, adj, strict=True):
            out[i] = float(a)
    return out


def at_floor(p, n) -> bool:
    """A bootstrap P at its floor 2 / (n + 1): no resample on the other side, so the value is an upper bound."""
    return p is not None and bool(n) and abs(p - 2 / (n + 1)) < 1e-12


def holm_bounds(ps: list, bounds: list) -> tuple[list, list]:
    """Holm's adjusted P values (equal to ``holm``) and whether each is an upper bound: it is when the running maximum
    that sets it comes from a raw P that is itself a bound."""
    idx = sorted((i for i, p in enumerate(ps) if p is not None), key=lambda i: ps[i])
    m, adj, flag = len(idx), [None] * len(ps), [False] * len(ps)
    run, run_b = 0.0, False
    for j, i in enumerate(idx):
        c = (m - j) * ps[i]
        if c > run:
            run, run_b = c, bounds[i]
        elif c == run:
            run_b = run_b or bounds[i]
        adj[i], flag[i] = min(1.0, run), bool(run_b and run < 1.0)
    ref = holm(ps)
    assert all((a is None and b is None) or abs(a - b) < 1e-12 for a, b in zip(adj, ref, strict=True))
    return adj, flag


def assign_holm(rows: list, key: str = "p", out: str = "p_holm") -> None:
    ps = [r.get(key) for r in rows]
    bounds = [at_floor(r.get(key), r.get("draws")) for r in rows]
    adj, flag = holm_bounds(ps, bounds)
    for r, a, b, f in zip(rows, adj, flag, bounds, strict=True):
        r[out], r[f"{out}_bound"], r[f"{key}_bound"] = a, b, f


def close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= TOL * max(1.0, abs(float(a)), abs(float(b)))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- families 1-3: the F1 comparisons (tools/paper_f2.py) --------------------------------------------------------

def f1_inputs(root: Path, name: str) -> dict:
    """The arrays ``tools/paper_f2.py`` paired on: the subset's anchors, A1's labels, the 128-rollout share and A1's
    stored comparators, in its order."""
    import pandas as pd

    sub, a1d, design = SUBSETS[name]
    sd, a1 = root / sub, root / a1d
    an = pd.read_parquet(sd / "anchors.parquet").drop_duplicates("anchor").reset_index(drop=True)
    pa1 = pd.read_parquet(a1 / "predictions.parquet")
    ps = pd.read_parquet(sd / "predictions.parquet")
    others = [pd.read_parquet(a1 / f) for f in ("baseline_predictions.parquet", "tabpfn_v3_5_predictions.parquet",
                                                "probe_predictions.parquet") if (a1 / f).exists()]

    def preds(prog: str) -> dict:
        out = {"twin_128": ps[ps["program"] == prog].set_index("anchor")["p"].reindex(an["anchor"]).to_numpy(float),
               "twin_a1_32": pa1[pa1["program"] == prog].set_index("anchor")["p"].reindex(an["anchor"])
               .to_numpy(float)}
        for d_ in others:
            d_ = d_[d_["program"] == prog]
            for m_, g2 in d_.groupby("model"):
                if m_ in ("lgbm_pooled", "tabpfn_v3_5", "twin_probe"):
                    out[m_] = g2.drop_duplicates("anchor").set_index("anchor")["p"].reindex(an["anchor"]).to_numpy(
                        float)
        return out

    return {"w": an["weight"].to_numpy(dtype=float), "ds": an["dataset"].to_numpy(),
            "stay": an["twin"].astype(str).to_numpy(), "preds": preds,
            "lab": {p: g.set_index("anchor")["label"].reindex(an["anchor"]).to_numpy(dtype=float)
                    for p, g in pa1.groupby("program")},
            "f2": json.loads((sd / "f2.json").read_text()),
            "files": [sd / "anchors.parquet", sd / "predictions.parquet", sd / "f2.json", a1 / "predictions.parquet"]}


def _paired_slow(yy, aa, bb, w_):
    from icutwin.eval import bench

    return (bench.auroc_or_none(yy, aa, w_) or np.nan) - (bench.auroc_or_none(yy, bb, w_) or np.nan)


def _paired_fast(yy, aa, bb, w_):
    from icutwin.eval import bench

    return bench.auroc_fast(yy, aa, w_) - bench.auroc_fast(yy, bb, w_)


def f1_exact(x: dict, prog: str, other: str, stored: dict | None, stage: str) -> dict:
    """``paper_f2.paired`` rerun: reproduce the stored difference and interval (stage ``check``), then the P value
    from 10,000 resamples (stage ``p``)."""
    from icutwin.eval import bench

    y, ps = x["lab"][prog], x["preds"](prog)
    p1, p2, w, stay, ds = ps["twin_128"], ps.get(other), x["w"], x["stay"], x["ds"]
    if p2 is None:
        return {"error": f"no {other}"}
    ok = np.isfinite(y) & np.isfinite(p1) & np.isfinite(p2)
    if not ok.sum() or not 0 < y[ok].sum() < ok.sum():
        return {"error": "one class"}
    d = bench.auroc_or_none(y[ok], p1[ok], w[ok]) - bench.auroc_or_none(y[ok], p2[ok], w[ok])
    out = {"estimate": float(d), "n": int(ok.sum()), "positives": int(y[ok].sum())}
    if stage == "check":
        ci = bench.bootstrap(_paired_slow, stay[ok], y[ok], p1[ok], p2[ok], w[ok], n=B_CI, strata=ds[ok])
        out["ci"] = list(ci)
        out["reproduced"] = bool(stored and close(d, stored["diff"]) and close(ci[0], stored["ci"][0])
                                 and close(ci[1], stored["ci"][1]))
        return out
    memo = x.setdefault("p_memo", {})
    if (prog, other) not in memo:
        draws = bench.bootstrap_draws(_paired_fast, stay[ok], y[ok], p1[ok], p2[ok], w[ok], n=B_P, seed=SEED_P,
                                      strata=ds[ok])
        memo[(prog, other)] = p_boot(draws, d)
    out["p"], out["draws"] = memo[(prog, other)]
    return out


# ---- family 4: amendment 10's comparisons (tools/paper_comparators.py) ------------------------------------------

def comparator_inputs(root: Path, name: str, prog: str) -> dict | None:
    import pandas as pd
    from paper_comparators import scores_for

    sub, a1d, design = SUBSETS[name]
    tw = pd.read_parquet(root / sub / "predictions.parquet")
    a1_tw = pd.read_parquet(root / a1d / "predictions.parquet")
    clin = pd.read_parquet(root / "f10/clinical/scores.parquet")
    sc = clin[clin["design"] == design].drop_duplicates("anchor").set_index("anchor")
    t = tw[tw["program"] == prog].drop_duplicates("anchor").set_index("anchor")
    t = t[t.index.isin(set(a1_tw.loc[a1_tw["program"] == prog, "anchor"]))]
    if len(t) < 30 or t["label"].nunique() < 2:
        return None
    s = sc.reindex(t.index)
    return {"y": t["label"].to_numpy(dtype=np.float64), "w": t["weight"].to_numpy(dtype=np.float64),
            "stay": t["stay"].astype(str).to_numpy(), "ds": t["dataset"].astype(str).to_numpy(),
            "ref": t["p"].to_numpy(dtype=np.float64),
            "scores": {c: s[c].to_numpy(dtype=np.float64) for c in scores_for(design, prog) if c in CLINICAL
                       and c in s}}


def comparator_draws(x: dict, score: str, n: int, seed: int) -> np.ndarray:
    """``paper_comparators.evaluate``'s paired differences (the reference minus the score) on ``n`` resamples."""
    from icutwin.eval import bench

    y, w, r, p = x["y"], x["w"], x["ref"], x["scores"][score]
    idx = bench.resamples(x["stay"], n=n, seed=seed, strata=x["ds"])
    m, r_ok = np.isfinite(p), np.isfinite(r)
    both = m & r_ok
    out = np.full(n, np.nan)
    for i, ix in enumerate(idx):
        if both.all():
            out[i] = bench.auroc_fast(y[ix], r[ix], w[ix]) - bench.auroc_fast(y[ix], p[ix], w[ix])
        else:
            kb = ix[both[ix]]
            out[i] = bench.auroc_fast(y[kb], r[kb], w[kb]) - bench.auroc_fast(y[kb], p[kb], w[kb])
    return out


# ---- the families ------------------------------------------------------------------------------------------------

def headline(root: Path, stage: str, cache: dict) -> dict:
    """Families 1-3 (stage ``check``: reproductions only; stage ``p``: P values)."""
    res = {1: [], 2: [], 3: []}
    for name, prog in HEADLINE:
        x = cache.setdefault(name, f1_inputs(root, name))
        f2 = x["f2"]
        e2 = prog in E2_PROGRAMS[name]
        for fam, (other, label) in VERSUS.items():
            row = {"family": fam, "subset": name, "program": prog, "endpoint": NAMES.get(prog, prog),
                   "comparison": f"Didymos minus {label}" if fam < 3 else "follow-up minus pre-specified",
                   "comparator": other}
            if e2:
                key = "e2_minus_twin_a1_32" if fam == 3 else f"e2_minus_{other}"
                st = (f2.get("f2", {}).get(prog, {}).get("paired") or {}).get(key)
                row.update({"estimate_kind": "hazard", "method": "from the stored interval (Altman and Bland)"})
                if st:
                    row.update({"estimate": st["diff"], "ci": st["ci"], "n": st.get("n"),
                                "positives": st.get("positives"),
                                "p": p_ci(st["diff"], *st["ci"]) if stage == "p" else None})
            else:
                st = (f2.get("f1", {}).get(prog, {}).get("paired") or {}).get(f"twin_128_minus_{other}")
                r = f1_exact(x, prog, other, st, stage)
                row.update({"estimate_kind": "sampled", "method": "exact paired bootstrap (10,000 resamples)",
                            **{k: v for k, v in r.items() if k != "ci"}})
                row["ci"] = st["ci"] if st else r.get("ci")
                if stage == "check":
                    row["reproduced"] = r.get("reproduced")
            res[fam].append(row)
    return res


def approximation_check(root: Path, cache: dict) -> dict:
    """The CI-derived P against the exact P on every sampled-estimate comparison of the F1 block (families 1-3)."""
    rows = []
    for name in SUBSETS:
        x = cache.setdefault(name, f1_inputs(root, name))
        for prog, blk in x["f2"].get("f1", {}).items():
            for other in ("lgbm_pooled", "tabpfn_v3_5", "twin_a1_32"):
                st = (blk.get("paired") or {}).get(f"twin_128_minus_{other}")
                if not st:
                    continue
                r = f1_exact(x, prog, other, st, "p")
                pa = p_ci(st["diff"], *st["ci"])
                if r.get("p") is not None and pa is not None:
                    rows.append({"subset": name, "program": prog, "comparator": other, "p_exact": r["p"],
                                 "p_from_ci": pa})
    if not rows:
        return {"comparisons": 0}
    le = np.log10([max(r["p_exact"], 1e-12) for r in rows])
    la = np.log10([max(r["p_from_ci"], 1e-12) for r in rows])
    same = [(r["p_exact"] < 0.05) == (r["p_from_ci"] < 0.05) for r in rows]
    return {"comparisons": len(rows), "median_abs_log10_difference": float(np.median(np.abs(le - la))),
            "max_abs_log10_difference": float(np.max(np.abs(le - la))), "same_decision_at_0.05": float(np.mean(same)),
            "rows": rows}


def clinical(root: Path, stage: str) -> list:
    """Family 4: the sampled estimate minus the best clinical score per endpoint."""
    from paper_comparators import evaluate

    comp = json.loads((root / "f10/comparators.json").read_text())
    rows = []
    for name, prog in HEADLINE:
        x = comparator_inputs(root, name, prog)
        stored = comp["subsets"].get(name, {}).get(prog, {}).get("models", {})
        cands = {c: stored[c] for c in (x["scores"] if x else {}) if c in stored}
        if not x or not cands:
            continue
        best = max(cands, key=lambda c: cands[c]["auroc"])
        st = cands[best]
        row = {"family": 4, "subset": name, "program": prog, "endpoint": NAMES.get(prog, prog),
               "comparison": f"Didymos (sampled) minus {SCORE_NAMES[best]}", "comparator": best,
               "estimate_kind": "sampled", "method": "exact paired bootstrap (10,000 resamples)",
               "estimate": st["ref_minus"], "ci": st["ref_minus_ci"], "n": st["n"], "positives": st["positives"],
               "comparator_auroc": st["auroc"]}
        if stage == "check":
            ev = evaluate(x["y"], x["w"], x["stay"], x["ds"], {"twin_128": x["ref"], best: x["scores"][best]},
                          "twin_128", B_CI)
            r = ev.get(best, {})
            row["reproduced"] = bool(close(r.get("ref_minus"), st["ref_minus"])
                                     and close(r.get("ref_minus_ci", [None])[0], st["ref_minus_ci"][0])
                                     and close(r.get("ref_minus_ci", [None, None])[1], st["ref_minus_ci"][1]))
        else:
            row["p"], row["draws"] = p_boot(comparator_draws(x, best, B_P, SEED_P), st["ref_minus"])
        rows.append(row)
    return rows


def calibration(root: Path, toolkit: Path, stage: str) -> list:
    """Family 5: on V1's draws (rolling anchors, 32 rollouts), O/E = 1 raw and recalibrated, and the changes in O/E
    and ECE, for the programs whose maps are in force."""
    import pandas as pd

    from icutwin.eval import bench
    from icutwin.toolkit import recalibration as rc

    v1 = toolkit / "v1/forecast"
    if not (v1 / "e2.parquet").exists():
        return []
    maps = rc.load(toolkit / "recalibration/MUC32_main.json")
    a1 = pd.read_parquet(root / "a1/test_R/predictions.parquet")
    preds, e2 = pd.read_parquet(v1 / "predictions.parquet"), pd.read_parquet(v1 / "e2.parquet")
    who = preds.drop_duplicates("anchor").set_index("anchor")[["stay", "dataset"]]
    v2 = json.loads((toolkit / "v2/v2.json").read_text())["results"]
    v2b = json.loads((toolkit / "v2/v2b.json").read_text())["programs"]
    e2p = ("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880")

    def oe(yy, pp, ww):
        return float(np.average(yy, weights=ww) / max(np.average(pp, weights=ww), 1e-9))

    rows = []
    for prog in sorted(maps):
        lab = a1[a1["program"] == prog][["anchor", "label", "weight"]]
        src = e2[e2["program"] == prog].rename(columns={"e2": "p"}) if prog in e2p else preds[preds["program"] == prog]
        x = lab.merge(src[["anchor", "p"]], on="anchor")
        x = x[np.isfinite(x["label"]) & np.isfinite(x["p"])]
        y, p, w = x["label"].to_numpy(), x["p"].to_numpy(), x["weight"].to_numpy()
        pc, ok = rc.apply(maps, prog, p)
        if not ok:
            continue
        g, s = who.reindex(x["anchor"])["stay"].astype(str).to_numpy(), who.reindex(x["anchor"])["dataset"].to_numpy()
        raw, cal = bench.binary_metrics(y, p, w), bench.binary_metrics(y, pc, w)
        kind = maps[prog].get("kind", "isotonic")
        ref = (v2b[prog]["evaluation"]["32"] if kind == "logistic" else v2.get(f"{prog}|32"))
        stats = {"oe_raw": (lambda yy, pp, qq, ww: oe(yy, pp, ww), raw["observed_over_expected"], 1.0),
                 "oe_recalibrated": (lambda yy, pp, qq, ww: oe(yy, qq, ww), cal["observed_over_expected"], 1.0),
                 "oe_change": (lambda yy, pp, qq, ww: oe(yy, qq, ww) - oe(yy, pp, ww),
                               cal["observed_over_expected"] - raw["observed_over_expected"], 0.0),
                 "ece_change": (lambda yy, pp, qq, ww: bench.ece(yy, qq, ww) - bench.ece(yy, pp, ww),
                                cal["ece"] - raw["ece"], 0.0)}
        for stat, (fn, est, null) in stats.items():
            row = {"family": 5, "subset": "V1 (rolling anchors, 32 rollouts)", "program": prog,
                   "endpoint": NAMES.get(prog, prog), "comparison": stat, "map": kind,
                   "estimate_kind": "hazard" if prog in e2p else "sampled", "estimate": float(est), "null": null,
                   "n": int(len(y)), "positives": int(y.sum()),
                   "method": "paired bootstrap over stays within datasets (CI 1,000, seed 0; P 10,000, seed 1)"}
            if stage == "check":
                row["reproduced"] = bool(ref and close(raw["observed_over_expected"],
                                                       ref["raw"]["observed_over_expected"])
                                         and close(cal["observed_over_expected"],
                                                   ref["recalibrated"]["observed_over_expected"])
                                         and close(raw["ece"], ref["raw"]["ece"])
                                         and close(cal["ece"], ref["recalibrated"]["ece"]))
            else:
                row["ci"] = list(bench.bootstrap(fn, g, y, p, pc, w, n=B_CI, seed=0, strata=s))
                row["p"], row["draws"] = p_boot(bench.bootstrap_draws(fn, g, y, p, pc, w, n=B_P, seed=SEED_P,
                                                                      strata=s), est, null)
            rows.append(row)
    return rows


def whatif(root: Path) -> tuple[list, list]:
    """Families 6 (Didymos's factual contrast minus the observational one) and 7 (the null arm's sign share)."""
    from scipy.stats import binomtest, norm

    c1b = json.loads((root / "f5/c1b.json").read_text())
    agree, null = [], []
    for tr, resp in c1b["factual"].items():
        for r, x in resp.items():
            if not isinstance(x, dict) or "observational_contrast" not in x or "twin_factual_contrast" not in x:
                continue
            o, t = x["observational_contrast"], x["twin_factual_contrast"]
            if None in (o.get("mean"), t.get("mean")) or None in (*o.get("ci", [None]), *t.get("ci", [None])):
                continue
            se = np.hypot((o["ci"][1] - o["ci"][0]) / 3.919927969, (t["ci"][1] - t["ci"][0]) / 3.919927969)
            d = t["mean"] - o["mean"]
            agree.append({"family": 6, "treatment": tr, "response": r, "comparison": "Didymos minus observational",
                          "observational": o["mean"], "observational_ci": o["ci"],
                          "observational_pairs": o.get("pairs"),
                          "didymos": t["mean"], "didymos_ci": t["ci"], "didymos_pairs": t.get("pairs"),
                          "estimate": float(d), "ci": [float(d - 1.959963984540054 * se),
                                                       float(d + 1.959963984540054 * se)],
                          "p": float(2 * norm.sf(abs(d) / se)) if se > 0 else None,
                          "method": "normal, the two intervals' standard errors combined as if independent"})
    for tr, resp in c1b["aa"].items():
        for r, x in resp.items():
            if not isinstance(x, dict) or not x.get("prefixes"):
                continue
            n = int(x["prefixes"])
            k = int(round(x["share_expected_sign"] * n))
            null.append({"family": 7, "treatment": tr, "response": r, "comparison": "null-arm sign share = 0.5",
                         "estimate": k / n, "k": k, "n": n, "null_mean": x.get("mean"),
                         "null_mean_ci": x.get("mean_ci"),
                         "ci": list(binomtest(k, n, 0.5).proportion_ci(method="exact")),
                         "p": float(binomtest(k, n, 0.5).pvalue), "method": "exact two-sided binomial"})
    return agree, null


def spearman_exact(x, y) -> tuple[float, float]:
    """Spearman's r and its exact two-sided permutation P (every ordering; no ties assumed)."""
    from scipy.stats import rankdata

    rx, ry = rankdata(x), rankdata(y)
    n = len(rx)
    r = float(np.corrcoef(rx, ry)[0, 1])
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int16)
    ryp = ry[perms]
    rr = ((rx - rx.mean()) * (ryp - ry.mean())).sum(1) / np.sqrt(((rx - rx.mean()) ** 2).sum()
                                                                 * ((ry - ry.mean()) ** 2).sum())
    return r, float(np.mean(np.abs(rr) >= abs(r) - 1e-12))


def leakage(root: Path) -> list:
    """Family 8: per dataset, membership AUC against the member/non-member NLL gap over the checkpoints."""
    a3 = json.loads((root / "a3/a3.json").read_text())
    rows = []
    for ds, x in sorted(a3["frontier"].items()):
        pts = [p for p in x["points"] if p.get("auc") is not None and p.get("nll_gap") is not None]
        if len(pts) < 4:
            continue
        r, p = spearman_exact([q["nll_gap"] for q in pts], [q["auc"] for q in pts])
        rows.append({"family": 8, "dataset": ds, "comparison": "Spearman(membership AUC, NLL gap)", "estimate": r,
                     "n": len(pts), "p": p, "method": "exact permutation (all orderings)",
                     "checkpoints": [q["run"] for q in pts]})
    return rows


def augmentation(root: Path) -> list:
    """Family 9: per design, real-stay count and twin multiple, the cells' mean gains: a sign test and a Wilcoxon
    signed-rank test."""
    from scipy.stats import binomtest, wilcoxon

    rows = []
    for design in ("L24", "R"):
        f = root / f"f8/aug_{design}.json"
        if not f.exists():
            continue
        aug = json.loads(f.read_text())
        cells = {}
        for _prog, by in aug["programs"].items():
            for _ds, x in by.items():
                if x.get("test_positives", 0) < 20:      # the materials' summary cells (paper_results_followup)
                    continue
                for c, v in (x.get("cells") or {}).items():
                    if re.fullmatch(r"n\d+_m\d+", c) and v and v.get("diff_mean") is not None \
                            and np.isfinite(v["diff_mean"]):
                        cells.setdefault(c, []).append(v["diff_mean"])
        for c in sorted(cells, key=lambda s: (int(s.split("_")[0][1:]), s)):
            d = np.asarray(cells[c], float)
            nz = d[d != 0]
            pos = int((nz > 0).sum())
            sign_p = float(binomtest(pos, len(nz), 0.5).pvalue) if len(nz) else None
            wil_p = float(wilcoxon(d).pvalue) if len(nz) >= 2 else None
            rows.append({"family": 9, "design": design, "cell": c, "cells": int(len(d)), "positive": pos,
                         "estimate": float(np.mean(d)), "median": float(np.median(d)), "p_sign": sign_p,
                         "p_wilcoxon": wil_p, "method": "two-sided sign test; two-sided Wilcoxon signed-rank"})
    return rows


def attribution(root: Path) -> list:
    """Family 10: knockout's agreement with editing minus attention rollout's (group sums), per anchor."""
    from scipy.stats import wilcoxon

    rows_in = [json.loads(s) for s in (root / "f4/attn_diag/per_anchor.jsonl").read_text().splitlines() if s.strip()]
    out = []
    groups = {"pooled": rows_in}
    for r in rows_in:
        groups.setdefault(r["program"], []).append(r)
    for name, rs in groups.items():
        d = np.array([r["edit_knockout_sum"] - r["edit_rollout_sum"] for r in rs
                      if r.get("edit_knockout_sum") is not None and r.get("edit_rollout_sum") is not None], float)
        d = d[np.isfinite(d)]
        if len(d) < 5:
            continue
        out.append({"family": 10, "program": name, "endpoint": NAMES.get(name, name),
                    "comparison": "knockout minus rollout, agreement with editing", "estimate": float(np.median(d)),
                    "n": int(len(d)), "p": float(wilcoxon(d).pvalue), "method": "two-sided Wilcoxon signed-rank"})
    return out


def trials(toolkit: Path) -> list:
    """The trial simulations (V4): Clopper-Pearson intervals for the type I error; Monte Carlo intervals for the
    sample-size ratio from a parametric bootstrap of the two simulated powers."""
    from scipy.stats import binomtest

    from icutwin.toolkit.trial import n_for_power

    f = toolkit / "v4/v4.json"
    if not f.exists():
        return []
    v4 = json.loads(f.read_text())
    rng = np.random.default_rng(SEED_P)
    rows = []
    for cell, c in v4["cells"].items():
        alpha, target = float(c.get("alpha", 0.05)), float(c.get("power", 0.8))
        for score, s in c.get("scores", {}).items():
            a = (s or {}).get("adjustment")
            if not a:
                continue
            sims = 2000
            n0 = int(a["n_simulated"])
            pa = rng.binomial(sims, a["power_adjusted"], size=B_P) / sims
            pu = rng.binomial(sims, a["power_unadjusted"], size=B_P) / sims
            ratio = np.array([n_for_power(x, n0, alpha, target) / n_for_power(u, n0, alpha, target)
                              for x, u in zip(pa, pu, strict=True)])
            ratio = ratio[np.isfinite(ratio)]
            t1 = {}
            for k_ in ("type1_unadjusted", "type1_adjusted"):
                k = int(round(a[k_] * sims))
                t1[k_] = {"estimate": a[k_], "k": k, "n": sims,
                          "ci": list(binomtest(k, sims, 0.5).proportion_ci(method="exact"))}
            rows.append({"family": "trial", "cell": cell, "score": score, "n_ratio": a["n_ratio"],
                         "n_ratio_ci": [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))],
                         "power_unadjusted": a["power_unadjusted"], "power_adjusted": a["power_adjusted"],
                         "simulations": sims, **t1,
                         "method": "Clopper-Pearson; parametric bootstrap of the two powers (10,000, seed 1)"})
    return rows


# ---- amendment 14: every scored endpoint (family 11) and label efficiency (item 12) -------------------------------

BREADTH = ("R", "L24", "L48")
B_JOINT, SEED_JOINT = 1000, 2


def program_family(prog: str, fams: dict) -> str:
    """The catalog's family of a program (``remaining_los_gt_7d`` comes from the F7 band program)."""
    base = prog.partition("@")[0]
    return "F7" if base == "remaining_los_gt_7d" else fams[base]


def catalog_families() -> dict:
    import yaml

    from icutwin.factory import programs as fp

    cat = yaml.safe_load(fp.CATALOG_YAML.read_text(encoding="utf-8"))
    return {e["name"]: e["family"] for e in cat["programs"] if isinstance(e, dict) and "name" in e}


def breadth_inputs(root: Path, design: str) -> dict:
    """``tools/paper_compare.py``'s A1 pairing for every program of a design: the common anchors (those every
    predictor scored) in its order, the Twin's share, LightGBM's prediction and the stored comparison."""
    import pandas as pd

    dd = root / "a1" / f"test_{design}"
    tw = pd.read_parquet(dd / "predictions.parquet")
    tw["model"] = "twin"
    parts = [tw[["anchor", "dataset", "program", "model", "p", "label", "weight", "stay"]]]
    files = [dd / "predictions.parquet", dd / "compare.json"]
    for f in sorted(dd.glob("*_predictions.parquet")):          # baselines, the probe, TabPFN (v2, 3.5)
        parts.append(pd.read_parquet(f))
        files.append(f)
    lab = tw[["anchor", "program", "label", "weight", "stay"]].drop_duplicates(["anchor", "program"])
    allp = pd.concat([p.drop(columns=[c for c in ("label", "weight", "stay") if c in p], errors="ignore")
                      for p in parts]).merge(lab, on=["anchor", "program"])
    stored = json.loads((dd / "compare.json").read_text())["a1"]
    progs = {}
    for prog, g in allp.groupby("program"):
        st = stored.get(prog) or {}
        if st.get("twin_minus_lgbm") is None:
            continue
        wide = g.pivot_table(index="anchor", columns="model", values="p", aggfunc="first")
        meta = g.drop_duplicates("anchor").set_index("anchor").loc[wide.index]
        common = wide.dropna()
        m = wide.index.isin(common.index)
        progs[prog] = {"y": meta["label"].to_numpy()[m], "w": meta["weight"].to_numpy()[m],
                       "ds": meta["dataset"].to_numpy()[m], "stay": meta["stay"].to_numpy()[m],
                       "twin": common["twin"].to_numpy(), "lgbm": common["lgbm_pooled"].to_numpy(),
                       "stored": {"diff": st["twin_minus_lgbm"], "ci": st["twin_minus_lgbm_ci"]}}
    return {"programs": progs, "files": files}


def _breadth_check(item):
    """One program's stored difference and interval, recomputed (``paper_compare``'s bootstrap)."""
    from icutwin.eval import bench

    key, x = item
    d = _paired_slow(x["y"], x["twin"], x["lgbm"], x["w"])
    ci = bench.bootstrap(_paired_slow, x["stay"], x["y"], x["twin"], x["lgbm"], x["w"], n=B_CI, strata=x["ds"])
    st = x["stored"]
    ok = close(d, st["diff"]) and close(ci[0], st["ci"][0]) and close(ci[1], st["ci"][1])
    return key, {"estimate": float(d), "ci": [float(ci[0]), float(ci[1])], "reproduced": bool(ok)}


def _breadth_p(item):
    """One program's P value from 10,000 paired resamples (seed 1)."""
    from icutwin.eval import bench

    key, x = item
    d = _paired_slow(x["y"], x["twin"], x["lgbm"], x["w"])
    draws = bench.bootstrap_draws(_paired_fast, x["stay"], x["y"], x["twin"], x["lgbm"], x["w"], n=B_P,
                                  seed=SEED_P, strata=x["ds"])
    p, n = p_boot(draws, d)
    return key, {"p": p, "draws": n}


def bh_bounds(ps: list, bounds: list) -> tuple[list, list]:
    """Benjamini-Hochberg q values (equal to statsmodels' ``fdr_bh``) and whether each is an upper bound.

    q at rank i is the minimum of m p / rank over ranks i and above. It is an upper bound when any term it could take
    is one: its own P, or the q above it. Ties put bounds first, since a bound's true P lies at or below its value."""
    from statsmodels.stats.multitest import multipletests

    idx = sorted((i for i, p in enumerate(ps) if p is not None), key=lambda i: (ps[i], not bounds[i]))
    m, q, flag = len(idx), [None] * len(ps), [False] * len(ps)
    run, run_b = 1.0, False
    for j in range(m - 1, -1, -1):
        i = idx[j]
        run = min(run, ps[i] * m / (j + 1))
        run_b = run_b or bool(bounds[i])
        q[i], flag[i] = min(1.0, run), bool(run_b and run < 1.0)
    if idx:
        ref = multipletests([ps[i] for i in idx], method="fdr_bh")[1]
        assert all(abs(q[i] - r) < 1e-12 for i, r in zip(idx, ref, strict=True))
    return q, flag


def breadth_joint(inp: dict, keys: list, n: int = B_JOINT, seed: int = SEED_JOINT) -> np.ndarray:
    """Every program's paired difference on one joint bootstrap: the held-out stays (identified by dataset and stay)
    resampled within datasets once per resample, for every program of every anchor set (rows: resamples; columns:
    ``keys``)."""
    stay_ds, ids = {}, {}
    for k in keys:
        x = inp[k[0]]["programs"][k[1]]
        ids[k] = np.array([f"{d}|{s}" for d, s in zip(x["ds"], x["stay"], strict=True)])
        for s, d in zip(ids[k], x["ds"], strict=True):
            stay_ds.setdefault(s, str(d))
    stays = np.array(sorted(stay_ds))
    dss = np.array([stay_ds[s] for s in stays])
    pos = {s: i for i, s in enumerate(stays)}
    where = {k: np.array([pos[s] for s in ids[k]]) for k in keys}
    strata = [np.flatnonzero(dss == g) for g in np.unique(dss)]
    arrays = {k: tuple(np.asarray(inp[k[0]]["programs"][k[1]][c], dtype=np.float64) for c in ("y", "twin", "lgbm", "w"))
              for k in keys}
    rng = np.random.default_rng(seed)
    out = np.full((n, len(keys)), np.nan)
    for i in range(n):
        count = np.zeros(len(stays), dtype=np.int64)
        for members in strata:
            np.add.at(count, rng.choice(members, size=len(members)), 1)
        for j, k in enumerate(keys):
            y, a, b, w = arrays[k]
            ix = np.repeat(np.arange(len(y)), count[where[k]])
            out[i, j] = _paired_fast(y[ix], a[ix], b[ix], w[ix])
    return out


def breadth(root: Path, workers: int) -> tuple[list, dict, list]:
    """Family 11: every program A1 compared with LightGBM on R, L24 and L48. The stored intervals are reproduced first
    (the function stops otherwise); then the P values, Benjamini-Hochberg q values and Holm's adjusted P values, and
    the joint-bootstrap summaries."""
    from concurrent.futures import ProcessPoolExecutor

    import paper_figstyle as st

    fams = catalog_families()
    inp = {d: breadth_inputs(root, d) for d in BREADTH}
    keys = [(d, p) for d in BREADTH for p in sorted(inp[d]["programs"])]
    items = [(k, inp[k[0]]["programs"][k[1]]) for k in keys]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        checks = dict(ex.map(_breadth_check, items, chunksize=2))
    bad = [k for k, r in checks.items() if not r["reproduced"]]
    print(json.dumps({"family 11 reproductions": len(checks), "failed": len(bad)}), flush=True)
    if bad:
        print(json.dumps({"not reproduced": [list(k) for k in bad]}), flush=True)
        raise NotReproduced("family 11: stored intervals not reproduced; no P value computed")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        pvals = dict(ex.map(_breadth_p, items, chunksize=2))
    rows = []
    for k in keys:
        x = inp[k[0]]["programs"][k[1]]
        rows.append({"family": 11, "design": k[0], "program": k[1], "endpoint": st.program_name(k[1]),
                     "catalog_family": program_family(k[1], fams), "n": int(len(x["y"])),
                     "positives": int(np.nansum(np.asarray(x["y"], dtype=float))), "estimate": checks[k]["estimate"],
                     "ci": x["stored"]["ci"], "p": pvals[k]["p"], "draws": pvals[k]["draws"],
                     "method": "exact paired bootstrap (10,000 resamples), A1's common anchors"})
    bounds = [at_floor(r["p"], r["draws"]) for r in rows]
    q, qb = bh_bounds([r["p"] for r in rows], bounds)
    for r, b, qq, f in zip(rows, bounds, q, qb, strict=True):
        r["p_bound"], r["q_bh"], r["q_bh_bound"] = b, qq, f
    assign_holm(rows)
    joint = breadth_joint(inp, keys)
    est = np.array([r["estimate"] for r in rows])
    summary = {"resamples": B_JOINT, "seed": SEED_JOINT, "groups": {}}
    for name in ("all", *BREADTH):
        cols = [j for j, k in enumerate(keys) if name == "all" or k[0] == name]
        e, jb = est[cols], joint[:, cols]
        sel = [rows[j] for j in cols]
        summary["groups"][name] = {
            "programs": len(cols), "positive": int((e > 0).sum()), "negative": int((e < 0).sum()),
            "median": float(np.median(e)),
            "median_ci": [float(v) for v in np.percentile(np.nanmedian(jb, axis=1), [2.5, 97.5])],
            "share_positive": float(np.mean(e > 0)),
            "share_positive_ci": [float(v) for v in np.percentile(np.mean(jb > 0, axis=1), [2.5, 97.5])],
            "ci_above_zero": sum(r["ci"][0] > 0 for r in sel), "ci_below_zero": sum(r["ci"][1] < 0 for r in sel),
            "bh_favour_didymos": sum(r["q_bh"] <= 0.05 and r["estimate"] > 0 for r in sel),
            "bh_favour_lightgbm": sum(r["q_bh"] <= 0.05 and r["estimate"] < 0 for r in sel),
            "holm_favour_didymos": sum(r["p_holm"] <= 0.05 and r["estimate"] > 0 for r in sel),
            "holm_favour_lightgbm": sum(r["p_holm"] <= 0.05 and r["estimate"] < 0 for r in sel)}
    summary["catalog_families"] = {}
    for f in st.FAMILY:
        sel = [r for r in rows if r["catalog_family"] == f]
        if sel:
            summary["catalog_families"][f] = {
                "name": st.FAMILY[f], "programs": len(sel), "positive": sum(r["estimate"] > 0 for r in sel),
                "bh_favour_didymos": sum(r["q_bh"] <= 0.05 and r["estimate"] > 0 for r in sel),
                "bh_favour_lightgbm": sum(r["q_bh"] <= 0.05 and r["estimate"] < 0 for r in sel),
                "median": float(np.median([r["estimate"] for r in sel]))}
    files = sorted({str(f) for d in BREADTH for f in inp[d]["files"]})
    return rows, summary, files


def label_efficiency(root: Path) -> tuple[list, dict]:
    """Item 12: the training size at which LightGBM per dataset matches the Twin's per-dataset AUROC, per primary
    endpoint (interpolated linearly in AUROC against log10 n between the first size that reaches it and the one
    before; 'at most' the smallest size, or 'not reached')."""
    import paper_figstyle as st

    rows = []
    for design in ("R", "L24"):
        bl = json.loads((root / f"a1/test_{design}/baselines.json").read_text())["programs"]
        a1 = json.loads((root / f"a1/test_{design}/a1.json").read_text())["programs"]
        for prog in PRIMARY[design]:
            eff = (bl.get(prog) or {}).get("label_efficiency") or {}
            tw = (a1.get(prog) or {}).get("datasets") or {}
            for ds in st.DATASETS:
                e, t = eff.get(ds) or {}, (tw.get(ds) or {}).get("auroc")
                curve = sorted((int(k), float(v)) for k, v in e.items() if v is not None)
                if t is None or not curve:
                    continue
                ns, vs = [c[0] for c in curve], [c[1] for c in curve]
                first = next((i for i, v in enumerate(vs) if v >= t), None)
                if first is None:
                    status, n_match = "not reached", None
                elif first == 0:
                    status, n_match = "at most", float(ns[0])
                else:
                    x0, x1, a0, a1_ = np.log10(ns[first - 1]), np.log10(ns[first]), vs[first - 1], vs[first]
                    status, n_match = "interpolated", float(10 ** (x0 + (t - a0) / (a1_ - a0) * (x1 - x0)))
                rows.append({"design": design, "program": prog, "endpoint": st.program_name(prog), "dataset": ds,
                             "didymos": float(t), "curve": {str(n): v for n, v in curve}, "max_n": ns[-1],
                             "status": status, "n_match": n_match,
                             "over_1000": status == "not reached" or (n_match is not None and n_match > 1000)})
    summary = {"cells": len(rows), "over_1000_or_not_reached": sum(r["over_1000"] for r in rows),
               "not_reached": sum(r["status"] == "not reached" for r in rows),
               "at_most_smallest": sum(r["status"] == "at most" for r in rows)}
    summary["share_over_1000_or_not_reached"] = summary["over_1000_or_not_reached"] / max(1, summary["cells"])
    return rows, summary


# ---- amendment 15: ranks across endpoints (items 13 and 14) ------------------------------------------------------

B_RANK, SEED_RANK = 10_000, 3
RANK_LEARNED = ("lgbm_pooled", "lgbm_per_dataset", "logistic_pooled", "tabpfn_v3_5", "tabpfn_v2", "logistic_l1",
                "logistic_enet", "cart", "random_forest", "extra_trees", "naive_bayes", "knn", "mlp")
RANK_CLINICAL = ("sofa", "news", "mews", "qsofa", "sirs")
BREADTH_PREDICTORS = ("twin", "lgbm_pooled", "logistic_pooled", "twin_probe")     # scored on every program


def rank_summary(mat: np.ndarray, names: list, ref: str, seed: int = SEED_RANK, n_boot: int = B_RANK) -> dict:
    """Ranks of the predictors (columns) over tasks (rows), rank 1 the highest AUROC with mid-ranks for ties: mean ranks
    with a bootstrap interval over tasks, firsts, the Friedman test, and ``ref`` against every other predictor
    (two-sided Wilcoxon signed-rank on the paired AUROCs, Holm-adjusted)."""
    from scipy.stats import friedmanchisquare, rankdata, wilcoxon

    ranks = np.vstack([rankdata(-row) for row in mat])
    rng = np.random.default_rng(seed)
    boot = ranks[rng.integers(0, len(mat), size=(n_boot, len(mat)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    fr = friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
    r = names.index(ref)
    comps = []
    for j, n in enumerate(names):
        if j == r:
            continue
        d = mat[:, r] - mat[:, j]
        comps.append({"comparator": n, "median_difference": float(np.median(d)), "wins": int((d > 0).sum()),
                      "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
                      "p": float(wilcoxon(d, alternative="two-sided").pvalue)})
    assign_holm(comps)
    order = np.argsort(ranks.mean(0), kind="stable")
    return {"tasks": int(len(mat)), "predictors": [
        {"predictor": names[j], "mean_rank": float(ranks[:, j].mean()), "mean_rank_ci": [float(lo[j]), float(hi[j])],
         "firsts": int((ranks[:, j] == 1).sum()), "mean_auroc": float(mat[:, j].mean())} for j in order],
        "friedman": {"statistic": float(fr.statistic), "p": float(fr.pvalue)}, "reference": ref,
        "comparisons": comps, "ranks": ranks.tolist(), "bootstrap": {"resamples": n_boot, "seed": seed}}


def ranks_primary(root: Path) -> tuple[dict, list]:
    """Item 13: every predictor scored on the 12 primary endpoints where all learned comparators were fitted (the
    follow-up anchors); Didymos's current estimate is the hazard estimate where it applies."""
    comp = json.loads((root / "f10/comparators.json").read_text())
    tasks = [k for k in HEADLINE if not (k[0] == "L24x" and k[1] == "decompensation@1440")]
    names = ["twin_current", "twin_probe", *RANK_LEARNED, *RANK_CLINICAL]
    rows = []
    for sub, prog in tasks:
        models = comp["subsets"][sub][prog]["models"]
        if prog in E2_PROGRAMS[sub]:
            cur = json.loads((root / f"f1/{sub}/f2.json").read_text())["f2"][prog]["estimators"]["e2"]["auroc"]
        else:
            cur = models["twin_128"]["auroc"]
        row = [cur] + [(models.get(n) or {}).get("auroc") for n in names[1:]]
        rows.append(row)
    keep = [j for j in range(len(names)) if all(r[j] is not None for r in rows)]
    mat = np.array([[r[j] for j in keep] for r in rows], dtype=float)
    out = rank_summary(mat, [names[j] for j in keep], "twin_current")
    out["endpoints"] = [{"subset": s, "program": p, "endpoint": NAMES.get(p, p)} for s, p in tasks]
    out["dropped"] = [names[j] for j in range(len(names)) if j not in keep]
    out["auroc"] = mat.tolist()
    out["names"] = [names[j] for j in keep]
    return out, [str(root / "f10/comparators.json")] + [str(root / f"f1/{s}/f2.json") for s in ("R128", "L24x")]


def ranks_breadth(root: Path) -> tuple[dict, list]:
    """Item 14: the predictors A1 scored on every program, on each program's common anchors (``compare.json``)."""
    rows, tasks, files = [], [], []
    for design in BREADTH:
        f = root / f"a1/test_{design}/compare.json"
        files.append(str(f))
        for prog, x in sorted(json.loads(f.read_text())["a1"].items()):
            if x.get("twin_minus_lgbm") is None:
                continue
            vals = [x.get(n) for n in BREADTH_PREDICTORS]
            if any(v is None for v in vals):
                raise ValueError(f"{design} {prog}: a predictor of item 14 is not scored")
            rows.append(vals)
            tasks.append({"design": design, "program": prog})
    out = rank_summary(np.array(rows, dtype=float), list(BREADTH_PREDICTORS), "twin")
    out["programs"] = tasks
    out["names"] = list(BREADTH_PREDICTORS)
    return out, files


def amendment15(root: Path, out: Path, tables: str | None) -> None:
    """Amendment 15's rank summaries, added to an existing ``stats.json`` with their own git sha and inputs."""
    t0 = time.time()
    git = _sha()
    f = out / "stats.json"
    rep = json.loads(f.read_text(encoding="utf-8"))
    primary, f1 = ranks_primary(root)
    breadth_r, f2 = ranks_breadth(root)
    rep["ranks"] = {"primary": primary, "breadth": breadth_r}
    rep["amendment15"] = {"git": git, "inputs": {p: sha256(Path(p)) for p in sorted(set(f1 + f2))},
                          "seconds": round(time.time() - t0, 1)}
    f.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8", newline="\n")
    if tables:
        write_amendment15_tables(rep, Path(tables))
    top = primary["predictors"][:3]
    print(json.dumps({"primary_top3": [(t["predictor"], round(t["mean_rank"], 2)) for t in top],
                      "primary_friedman_p": primary["friedman"]["p"],
                      "breadth": [(t["predictor"], round(t["mean_rank"], 2)) for t in breadth_r["predictors"]],
                      "breadth_friedman_p": breadth_r["friedman"]["p"]}), flush=True)


def write_amendment15_tables(rep: dict, out: Path) -> None:
    """The rank tables, and Table 2 (the leaderboard) as CSV and Markdown, each endpoint's best in bold."""
    import paper_figstyle as st

    out.mkdir(parents=True, exist_ok=True)
    for key in ("primary", "breadth"):
        r = rep["ranks"][key]
        comps = {c["comparator"]: c for c in r["comparisons"]}
        with open(out / f"stats_ranks_{key}.csv", "w", newline="\n", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["predictor", "mean rank (95% CI)", "firsts", "mean AUROC", "Didymos minus, median",
                        "Didymos wins/ties/losses", "P (Wilcoxon)", "P (Holm)"])
            for p in r["predictors"]:
                c = comps.get(p["predictor"])
                w.writerow([st.PREDICTOR.get(p["predictor"], p["predictor"]),
                            f"{p['mean_rank']:.2f} ({p['mean_rank_ci'][0]:.2f} to {p['mean_rank_ci'][1]:.2f})",
                            p["firsts"], fmt(p["mean_auroc"]),
                            "-" if not c else fmt(c["median_difference"]),
                            "-" if not c else f"{c['wins']}/{c['ties']}/{c['losses']}",
                            "-" if not c else fmt_p(c["p"]), "-" if not c else fmt_p(c["p_holm"])])
    r = rep["ranks"]["primary"]
    mat = np.asarray(r["auroc"])
    names = r["names"]
    order = [names.index(p["predictor"]) for p in r["predictors"]]
    mean_rank = {p["predictor"]: p["mean_rank"] for p in r["predictors"]}
    heads = [st.ENDPOINT_SHORT.get(e["program"], e["endpoint"]) + (" (L)" if e["subset"] == "L24x" else "")
             for e in r["endpoints"]]
    best = mat.max(1)
    with open(out / "table2_leaderboard.csv", "w", newline="\n", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["predictor", *heads, "mean rank"])
        for j in order:
            w.writerow([st.PREDICTOR.get(names[j], names[j]), *[f"{v:.3f}" for v in mat[:, j]],
                        f"{mean_rank[names[j]]:.2f}"])
    md = ["| Predictor | " + " | ".join(heads) + " | Mean rank |", "|---" * (len(heads) + 2) + "|"]
    for j in order:                       # bold: the best at the precision shown (ties at three decimals all bold)
        cells = [f"**{v:.3f}**" if f"{v:.3f}" == f"{b:.3f}" else f"{v:.3f}" for v, b in zip(mat[:, j], best,
                                                                                          strict=True)]
        md.append(f"| {st.PREDICTOR.get(names[j], names[j])} | " + " | ".join(cells) +
                  f" | {mean_rank[names[j]]:.2f} |")
    (out / "table2_leaderboard.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")


# ---- amendment 16: families 15-20 --------------------------------------------------------------------------------

L48X_ESTIMATE = {"icu_death@-1": "e2", "hospital_death@0": "twin_128", "remaining_los_gt_7d": "twin_128",
                 "decompensation@1440": "e2"}


def ext_rows(root: Path, clif: Path, fam: int) -> list:
    """Families 15-17 (16a; ``runs/paper/ext/ext.json``, ``tools/paper_external.py analyze``): Didymos's estimate minus
    transferred LightGBM (15) or the local cross-fitted LightGBM (16), and the sampled estimate minus the best clinical
    score (17), per scored endpoint and database. The stored 1,000-resample interval is reproduced first (rows carry
    ``reproduced``); the P value is computed later, from 10,000 resamples."""
    from paper_external import pairs

    ext = json.loads((root / "ext/ext.json").read_text(encoding="utf-8"))
    rows = []
    for (design, d, prog), x in sorted(pairs(root / "ext", clif).items()):
        r = ext["designs"][design][d][prog]
        if fam in (15, 16):
            ref, comp = "twin", {15: "lgbm_pooled", 16: "lgbm_local"}[fam]
            stored = r["predictors"].get(comp)
        else:
            ref = "twin_sampled"
            cands = {k: v for k, v in r["sampled_vs_scores"].items() if k != "twin_sampled" and v.get("auroc")}
            if not cands:
                continue
            comp = max(cands, key=lambda k: cands[k]["auroc"])
            stored = cands[comp]
        if not stored or "ref_minus" not in stored or comp not in x:
            continue
        inp = {"y": x["y"].to_numpy(float), "w": x["w"].to_numpy(float), "stay": x["stay"].astype(str).to_numpy(),
               "ds": np.full(len(x), d), "ref": x[ref].to_numpy(float), "scores": {comp: x[comp].to_numpy(float)}}
        both = np.isfinite(inp["ref"]) & np.isfinite(inp["scores"][comp])
        ci = [float(v) for v in np.nanpercentile(comparator_draws(inp, comp, B_CI, 0), [2.5, 97.5])]
        rows.append({"family": fam, "design": design, "dataset": d, "program": prog, "endpoint": NAMES.get(prog, prog),
                     "comparison": f"{ref} minus {comp}", "comparator": comp,
                     "estimate_kind": r["estimate"] if ref == "twin" else "sampled",
                     "estimate": stored["ref_minus"], "ci": stored["ref_minus_ci"], "n": int(both.sum()),
                     "positives": int(inp["y"][both].sum()),
                     "reproduced": bool(close(ci[0], stored["ref_minus_ci"][0]) and close(ci[1], stored["ref_minus_ci"][1])),
                     "_inp": inp})
    return rows


def l48x_inputs(root: Path) -> dict:
    """L48x's arrays in ``tools/paper_f2.py``'s order: A1's labels, the 128-rollout share, E2 (``estimates.parquet``),
    A1's 32-rollout share and LightGBM pooled (A1's stored predictions, reindexed)."""
    import pandas as pd

    sd, a1 = root / "f1/L48x", root / "a1/test_L48"
    an = pd.read_parquet(sd / "anchors.parquet").drop_duplicates("anchor").reset_index(drop=True)
    pa1 = pd.read_parquet(a1 / "predictions.parquet")
    ps = pd.read_parquet(sd / "predictions.parquet")
    est = pd.read_parquet(sd / "estimates.parquet")
    bl = pd.read_parquet(a1 / "baseline_predictions.parquet")

    def col(df, prog):
        return df[df["program"] == prog].drop_duplicates("anchor").set_index("anchor")["p"].reindex(an["anchor"]) \
            .to_numpy(float)

    def preds(prog: str) -> dict:
        return {"twin_128": col(ps, prog), "e2": col(est[est["estimator"] == "e2"], prog),
                "twin_a1_32": col(pa1, prog), "lgbm_pooled": col(bl[bl["model"] == "lgbm_pooled"], prog)}

    return {"w": an["weight"].to_numpy(dtype=float), "ds": an["dataset"].to_numpy(),
            "stay": an["twin"].astype(str).to_numpy(), "preds": preds,
            "lab": {p: g.set_index("anchor")["label"].reindex(an["anchor"]).to_numpy(dtype=float)
                    for p, g in pa1.groupby("program")},
            "f2": json.loads((sd / "f2.json").read_text()),
            "files": [sd / "anchors.parquet", sd / "predictions.parquet", sd / "estimates.parquet", sd / "f2.json",
                      a1 / "predictions.parquet", a1 / "baseline_predictions.parquet"]}


def l48x_rows(x: dict, fam: int) -> list:
    """Families 18 (minus LightGBM) and 19 (minus A1's 32-rollout share) on L48x's 4 endpoints (16b), each with the
    estimate the amendment names; the stored interval (``f2.json``) reproduced first."""
    from icutwin.eval import bench

    other = {18: "lgbm_pooled", 19: "twin_a1_32"}[fam]
    rows = []
    for prog, est in L48X_ESTIMATE.items():
        y, ps = x["lab"].get(prog), x["preds"](prog)
        if y is None:
            continue
        p1, p2, w, stay, ds = ps[est], ps[other], x["w"], x["stay"], x["ds"]
        ok = np.isfinite(y) & np.isfinite(p1) & np.isfinite(p2)
        if not ok.sum() or not 0 < y[ok].sum() < ok.sum():
            continue
        sec = x["f2"]["f2"] if est == "e2" else x["f2"]["f1"]
        stored = (sec.get(prog, {}).get("paired") or {}).get(f"{'e2' if est == 'e2' else 'twin_128'}_minus_{other}")
        d = bench.auroc_or_none(y[ok], p1[ok], w[ok]) - bench.auroc_or_none(y[ok], p2[ok], w[ok])
        ci = bench.bootstrap(_paired_slow, stay[ok], y[ok], p1[ok], p2[ok], w[ok], n=B_CI, strata=ds[ok])
        rows.append({"family": fam, "design": "L48x", "program": prog, "endpoint": NAMES.get(prog, prog),
                     "comparison": f"{est} minus {other}", "estimate_kind": "hazard (E2)" if est == "e2" else "sampled",
                     "estimate": float(d), "ci": list(ci), "n": int(ok.sum()), "positives": int(y[ok].sum()),
                     "reproduced": bool(stored and close(d, stored["diff"]) and close(ci[0], stored["ci"][0])
                                        and close(ci[1], stored["ci"][1])),
                     "_l48x": (stay[ok], y[ok], p1[ok], p2[ok], w[ok], ds[ok])})
    return rows


def knockout_rows(root: Path) -> list:
    """Family 20 (16c): per anchor, rho(knockout, generator) minus rho(editing, generator)
    (``runs/paper/f4r/knockout_vs_edit.parquet``); two-sided Wilcoxon signed-rank, pooled and per endpoint."""
    import pandas as pd
    from scipy.stats import wilcoxon

    k = pd.read_parquet(root / "f4r/knockout_vs_edit.parquet")
    k = k[np.isfinite(k["rho_knockout"]) & np.isfinite(k["rho_edit"])]
    rows = []
    for prog, g in [("pooled", k), *k.groupby("program")]:
        dif = (g["rho_knockout"] - g["rho_edit"]).to_numpy()
        nz = dif[dif != 0]
        p = float(wilcoxon(nz, alternative="two-sided").pvalue) if len(nz) else None
        rows.append({"family": 20, "program": prog, "endpoint": NAMES.get(prog, prog), "anchors": int(len(g)),
                     "knockout_median": float(np.median(g["rho_knockout"])), "edit_median": float(np.median(g["rho_edit"])),
                     "difference_median": float(np.median(dif)), "wins": int((dif > 0).sum()),
                     "ties": int((dif == 0).sum()), "losses": int((dif < 0).sum()), "p": p})
    return rows


def amendment16(root: Path, clif: Path, out: Path, tables: str | None) -> None:
    """Amendment 16's families 15-20, added to an existing ``stats.json`` with their own git sha and inputs: every
    stored interval reproduced first (the script stops before any P value if one differs), then exact paired-bootstrap
    P values (10,000 resamples, seed 1, strata = database) and Holm within each family (family 20: across the 4
    endpoints, the pooled test beside them)."""
    from icutwin.eval import bench

    t0 = time.time()
    git = _sha()
    fams = {f: ext_rows(root, clif, f) for f in (15, 16, 17)}
    lx = l48x_inputs(root) if (root / "f1/L48x/estimates.parquet").exists() else None
    for f in (18, 19):
        fams[f] = l48x_rows(lx, f) if lx else []
    rows = [r for f in fams.values() for r in f]
    bad = [r for r in rows if not r["reproduced"]]
    print(json.dumps({"reproductions": len(rows), "failed": len(bad)}), flush=True)
    if bad:
        for r in bad:
            print(json.dumps({k: r.get(k) for k in ("family", "design", "dataset", "program", "comparison")}))
        raise NotReproduced("stored intervals not reproduced; no P value computed")
    for r in rows:
        if "_inp" in r:
            draws = comparator_draws(r["_inp"], r["comparator"], B_P, SEED_P)
        else:
            stay, y, p1, p2, w, ds = r["_l48x"]
            draws = bench.bootstrap_draws(_paired_fast, stay, y, p1, p2, w, n=B_P, seed=SEED_P, strata=ds)
        r["p"], r["draws"] = p_boot(draws, r["estimate"])
        r.pop("_inp", None)
        r.pop("_l48x", None)
    for f in fams.values():
        assign_holm(f)
    fams[20] = knockout_rows(root) if (root / "f4r/knockout_vs_edit.parquet").exists() else []
    assign_holm([r for r in fams[20] if r["program"] != "pooled"])
    f = out / "stats.json"
    rep = json.loads(f.read_text(encoding="utf-8"))
    rep["families"].update({str(k): v for k, v in fams.items()})
    files = [root / "ext/ext.json", root / "ext/gate.json", root / "f4r/knockout_vs_edit.parquet",
             *(lx["files"] if lx else [])]
    for design in ("R", "L24e"):
        files += sorted((root / "ext" / design).glob("*.parquet"))
    rep["amendment16"] = {"git": git, "reproductions": len(rows), "resamples_p": B_P, "seed": SEED_P,
                          "inputs": {str(p): sha256(p) for p in files if p.exists()},
                          "seconds": round(time.time() - t0, 1)}
    f.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8", newline="\n")
    if tables:
        write_amendment16_tables(rep, Path(tables))
    print(json.dumps({k: [(r.get("dataset", ""), r["program"], fmt(r.get("estimate")), fmt_p(r.get("p_holm")))
                          for r in v] for k, v in fams.items() if k != 20}, default=str)[:3000], flush=True)


def write_amendment16_tables(rep: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def ps(r) -> list:
        return [fmt_p(r.get("p"), r.get("p_bound")), fmt_p(r.get("p_holm"), r.get("p_holm_bound"))]

    fams = rep["families"]
    for fam in ("15", "16", "17", "18", "19"):
        with open(out / f"stats_family{fam}.csv", "w", newline="\n", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["design", "database", "program", "endpoint", "comparison", "estimate", "events/n",
                        "estimate (95% CI)", "P", "P (Holm)"])
            for r in fams.get(fam, []):
                ci = r.get("ci") or [None, None]
                w.writerow([r["design"], r.get("dataset", "pooled seven"), r["program"], r["endpoint"],
                            r["comparison"], r.get("estimate_kind"), f"{r['positives']}/{r['n']}",
                            f"{fmt(r['estimate'])} ({fmt(ci[0])} to {fmt(ci[1])})", *ps(r)])
    with open(out / "stats_family20.csv", "w", newline="\n", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["program", "endpoint", "anchors", "rho knockout (median)", "rho editing (median)",
                    "difference (median)", "knockout higher/ties/lower", "P (Wilcoxon)", "P (Holm)"])
        for r in fams.get("20", []):
            w.writerow([r["program"], r["endpoint"], r["anchors"], fmt(r["knockout_median"]), fmt(r["edit_median"]),
                        fmt(r["difference_median"]), f"{r['wins']}/{r['ties']}/{r['losses']}", fmt_p(r.get("p")),
                        fmt_p(r.get("p_holm")) if r["program"] != "pooled" else "-"])


FAM17 = {21: (("Z1", "Z0"), ("F1", "Z0"), ("F1R", "Z0")), 22: (("F1R", "lgbm_ftlabel"),),
         23: (("F1R", "tabpfn_v3_5"),)}


def ext17_rows(root: Path, fam: int) -> list:
    """Families 21-23 (amendment 17; ``runs/paper/ext17/{ext17.json,scores.parquet}``, ``tools/paper_ext17.py
    analyze``) on the TEST anchors, per scored endpoint and database with at least 10 TEST positives: each arm minus
    Z0 (21), F1+R minus the local LightGBM on every adaptation label (22) and minus TabPFN-3.5 transferred (23). The
    stored 1,000-resample interval is reproduced first; the P value comes later, from 10,000 resamples."""
    import pandas as pd

    ext = json.loads((root / "ext17/ext17.json").read_text(encoding="utf-8"))
    sc = pd.read_parquet(root / "ext17/scores.parquet")
    rows = []
    for (design, d, prog), x in sc.groupby(["design", "dataset", "program"], sort=True):
        r = ext["designs"][design][d][prog]
        if r["descriptive_only"]:
            continue
        x = x.reset_index(drop=True)
        for ref, comp in FAM17[fam]:
            stored = r["predictors"].get(comp) if ref == "F1R" else r["arms_vs_z0"].get(ref)
            if not stored or "ref_minus" not in stored:
                continue
            inp = {"y": x["y"].to_numpy(float), "w": x["w"].to_numpy(float), "stay": x["stay"].astype(str).to_numpy(),
                   "ds": np.full(len(x), d), "ref": x[ref].to_numpy(float), "scores": {comp: x[comp].to_numpy(float)}}
            both = np.isfinite(inp["ref"]) & np.isfinite(inp["scores"][comp])
            ci = [float(v) for v in np.nanpercentile(comparator_draws(inp, comp, B_CI, 0), [2.5, 97.5])]
            rows.append({"family": fam, "design": design, "dataset": d, "program": prog,
                         "endpoint": NAMES.get(prog, prog), "comparison": f"{ref} minus {comp}", "arm": ref,
                         "comparator": comp, "estimate": stored["ref_minus"], "ci": stored["ref_minus_ci"],
                         "n": int(both.sum()), "positives": int(inp["y"][both].sum()),
                         "reproduced": bool(close(ci[0], stored["ref_minus_ci"][0])
                                            and close(ci[1], stored["ref_minus_ci"][1])),
                         "_inp": inp})
    return rows


def amendment17(root: Path, out: Path, tables: str | None) -> None:
    """Amendment 17's families 21-24, added to an existing ``stats.json`` with their own git sha and inputs: every
    stored interval reproduced first (the script stops before any P value if one differs), then exact paired-bootstrap
    P values (10,000 resamples, seed 1, strata = database) and Holm within each arm (21) and within 22 and 23; family
    24 is ``ext17.json``'s rank summary (``tools/paper_ext17.py analyze``)."""
    t0 = time.time()
    git = _sha()
    fams = {f: ext17_rows(root, f) for f in (21, 22, 23)}
    rows = [r for f in fams.values() for r in f]
    bad = [r for r in rows if not r["reproduced"]]
    print(json.dumps({"reproductions": len(rows), "failed": len(bad)}), flush=True)
    if bad:
        for r in bad:
            print(json.dumps({k: r.get(k) for k in ("family", "design", "dataset", "program", "comparison")}))
        raise NotReproduced("stored intervals not reproduced; no P value computed")
    for r in rows:
        draws = comparator_draws(r.pop("_inp"), r["comparator"], B_P, SEED_P)
        r["p"], r["draws"] = p_boot(draws, r["estimate"])
    for arm in ("Z1", "F1", "F1R"):
        assign_holm([r for r in fams[21] if r["arm"] == arm])
    for f in (22, 23):
        assign_holm(fams[f])
    ext = json.loads((root / "ext17/ext17.json").read_text(encoding="utf-8"))
    fams[24] = ext.get("ranks", {})
    f = out / "stats.json"
    rep = json.loads(f.read_text(encoding="utf-8"))
    rep["families"].update({str(k): v for k, v in fams.items()})
    files = [root / "ext17/ext17.json", root / "ext17/scores.parquet", root / "ext17/split.json"]
    rep["amendment17"] = {"git": git, "reproductions": len(rows), "resamples_p": B_P, "seed": SEED_P,
                          "split_sha256": ext.get("split_sha256"),
                          "inputs": {str(p): sha256(p) for p in files if p.exists()},
                          "seconds": round(time.time() - t0, 1)}
    f.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8", newline="\n")
    if tables:
        write_amendment17_tables(rep, Path(tables))
    print(json.dumps({k: [(r["dataset"], r["program"], r["arm"], fmt(r.get("estimate")), fmt_p(r.get("p_holm")))
                          for r in v] for k, v in fams.items() if k != 24}, default=str)[:3000], flush=True)


def write_amendment17_tables(rep: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fams = rep["families"]
    with open(out / "stats_family21-23.csv", "w", newline="\n", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["family", "design", "database", "program", "endpoint", "comparison", "events/n",
                    "estimate (95% CI)", "P", "P (Holm)"])
        for fam in ("21", "22", "23"):
            for r in fams.get(fam, []):
                ci = r.get("ci") or [None, None]
                w.writerow([fam, r["design"], r["dataset"], r["program"], r["endpoint"], r["comparison"],
                            f"{r['positives']}/{r['n']}", f"{fmt(r['estimate'])} ({fmt(ci[0])} to {fmt(ci[1])})",
                            fmt_p(r.get("p"), r.get("p_bound")), fmt_p(r.get("p_holm"), r.get("p_holm_bound"))])
    with open(out / "stats_family24.csv", "w", newline="\n", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["set", "predictor", "mean rank", "95% CI", "firsts", "mean AUROC", "tasks"])
        for key in ("all", "without_local"):
            s = (fams.get("24") or {}).get(key)
            if not s:
                continue
            for x in s["predictors"]:
                ci = x.get("mean_rank_ci") or [None, None]
                w.writerow([key, x["predictor"], fmt(x["mean_rank"], 2), f"{fmt(ci[0], 2)} to {fmt(ci[1], 2)}",
                            x["firsts"], fmt(x["mean_auroc"]), s["tasks"]])


# ---- output ------------------------------------------------------------------------------------------------------

def fmt_p(p, bound: bool = False) -> str:
    if p is None:
        return "-"
    if p < 0.001:
        return "<0.001"
    if bound:                                    # an upper bound: two significant figures, rounded up
        e = int(np.floor(np.log10(p))) - 1
        return "<=" + f"{np.ceil(p / 10 ** e - 1e-9) * 10 ** e:.{max(0, -e)}f}"
    return f"{p:.2f}" if p >= 0.1 else f"{float(f'{p:.2g}'):g}"


def fmt(x, d=3) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def write_tables(rep: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def write(name: str, head: list, rows: list) -> None:
        with open(out / f"stats_{name}.csv", "w", newline="\n", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(head)
            w.writerows(rows)

    def with_ci(est, ci, d=3) -> str:
        ci = ci or [None, None]
        return f"{fmt(est, d)} ({fmt(ci[0], d)} to {fmt(ci[1], d)})"

    def ps(r) -> list:
        return [fmt_p(r.get("p"), r.get("p_bound")), fmt_p(r.get("p_holm"), r.get("p_holm_bound"))]

    fams = rep["families"]
    for fam in ("1", "2", "3", "4"):
        write(f"family{fam}", ["subset", "program", "endpoint", "comparison", "estimate", "events/n",
                               "estimate (95% CI)", "P", "P (Holm)", "method"],
              [[r["subset"], r["program"], r["endpoint"], r["comparison"], r.get("estimate_kind"),
                f"{r.get('positives')}/{r.get('n')}", with_ci(r.get("estimate"), r.get("ci")), *ps(r),
                r.get("method")] for r in fams.get(fam, [])])
    write("family5", ["program", "endpoint", "map", "statistic", "estimate (95% CI)", "P", "P (Holm)"],
          [[r["program"], r["endpoint"], r["map"], r["comparison"], with_ci(r["estimate"], r.get("ci")), *ps(r)]
           for r in fams.get("5", [])])
    write("family6", ["treatment", "response", "observational (95% CI)", "Didymos (95% CI)", "difference (95% CI)",
                      "P", "P (Holm)"],
          [[r["treatment"], r["response"], with_ci(r["observational"], r["observational_ci"], 2),
            with_ci(r["didymos"], r["didymos_ci"], 2), with_ci(r["estimate"], r["ci"], 2), *ps(r)]
           for r in fams.get("6", [])])
    write("family7", ["treatment", "response", "expected-sign share (95% CI)", "k/n", "P", "P (Holm)"],
          [[r["treatment"], r["response"], with_ci(r["estimate"], r["ci"], 2), f"{r['k']}/{r['n']}", *ps(r)]
           for r in fams.get("7", [])])
    write("family8", ["dataset", "checkpoints", "Spearman r", "P", "P (Holm)"],
          [[r["dataset"], r["n"], fmt(r["estimate"], 2), *ps(r)] for r in fams.get("8", [])])
    write("family9", ["design", "cell", "cells", "positive", "mean gain", "P (sign)", "P (sign, Holm)",
                      "P (Wilcoxon)", "P (Wilcoxon, Holm)"],
          [[r["design"], r["cell"], r["cells"], r["positive"], fmt(r["estimate"]), fmt_p(r.get("p_sign")),
            fmt_p(r.get("p_sign_holm")), fmt_p(r.get("p_wilcoxon")), fmt_p(r.get("p_wilcoxon_holm"))]
           for r in fams.get("9", [])])
    write("family10", ["program", "endpoint", "anchors", "median difference", "P", "P (Holm, per endpoint)"],
          [[r["program"], r["endpoint"], r["n"], fmt(r["estimate"], 2), *ps(r)] for r in fams.get("10", [])])
    write("trials", ["cell", "score", "sample-size ratio (Monte Carlo 95% interval)",
                     "type I error, unadjusted (95% CI)", "type I error, adjusted (95% CI)"],
          [[r["cell"], r["score"], with_ci(r["n_ratio"], r["n_ratio_ci"], 2),
            with_ci(r["type1_unadjusted"]["estimate"], r["type1_unadjusted"]["ci"]),
            with_ci(r["type1_adjusted"]["estimate"], r["type1_adjusted"]["ci"])] for r in rep.get("trials", [])])
    write("approximation", ["subset", "program", "comparator", "P exact", "P from the interval"],
          [[r["subset"], r["program"], r["comparator"], fmt_p(r["p_exact"]), fmt_p(r["p_from_ci"])]
           for r in rep.get("approximation_check", {}).get("rows", [])])


def write_amendment14_tables(rep: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def write(name: str, head: list, rows: list) -> None:
        with open(out / f"stats_{name}.csv", "w", newline="\n", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(head)
            w.writerows(rows)

    write("family11", ["design", "program", "endpoint", "catalog family", "events/n", "Didymos minus LightGBM (95% CI)",
                       "P", "q (Benjamini-Hochberg)", "P (Holm)"],
          [[r["design"], r["program"], r["endpoint"], r["catalog_family"], f"{r['positives']}/{r['n']}",
            f"{fmt(r['estimate'])} ({fmt(r['ci'][0])} to {fmt(r['ci'][1])})", fmt_p(r["p"], r["p_bound"]),
            fmt_p(r["q_bh"], r["q_bh_bound"]), fmt_p(r["p_holm"], r["p_holm_bound"])]
           for r in rep["families"]["11"]])
    sizes = ("100", "300", "1000", "3000")
    write("label_efficiency", ["design", "program", "endpoint", "dataset", "Didymos AUROC", *[f"LightGBM, n={n}"
                                                                                           for n in sizes],
                               "LightGBM, all", "all (n)", "n to match"],
          [[r["design"], r["program"], r["endpoint"], r["dataset"], fmt(r["didymos"]),
            *[fmt(r["curve"].get(n)) for n in sizes], fmt(r["curve"][str(r["max_n"])]), r["max_n"],
            {"not reached": "not reached", "at most": f"at most {fmt(r['n_match'], 0)}"}.get(
                r["status"], fmt(r["n_match"], 0))] for r in rep["label_efficiency"]])


def amendment14(root: Path, out: Path, tables: str | None, workers: int) -> None:
    """Amendment 14's statistics, added to an existing ``stats.json`` (families 1-10 and the trials are left as they
    are), with their own git sha, inputs and timing."""
    t0 = time.time()
    git = _sha()
    f = out / "stats.json"
    rep = json.loads(f.read_text(encoding="utf-8"))
    rows, summary, files = breadth(root, workers)
    rep["families"]["11"] = rows
    rep["breadth"] = summary
    rep["label_efficiency"], rep["label_efficiency_summary"] = label_efficiency(root)
    files += [str(root / f"a1/test_{d}/{n}") for d in ("R", "L24") for n in ("baselines.json", "a1.json")]
    rep["amendment14"] = {"git": git, "inputs": {p: sha256(Path(p)) for p in sorted(set(files)) if Path(p).exists()},
                          "resamples_p": B_P, "seed_p": SEED_P, "resamples_joint": B_JOINT, "seed_joint": SEED_JOINT,
                          "seconds": round(time.time() - t0, 1)}
    f.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8", newline="\n")
    if tables:
        write_amendment14_tables(rep, Path(tables))
    print(json.dumps({"seconds": rep["amendment14"]["seconds"], "breadth": summary["groups"]["all"],
                      "label_efficiency": rep["label_efficiency_summary"]}), flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/paper")
    ap.add_argument("--toolkit", default="runs/toolkit")
    ap.add_argument("--out", default="runs/paper/stats")
    ap.add_argument("--tables", default=None)
    ap.add_argument("--check-only", action="store_true", help="reproduce the stored intervals and stop")
    ap.add_argument("--amendment14", action="store_true",
                    help="add amendment 14's family 11 and label efficiency to an existing stats.json")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--amendment15", action="store_true",
                    help="add amendment 15's rank summaries (items 13 and 14) to an existing stats.json")
    ap.add_argument("--amendment16", action="store_true",
                    help="add amendment 16's families 15-20 to an existing stats.json")
    ap.add_argument("--clif", default="data/clif_canonical/full", help="amendment 16a: the external CLIF (patients)")
    ap.add_argument("--amendment17", action="store_true",
                    help="add amendment 17's families 21-24 to an existing stats.json")
    a = ap.parse_args(argv)
    root, toolkit = Path(a.root), Path(a.toolkit)
    if a.amendment17:
        amendment17(root, Path(a.out), a.tables)
        return
    if a.amendment16:
        amendment16(root, Path(a.clif), Path(a.out), a.tables)
        return
    if a.amendment14:
        amendment14(root, Path(a.out), a.tables, a.workers)
        return
    if a.amendment15:
        amendment15(root, Path(a.out), a.tables)
        return
    t0 = time.time()
    git = _sha()                                   # at the start: the code that runs, whatever is committed later
    cache: dict = {}

    # 1. reproductions: every stored interval this script recomputes, before any P value
    checks = {"headline": headline(root, "check", cache), "clinical": clinical(root, "check"),
              "calibration": calibration(root, toolkit, "check")}
    bad = [r for fam in checks["headline"].values() for r in fam if r.get("reproduced") is False]
    bad += [r for r in checks["clinical"] + checks["calibration"] if r.get("reproduced") is False]
    n_checked = sum(1 for fam in checks["headline"].values() for r in fam if "reproduced" in r) + \
        sum(1 for r in checks["clinical"] + checks["calibration"] if "reproduced" in r)
    print(json.dumps({"reproductions": n_checked, "failed": len(bad),
                      "seconds": round(time.time() - t0, 1)}), flush=True)
    if bad:
        for r in bad:
            print(json.dumps({k: r.get(k) for k in ("family", "subset", "program", "comparator", "comparison")}),
                  flush=True)
        raise NotReproduced("stored intervals not reproduced; no P value computed")
    if a.check_only:
        return

    # 2. the P values
    fam = headline(root, "p", cache)
    rep = {"git": git, "amendment": 13, "seed": SEED_P, "resamples_p": B_P, "resamples_ci": B_CI,
           "reproductions": n_checked, "families": {str(k): v for k, v in fam.items()}}
    rep["families"]["4"] = clinical(root, "p")
    rep["families"]["5"] = calibration(root, toolkit, "p")
    rep["families"]["6"], rep["families"]["7"] = whatif(root)
    rep["families"]["8"] = leakage(root)
    rep["families"]["9"] = augmentation(root)
    rep["families"]["10"] = attribution(root)
    rep["trials"] = trials(toolkit)
    # Holm within each family (family 5 within each statistic; family 9 within each test; family 10 per endpoint);
    # a bootstrap P at its floor, and any adjusted value it sets, is marked as an upper bound
    for k in ("1", "2", "3", "4", "6", "7", "8"):
        assign_holm(rep["families"][k])
    for stat in ("oe_raw", "oe_recalibrated", "oe_change", "ece_change"):
        assign_holm([r for r in rep["families"]["5"] if r["comparison"] == stat])
    for test in ("p_sign", "p_wilcoxon"):
        assign_holm(rep["families"]["9"], key=test, out=f"{test}_holm")
    assign_holm([r for r in rep["families"]["10"] if r["program"] != "pooled"])
    rep["approximation_check"] = approximation_check(root, cache)
    inputs = sorted({str(p) for x in cache.values() for p in x["files"]} |
                    {str(root / f) for f in ("f10/comparators.json", "f10/clinical/scores.parquet", "f5/c1b.json",
                                             "a3/a3.json", "f8/aug_L24.json", "f8/aug_R.json",
                                             "f4/attn_diag/per_anchor.jsonl")} |
                    {str(toolkit / f) for f in ("v1/forecast/e2.parquet", "v1/forecast/predictions.parquet",
                                                "recalibration/MUC32_main.json", "v2/v2.json", "v2/v2b.json",
                                                "v4/v4.json")})
    rep["inputs"] = {p: sha256(Path(p)) for p in inputs if Path(p).exists()}
    rep["seconds"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8", newline="\n")
    if a.tables:
        write_tables(rep, Path(a.tables))
    print(json.dumps({"seconds": rep["seconds"], "approximation": {k: v for k, v in rep["approximation_check"].items()
                                                                  if k != "rows"}}), flush=True)


if __name__ == "__main__":
    main()
