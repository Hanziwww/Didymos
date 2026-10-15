"""Paper P amendment 10 (``docs/p1_paper_protocol.md``): every comparator against the Twin, on A1's held-out anchors
and on F1's subsets.

Predictors of a design (``--a1``: ``runs/paper/a1/test_<design>``):
* the Twin: A1's (32 rollouts, ``predictions.parquet``); on a subset, the 128-rollout share (``<subset>/
  predictions.parquet``), with E2's AUROC beside it (``<subset>/f2.json``; its per-anchor values were not kept);
* A1's comparators: LightGBM pooled and per dataset, L2 logistic regression (``baseline_predictions.parquet``),
  TabPFN-3.5 and v2, the Twin probe;
* amendment 10's learned baselines (``--ml``: ``ml_<design>*/ml_predictions.parquet``) and clinical scores
  (``--clinical``: ``scores.parquet``), each score on the endpoints the amendment assigns it (``scores_for``).

Per design or subset and endpoint, on the anchors with the Twin's label (as A1's comparisons; on a subset, also with
a label in A1's predictions, as F1 and F2, so E2's AUROC is on the same anchors):
* each predictor's AUROC (weighted as A1), with a 95% interval by a bootstrap over stays within datasets (``--boot``
  resamples shared by all predictors), and its AUPRC; for probabilities also the Brier score, O/E and calibration
  slope; the AUROC per dataset;
* the paired difference, the Twin minus the predictor, on the anchors both scored, from the same resamples.

The datasets' own scores are read on their datasets' anchors only, with the Twin's and LightGBM's AUROC on the same
anchors. Writes ``<out>/comparators.json``.

    python tools/paper_comparators.py --a1 runs/paper/a1 --subsets runs/paper/f1/R128 runs/paper/f1/L24x \\
        --ml runs/paper/f10 --clinical runs/paper/f10/clinical/scores.parquet --out runs/paper/f10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import PRIMARY, _sha  # noqa: E402

ENDPOINTS = {"L24": (*PRIMARY["L24"], "decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440",
                     "rrt_onset@2880"),
             "L48": PRIMARY["L48"], "R": PRIMARY["R"]}
SUBSET_DESIGN = {"R128": "R", "L24x": "L24", "L48x": "L48"}
# amendment 16b: L48x adds death within 24 h at the landmark and the three treatments (secondary)
SUBSET_ENDPOINTS = {"L48x": (*PRIMARY["L48"], "decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440",
                             "rrt_onset@2880")}
SCREENS = ("news", "mews", "qsofa", "sirs")
INDICATORS = {"decompensation@1440": ("shock_index", "lactate"), "imv_onset@1440": ("neg_rox",),
              "vasopressor_onset@1440": ("shock_index", "neg_map"), "rrt_onset@2880": ("kdigo", "creatinine"),
              "aki_2@2880": ("kdigo", "creatinine"), "hypotension@240": ("neg_map",),
              "hyperkalemia_top@1440": ("potassium",), "thrombocytopenia@2880": ("neg_platelets",),
              "sepsis_ase@1440": ("lactate",)}
NATIVE = {"icu_death@-1": ("apache_icu_mort", "saps3_prob", "asa"),
          "hospital_death@0": ("apache_hosp_mort", "saps3_prob", "asa"),
          "remaining_los_gt_7d": ("apache_icu_los", "saps3_prob", "asa"),
          "decompensation@1440": ("apache_icu_mort", "saps3_prob", "asa")}
NATIVE_DATASET = {"apache_icu_mort": "eicu", "apache_hosp_mort": "eicu", "apache_icu_los": "eicu",
                  "saps3_prob": "sicdb", "asa": "inspire"}
LEARNED = ("lgbm_pooled", "lgbm_per_dataset", "logistic_pooled", "tabpfn_v3_5", "tabpfn_v2", "twin_probe",
           "logistic_l1", "logistic_enet", "cart", "random_forest", "extra_trees", "naive_bayes", "knn", "mlp")
PROBABILITY = (*LEARNED, "sapsii_prob", "oasis_prob", "apache_icu_mort", "apache_hosp_mort", "saps3_prob")


def scores_for(design: str, prog: str) -> list[str]:
    """The clinical scores the amendment assigns an endpoint: SOFA and the screens everywhere, SAPS II and OASIS at
    the landmarks, the endpoint's bedside indicators."""
    out = ["sofa", *SCREENS]
    if design in ("L24", "L48"):
        out += ["sapsii_prob", "oasis_prob"]
    return out + list(INDICATORS.get(prog, ()))


def evaluate(y, w, stay, ds, preds: dict, ref: str, boot: int) -> dict:
    """Metrics of every predictor (``preds``: name -> probabilities or scores, NaN where it has none) and its paired
    difference from ``ref``, from one set of resamples."""
    from sklearn.metrics import average_precision_score

    from icutwin.eval import bench

    idx = bench.resamples(stay, n=boot, seed=0, strata=ds)
    out = {}
    r = preds[ref]
    r_ok = np.isfinite(r)
    r_full = [bench.auroc_fast(y[ix], r[ix], w[ix]) if r_ok.all() else np.nan for ix in idx]
    for name, p in preds.items():
        m = np.isfinite(p)
        if m.sum() < 30 or len(np.unique(y[m])) < 2:
            continue
        res = {"n": int(m.sum()), "positives": int(y[m].sum()), "auroc": bench.auroc_fast(y[m], p[m], w[m]),
               "auprc": float(average_precision_score(y[m], p[m], sample_weight=w[m]))}
        if name in PROBABILITY or name.startswith("twin"):
            mm = bench.binary_metrics(y[m], np.clip(p[m], 0, 1), w[m])
            res.update({k: mm.get(k) for k in ("brier", "observed_over_expected", "calibration_slope")})
        both = m & r_ok
        au, diff = [], []
        for ix, rf in zip(idx, r_full, strict=True):
            k = ix[m[ix]]
            au.append(bench.auroc_fast(y[k], p[k], w[k]))
            if name != ref:
                if both.all():                          # the reference's AUROC on the full resample, computed once
                    diff.append(rf - au[-1])
                else:
                    kb = ix[both[ix]]
                    diff.append(bench.auroc_fast(y[kb], r[kb], w[kb]) - bench.auroc_fast(y[kb], p[kb], w[kb]))
        au = np.asarray(au)
        res["auroc_ci"] = [float(x) for x in np.nanpercentile(au, [2.5, 97.5])]
        if name != ref:
            d = np.asarray(diff)
            res["ref_minus"] = bench.auroc_fast(y[both], r[both], w[both]) - bench.auroc_fast(y[both], p[both], w[both])
            res["ref_minus_ci"] = [float(x) for x in np.nanpercentile(d, [2.5, 97.5])]
        res["datasets"] = {}
        for d_ in np.unique(ds[m]):
            md = m & (ds == d_)
            if len(np.unique(y[md])) == 2:
                res["datasets"][str(d_)] = {"n": int(md.sum()), "positives": int(y[md].sum()),
                                            "auroc": bench.auroc_fast(y[md], p[md], w[md])}
        out[name] = res
    return out


def native(y, w, stay, ds, preds: dict, prog: str, ref: str, boot: int) -> dict:
    """The datasets' own scores on their datasets' anchors, with ``ref`` and LightGBM on the same anchors."""
    out = {}
    for name in NATIVE.get(prog, ()):
        if name not in preds:
            continue
        m = np.isfinite(preds[name]) & (ds == NATIVE_DATASET[name]) & np.isfinite(preds[ref])
        if m.sum() < 30 or len(np.unique(y[m])) < 2:
            continue
        sub = {k: preds[k][m] for k in (name, ref, "lgbm_pooled") if k in preds}
        res = evaluate(y[m], w[m], stay[m], ds[m], sub, ref, boot)
        out[name] = {"dataset": NATIVE_DATASET[name], **res}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("a1", "ml", "clinical", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--subsets", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--only-subsets", action="store_true",
                    help="amendment 16b: the subsets alone (comparators_<subset>.json), the designs not rerun")
    ap.add_argument("--extra", nargs="*", default=[], help="more prediction files (anchor, program, model, p)")
    a = ap.parse_args(argv)
    import pandas as pd

    t0 = time.time()
    clin = pd.read_parquet(a.clinical)
    rep = {"git": _sha(), "boot": a.boot, "designs": {}, "subsets": {}}

    def comparators(design: str) -> pd.DataFrame:
        dd = Path(a.a1) / f"test_{design}"
        parts = [pd.read_parquet(f)[["anchor", "program", "model", "p"]] for f in sorted(dd.glob("*_predictions.parquet"))]
        for ml in sorted(Path(a.ml).glob(f"ml_{design}*/ml_predictions.parquet")):   # one directory per lane
            parts.append(pd.read_parquet(ml)[["anchor", "program", "model", "p"]])
        for f in a.extra:                                  # amendment 16b: predictions fitted later, kept apart
            x = pd.read_parquet(f)[["anchor", "program", "model", "p"]]
            have = pd.concat(parts).set_index(["anchor", "program", "model"]).index
            parts.append(x[~x.set_index(["anchor", "program", "model"]).index.isin(have)])
        return pd.concat(parts, ignore_index=True)

    def run(twin_file: Path, design: str, ref: str, e2: dict | None = None, subset: bool = False) -> dict:
        tw = pd.read_parquet(twin_file)
        comp = comparators(design)
        sc = clin[clin["design"] == design].drop_duplicates("anchor").set_index("anchor")
        a1_tw = pd.read_parquet(Path(a.a1) / f"test_{design}" / "predictions.parquet") if subset else None
        res = {}
        for prog in SUBSET_ENDPOINTS.get(twin_file.parent.name, ENDPOINTS[design]) if subset else ENDPOINTS[design]:
            t = tw[tw["program"] == prog].drop_duplicates("anchor").set_index("anchor")
            if subset:                                   # F1/F2's anchors: those with a label in A1's predictions
                t = t[t.index.isin(set(a1_tw.loc[a1_tw["program"] == prog, "anchor"]))]
            if len(t) < 30 or t["label"].nunique() < 2:
                continue
            y, w = t["label"].to_numpy(dtype=np.float64), t["weight"].to_numpy(dtype=np.float64)
            stay, ds = t["stay"].astype(str).to_numpy(), t["dataset"].astype(str).to_numpy()
            preds = {ref: t["p"].to_numpy(dtype=np.float64)}
            c = comp[comp["program"] == prog]
            for model, g in c.groupby("model"):
                preds[model] = g.drop_duplicates("anchor").set_index("anchor")["p"].reindex(t.index).to_numpy(
                    dtype=np.float64)
            s = sc.reindex(t.index)
            for col in scores_for(design, prog) + list(NATIVE.get(prog, ()) if design in ("L24", "L48") else ()):
                if col in s:
                    preds[col] = s[col].to_numpy(dtype=np.float64)
            pooled = {k: v for k, v in preds.items() if k not in NATIVE_DATASET}
            res[prog] = {"anchors": int(len(t)), "positives": int(y.sum()),
                         "models": evaluate(y, w, stay, ds, pooled, ref, a.boot),
                         "native": native(y, w, stay, ds, preds, prog, ref, a.boot)
                         if design in ("L24", "L48") and not subset else {}}
            if e2 and prog in e2:
                res[prog]["e2_auroc"] = e2[prog]
            print(json.dumps({"design": design, "ref": ref, "program": prog, "anchors": len(t),
                              "seconds": round(time.time() - t0, 1)}), flush=True)
        return res

    for design in () if a.only_subsets else ENDPOINTS:
        rep["designs"][design] = run(Path(a.a1) / f"test_{design}" / "predictions.parquet", design, "twin")
    for sd in map(Path, a.subsets):
        name = sd.name
        f2 = json.loads((sd / "f2.json").read_text()) if (sd / "f2.json").exists() else {}
        e2 = {p: x["estimators"]["e2"]["auroc"] for p, x in f2.get("f2", {}).items() if "estimators" in x}
        rep["subsets"][name] = run(sd / "predictions.parquet", SUBSET_DESIGN[name], "twin_128", e2, subset=True)
    rep["seconds"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "comparators.json" if not a.only_subsets else \
        "comparators_" + "_".join(Path(x).name for x in a.subsets) + ".json"
    (out / name).write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8", newline="\n")
    print(json.dumps({"seconds": rep["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
