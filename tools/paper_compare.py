"""Paper P: the comparisons across A1's predictors and the analyses that read their per-anchor predictions
(``docs/p1_paper_protocol.md`` A1, A2, A5, B2), for one held-out design directory:

* A1: per program, every predictor's AUROC on the anchors all of them scored (the Twin's rollouts, the Twin probe,
  LightGBM pooled and per dataset, logistic regression), and the paired bootstrap difference Twin minus LightGBM
  pooled (stays resampled within datasets);
* A2: the Twin's isotonic recalibration fitted on the tuning-split dry run of the same design (``--dev``) and applied
  here: calibration before and after; O/E by ICU day of the anchor;
* A5: the Twin's and LightGBM's AUROC, O/E and calibration slope by sex, age band, dataset, admission type and race
  (MIMIC, eICU, NWICU), primary endpoints, subgroups with at least 30 positives;
* B2 (``--unknown``, the unknown-site run of the same anchors): the Twin's AUROC and O/E with the site known and
  unknown, on the shared anchors.

Writes ``<out>/compare.json`` (aggregates).

    python tools/paper_compare.py --design-dir runs/paper/a1/test_L24 --dev runs/paper/a1/dev_L24 \\
        --store data/factory/stores/real_test --unknown runs/paper/a1/b2_L24_unknown --out runs/paper/a1/test_L24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import PRIMARY, _sha  # noqa: E402

AGE_BANDS = (18, 45, 65, 80, 200)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("design-dir", "store", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--unknown", default=None)
    ap.add_argument("--tokens", default="data/tokens/full_r4")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(argv)
    import pandas as pd
    from sklearn.isotonic import IsotonicRegression

    from icutwin.eval import bench
    from icutwin.factory import store as fst

    dd = Path(a.design_dir)
    design = json.loads((dd / "design.json").read_text())["design"]
    tw = pd.read_parquet(dd / "predictions.parquet")
    tw["model"] = "twin"
    parts = [tw[["anchor", "dataset", "program", "model", "p", "label", "weight", "stay"]]]
    for f in sorted(dd.glob("*_predictions.parquet")):     # baselines, the probe, TabPFN (v2, 3.5)
        parts.append(pd.read_parquet(f))
    lab = tw[["anchor", "program", "label", "weight", "stay"]].drop_duplicates(["anchor", "program"])
    allp = pd.concat([p.drop(columns=[c for c in ("label", "weight", "stay") if c in p], errors="ignore")
                      for p in parts]).merge(lab, on=["anchor", "program"])
    an = pd.read_parquet(dd / "anchors.parquet").set_index("anchor")
    allp["day"] = np.minimum(an.loc[allp["anchor"], "t_min"].to_numpy() // 1440 + 1, 4)
    # subgroup attributes from the store and the stays' prefix (race)
    st = fst.Store(Path(a.store)).stays.set_index("stay_id")
    stay = allp["stay"].astype(str)
    allp["sex"] = st.loc[stay, "sex"].astype(str).to_numpy()
    allp["age_band"] = np.digitize(st.loc[stay, "age"].to_numpy(dtype=float), AGE_BANDS[1:-1])
    allp["admission_type"] = st.loc[stay, "admission_type"].astype(str).to_numpy()
    race = {}
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(Path(a.tokens) / "vocab.json")
    for r in bench.real_rows(a.tokens, "held_out", set(stay), datasets=("mimic", "eicu", "nwicu")):
        for tok in r["ids"][:16]:                    # the prefix's race token
            m = v.meta[int(tok)]
            if m.get("cls") == "prefix" and m.get("field") == "race":
                race[r["stay_id"]] = str(m.get("value"))
    allp["race"] = [race.get(s, "not recorded") for s in stay]
    rep = {"design": design, "git": _sha(), "a1": {}, "a2": {}, "a5": {}, "b2": {}}
    for prog, g in allp.groupby("program"):
        wide = g.pivot_table(index="anchor", columns="model", values="p", aggfunc="first")
        meta = g.drop_duplicates("anchor").set_index("anchor").loc[wide.index]
        y, w, ds = meta["label"].to_numpy(), meta["weight"].to_numpy(), meta["dataset"].to_numpy()
        common = wide.dropna()
        res = {"anchors_common": int(len(common))}
        if len(common) >= 30 and len(np.unique(y[wide.index.isin(common.index)])) == 2:
            m = wide.index.isin(common.index)
            for model in common.columns:
                res[model] = bench.auroc_or_none(y[m], common[model].to_numpy(), w[m])
            if "twin" in common and "lgbm_pooled" in common:
                diff = lambda yy, a_, b_, ww: (bench.auroc_or_none(yy, a_, ww) or np.nan) - (  # noqa: E731
                    bench.auroc_or_none(yy, b_, ww) or np.nan)
                res["twin_minus_lgbm"] = res["twin"] - res["lgbm_pooled"] if res.get("twin") and res.get(
                    "lgbm_pooled") else None
                res["twin_minus_lgbm_ci"] = bench.bootstrap(diff, meta["stay"].to_numpy()[m], y[m],
                                                            common["twin"].to_numpy(), common["lgbm_pooled"].to_numpy(),
                                                            w[m], n=a.boot, strata=ds[m])
        rep["a1"][prog] = res
    # A2: recalibration and O/E by ICU day
    dev = pd.read_parquet(Path(a.dev) / "predictions.parquet") if a.dev else None
    twp = allp[allp["model"] == "twin"]
    for prog, g in twp.groupby("program"):
        y, p, w = g["label"].to_numpy(), g["p"].to_numpy(), g["weight"].to_numpy()
        if len(np.unique(y)) != 2 or not np.isin(y, (0.0, 1.0)).all():   # 0/1 outcomes only (not regressions)
            continue
        res = {"raw": bench.binary_metrics(y, p, w)}
        if dev is not None:
            d = dev[dev["program"] == prog]
            if len(d) >= 50 and len(np.unique(d["label"])) == 2:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(d["p"], d["label"],
                                                                                    sample_weight=d["weight"])
                res["recalibrated"] = bench.binary_metrics(y, iso.predict(p), w)
        res["by_icu_day"] = {}
        for day in sorted(g["day"].unique()):
            m = g["day"].to_numpy() == day
            if y[m].sum() >= 10:
                res["by_icu_day"][str(int(day)) if day < 4 else "4+"] = {
                    "anchors": int(m.sum()), "observed_over_expected":
                        float(np.average(y[m], weights=w[m]) / max(np.average(p[m], weights=w[m]), 1e-9))}
        rep["a2"][prog] = res
    # A5: subgroups, primary endpoints
    for prog in PRIMARY.get(design, ()):
        g = allp[(allp["program"] == prog) & allp["model"].isin(["twin", "lgbm_pooled"])]
        rep["a5"][prog] = {}
        for attr in ("sex", "age_band", "dataset", "admission_type", "race"):
            rep["a5"][prog][attr] = {}
            for val, h in g.groupby(attr):
                out = {}
                for model, hh in h.groupby("model"):
                    y = hh["label"].to_numpy()
                    if y.sum() >= 30 and (1 - y).sum() >= 30:
                        mm = bench.binary_metrics(y, hh["p"].to_numpy(), hh["weight"].to_numpy())
                        out[model] = {k: mm.get(k) for k in ("n", "positives", "auroc", "observed_over_expected",
                                                            "calibration_slope")}
                    else:
                        out[model] = {"n": int(len(y)), "positives": int(y.sum()), "scored": False}
                rep["a5"][prog][attr][str(val)] = out
    # B2: unknown site against known, same anchors
    if a.unknown and (Path(a.unknown) / "predictions.parquet").exists():
        un = pd.read_parquet(Path(a.unknown) / "predictions.parquet")
        un["anchor"] = un["anchor"].astype(str)
        for prog, g in un.groupby("program"):
            k = twp[twp["program"] == prog].set_index("anchor")
            common = g[g["anchor"].isin(k.index)]
            if len(common) < 30 or len(np.unique(common["label"])) < 2:
                continue
            kk = k.loc[common["anchor"]]
            y, w = common["label"].to_numpy(), common["weight"].to_numpy()
            rep["b2"][prog] = {"anchors": int(len(common)),
                               "known": bench.binary_metrics(y, kk["p"].to_numpy(), w),
                               "unknown": bench.binary_metrics(y, common["p"].to_numpy(), w)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "compare.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({p: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                      for p, r in rep["a1"].items() if p in PRIMARY.get(design, ())}, indent=1, default=str))


if __name__ == "__main__":
    main()
