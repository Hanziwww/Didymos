"""icutwin on the GPU host: Paper P amendment 11, V2 (E2 recalibration; ``docs/p1_paper_protocol.md``).

* ``--part fit``: the toolkit's batch forecast on amendment 1's tuning-split R anchors (``runs/paper/a1/dev_R``: 100
  per dataset, their inclusion weights), 32 rollouts at that design's caps, with each anchor's alignment row (its own
  stay decoded on its clock) as the label; an isotonic map per program (E2 for the four hazard programs, the share for
  A1's other R primary programs; weighted, clipped): ``runs/toolkit/recalibration/<run>.json`` and
  ``runs/toolkit/v2/fit.json``.
* ``--part evaluate`` (CPU): on V1's draws (R128, ``runs/toolkit/v1/forecast``), with A1's labels, at 32 rollouts and
  at the first 16 (the shares re-read from V1's store, E2 from its kept hazard paths): O/E, calibration slope and
  intercept, ECE, Brier and AUROC, raw and recalibrated: ``runs/toolkit/v2/v2.json``.
    python tools/server/toolkit_v2.py --part fit --engines 3 --workers 5
"""
import argparse
import json
from pathlib import Path

import numpy as np

PRIMARY_R = ("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880", "aki_2@2880",
             "sepsis_ase@1440", "hypotension@240", "hyperkalemia_top@1440", "thrombocytopenia@2880")
E2_PROGS = ("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880")


def session(a):
    from icutwin.toolkit.session import Paths, Session

    return Session(Paths.default(), backend="vllm", engines=a.engines, gpu_memory_utilization=a.gpu,
                   workers=a.workers)


def fit(s, a, out: Path) -> None:
    import pandas as pd

    from icutwin.eval import bench
    from icutwin.toolkit import forecast, inputs
    from icutwin.toolkit import recalibration as rc

    dd = s.paths.paper / "a1" / "dev_R"
    an = pd.read_parquet(dd / "anchors.parquet").drop_duplicates("anchor").reset_index(drop=True)
    if a.limit:
        an = an.groupby("dataset").head(a.limit).reset_index(drop=True)
    caps = json.loads((dd / "design.json").read_text())["caps"]
    rows = {r["stay_id"]: r for r in bench.real_rows(s.paths.tokens, "tuning", set(an["twin"].astype(str)))}
    v = s.vocab
    pats, anchors, w = {}, [], {}
    for x in an.itertuples():
        sid = str(x.twin)
        if sid not in pats:
            pats[sid] = inputs.from_ids(rows[sid]["ids"], sid, x.dataset, "tuning")
        anc = pats[sid].at(v, float(x.hours))
        anchors.append(anc)
        w[anc.id] = float(x.weight)
    job = out / "fit_forecast"
    forecast.batch(s, anchors, job, n=a.n or 32, caps=caps, seed=21, design="R", weights=w, observed=True)
    pr = pd.read_parquet(job / "predictions.parquet")
    e2 = pd.read_parquet(job / "e2.parquet")
    maps, rep = {}, {"anchors": len(anchors), "programs": {}}
    for prog in PRIMARY_R:
        x = pr[pr["program"] == prog][["anchor", "weight", "label", "p"]]
        if prog in E2_PROGS:
            x = x.drop(columns="p").merge(e2[e2["program"] == prog][["anchor", "e2"]].rename(columns={"e2": "p"}),
                                          on="anchor")
        x = x[np.isfinite(x["label"]) & np.isfinite(x["p"])]
        if len(x) < 50 or not 0 < x["label"].sum() < len(x):
            rep["programs"][prog] = {"anchors": len(x), "fitted": False}
            continue
        m = rc.fit(x["label"].to_numpy(), x["p"].to_numpy(), x["weight"].to_numpy())
        m["source"] = f"amendment 11 V2: tuning R anchors, {'E2' if prog in E2_PROGS else 'share'}, 32 rollouts"
        maps[prog] = m
        rep["programs"][prog] = {"anchors": m["n"], "positives": m["positives"], "fitted": True,
                                 "estimator": "e2" if prog in E2_PROGS else "share",
                                 "oe_raw": float(np.sum(x["weight"] * x["label"]) / np.sum(x["weight"] * x["p"]))}
    rc.save(maps, rc.path_of(s))
    (out / "fit.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)


def metrics(y, p, w) -> dict:
    from icutwin.eval import bench

    r = bench.binary_metrics(y, p, w)
    return {k: r.get(k) for k in ("auroc", "brier", "ece", "observed_over_expected", "calibration_slope",
                                  "calibration_intercept", "mean_predicted", "prevalence")}


def evaluate(s, a, out: Path) -> None:
    import pandas as pd

    from icutwin.factory import store as fst
    from icutwin.toolkit import forecast
    from icutwin.toolkit import labels as tl
    from icutwin.toolkit import recalibration as rc

    v1 = s.paths.out / "v1" / "forecast"
    maps = rc.load(rc.path_of(s))
    a1 = pd.read_parquet(s.paths.paper / "a1/test_R/predictions.parquet")
    an = pd.read_parquet(s.paths.paper / "f1/R128/anchors.parquet").drop_duplicates("anchor").set_index("anchor")
    preds = {32: pd.read_parquet(v1 / "predictions.parquet")}
    e2 = {32: pd.read_parquet(v1 / "e2.parquet")}
    # the first 16 rollouts: the shares re-read from V1's store, E2 from its kept hazard paths
    sets = [tl.RolloutSet(x, x.split("@")[0], int(an.loc[x, "t_min"]), np.zeros(0, np.int32), [None] * 16)
            for x in an.index]
    r16 = tl.readout(fst.Store(v1 / "store"), sets, design="R")
    preds[16] = pd.DataFrame([{"anchor": k, "program": p, "p": x["p"]} for k, r in r16.items()
                              for p, x in r["binary"].items()])
    by, edges = forecast.load_paths(v1 / "hazards.npz")
    e2[16] = pd.DataFrame([{"anchor": k, "program": p, "e2": x["e2"]} for k, ps in by.items()
                           for p, x in forecast.e2_summary(ps[:16], edges)["programs"].items()])
    rep = {"maps": sorted(maps), "results": {}}
    for n in (32, 16):
        for prog in PRIMARY_R:
            lab = a1[a1["program"] == prog][["anchor", "label", "weight"]]
            src = e2[n][e2[n]["program"] == prog].rename(columns={"e2": "p"}) if prog in E2_PROGS else \
                preds[n][preds[n]["program"] == prog]
            x = lab.merge(src[["anchor", "p"]], on="anchor")
            x = x[np.isfinite(x["label"]) & np.isfinite(x["p"])]
            y, p, w = x["label"].to_numpy(), x["p"].to_numpy(), x["weight"].to_numpy()
            pc, ok = rc.apply(maps, prog, p)
            rep["results"][f"{prog}|{n}"] = {"rollouts": n, "anchors": len(x), "raw": metrics(y, p, w),
                                             "recalibrated": metrics(y, pc, w) if ok else None}
            if ok:
                d = rep["results"][f"{prog}|{n}"]
                d["auroc_change"] = d["recalibrated"]["auroc"] - d["raw"]["auroc"]
                d["auroc_within_0005"] = bool(abs(d["auroc_change"]) <= 0.005)
    (out / "v2.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: {"oe_raw": x["raw"]["observed_over_expected"],
                          "oe_cal": (x["recalibrated"] or {}).get("observed_over_expected"),
                          "auroc_ok": x.get("auroc_within_0005")} for k, x in rep["results"].items()}, indent=1),
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("fit", "evaluate"), required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--engines", type=int, default=3)
    ap.add_argument("--gpu", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    s = session(a)
    out = Path(a.out) if a.out else s.paths.out / "v2"
    out.mkdir(parents=True, exist_ok=True)
    {"fit": fit, "evaluate": evaluate}[a.part](s, a, out)


if __name__ == "__main__":
    main()
