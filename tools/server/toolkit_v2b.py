"""icutwin on the GPU host (CPU work): Paper P amendment 11, V2b (logistic recalibration of the treatment starts; post
hoc, the study lead's decision after V2, recorded in the protocol before it was computed).

For new IMV and new vasopressor within 24 h and new RRT within 48 h: logit(p') = a + b x logit(p), p the E2 risk
(``toolkit.recalibration.fit_logistic``), fitted on V2's tuning-split fit (``runs/toolkit/v2/fit_forecast``: the same
anchors, 32 rollouts, alignment-row labels, inclusion weights). Evaluated on V1's draws (R128, A1's labels) at 32
rollouts and at the first 16 (E2 from V1's kept hazard paths): O/E, calibration slope and intercept, ECE, Brier,
AUROC, raw and recalibrated. A map is adopted when b > 0 and, at both 32 and 16 rollouts, its O/E is closer to 1 and
its ECE no larger. Adopted maps join the toolkit's recalibration (``runs/toolkit/recalibration/<run>.json``, beside
V2's adopted isotonic maps). Writes ``runs/toolkit/v2/v2b.json``.

    python tools/server/toolkit_v2b.py
"""
import json

import numpy as np

PROGS = ("imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880")


def metrics(y, p, w) -> dict:
    from icutwin.eval import bench

    r = bench.binary_metrics(y, p, w)
    return {k: r.get(k) for k in ("auroc", "brier", "ece", "observed_over_expected", "calibration_slope",
                                  "calibration_intercept", "mean_predicted", "prevalence")}


def main() -> None:
    import pandas as pd

    from icutwin.toolkit import forecast
    from icutwin.toolkit import recalibration as rc
    from icutwin.toolkit.session import Paths

    paths = Paths.default()
    v2, v1 = paths.out / "v2", paths.out / "v1" / "forecast"
    pr = pd.read_parquet(v2 / "fit_forecast/predictions.parquet")
    e2_fit = pd.read_parquet(v2 / "fit_forecast/e2.parquet")
    a1 = pd.read_parquet(paths.paper / "a1/test_R/predictions.parquet")
    draws = {32: pd.read_parquet(v1 / "e2.parquet")}
    by, edges = forecast.load_paths(v1 / "hazards.npz")
    draws[16] = pd.DataFrame([{"anchor": k, "program": p, "e2": x["e2"]} for k, ps in by.items()
                              for p, x in forecast.e2_summary(ps[:16], edges)["programs"].items()])
    rep = {"method": "logit(p') = a + b logit(p), weighted ML on the tuning split's E2 (32 rollouts)",
           "programs": {}}
    adopted = {}
    for prog in PROGS:
        x = pr[pr["program"] == prog][["anchor", "weight", "label"]].merge(
            e2_fit[e2_fit["program"] == prog][["anchor", "e2"]], on="anchor")
        x = x[np.isfinite(x["label"]) & np.isfinite(x["e2"])]
        m = rc.fit_logistic(x["label"].to_numpy(), x["e2"].to_numpy(), x["weight"].to_numpy())
        m["source"] = "amendment 11 V2b: tuning R anchors, E2, 32 rollouts, logistic"
        res = {"fit": m, "evaluation": {}}
        better = []
        for n, src in draws.items():
            z = a1[a1["program"] == prog][["anchor", "label", "weight"]].merge(
                src[src["program"] == prog][["anchor", "e2"]], on="anchor")
            z = z[np.isfinite(z["label"]) & np.isfinite(z["e2"])]
            y, p, w = z["label"].to_numpy(), z["e2"].to_numpy(), z["weight"].to_numpy()
            raw, cal = metrics(y, p, w), metrics(y, rc.transform(m, p), w)
            ok = abs(cal["observed_over_expected"] - 1) < abs(raw["observed_over_expected"] - 1) and \
                cal["ece"] <= raw["ece"]
            better.append(ok)
            res["evaluation"][str(n)] = {"anchors": len(z), "raw": raw, "recalibrated": cal, "improved": bool(ok),
                                         "auroc_change": cal["auroc"] - raw["auroc"]}
        res["adopted"] = bool(m["b"] > 0 and all(better))
        if res["adopted"]:
            adopted[prog] = {**m, "adopted": "amendment 11 V2b: b > 0; O/E closer to 1 and ECE no larger at 32 and 16"}
        rep["programs"][prog] = res
        print(json.dumps({prog: {"a": m["a"], "b": m["b"], "adopted": res["adopted"],
                                 **{n: (e["raw"]["observed_over_expected"], e["recalibrated"]["observed_over_expected"],
                                        e["raw"]["ece"], e["recalibrated"]["ece"]) for n, e in res["evaluation"].items()}}}),
              flush=True)
    in_force = paths.out / "recalibration" / "MUC32_main.json"
    maps = rc.load(in_force)
    maps.update(adopted)
    rc.save(maps, in_force)
    rep["adopted"] = sorted(adopted)
    rep["maps_in_force"] = sorted(maps)
    (v2 / "v2b.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"adopted": rep["adopted"], "maps_in_force": rep["maps_in_force"]}), flush=True)


if __name__ == "__main__":
    main()
