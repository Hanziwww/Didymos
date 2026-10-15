"""icutwin on the GPU host (CPU work): Paper P amendment 11, V1's added check (post hoc, recorded in the protocol
before it was computed). For each of A1's nine primary R programs, each rollout's label in V1's store (the label
engine at the anchor's prediction time), the toolkit's 32-rollout share resampled 1,000 times with the rollouts drawn
with replacement within each anchor (anchors fixed; a share needs half its rollouts valid, as A1): the standard
deviation of its AUROC and O/E against A1's labels. Two independent replicates on the same anchors should differ by
at most 1.96 x sqrt(2) times it. Writes ``runs/toolkit/v1/v1_noise.json``.

    python tools/server/toolkit_v1_noise.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

PRIMARY_R = ("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880", "aki_2@2880",
             "sepsis_ase@1440", "hypotension@240", "hyperkalemia_top@1440", "thrombocytopenia@2880")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import pandas as pd

    from icutwin.eval import bench
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst
    from icutwin.toolkit import labels as tl
    from icutwin.toolkit.session import Paths

    paths = Paths.default()
    v1 = paths.out / "v1"
    rep = json.loads((v1 / "v1.json").read_text())
    store = fst.Store(v1 / "forecast" / "store")
    an = pd.read_parquet(paths.paper / "f1/R128/anchors.parquet").drop_duplicates("anchor").reset_index(drop=True)
    a1 = pd.read_parquet(paths.paper / "a1/test_R/predictions.parquet")
    idx = {s: i for i, s in enumerate(store.stays["stay_id"].tolist())}
    n_roll = 32
    st = np.array([[idx[f"{x}#{k}"] for k in range(n_roll)] for x in an["anchor"]], dtype=np.int64)
    rows = fl.Rows(st.reshape(-1), np.repeat(an["t_min"].to_numpy(dtype=np.int64), n_roll))
    ev = fpr.Evaluator(store)
    progs = {tl.key(p): p for p in fpr.catalog(store.channels)}
    rng = np.random.default_rng(a.seed)
    pick = rng.integers(0, n_roll, size=(a.boot, len(an), n_roll), dtype=np.int8)
    out = {"boot": a.boot, "rule": "|toolkit - A1| <= 1.96 x sqrt(2) x the rollout-bootstrap SD", "programs": {}}
    for name in PRIMARY_R:
        b = ev.evaluate(progs[name], rows)
        yr = np.where(b.valid, b.y, np.nan).astype(np.float64).reshape(len(an), n_roll)
        lab = a1[a1["program"] == name].drop_duplicates("anchor").set_index("anchor")
        y = lab["label"].reindex(an["anchor"]).to_numpy(float)
        w = an["weight"].to_numpy(float)
        p_a1 = lab["p"].reindex(an["anchor"]).to_numpy(float)
        nv = np.isfinite(yr).sum(1)
        p_tk = np.where(nv >= 16, np.nansum(yr, 1) / np.maximum(nv, 1), np.nan)
        ok = np.isfinite(y) & np.isfinite(p_a1) & np.isfinite(p_tk)
        au, oe = [], []
        for k in range(a.boot):
            sub = np.take_along_axis(yr, pick[k], axis=1)
            nvk = np.isfinite(sub).sum(1)
            pk = np.where(nvk >= 16, np.nansum(sub, 1) / np.maximum(nvk, 1), np.nan)
            m = ok & np.isfinite(pk)
            au.append(bench.auroc_fast(y[m], pk[m], w[m]))
            oe.append(float(np.sum(w[m] * y[m]) / max(np.sum(w[m] * pk[m]), 1e-12)))
        sd_au, sd_oe = float(np.nanstd(au)), float(np.nanstd(oe))
        x = rep["forecast"][name]
        d_au, d_oe = x["auroc_toolkit"] - x["auroc_a1"], x["oe_toolkit"] - x["oe_a1"]
        lim_au, lim_oe = 1.96 * np.sqrt(2) * sd_au, 1.96 * np.sqrt(2) * sd_oe
        out["programs"][name] = {"anchors": int(ok.sum()), "positives": int(np.nansum(y[ok])),
                                 "auroc_a1": x["auroc_a1"], "auroc_toolkit": x["auroc_toolkit"],
                                 "auroc_reproduced": bench.auroc_or_none(y[ok], p_tk[ok], w[ok]),
                                 "sd_auroc_rollouts": sd_au, "auroc_difference": d_au, "auroc_limit": lim_au,
                                 "oe_a1": x["oe_a1"], "oe_toolkit": x["oe_toolkit"], "sd_oe_rollouts": sd_oe,
                                 "oe_difference": d_oe, "oe_limit": lim_oe,
                                 "original_rule_pass": x["pass"],
                                 "pass": bool(abs(d_au) <= lim_au and abs(d_oe) <= lim_oe)}
        print(json.dumps({name: {k: out["programs"][name][k] for k in ("auroc_difference", "auroc_limit",
                                                                        "oe_difference", "oe_limit", "pass")}}),
              flush=True)
    others = all(x["pass"] for part in ("e2", "whatif") for x in rep[part].values())
    out["pass"] = bool(others and all(x["pass"] for x in out["programs"].values()))
    out["e2_and_whatif_pass"] = others
    Path(v1 / "v1_noise.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"pass": out["pass"]}), flush=True)


if __name__ == "__main__":
    main()
