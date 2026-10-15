"""The alignment-diversity frontier of the label sampler (plan Phase F, "Label sampling" 6; Q&A 12): the saved pool
raked again for alpha in {0, 0.5, 0.7, 1} x eps in {0, 0.1, 0.2} (the attributes are the pool's; nothing is
evaluated again), each point's alignment, the KL divergence of the raked groups' realised attribute shares to P_clin
(averaged over axes, groups weighted by the study lead's mix), against its diversity, the effective number of distinct
label functions (exp of the entropy of the clusters' weight shares, raked groups pooled). Two references: the clinical
catalog alone (aligned, narrow) and the pool's uniform base (diverse, unaligned). Aggregates only.

    python tools/factory_frontier.py --pool data/factory/pools/bank_dev --out runs/factory/frontier.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ALPHAS = (0.0, 0.5, 0.7, 1.0)
EPSS = (0.0, 0.1, 0.2)
RAKED = ("threshold", "values", "functional")


def point(df, w, targets: dict, mix: dict) -> dict:
    from icutwin.factory import pool

    d = df.assign(weight=w)
    rep = pool.axis_report(d, targets)
    kl, tv, tot = 0.0, 0.0, 0.0
    for g in RAKED:
        if g not in rep:
            continue
        axes = rep[g]
        kl += mix[g] * float(np.mean([r["kl_to_p_clin"] for r in axes.values()]))
        tv += mix[g] * float(np.mean([r["tv_to_p_star"] for r in axes.values()]))
        tot += mix[g]
    m = d["group"].isin(RAKED).to_numpy() & (w > 0)
    mass = d[m].groupby(["group", "cluster"])["weight"].sum()
    share = (mass.groupby(level=0).transform(lambda s: s / s.sum()) *
             mass.index.get_level_values(0).map(lambda g: mix[g] / tot)).to_numpy()
    share = share[share > 0]
    return {"kl_to_p_clin": round(kl / tot, 4), "tv_to_p_star": round(tv / tot, 4),
            "effective_label_functions": round(float(np.exp(-(share * np.log(share)).sum())), 1),
            "effective_size": round(float(1 / (share ** 2).sum()), 1)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import pandas as pd

    from icutwin.factory import episodes, pool

    df = pd.read_parquet(Path(a.pool) / "pool.parquet")
    base_t = pool.load_targets()
    cfg = episodes.load_config()["families"]
    mix = {g: cfg[g]["share"] for g in RAKED}
    out = {"grid": []}
    for al in ALPHAS:
        for ep in EPSS:
            t = {**base_t, "alpha": al, "eps": ep}
            w = pool.rake(df, t)
            out["grid"].append({"alpha": al, "eps": ep, **point(df, w, t, mix)})
            print(json.dumps(out["grid"][-1]), flush=True)
    _u, inv, cnt = np.unique(df["cluster"].to_numpy(), return_inverse=True, return_counts=True)
    out["uniform_base"] = point(df, 1.0 / cnt[inv], base_t, mix)
    cat = df["name"].fillna("") != ""                    # the clinical catalog's programs alone, as one group
    if cat.any():
        dc = df[cat].assign(group="threshold")
        out["catalog_alone"] = point(dc, np.ones(int(cat.sum())), base_t, {"threshold": 1.0, "values": 0.0,
                                                                            "functional": 0.0})
    out["knee"] = {"alpha": base_t["alpha"], "eps": base_t["eps"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("uniform_base", "catalog_alone", "knee") if k in out}, indent=1))


if __name__ == "__main__":
    main()
