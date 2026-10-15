"""Build the label sampler's pool on a store (plan Phase F, S3 / S6b; ``factory.pool``) and report it: programs per
group, label functions (clusters), the raked shares on each axis against P* and against P_clin, and the cells
whose effective number of distinct label functions is under the minimum (to be filled, "Label sampling" 4).
Aggregates only. Runs on the host:

    python tools/factory_pool.py --store data/factory/stores/bank_dev --out data/factory/pools/bank_dev --n-fh 3000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-fh", type=int, default=3000)
    ap.add_argument("--n-stays", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rerake", action="store_true", help="rake the saved pool at --out again (the targets changed)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from icutwin.factory import pool
    from icutwin.factory import store as fst

    t0 = time.time()
    st = fst.Store(Path(a.store))
    targets = pool.load_targets()
    if a.rerake:                                    # the attributes are kept: only the weights change
        import pandas as pd

        df = pd.read_parquet(Path(a.out) / "pool.parquet")
        for k, w in pool.rake_stages(df, targets).items():
            df[k] = w
        old = json.loads((Path(a.out) / "pool.json").read_text(encoding="utf-8"))
        cells = pool.cells(df, targets)
        low = [c for c in cells if c["ess"] < targets["min_ess"] and c["share"] > 0.002]
        old.update({"reraked": round(time.time() - t0, 1), "axes": pool.axis_report(df, targets),
                    "axes_s1": pool.axis_report(df.assign(weight=df["weight_s1"]), targets),
                    "cells": len(cells), "cells_under_min_ess": len(low),
                    "under_min_ess": sorted(low, key=lambda c: -c["share"])[:40]})
        df.to_parquet(Path(a.out) / "pool.parquet", index=False)
        (Path(a.out) / "pool.json").write_text(json.dumps(old, indent=1), encoding="utf-8")
        for g, axes in old["axes_s1"].items():
            print("stage 1", g, {ax: (r["tv_to_p_star"], r["kl_to_p_clin"]) for ax, r in axes.items()})
        return
    df, panel = pool.build(st, n_fh=a.n_fh, seed=a.seed, n_stays=a.n_stays, targets=targets, workers=a.workers)
    cells = pool.cells(df, targets)
    low = [c for c in cells if c["ess"] < targets["min_ess"] and c["share"] > 0.002]
    info = {"store": str(a.store), "seed": a.seed, "n_fh": a.n_fh, "seconds": round(time.time() - t0, 1),
            "groups": {g: {"programs": int((df["group"] == g).sum()),
                           "with_weight": int(((df["group"] == g) & (df["weight"] > 0)).sum()),
                           "clusters": int(df.loc[df["group"] == g, "cluster"].nunique())} for g in pool.GROUPS},
            "axes": pool.axis_report(df, targets),
            "axes_s1": pool.axis_report(df.assign(weight=df["weight_s1"]), targets), "cells": len(cells), "cells_under_min_ess": len(low),
            "under_min_ess": sorted(low, key=lambda c: -c["share"])[:40]}
    pool.save(df.drop(columns=["sig"]), panel, Path(a.out), info)
    print(json.dumps({k: info[k] for k in ("seconds", "groups", "cells", "cells_under_min_ess")}, indent=1))
    for g, axes in info["axes"].items():
        print(g, {ax: (r["tv_to_p_star"], r["kl_to_p_clin"]) for ax, r in axes.items()})


if __name__ == "__main__":
    main()
