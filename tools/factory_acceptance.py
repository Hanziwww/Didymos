"""The episode sampler's acceptance per pool cell (plan Phase F, "Label sampling" 5: a cell's weight divided by its
measured acceptance): ``--n`` episodes of a store with a pool (its weights as raked, any earlier acceptance file set
aside), every attempt's program and whether it became an episode, the share accepted per (group, label, horizon
band, prevalence band) cell, written to ``<pool>/acceptance.json`` so ``pool.load`` divides the weights by it.
Episode i stays a function of (store, pool with its acceptance file, seed, i). Aggregates only.

    python tools/factory_acceptance.py --store data/factory/stores/bank_dev --pool data/factory/pools/bank_dev \\
        --n 20000 --workers 12
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_S: dict = {}


def _job(args) -> list:
    store, pool_dir, vocab, seed, idx = args
    if _S.get("key") != (store, pool_dir, seed):
        import pandas as pd

        from icutwin.factory import episodes, pool
        from icutwin.factory import store as fst
        from icutwin.tokenization.vocab import Vocab

        df = pd.read_parquet(Path(pool_dir) / "pool.parquet")
        if "weight_s1" in df:                                     # stage 1's raking, without acceptance
            df["weight"] = df["weight_s1"]
        progs, weights = pool.load_programs(df)
        s = episodes.Sampler(fst.Store(Path(store)), Vocab.load(Path(vocab)), seed=seed, programs=progs,
                             weights=weights, targets=pool.load_targets(), support=pool.load_support(df))
        _S.update(key=(store, pool_dir, seed), s=s)
    from icutwin.factory.episodes import NoEpisode

    s = _S["s"]
    s.attempt_log = []
    sites = []
    for i in idx:
        try:
            sites.append(s.recipe(i)["site"])
        except NoEpisode:
            pass
    return s.attempt_log, sites


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--vocab", default="data/tokens/full_r4/vocab.json")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-attempts", type=int, default=20)
    a = ap.parse_args(argv)
    import pandas as pd

    from icutwin.factory import pool

    t0 = time.time()
    jobs = [(a.store, a.pool, a.vocab, a.seed, list(range(k, min(k + 100, a.n)))) for k in range(0, a.n, 100)]
    with ProcessPoolExecutor(a.workers) as ex:
        parts = list(ex.map(_job, jobs))
    log = [x for lg, _s in parts for x in lg]
    used = Counter(x for _lg, s in parts for x in s)
    df = pd.read_parquet(Path(a.pool) / "pool.parquet", columns=["hash", *pool.ACCEPT_KEYS])
    key = {h: pool.cell_key(r) for h, r in zip(df["hash"], df[list(pool.ACCEPT_KEYS)].to_dict("records"),
                                                strict=True)}
    tried, took = Counter(), Counter()
    for h, ok in log:
        k = key.get(h)
        if k is None:                                   # a fresh functional head, not in the pool
            continue
        tried[k] += 1
        took[k] += bool(ok)
    cells = {k: round(took[k] / n, 4) for k, n in tried.items() if n >= a.min_attempts}
    from icutwin.factory import store as fst

    st = fst.Store(Path(a.store))
    share = st.stays.loc[st.stays["split"] == "train", "dataset"].value_counts(normalize=True).to_dict() \
        if "split" in st.stays else st.stays["dataset"].value_counts(normalize=True).to_dict()
    tot = sum(used.values())
    sites = {d: round(float(np.clip(share[d] / (used[d] / tot), 0.5, 2.0)), 4) for d in share if used.get(d)}
    out = {"store": a.store, "seed": a.seed, "episodes": a.n, "attempts": len(log),
           "accepted": int(sum(ok for _h, ok in log)), "cells": cells,
           "cells_too_few_attempts": sum(n < a.min_attempts for n in tried.values()),
           "sites": sites, "site_use": {d: round(n / tot, 4) for d, n in used.items()},
           "site_share": {d: round(v, 4) for d, v in share.items()},
           "seconds": round(time.time() - t0, 1)}
    (Path(a.pool) / "acceptance.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    acc = sorted(cells.values())
    print(json.dumps({k: out[k] for k in ("episodes", "attempts", "accepted", "cells_too_few_attempts", "seconds")}
                     | {"cells": len(cells), "acceptance_p10_p50_p90": [acc[len(acc) // 10], acc[len(acc) // 2],
                                                                         acc[9 * len(acc) // 10]] if acc else None},
                     indent=1))


if __name__ == "__main__":
    main()
