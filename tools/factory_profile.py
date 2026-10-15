"""Where an episode's time and its redraws go (plan Phase F, S3 throughput and acceptance): ``--n`` episodes in one
process under cProfile, the sampler's redraw reasons, and attempts by family. Aggregates only.

    python tools/factory_profile.py --store data/factory/stores/bank_dev --pool data/factory/pools/bank_dev --n 200
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from collections import Counter
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--vocab", default="data/tokens/full_r4/vocab.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args(argv)
    from icutwin.factory import episodes, pool
    from icutwin.factory import store as fst
    from icutwin.tokenization.vocab import Vocab

    t0 = time.time()
    st = fst.Store(Path(a.store))
    progs, weights, support = pool.load(Path(a.pool)) if a.pool else (None, None, None)
    s = episodes.Sampler(st, Vocab.load(Path(a.vocab)), seed=a.seed, programs=progs, weights=weights,
                         targets=pool.load_targets(), support=support)
    print(json.dumps({"setup_s": round(time.time() - t0, 1)}), flush=True)
    attempts, fams, none = Counter(), Counter(), 0
    prof = cProfile.Profile()
    t1 = time.time()
    prof.enable()
    for i in range(a.start, a.start + a.n):
        try:
            r = s.recipe(i)
        except episodes.NoEpisode:
            none += 1
            continue
        s.materialize(r)
        attempts[r["program"]["family"]] += r["attempts"]
        fams[r["program"]["family"]] += 1
    prof.disable()
    wall = time.time() - t1
    print(json.dumps({"episodes": sum(fams.values()), "no_episode": none, "seconds": round(wall, 1),
                      "per_episode_s": round(wall / max(sum(fams.values()), 1), 3), "stats": s.stats,
                      "attempts_per_family": {f: round(attempts[f] / fams[f], 2) for f in fams}}, indent=1))
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(a.top)
    print(buf.getvalue())


if __name__ == "__main__":
    main()
