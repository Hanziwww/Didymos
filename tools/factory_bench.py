"""The episode factory's S3 gate on the host (plan Phase F): episodes from a store and its pool in ``--workers``
processes, each episode a function of its index. Reports throughput (episodes and events per second), acceptance
(accepted over drawn), contract failures (the gate: none), the realised mix (family groups, label types, cadences,
context policies, descriptor and anonymization shares, context sizes), and determinism (the same indices with one
worker and with several give identical episodes). Aggregates only.

    python tools/factory_bench.py --store data/factory/stores/bank_dev --pool data/factory/pools/bank_dev \\
        --n 10000 --workers 12 --out runs/factory/bench_s3.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_S: dict = {}


def _sampler(store: str, pool_dir: str | None, vocab: str, stage: int, seed: int, split: str):
    key = (store, pool_dir, stage, seed, split)
    if _S.get("key") != key:
        from icutwin.factory import episodes, pool
        from icutwin.factory import store as fst
        from icutwin.tokenization.vocab import Vocab

        st = fst.Store(Path(store))
        v = Vocab.load(Path(vocab))
        progs, weights, support = pool.load(Path(pool_dir)) if pool_dir else (None, None, None)
        _S.update(key=key, s=episodes.Sampler(st, v, stage=stage, seed=seed, programs=progs, weights=weights,
                                              targets=pool.load_targets(), split=split, support=support))
    return _S["s"]


def digest(ep: dict) -> str:
    h = hashlib.sha1(json.dumps(ep["recipe"], sort_keys=True).encode())
    for part in ("context", "query"):
        for k in sorted(ep[part]):
            x = ep[part][k]
            if isinstance(x, np.ndarray):
                h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def _job(args) -> list[dict]:
    store, pool_dir, vocab, stage, seed, split, idx = args
    from icutwin.factory import contracts
    from icutwin.factory.episodes import NoEpisode

    s = _sampler(store, pool_dir, vocab, stage, seed, split)
    before = dict(s.stats)
    out = []
    for i in idx:
        t0 = time.time()
        try:
            ep = s.episode(i)
        except NoEpisode:
            s.stats["no_episode"] = s.stats.get("no_episode", 0) + 1
            continue
        t1 = time.time()
        fails = contracts.episode_failures(ep)
        r = ep["recipe"]
        out.append({"i": i, "seconds": t1 - t0, "fails": fails, "digest": digest(ep),
                    "group": r["program"]["family"], "label": ep["task"]["label"], "cadence": r["cadence"][0],
                    "policy": r["policy"], "descriptor": r["descriptor"], "anon": r["anonymize"] is not None,
                    "context": len(r["context"]), "queries": len(r["queries"]),
                    "times": sum(len(q["times"]) for q in r["queries"]), "attempts": r["attempts"],
                    "filter_stage": r["filter"].get("stage"),
                    "events": int(len(ep["context"]["ev_t"]) + len(ep["query"]["ev_t"])),
                    "bytes": int(sum(x.nbytes for p in ("context", "query") for x in ep[p].values()
                                     if isinstance(x, np.ndarray)))})
    out.append({"stats": {k: v - before.get(k, 0) for k, v in s.stats.items() if v != before.get(k, 0)}})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--vocab", default="data/tokens/full_r4/vocab.json")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--check", type=int, default=200, help="indices recomputed with one worker (determinism)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from icutwin.factory.pool import GROUPS

    chunk = 50
    jobs = [(a.store, a.pool, a.vocab, a.stage, a.seed, a.split, list(range(k, min(k + chunk, a.n))))
            for k in range(0, a.n, chunk)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for part in ex.map(_job, jobs) for r in part]
    wall = time.time() - t0
    stats: Counter = Counter()
    for r in rows:
        stats.update(r.get("stats", {}))
    res = [r for r in rows if "stats" not in r]
    single = [r for r in _job((a.store, a.pool, a.vocab, a.stage, a.seed, a.split, list(range(a.check))))
              if "stats" not in r]
    same = sum(x["digest"] == y["digest"] for x, y in zip(single, res[:a.check]))
    fam2grp = {f: g for g, fs in GROUPS.items() for f in fs}
    n = len(res)
    sh = lambda key: {k: round(c / n, 4) for k, c in Counter(r[key] for r in res).most_common()}  # noqa: E731
    report = {"episodes": n, "workers": a.workers, "wall_s": round(wall, 1),
              "episodes_per_s": round(n / wall, 2), "events_per_s": round(sum(r["events"] for r in res) / wall),
              "per_episode_s_p50_p95": [round(float(np.quantile([r["seconds"] for r in res], q)), 3)
                                        for q in (0.5, 0.95)],
              "acceptance": round(n / sum(r["attempts"] for r in res), 4),
              "redraws": dict(stats.most_common()),
              "contract_failures": sum(bool(r["fails"]) for r in res),
              "failure_examples": [r["fails"][:3] for r in res if r["fails"]][:5],
              "determinism": {"checked": a.check, "identical": same},
              "groups": {k: round(c / n, 4) for k, c in Counter(fam2grp.get(r["group"], r["group"])
                                                              for r in res).most_common()},
              "families": sh("group"), "labels": sh("label"), "cadences": sh("cadence"), "policies": sh("policy"),
              "descriptor_shown": round(float(np.mean([r["descriptor"] for r in res])), 4),
              "anonymized": round(float(np.mean([r["anon"] for r in res])), 4),
              "filter_stages": sh("filter_stage"),
              "context_size_p5_50_95": [int(np.quantile([r["context"] for r in res], q)) for q in (0.05, 0.5, 0.95)],
              "zero_context": round(float(np.mean([r["context"] == 0 for r in res])), 4),
              "events_p50_p95": [int(np.quantile([r["events"] for r in res], q)) for q in (0.5, 0.95)],
              "megabytes_p50_p95": [round(float(np.quantile([r["bytes"] for r in res], q)) / 2**20, 2)
                                    for q in (0.5, 0.95)]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
