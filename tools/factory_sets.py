"""Evaluation sets and context banks (plan Phase F, S7; Factory design 11), written on the host from a stay store with
the clinical catalog's programs, never part of a training output. Internal and DUA-derived (real stays): they stay on
the approved hosts.
* ``episodes``: a frozen dump of trajectory episodes (``stream.dump``; the store's own split);
* ``banks``: context banks per (site, program): ``n`` complete trajectories with the program's label timeline
  (``Sampler.context_bank``), named by site and program hash, so a twin bank and a real episode of the same site and
  program form a switchable-context pair (twin context, real queries);
* ``queries``: full-trajectory query sets per program: stays queried at every cadence time (alarm metrics, lead
  time; ``Sampler.full_queries``).
Each part has a manifest with its files' SHA-256.

    python tools/factory_sets.py --store data/factory/stores/real_test --out data/factory/sets/real_test \\
        --episodes 2000 --bank-size 64 --query-stays 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", default="data/tokens/full_r4/vocab.json")
    ap.add_argument("--split", default=None)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--bank-size", type=int, default=64)
    ap.add_argument("--query-stays", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import numpy as np

    from icutwin.factory import episodes
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst
    from icutwin.factory import stream
    from icutwin.tokenization.vocab import Vocab

    out = Path(a.out)
    st = fst.Store(Path(a.store))
    s = episodes.Sampler(st, Vocab.load(Path(a.vocab)), seed=a.seed, split=a.split)
    rep: dict = {"store": a.store, "split": a.split}
    if a.episodes:
        es = stream.EpisodeStream(Path(a.store), Path(a.vocab), None, 1, a.seed, a.split)
        man = stream.dump(es, list(range(a.episodes)), out / "episodes", name="episodes")
        rep["episodes"] = {"shards": len(man["shards"])}
    progs = [p for p in fpr.catalog(st.channels) if p.label in ("binary", "hosp_death")]
    for part in ("banks", "queries"):
        (out / part).mkdir(parents=True, exist_ok=True)
    banks, queries = {}, {}
    rng = np.random.default_rng(a.seed)
    for p in progs:
        cadence = ("landmark", 1440) if p.label == "hosp_death" else ("rolling", 60)
        for site in s.site_names:
            ep = s.context_bank(p, cadence, str(site), a.bank_size, a.seed)
            if len(ep["recipe"]["context"]) < 2:
                continue
            f = out / "banks" / f"{site}__{p.hash()[:16]}.npz"
            banks[f.name] = {"site": str(site), "program": p.name, "horizon": p.horizon, "hash": p.hash(),
                             "patients": len(ep["recipe"]["context"]), "sha256": stream.write_shard([ep], f)}
            stays = s.by_site[site]
            pick = np.sort(rng.choice(stays, size=min(a.query_stays, len(stays)), replace=False))
            q = s.full_queries(p, cadence, [int(i) for i in pick])
            g = out / "queries" / f"{site}__{p.hash()[:16]}.npz"
            queries[g.name] = {"site": str(site), "program": p.name, "horizon": p.horizon, "hash": p.hash(),
                               "stays": len(q["recipe"]["queries"]),
                               "times": int(sum(len(x["times"]) for x in q["recipe"]["queries"])),
                               "sha256": stream.write_shard([q], g)}
    (out / "banks" / "manifest.json").write_text(json.dumps(banks, indent=1), encoding="utf-8")
    (out / "queries" / "manifest.json").write_text(json.dumps(queries, indent=1), encoding="utf-8")
    rep.update({"banks": len(banks), "query_sets": len(queries)})
    (out / "manifest.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
