"""Build the factory's stay stores (plan Phase F, S1; ``factory.store``). Runs on the host with the project environment;
internal, DUA-derived outputs under ``data/factory/stores`` (never committed).

    python tools/factory_store.py bank --bank data/bank/MUC32_main_bank --tokens data/tokens/full_r4 \\
        --out data/factory/stores/bank_dev --val-pct 10 --workers 8
    python tools/factory_store.py tokens --tokens data/tokens/full_r4 --split tuning --label dev \\
        --out data/factory/stores/real_dev [--per-dataset 3000]
    python tools/factory_store.py meds --meds data/meds/full --derived data/derived --tokens data/tokens/full_r4 \\
        --split tuning --label dev_raw --out data/factory/stores/real_raw_dev --per-dataset 500
    python tools/factory_store.py stats --store data/factory/stores/bank_dev
    python tools/factory_store.py marginals --tokens data/tokens/full_r4
    python tools/factory_store.py supply --tokens data/tokens/full_r4 --shards data/supply/<run> \\
        --out data/factory/stores/<pool> --segment seg_0000 --workers 16
    python tools/factory_store.py weights --out data/factory/stores/<pool> --plan data/supply/<run>/plan.json \\
        --marginals data/tokens/full_r4/supply_marginals.json

``bank``: one segment per bank part (``part<k>``), each twin's split by a hash of its id (``store.twin_split``).
``generic``: a generic-trajectory store (``factory.generic``; the switchable general-task source).
``index``: the key-major copies added to stores built before them (``store.index_segments``).
``marginals``: the training stays' aggregates a supply plan and its weights need (``generate.supply.marginals``).
``supply``: a supply run's shards (``generate.supply.produce``) appended as a segment, with each twin's cell and
request; ``weights``: the pool's importance weights for its segments (``generate.supply``; goals checked).
``tokens``: real stays' token sequences through the twins' decode (the real-through-tokens store), one segment per
dataset, a seeded sample of ``per-dataset`` stays when given (``--all-splits``: every split's stays, for an
inference-only database). ``meds``: real stays as recorded (the real-raw store,
for the quantization report), one segment per dataset. ``stats``: the training twins' channel statistics into the
store's ``channels.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATASETS = ("mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire")
SPLITS = ("train", "tuning", "held_out")


def _sample(n: int, k: int | None, seed: int) -> np.ndarray:
    if k is None or k >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))


def cmd_bank(a) -> None:
    from icutwin.factory import store

    tokens = Path(a.tokens)
    parts = sorted(p for p in Path(a.bank).iterdir() if p.is_dir() and p.name.startswith("part"))
    for k, part in enumerate(parts):
        rows = []
        for f in sorted(part.glob("*.parquet")):
            for r in pq.read_table(f, columns=["twin", "dataset", "ids", "weight"]).to_pylist():
                rows.append({"stay_id": r["twin"], "ids": r["ids"], "weight": r["weight"], "cell": "base",
                             "split": store.twin_split(r["twin"], a.val_pct)})
        t0 = time.time()
        e = store.build(rows, tokens / "vocab.json", Path(a.out), f"seg_{k:04d}",
                        {"kind": "bank", "bank": str(a.bank), "part": part.name}, seed=a.seed,
                        persistence=tokens / "rank_persistence.json", workers=a.workers, chunk=a.chunk)
        print(json.dumps({"segment": e["name"], "stays": e["stays"], "events": e["events"],
                          "seconds": round(time.time() - t0, 1)}), flush=True)


def cmd_supply(a) -> None:
    """Append a supply run's shards (``generate.supply.produce``; every ``part*`` directory) as one segment: each
    twin with its cell, its request and its split by a hash of its id. A twin whose tokens repeat an earlier
    segment's or another part's is left out (the parts' hash sets are their own)."""
    from icutwin.factory import store
    from icutwin.generate import bank

    tokens = Path(a.tokens)
    root = Path(a.shards)
    parts = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("part")) or [root]
    seen = set()
    if (Path(a.out) / "hashes.txt").exists():
        seen = set((Path(a.out) / "hashes.txt").read_text().split())
    rows, dup = [], 0
    for part in parts:
        for f in sorted(part.glob("*.parquet")):
            for r in pq.read_table(f, columns=["twin", "ids", "cell", "conditions", "goal_met"]).to_pylist():
                h = bank.seq_hash(r["ids"])
                if h in seen:
                    dup += 1
                    continue
                seen.add(h)
                rows.append({"stay_id": r["twin"], "ids": r["ids"], "weight": float("nan"), "cell": r["cell"],
                             "conditions": r["conditions"], "goal_met": r["goal_met"],
                             "split": store.twin_split(r["twin"], a.val_pct)})
    t0 = time.time()
    e = store.build(rows, tokens / "vocab.json", Path(a.out), a.segment,
                    {"kind": "supply", "shards": str(root), "parts": [p.name for p in parts]}, seed=a.seed,
                    persistence=tokens / "rank_persistence.json", workers=a.workers, chunk=a.chunk)
    (Path(a.out) / "hashes.txt").write_text("\n".join(sorted(seen)))
    sec = time.time() - t0
    print(json.dumps({"segment": e["name"], "stays": e["stays"], "events": e["events"], "cross_part_duplicates": dup,
                      "seconds": round(sec, 1),
                      "twins_per_core_hour": round(e["stays"] / max(sec * a.workers / 3600, 1e-9), 1)}), flush=True)


def cmd_weights(a) -> None:
    """The pool's importance weights (``generate.supply``) for the store's segments: realized attributes, the goal
    checks on the derived concepts (a twin that missed a goal is rejected), the balance-heuristic weights from the
    realized cells."""
    from icutwin.factory import store
    from icutwin.generate import supply

    plan = supply.SupplyPlan.load(Path(a.plan))
    marg = json.loads(Path(a.marginals).read_text(encoding="utf-8"))
    st = store.Store(Path(a.out))
    attrs = supply.store_attributes(st)
    met = supply.goal_check(attrs)
    rejected = np.array([m is False for m in met], dtype=bool)        # a goal missed, as the weights read it
    keep = [x for x, r in zip(attrs, rejected, strict=True) if not r]
    counts: dict = {}
    for x in keep:
        counts[x["cell"]] = counts.get(x["cell"], 0) + 1
    w = np.zeros(len(attrs))
    w[~rejected] = plan.weights(keep, marg, counts=counts, shrink=a.shrink)
    info = store.set_weights(Path(a.out), [x["stay_id"] for x in attrs], w, met, rejected,
                             {"plan": plan.name, "marginals": str(a.marginals), "shrink": a.shrink,
                              "counts": counts})
    print(json.dumps({"stays": len(attrs), "rejected": int(rejected.sum()),
                      "weight_p01_p50_p99": np.quantile(w[~rejected], [0.01, 0.5, 0.99]).round(4).tolist()
                      if (~rejected).any() else None, "segments": info["segments"]}), flush=True)


def cmd_generic(a) -> None:
    from icutwin.factory import generic
    from icutwin.factory import store as fst

    e = generic.build_store(Path(a.out), n_stays=a.n_stays, n_channels=a.n_channels, seed=a.seed)
    st = fst.Store(Path(a.out))
    print(json.dumps({"stays": e["stays"], "events": e["events"], "channels": len(st.channels),
                      "train": int((st.stays["split"] == "train").sum())}), flush=True)


def cmd_index(a) -> None:
    from icutwin.factory import store

    for s in a.store:
        print(json.dumps({"store": s, "indexed": store.index_segments(Path(s))}), flush=True)


def cmd_marginals(a) -> None:
    from icutwin.generate import supply

    m = supply.marginals(Path(a.tokens), list(a.datasets), a.split)
    out = Path(a.out or Path(a.tokens) / "supply_marginals.json")
    out.write_text(json.dumps(m), encoding="utf-8")
    print(json.dumps({"out": str(out), "datasets": m["datasets"],
                      "strata": {d: len(s) for d, s in m["strata"].items()}}), flush=True)


def cmd_tokens(a) -> None:
    from icutwin.factory import store

    import pyarrow as pa

    tokens = Path(a.tokens)
    if not (a.all_splits or a.split):
        raise SystemExit("tokens: give --split or --all-splits")
    splits = SPLITS if a.all_splits else (a.split,)
    for k, ds in enumerate(a.datasets):
        t = pa.concat_tables([pq.read_table(tokens / ds / f"{sp}.parquet", columns=["stay_id", "ids"]) for sp in splits])
        idx = _sample(t.num_rows, a.per_dataset, a.seed + k)
        rows = [{"stay_id": r["stay_id"], "ids": r["ids"], "split": a.label, "weight": 1.0, "cell": "real"}
                for r in t.take(idx).to_pylist()]
        t0 = time.time()
        e = store.build(rows, tokens / "vocab.json", Path(a.out), f"seg_{ds}",
                        {"kind": "real_tokens", "split": "+".join(splits), "dataset": ds, "sampled": len(rows),
                         "of": t.num_rows}, seed=a.seed, persistence=tokens / "rank_persistence.json",
                        workers=a.workers, chunk=a.chunk)
        print(json.dumps({"segment": e["name"], "stays": e["stays"], "events": e["events"],
                          "seconds": round(time.time() - t0, 1)}), flush=True)


def cmd_meds(a) -> None:
    from icutwin.factory import store
    from icutwin.tokenization.stays import iter_stays

    tokens = Path(a.tokens)
    for k, ds in enumerate(a.datasets):
        t0 = time.time()
        stays = list(iter_stays(Path(a.meds) / ds, a.split, Path(a.derived) / ds))
        idx = _sample(len(stays), a.per_dataset, a.seed + k)
        stays = [stays[i] for i in idx]
        der = store.derived_of(Path(a.derived) / ds)
        e = store.build_from_stays(stays, der, tokens / "vocab.json", Path(a.out), f"seg_{ds}",
                                   {"kind": "real_meds", "split": a.split, "dataset": ds}, split=a.label)
        print(json.dumps({"segment": e["name"], "stays": e["stays"], "events": e["events"],
                          "seconds": round(time.time() - t0, 1)}), flush=True)


def cmd_stats(a) -> None:
    from icutwin.factory import store

    st = store.Store(Path(a.store))
    stats = store.channel_stats(st, split=a.split)
    ch = st.channels
    ch.stats = stats
    ch.save(Path(a.store) / "channels.json")
    print(json.dumps({"channels_with_stats": len(stats)}), flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("bank", "tokens", "meds"):
        g = sub.add_parser(name)
        g.add_argument("--tokens", required=True)
        g.add_argument("--out", required=True)
        g.add_argument("--seed", type=int, default=0)
        g.add_argument("--workers", type=int, default=1)
        g.add_argument("--chunk", type=int, default=500)
        if name == "bank":
            g.add_argument("--bank", required=True)
            g.add_argument("--val-pct", type=float, default=10.0)
        else:
            g.add_argument("--split", required=name == "meds")
            g.add_argument("--label", required=True)
            if name == "tokens":   # an inference-only database (stage.spec.EXTERNAL): every stay is a test stay
                g.add_argument("--all-splits", action="store_true")
            g.add_argument("--per-dataset", type=int, default=None)
            g.add_argument("--datasets", nargs="+", default=list(DATASETS))
        if name == "meds":
            g.add_argument("--meds", required=True)
            g.add_argument("--derived", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--store", required=True)
    s.add_argument("--split", default="train")
    g = sub.add_parser("supply")
    g.add_argument("--tokens", required=True)
    g.add_argument("--shards", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--segment", required=True)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workers", type=int, default=1)
    g.add_argument("--chunk", type=int, default=250)
    g.add_argument("--val-pct", type=float, default=10.0)
    g = sub.add_parser("weights")
    g.add_argument("--out", required=True)
    g.add_argument("--plan", required=True)
    g.add_argument("--marginals", required=True)
    g.add_argument("--shrink", type=float, default=100.0)
    g = sub.add_parser("generic")
    g.add_argument("--out", required=True)
    g.add_argument("--n-stays", type=int, default=5000)
    g.add_argument("--n-channels", type=int, default=24)
    g.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("index")
    g.add_argument("--store", nargs="+", required=True)
    g = sub.add_parser("marginals")
    g.add_argument("--tokens", required=True)
    g.add_argument("--split", default="train")
    g.add_argument("--datasets", nargs="+", default=list(DATASETS))
    g.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    {"bank": cmd_bank, "tokens": cmd_tokens, "meds": cmd_meds, "stats": cmd_stats, "supply": cmd_supply,
     "weights": cmd_weights, "marginals": cmd_marginals, "index": cmd_index,
     "generic": cmd_generic}[a.cmd](a)


if __name__ == "__main__":
    main()
