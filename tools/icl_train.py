"""TrajPFN's learning confirmation (plan Phase G, G3-G4): evaluation sets and training runs.

    # frozen evaluation episodes: twin validation, twin label-shift, real dev
    python tools/icl_train.py make-set --store data/factory/stores/bank_dev --pool data/factory/pools/bank_dev \\
        --out data/icl/sets/twin_val --n 2000 --split val --seed 1001
    python tools/icl_train.py make-set --store data/factory/stores/real_dev --pool data/factory/pools/bank_dev \\
        --stats-from data/factory/stores/bank_dev --out data/icl/sets/real_dev --n 2000 --seed 1002
    # a run (resumes from its checkpoint)
    python tools/icl_train.py train --run-dir runs/icl/S_pilot --preset S --episodes 200000 \\
        --eval-set data/icl/sets/twin_val

Runs, checkpoints and sets stay under ``D:\\ulife`` (gitignored; DUA-derived, internal).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import typing


def main(argv: list[str] | None = None) -> None:
    from icutwin.icl import train as T

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("make-set")
    g.add_argument("--store", required=True)
    g.add_argument("--pool", default="data/factory/pools/bank_dev")
    g.add_argument("--vocab", default="data/tokens/full_r4/vocab.json")
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=2000)
    g.add_argument("--seed", type=int, default=1001)
    g.add_argument("--split", default=None)
    g.add_argument("--stage", type=int, default=1)
    g.add_argument("--stats-from", default=None)
    t = sub.add_parser("train")
    defaults = {"store": "data/factory/stores/bank_dev", "pool": "data/factory/pools/bank_dev",
                "vocab": "data/tokens/full_r4/vocab.json"}
    hints = typing.get_type_hints(T.TrainConfig)
    for f in dataclasses.fields(T.TrainConfig):
        name = "--" + f.name.replace("_", "-")
        if f.name == "model_overrides":
            t.add_argument(name, type=json.loads, default={})
            continue
        typ = hints[f.name]
        base = typing.get_args(typ)[0] if typing.get_origin(typ) in (typing.Union, getattr(__import__("types"),
                                                                                             "UnionType", None)) else typ
        default = defaults.get(f.name, f.default if f.default is not dataclasses.MISSING else None)
        if base is bool:
            t.add_argument(name, type=lambda s: s.lower() in ("1", "true", "yes"), default=default)
        else:
            t.add_argument(name, type=base if base in (int, float, str) else str, default=default,
                           required=default is None and f.name == "run_dir")
    a = ap.parse_args(argv)
    if a.cmd == "make-set":
        man = T.make_set(a.store, a.pool, a.vocab, a.out, a.n, a.seed, a.split, a.stage, a.stats_from)
        print(json.dumps({k: v for k, v in man.items() if k != "shards"}))
        return
    cfg = T.TrainConfig(**{f.name: getattr(a, f.name) for f in dataclasses.fields(T.TrainConfig)})
    print(json.dumps(T.train(cfg)))


if __name__ == "__main__":
    main()
