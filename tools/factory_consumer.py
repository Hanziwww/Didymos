"""The reference consumer on the host's episodes (plan Phase F, S9): ``--n`` episodes of a store and its pool through
``factory.consumer.Consumer`` (randomly initialized, never trained): a forward pass and the masked loss with its
gradients per episode, every output finite, the time per episode by task type. It proves the format is consumable at
production sizes; it is not the architecture. Aggregates only.

    python tools/factory_consumer.py --store data/factory/stores/bank_dev --pool data/factory/pools/bank_dev \\
        --n 200 --device cuda --out runs/factory/consumer.json
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
    ap.add_argument("--pool", default=None)
    ap.add_argument("--vocab", default="data/tokens/full_r4/vocab.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import numpy as np
    import torch

    from icutwin.factory import episodes, pool
    from icutwin.factory import store as fst
    from icutwin.factory.consumer import Consumer
    from icutwin.tokenization.vocab import Vocab

    st = fst.Store(Path(a.store))
    progs, weights, support = pool.load(Path(a.pool)) if a.pool else (None, None, None)
    s = episodes.Sampler(st, Vocab.load(Path(a.vocab)), stage=a.stage, seed=a.seed, programs=progs, weights=weights,
                         targets=pool.load_targets(), support=support)
    m = Consumer(len(st.channels)).to(a.device)
    by: dict = {}
    bad = 0
    for i in range(a.n):
        try:
            ep = s.episode(i)
        except episodes.NoEpisode:
            continue
        t0 = time.time()
        outs = m(ep)
        loss = m.loss(ep, outs)
        m.zero_grad()
        loss.backward()
        sec = time.time() - t0
        ok = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(o["out"]).all()) for o in outs) and all(
            bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None)
        bad += not ok
        r = by.setdefault(ep["task"]["label"], {"episodes": 0, "seconds": [], "events": [], "non_finite": 0})
        r["non_finite"] += not ok
        r["episodes"] += 1
        r["seconds"].append(sec)
        r["events"].append(int(len(ep["context"]["ev_t"]) + len(ep["query"]["ev_t"])))
    rep = {"device": a.device, "stage": a.stage, "episodes": sum(r["episodes"] for r in by.values()),
           "non_finite": bad, "parameters": int(sum(p.numel() for p in m.parameters())),
           "by_label": {k: {"episodes": r["episodes"], "non_finite": r["non_finite"],
                            "seconds_p50": round(float(np.median(r["seconds"])), 3),
                            "events_p50": int(np.median(r["events"]))} for k, r in by.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
