"""icutwin on the AutoDL host: structure of goal-first twins on disk (aggregates only): token length quantiles,
the share with an ICU discharge, with an end block, the end block's length, and the share of tokens after the
ICU discharge; against the real held-out stays of the dataset.
    python gf_twin_stats.py <tokens root> <twins parquet> <dataset>"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from icutwin.eval import privacy as pv
from icutwin.eval.forecast import Reader
from icutwin.tokenization.vocab import Vocab

tokens, twins, ds = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
v = Vocab.load(tokens / "vocab.json")
rd = Reader(v)
end = v.id("[END]")


def stats(seqs, name):
    lens = np.array([len(s) for s in seqs])
    dis = [next((k for k, t in enumerate(s) if t in rd.dis), None) for s in seqs]
    ends = [list(s).index(end) if end in s else None for s in seqs]
    endlen = np.array([len(s) - e for s, e in zip(seqs, ends) if e is not None])
    after = np.array([(e - d) for d, e in zip(dis, ends) if d is not None and e is not None])
    hours = [rd.outcomes(s)["icu_los_hours"] for s in seqs]
    h = np.array([x for x in hours if x is not None])
    print(f"{name}: n {len(seqs)} length p50 {np.median(lens):.0f} p90 {np.quantile(lens, .9):.0f} "
          f"p99 {np.quantile(lens, .99):.0f} | ICU discharge {np.mean([d is not None for d in dis]):.3f} | "
          f"end block {np.mean([e is not None for e in ends]):.3f}, its length p50 {np.median(endlen):.0f} "
          f"p99 {np.quantile(endlen, .99):.0f} | tokens from discharge to [END] p50 {np.median(after):.0f} "
          f"p99 {np.quantile(after, .99):.0f} | LOS h p50 {np.median(h):.0f} p90 {np.quantile(h, .9):.0f}", flush=True)


syn = [np.asarray(x) for x in pq.read_table(twins).column("ids").to_pylist()]
stats(syn, "twins")
held = pv.first_windows(tokens, ds, "held_out")
rng = np.random.default_rng(0)
stats([held[i] for i in rng.choice(len(held), size=min(1000, len(held)), replace=False)], "real ")
