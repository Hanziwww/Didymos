"""icutwin on the AutoDL host: what tells the pilot's samples from real held-out stays, as aggregates only (no stay,
no sample): per dataset, tokens per stay by token family, the interval mix, the tokens most over- and
under-represented (frequency per stay, synthetic over real) and the discriminator's strongest unigram features.
Run while the pilot's samples exist (they are deleted with the pilot's report).

    .venv/bin/python tools/pilot_diagnose.py <tokens root> <pilot samples dir> [datasets...]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from icutwin.eval import privacy as pv
from icutwin.tokenization.vocab import Vocab

tokens, samples = Path(sys.argv[1]), Path(sys.argv[2])
datasets = sys.argv[3:] or ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
v = Vocab.load(tokens / "vocab.json")
fam = []
for m in v.meta:
    c = m.get("cls")
    fam.append(f"{c}:{(m.get('code') or '').split('//')[0]}" if c in ("event", "end") else c)
rng = np.random.default_rng(0)
out = {}
for ds in datasets:
    f = samples / f"{ds}.parquet"
    if not f.exists():
        continue
    syn = [np.asarray(x) for x in pq.read_table(f).column("ids").to_pylist()]
    held = pv.first_windows(tokens, ds, "held_out")
    held = [held[i] for i in rng.choice(len(held), size=min(len(syn), len(held)), replace=False)]
    rep = {}
    for name, seqs in (("real", held), ("synthetic", syn)):
        fc = Counter(fam[t] for s in seqs for t in s)
        top = []                                  # per stay: its most repeated event token's count and share
        for s in seqs:
            ev = [t for t in s if fam[t].startswith("event:")]
            n = Counter(ev).most_common(1)[0][1] if ev else 0
            top.append((n, n / max(len(ev), 1)))
        top = np.array(top)
        rep[name] = {"tokens_per_stay": float(np.mean([len(s) for s in seqs])),
                     "tokens_p50_p90_p99": [float(np.percentile([len(s) for s in seqs], q)) for q in (50, 90, 99)],
                     "top_event_count_p50_p99": [float(np.percentile(top[:, 0], q)) for q in (50, 99)],
                     "top_event_share_p50_p99": [round(float(np.percentile(top[:, 1], q)), 3) for q in (50, 99)],
                     "stays_with_an_event_over_200": int((top[:, 0] > 200).sum()),
                     "families_per_stay": {k: round(n / len(seqs), 1) for k, n in fc.most_common(14)},
                     "intervals": {v.tokens[t]: round(n / len(seqs), 2) for t, n in sorted(
                         Counter(t for s in seqs for t in s if v.meta[t].get("cls") == "interval").items())}}
    cr = Counter(t for s in held for t in s)
    cs = Counter(t for s in syn for t in s)
    ratio = []
    for t in set(cr) | set(cs):
        a, b = cr.get(t, 0), cs.get(t, 0)
        if a + b >= 200:
            ratio.append((np.log((b + 1) / (a + 1)), v.tokens[t], a, b))
    ratio.sort()
    rep["over"] = [(tok, a, b) for _r, tok, a, b in ratio[-12:][::-1]]
    rep["under"] = [(tok, a, b) for _r, tok, a, b in ratio[:12]]
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import LogisticRegression

    from icutwin.eval.fidelity import _matrix

    x = _matrix(held + syn, len(v.tokens))[:, :len(v.tokens)]
    y = np.r_[np.zeros(len(held)), np.ones(len(syn))]
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(TfidfTransformer(sublinear_tf=True).fit_transform(x), y)
    w = clf.coef_[0]
    top = np.argsort(w)
    rep["discriminator_synthetic_side"] = [(v.tokens[i], round(float(w[i]), 2)) for i in top[-12:][::-1]]
    rep["discriminator_real_side"] = [(v.tokens[i], round(float(w[i]), 2)) for i in top[:12]]
    out[ds] = rep
    print(ds, json.dumps(rep), flush=True)
