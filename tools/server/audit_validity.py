"""icutwin on the AutoDL host: why the M7 audit samples of a run are grammar-invalid, as aggregates only (the protocol
reports the samples' validity; this breaks it down by the automaton's rule and by the position of the first
violation). Reads runs/audit/m7_<run>/audit_samples before the purge.

    .venv/bin/python tools/audit_validity.py <run> <tokens root>
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from icutwin.tokenization.fsm import Automaton, compile_tables
from icutwin.tokenization.vocab import Vocab

run, tokens = sys.argv[1], Path(sys.argv[2])
d = Path(f"/root/autodl-tmp/icutwin/runs/audit/m7_{run}/audit_samples")
v = Vocab.load(tokens / "vocab.json")
t = compile_tables(v)
eos = v.id("[EOS]")
rules, first_pos, lengths, ended, cls_at = Counter(), [], [], 0, Counter()
n = 0
for f in sorted(d.glob("shard_*.parquet")):
    col = pq.read_table(f).column("ids").to_pylist()
    for ids in col:
        n += 1
        lengths.append(len(ids))
        ended += int(ids[-1] == eos)
        a = Automaton(t)
        bad = None
        for pos, tok in enumerate(ids):
            if not a.step(tok):
                bad = (pos, tok, a.violation)
                break
        if bad is None:
            rules["valid" if a.complete() else "incomplete (no end block before the cap)"] += 1
            continue
        pos, tok, why = bad
        rules[re.sub(r"\d+", "#", str(why).split(":")[0])[:80]] += 1
        first_pos.append(pos / max(len(ids), 1))
        m = v.meta[tok]
        cls_at[f"{m.get('cls')}:{(m.get('code') or v.tokens[tok]).split('//')[0]}"] += 1
q = lambda x: [round(float(np.quantile(x, p)), 3) for p in (0.1, 0.5, 0.9)] if len(x) else None  # noqa: E731
print(json.dumps({"samples": n, "ended_with_eos": ended, "length_p10_p50_p90": q(lengths),
                  "first_violation": dict(rules.most_common()),
                  "violation_position_share_of_length_p10_p50_p90": q(first_pos),
                  "violating_token_class": dict(cls_at.most_common(15))}, indent=1))
