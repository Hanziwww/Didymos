"""icutwin on the AutoDL host: the tuning NLL a run's metrics.jsonl records at one evaluation step (nothing if not
there yet).
    python eval_nll_at.py <metrics.jsonl> <step>"""
import json
import sys
from pathlib import Path

p, step = Path(sys.argv[1]), int(sys.argv[2])
if p.exists():
    for line in p.read_text().splitlines():
        if '"eval"' in line:
            r = json.loads(line)
            if r.get("step") == step:
                print(r["eval"]["nll"])
                break
