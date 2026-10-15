"""icutwin on the AutoDL host: the tuning NLL at every evaluation of the given runs (metrics.jsonl).
    python nll_traj.py M32_main MC32_main ..."""
import json
import sys
from pathlib import Path

R = Path("/root/autodl-tmp/icutwin/runs/train/full")
for run in sys.argv[1:]:
    p = R / run / "metrics.jsonl"
    if not p.exists():
        print(run, "-")
        continue
    ev = [json.loads(x) for x in p.read_text().splitlines() if '"eval"' in x]
    print(run.ljust(12), " ".join(f"{e['step']}:{e['eval']['nll']:.4f}" for e in ev))
