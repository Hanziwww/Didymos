"""icutwin on the AutoDL host: tuning NLL of training runs at their shared evaluation steps (from metrics.jsonl),
overall and per dataset:
    python nll_compare.py <run> <run> ...      (names under runs/train/full)"""
import json
import sys
from pathlib import Path

R = Path("/root/autodl-tmp/icutwin/runs/train/full")
DS = ["aumc", "eicu", "hirid", "inspire", "mimic", "nwicu", "sicdb"]
evals = {}
for run in sys.argv[1:]:
    rows = [json.loads(x) for x in (R / run / "metrics.jsonl").read_text().splitlines() if '"eval"' in x]
    evals[run] = {r["step"]: r["eval"] for r in rows}
w = max(map(len, evals))
print("step".rjust(5), "run".ljust(w), "all".rjust(6), *(d[:6].rjust(6) for d in DS))
for step in sorted(set().union(*(e.keys() for e in evals.values()))):
    for run, e in evals.items():
        if step in e:
            x = e[step]
            print(str(step).rjust(5), run.ljust(w), f"{x['nll']:.3f}".rjust(6),
                  *(f"{x['by_dataset'][d]:.3f}".rjust(6) for d in DS))
