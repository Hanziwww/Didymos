"""icutwin on the AutoDL host: the M7 P1 membership attack of candidates side by side (aggregates):
    python p1_compare.py <run> <run> ...      (runs/audit/m7_<run>/p1.json)"""
import json
import sys
from pathlib import Path

A = Path("/root/autodl-tmp/icutwin/runs/audit")
runs = {r: json.loads((A / f"m7_{r}" / "p1.json").read_text()) for r in sys.argv[1:]}
w = max(map(len, runs))
groups = list(next(iter(runs.values()))["groups"])
print("group".ljust(8), "run".ljust(w), "AUC [95% CI]".ljust(24), "TPR@1%FPR".ljust(9), "NLL members/non", "status")
for g in groups:
    for r, p in runs.items():
        x = p["groups"].get(g)
        if x:
            a, t = x["auc"], x["tpr_at_1pct_fpr"]
            print(g.ljust(8), r.ljust(w), f"{a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]".ljust(24), f"{t[0]:.3f}".ljust(9),
                  f"{x['mean_nll_members']:.3f}/{x['mean_nll_nonmembers']:.3f}".ljust(15), x["status"])
for r, p in runs.items():
    print(r, "P1", p["status"])
