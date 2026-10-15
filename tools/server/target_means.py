"""icutwin on the AutoDL host: the bank criteria (A1-A7) on the mean of two audit replicates, each criterion's value
against its threshold and the relative distance (aggregates only).
    python target_means.py <audit dir 0> <audit dir 1>"""
import json
import sys
from pathlib import Path

import numpy as np

DS = ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
FAM = ["vitals", "labs", "medications", "respiratory", "timing"]
dirs = [Path(x) for x in sys.argv[1:3]]
audits = [json.loads((d / "audit.json").read_text()) for d in dirs]
quals = [json.loads((d / "quality.json").read_text()) for d in dirs]


def row(name, vals, lo=None, hi=None):
    m = float(np.mean(vals))
    ok = (lo is None or m >= lo) and (hi is None or m <= hi)
    edge = hi if hi is not None and m > hi else lo if lo is not None and m < lo else None
    rel = "" if ok else f" (off by {abs(m - edge):.3f}, {abs(m - edge) / abs(edge):.1%} of the threshold)"
    band = f">= {lo}" if hi is None else f"<= {hi}" if lo is None else f"{lo}-{hi}"
    print(f"  {name:38s} {' / '.join(f'{x:.3f}' for x in vals)}  mean {m:.3f}  [{band}] {'met' if ok else 'MISSED'}{rel}")


for ds in DS:
    vs = [a[ds]["twins"] for a in audits]
    r = lambda t, k: t["tails"]["twins"][k] / t["tails"]["real"][k]  # noqa: E731
    print(ds)
    row("A1 first-24h AUC", [t["views"]["first_24h"] for t in vs], hi=0.70)
    row("A2 length-matched AUC", [t["views"]["length_matched"] for t in vs], hi=0.80)
    row("A3 max family AUC", [max(t["views"][f] for f in FAM if t["views"].get(f) is not None) for t in vs], hi=0.80)
    row("A4 events p90 ratio", [r(t, "events_p90") for t in vs], 0.80, 1.20)
    row("A4 events p99 ratio", [r(t, "events_p99") for t in vs], 0.80, 1.20)
    row("A5 top-event p99 ratio", [r(t, "top_event_p99") for t in vs], 0.75, 1.25)
print("pooled")
dv = [q["pooled"]["candidates"]["model"]["metrics"] for q in quals]
pc = [q["pooled"]["positive_control"] for q in quals]
row("A6 Vendi ratio", [d["diversity"]["vendi_ratio"] for d in dv], lo=0.95)
need = float(np.mean([p["diversity"]["rare_code_recall"] - 0.03 for p in pc]))
row("A6 rare-code recall", [d["diversity"]["rare_code_recall"] for d in dv], lo=round(need, 3))
need = float(np.mean([p["density"] - 0.05 for p in pc]))
row("A7 harness density", [d["density"] for d in dv], lo=round(need, 3))
for task in quals[0]["utility"]["sets"]["model"]:
    row(f"A7 utility {task}", [q["utility"]["sets"]["model"][task]["utility_ratio"] for q in quals], lo=0.90)
