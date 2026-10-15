"""icutwin on the AutoDL host: per-dataset masked onset O/E after 24 h (observed/expected) and the pooled O/E by ICU
day, for result sets named run[:tag] (files runs/eval/<run>/calibration_masked[_<tag>]_<dataset>.json).
    python cmp_calm.py M32_main M32_main:train MC32_main"""
import json
import sys
from pathlib import Path

R = Path("/root/autodl-tmp/icutwin/runs/eval")
DS = ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
OUT = ("imv", "vasopressor", "rrt", "icu_discharge", "icu_death")
DAYS = ("1", "2", "3-7", "8+")
sets = sys.argv[1:] or ["M32_main", "MC32_main"]


def load(name, ds):
    run, _, tag = name.partition(":")
    p = R / run / (f"calibration_masked_{tag}_{ds}.json" if tag else f"calibration_masked_{ds}.json")
    return json.loads(p.read_text())[ds]["masked"] if p.exists() else None


for ds in DS:
    for name in sets:
        c = load(name, ds)
        if c is None:
            continue
        row = []
        for k in OUT:
            a = c[k]["after_24h"]
            oe = a["o_e"]
            row.append(f"{k[:4]} {oe:.2f} ({a['observed']:.0f}/{a['expected']:.0f})" if oe is not None else f"{k[:4]} -")
        print(ds.ljust(8), name.ljust(16), "  ".join(row))
print()
for name in sets:
    tot = {k: [0.0, 0.0] for k in OUT}
    day = {k: [[0.0, 0.0] for _ in DAYS] for k in OUT}
    for ds in DS:
        c = load(name, ds)
        if c is None:
            continue
        for k in OUT:
            tot[k][0] += c[k]["after_24h"]["observed"]
            tot[k][1] += c[k]["after_24h"]["expected"]
            for j, d in enumerate(DAYS):
                day[k][j][0] += c[k][d]["observed"]
                day[k][j][1] += c[k][d]["expected"]
    print("pooled", name.ljust(16), "  ".join(f"{k[:4]} {o / e:.3f} ({o:.0f}/{e:.0f})" for k, (o, e) in tot.items() if e))
    print("  by day", " ".join(f"{k[:4]} " + "/".join(f"{o / e:.2f}" if e else "-" for o, e in day[k])
                               for k in OUT[:3]))
