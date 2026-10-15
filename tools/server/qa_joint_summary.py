"""icutwin on the AutoDL host: does the treatment-onset excess follow time in the ICU or the context's length?
Per dataset and treatment, observed over expected first events by ICU day (rows) and position of the predicted token
(columns), observed/expected in brackets (M32_main, runs/eval/M32_main/calibration_qa2.json); then M_main (8k windows
with state blocks) against M32_main by ICU day (raw calibration). Aggregates only."""
import json
from pathlib import Path

E = Path("/root/autodl-tmp/icutwin/runs/eval")
DS = ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
POS = ["0-2k", "2k-8k", "8k-16k", "16k+"]
DAYS = ["1", "2", "3-7", "8+"]


def cell(c):
    if c["expected"] < 0.5 and c["observed"] == 0:
        return "".rjust(17)
    oe = "-" if c["o_e"] is None else f"{c['o_e']:.2f}"
    return f"{oe} ({c['observed']:.0f}/{c['expected']:.1f})".rjust(17)


q = json.loads((E / "M32_main" / "calibration_qa2.json").read_text())
for ds in DS:
    if ds not in q:
        continue
    for k in ("imv", "vasopressor", "rrt"):
        t = q[ds][k]["by_day_position"]
        print(f"{ds} {k}".ljust(20), "".join(p.rjust(17) for p in POS))
        for d in DAYS:
            print(f"   day {d}".ljust(20), "".join(cell(t[d][p]) for p in POS))
m32 = json.loads((E / "M32_main" / "calibration.json").read_text())
mm = E / "M_main" / "calibration.json"
if mm.exists():
    m8 = json.loads(mm.read_text())
    print("\nBY ICU DAY, raw O/E (1, 2, 3-7, 8+): M_main (8k windows) | M32_main (32k)")
    for ds in DS:
        if ds in m8:
            print(ds.ljust(8), "  ".join(
                k[:5] + " " + "/".join("-" if m8[ds][k][d]["o_e"] is None else f"{m8[ds][k][d]['o_e']:.2f}" for d in DAYS)
                + " | " + "/".join("-" if m32[ds][k][d]["o_e"] is None else f"{m32[ds][k][d]['o_e']:.2f}" for d in DAYS)
                for k in ("imv", "vasopressor", "rrt")))
