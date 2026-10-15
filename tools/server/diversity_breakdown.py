"""icutwin on the AutoDL host: the diversity and density measures of audits per dataset against their positive
control (real against real), to see where the pooled A6 / A7 density gaps come from (aggregates only).
    python diversity_breakdown.py <audit dir> [...]"""
import json
import sys
from pathlib import Path

for d in map(Path, sys.argv[1:]):
    q = json.loads((d / "quality.json").read_text())
    print(f"== {d.parent.name}/{d.name}   (twins / positive control)")
    rows = list(q["datasets"].items()) + [("pooled", q["pooled"])]
    for ds, r in rows:
        m, pc = r["candidates"]["model"]["metrics"], r["positive_control"]
        dv, pd = m["diversity"], pc.get("diversity", {})
        print(f"  {ds:8s} vendi ratio {dv.get('vendi_ratio', float('nan')):.3f}  "
              f"rare-code recall {dv.get('rare_code_recall', float('nan')):.3f}/{pd.get('rare_code_recall', float('nan')):.3f}  "
              f"density {m['density']:.3f}/{pc['density']:.3f}  coverage {m['coverage']:.3f}/{pc['coverage']:.3f}  "
              f"precision {m['precision']:.3f}/{pc['precision']:.3f}  recall {m['recall']:.3f}/{pc['recall']:.3f}  "
              f"duplicates {dv.get('duplicate_rate', float('nan')):.3f}")
