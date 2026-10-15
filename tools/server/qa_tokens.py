"""icutwin on the AutoDL host: the treatment QA's tokens (``calibration --qa``): per dataset and treatment, the tokens
carrying the expected first events after the admission (expected / observed first events by token).
    python qa_tokens.py <run> [top]"""
import json
import sys
from pathlib import Path

run = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 6
q = json.loads((Path("/root/autodl-tmp/icutwin/runs/eval") / run / "calibration_qa.json").read_text())
for ds, r in q.items():
    for k in ("imv", "vasopressor", "rrt"):
        toks = r[k]["tokens"][:top]
        e = sum(x["expected"] for x in r[k]["tokens"])
        o = sum(x["observed"] for x in r[k]["tokens"]) + r[k]["observed_tokens_not_listed"]
        print(f"{ds[:5]:5} {k[:4]:4} listed E {e:6.1f} all O {o:3d} |",
              " | ".join(f"{x['token'][-44:]} {x['expected']:.1f}/{x['observed']}" for x in toks))
