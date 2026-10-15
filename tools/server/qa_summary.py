"""icutwin on the AutoDL host: round 3's treatment QA of M32_main in compact text (aggregates only): per dataset and
treatment, observed over expected first events by position of the predicted token, and the set's tokens carrying
the most expected first events (expected / observed)."""
import json
from pathlib import Path

q = json.loads(Path("/root/autodl-tmp/icutwin/runs/eval/M32_main/calibration_qa.json").read_text())
for ds, x in q.items():
    for k in ("imv", "vasopressor", "rrt"):
        pos = {n: (None if v["o_e"] is None else round(v["o_e"], 2), int(v["observed"]), round(v["expected"], 1))
               for n, v in x[k]["by_position"].items()}
        print(ds, k, "| by position (O/E, observed, expected):", pos)
        top = x[k]["tokens"][:5]
        print("    top tokens:", "; ".join(f"{y['token'].split('//', 1)[1][:48]} {y['expected']:.1f}/{y['observed']}"
                                           for y in top), "| first events on other tokens:",
              x[k]["observed_tokens_not_listed"])
