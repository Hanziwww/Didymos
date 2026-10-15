"""icutwin on the AutoDL host: round 2's diagnosis in compact text (aggregates only) from runs/eval/M32_main/auth:
the views before and after the canonical header order, drift, hazard, coherence, tails, the model checks and the
targets."""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/icutwin/runs/eval/M32_main/auth")
r = lambda n: json.loads((D / n).read_text()) if (D / n).exists() else None  # noqa: E731
base, canon, model = r("audit.json"), r("audit_canon.json"), r("model.json")
DS = ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
V = ["first_6h", "first_24h", "first_72h", "whole", "length_matched", "vitals", "labs", "medications", "respiratory",
     "timing", "order"]
f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
print("VIEWS (twins | canonical header | positive control)")
print("ds".ljust(9) + " ".join(k[:8].rjust(8) for k in V))
for ds in DS + ["pooled"]:
    for name, a in (("base", base), ("canon", canon)):
        if a and ds in a:
            print((ds[:5] + " " + name[:3]).ljust(9) + " ".join(f(a[ds]["twins"]["views"].get(k)).rjust(8) for k in V))
    if canon and ds in canon:
        print((ds[:5] + " pos").ljust(9) + " ".join(f(canon[ds]["positive_control"]["views"].get(k)).rjust(8) for k in V))
print("\nTAILS, DRIFT, HAZARD (canonical)")
for ds in DS:
    a = (canon or base)[ds]["twins"]
    tr, tt = a["tails"]["real"], a["tails"]["twins"]
    print(ds, "events p50/p90/p99 real", [round(tr[k]) for k in ("events_p50", "events_p90", "events_p99")],
          "twins", [round(tt[k]) for k in ("events_p50", "events_p90", "events_p99")],
          "| top-event p99 real", round(tr["top_event_p99"]), "twins", round(tt["top_event_p99"]))
    dr, dt = a["drift"]["real"]["repeat_share_by_hours"], a["drift"]["twins"]["repeat_share_by_hours"]
    print("   repeat share by hours real", {k: f(x) for k, x in dr.items()}, "twins", {k: f(x) for k, x in dt.items()})
    er, et = a["drift"]["real"]["events_per_hour_by_hours"], a["drift"]["twins"]["events_per_hour_by_hours"]
    print("   events/hour real", {k: f(x) for k, x in er.items()}, "twins", {k: f(x) for k, x in et.items()})
    hr, ht = a["hazard"]["real"], a["hazard"]["twins"]
    print("   discharge hazard d1-7 real", [f(x) for x in hr[:7]], "twins", [f(x) for x in ht[:7]],
          "| d14-21 real", [f(x) for x in hr[13:21]], "twins", [f(x) for x in ht[13:21]])
    cr, ct = a["coherence"]["real"], a["coherence"]["twins"]
    print("   lag-1 same bin", {c.split("//")[1][:10]: (f(cr[c]["same_bin"]), f(ct[c]["same_bin"])) for c in cr})
if model:
    print("\nMODEL CHECKS")
    for ds, m in model.items():
        print(ds, "NLL by position real", {k: f(x) for k, x in m["nll_by_position"]["real"].items()},
              "twins", {k: f(x) for k, x in m["nll_by_position"]["twins"].items()})
        print("   temperature", {k: round(x, 4) for k, x in m["temperature_nll"].items()})
        mm = m["masked_mass"]
        print("   masked mass", {k: (x["positions"], f(x["grammar"]), f(x["grammar_support"]), f(x["share_over_1pct"]))
                                 for k, x in mm.items()})
for name in ("targets.json", "targets_canon.json"):
    t = r(name)
    if t:
        print("\n" + name, json.dumps(t))
