"""icutwin on the AutoDL host: every bank criterion (A1-A7, docs/m7_authenticity.md) of the given audits with its
value and its distance from the threshold (aggregates only), so that near misses can be read.
    python target_margins.py <audit dir> [<audit dir> ...]"""
import json
import sys
from pathlib import Path

DS = ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
FAM = ["vitals", "labs", "medications", "respiratory", "timing"]


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


for d in map(Path, sys.argv[1:]):
    a, q = load(d / "audit.json"), load(d / "quality.json")
    print(f"== {d.parent.name}/{d.name}")
    for ds in DS:
        if not a or ds not in a:
            continue
        t = a[ds]["twins"]
        vw, tr, tt = t["views"], t["tails"]["real"], t["tails"]["twins"]
        r = {k: tt[k] / tr[k] for k in ("events_p90", "events_p99", "top_event_p99") if tr[k]}
        fam = max(vw[f] for f in FAM if vw.get(f) is not None)
        miss = []
        if vw["first_24h"] > 0.70:
            miss.append(f"A1 AUC {vw['first_24h']:.3f} > 0.70 (+{vw['first_24h'] - 0.70:.3f})")
        if vw["length_matched"] > 0.80:
            miss.append(f"A2 AUC {vw['length_matched']:.3f} > 0.80")
        if fam > 0.80:
            miss.append(f"A3 family AUC {fam:.3f} > 0.80")
        for k in ("events_p90", "events_p99"):
            if not 0.80 <= r[k] <= 1.20:
                edge = 1.20 if r[k] > 1.20 else 0.80
                miss.append(f"A4 {k} ratio {r[k]:.3f} outside 0.80-1.20 (by {abs(r[k] - edge):.3f})")
        if not 0.75 <= r["top_event_p99"] <= 1.25:
            edge = 1.25 if r["top_event_p99"] > 1.25 else 0.75
            miss.append(f"A5 top-event p99 ratio {r['top_event_p99']:.3f} outside 0.75-1.25 (by {abs(r['top_event_p99'] - edge):.3f})")
        print(f"  {ds:8s} " + ("; ".join(miss) if miss else "all met"))
    if q:
        pos = q["pooled"]["positive_control"]["diversity"]
        cand = q["pooled"]["candidates"]["model"]
        dv = cand["metrics"]["diversity"]
        need = pos["rare_code_recall"] - 0.03
        print(f"  pooled   A6 Vendi ratio {dv['vendi_ratio']:.3f} (>= 0.95, short {0.95 - dv['vendi_ratio']:+.3f}); "
              f"rare-code recall {dv['rare_code_recall']:.3f} (>= {need:.3f}, short {need - dv['rare_code_recall']:+.3f})")
        flags = [f for f in cand["flags"] if f not in ("discriminator_auc", "length_p99_logratio", "vendi_ratio",
                                                       "vendi_ratio_by_dataset_min")]
        bands = cand.get("bands") or cand.get("checks") or {}
        print(f"  pooled   A7 harness flags beyond the exempt ones: {flags}; "
              f"{ {f: bands.get(f) for f in flags} if bands else ''}")
        util = q.get("utility", {}).get("sets", {}).get("model", {})
        print("  pooled   A7 utility " + ", ".join(f"{k} {x['utility_ratio']:.3f}" for k, x in util.items())
              + " (each >= 0.90)")
