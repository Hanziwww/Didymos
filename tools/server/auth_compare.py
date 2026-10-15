"""icutwin on the AutoDL host: two authenticity audits side by side in compact text (aggregates only), e.g. round 2's
canonical-header diagnosis against the remediated run (docs/m7_authenticity.md, step 4):
    python auth_compare.py <label>=<dir>[:<suffix>] ...
reads <dir>/audit<suffix>.json, targets<suffix>.json and quality<suffix>.json (each optional)."""
import json
import sys
from pathlib import Path

DS = ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
FAM = ["vitals", "labs", "medications", "respiratory", "timing"]
f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"  # noqa: E731


def load(spec):
    label, rest = spec.split("=", 1)
    d, _, suf = rest.partition(":")
    r = lambda n: json.loads((Path(d) / f"{n}{suf}.json").read_text()) if (Path(d) / f"{n}{suf}.json").exists() else None  # noqa: E731
    return label, r("audit"), r("targets"), r("quality")


runs = [load(s) for s in sys.argv[1:]]
w = max(len(x[0]) for x in runs)
print("VIEWS: first 24h (A1) | length-matched (A2) | whole | max family (A3) | order")
for ds in DS + ["pooled"]:
    for label, a, _, _ in runs:
        if a and ds in a:
            v = a[ds]["twins"]["views"]
            fam = max((v[k] for k in FAM if v.get(k) is not None), default=None)
            print(ds.ljust(8), label.ljust(w), *(f(v.get(k)).rjust(6) for k in ("first_24h", "length_matched", "whole")),
                  f(fam).rjust(6), f(v.get("order")).rjust(6))
print("\nTAILS, twins/real: events p50 p90 p99 (A4 p90, p99) | top event p99 (A5) | discharge hazard d1 d2 d3 real/twins")
for ds in DS:
    for label, a, _, _ in runs:
        if a and ds in a:
            t = a[ds]["twins"]
            tr, tt = t["tails"]["real"], t["tails"]["twins"]
            q = lambda k: tt[k] / tr[k] if tr[k] else None  # noqa: E731
            hr, ht = t["hazard"]["real"], t["hazard"]["twins"]
            print(ds.ljust(8), label.ljust(w), *(f(q(k), 2).rjust(5) for k in ("events_p50", "events_p90", "events_p99")),
                  "|", f(q("top_event_p99"), 2).rjust(5), "|", " ".join(f"{f(hr[i], 2)}/{f(ht[i], 2)}" for i in range(3)))
print("\nPOOLED HARNESS: vendi ratio | rare-code recall (control) | flags | utility ratios | grammar")
for label, _, _, qu in runs:
    if qu:
        c, pos = qu["pooled"]["candidates"]["model"], qu["pooled"]["positive_control"]
        d = c["metrics"]["diversity"]
        util = {t: round(x["utility_ratio"], 3) for t, x in qu.get("utility", {}).get("sets", {}).get("model", {}).items()}
        gram = min(r.get("grammar_valid", 0) for r in qu["datasets"].values())
        print(label.ljust(w), f(d["vendi_ratio"]), f(d["rare_code_recall"]), f"({f(pos['diversity']['rare_code_recall'])})",
              c["flags"], util, f(gram))
print("\nTARGETS (pass marks per dataset: A1 A2 A3 A4 A5)")
for ds in DS:
    for label, _, t, _ in runs:
        if t and ds in t:
            print(ds.ljust(8), label.ljust(w), " ".join(("A" + k[1] if t[ds][k] else "--") for k in ("A1", "A2", "A3", "A4", "A5")))
for label, _, t, _ in runs:
    if t and "pooled" in t:
        print("pooled".ljust(8), label.ljust(w), t["pooled"])
for label, _, t, _ in runs:       # the third run's adoption rule counts the per-dataset targets met
    if t:
        met = sum(bool(t[ds][k]) for ds in DS if ds in t for k in ("A1", "A2", "A3", "A4", "A5"))
        print(f"{label}: per-dataset targets met {met} of {5 * sum(ds in t for ds in DS)}")
