"""icutwin on the AutoDL host: round 3's screen of the candidates against M32_main in compact text (aggregates only;
docs/m7_refinement.md, amendment 1): tuning NLL; P1 membership AUC; S4, the masked teacher-forced onset calibration
after 24 h pooled over the datasets (and its cells within 0.85-1.15), with the pooled O/E by ICU day; and from the
twin screen (300 per dataset, length-resampled): S1 treatment rates, runaway share, length and top-event tails,
the first-24-hour unigram AUC.
    python round3_screen.py [runs...]      (default M32_main MC32_main MCN32_main)"""
import json
import sys
from pathlib import Path

R = Path("/root/autodl-tmp/icutwin/runs")
RUNS = sys.argv[1:] or ["M32_main", "MC32_main", "MCN32_main"]
DS = ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
TR = ["imv", "vasopressor", "rrt"]
DAYS = ["1", "2", "3-7", "8+"]
f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"  # noqa: E731


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


print("TUNING NLL (final) and P1 membership AUC per dataset")
for run in RUNS:
    m = R / "train/full" / run / "metrics.jsonl"
    ev = [json.loads(x) for x in m.read_text().splitlines() if '"eval"' in x][-1]["eval"] if m.exists() else None
    p1 = load(R / "audit" / f"m7_{run}" / "p1.json")
    print(run.ljust(11), "NLL", f(ev["nll"] if ev else None, 4),
          "| P1", p1["status"] if p1 else "-", " ".join(f"{d[:4]} {p1['groups'][d]['auc'][0]:.3f}" for d in DS) if p1 else "")
print("\nS4: MASKED ONSET O/E AFTER 24 H, pooled (observed/expected) | cells within 0.85-1.15 | pooled O/E by ICU day")
for run in RUNS:
    c = load(R / "eval" / run / "calibration_masked.json")
    if not c:
        continue
    pooled, cells, byday = [], 0, []
    for k in TR:
        o = sum(c[d]["masked"][k]["after_24h"]["observed"] for d in DS if d in c)
        e = sum(c[d]["masked"][k]["after_24h"]["expected"] for d in DS if d in c)
        pooled.append(f"{k[:4]} {f(o / e if e else None, 2)} ({o:.0f}/{e:.0f})")
        cells += sum(0.85 <= (c[d]["masked"][k]["after_24h"]["o_e"] or 0) <= 1.15 for d in DS if d in c)
        days = []
        for dy in DAYS:
            o = sum(c[d]["masked"][k][dy]["observed"] for d in DS if d in c)
            e = sum(c[d]["masked"][k][dy]["expected"] for d in DS if d in c)
            days.append(f(o / e if e else None, 2))
        byday.append(f"{k[:4]} " + "/".join(days))
    print(run.ljust(11), " | ".join(pooled), f"| {cells} of {3 * len([d for d in DS if d in c])}", "|", "  ".join(byday))
print("\nTWINS (length-resampled): S1 new onset after 24 h, twins/real per dataset (positives) | runaways | tails")
for run in RUNS:
    d = R / "eval" / run / ("auth_d" if run == "M32_main" else "screen")
    q, a, x = load(d / "quality.json"), load(d / "audit.json"), load(d / "explain.json")
    if not q:
        continue
    print(f"{run} ({d.name})")
    for k in TR:
        cells = []
        for ds in DS:
            p = q["datasets"].get(ds, {}).get("prevalence")
            if p:
                tw, re = p["twins"][k], p["real"][k]
                cells.append(f"{ds[:4]} {f(tw['rate'], 3)}/{f(re['rate'], 3)}({tw['positives']}/{re['positives']})")
        pp = q.get("pooled_prevalence", {})
        pooled = f"pooled {f(pp['twins'][k]['rate'], 3)}/{f(pp['real'][k]['rate'], 3)}" if pp else ""
        print("   ", k[:4].ljust(5), " ".join(cells), "|", pooled)
    run_share = [f"{ds[:4]} {f(q['datasets'][ds]['prevalence']['twins']['no_icu_discharge'], 3)}" for ds in DS
                 if ds in q["datasets"]]
    print("    no ICU discharge (twins):", " ".join(run_share))
    if a:
        tails = []
        for ds in DS:
            if ds in a:
                tr, tt = a[ds]["twins"]["tails"]["real"], a[ds]["twins"]["tails"]["twins"]
                tails.append(f"{ds[:4]} {tt['events_p99'] / tr['events_p99']:.2f}/{tt['top_event_p99'] / tr['top_event_p99']:.2f}")
        print("    events p99 / top-event p99 ratios:", " ".join(tails))
    if x:
        u = [x[ds]["first_24h"]["unigram_auc"] for ds in DS if ds in x]
        print("    first-24-h unigram AUC: mean", f(sum(u) / len(u)), " ".join(f"{ds[:4]} {x[ds]['first_24h']['unigram_auc']:.3f}"
                                                                      for ds in DS if ds in x))
