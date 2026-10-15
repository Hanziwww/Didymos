"""icutwin on the AutoDL host: round 3's step-0 diagnosis of M32_main in compact text (aggregates only):
    python round3_diag_summary.py
0a teacher-forced hazard calibration (raw; under the generation mask), 0b rollout O/E by horizon, 0c per-dataset
treatment rates and runaway shares of twins against real stays, 0d checkpoint averages."""
import json
from pathlib import Path

E = Path("/root/autodl-tmp/icutwin/runs/eval/M32_main")
DS = ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
r = lambda n: json.loads((E / n).read_text()) if (E / n).exists() else None  # noqa: E731
f = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"  # noqa: E731

cal, calm = r("calibration.json"), r("calibration_masked.json")
OUT = ("imv", "vasopressor", "rrt", "icu_discharge", "icu_death")
print("0a TEACHER-FORCED O/E after 24 h (raw | masked [95% CI]); observed / expected")
for ds in DS:
    sources = [("raw", (cal or {}).get(ds)), ("masked", (calm or {}).get(ds, {}).get("masked"))]
    for name, rows in sources:
        if not rows:
            continue
        cells = []
        for k in OUT:
            x = rows[k]["after_24h"]
            ci = x.get("o_e_ci") or [None, None]
            cells.append(f"{k[:5]} {f(x['o_e'])} [{f(ci[0])},{f(ci[1])}] {x['observed']:.0f}/{x['expected']:.1f}")
        print(ds.ljust(8), name.ljust(6), " | ".join(cells))
if cal:
    print("   by ICU day, raw O/E (1, 2, 3-7, 8+):")
    for ds in DS:
        if ds in cal:
            print("  ", ds.ljust(8), "  ".join(f"{k[:5]} " + "/".join(f(cal[ds][k][d]["o_e"]) for d in ("1", "2", "3-7", "8+"))
                                               for k in OUT))

fc = r("forecast_h.json")
if fc:
    print("\n0b ROLLOUTS from real 24-h prefixes: O/E at 6 / 24 / 72 h / whole stay (real prevalence at 72 h)")
    for ds in DS:
        if ds not in fc:
            continue
        h = fc[ds].get("horizons", {})
        cells = []
        for k in ("imv", "vasopressor", "rrt", "icu_discharge", "icu_death"):
            oe = [h.get(str(x), {}).get(k, {}).get("observed_over_expected") for x in (6, 24, 72)]
            whole = fc[ds].get(k, {}).get("observed_over_expected") if k in fc[ds] else None
            prev = h.get("72", {}).get(k, {}).get("prevalence")
            cells.append(f"{k[:5]} {'/'.join(f(x) for x in oe)}/{f(whole)} ({f(prev)})")
        print(ds.ljust(8), " | ".join(cells))
        d = fc[ds]
        print("         whole stay: death", f(d["death"].get("observed_over_expected")), "los>7d",
              f(d["icu_los_gt_7d"].get("observed_over_expected")), "| LOS spread-skill",
              f(d["icu_los_hours"].get("spread_skill")), "censored", f(d["icu_los_hours"].get("rollouts_censored")))

q = r("auth_d/quality.json")
if q:
    print("\n0c TWINS (round-2 sampler, seed 1) against held-out: new onset after 24 h, rate twins / real (positives)")
    for ds in DS:
        p = q["datasets"].get(ds, {}).get("prevalence")
        if not p:
            continue
        cells = [f"{k} {f(p['twins'][k]['rate'], 3)}/{f(p['real'][k]['rate'], 3)} ({p['twins'][k]['positives']}/{p['real'][k]['positives']})"
                 for k in ("imv", "vasopressor", "rrt", "death", "icu_los_gt_7d")]
        print(ds.ljust(8), " | ".join(cells), "| no ICU discharge", f(p["twins"]["no_icu_discharge"], 3), "/",
              f(p["real"]["no_icu_discharge"], 3))
    t = r("auth_d/targets.json")
    if t:
        met = sum(bool(t[ds][k]) for ds in DS if ds in t for k in ("A1", "A2", "A3", "A4", "A5"))
        print("   audit (seed 1): per-dataset targets met", met, "of 35; pooled", t.get("pooled"))

avg = r("ckpt_average.json")
if avg:
    print("\n0d CHECKPOINT AVERAGES (tuning NLL):", {k: round(v["nll"], 4) for k, v in avg.items()})
