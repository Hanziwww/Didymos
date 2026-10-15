"""icutwin on the AutoDL host: round 3's tallies (docs/m7_refinement.md, "Targets and decision"; aggregates only).
Per label (a model, or a model under a sampler):
* the per-dataset A1-A5 targets met by each audit replicate and their mean, and the pooled A6/A7 parts each meets;
* S1: new-onset IMV, vasopressor and RRT after the 24-hour landmark, twins against the real rate within +-25% per
  dataset, the datasets with fewer than 30 real events pooled per treatment (one cell); per length-only audit;
* S2: rollout observed over expected within 0.8-1.25 for death, IMV, vasopressor, RRT and ICU stay over 7 days per
  dataset, the same pooling (observed events).
    python round3_tally.py --audits LABEL=DIR[,DIR] ... --s1 LABEL=DIR[,DIR] ... --s2 LABEL=FILE ..."""
import argparse
import json
from pathlib import Path

DS = ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
TR = ["imv", "vasopressor", "rrt"]
S2 = ["death", "imv", "vasopressor", "rrt", "icu_los_gt_7d"]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def within(x, ref, lo, hi):
    return ref > 0 and lo <= x / ref <= hi


def s1_cells(q: dict) -> tuple[int, int, list]:
    met = total = 0
    cells = []
    for k in TR:
        small = [0, 0, 0, 0]                       # twin positives, twin n, real positives, real n
        for ds in DS:
            p = q["datasets"].get(ds, {}).get("prevalence")
            if not p:
                continue
            tw, re = p["twins"][k], p["real"][k]
            if re["positives"] >= 30:
                ok = within(tw["rate"] or 0.0, re["rate"] or 0.0, 0.75, 1.25)
                met, total = met + ok, total + 1
                cells.append(f"{k[:4]}:{ds[:4]} {tw['rate'] / re['rate']:.2f}{'' if ok else '!'}")
            else:
                small = [small[0] + tw["positives"], small[1] + tw["n"], small[2] + re["positives"], small[3] + re["n"]]
        if small[3]:
            r_tw, r_re = small[0] / max(small[1], 1), small[2] / small[3]
            ok = within(r_tw, r_re, 0.75, 1.25)
            met, total = met + ok, total + 1
            cells.append(f"{k[:4]}:pooled({small[2]}) {r_tw / r_re if r_re else float('nan'):.2f}{'' if ok else '!'}")
    return met, total, cells


def s2_cells(f: dict) -> tuple[int, int, list]:
    met = total = 0
    cells = []
    for k in S2:
        small = [0.0, 0.0]                         # observed, expected events
        for ds in DS:
            r = f.get(ds, {}).get(k)
            if not r or r.get("prevalence") is None:
                continue
            obs, exp = r["prevalence"] * r["stays"], r["mean_predicted"] * r["stays"]
            if obs >= 30:
                ok = within(obs, exp, 0.8, 1.25)
                met, total = met + ok, total + 1
                cells.append(f"{k[:4]}:{ds[:4]} {obs / exp:.2f}{'' if ok else '!'}")
            else:
                small = [small[0] + obs, small[1] + exp]
        if small[1]:
            ok = within(small[0], small[1], 0.8, 1.25)
            met, total = met + ok, total + 1
            cells.append(f"{k[:4]}:pooled({small[0]:.0f}) {small[0] / small[1]:.2f}{'' if ok else '!'}")
    return met, total, cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audits", nargs="*", default=[])
    ap.add_argument("--s1", nargs="*", default=[])
    ap.add_argument("--s2", nargs="*", default=[])
    ap.add_argument("--cells", action="store_true", help="list every S1 and S2 cell")
    a = ap.parse_args()
    for spec in a.audits:
        label, dirs = spec.split("=", 1)
        counts = []
        for d in dirs.split(","):
            t = load(Path(d) / "targets.json")
            if t is None:
                print(f"{label}: {d} has no targets.json")
                continue
            n = sum(bool(t[ds][k]) for ds in DS if ds in t for k in ("A1", "A2", "A3", "A4", "A5"))
            counts.append(n)
            print(f"{label}: {Path(d).name} A1-A5 met {n} of {5 * sum(ds in t for ds in DS)}; pooled {t.get('pooled')}")
        if counts:
            print(f"{label}: mean A1-A5 count {sum(counts) / len(counts):.1f} over {len(counts)} replicate(s)")
    for spec in a.s1:
        label, dirs = spec.split("=", 1)
        res = []
        for d in dirs.split(","):
            q = load(Path(d) / "quality.json")
            if q is None:
                print(f"{label}: {d} has no quality.json")
                continue
            met, total, cells = s1_cells(q)
            res.append(met)
            print(f"{label}: {Path(d).name} S1 cells met {met} of {total}" + (f"  [{'; '.join(cells)}]" if a.cells else ""))
        if res:
            print(f"{label}: mean S1 cells met {sum(res) / len(res):.1f}")
    for spec in a.s2:
        label, path = spec.split("=", 1)
        f = load(Path(path))
        if f is None:
            print(f"{label}: no {path}")
            continue
        met, total, cells = s2_cells(f)
        print(f"{label}: S2 cells met {met} of {total}" + (f"  [{'; '.join(cells)}]" if a.cells else ""))


if __name__ == "__main__":
    main()
