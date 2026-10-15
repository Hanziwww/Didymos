"""icutwin on the AutoDL host: round 3's pre-registered selection rules applied to their results (docs/m7_refinement.md;
aggregates only). Prints the reasoning and, last, one line for the chain script:
    python round3_select.py goalfirst            -> "GOALFIRST 1" or "GOALFIRST 0" (amendment 3)
    python round3_select.py finalist [runs...]   -> "FINALIST <run>" or "FINALIST none" (amendments 1 and 2)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from round3_tally import DS, TR, load, s1_cells  # noqa: E402

R = Path("/root/autodl-tmp/icutwin/runs")
REF = "M32_main"
S3_MAX = 1.571


def a15(t: dict) -> int:
    return sum(bool(t[ds][k]) for ds in DS if ds in t for k in ("A1", "A2", "A3", "A4", "A5"))


def goalfirst() -> None:
    e = R / "eval" / REF
    ref_t = [load(e / d / "targets.json") for d in ("auth_c", "auth_d")]
    gf_t = [load(e / d / "targets.json") for d in ("auth_g0", "auth_g1")]
    gf_q = [load(e / d / "quality.json") for d in ("auth_g0", "auth_g1")]
    if None in ref_t or None in gf_t or None in gf_q:
        print("missing audits: goal-first not adopted")
        print("GOALFIRST 0")
        return
    ref_a = sum(a15(t) for t in ref_t) / 2
    ref_s1 = s1_cells(load(e / "auth_d" / "quality.json"))[0]
    gf_a = sum(a15(t) for t in gf_t) / 2
    gf_s1 = sum(s1_cells(q)[0] for q in gf_q) / 2
    lost = [k for k, ok in ref_t[0]["pooled"].items() if ok and ref_t[1]["pooled"].get(k)
            and not all(t["pooled"].get(k) for t in gf_t)]
    for q, name in zip(gf_q, ("auth_g0", "auth_g1")):
        print(name, "goal agreement:", {ds: q["datasets"][ds].get("goals") for ds in DS if ds in q["datasets"]}
              if any("goals" in q["datasets"][ds] for ds in q["datasets"]) else "not recorded")
    print(f"authenticity: A1-A5 mean {gf_a:.1f} against {ref_a:.1f} (better >= 24.5, no worse >= 21.5)")
    print(f"treatment frequency: S1 cells {gf_s1:.1f} against {ref_s1} (better >= 6, no worse >= 2)")
    print(f"pooled A6/A7 parts the reference meets and goal-first loses: {lost or 'none'}")
    better_a, nw_a = gf_a >= 24.5, gf_a >= 21.5
    better_s, nw_s = gf_s1 >= 6, gf_s1 >= 2
    adopt = not lost and ((better_a and nw_s) or (better_s and nw_a))
    print(f"rule: {'adopted' if adopt else 'not adopted'}")
    print(f"GOALFIRST {int(adopt)}")


def pooled_s4(run: str):
    c = load(R / "eval" / run / "calibration_masked.json")
    if not c:
        return None
    out = {}
    for k in TR:
        o = sum(c[d]["masked"][k]["after_24h"]["observed"] for d in DS if d in c)
        x = sum(c[d]["masked"][k]["after_24h"]["expected"] for d in DS if d in c)
        out[k] = o / x if x else None
    return out


def final_nll(run: str):
    m = R / "train/full" / run / "metrics.jsonl"
    if not m.exists():
        return None
    ev = [json.loads(x) for x in m.read_text().splitlines() if '"eval"' in x]
    return ev[-1]["eval"]["nll"] if ev else None


def finalist(runs: list, screen: str = "screen") -> None:
    ref = pooled_s4(REF)
    rows = []
    for run in runs:
        nll, p1 = final_nll(run), load(R / "audit" / f"m7_{run}" / "p1.json")
        q = load(R / "eval" / run / screen / "quality.json")
        x = load(R / "eval" / run / screen / "explain.json")
        s4 = pooled_s4(run)
        if nll is None or p1 is None or q is None or x is None or s4 is None:
            print(f"{run}: incomplete (NLL, P1, screen or S4 missing): out")
            continue
        s1 = s1_cells(q)
        auc = sum(x[ds]["first_24h"]["unigram_auc"] for ds in DS if ds in x) / sum(ds in x for ds in DS)
        closer = sum(abs(1 - s4[k]) < abs(1 - ref[k]) for k in TR)
        ok = nll <= S3_MAX and p1["status"] == "PASS"
        print(f"{run}: tuning NLL {nll:.4f} (S3 {'holds' if nll <= S3_MAX else 'fails'}), P1 {p1['status']}, "
              f"S1 cells {s1[0]} of {s1[1]} {s1[2]}, first-24-h unigram AUC {auc:.3f}, "
              f"S4 {', '.join(f'{k[:4]} {s4[k]:.3f}' for k in TR)} ({closer} of 3 closer to 1 than {REF})")
        if ok:
            rows.append((-s1[0], auc, nll, run, closer))
    if not rows:
        print("no candidate holds S3 and P1")
        print("FINALIST none")
        return
    rows.sort()
    _s1, _auc, _nll, best, closer = rows[0]
    print(f"most S1 cells, then the lower unigram AUC, then the lower NLL: {best}")
    if closer < 2:
        print(f"{best} brings S4 closer to 1 for {closer} of 3 treatments (< 2): the round is reported, no decision run")
        print("FINALIST none")
        return
    print(f"FINALIST {best}")


if __name__ == "__main__":
    if sys.argv[1] == "goalfirst":
        goalfirst()
    else:                  # finalist [--screen DIR] [runs...]
        args = sys.argv[2:]
        screen = "screen"
        if args[:1] == ["--screen"]:
            screen, args = args[1], args[2:]
        finalist(args or ["MC32_main", "MU32_main", "MUH32_main"], screen)
