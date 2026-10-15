"""icutwin on the AutoDL host: round 3's adoption rule (docs/m7_refinement.md, "Targets and decision", amendments 1-3)
applied to the decision run of a finalist against M32_main (aggregates only):
* the finalist's M7 gate passes; S3 (tuning NLL at most 1.571) holds;
* no pooled A6/A7 part the reference meets in both its decision audits fails in either of the finalist's;
* better on one axis and no worse on the other: authenticity, the mean A1-A5 count of the two decision audits
  (better: reference + 2; no worse: reference - 1); calibration, S1 + S2 cells (better: + 3; no worse: - 1). S1 of
  the finalist from its length-only audit (auth_s1), of the reference from its audit under the same draws (auth_d,
  or with goal-first draws the mean of auth_g0 and auth_g1); S2 from each one's rollouts (forecast_h.json).
    python round3_adopt.py <finalist> <goal-first: 0|1>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from round3_tally import load, s1_cells, s2_cells  # noqa: E402
from round3_select import a15, final_nll  # noqa: E402

R = Path("/root/autodl-tmp/icutwin/runs")
REF = "M32_main"


def main() -> None:
    fin, gf = sys.argv[1], sys.argv[2] == "1"
    ef, er = R / "eval" / fin, R / "eval" / REF
    gate = (load(R / "audit" / f"m7_{fin}" / "report.json") or {}).get("gate")
    nll = final_nll(fin)
    ft = [load(ef / d / "targets.json") for d in ("auth_r0", "auth_r1")]
    rt = [load(er / d / "targets.json") for d in ("auth_ref0", "auth_ref1")]
    if None in ft or None in rt:
        print("decision audits missing: no adoption")
        print("ADOPT 0")
        return
    fa, ra = sum(map(a15, ft)) / 2, sum(map(a15, rt)) / 2
    parts = set(rt[0]["pooled"]) | set(rt[1]["pooled"])
    lost = [k for k in sorted(parts) if all(t["pooled"].get(k) for t in rt) and not all(t["pooled"].get(k) for t in ft)]
    fs1 = s1_cells(load(ef / "auth_s1" / "quality.json"))[0]
    if gf:
        rs1 = sum(s1_cells(load(er / d / "quality.json"))[0] for d in ("auth_g0", "auth_g1")) / 2
    else:
        rs1 = s1_cells(load(er / "auth_d" / "quality.json"))[0]
    fs2 = s2_cells(load(ef / "forecast_h.json"))[0] if load(ef / "forecast_h.json") else None
    rs2 = s2_cells(load(er / "forecast_h.json"))[0]
    print(f"gate {gate}; tuning NLL {nll:.4f} (S3 {'holds' if nll <= 1.571 else 'fails'})")
    print(f"authenticity: A1-A5 mean {fa:.1f} ({[a15(t) for t in ft]}) against {ra:.1f} ({[a15(t) for t in rt]})")
    print(f"pooled parts: finalist {[t['pooled'] for t in ft]}; reference {[t['pooled'] for t in rt]}; lost {lost or 'none'}")
    if fs2 is None:
        print("no rollouts for the finalist: no adoption")
        print("ADOPT 0")
        return
    fc, rc = fs1 + fs2, rs1 + rs2
    print(f"calibration: S1 {fs1} + S2 {fs2} = {fc} against S1 {rs1} + S2 {rs2} = {rc}")
    better_a, nw_a = fa >= ra + 2, fa >= ra - 1
    better_c, nw_c = fc >= rc + 3, fc >= rc - 1
    adopt = gate == "PASS" and nll <= 1.571 and not lost and ((better_a and nw_c) or (better_c and nw_a))
    print(f"axes: authenticity {'better' if better_a else 'no worse' if nw_a else 'worse'}, "
          f"calibration {'better' if better_c else 'no worse' if nw_c else 'worse'}")
    print(f"rule: {fin} {'replaces' if adopt else 'does not replace'} {REF}")
    print(f"ADOPT {int(adopt)}")


if __name__ == "__main__":
    main()
