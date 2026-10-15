"""Paper P amendment 11: record V3's final verdict (``runs/toolkit/v3/v3_final.json``: V3x's loss test with V3's
extraction, intervened and copying-control parts) in the M7 gate report, the study lead's decision ("enable V3 access").

Only a PASS with valid controls is recorded: ``conditional_and_counterfactual_sampling`` (its status, the assessment,
the file, the date and the headline numbers) at the report's top level, and the PENDING entry of ``not_in_v1``
replaced by a pointer to it. ``--accept-inconclusive <dataset>`` (the study lead's decision, in the protocol) records a
PASS whose only non-PASS parts are those datasets' INCONCLUSIVE loss tests, each stated in the record. The gate's own
fields (``gate``, P1-P5 and their controls) are unchanged; the report as it was is kept beside it
(``report_before_v3.json``). Exits 2 without recording anything otherwise.

    python tools/server/gate_record_v3.py [--accept-inconclusive inspire]
"""
import argparse
import json
import shutil
import sys
from datetime import UTC, datetime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept-inconclusive", nargs="*", default=[])
    args = ap.parse_args()
    from icutwin.toolkit.session import Paths

    paths = Paths.default()
    final = json.loads((paths.out / "v3" / "v3_final.json").read_text())
    loss = final["a_loss"]
    exceptions = {d: {"status": x["status"], "members": x["members"], "nonmembers": x["nonmembers"], "auc": x["auc"],
                      "auc_ci": x["auc_ci"], "tpr_1fpr": x["tpr_1fpr"], "tpr_ci": x["tpr_ci"],
                      "reason": ("sample-limited: every eligible held-out stay used "
                                 "(the study lead's decision to record)")}
                  for d, x in loss.items() if d in args.accept_inconclusive and x["status"] == "INCONCLUSIVE"}
    others = [x["status"] for d, x in loss.items() if d not in exceptions] + \
        [final[k][f] for k in ("b_extraction", "c_intervened") for f in ("exact32_status", "jaccard_status")
         if final[k].get(f)]
    passed = final["status"] == "PASS" or (bool(exceptions) and all(x == "PASS" for x in others))
    if not passed or not final["controls_valid"]:
        print(json.dumps({"recorded": False, "status": final["status"], "controls_valid": final["controls_valid"]}))
        sys.exit(2)
    rep_f = paths.audit / "report.json"
    backup = paths.audit / "report_before_v3.json"
    if not backup.exists():
        shutil.copy(rep_f, backup)
    report = json.loads(rep_f.read_text())
    a = final["a_loss"]["pooled"]
    report["conditional_and_counterfactual_sampling"] = {
        "status": "PASS", "assessment": "Paper P amendment 11 V3 (loss test at P1's sizes: V3x)",
        "file": "prompted.json", "recorded": datetime.now(UTC).date().isoformat(),
        "loss_auc_pooled": a["auc"], "loss_auc_ci": a["auc_ci"], "tpr_1fpr_pooled": a["tpr_1fpr"],
        "tpr_1fpr_ci": a["tpr_ci"], "exact32_difference": final["b_extraction"]["exact32_difference"],
        "intervened_exact32_difference": final["c_intervened"]["exact32_difference"],
        "controls_valid": final["controls_valid"], "rule_verdict": final["status"], "exceptions": exceptions}
    note = f"; {', '.join(exceptions)} inconclusive, sample-limited" if exceptions else ""
    report.setdefault("not_in_v1", {})["conditional_and_counterfactual_sampling"] = \
        f"PASS (Paper P amendment 11 V3{note}; see conditional_and_counterfactual_sampling, prompted.json)"
    rep_f.write_text(json.dumps(report, indent=1))
    print(json.dumps({"recorded": True, "report": str(rep_f), "backup": str(backup)}))


if __name__ == "__main__":
    main()
