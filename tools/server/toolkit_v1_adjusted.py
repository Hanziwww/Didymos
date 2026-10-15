"""Paper P amendment 11, V1 under the study lead's adjusted criterion (post hoc, set after the original rule's and the
added check's results were known; ``docs/p1_paper_protocol.md``). A program conforms when the toolkit's AUROC and O/E
both lie within the added check's limit (1.96 x sqrt(2) x the rollout-bootstrap SD, ``runs/toolkit/v1/v1_noise.json``)
of either paper run on the same anchors: A1 (32 rollouts) or F1 (128 rollouts: its curve at 32 rollouts, the mean over
random 32-rollout subsets, and its O/E). V1 passes when all nine programs conform and E2 and the what-if checks pass.
Writes ``runs/toolkit/v1/v1_adjusted.json`` with the three verdicts.

    python tools/server/toolkit_v1_adjusted.py
"""
import json

from icutwin.toolkit.session import Paths


def main() -> None:
    paths = Paths.default()
    v1 = paths.out / "v1"
    rep = json.loads((v1 / "v1.json").read_text())
    noise = json.loads((v1 / "v1_noise.json").read_text())
    f1 = json.loads((paths.paper / "f1/R128/a1.json").read_text())["programs"]
    out = {"criterion": "within the noise limit of A1 or of F1 (curve at 32; O/E at 128), AUROC and O/E both",
           "programs": {}}
    for prog, x in noise["programs"].items():
        ref_f1 = {"auroc": f1[prog]["rollout_curve"]["32"]["auroc"],
                  "oe": f1[prog]["pooled"]["observed_over_expected"]}
        near = {"a1": abs(x["auroc_toolkit"] - x["auroc_a1"]) <= x["auroc_limit"]
                and abs(x["oe_toolkit"] - x["oe_a1"]) <= x["oe_limit"],
                "f1": abs(x["auroc_toolkit"] - ref_f1["auroc"]) <= x["auroc_limit"]
                and abs(x["oe_toolkit"] - ref_f1["oe"]) <= x["oe_limit"]}
        out["programs"][prog] = {"auroc_toolkit": x["auroc_toolkit"], "auroc_a1": x["auroc_a1"],
                                 "auroc_f1_curve32": ref_f1["auroc"], "oe_toolkit": x["oe_toolkit"], "oe_a1": x["oe_a1"],
                                 "oe_f1": ref_f1["oe"], "auroc_limit": x["auroc_limit"], "oe_limit": x["oe_limit"],
                                 "within_a1": bool(near["a1"]), "within_f1": bool(near["f1"]),
                                 "original_rule": rep["forecast"][prog]["pass"], "added_check": x["pass"],
                                 "pass": bool(near["a1"] or near["f1"])}
    out["e2_and_whatif_pass"] = noise["e2_and_whatif_pass"]
    out["verdicts"] = {"original_rule": rep["pass"], "added_check": noise["pass"],
                       "adjusted": bool(noise["e2_and_whatif_pass"] and all(x["pass"] for x in out["programs"].values()))}
    out["pass"] = out["verdicts"]["adjusted"]
    (v1 / "v1_adjusted.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"verdicts": out["verdicts"], **{p: (x["within_a1"], x["within_f1"]) for p, x in
                                                      out["programs"].items()}}, indent=1), flush=True)


if __name__ == "__main__":
    main()
