"""icutwin on the AutoDL host: the M7 gate's results as compact text (aggregates only, from runs/audit/m7_M_main)."""
import json
from pathlib import Path

import sys
D = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/icutwin/runs/audit/m7_M_main")
r = lambda n: json.loads((D / f"{n}.json").read_text()) if (D / f"{n}.json").exists() else None  # noqa: E731
f = lambda t, k=3: f"{t[0]:.{k}f} [{t[1]:.{k}f}, {t[2]:.{k}f}]"  # noqa: E731
p1, p2, syn, rep = r("p1"), r("p2"), r("synthetic"), r("report")
if p1:
    print("P1", p1["status"], p1.get("model", {}).get("weights_sha256", "")[:12])
    for d, g in p1["groups"].items():
        print(f"  {d:8s} n={g['members']}/{g['nonmembers']} AUC {f(g['auc'])} TPR@1%FPR {f(g['tpr_at_1pct_fpr'])} "
              f"NLL {g['mean_nll_members']:.3f}/{g['mean_nll_nonmembers']:.3f} {g['status']}")
if p2:
    print("P2", p2["status"])
    for s, rows in p2["settings"].items():
        for d, g in rows.items():
            print(f"  {s} {d:8s} n={g['members']}/{g['nonmembers']} exact {g['exact_members']:.4f}/{g['exact_nonmembers']:.4f} "
                  f"diff {f(g['difference'], 4)} prefix {g['matched_prefix_members']:.1f}/{g['matched_prefix_nonmembers']:.1f} {g['status']}")
if syn:
    print("samples", syn["samples"], "grammar-valid", round(syn["grammar_valid_share"], 4), syn["datasets_of_samples"])
    for g in ("model", "private_reference", "copying_generator"):
        p3 = syn["p3"][g]
        print(f"P3 {g}: {p3['status']} {p3['statuses']} copies {f(p3['copy_rate_difference'], 4)} (baseline "
              f"{p3['copy_rate_baseline']:.4f}) DCR<q05 {f(p3['share_below_dcr_q05'])} NNDR<q05 {f(p3['share_below_nndr_q05'])} "
              f"not authentic {p3['not_authentic']:.3f} (baseline {p3['not_authentic_baseline']:.3f})")
        if g == "model":
            for d, x in p3["datasets"].items():
                print(f"    {d:8s} n={x['samples']} DCR<q05 {x['below_dcr_q05']:.3f} NNDR<q05 {x['below_nndr_q05']:.3f} "
                      f"copies {x['copies']:.4f} DCR median {x['dcr_median']:.3f} not authentic {x['not_authentic']:.3f}")
        p4 = syn["p4"][g]
        print(f"P4 {g}: {p4['status']} AUC {f(p4['auc'])}")
        p5 = syn["p5"][g]
        if "difference" in p5:
            print(f"P5 {g}: {p5['status']} BA {p5['balanced_accuracy_members']:.3f}/{p5['balanced_accuracy_nonmembers']:.3f} "
                  f"diff {f(p5['difference'])} n={p5['members']}/{p5['nonmembers']} prevalence {p5['prevalence']:.3f} "
                  f"samples complete {p5['samples_complete']}")
        else:
            print(f"P5 {g}: {p5}")
if rep:
    print("REPORT", json.dumps(rep))
