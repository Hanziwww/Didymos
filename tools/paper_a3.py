"""Paper P, A3: the privacy gate of the adopted model and the fit-privacy frontier (``docs/p1_paper_protocol.md`` A3),
compiled from the existing M7 audits (``runs/audit/m7_<run>/{p1,p2,p5v2,synthetic,report}.json``) and the runs'
training reports. The frontier: per checkpoint and dataset, P1's membership AUC (with its interval and the TPR at 1%
FPR) against the mean held-out NLL of the attacked stays, and the optimizer; per dataset, the rank correlation of AUC
with the member/non-member NLL gap and with held-out NLL, AdamW and Muon checkpoints together. Writes
``<out>/a3.json`` (aggregates).

    python tools/paper_a3.py --audit runs/audit --train runs/train/full --out runs/paper/a3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import _sha  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("audit", "train", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--adopted", default="MUC32_main")
    a = ap.parse_args(argv)
    from scipy.stats import spearmanr

    audit, train = Path(a.audit), Path(a.train)
    rep = {"git": _sha(), "adopted": a.adopted, "checkpoints": {}, "frontier": {}}
    pts: dict = {}
    for d in sorted(audit.glob("m7_*")):
        run = d.name[3:]
        p1 = json.loads((d / "p1.json").read_text()) if (d / "p1.json").exists() else None
        cfg = json.loads((train / run / "config.json").read_text()) if (train / run / "config.json").exists() else {}
        rep_t = json.loads((train / run / "train_report.json").read_text()) \
            if (train / run / "train_report.json").exists() else {}
        nll = (rep_t.get("final_eval") or {}).get("nll")
        c = {"optimizer": cfg.get("optimizer", "adamw"), "preset": cfg.get("preset"), "tuning_nll": nll,
             "tokens": rep_t.get("tokens"), "p1_status": p1.get("status") if p1 else None, "p1": {}}
        for g, x in (p1 or {}).get("groups", {}).items():
            auc = x["auc"]
            c["p1"][g] = {"auc": auc[0], "auc_ci": auc[1:], "tpr_at_1pct_fpr": x.get("tpr_at_1pct_fpr"),
                          "nll_members": x.get("mean_nll_members"), "nll_nonmembers": x.get("mean_nll_nonmembers"),
                          "status": x.get("status")}
            if g != "pooled" and x.get("mean_nll_nonmembers") is not None:
                pts.setdefault(g, []).append((run, c["optimizer"], auc[0], float(x["mean_nll_nonmembers"]),
                                              float(x["mean_nll_nonmembers"]) - float(x["mean_nll_members"])))
        if (d / "report.json").exists():
            c["gate"] = json.loads((d / "report.json").read_text()).get("gate")
        rep["checkpoints"][run] = c
    for g, rows in sorted(pts.items()):
        auc = np.array([r[2] for r in rows])
        held = np.array([r[3] for r in rows])
        gap = np.array([r[4] for r in rows])
        rep["frontier"][g] = {"points": [{"run": r[0], "optimizer": r[1], "auc": r[2], "nll_nonmembers": r[3],
                                          "nll_gap": r[4]} for r in rows],
                              "spearman_auc_vs_gap": float(spearmanr(auc, gap).statistic) if len(rows) > 2 else None,
                              "spearman_auc_vs_heldout_nll": float(spearmanr(auc, held).statistic)
                              if len(rows) > 2 else None}
    ad = audit / f"m7_{a.adopted}"
    for f in ("p2.json", "p5v2.json", "synthetic.json", "report.json"):
        if (ad / f).exists():
            rep[f"adopted_{f[:-5]}"] = json.loads((ad / f).read_text())
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "a3.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({g: {k: v for k, v in x.items() if k != "points"} for g, x in rep["frontier"].items()}, indent=1))


if __name__ == "__main__":
    main()
