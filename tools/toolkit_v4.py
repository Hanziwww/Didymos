"""Paper P amendment 11, V4: the trial designer on the kept held-out predictions (CPU; ``docs/p1_paper_protocol.md``).

Cells: ICU and in-hospital death on L24x (128-rollout shares); new IMV and new vasopressor within 24 h, new RRT and AKI
stage 2 within 48 h on L24 (32-rollout shares); relative risks 0.8 and 0.7; the Twin, SOFA, SAPS II, OASIS, NEWS and
LightGBM as prognostic scores (APACHE IVa on its eICU anchors, with the Twin and LightGBM on the same anchors, apart);
enrichment fractions 1, 0.75, 0.5, 0.25; 2,000 simulated trials per score for the adjusted analysis
(``toolkit.trial``). Checks: the type I error at relative risk 1 within 0.04-0.06 for both analyses; the unadjusted
simulated size within 5% of the analytic one. Writes ``runs/toolkit/v4/v4.json``.

    python tools/toolkit_v4.py --workers 12
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ENDPOINTS = (("L24x", "icu_death@-1"), ("L24x", "hospital_death@0"), ("L24", "imv_onset@1440"),
             ("L24", "vasopressor_onset@1440"), ("L24", "rrt_onset@2880"), ("L24", "aki_2@2880"))
SCORES = ("twin", "sofa", "sapsii", "oasis", "news", "lgbm_pooled")
RRS = (0.8, 0.7)


def cell(job: tuple) -> tuple:
    key, spec = job
    from icutwin.toolkit.session import Paths, Session

    s = Session(Paths.default())
    return key, s.trial(spec)["outputs"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--simulations", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from icutwin.toolkit.session import Paths

    out = Path(a.out) if a.out else Paths.default().out / "v4"
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, (design, ep) in enumerate(ENDPOINTS):
        for j, rr in enumerate(RRS):
            base = {"design": design, "endpoint": ep, "effect": {"rr": rr}, "simulations": a.simulations,
                    "seed": 100 * i + j}
            jobs.append((f"{design}|{ep}|{rr}", {**base, "scores": list(SCORES)}))
            jobs.append((f"{design}|{ep}|{rr}|eicu", {**base, "seed": 100 * i + j + 50,
                                                      "scores": ["twin", "apache_iva", "lgbm_pooled"],
                                                      "population": {"datasets": ["eicu"]}}))
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(cell, jobs))
    checks = {"type1_in_0.04_0.06": {}, "unadjusted_within_5pct": {}}
    for k, o in res.items():
        for name, sc in o["scores"].items():
            adj = sc.get("adjustment")
            if not adj:
                continue
            checks["type1_in_0.04_0.06"][f"{k}|{name}"] = bool(0.04 <= adj["type1_unadjusted"] <= 0.06
                                                               and 0.04 <= adj["type1_adjusted"] <= 0.06)
            checks["unadjusted_within_5pct"][f"{k}|{name}"] = bool(abs(adj["unadjusted_vs_analytic"] - 1) <= 0.05)
    rep = {"cells": res, "checks": checks,
           "checks_passed": {c: (sum(x.values()), len(x)) for c, x in checks.items()}}
    (out / "v4.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8", newline="\n")
    print(json.dumps(rep["checks_passed"]), flush=True)


if __name__ == "__main__":
    main()
