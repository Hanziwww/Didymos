"""A twin-supply plan (plan Phase F, S1b; ``generate.supply.SupplyPlan``) from the training stays' marginals
(``tools/factory_store.py marginals``): its mechanisms, cells and requests, written as JSON for the producer
(``tools/server/supply_produce.py``). Aggregates only.

    python tools/supply_plan.py --marginals data/tokens/full_r4/supply_marginals.json --total 2000 --max-strata 6 \\
        --name s1b --out data/supply/s1b/plan.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--marginals", required=True)
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--power", type=float, default=0.5)
    ap.add_argument("--strata-min", type=int, default=20)
    ap.add_argument("--max-strata", type=int, default=None)
    ap.add_argument("--site-min", type=int, default=200)
    ap.add_argument("--goal-min", type=int, default=20)
    a = ap.parse_args(argv)
    from icutwin.generate import supply

    marg = json.loads(Path(a.marginals).read_text(encoding="utf-8"))
    plan = supply.SupplyPlan.build(marg, a.total, name=a.name, power=a.power, strata_min=a.strata_min,
                                   max_strata=a.max_strata, site_min=a.site_min, goal_min=a.goal_min)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plan.save(Path(a.out))
    by: dict = {}
    for c in plan.cells:
        k = c.mechanism if c.mechanism != "goal" else f"goal:{c.goal[0]}"
        by.setdefault(k, {})[c.dataset] = by.get(k, {}).get(c.dataset, 0) + c.quota
    print(json.dumps({"total": plan.total, "mechanisms": plan.mechanisms(), "cells": len(plan.cells),
                      "by_mechanism_and_dataset": by, "out": a.out}, indent=1))


if __name__ == "__main__":
    main()
