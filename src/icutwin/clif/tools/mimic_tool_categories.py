"""Build ``maps/mimic/tool_{intermittent,continuous}_categories.csv``: the categories CLIF-MIMIC can emit per table.

CLIF-MIMIC builds both medication tables from ICU ``inputevents`` items listed in its mapping (decision
INTERMITTENT or CONTINUOUS, or BOTH for items charted as infusion or bolus). The MIMIC supplement adds ``emar``
administrations, and inside ICU stays a parenteral ``emar`` dose of a drug MetaVision charts is the same
administration the tool already emits from ``inputevents`` (60-94% of intermittent ones have a tool row within
60 min; infusions run in ``inputevents``). Names are given in canonical CLIF 3.0 form: clifpy's value normalizer
plus ``value_fixes.yaml`` of the table.

    python -m icutwin.clif.tools.mimic_tool_categories \
        --mapping "refs/CLIF-MIMIC/data/mappings/mimic-to-clif-mappings - med_category.csv" \
        --out-dir src/icutwin/clif/maps/mimic
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from clifpy.utils.crosswalk import normalize_category_value

from icutwin.clif.harmonize import value_fixes

DECISIONS = {"intermittent": ("INTERMITTENT", "BOTH"), "continuous": ("CONTINUOUS", "BOTH")}


def tool_categories(mapping: Path, kind: str = "intermittent") -> dict[str, list[str]]:
    """canonical category -> CLIF-MIMIC itemids of the decisions that reach ``medication_admin_<kind>``."""
    renames = (value_fixes().get(f"medication_admin_{kind}") or {}).get("med_category") or {}
    out: dict[str, list[str]] = defaultdict(list)
    for r in csv.DictReader(mapping.read_text(encoding="utf-8").splitlines()):
        if r["decision"] not in DECISIONS[kind]:
            continue
        cat = normalize_category_value(r["med_category"]).replace(" ", "_")
        fix = renames.get(cat) or {}
        # a conditional fix (a list: e.g. albumin split by concentration) can give any of its targets
        for target in ([f["to"] for f in fix] if isinstance(fix, list) else [fix.get("to", cat)]):
            out[target].append(r["itemid"])
    return dict(sorted(out.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    for kind, decisions in DECISIONS.items():
        with (args.out_dir / f"tool_{kind}_categories.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["med_category", "itemids", "provenance"])
            for cat, ids in tool_categories(args.mapping, kind).items():
                w.writerow([cat, " ".join(ids), f"CLIF-MIMIC mapping med_category.csv, decision {' or '.join(decisions)}"])


if __name__ == "__main__":
    main()
