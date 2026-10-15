"""Build ``maps/aumc/tool_intermittent_items.csv``: the drugitems CLIF-AmsterdamUMCdb maps to intermittent doses.

CLIF-AmsterdamUMCdb builds ``medication_admin_intermittent`` from a fixed list of 62 item ids
(``mappings/medication_admin_intermittent.yaml``). The AUMC supplement adds the non-continuous administrations
of every other item with a CLIF category; it never re-maps an item of this list.

    python -m icutwin.clif.tools.aumc_tool_items \
        --mapping refs/CLIF-AmsterdamUMCdb/mappings/medication_admin_intermittent.yaml \
        --out src/icutwin/clif/maps/aumc/tool_intermittent_items.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def tool_items(mapping: Path) -> list[dict]:
    spec = yaml.safe_load(mapping.read_text(encoding="utf-8"))
    return sorted(({"itemid": str(i["itemid"]), "item": i["item"], "med_category": i["med_category"]}
                   for i in spec["drugitems"]["items"]), key=lambda r: int(r["itemid"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["itemid", "item", "med_category", "provenance"], lineterminator="\n")
        w.writeheader()
        for r in tool_items(args.mapping):
            w.writerow({**r, "provenance": "CLIF-AmsterdamUMCdb mappings/medication_admin_intermittent.yaml"})


if __name__ == "__main__":
    main()
