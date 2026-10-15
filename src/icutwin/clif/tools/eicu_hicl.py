"""Build ``maps/eicu/hicl.csv``: eICU HICL ingredient code -> lexicon ingredient.

eICU's ``medication`` table leaves ``drugname`` empty for about a third of the orders and gives only the First
Databank HICL sequence number, an ingredient-level code (one HICL covers every strength, formulation and
route of an ingredient combination). Orders that carry both a name and a HICL tell which ingredient a HICL
stands for: their names are resolved with the shared lexicon (ignoring the formulation row, because a HICL
does not encode the formulation), and a HICL is mapped only when one ingredient accounts for at least
``--min-purity`` of its named orders, over at least ``--min-named`` orders.

    python -m icutwin.clif.tools.eicu_hicl --archive dataset/eICU/eICU-2.0.tar.zst \
        --out src/icutwin/clif/maps/eicu/hicl.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.csv as pacsv

from icutwin.clif import lexicon
from icutwin.stage.archive import iter_members

FORMULATION = "non_systemic_formulation"


def count_names(archive: Path, prefix: str = "eICU-2.0/") -> tuple[Counter, Counter]:
    """(hicl, lower(drugname)) counts over named orders, and order counts per HICL without a name."""
    named, unnamed = Counter(), Counter()
    for member, stream in iter_members(archive, prefix):
        if member.rel != "medication.csv":
            continue
        opts = pacsv.ConvertOptions(include_columns=["drugname", "drughiclseqno"],
                                    column_types={"drugname": "string", "drughiclseqno": "string"})
        reader = pacsv.open_csv(stream, read_options=pacsv.ReadOptions(block_size=1 << 26), convert_options=opts)
        for batch in reader:
            names, hicls = batch.column("drugname").to_pylist(), batch.column("drughiclseqno").to_pylist()
            for n, h in zip(names, hicls):
                if not h:
                    continue
                if n and n.strip():
                    named[(h, n.strip().lower())] += 1
                else:
                    unnamed[h] += 1
        break
    return named, unnamed


def build(named: Counter, unnamed: Counter, min_purity: float, min_named: int) -> list[dict]:
    res = lexicon.resolve(sorted({n for _, n in named}), exclude=(FORMULATION,))
    by_hicl: dict[str, Counter] = defaultdict(Counter)
    example: dict[tuple[str, str], tuple[str, int]] = {}
    for (h, n), k in named.items():
        ing = (res.get(n) or {}).get("ingredient") or "(unresolved)"
        by_hicl[h][ing] += k
        if k > example.get((h, ing), ("", 0))[1]:
            example[(h, ing)] = (n, k)
    rows = []
    for h, dist in sorted(by_hicl.items(), key=lambda x: int(x[0])):
        total = sum(dist.values())
        ing, k = dist.most_common(1)[0]
        if ing == "(unresolved)" or total < min_named or k / total < min_purity:
            continue
        rows.append({"drughiclseqno": h, "ingredient": ing, "purity": round(k / total, 4), "named_orders": total,
                     "unnamed_orders": unnamed.get(h, 0), "example_name": example[(h, ing)][0]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-purity", type=float, default=0.9)
    ap.add_argument("--min-named", type=int, default=5)
    args = ap.parse_args()
    named, unnamed = count_names(args.archive)
    rows = build(named, unnamed, args.min_purity, args.min_named)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    covered = sum(r["unnamed_orders"] for r in rows)
    print(f"HICL codes mapped: {len(rows)}; unnamed orders covered: {covered} of {sum(unnamed.values())}")


if __name__ == "__main__":
    main()
