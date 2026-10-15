"""icutwin on the AutoDL host (2026-10-07): the category -> group pairs of the full canonical CLIF (aggregates: names
and counts only), for generate.clif_out's group columns."""
import json
from pathlib import Path

import duckdb

ROOT = Path("/root/autodl-tmp/icutwin/data/clif_canonical/full")
PAIRS = {"output": ("output_category", "output_group"), "input": ("input_category", "input_group"),
         "medication_admin_continuous": ("med_category", "med_group"),
         "medication_admin_intermittent": ("med_category", "med_group"),
         "patient_assessments": ("assessment_category", "assessment_group"),
         "medication_admin_continuous:mar": ("mar_action_category", "mar_action_group"),
         "microbiology_culture": ("organism_category", "organism_group")}
con = duckdb.connect()
con.execute("SET threads=4")
out = {}
for key, (cat, grp) in PAIRS.items():
    table = key.split(":")[0]
    files = sorted(ROOT.glob(f"*/clif_{table}.parquet"))
    if not files:
        continue
    lst = ", ".join(f"'{f}'" for f in files)
    rows = con.execute(f"""SELECT {cat}, {grp}, count(*) FROM read_parquet([{lst}], union_by_name=true)
                           WHERE {cat} IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 3 DESC""").fetchall()
    pairs: dict = {}
    for c, g, n in rows:
        pairs.setdefault(c, {})[str(g)] = n
    out[key] = {"columns": [cat, grp], "pairs": pairs,
                "not_functional": {c: v for c, v in pairs.items() if len([g for g in v if g != "None"]) > 1}}
print(json.dumps(out))
