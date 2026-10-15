"""Parity of ``icutwin.meds.elf`` with CLIF_MEDS on the domains our build takes from our engine.

Runs our engine with the exact concept configs a CLIF_MEDS run used (``<meds_dir>/_concept_config``) on the
same canonical CLIF directory and compares the events as multisets (``EXCEPT ALL`` both ways): the domains
CLIF_MEDS converts generically, and the qualified domains (``meds/qualifiers.yaml``: medications with
CLIF_MEDS's clifpy unit conversion, vitals, assessments), whose codes are collapsed to their parent ELF code
for the comparison.

    python -m icutwin.meds.parity --clif data/clif_canonical/subset/mimic --meds data/meds_elf/subset/mimic
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from icutwin.clif.conform import connect
from icutwin.meds.build import base_code_levels, extension_configs, qualifiers
from icutwin.meds.elf import EVENT_COLUMNS, convert_domain, load_config, qualified_config

GENERIC_DOMAINS = ("PATIENT", "HOSP", "ADT", "VITAL", "LAB", "POS", "CODE_STATUS", "HOSP_DX", "PROC", "PA")
# CLIF_MEDS's medication processors convert doses with the clifpy of CLIF_MEDS's own environment (0.4.x; ours is
# the project's pinned 2.1.0). Found in M4: its event times come out in the host's local time
# zone (8 h early on a US Pacific host, pytz's standard time for dates after 2037; none on a UTC host), and its
# weight lookup differs on a few events. Our engine keeps canonical wall-clock times, so these domains are
# compared on every column but time (the report gives the time differences), and the only difference allowed is
# a unit conversion of an otherwise identical event (same subject, hospitalization, drug, action and route)
SHIFTED_DOMAINS = ("MED_CON", "MED_INT")


def _p(path: Path) -> str:
    return str(path).replace("\\", "/")


def check(clif_dir: Path, meds_dir: Path, work_dir: Path) -> dict:
    con = connect()
    report = {}
    qualified = {d for d in qualifiers() if d not in extension_configs()}
    for domain in sorted(set(GENERIC_DOMAINS) | qualified):
        ref = meds_dir / "data" / f"{domain}.parquet"
        if not ref.exists():
            continue
        cfg = load_config(meds_dir / "_concept_config" / f"{domain}.yaml")
        code = "code"
        if domain in qualified:   # our qualified events, each code collapsed to the ELF code it extends
            q = qualifiers()[domain]
            n = base_code_levels(cfg)
            cfg = qualified_config(cfg, q["level"], q.get("where"))
            code = f"array_to_string(string_split(code, '//')[1:{n}], '//') AS code"
        ours = work_dir / f"{domain}.parquet"
        events = convert_domain(con, clif_dir, cfg, domain, ours)
        cols = list(EVENT_COLUMNS)
        if "hospitalization_id" in pq.read_schema(ref).names and "hospitalization_id" in pq.read_schema(ours).names:
            cols.append("hospitalization_id")
        shifted = domain in SHIFTED_DOMAINS
        if shifted:
            cols = [c for c in cols if c != "time"]
        sel = ", ".join(cols)
        ours_sel = ", ".join(code if c == "code" else c for c in cols)
        diff = {}
        for name, a, b in (("only_clif_meds", f"SELECT {sel} FROM read_parquet('{_p(ref)}')",
                            f"SELECT {ours_sel} FROM read_parquet('{_p(ours)}')"),
                           ("only_icutwin", f"SELECT {ours_sel} FROM read_parquet('{_p(ours)}')",
                            f"SELECT {sel} FROM read_parquet('{_p(ref)}')")):
            q_ = f"{a} EXCEPT ALL {b}"
            diff[name] = con.execute(f"SELECT count(*) FROM ({q_})").fetchone()[0]
            diff[name + "_examples"] = [list(map(str, r)) for r in con.execute(f"{q_} LIMIT 3").fetchall()]
        report[domain] = {"clif_meds": pq.ParquetFile(ref).metadata.num_rows, "icutwin": events,
                          "qualified": domain in qualified, "time_compared": not shifted, **diff}
        if shifted and (diff["only_clif_meds"] or diff["only_icutwin"]):
            # the differing events must pair up once unit and value are set aside: a unit conversion difference
            no_unit = ("subject_id, hospitalization_id, split_part(code, '//', 2) AS drug, "
                       "split_part(code, '//', 4) AS action")
            a = f"SELECT {sel} FROM read_parquet('{_p(ref)}') EXCEPT ALL SELECT {ours_sel} FROM read_parquet('{_p(ours)}')"
            b = f"SELECT {ours_sel} FROM read_parquet('{_p(ours)}') EXCEPT ALL SELECT {sel} FROM read_parquet('{_p(ref)}')"
            unpaired = con.execute(f"""SELECT (SELECT count(*) FROM (SELECT {no_unit} FROM ({a}) EXCEPT ALL
                                                                  SELECT {no_unit} FROM ({b}))),
                                              (SELECT count(*) FROM (SELECT {no_unit} FROM ({b}) EXCEPT ALL
                                                                  SELECT {no_unit} FROM ({a})))""").fetchone()
            if not any(unpaired):
                report[domain]["unit_conversion_differs"] = diff["only_icutwin"]
                diff["only_clif_meds"] = diff["only_icutwin"] = 0
                report[domain].update(only_clif_meds=0, only_icutwin=0)
        if shifted:   # hours CLIF_MEDS's event times lie before ours (same subject, parent code and value)
            report[domain]["clif_meds_time_shift_hours"] = dict(con.execute(f"""
                WITH o AS (SELECT subject_id, {code}, numeric_value, time, row_number() OVER () AS k
                           FROM read_parquet('{_p(ours)}')),
                     r AS (SELECT subject_id, code, numeric_value, time FROM read_parquet('{_p(ref)}'))
                SELECT round(epoch(o.time - r.time) / 3600, 2) AS h, count(DISTINCT o.k) AS n
                FROM o JOIN r USING (subject_id, code, numeric_value)
                WHERE abs(epoch(o.time - r.time)) < 86400 GROUP BY 1 ORDER BY 2 DESC LIMIT 3""").fetchall())
    report["identical"] = all(v["only_clif_meds"] == 0 and v["only_icutwin"] == 0 for v in report.values()
                              if isinstance(v, dict))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clif", type=Path, required=True)
    ap.add_argument("--meds", type=Path, required=True)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        print(json.dumps(check(args.clif, args.meds, Path(tmp)), indent=1))


if __name__ == "__main__":
    main()
