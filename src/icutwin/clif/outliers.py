"""CLIF outlier handling as SQL, using clifpy's official ``outlier_config.yaml`` (values out of range -> NULL).

clifpy's ``apply_outlier_handling`` works on in-memory frames; this is the same rule in DuckDB so it scales
to full tables. Parity with the official implementation is enforced by ``tests/test_clif_outliers.py``.
Values are nulled, never clamped (clamping, as ELF/CLIF_MEDS does, would distort the distributions the
generator must learn). ``outlier_extensions.yaml`` adds ranges for categories the official (2.1-era) file
lacks; it may only fill gaps, never override an official range.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from importlib import resources
from pathlib import Path

import clifpy
import duckdb
import pyarrow.parquet as pq
import yaml

from icutwin.clif.units import clean_dose_unit

# category column that selects the range for category-dependent value columns
_CATEGORY_COL = {
    "vitals": "vital_category",
    "labs": "lab_category",
    "patient_assessments": "assessment_category",
    "medication_admin_continuous": "med_category",
    "medication_admin_intermittent": "med_category",
    "invasive_hemodynamics": "measurement_category",
    "mcs": "setting_category",
}
_UNIT_COL = {"medication_admin_continuous": "med_dose_unit", "medication_admin_intermittent": "med_dose_unit"}


@lru_cache(maxsize=1)
def official_outlier_config() -> dict:
    path = Path(clifpy.__file__).parent / "schemas" / "outlier_config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["tables"]


@lru_cache(maxsize=1)
def outlier_extensions() -> dict:
    text = resources.files("icutwin.clif").joinpath("outlier_extensions.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)["tables"]


def outlier_overlaps() -> list[str]:
    """Extension keys that also exist in the official config (must be empty)."""
    official, out = official_outlier_config(), []
    for table, cols in outlier_extensions().items():
        for col, cats in cols.items():
            have = {k.lower() for k in ((official.get(table) or {}).get(col) or {})}
            out += [f"{table}.{col}.{c}" for c in cats if c.lower() in have]
    return out


@lru_cache(maxsize=2)
def outlier_config(extensions: bool = True) -> dict:
    cfg = copy.deepcopy(official_outlier_config())
    if not extensions:
        return cfg
    if outlier_overlaps():
        raise ValueError(f"outlier_extensions.yaml overrides official ranges: {outlier_overlaps()}")
    for table, cols in outlier_extensions().items():
        for col, cats in cols.items():
            cfg.setdefault(table, {}).setdefault(col, {}).update(cats)
    return cfg


def _lit(v: str) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def _bounds(spec: dict, expr: str) -> str:
    parts = []
    if spec.get("min") is not None:
        parts.append(f"{expr} < {float(spec['min'])}")
    if spec.get("max") is not None:
        parts.append(f"{expr} > {float(spec['max'])}")
    return " OR ".join(parts) or "false"


def _out_of_range(table: str, col: str, spec: dict) -> str | None:
    """SQL boolean that is true when ``col`` holds an outlier under the official ranges."""
    expr = f'"{col}"'
    if "min" in spec or "max" in spec:
        return _bounds(spec, expr)
    cat = _CATEGORY_COL.get(table)
    if cat is None:
        return None
    unit = _UNIT_COL.get(table)
    whens = []
    # categories match case-insensitively, as in clifpy (its config mixes 'RASS', 'NRS' with 'gcs_eye')
    for category, sub in spec.items():
        if unit and isinstance(sub, dict) and not ({"min", "max"} & set(sub)):
            for u, rng in sub.items():  # medications: ranges per (category, unit); the official spelling
                # or its canonical (cleaned) form, as harmonize writes dose units
                spellings = ", ".join(_lit(x) for x in sorted({str(u).lower(), clean_dose_unit(str(u))[0]}))
                whens.append(f"WHEN lower(\"{cat}\") = {_lit(category.lower())} "
                             f"AND lower(\"{unit}\") IN ({spellings}) THEN ({_bounds(rng, expr)})")
        else:
            whens.append(f"WHEN lower(\"{cat}\") = {_lit(category.lower())} THEN ({_bounds(sub, expr)})")
    return f"CASE {' '.join(whens)} ELSE false END" if whens else None


def apply_outliers(con: duckdb.DuckDBPyConnection, src: Path, dst: Path, table: str,
                   extensions: bool = True) -> dict:
    spec = outlier_config(extensions).get(table) or {}
    names = pq.read_schema(src).names
    select, rules = [], []
    for col in names:
        cond = _out_of_range(table, col, spec[col]) if col in spec else None
        if cond is None:
            select.append(f'"{col}"')
            continue
        rules.append(col)
        select.append(f'CASE WHEN ({cond}) THEN NULL ELSE "{col}" END AS "{col}"')
    src_sql = str(src).replace("\\", "/")
    dst_sql = str(dst).replace("\\", "/")
    con.execute(f"COPY (SELECT {', '.join(select)} FROM read_parquet('{src_sql}')) TO '{dst_sql}' "
                f"(FORMAT parquet, COMPRESSION zstd)")
    nulled = {}
    for col in rules:
        before = con.execute(f"SELECT count(\"{col}\") FROM read_parquet('{src_sql}')").fetchone()[0]
        after = con.execute(f"SELECT count(\"{col}\") FROM read_parquet('{dst_sql}')").fetchone()[0]
        nulled[col] = before - after
    return {"rule": "clifpy_outlier_config" + ("+icutwin_extensions" if extensions else ""), "nulled": nulled}
