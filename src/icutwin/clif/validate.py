"""S3 gate: CLIFpy validation of a canonical CLIF 3.0 directory.

Every table is validated with clifpy's 3.0 schemas (tables without a clifpy class get a dynamic one, since
clifpy derives the table name from the class name), our ``ext_`` tables with ``ext_schemas.yaml``. Findings
are split into
* hard errors, which fail the gate: missing required columns, data types other than the schema's, category
  values outside CLIF 3.0 plus our declared extensions;
* soft findings (clifpy's data-quality checks: missingness, chronology, duplicates, conditional
  requirements, distribution shifts; required columns without any value, which a source may not record),
  reported with their size; a soft finding above its threshold is listed under ``flags`` for review.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path

import pandas as pd

from icutwin.clif.conform import allowed_values, ext_schemas, table_schema

log = logging.getLogger(__name__)

REPORT = "_clifpy_validation.json"
HARD = {"Missing Required Columns", "Data Type Mismatch", "Invalid Categorical Values"}
# soft findings above these percentages are flagged for review
FLAG_PERCENT = {"Chronological Order Violation": 1.0, "Duplicate Composite Keys": 5.0,
                "Medication Dose Unit Inconsistency": 5.0, "Overlapping Time Periods": 1.0}
# canonical identity columns clifpy's composite keys lack: arterial and cuff blood pressure are separate
# measurements at the same time (harmonize keeps the measurement site); specimens collected together are
# separate cultures (AmsterdamUMCdb releases culture orders without results)
KEY_EXTENSIONS = {"vitals": ("meas_site_name",), "microbiology_culture": ("fluid_category", "method_category")}


def _details(err: dict) -> dict:
    d = err.get("details")
    if isinstance(d, dict):
        return d
    try:
        return ast.literal_eval(str(d))
    except (ValueError, SyntaxError):
        return {}


def _clifpy_errors(table: str, df: pd.DataFrame, out_dir: Path) -> list[dict]:
    from clifpy.tables.base_table import BaseTable
    cls = type("".join(p.capitalize() for p in table.split("_")), (BaseTable,), {})
    obj = cls(data_directory=str(out_dir), filetype="parquet", timezone="UTC", output_directory=str(out_dir),
              data=df, clif_version="3.0")
    obj.validate()
    return list(obj.errors or [])


def _replacement_characters(df: pd.DataFrame) -> list[dict]:
    """Text holding U+FFFD: a source decoded with the wrong encoding (always flagged)."""
    out = []
    for col in df.columns:
        if str(df[col].dtype) in ("string", "object"):
            n = int(df[col].astype("string").str.contains("�", regex=False).fillna(False).sum())
            if n:
                out.append({"type": "Replacement Characters", "description": f"{col}: {n} rows",
                            "percent": round(100 * n / max(len(df), 1), 2)})
    return out


def _rekey_duplicates(table: str, df: pd.DataFrame, errors: list[dict]) -> list[dict]:
    """Recount clifpy's composite-key duplicates with the canonical identity columns it lacks."""
    extra = [c for c in KEY_EXTENSIONS.get(table, ()) if c in df.columns]
    if not extra:
        return errors
    out = []
    for e in errors:
        m = re.search(r"composite key \(([^)]*)\)", str(e.get("description", "")))
        if e.get("type") != "Duplicate Composite Keys" or not m:
            out.append(e)
            continue
        key = [k.strip() for k in m.group(1).split(",")] + extra
        n = int(df.duplicated(subset=key).sum())
        if n == 0:
            continue
        det = {**_details(e), "percent": round(100 * n / max(len(df), 1), 2), "key": key}
        out.append({**e, "description": f"{n} duplicate records on canonical key ({', '.join(key)})", "details": det})
    return out


def _classify(table: str, errors: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    hard, soft, flags = [], [], []
    for e in errors:
        kind, det = str(e.get("type")), _details(e)
        rec = {"type": kind, "description": str(e.get("description", ""))[:300]}
        if kind == "Invalid Categorical Values":
            col = det.get("column")
            ok = allowed_values(table, col) or set()
            bad = [v for v in det.get("top_invalid", []) if v.get("value") not in ok]
            if not bad:             # every flagged value is a declared extension
                soft.append({**rec, "declared_extensions": [v.get("value") for v in det.get("top_invalid", [])]})
                continue
            rec["values"] = bad
        if kind in HARD:
            hard.append(rec)
            continue
        pct = det.get("percent")
        if pct is not None:
            rec["percent"] = pct
        soft.append(rec)
        if pct is not None and kind in FLAG_PERCENT and float(pct) > FLAG_PERCENT[kind]:
            flags.append(rec)
    return hard, soft, flags


def _ext_errors(table: str, df: pd.DataFrame) -> list[dict]:
    hard = []
    for col in ext_schemas()[table]["columns"]:
        name = col["name"]
        if name not in df.columns:
            if col.get("required"):
                hard.append({"type": "Missing Required Columns", "description": name})
            continue
        if col.get("required") and df[name].isna().any():
            hard.append({"type": "Null Required Values", "description": f"{name}: {int(df[name].isna().sum())} rows"})
        pv = col.get("permissible_values")
        if pv:
            bad = sorted(set(df[name].dropna().astype(str)) - {str(v) for v in pv})
            if bad:
                hard.append({"type": "Invalid Categorical Values", "description": name, "values": bad[:20]})
    key = ext_schemas()[table].get("primary_key")
    if key and set(key) <= set(df.columns) and (n := int(df.duplicated(subset=key).sum())):
        hard.append({"type": "Duplicate Primary Key", "description": f"{', '.join(key)}: {n} duplicate rows"})
    return hard


def validate_canonical_dir(clif_dir: Path, work_dir: Path | None = None) -> dict:
    logging.getLogger("clifpy").setLevel(logging.WARNING)
    work = work_dir or (clif_dir / "_clifpy")
    work.mkdir(parents=True, exist_ok=True)
    report: dict = {"clif_dir": str(clif_dir), "clif_version": "3.0", "tables": {}, "flags": {}}
    for f in sorted(clif_dir.glob("clif_*.parquet")):
        table = f.stem.removeprefix("clif_")
        # nullable dtypes keep the stored types: numpy-backed pandas turns an INT column with NULLs into
        # float64, which clifpy would report as a type mismatch that is not in the data
        df = pd.read_parquet(f, dtype_backend="numpy_nullable")
        if table.startswith("ext_"):
            hard, soft, flags = _ext_errors(table, df), [], []
            if table == "ext_icu_stay":
                bad = int((df["icu_out_dttm"] <= df["icu_in_dttm"]).sum())
                if bad:
                    rec = {"type": "Chronological Order Violation", "description": f"icu_out <= icu_in: {bad} stays",
                           "percent": round(100 * bad / max(len(df), 1), 2)}
                    soft.append(rec)
                    if rec["percent"] > FLAG_PERCENT["Chronological Order Violation"]:
                        flags.append(rec)
        else:
            try:
                table_schema(table)
            except KeyError:
                report["tables"][table] = {"skipped": "no CLIF 3.0 schema"}
                continue
            hard, soft, flags = _classify(table, _rekey_duplicates(table, df, _clifpy_errors(table, df, work)))
            for col in table_schema(table)["columns"]:
                if col.get("required") and col["name"] in df.columns and len(df) and df[col["name"]].isna().all():
                    soft.append({"type": "Required Column Without Values", "description": col["name"]})
        for rec in _replacement_characters(df):   # text decoded with the wrong encoding upstream
            soft.append(rec)
            flags.append(rec)
        report["tables"][table] = {"rows": len(df), "hard_errors": hard, "soft": soft}
        if flags:
            report["flags"][table] = flags
        log.info("validated %s: rows=%d hard=%d soft=%d flags=%d", table, len(df), len(hard), len(soft), len(flags))
    report["n_hard_errors"] = sum(len(t.get("hard_errors", [])) for t in report["tables"].values())
    report["passed"] = report["n_hard_errors"] == 0
    (clif_dir / REPORT).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
