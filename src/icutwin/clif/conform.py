"""Schema-driven CLIF writer used by our own adapters (target: CLIF 3.0 schemas shipped with clifpy).

Every table is written through :class:`ClifWriter`, which
* rejects columns the schema does not define (no silent extras),
* casts each column to its schema type (timestamps as UTC ``TIMESTAMPTZ``),
* fails if a standardized column holds a value outside the 3.0 permissible values or the declared
  ``ext_`` extensions (``extensions.yaml``), so vocabulary drift cannot reach downstream stages.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import yaml
from clifpy.utils.crosswalk import load_schema

CANONICAL_VERSION = "3.0"
_DUCK = {"VARCHAR": "VARCHAR", "DATETIME": "TIMESTAMPTZ", "DATE": "DATE", "DOUBLE": "DOUBLE", "FLOAT": "DOUBLE",
         "INT": "INTEGER", "INTEGER": "INTEGER", "BIGINT": "BIGINT", "BOOLEAN": "BOOLEAN"}


@lru_cache(maxsize=1)
def ext_schemas() -> dict:
    text = resources.files("icutwin.clif").joinpath("ext_schemas.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


@lru_cache(maxsize=1)
def extension_columns() -> dict[str, list[dict]]:
    """``extension_columns.yaml``: declared extension columns of CLIF 3.0 tables."""
    text = resources.files("icutwin.clif").joinpath("extension_columns.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


@lru_cache(maxsize=None)
def table_schema(table: str) -> dict:
    """The CLIF 3.0 schema of a table (2.1 where 3.0 lacks it) with our declared extension columns, or the
    schema of one of our ``ext_`` tables."""
    if table.startswith("ext_"):
        return ext_schemas()[table]
    schema = load_schema(table, CANONICAL_VERSION) or load_schema(table, "2.1")
    if schema is None:
        raise KeyError(f"no CLIF schema for table {table!r}")
    extra = extension_columns().get(table) or []
    if extra:
        names = {c["name"] for c in schema["columns"]}
        clash = [c["name"] for c in extra if c["name"] in names]
        if clash:
            raise ValueError(f"{table}: declared extension columns {clash} are CLIF 3.0 columns")
        schema = {**schema, "columns": [*schema["columns"],
                                        *({k: v for k, v in c.items() if k != "why"} for c in extra)]}
    return schema


MEDICATION_TABLES = ("medication_admin_continuous", "medication_admin_intermittent")
# registry kind -> origin named in MEDS code descriptions ("[not CLIF 3.0 vocabulary: <origin> extension]")
EXTENSION_ORIGINS = {"consortium": "consortium practice", "local": "local", "concept": "local concept"}


@lru_cache(maxsize=1)
def medication_extensions() -> tuple[dict, ...]:
    """``medication_extensions.csv``: every medication category outside the CLIF 3.0 list of its table.

    Columns: ``table``, ``med_category``, ``kind`` (consortium: emitted by an official CLIF ETL; local: a CLIF
    3.0 term of another medication table; concept: ``ext_`` drug without any CLIF 3.0 category),
    ``ingredients`` (shared drug lexicon ingredients the category applies to, ``|``-separated), ``med_group``,
    ``routes`` and ``units`` (``|``-separated), ``untyped_iv`` ('continuous': an IV record whose source states
    no administration type stays an infusion of this drug), ``clif_3_0`` (where CLIF 3.0 lists the term),
    ``source`` and ``why``. The lexicon gives these categories to the ingredients (:func:`icutwin.clif.lexicon.rows`).
    """
    text = resources.files("icutwin.clif").joinpath("medication_extensions.csv").read_text(encoding="utf-8")
    return tuple(csv.DictReader(text.splitlines()))


def _split(value: str) -> list[str]:
    return [v for v in (value or "").split("|") if v]


@lru_cache(maxsize=1)
def extensions() -> dict:
    """Declared additions to the CLIF 3.0 vocabularies: ``extensions.yaml`` plus the medication registry
    (:func:`medication_extensions`); ``medication_orders`` accepts the union of the administration tables'
    additions that its own CLIF 3.0 list lacks (entries of ``extensions.yaml`` take precedence)."""
    text = resources.files("icutwin.clif").joinpath("extensions.yaml").read_text(encoding="utf-8")
    out = yaml.safe_load(text) or {}
    for r in medication_extensions():
        cats = out.setdefault(r["table"], {}).setdefault("med_category", {})
        if r["med_category"] in cats:
            raise ValueError(f"{r['table']}.{r['med_category']} is declared twice (medication_extensions.csv)")
        cats[r["med_category"]] = {
            "origin": EXTENSION_ORIGINS[r["kind"]], "kind": r["kind"], "group": r["med_group"] or None,
            "ingredients": _split(r["ingredients"]), "routes": _split(r["routes"]), "units": _split(r["units"]),
            "untyped_iv": r["untyped_iv"] or None, "clif_3_0": r["clif_3_0"], "source": r["source"], "why": r["why"]}
    orders = out.setdefault("medication_orders", {}).setdefault("med_category", {})
    standard = set(_permissible("medication_orders", "med_category"))
    for table in ("medication_admin_intermittent", "medication_admin_continuous"):   # intermittent group first
        for cat, spec in ((out.get(table) or {}).get("med_category") or {}).items():
            if cat not in standard and cat not in orders:
                orders[cat] = {**spec, "why": f"as {table}: {spec.get('why', '')}"}
    return out


def _permissible(table: str, column: str) -> list:
    for col in table_schema(table)["columns"]:
        if col["name"] == column:
            return list(col.get("permissible_values") or [])
    return []


def allowed_values(table: str, column: str) -> set | None:
    for col in table_schema(table)["columns"]:
        if col["name"] == column and col.get("permissible_values"):
            ext = (extensions().get(table) or {}).get(column) or {}
            return set(col["permissible_values"]) | set(ext)
    return None


def spill_dir(base: Path | None = None) -> Path:
    """A spill directory of its own for one DuckDB connection, under ``base`` (default: ``ICUTWIN_TMP``, else the
    system's temp directory). DuckDB names its spill files without a process or connection id
    (``duckdb_temp_storage_S64K-2.tmp``), and an in-memory database spills to ``.tmp`` in the working directory
    by default, so two connections spilling to one directory overwrite each other's blocks."""
    base = Path(base) if base is not None else Path(os.environ.get("ICUTWIN_TMP") or tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"duckdb-{os.getpid()}-", dir=base))


def connect(temp_dir: Path | None = None, threads: int | None = None) -> duckdb.DuckDBPyConnection:
    """The pipeline's DuckDB connection. Where several pipeline processes share a host, ``ICUTWIN_DUCKDB_THREADS``
    and ``ICUTWIN_DUCKDB_MEMORY_LIMIT`` (e.g. ``120GB``) bound each one (DuckDB's defaults: every core, 80% of
    the memory, per process). Every connection spills to a directory of its own (:func:`spill_dir`, under
    ``temp_dir`` when given)."""
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")  # naive->TIMESTAMPTZ casts must never use the machine's local zone
    con.execute(f"SET temp_directory='{spill_dir(temp_dir).as_posix()}'")
    threads = threads or int(os.environ.get("ICUTWIN_DUCKDB_THREADS") or 0)
    if threads:
        con.execute(f"SET threads={threads}")
    if limit := os.environ.get("ICUTWIN_DUCKDB_MEMORY_LIMIT"):
        con.execute(f"SET memory_limit='{limit}'")
    return con


@dataclass
class ClifWriter:
    con: duckdb.DuckDBPyConnection
    out_dir: Path
    written: dict = field(default_factory=dict)

    def write(self, table: str, sql: str, order_by: tuple[str, ...] = (), part: str | None = None) -> dict:
        """Validate and write ``clif_<table>.parquet`` (or the supplement part ``clif_<table>+<part>.parquet``)."""
        schema = table_schema(table)
        columns = [(c["name"], _DUCK[c["data_type"].upper()]) for c in schema["columns"]]
        names = {n for n, _ in columns}
        src_cols = self.con.sql(sql).columns
        unknown = [c for c in src_cols if c not in names]
        if unknown:
            raise ValueError(f"{table}: columns not in the CLIF {CANONICAL_VERSION} schema: {unknown}")
        select = ", ".join(
            f'CAST("{n}" AS {t}) AS "{n}"' if n in src_cols else f'CAST(NULL AS {t}) AS "{n}"' for n, t in columns
        )
        order = f" ORDER BY {', '.join(order_by)}" if order_by else ""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        dst = self.out_dir / (f"clif_{table}+{part}.parquet" if part else f"clif_{table}.parquet")
        self.con.execute(f"COPY (SELECT {select} FROM ({sql}) src{order}) TO '{dst.as_posix()}' "
                         f"(FORMAT parquet, COMPRESSION zstd)")
        stats = {"rows": pq.ParquetFile(dst).metadata.num_rows, "invalid": {}, "null_required": {}}
        for col in schema["columns"]:
            name = col["name"]
            ok = allowed_values(table, name)
            if ok is not None:
                bad = self.con.execute(
                    f'SELECT "{name}", count(*) FROM read_parquet(?) WHERE "{name}" IS NOT NULL GROUP BY 1',
                    [dst.as_posix()]).fetchall()
                bad = {str(v): n for v, n in bad if v not in ok and str(v) not in ok}
                if bad:
                    stats["invalid"][name] = bad
            if col.get("required") and stats["rows"]:
                nulls = self.con.execute(f'SELECT count(*) FROM read_parquet(?) WHERE "{name}" IS NULL',
                                         [dst.as_posix()]).fetchone()[0]
                if nulls:
                    stats["null_required"][name] = round(nulls / stats["rows"], 4)
        if stats["invalid"]:
            raise ValueError(f"{table}: values outside CLIF {CANONICAL_VERSION} + extensions: "
                             f"{json.dumps(stats['invalid'])[:2000]}")
        self.written[table] = stats
        return stats
