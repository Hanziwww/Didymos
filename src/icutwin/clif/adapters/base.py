"""Base class for our own raw -> CLIF adapters (datasets without a published CLIF ETL).

An adapter reads the parquet mirror (S1), joins versioned mapping CSVs (``clif/maps/<dataset>/``, every
row carries provenance), and emits each CLIF table as SQL that :class:`ClifWriter` validates and
writes. Adapters emit CLIF 3.0 category values directly, so the S2b crosswalk is a no-op for them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

import pyarrow as pa

from icutwin.clif.conform import ClifWriter, connect

log = logging.getLogger(__name__)

ADAPTER_REPORT = "_adapter_report.json"


def minutes_after(origin_sql: str, minutes_sql: str) -> str:
    """SQL for ``origin + minutes`` with microsecond precision (minutes may be fractional)."""
    # DOUBLE, not DECIMAL: decimal literals (e.g. 365.25) would overflow DECIMAL(18) at microsecond scale
    return f"({origin_sql} + to_microseconds(CAST(round(CAST(({minutes_sql}) AS DOUBLE) * 60000000) AS BIGINT)))"


class ClifAdapter:
    dataset: str = ""
    tables: tuple[str, ...] = ()
    part: str | None = None          # supplements write clif_<table>+<part>.parquet next to a reference tool's output
    report_name: str = ADAPTER_REPORT

    def __init__(self, parquet_dir: Path, out_dir: Path, temp_dir: Path, threads: int | None = None):
        self.parquet_dir = parquet_dir
        self.out_dir = out_dir
        self.con = connect(temp_dir, threads)
        self.writer = ClifWriter(self.con, out_dir)
        self._maps: dict[str, str] = {}
        self.audit: dict[str, dict] = {}  # per-table mapping coverage, written to the adapter report

    # -- inputs -------------------------------------------------------------------------------------
    def src(self, rel: str) -> str:
        """``read_parquet`` over a mirrored source table (``rel`` may be a glob, e.g. HiRID parts)."""
        path = (self.parquet_dir / rel).as_posix()
        return f"read_parquet('{path}', union_by_name=true)"

    def _register_csv(self, table: str, key: str, *parts: str, text: str | None = None) -> str:
        """Load a packaged CSV (or ``text``) as an all-VARCHAR table (empty fields -> NULL) with its row order
        ``_row``.

        Parsed with Python's csv module rather than DuckDB's sniffer: mapping files hold regexes with commas
        and quotes, and a dialect guess must never decide how a mapping is read.
        """
        if key not in self._maps:
            if text is None:
                text = resources.files("icutwin.clif").joinpath(*parts).read_text(encoding="utf-8")
            self._maps[key] = hashlib.sha256(text.encode()).hexdigest()
            header, *rows = list(csv.reader(text.splitlines()))
            columns = {c: pa.array([(r[i] or None) if i < len(r) else None for r in rows], pa.string())
                       for i, c in enumerate(header)}
            columns["_row"] = pa.array(range(1, len(rows) + 1), pa.int64())
            self.con.register(f"_csv_{table}", pa.table(columns))
            self.con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _csv_{table}")
            self.con.unregister(f"_csv_{table}")
        return table

    def map(self, name: str) -> str:
        """Register ``clif/maps/<dataset>/<name>.csv`` as table ``map_<name>``."""
        return self._register_csv(f"map_{name}", name, "maps", self.dataset, f"{name}.csv")

    def lexicon(self, name: str) -> str:
        """Register the shared effective lexicon (``clif/lexicon/<name>.csv``, for ``drugs`` with the medication
        registry's extension categories; :func:`icutwin.clif.lexicon.rows`) as table ``lex_<name>``."""
        from icutwin.clif.lexicon import csv_text
        return self._register_csv(f"lex_{name}", f"lexicon/{name}", text=csv_text(name))

    def regex_lookup(self, name: str, texts_sql: str, map_name: str, cols: tuple[str, ...],
                     table: str | None = None) -> str:
        """Resolve distinct texts through an ordered regex map; returns table ``lk_<name>(txt, *cols)``.

        ``texts_sql`` yields a column ``txt`` (already normalised, e.g. lower-cased). The map (a dataset map,
        or an already registered ``table`` such as a lexicon) has columns ``priority`` and ``pattern`` (RE2,
        partial match); the lowest matching priority wins, so specific patterns must precede generic ones.
        Unmatched texts get NULLs; ``__skip__`` rows are returned as such.
        """
        m = table or self.map(map_name)
        picked = ", ".join(f"m.{c}" for c in cols)
        self.con.execute(f"""
            CREATE OR REPLACE TABLE lk_{name} AS
            SELECT txt, {', '.join(cols)} FROM (
                SELECT d.txt, {picked},
                       row_number() OVER (PARTITION BY d.txt ORDER BY CAST(m.priority AS INTEGER) NULLS LAST, m._row) AS k
                FROM (SELECT DISTINCT txt FROM ({texts_sql}) WHERE txt IS NOT NULL) d
                LEFT JOIN {m} m ON regexp_matches(d.txt, m.pattern))
            WHERE k = 1""")
        return f"lk_{name}"

    def coverage(self, table: str, key: str, sql: str, top: int = 25) -> None:
        """Record mapping coverage: ``sql`` yields (txt, n, mapped BOOLEAN); unmapped texts are listed by count."""
        rows = self.con.execute(f"SELECT txt, sum(n), bool_or(mapped) FROM ({sql}) GROUP BY txt").fetchall()
        total = sum(int(n) for _, n, _ in rows)
        unmapped = sorted(((t, int(n)) for t, n, ok in rows if not ok), key=lambda r: -r[1])
        self.audit.setdefault(table, {})[key] = {
            "rows": total, "rows_unmapped": sum(n for _, n in unmapped),
            "top_unmapped": dict(unmapped[:top]),
        }

    # -- run ----------------------------------------------------------------------------------------
    def build(self, table: str) -> tuple[str, tuple[str, ...]]:
        """Return (sql, order_by) for one CLIF table; subclasses implement ``sql_<table>``."""
        return getattr(self, f"sql_{table}")()

    def run(self, tables: tuple[str, ...] | None = None) -> dict:
        started = time.time()
        report: dict = {"dataset": self.dataset, "adapter": type(self).__name__, "tables": {}}
        for table in tables or self.tables:
            t0 = time.time()
            sql, order_by = self.build(table)
            stats = self.writer.write(table, sql, order_by, part=self.part)
            stats["seconds"] = round(time.time() - t0, 2)
            report["tables"][table] = stats
            log.info("%s: wrote clif_%s rows=%d (%.1fs)", self.dataset, table, stats["rows"], stats["seconds"])
        report["maps_sha256"] = dict(sorted(self._maps.items()))
        report["audit"] = self.audit
        report["seconds"] = round(time.time() - started, 1)
        report["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        (self.out_dir / self.report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
