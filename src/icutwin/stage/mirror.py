"""Native CSV layout -> parquet mirror (DuckDB), with a cross-parser row-count check.

Row counts of filtered members come from the pyarrow parse in staging; every other CSV is counted
with pyarrow here. DuckDB's parquet row count must match exactly, so a CSV mis-parse (quoting,
embedded newlines) in either reader fails loudly instead of silently dropping rows.

Latin-1 sources: DuckDB's latin-1 decoder rejects the C1 range (bytes 0x80-0x9F), which text typed on Windows
code pages carries (AmsterdamUMCdb's ``listitems`` at full scale). Latin-1 maps every byte to its code point,
so such a file is transcoded to UTF-8 (byte-exact) and read again with the same DuckDB typing.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from icutwin.stage.staging import MANIFEST

log = logging.getLogger(__name__)

MIRROR_MANIFEST = "_mirror_manifest.json"
_FALLBACK_MAX_BYTES = 50 << 20


def _latin1_to_utf8(src: Path, dst: Path) -> None:
    """A UTF-8 copy of a latin-1 CSV: every byte becomes its code point (C1 bytes included), nothing is lost."""
    with (gzip.open(src, "rb") if src.name.endswith(".gz") else src.open("rb")) as fin, dst.open("wb") as fout:
        while chunk := fin.read(64 << 20):
            fout.write(chunk.decode("latin-1").encode("utf-8"))


def _copy(con: duckdb.DuckDBPyConnection, csv_path: Path, dst: Path, encoding: str, sample_size: int,
          delimiter: str = ",") -> None:
    # ',' keeps DuckDB's sniffer as before; any other delimiter (RICD's ';') is stated, not guessed
    delim = "" if delimiter == "," else f", delim='{delimiter}'"
    con.execute(
        f"COPY (SELECT * FROM read_csv('{_sql_path(csv_path)}', header=true, sample_size={sample_size}, encoding='{encoding}'{delim})) "
        f"TO '{_sql_path(dst)}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)"
    )


def _table_rel(rel: Path) -> Path:
    name = rel.name
    if name.endswith(".csv.gz"):
        name = name[: -len(".csv.gz")]
    elif name.endswith(".csv"):
        name = name[: -len(".csv")]
    else:
        raise ValueError(f"not a CSV file: {rel}")
    return rel.with_name(f"{name}.parquet")


def _count_csv_rows(path: Path, encoding: str, delimiter: str = ",") -> int:
    read = pacsv.ReadOptions(block_size=32 << 20, encoding=encoding.replace('-', ''))
    parse = pacsv.ParseOptions(newlines_in_values=True, delimiter=delimiter)
    # all-string conversion: counting must not depend on type inference
    with pacsv.open_csv(path, read_options=read, parse_options=parse) as reader:
        names = reader.schema.names
    convert = pacsv.ConvertOptions(column_types={n: "string" for n in names})
    rows = 0
    with pacsv.open_csv(path, read_options=read, parse_options=parse, convert_options=convert) as reader:
        for batch in reader:
            rows += batch.num_rows
    return rows


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def mirror_dataset(
    native_dir: Path,
    out_dir: Path,
    temp_dir: Path,
    threads: int = 8,
    memory_limit: str = "12GB",
    sample_size: int = -1,
) -> dict:
    started = time.time()
    staged = json.loads((native_dir / MANIFEST).read_text(encoding="utf-8"))
    encoding = staged.get("encoding", "utf-8")
    delimiter = staged.get("delimiter", ",")
    staged_rows = {Path(m["path"]).resolve(): m["rows_out"] for m in staged["members"] if m.get("rows_out") is not None}
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{_sql_path(temp_dir)}'")
    con.execute("SET preserve_insertion_order=true")

    tables = []
    for csv_path in sorted(p for p in native_dir.rglob("*") if p.name.endswith((".csv", ".csv.gz"))):
        rel = csv_path.relative_to(native_dir)
        dst = out_dir / _table_rel(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        converter = "duckdb"
        try:
            _copy(con, csv_path, dst, encoding, sample_size, delimiter)
        except duckdb.Error as err:
            if encoding == "latin-1" and "not latin-1 encoded" in str(err):   # C1 bytes (module docstring)
                utf8 = temp_dir / f"{csv_path.name}.utf8.csv"
                try:
                    _latin1_to_utf8(csv_path, utf8)
                    _copy(con, utf8, dst, "utf-8", sample_size)
                finally:
                    utf8.unlink(missing_ok=True)
                converter = "duckdb-utf8"
                log.info("%s: latin-1 with C1 bytes, read as its UTF-8 transcoding", rel)
            # Small reference files with irregular formatting (e.g. mixed CRLF/LF line endings) defeat the
            # DuckDB sniffer; pyarrow parses them. Large data tables must never fall back silently.
            elif csv_path.stat().st_size > _FALLBACK_MAX_BYTES:
                raise
            else:
                table = pacsv.read_csv(csv_path, read_options=pacsv.ReadOptions(encoding=encoding.replace('-', '')),
                                       parse_options=pacsv.ParseOptions(newlines_in_values=True,
                                                                        delimiter=delimiter))
                pq.write_table(table, dst, compression="zstd")
                converter = "pyarrow-fallback"
                log.warning("DuckDB could not parse %s; converted with pyarrow", rel)
        expected = staged_rows.get(csv_path.resolve())
        source = "staging"
        if expected is None:
            expected, source = _count_csv_rows(csv_path, encoding, delimiter), "pyarrow"
        rows = pq.ParquetFile(dst).metadata.num_rows
        if rows != expected:
            raise RuntimeError(f"row-count mismatch for {rel}: parquet={rows} {source}={expected}")
        tables.append({"table": str(_table_rel(rel)).replace("\\", "/"), "rows": rows, "checked_against": source,
                       "converter": converter,
                       "seconds": round(time.time() - t0, 2)})
        log.info("mirrored %s rows=%d (%.1fs)", rel, rows, time.time() - t0)

    manifest = {
        "dataset": staged["dataset"],
        "native_dir": str(native_dir),
        "staging_mode": staged["mode"],
        "frac": staged["frac"],
        "tables": tables,
        "seconds": round(time.time() - started, 1),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / MIRROR_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
