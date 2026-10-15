"""CSV member processing for staging: key-column filtering and byte-exact copies.

Filtering parses with pyarrow (RFC 4180, quoted newlines allowed) and keeps every column as a string,
so filtered rows carry exactly the source text values. Unquoted empty fields are read as null and written
back as empty fields, and values are re-quoted minimally (Python ``csv.QUOTE_MINIMAL``), which reproduces
the sources' own formatting (bare numbers, empty missing values) so downstream readers infer the same types.
Non-UTF-8 sources (AmsterdamUMCdb is Latin-1) are decoded on read and re-encoded on write; Latin-1
maps every byte to one code point, so this round trip is byte-exact.
"""

from __future__ import annotations

import csv
import gzip
import io
import shutil
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

_BLOCK = 32 << 20
_COPY_CHUNK = 16 << 20
_PYARROW_ENCODING = {"utf-8": "utf8", "latin-1": "latin1"}


@dataclass
class MemberStats:
    rows_in: int | None
    rows_out: int | None
    bytes_out: int


def read_header(stream: io.BufferedReader, encoding: str = "utf-8") -> list[str]:
    """Consume and parse the header line (handles a UTF-8 BOM and quoted names)."""
    line = stream.readline()
    if not line:
        raise ValueError("empty CSV member: no header line")
    text = line.decode("utf-8-sig" if encoding == "utf-8" else encoding)
    names = next(csv.reader([text.rstrip("\r\n")]))
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate column names in header: {names}")
    return names


def _string_reader(stream: io.BufferedReader, names: list[str], encoding: str) -> pacsv.CSVStreamingReader:
    return pacsv.open_csv(
        stream,
        read_options=pacsv.ReadOptions(column_names=names, block_size=_BLOCK, use_threads=True,
                                       encoding=_PYARROW_ENCODING[encoding]),
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
        # only *unquoted* empty fields become null; "NULL"/"NA" text and quoted "" stay strings
        convert_options=pacsv.ConvertOptions(column_types={n: pa.string() for n in names}, null_values=[""],
                                             strings_can_be_null=True, quoted_strings_can_be_null=False),
    )


def read_all_strings(stream: io.BufferedReader, encoding: str = "utf-8") -> pa.Table:
    names = read_header(stream, encoding)
    return _string_reader(stream, names, encoding).read_all()


def _open_sink(path: Path, compression: str) -> pa.NativeFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compression == "gzip":
        return pa.CompressedOutputStream(str(path), "gzip")
    if compression == "none":
        return pa.OSFile(str(path), "wb")
    raise ValueError(f"unknown compression: {compression}")


def _needs_quoting(batch: pa.RecordBatch) -> bool:
    return any(pc.any(pc.match_substring_regex(col, r'[",\r\n]')).as_py() for col in batch.columns)


def _csv_bytes(batch: pa.RecordBatch, header: bool, encoding: str) -> bytes:
    if not header and not _needs_quoting(batch):
        # fast path: identical output to QUOTE_MINIMAL when no value contains a delimiter, quote or newline
        buf = pa.BufferOutputStream()
        pacsv.write_csv(batch, buf, write_options=pacsv.WriteOptions(include_header=False, quoting_style="none"))
        data = buf.getvalue().to_pybytes()
        return data if encoding == "utf-8" else data.decode("utf-8").encode(encoding)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    if header:
        writer.writerow(batch.schema.names)
    writer.writerows(zip(*(col.to_pylist() for col in batch.columns)))  # None -> empty field
    return out.getvalue().encode(encoding)


def filter_csv(
    stream: io.BufferedReader,
    dst: Path,
    compression: str,
    key_col: str | None,
    keys: pa.Array | None,
    encoding: str = "utf-8",
) -> MemberStats:
    """Write rows whose ``key_col`` value is in ``keys`` (all rows if ``keys`` is None)."""
    names = read_header(stream, encoding)
    if key_col is not None and key_col not in names:
        raise KeyError(f"key column {key_col!r} not in header {names}")
    reader = _string_reader(stream, names, encoding)
    rows_in = rows_out = 0
    sink = _open_sink(dst, compression)
    try:
        sink.write(_csv_bytes(pa.RecordBatch.from_pylist([], schema=reader.schema), True, encoding))
        for batch in reader:
            rows_in += batch.num_rows
            if keys is not None:
                batch = batch.filter(pc.is_in(batch.column(key_col), value_set=keys))
            if batch.num_rows:
                sink.write(_csv_bytes(batch, False, encoding))
                rows_out += batch.num_rows
    finally:
        sink.close()
    return MemberStats(rows_in=rows_in, rows_out=rows_out, bytes_out=dst.stat().st_size)


def copy_member(stream: io.BufferedReader, dst: Path, compression: str) -> MemberStats:
    """Byte-exact copy, optionally gzip-compressed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if compression == "gzip":
        with gzip.open(dst, "wb", compresslevel=6) as out:
            shutil.copyfileobj(stream, out, _COPY_CHUNK)
    elif compression == "none":
        with open(dst, "wb") as out:
            shutil.copyfileobj(stream, out, _COPY_CHUNK)
    else:
        raise ValueError(f"unknown compression: {compression}")
    return MemberStats(rows_in=None, rows_out=None, bytes_out=dst.stat().st_size)
