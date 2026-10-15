"""ClifWriter: schema columns only, typed output, UTC timestamps, strict vocabularies."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from icutwin.clif.conform import ClifWriter, allowed_values, connect


@pytest.fixture()
def writer(tmp_path: Path) -> ClifWriter:
    return ClifWriter(connect(), tmp_path)


def test_writes_full_schema_with_types(writer: ClifWriter, tmp_path: Path) -> None:
    sql = ("SELECT 'H1' AS hospitalization_id, TIMESTAMP '2016-01-01 10:30:00' AS recorded_dttm, "
           "'heart_rate' AS vital_category, '88' AS vital_value")
    stats = writer.write("vitals", sql)
    table = pq.read_table(tmp_path / "clif_vitals.parquet")
    assert {"hospitalization_id", "recorded_dttm", "measured_dttm", "vital_name", "vital_category",
            "vital_value", "meas_site_name"} == set(table.schema.names)  # full CLIF 3.0 vitals schema
    assert str(table.schema.field("recorded_dttm").type) == "timestamp[us, tz=UTC]"
    assert table.column("recorded_dttm")[0].as_py() == datetime(2016, 1, 1, 10, 30, tzinfo=UTC)  # no local shift
    assert table.column("vital_value")[0].as_py() == 88.0
    assert stats["rows"] == 1 and "meas_site_name" not in stats["null_required"]


def test_rejects_unknown_columns(writer: ClifWriter) -> None:
    with pytest.raises(ValueError, match="not in the CLIF"):
        writer.write("vitals", "SELECT 'H1' AS hospitalization_id, 1 AS not_a_clif_column")


def test_rejects_values_outside_vocabulary(writer: ClifWriter) -> None:
    sql = ("SELECT 'H1' AS hospitalization_id, TIMESTAMP '2016-01-01' AS recorded_dttm, "
           "'IMV' AS device_category")  # 2.1 spelling; canonical 3.0 is 'imv'
    with pytest.raises(ValueError, match="device_category"):
        writer.write("respiratory_support", sql)


def test_extension_values_are_allowed(writer: ClifWriter) -> None:
    assert "ext_base_excess" in allowed_values("labs", "lab_category")
    sql = ("SELECT 'H1' AS hospitalization_id, TIMESTAMP '2016-01-01' AS lab_result_dttm, "
           "'ext_base_excess' AS lab_category, '-2.0' AS lab_value, -2.0 AS lab_value_numeric, 'mmol/l' AS reference_unit")
    assert writer.write("labs", sql)["rows"] == 1


def test_ext_icu_stay_schema_is_enforced(writer: ClifWriter) -> None:
    ok = ("SELECT 'inspire:1' AS stay_id, 'P1' AS patient_id, 'H1' AS hospitalization_id, 'inspire' AS dataset, "
          "'inspire' AS site_id, TIMESTAMP '2016-01-01' AS icu_in_dttm, TIMESTAMP '2016-01-02' AS icu_out_dttm, "
          "false AS clock_valid, 'or' AS admission_source, 'elective_surgery' AS surgical_status, 1 AS stay_seq, "
          "'alive' AS icu_outcome")
    assert writer.write("ext_icu_stay", ok)["rows"] == 1
    with pytest.raises(ValueError, match="admission_source"):
        writer.write("ext_icu_stay", ok.replace("'or' AS admission_source", "'operating room' AS admission_source"))


def test_connect_takes_host_limits_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("ICUTWIN_DUCKDB_THREADS", "3")
    monkeypatch.setenv("ICUTWIN_DUCKDB_MEMORY_LIMIT", "2GB")
    con = connect()
    assert con.execute("SELECT current_setting('threads')").fetchone()[0] == 3
    assert con.execute("SELECT current_setting('memory_limit')").fetchone()[0] in ("2.0 GB", "1.8 GiB", "1.9 GiB")
    assert connect(threads=5).execute("SELECT current_setting('threads')").fetchone()[0] == 5   # an explicit value wins


def test_every_connection_spills_to_a_directory_of_its_own(monkeypatch, tmp_path) -> None:
    """DuckDB names spill files without a process or connection id, and an in-memory database spills to '.tmp'
    in the working directory: parallel steps of one checkout overwrote each other's blocks (server, 2026-10-05)."""
    monkeypatch.setenv("ICUTWIN_TMP", str(tmp_path))
    dirs = [Path(c.execute("SELECT current_setting('temp_directory')").fetchone()[0])
            for c in (connect(), connect(), connect(tmp_path / "x"))]
    assert len(set(dirs)) == 3 and all(d.is_dir() for d in dirs)
    assert dirs[0].parent == tmp_path and dirs[2].parent == tmp_path / "x"
