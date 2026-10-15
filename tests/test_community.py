"""Community MEDS ETL runner: script paths on Windows (WSL2) and on a Linux host, run scripts and the HiRID input
staging (no ETL is run)."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import duckdb
import pytest

from icutwin.clif import community as cm


@pytest.fixture(autouse=True)
def _windows_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The script expectations below are the Windows (WSL2) rendering, on any machine running the tests."""
    monkeypatch.setattr(cm, "NATIVE", False)
    if os.name != "nt":   # Windows drive paths do not resolve on Linux: the mount mapping alone
        monkeypatch.setattr(cm, "wsl_path", lambda p: "/mnt/" + p.as_posix()[0].lower() + p.as_posix()[2:])


@pytest.mark.skipif(os.name != "nt", reason="Windows paths")
def test_wsl_path() -> None:
    assert cm.wsl_path(Path("D:/project/data/x")) == "/mnt/d/project/data/x"


def test_linux_host_runs_scripts_natively(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On a Linux host (the server) the scripts get the paths as they are and run under the host's bash."""
    import subprocess

    monkeypatch.setattr(cm, "NATIVE", True)
    out = tmp_path / "o" / "eicu"
    s = cm.script("eicu", cm.ETLS["eicu"], tmp_path / "r" / "eICU_MEDS", tmp_path / "i", out)
    assert f'root_output_dir="{out.resolve().as_posix()}"' in s and "/mnt/" not in s
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    assert cm.run_script("echo ok\n", tmp_path / "s.sh", tmp_path / "s.log") == 0
    assert calls == [["bash", (tmp_path / "s.sh").resolve().as_posix()]]


@pytest.mark.parametrize(("name", "expected"), [
    ("mimic", 'meds-extract-run spec=MIMIC-IV output_dir="/mnt/d/o/mimic" do_download=false input_dir="/mnt/d/i"'),
    ("inspire", 'MEDS_extract-INSPIRE "/mnt/d/o/inspire" --raw-input-dir "/mnt/d/i" --do-download false'),
    ("eicu", 'MEDS_extract-eICU root_output_dir="/mnt/d/o/eicu" raw_input_dir="/mnt/d/i" do_download=False'),
])
def test_run_command_per_runner(name: str, expected: str) -> None:
    etl = cm.ETLS[name]
    s = cm.script(name, etl, Path("D:/r") / etl.repo, Path("D:/i"), Path("D:/o") / name)
    assert expected in s and s.startswith("#!/usr/bin/env bash\nset -euo pipefail")
    assert "\r" not in s
    if etl.runner == "extract":
        assert "uv sync --frozen" in s                       # locked install
    else:
        assert "uv pip sync" in s and cm.MIRROR in s          # locked dependencies, from the mirror
    assert "_environment.txt" in s


def test_environments_are_locked() -> None:
    s = cm.script("aumc", cm.ETLS["aumc"], Path("D:/r/AUMCdb_MEDS"), Path("D:/i"), Path("D:/o/aumc"))
    assert 'uv pip sync --python "$VENV/bin/python"' in s and '--no-deps "/mnt/d/r/AUMCdb_MEDS"' in s
    assert 'cd "/mnt/d/r/AUMCdb_MEDS" && MEDS_extract-AUMCdb' in s
    assert "diff <(grep -v '^#' \"$LOCK\"" in s                     # a run fails on any version drift
    for name, etl in cm.ETLS.items():                                # every ETL has a lock holding its pins
        locked = cm.lock_file(etl).read_text(encoding="utf-8")
        assert all(p in locked.splitlines() for p in etl.pins), name
    assert cm.ETLS["hirid"].pins == ("polars==1.26.0",) and cm.ETLS["aumc"].pins == ("meds-transforms==0.2.4",)


def test_a_lock_without_the_pins_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "HiRID_MEDS.txt").write_text("# header\npolars==1.29.0\n", encoding="utf-8")
    monkeypatch.setattr(cm, "LOCKS", tmp_path)
    with pytest.raises(ValueError, match="polars==1.26.0"):
        cm.lock_file(cm.ETLS["hirid"])


def test_hirid_staging(tmp_path: Path) -> None:
    raw, stage = tmp_path / "raw", tmp_path / "stage"
    obs, ph, ref = raw / "raw_stage/observation_tables/csv", raw / "raw_stage/pharma_records/csv", raw / "reference"
    for d in (obs, ph, ref):
        d.mkdir(parents=True)
    (obs / "part-0.csv").write_text("datetime,entertime,patientid,status,stringvalue,type,value,variableid\n"
                                    "2150-01-01 08:00:00.000,2150-01-01 08:01:00.000,7,8,,,80.0,200\n")
    (obs / "part-1.csv").write_text("datetime,entertime,patientid,status,stringvalue,type,value,variableid\n")
    (ph / "part-0.csv").write_text(
        "patientid,pharmaid,givenat,enteredentryat,givendose,cumulativedose,fluidamount_calc,cumulfluidamount_calc,"
        "doseunit,route,infusionid,typeid,subtypeid,recordstatus\n"
        "7,1000251,2150-01-01 09:00:00.000,2150-01-01 09:01:00.000,50.0,50.0,,,µg,iv-inj,3,1,8.0,780\n")
    (ref / "general_table.csv").write_text("patientid,admissiontime,sex,age,discharge_status\n7,2150-01-01 07:00:00,M,60,alive\n")
    (stage / "stale").mkdir(parents=True)                     # rebuilt from scratch
    cm.stage_hirid(raw, stage)
    assert not (stage / "stale").exists()
    files = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
    assert files == ["raw_stage/observation_tables/parquet/part-0.parquet",   # header-only part-1 skipped
                     "raw_stage/pharma_records_parquet/parquet/part-0.parquet", "reference_data.tar.gz"]
    types = dict(duckdb.sql(f"DESCRIBE SELECT * FROM '{(stage / files[0]).as_posix()}'").fetchall()[i][:2]
                 for i in range(8))
    assert types["datetime"] == "TIMESTAMP" and types["patientid"] == "BIGINT" and types["value"] == "DOUBLE"
    with tarfile.open(stage / "reference_data.tar.gz") as tar:
        assert tar.getnames() == ["general_table.csv"]


def test_hirid_staging_keeps_quoted_commas_after_the_sniff_sample(tmp_path: Path) -> None:
    """A quoted route with a comma ("i.ventr, EVD") first seen after DuckDB's sniffing sample stays one field."""
    raw, stage = tmp_path / "raw", tmp_path / "stage"
    ph, ref = raw / "raw_stage/pharma_records/csv", raw / "reference"
    obs = raw / "raw_stage/observation_tables/csv"
    for d in (obs, ph, ref):
        d.mkdir(parents=True)
    (obs / "part-0.csv").write_text("datetime,entertime,patientid,status,stringvalue,type,value,variableid\n")
    head = ("patientid,pharmaid,givenat,enteredentryat,givendose,cumulativedose,fluidamount_calc,"
            "cumulfluidamount_calc,doseunit,route,infusionid,typeid,subtypeid,recordstatus\n")
    row = "7,1000263,2194-12-26 08:00:00.000,2194-12-26 10:11:10.850,0.0,0.0,0.0,0.0,mg,{route},{i},1,8.0,544\n"
    (ph / "part-0.csv").write_text(head + "".join(row.format(route="iv", i=i) for i in range(25000))
                                   + row.format(route='"i.ventr, EVD"', i=25000))
    (ref / "general_table.csv").write_text("patientid,admissiontime,sex,age\n7,2194-12-26 07:00:00,M,60\n")
    cm.stage_hirid(raw, stage)
    got = duckdb.sql(f"SELECT route, infusionid FROM '{(stage / 'raw_stage/pharma_records_parquet/parquet/part-0.parquet').as_posix()}' "
                     "WHERE infusionid = 25000").fetchall()
    assert got == [("i.ventr, EVD", 25000)]
