"""S3 gate: CLIFpy 3.0 validation of canonical directories (clifpy demo data and hand-built tables)."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import clifpy
import pandas as pd
import pytest

from icutwin.clif.harmonize import harmonize_dir
from icutwin.clif.validate import REPORT, validate_canonical_dir

DEMO = Path(clifpy.__file__).parent / "data" / "clif_demo"


@pytest.fixture()
def canonical(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy2(DEMO / "clif_position.parquet", raw / "clif_position.parquet")
    harmonize_dir(raw, tmp_path / "canon", tmp_path / "duck")
    return tmp_path / "canon"


def test_conformant_directory_passes(canonical: Path, tmp_path: Path) -> None:
    report = validate_canonical_dir(canonical, tmp_path / "work")
    assert report["passed"] and report["n_hard_errors"] == 0
    assert report["tables"]["position"]["rows"] > 0
    assert (canonical / REPORT).exists()


def test_value_outside_clif30_is_a_hard_error(canonical: Path, tmp_path: Path) -> None:
    f = canonical / "clif_position.parquet"
    df = pd.read_parquet(f)
    df.loc[df.index[0], "position_category"] = "sitting"
    df.to_parquet(f)
    report = validate_canonical_dir(canonical, tmp_path / "work")
    assert not report["passed"]
    (err,) = [e for e in report["tables"]["position"]["hard_errors"] if e["type"] == "Invalid Categorical Values"]
    assert [v["value"] for v in err["values"]] == ["sitting"]


def _stay(stay: str, t_in: datetime, t_out: datetime) -> dict:
    return {"stay_id": stay, "patient_id": "P", "hospitalization_id": "H", "dataset": "eicu", "site_id": "1",
            "icu_in_dttm": t_in, "icu_out_dttm": t_out, "unit_type": "general_icu", "clock_valid": True,
            "admission_source": "ed", "surgical_status": "unknown", "stay_seq": 1, "icu_outcome": "alive"}


def test_ext_icu_stay_schema_and_chronology(tmp_path: Path) -> None:
    d = tmp_path / "canon"
    d.mkdir()
    t = lambda h: datetime(2100, 1, 1, h, tzinfo=UTC)  # noqa: E731
    pd.DataFrame([_stay("s1", t(1), t(5)), _stay("s2", t(3), t(3))]).to_parquet(d / "clif_ext_icu_stay.parquet")
    report = validate_canonical_dir(d, tmp_path / "work")
    assert report["passed"]                                   # chronology is a data-quality finding, not schema
    (flag,) = report["flags"]["ext_icu_stay"]
    assert flag["type"] == "Chronological Order Violation" and flag["percent"] == 50.0
    bad = pd.DataFrame([{**_stay("s1", t(1), t(5)), "admission_source": "helicopter"}]).drop(columns="icu_outcome")
    bad.to_parquet(d / "clif_ext_icu_stay.parquet")
    report = validate_canonical_dir(d, tmp_path / "work")
    assert {e["type"] for e in report["tables"]["ext_icu_stay"]["hard_errors"]} == {
        "Missing Required Columns", "Invalid Categorical Values"}


def test_int_column_with_nulls_is_not_a_type_mismatch(tmp_path: Path) -> None:
    from icutwin.clif.conform import connect
    from icutwin.clif.harmonize import conform_to_schema

    d = tmp_path / "canon"
    d.mkdir()
    f = d / "clif_hospitalization.parquet"
    t = lambda h: datetime(2100, 1, 1, h, tzinfo=UTC)  # noqa: E731
    pd.DataFrame({"patient_id": ["P1", "P2"], "hospitalization_id": ["H1", "H2"], "admission_dttm": [t(1), t(2)],
                  "discharge_dttm": [t(5), t(6)], "age_at_admission": [70, None],  # stored INT with a NULL
                  "admission_type_category": ["ed", "ed"], "discharge_category": ["home", "expired"]}).to_parquet(f)
    conform_to_schema(connect(), f, "hospitalization")
    report = validate_canonical_dir(d, tmp_path / "work")
    assert report["tables"]["hospitalization"]["hard_errors"] == []


def test_arterial_and_cuff_pressure_at_one_time_are_not_duplicates(tmp_path: Path) -> None:
    from icutwin.clif.conform import connect
    from icutwin.clif.harmonize import conform_to_schema

    d = tmp_path / "canon"
    d.mkdir()
    f = d / "clif_vitals.parquet"
    t = datetime(2100, 1, 1, 10, tzinfo=UTC)
    rows = [("H1", "sbp", "arterial", 120.0), ("H1", "sbp", "cuff", 110.0), ("H2", "sbp", "cuff", 100.0),
            ("H2", "sbp", "cuff", 101.0)]   # H2: a real duplicate
    pd.DataFrame({"hospitalization_id": [r[0] for r in rows], "recorded_dttm": [t] * 4,
                  "vital_category": [r[1] for r in rows], "meas_site_name": [r[2] for r in rows],
                  "vital_value": [r[3] for r in rows]}).to_parquet(f)
    conform_to_schema(connect(), f, "vitals")
    soft = validate_canonical_dir(d, tmp_path / "work")["tables"]["vitals"]["soft"]
    (dup,) = [e for e in soft if e["type"] == "Duplicate Composite Keys"]
    assert dup["percent"] == 25.0 and dup["description"].startswith("1 duplicate records")


def test_duplicate_stay_ids_are_a_hard_error(tmp_path: Path) -> None:
    d = tmp_path / "canon"
    d.mkdir()
    t = lambda h: datetime(2100, 1, 1, h, tzinfo=UTC)  # noqa: E731
    pd.DataFrame([_stay("s1", t(1), t(5)), {**_stay("s1", t(1), t(5)), "hospitalization_id": "H2"}]).to_parquet(
        d / "clif_ext_icu_stay.parquet")
    report = validate_canonical_dir(d, tmp_path / "work")
    assert not report["passed"]
    assert [e["type"] for e in report["tables"]["ext_icu_stay"]["hard_errors"]] == ["Duplicate Primary Key"]


def test_replacement_characters_are_flagged(canonical: Path, tmp_path: Path) -> None:
    f = canonical / "clif_position.parquet"
    df = pd.read_parquet(f)
    df.loc[df.index[0], "position_name"] = "Buikligging �"
    df.to_parquet(f)
    report = validate_canonical_dir(canonical, tmp_path / "work")
    assert report["passed"] and report["flags"]["position"][0]["type"] == "Replacement Characters"
