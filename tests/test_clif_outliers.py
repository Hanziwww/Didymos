"""SQL outlier handling must null exactly the cells clifpy's official apply_outlier_handling nulls."""

from __future__ import annotations

import importlib
from pathlib import Path

import clifpy
import numpy as np
import pandas as pd
import pytest
from clifpy.utils.outlier_handler import apply_outlier_handling

from icutwin.clif.conform import connect
from icutwin.clif.outliers import apply_outliers, outlier_config, outlier_overlaps

DEMO = Path(clifpy.__file__).parent / "data" / "clif_demo"
TABLES = {
    "vitals": ("clifpy.tables.vitals", "Vitals", ["vital_value"]),
    "labs": ("clifpy.tables.labs", "Labs", ["lab_value_numeric"]),
    "medication_admin_continuous": ("clifpy.tables.medication_admin_continuous", "MedicationAdminContinuous", ["med_dose"]),
    "patient_assessments": ("clifpy.tables.patient_assessments", "PatientAssessments", ["numerical_value"]),
    "respiratory_support": ("clifpy.tables.respiratory_support", "RespiratorySupport", ["fio2_set", "peep_set", "tidal_volume_obs"]),
}


def _synthetic_assessments() -> pd.DataFrame:
    """Lower-case CLIF 3.0 categories whose clifpy config keys are upper-case ('RASS', 'NRS', 'SAS')."""
    rows = [("rass", -6.0), ("rass", -5.0), ("rass", 4.0), ("rass", 5.0), ("nrs", -1.0), ("nrs", 10.0),
            ("nrs", 30.0), ("sas", 0.0), ("sas", 7.0), ("gcs_total", 2.0), ("gcs_total", 15.0), ("cam_total", None)]
    return pd.DataFrame({
        "hospitalization_id": ["H"] * len(rows),
        "recorded_dttm": pd.date_range("2100-01-01", periods=len(rows), freq="h", tz="UTC"),
        "assessment_category": [r[0] for r in rows], "numerical_value": [r[1] for r in rows],
    })


@pytest.mark.parametrize("table", [*sorted(TABLES), "patient_assessments_synthetic"])
def test_sql_outliers_match_clifpy(table: str, tmp_path: Path) -> None:
    if table == "patient_assessments_synthetic":
        df = _synthetic_assessments()
        table = "patient_assessments"
    else:
        df = pd.read_parquet(DEMO / f"clif_{table}.parquet").drop(columns=["__index_level_0__"], errors="ignore")
    module, cls, cols = TABLES[table]
    src = tmp_path / "src.parquet"
    df.to_parquet(src, index=False)

    obj = getattr(importlib.import_module(module), cls)(
        data_directory=str(tmp_path), filetype="parquet", timezone="UTC", output_directory=str(tmp_path / "o"),
        data=df.copy())
    apply_outlier_handling(obj)
    official = obj.df

    apply_outliers(connect(), src, tmp_path / "dst.parquet", table, extensions=False)
    ours = pd.read_parquet(tmp_path / "dst.parquet")

    assert len(ours) == len(official) == len(df)
    for col in cols:
        want = pd.to_numeric(official[col], errors="coerce").isna().to_numpy()
        got = pd.to_numeric(ours[col], errors="coerce").isna().to_numpy()
        mismatch = np.flatnonzero(want != got)
        assert mismatch.size == 0, f"{table}.{col}: {mismatch.size} cells differ, e.g. rows {mismatch[:5].tolist()}"


def test_extensions_only_fill_gaps() -> None:
    assert outlier_overlaps() == []
    merged, official = outlier_config(True), outlier_config(False)
    assert merged["labs"]["lab_value_numeric"]["hemoglobin"] == official["labs"]["lab_value_numeric"]["hemoglobin"]
    assert "glucose" in merged["labs"]["lab_value_numeric"] and "glucose" not in official["labs"]["lab_value_numeric"]


def test_extension_ranges_apply(tmp_path: Path) -> None:
    df = pd.DataFrame({"hospitalization_id": ["H"] * 4, "recorded_dttm": pd.date_range("2100-01-01", periods=4, freq="h", tz="UTC"),
                       "measurement_category": ["cvp", "cvp", "pa_systolic", "pa_systolic"],
                       "measurement_value": [12.0, 80.0, 40.0, 270.0]})
    src = tmp_path / "src.parquet"
    df.to_parquet(src, index=False)
    rep = apply_outliers(connect(), src, tmp_path / "dst.parquet", "invasive_hemodynamics")
    got = pd.read_parquet(tmp_path / "dst.parquet")["measurement_value"].tolist()
    assert got[0] == 12.0 and got[2] == 40.0 and pd.isna(got[1]) and pd.isna(got[3])
    assert rep["nulled"] == {"measurement_value": 2}
