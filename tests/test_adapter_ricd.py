"""RICD adapter on a hand-built mini dataset with known expected CLIF output (amendment 16a)."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from icutwin.clif.adapters.ricd import RicdAdapter

ORIGIN = datetime(2020, 1, 1, tzinfo=UTC)
START_10 = 8 * 60 + 10                               # hospitalization 10: admitted at 08:10
START_11 = (5 + 30) * 1440 + 12 * 60 + 30            # 11: after 10 (100 h = 5 days, plus 30), admitted at 12:30


def at(start: int, minutes: float) -> datetime:
    return ORIGIN + timedelta(minutes=start + minutes)


def _write(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture(scope="module")
def clif(tmp_path_factory: pytest.TempPathFactory) -> Path:
    src = tmp_path_factory.mktemp("ricd_pq")
    (src / "csv").mkdir()
    pt = dict(new_patient_id=1, sex="male", body_weight=80.0, height=175, transfer=True,
              admission_department="ICU", discharge_department="neurorehabilitation")
    _write(src / "csv/all_patients.parquet", [
        dict(pt, new_hosp_id=10, visit_number=1, age=60, fatal_outcome=False),
        dict(pt, new_hosp_id=11, visit_number=2, age=61, fatal_outcome=True, transfer=False,
             admission_department="neurorehabilitation")])
    seg = lambda h, n, dept, s, e, icu: dict(new_hosp_id=h, unit_stay_number=n, actual_department=dept,  # noqa: E731
                                             actual_department_admission_hour=s, actual_department_discharge_hour=e,
                                             ICU_period=icu)
    _write(src / "csv/patient_pathway.parquet", [
        seg(10, 1, "ICU", 0, 30, True), seg(10, 2, "neurorehabilitation", 30, 100, False),
        seg(11, 1, "neurorehabilitation", 0, 10, False), seg(11, 2, "ICU", 10, 50, True)])
    mon = lambda h, m, clock, p, v_: dict(new_hosp_id=h, post_admission_minutes=m, hour=clock, parameter=p,  # noqa: E731
                                          value=v_)
    _write(src / "csv/monitoring_data.parquet", [
        mon(10, 60, time(9, 10), "heart rate", 88.0), mon(10, 120, time(10, 10), "heart rate", 92.0),
        mon(10, 60, time(9, 10), "systolic BP", 120.0), mon(10, 120, time(10, 10), "diuresis", 200.0),
        mon(10, 120, time(10, 10), "pulse", 91.0),                       # duplicates heart rate: not mapped
        mon(11, 30, time(13, 0), "heart rate", 100.0)])
    lab = lambda h, hr, p, u, b, x: dict(new_hosp_id=h, post_admission_hours=hr, analysis_eng="panel",  # noqa: E731
                                         parameter_eng_short=p, unit_eng=u, biomaterial_eng=b, result_num=x)
    _write(src / "csv/lab_data.parquet", [
        lab(10, 2, "Creatinine", "µmol/L", "Venous blood", 88.4),
        lab(10, 2, "Glucose", "mmol/L", "Capillary blood", 5.0),
        lab(10, 2, "Glucose", "mmol/L", "Urine", 1.0)])                  # urine: not mapped
    rx = lambda h, hr, inn, route, method, dose=None, unit=None: dict(  # noqa: E731
        new_hosp_id=h, post_admission_hours=str(hr), inn_eng=inn, route_eng=route, method_eng=method, dose=dose,
        dose_unit_eng=unit)
    _write(src / "csv/therapy_prescriptions.parquet", [
        rx(10, 5, "norepinephrine", "iv", "infusion pump", 4.0, "mg"),
        rx(10, 29, "norepinephrine", "iv", "infusion pump", 4.0, "mg"),  # renewed a day later: the same episode
        rx(10, 3, "furosemide", "iv", "bolus", 20.0, "mg"),
        rx(10, 4, "heparin sodium", "s/c", None, 5.0, "thousand IU"),
        rx(10, 4, "Tropicamide + Phenylephrine", "ophth", None)])        # no CLIF route
    _write(src / "csv/detailed_sofa.parquet", [
        {"new_hosp_id": 10, "post_admission_hours": 7, "mechanical_ventilation": True, "FiO2 %": 40, "GCS_score": 10},
        {"new_hosp_id": 10, "post_admission_hours": 31, "mechanical_ventilation": False, "FiO2 %": 21,
         "GCS_score": 14}])
    _write(src / "csv/all_scales.parquet", [dict(new_hosp_id=10, post_admission_hours=8, scale_eng="GCS", result=12.0)])
    _write(src / "csv/ICD10_diagnoses.parquet", [
        dict(new_hosp_id=11, diagnosis_type="discharge diagnosis", ICD_10="I63.5"),
        dict(new_hosp_id=11, diagnosis_type="discharge diagnosis: complications", ICD_10="J18.9"),
        dict(new_hosp_id=11, diagnosis_type="admission diagnosis", ICD_10="G93.1")])
    out = tmp_path_factory.mktemp("ricd_clif")
    RicdAdapter(src, out, tmp_path_factory.mktemp("duck")).run()
    return out


def read(clif: Path, table: str) -> pd.DataFrame:
    return pd.read_parquet(clif / f"clif_{table}.parquet")


def test_hospitalizations_follow_each_other_on_their_clock(clif: Path) -> None:
    h = read(clif, "hospitalization").set_index("hospitalization_id")
    assert h.loc["10", "admission_dttm"] == at(START_10, 0) and h.loc["10", "discharge_dttm"] == at(START_10, 6000)
    assert h.loc["11", "admission_dttm"] == at(START_11, 0) and h.loc["11", "discharge_category"] == "expired"
    p = read(clif, "patient").iloc[0]
    assert p["death_dttm"] == at(START_11, 50 * 60)
    assert p["birth_date"] == (at(START_10, 0) - timedelta(days=round(60 * 365.25 + 182.625))).date()


def test_icu_stays_sources_and_outcomes(clif: Path) -> None:
    s = read(clif, "ext_icu_stay").set_index("stay_id")
    assert s.loc["ricd:10_1", "admission_source"] == "other_hospital" and s.loc["ricd:10_1", "icu_outcome"] == "alive"
    assert s.loc["ricd:11_1", "admission_source"] == "ward" and s.loc["ricd:11_1", "icu_outcome"] == "dead"
    assert s.loc["ricd:11_1", "icu_in_dttm"] == at(START_11, 600) and s["clock_valid"].all()


def test_monitoring_vitals_and_output(clif: Path) -> None:
    v = read(clif, "vitals")
    hr = v[(v.vital_category == "heart_rate") & (v.hospitalization_id == "10")].sort_values("recorded_dttm")
    assert hr["recorded_dttm"].tolist() == [at(START_10, 60), at(START_10, 120)] and len(hr) == 2
    assert v[v.vital_category == "sbp"].iloc[0]["meas_site_name"] == "cuff"
    assert {"weight_kg", "height_cm"} <= set(v["vital_category"])
    assert read(clif, "output").iloc[0]["output_volume"] == 200.0


def test_hourly_sources_at_the_end_of_their_hour(clif: Path) -> None:
    lab = read(clif, "labs").set_index("lab_category")
    assert lab.loc["creatinine", "lab_result_dttm"] == at(START_10, 180)
    assert lab.loc["creatinine", "lab_value_numeric"] == pytest.approx(1.0, rel=1e-3)
    assert lab.loc["glucose", "lab_value_numeric"] == pytest.approx(90.08) and len(lab) == 2
    rs = read(clif, "respiratory_support")
    assert rs["device_category"].tolist() == ["imv"] and rs["fio2_set"].tolist() == [pytest.approx(0.4)]
    assert rs["recorded_dttm"].tolist() == [at(START_10, 8 * 60)]
    gcs = read(clif, "patient_assessments").sort_values("recorded_dttm")["numerical_value"].tolist()
    assert gcs == [10.0, 12.0, 14.0]


def test_orders_infusions_and_doses(clif: Path) -> None:
    c = read(clif, "medication_admin_continuous").sort_values("admin_dttm")
    assert c["mar_action_category"].tolist() == ["start", "going", "stop"]
    assert c["admin_dttm"].tolist() == [at(START_10, 360), at(START_10, 1800), at(START_10, 2520)]
    assert c["med_dose_unit"].isna().all()
    i = read(clif, "medication_admin_intermittent").set_index("med_category")
    assert (i.loc["furosemide", "med_route_category"], i.loc["furosemide", "med_dose"]) == ("iv", 20.0)
    assert (i.loc["heparin", "med_route_category"], i.loc["heparin", "med_dose"], i.loc["heparin", "med_dose_unit"]) \
        == ("im", 5000.0, "u")
    assert "phenylephrine" not in set(i.index)


def test_discharge_diagnoses_only(clif: Path) -> None:
    dx = read(clif, "hospital_diagnosis").set_index("diagnosis_code")["diagnosis_primary"].to_dict()
    assert dx == {"I635": 1, "J189": 0}
