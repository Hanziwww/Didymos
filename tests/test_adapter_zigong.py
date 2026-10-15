"""zigongICU adapter on a hand-built mini dataset with known expected CLIF output (amendment 16a)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from icutwin.clif.adapters.zigong import ZigongAdapter

ORIGIN = datetime(2019, 1, 1, tzinfo=UTC)
NURSING = ("temperature", "heart_rate", "breathing", "Blood_oxygen_saturation", "Blood_pressure_high",
           "Blood_pressure_low", "Invasive_SBP", "Invasive_diastolic_blood_pressure", "blood_sugar",
           "Oxygen_inhalation_mode", "Breathing_pattern", "Endotracheal_intubation", "Vt_setting", "Vt_Supervisor",
           "Oxygen_flow", "open_one's_eyes", "motion", "language", "RASS_sedation_score", "Urine_volume")


def at(hours: float) -> datetime:
    return ORIGIN + timedelta(hours=hours)


def _write(path: Path, rows: list[dict]) -> None:
    if "ChartTime" in rows[0]:              # the release's nursing chart is text throughout
        schema = pa.schema([("column00", pa.int64()), ("INP_NO", pa.int64()), ("ChartTime", pa.string()),
                            *((c, pa.string()) for c in NURSING)])
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
        return
    pq.write_table(pa.Table.from_pylist(rows), path)


def _chart(k: int, inp: int, day: str, **values) -> dict:
    return {"column00": k, "INP_NO": inp, "ChartTime": day, **{c: values.get(c) for c in NURSING}}


@pytest.fixture(scope="module")
def clif(tmp_path_factory: pytest.TempPathFactory) -> Path:
    src = tmp_path_factory.mktemp("zigong_pq")
    _write(src / "dtBaseline.parquet", [
        dict(PATIENT_ID=1, INP_NO=11, Age="60", SEX="Male", AdmitDept="EICU headquarters", DISCHARGE_DATE_TIME=200.0),
        dict(PATIENT_ID=2, INP_NO=12, Age="50", SEX="Female", AdmitDept="Department of Infectious Diseases",
             DISCHARGE_DATE_TIME=300.0)])
    _write(src / "dtTransfer.parquet", [
        dict(INP_NO=11, StartTime=0.0, StopTime=50.0, TransferDept="EICU headquarters"),
        dict(INP_NO=11, StartTime=50.5, StopTime=100.0, TransferDept="Intensive care unit (ICU) ward a"),  # merged
        dict(INP_NO=11, StartTime=100.0, StopTime=200.0, TransferDept="Department of respiratory medicine"),
        dict(INP_NO=12, StartTime=0.0, StopTime=40.0, TransferDept="Department of Infectious Diseases"),
        dict(INP_NO=12, StartTime=40.0, StopTime=300.0, TransferDept="Intensive care unit (ICU) ward B")])
    _write(src / "dtOutCome.parquet", [
        dict(PATIENT_ID=1, Follow_Vital="Death", Death_Date="2000"),   # long after discharge: not in hospital
        dict(PATIENT_ID=2, Follow_Vital="Death", Death_Date="290.5")])  # the date of an in-hospital death
    _write(src / "dtICD.parquet", [
        dict(INP_NO=11, ICD_Code="J18.900", Status_Discharge="Recovered", Diagnosis_seq=1),
        dict(INP_NO=11, ICD_Code="M80700/3", Status_Discharge="Recovered", Diagnosis_seq=2),   # morphology
        dict(INP_NO=12, ICD_Code="A41.900", Status_Discharge="Dead", Diagnosis_seq=1)])
    _write(src / "dtNursingChart.parquet", [
        # the admission day (its date is 4.75 h before admission): two rows over the 19.25 charted hours
        _chart(1, 11, "-4.75", heart_rate="90", Blood_pressure_high="120", Blood_pressure_low="60",
               Oxygen_inhalation_mode="气插导管", Breathing_pattern="SIMV", **{"open_one's_eyes": "3→呼唤睁眼"},
               RASS_sedation_score="-2", Urine_volume="100"),
        _chart(2, 11, "-4.75", heart_rate="100", Oxygen_inhalation_mode="鼻塞", Oxygen_flow="3"),
        _chart(3, 11, "19.25", heart_rate="95", Oxygen_inhalation_mode="辅助", Breathing_pattern="BIPAP"),
        _chart(4, 11, "-21977", heart_rate="80"),            # an implausible date: dropped
        _chart(5, 11, "NA", heart_rate="81")])               # no date: dropped
    _write(src / "dtDrugs.parquet", [
        dict(PATIENT_ID=1, DrugName="Norepinephrine bitartrate injection", Drug_time=5.0),
        dict(PATIENT_ID=1, DrugName="Norepinephrine bitartrate injection", Drug_time=20.0),   # within 25 h: going
        dict(PATIENT_ID=1, DrugName="Norepinephrine bitartrate injection", Drug_time=60.0),   # a new episode
        dict(PATIENT_ID=1, DrugName="Deoxyepinephrine hydrochloride injection", Drug_time=10.0),
        dict(PATIENT_ID=1, DrugName="Furosemide Injection", Drug_time=30.0),
        dict(PATIENT_ID=1, DrugName="Enoxaparin sodium injection", Drug_time=31.0),
        dict(PATIENT_ID=1, DrugName="Hemofiltration replacement base fluid", Drug_time=40.0)])
    _write(src / "dtLab.parquet", [
        dict(INP_NO=11, LabTime=12.0, Category="Renal function", Item="Creatinine (enzymatic method) (CRE)",
             LabValue="88.4", Unit="umol/L"),
        dict(INP_NO=11, LabTime=12.0, Category="Blood analysis", Item="Neutrophil ratio (neu%)", LabValue="0.8",
             Unit=None)])
    out = tmp_path_factory.mktemp("zigong_clif")
    ZigongAdapter(src, out, tmp_path_factory.mktemp("duck")).run()
    return out


def read(clif: Path, table: str) -> pd.DataFrame:
    return pd.read_parquet(clif / f"clif_{table}.parquet")


def test_icu_stays_merge_transfers_and_place_deaths(clif: Path) -> None:
    s = read(clif, "ext_icu_stay").set_index("stay_id")
    assert list(s.index) == ["zigong:11_1", "zigong:12_1"]
    assert s.loc["zigong:11_1", "icu_in_dttm"] == at(0) and s.loc["zigong:11_1", "icu_out_dttm"] == at(100)
    assert s.loc["zigong:11_1", "icu_outcome"] == "alive" and s.loc["zigong:11_1", "admission_source"] == "ed"
    assert s.loc["zigong:12_1", "icu_outcome"] == "dead" and s.loc["zigong:12_1", "admission_source"] == "ward"
    assert not s["clock_valid"].any() and set(s["dataset"]) == {"zigong"}
    p = read(clif, "patient").set_index("patient_id")
    assert p.loc["2", "death_dttm"] == at(300) and p.loc["1", "death_dttm"] == at(2000)


def test_nursing_chart_rows_spread_over_their_day(clif: Path) -> None:
    v = read(clif, "vitals")
    hr = v[v.vital_category == "heart_rate"].sort_values("recorded_dttm")
    assert hr["recorded_dttm"].tolist() == [at(19.25 / 2), at(19.25), at(43.25)]
    assert hr["vital_value"].tolist() == [90.0, 100.0, 95.0]
    m = v[v.vital_category == "map"].iloc[0]
    assert m["vital_value"] == pytest.approx(80.0) and m["meas_site_name"] == "cuff"


def test_respiratory_support_from_airway_and_mode(clif: Path) -> None:
    rs = read(clif, "respiratory_support").sort_values("recorded_dttm")
    assert rs["device_category"].tolist() == ["imv", "nasal_cannula", "nippv"]
    assert rs["mode_category"].tolist()[0] == "simv" and rs["lpm_set"].tolist()[1] == 3.0


def test_assessments_and_output(clif: Path) -> None:
    pa_ = read(clif, "patient_assessments").set_index("assessment_category")["numerical_value"]
    assert pa_["gcs_eye"] == 3.0 and pa_["rass"] == -2.0
    o = read(clif, "output").iloc[0]
    assert o["output_group"] == "urine" and o["output_volume"] == 100.0


def test_infusion_episodes_and_doses(clif: Path) -> None:
    c = read(clif, "medication_admin_continuous").sort_values("admin_dttm")
    ne = c[c.med_category == "norepinephrine"]
    assert ne["mar_action_category"].tolist() == ["start", "going", "stop", "start", "stop"]
    assert ne["admin_dttm"].tolist() == [at(5), at(20), at(26), at(60), at(66)]
    assert "phenylephrine" in set(c["med_category"]) and c["med_dose_unit"].isna().all()
    i = read(clif, "medication_admin_intermittent").set_index("med_category")["med_route_category"]
    assert i["furosemide"] == "iv" and i["ext_enoxaparin"] == "im"
    assert read(clif, "crrt_therapy")["recorded_dttm"].tolist() == [at(40)]


def test_labs_and_diagnoses(clif: Path) -> None:
    lab = read(clif, "labs").set_index("lab_category")["lab_value_numeric"]
    assert lab["creatinine"] == pytest.approx(1.0, rel=1e-3) and lab["neutrophils_percent"] == pytest.approx(80.0)
    dx = read(clif, "hospital_diagnosis").set_index("hospitalization_id")
    assert dx.loc["11", "diagnosis_code"] == "J189" and dx.loc["12", "diagnosis_code"] == "A419"
    assert len(dx.loc[["11"]]) == 1                       # the morphology code is not a diagnosis
