"""SICdb adapter on a hand-built fixture (SICdb column layout, no real patients)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from icutwin.clif.adapters.sicdb import SicdbAdapter

START_100 = datetime(2015, 7, 1, tzinfo=UTC)            # first case: 1 July of the admission year
START_101 = START_100 + timedelta(days=30)                # second case: OffsetAfterFirstAdmission = 30 days


def at(case_start: datetime, seconds: int) -> datetime:
    return case_start + timedelta(seconds=seconds)


@pytest.fixture(scope="module")
def out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("sicdb")
    src = d / "src"
    src.mkdir()
    refs = [
        (735, "male", "Sex", None), (2, "INIC", "HospitalUnit", None), (3129, "Entlassung aus dem Krankenhaus", "DischargeType", None),
        (3130, "Sterbefall", "DischargeType", None), (2202, "lebend", "DischargeState", None),
        (2215, "verstorben", "DischargeState", None), (900, "Zentrale Notaufnahme", "ReferringUnit", None),
        (901, "No Surgery", "SurgicalAdmissionType", None), (902, "Unknown", "SurgicalSite", None),
        (903, "Urgent", "AdmissionUrgency", None), (904, "Yes", "YesNoUnknown", None),
        (905, "Unknown", "SurgicalAdmissionType", None), (906, "Herz: CABG", "SurgicalSite", None),
        (707, "HeartRateECG", "SignalFloat", "/min"), (2283, "FIO2", "RespiratorSetting", "%"),
        (723, "CRRT Bloodflow", "SignalFloat", "ml/min"), (730, "CRRT DialysateFlow", "SignalFloat", "ml/h"),
        (731, "CRRT SubstitutePrae", "SignalFloat", "ml/h"), (720, "EndotrachealTube", "Airway", None),
        (3045, "ArterialLine", "Line", None), (457, "Calcium (ZL)", "Laboratory", "mmol/L"),
        (1562, "Norepinephrin", "Drug", "g"), (1420, "Paracetamol", "Drug", "g"),
        (2081, "NaCl 0,9% Kurzinfusion", "Fluid", "ml"), (2079, "NaCl 0,9% Medikament", "Fluid", "ml"),
        (2046, "Erythrozytenkonzentrat", "Fluid", "ml"), (1680, "Propofol 2%", "Drug", "g"),
        (1681, "Sojabohnenöl Propofol", "Drug", "ml"),
    ]
    pd.DataFrame(refs, columns=["ReferenceGlobalID", "ReferenceValue", "ReferenceName", "ReferenceUnit"]).to_parquet(
        src / "d_references.parquet")
    base = dict(PatientID=7, AdmissionYear=2015, saps3=55, HospitalDischargeType=3129, HospitalStayDays=5,
                DischargeState=2202, DischargeUnit=None, EstimatedSurvivalObservationTime=None, Sex=735,
                WeightOnAdmission=80000, HeightOnAdmission=175, AgeOnAdmission=60, HospitalUnit=2, ReferringUnit=900,
                ICD10Main="I21.4", ICD10MainText="NSTEMI", DiagnosisT2=None, SurgicalSite=902, HoursOfCRRT=None,
                AdmissionUrgency=903, AdmissionFormHasSepsis=0, SurgicalAdmissionType=901, OrbisDataAvailable=1,
                HeartSurgeryAdditionalData=None, HeartSurgeryCPBTime=None, HeartSurgeryCrossClampTime=None,
                HeartSurgeryBeginOffset=None, HeartSurgeryEndOffset=None)
    pd.DataFrame([
        {**base, "CaseID": 100, "OffsetAfterFirstAdmission": 0, "ICUOffset": 3600, "TimeOfStay": 86400,
         "HospitalDischargeDay": 5, "OffsetOfDeath": None, "AdmissionFormHasSepsis": 904},     # sepsis on the form
        {**base, "CaseID": 101, "OffsetAfterFirstAdmission": 30 * 86400, "ICUOffset": 0, "TimeOfStay": 7200,
         "HospitalDischargeDay": 0, "OffsetOfDeath": 3600, "DischargeState": 2215, "HospitalDischargeType": 3130,
         "SurgicalAdmissionType": 905, "HeartSurgeryBeginOffset": 100},   # type and site unknown, cardiac surgery
    ]).to_parquet(src / "cases.parquet")
    sig = lambda i, data_id, off, val: {"id": i, "CaseID": 100, "DataID": data_id, "Offset": off, "Val": val,  # noqa: E731
                                        "cnt": 60, "rawdata": None}
    pd.DataFrame([sig(1, 707, 3600, 80.0), sig(2, 2283, 7200, 40.0), sig(3, 2283, 20000, 30.0),
                  sig(4, 723, 7200, 150.0), sig(5, 730, 7200, 1000.0), sig(6, 731, 7200, 500.0)]).to_parquet(
        src / "data_float_h.parquet")
    pd.DataFrame([{"id": 1, "CaseID": 100, "LaboratoryID": 457, "Offset": 5000, "LaboratoryValue": 2.0,
                   "LaboratoryType": None}]).to_parquet(src / "laboratory.parquet")
    pd.DataFrame([{"id": 11, "CaseID": 100, "DataID": 720, "Offset": 3600, "OffsetEnd": 10000, "Data": None},
                  {"id": 12, "CaseID": 100, "DataID": 3045, "Offset": 3600, "OffsetEnd": 50000, "Data": None}]).to_parquet(
        src / "data_range.parquet")
    pd.DataFrame(columns=["id", "CaseID", "FieldID", "RefID"]).to_parquet(src / "data_ref.parquet")
    med = lambda i, drug, off, end, single, amount, apm: {  # noqa: E731
        "id": i, "CaseID": 100, "PatientID": 7, "DrugID": drug, "Offset": off, "OffsetDrugEnd": end,
        "IsSingleDose": single, "Amount": amount, "AmountPerMinute": apm, "GivenState": 2193}
    pd.DataFrame([
        med(1, 1562, 3600, 7200, 0, 0.0006, 1e-5), med(2, 1562, 7200, 10800, 0, 0.0012, 2e-5),
        med(3, 1562, 9000, 12000, 0, 0.00025, 5e-6),                        # overlapping second syringe
        med(4, 1420, 4000, 4060, 1, 1.0, 1.0 / 60), med(5, 2081, 4000, 4060, 1, 100.0, 100.0 / 60),
        med(6, 2079, 4000, 4060, 1, 3.0, 3.0 / 60), med(7, 2046, 10000, 13600, 0, 250.0, 250.0 / 60),
        med(8, 1680, 20000, 23600, 0, 0.2, 0.2 / 60),        # propofol 0.2 g over 1 h
        med(9, 1681, 20000, 23600, 0, 10.0, 10.0 / 60),      # its lipid vehicle: part of the drug record
    ]).to_parquet(src / "medication.parquet")
    pd.DataFrame(columns=["id", "CaseID", "PatientID", "LogState", "Offset", "HospitalUnit"]).to_parquet(
        src / "unitlog.parquet")
    o = d / "out"
    SicdbAdapter(src, o, d / "duck").run()
    return o


def _read(out: Path, table: str) -> pd.DataFrame:
    return pd.read_parquet(out / f"clif_{table}.parquet")


def test_case_anchoring_outcomes_and_binning(out: Path) -> None:
    s = _read(out, "ext_icu_stay").set_index("stay_id")
    assert s.loc["sicdb:100", "icu_in_dttm"] == at(START_100, 3600) and s.loc["sicdb:100", "icu_out_dttm"] == at(START_100, 86400)
    assert s.loc["sicdb:101", "icu_in_dttm"] == START_101 and s.loc["sicdb:101", "stay_seq"] == 2
    assert s.loc["sicdb:101", "icu_outcome"] == "dead" and not s.loc["sicdb:100", "clock_valid"]
    assert s.loc["sicdb:100", "admission_source"] == "ed" and s.loc["sicdb:100", "surgical_status"] == "nonsurgical"
    assert s["adm_dx_group"].to_dict() == {"sicdb:100": "medical_sepsis", "sicdb:101": "surgical_cardiovascular"}
    p = _read(out, "patient").iloc[0]
    assert p["death_dttm"] == at(START_101, 3600) and p["sex_category"] == "male"
    h = _read(out, "hospitalization").set_index("hospitalization_id")
    assert h.loc["100", "discharge_dttm"] == START_100 + timedelta(days=5, hours=12)
    assert h.loc["101", "discharge_category"] == "expired" and h.loc["100", "age_at_admission"] == 62
    assert _read(out, "hospital_diagnosis")["diagnosis_code"].tolist() == ["I214", "I214"]


def test_hourly_signals_are_stamped_at_the_end_of_their_hour(out: Path) -> None:
    v = _read(out, "vitals")
    hr = v[v.vital_category == "heart_rate"].iloc[0]
    assert hr["recorded_dttm"] == at(START_100, 3600 + 3540) and hr["vital_value"] == 80.0
    w = v[v.vital_category == "weight_kg"].iloc[0]
    assert w["vital_value"] == 80.0 and w["recorded_dttm"] == at(START_100, 3600)
    assert _read(out, "labs").iloc[0]["lab_value_numeric"] == pytest.approx(8.016)


def test_ventilation_documented_by_airway_ranges(out: Path) -> None:
    r = _read(out, "respiratory_support").sort_values("recorded_dttm")
    assert r["device_category"].tolist() == ["imv", None]                    # second FiO2 after extubation
    assert r["fio2_set"].tolist() == [pytest.approx(0.4), pytest.approx(0.3)]
    a = _read(out, "airway").iloc[0]
    assert (a["airway_category"], a["placed_dttm"], a["removed_dttm"]) == ("ett", at(START_100, 3600), at(START_100, 10000))
    assert _read(out, "line").iloc[0]["line_category"] == "art_line"
    assert _read(out, "crrt_therapy").iloc[0]["crrt_mode_category"] == "cvvhdf"


def test_medication_periods_become_rate_steps(out: Path) -> None:
    m = _read(out, "medication_admin_continuous").query("med_category == 'norepinephrine'").sort_values("admin_dttm")
    got = [(int((t - START_100).total_seconds()), a, round(dose, 6)) for t, a, dose in
           zip(m["admin_dttm"], m["mar_action_category"], m["med_dose"])]
    assert got == [(3600, "start", 10.0), (7200, "dose_change", 20.0), (9000, "dose_change", 25.0),
                   (10800, "dose_change", 5.0), (12000, "stop", 0.0)]
    assert set(m["med_dose_unit"]) == {"mcg/min"}


def test_single_doses_vehicles_and_fluids(out: Path) -> None:
    i = _read(out, "medication_admin_intermittent").sort_values("med_category")
    assert i[["med_category", "med_dose", "med_dose_unit"]].values.tolist() == [
        ["acetaminophen", 1000.0, "mg"], ["sodium_chloride", 100.0, "ml"]]      # diluent 'NaCl Medikament' dropped
    inp = _read(out, "input")
    assert inp[["input_category", "input_volume"]].values.tolist() == [["blood_products", 250.0]]
    assert inp["recorded_dttm"].iloc[0] == at(START_100, 13600)                 # volume counted at period end


def test_propofol_is_a_drug_and_its_lipid_a_vehicle(out: Path) -> None:
    m = _read(out, "medication_admin_continuous").query("med_category == 'propofol'").sort_values("admin_dttm")
    assert m["mar_action_category"].tolist() == ["start", "stop"] and m["med_name"].iloc[0] == "Propofol 2%"
    assert "Sojabohnenöl Propofol" not in set(_read(out, "medication_admin_continuous")["med_name"])
