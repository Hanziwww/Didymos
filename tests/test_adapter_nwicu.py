"""NWICU adapter on a hand-built fixture (MIMIC-IV-style layout, no real patients)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from icutwin.clif.adapters.nwicu import NwicuAdapter

T = lambda h, m=0: datetime(2150, 3, 1, h, m, tzinfo=UTC)  # noqa: E731


def _ts(h: int, m: int = 0) -> pd.Timestamp:
    return pd.Timestamp(2150, 3, 1, h, m)


@pytest.fixture(scope="module")
def out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("nwicu")
    hosp, icu = d / "src/data/nw_hosp", d / "src/data/nw_icu"
    hosp.mkdir(parents=True)
    icu.mkdir(parents=True)
    pd.DataFrame([{"subject_id": 1, "gender": "F", "anchor_age": 70, "anchor_year": 2148, "anchor_year_group": "x",
                   "dod": pd.NaT}]).to_parquet(hosp / "patients.parquet")
    pd.DataFrame([{"subject_id": 1, "hadm_id": 10, "admittime": _ts(1), "dischtime": _ts(23), "deathtime": pd.NaT,
                   "admission_type": "EMERGENCY", "admit_provider_id": None, "admission_location": "EMERGENCY MEDICINE",
                   "discharge_location": "HOME WITH HOME HEALTH CARE", "insurance": None, "language": "ENGLISH",
                   "marital_status": None, "race": "WHITE", "edregtime": None, "edouttime": None,
                   "hospital_expire_flag": 0}]).to_parquet(hosp / "admissions.parquet")
    pd.DataFrame([{"subject_id": 1, "hadm_id": 10, "stay_id": 100, "first_careunit": "MICU", "last_careunit": "MICU",
                   "intime": _ts(2), "outtime": _ts(22), "los": 0.8},
                  {"subject_id": 1, "hadm_id": 10, "stay_id": 101, "first_careunit": "IMCU", "last_careunit": "IMCU",
                   "intime": _ts(22), "outtime": _ts(23), "los": 0.04}]).to_parquet(icu / "icustays.parquet")
    pd.DataFrame([(320045, "PULSE"), (323761, "TEMPERATURE"), (320050, "R ARTERIAL LINE BLOOD PRESSURE"),
                  (320050, "R ARTERIAL LINE BLOOD PRESSURE 2"), (787541, "MECHANICAL VENTILATION"),
                  (1, "INTUBATION"), (2, "EXTUBATION / DECANNULATION")],
                 columns=["itemid", "label"]).to_parquet(icu / "d_items.parquet")
    ce = lambda item, h, v: {"subject_id": 1, "hadm_id": 10, "stay_id": 100, "caregiver_id": None, "charttime": _ts(h),  # noqa: E731
                             "storetime": _ts(h), "itemid": item, "value": str(v), "valuenum": v, "valueuom": None, "warning": 0}
    pd.DataFrame([ce(320045, 3, 90.0), ce(323761, 3, 98.6), ce(320050, 3, 120.0)]).to_parquet(icu / "chartevents.parquet")
    pe = lambda item, h: {"subject_id": 1, "hadm_id": 10, "stay_id": 100, "starttime": _ts(h), "endtime": None, "itemid": item}  # noqa: E731
    pd.DataFrame([pe(1, 4), pe(787541, 4), pe(2, 10)]).to_parquet(icu / "procedureevents.parquet")
    pd.DataFrame([(100001, "Glucose"), (100075, "D-Dimer")], columns=["itemid", "label"]).assign(
        fluid="Blood", category=None).to_parquet(hosp / "d_labitems.parquet")
    lab = lambda item, h, v: {"labevent_id": h, "subject_id": 1, "hadm_id": 10, "specimen_id": 1, "itemid": item,  # noqa: E731
                              "charttime": _ts(h), "storetime": _ts(h, 30), "value": str(v), "valuenum": v,
                              "valueuom": None}
    pd.DataFrame([lab(100001, 5, 140.0), lab(100075, 5, 2000.0), lab(100001, 6, 9999999.0)]).to_parquet(
        hosp / "labevents.parquet")
    rx = [{"pharmacy_id": 1, "dose_val_rx": "0.05", "dose_unit_rx": "mcg/kg/min", "route": "Intravenous"},
          {"pharmacy_id": 2, "dose_val_rx": "650", "dose_unit_rx": "mg", "route": "Oral"},
          {"pharmacy_id": 3, "dose_val_rx": "5000", "dose_unit_rx": "UNIT", "route": "Injection"},
          {"pharmacy_id": 4, "dose_val_rx": "40", "dose_unit_rx": "mg", "route": "Injection"},
          {"pharmacy_id": 5, "dose_val_rx": "1", "dose_unit_rx": "g", "route": "Topical"},
          {"pharmacy_id": 6, "dose_val_rx": "2", "dose_unit_rx": "puff", "route": "Inhalation"},
          {"pharmacy_id": 7, "dose_val_rx": None, "dose_unit_rx": None, "route": "Injection"}]
    pd.DataFrame(rx).to_parquet(hosp / "prescriptions.parquet")
    em = lambda pid, med, h, m, ev: {"subject_id": 1, "hadm_id": 10, "emar_id": f"{h}{m}", "pharmacy_id": pid,  # noqa: E731
                                     "charttime": _ts(h, m), "medication": med, "event_txt": ev}
    ne = "NOREPINEPHRINE BITARTRATE-D5W 4 MG/250 ML (16 MCG/ML) IV SOLN"
    pd.DataFrame([em(1, ne, 4, 0, "Applied"), em(1, ne, 5, 0, "Confirmed"), em(1, ne, 14, 0, "Confirmed"),
                  em(2, "ACETAMINOPHEN  325 MG ORAL TAB", 6, 0, "Applied"),
                  em(2, "ACETAMINOPHEN  325 MG ORAL TAB", 7, 0, "Not Given"),
                  em(2, "ACETAMINOPHEN  325 MG ORAL TAB", 12, 0, "Confirmed"),  # not an administration
                  em(3, "HEPARIN (PORCINE)  5000 UNITS/ML INJ SOLN", 8, 0, "Applied"),
                  em(4, "FUROSEMIDE 10 MG/ML INJ SOLN", 9, 0, "Applied"),
                  em(5, "GENTAMICIN 0.1 % TOP CREA", 10, 0, "Applied"),
                  em(6, "ALBUTEROL SULFATE 90 MCG/ACTUATION INH HFAA", 11, 0, "Applied"),
                  em(7, "FENTANYL CITRATE (PF) 50 MCG/ML INJ SOLN", 11, 30, "Applied")]).to_parquet(hosp / "emar.parquet")
    pd.DataFrame([{"subject_id": 1, "hadm_id": 10, "seq_num": 1, "icd_code": "J96.00", "icd_version": 10}]).to_parquet(
        hosp / "diagnoses_icd.parquet")
    o = d / "out"
    NwicuAdapter(d / "src", o, d / "duck").run()
    return o


def _read(out: Path, table: str) -> pd.DataFrame:
    return pd.read_parquet(out / f"clif_{table}.parquet")


def test_identity_stays_and_units(out: Path) -> None:
    h = _read(out, "hospitalization").iloc[0]
    assert (h["age_at_admission"], h["admission_type_category"], h["discharge_category"]) == (72, "ed", "home")
    s = _read(out, "ext_icu_stay")
    assert s["stay_id"].tolist() == ["nwicu:100"] and s["unit_type"].tolist() == ["medical_icu"]   # IMCU is not an ICU stay
    adt = _read(out, "adt").set_index("location_name")
    assert adt.loc["IMCU", "location_category"] == "stepdown"
    v = _read(out, "vitals").set_index("vital_category")
    assert v.loc["temp_c", "vital_value"] == pytest.approx(37.0)
    assert v.loc["sbp", "meas_site_name"] == "arterial" and len(v.loc[["sbp"]]) == 1   # duplicated d_items label
    lab = _read(out, "labs").set_index("lab_category")
    assert len(lab) == 2 and lab.loc["d_dimer", "lab_value_numeric"] == pytest.approx(2.0)   # sentinel 9999999 dropped
    assert lab.loc["glucose_serum", "lab_result_dttm"] == T(5, 30)


def test_orders_document_ventilation_and_airway(out: Path) -> None:
    r = _read(out, "respiratory_support")
    assert set(r["device_category"]) == {"imv"} and set(r["recorded_dttm"]) == {T(4)}
    a = _read(out, "airway").iloc[0]
    assert (a["placed_dttm"], a["removed_dttm"], a["airway_category"]) == (T(4), T(10), "ett")


def test_medications(out: Path) -> None:
    c = _read(out, "medication_admin_continuous").sort_values("admin_dttm")
    got = [(t.hour, t.minute, a) for t, a in zip(c["admin_dttm"], c["mar_action_category"])]
    # 05:00 -> 14:00 gap > 6 h: stop 60 min after 05:00, restart at 14:00, stop 60 min after the last confirmation
    assert got == [(4, 0, "start"), (5, 0, "going"), (6, 0, "stop"), (14, 0, "start"), (15, 0, "stop")]
    assert set(c["med_dose_unit"]) == {"mcg/kg/min"} and c["med_dose"].dropna().tolist() == [0.0, 0.0]
    i = _read(out, "medication_admin_intermittent").sort_values("admin_dttm")
    cols = i[["med_category", "med_dose", "med_dose_unit", "med_route_category"]].astype(object)
    assert cols.where(cols.notna(), None).values.tolist() == [
        ["acetaminophen", 650.0, "mg", "enteral"],          # 'Not Given' and 'Confirmed' are not doses
        ["heparin", 5000.0, "unit", "im"],          # 'Injection' of prophylactic heparin: subcutaneous (CLIF 'im')
        ["furosemide", 40.0, "mg", "iv"],           # topical gentamicin: not a CLIF administration
        ["albuterol", 2.0, "puff", "inhaled"]]      # a count in a counting unit, as prescribed (M4); fentanyl
    # ordered without a dose unit states no administration type (bolus vial or drip): no record
