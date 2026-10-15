"""INSPIRE adapter on a hand-built mini dataset with known expected CLIF output."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from icutwin.clif.adapters.inspire import InspireAdapter

ORIGIN = datetime(2016, 1, 1, tzinfo=UTC)


def at(minutes: float) -> datetime:
    return ORIGIN + timedelta(minutes=minutes)


def _write(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture(scope="module")
def clif(tmp_path_factory: pytest.TempPathFactory) -> Path:
    src = tmp_path_factory.mktemp("inspire_pq")
    op = dict(op_id=401, subject_id=101, hadm_id=201, case_id=None, opdate=0, age=70, sex="M", weight=70.0,
              height=170.0, race="Asian", asa=3, emop=0, department="CTS", antype="General", icd10_pcs="021009W",
              orin_time=600, orout_time=900, opstart_time=630, opend_time=870, admission_time=0,
              discharge_time=10000, anstart_time=610, anend_time=880, cpbon_time=700, cpboff_time=800,
              icuin_time=920, icuout_time=3000, inhosp_death_time=None, allcause_death_time=None)
    op2 = dict(op, op_id=402, subject_id=102, hadm_id=202, age=55, sex="F", emop=1, cpbon_time=None, cpboff_time=None,
               icd10_pcs="0DTJ0",                                                   # resection in the GI system
               orin_time=100, orout_time=200, opstart_time=110, opend_time=190, icuin_time=210, icuout_time=500,
               discharge_time=600, inhosp_death_time=450)
    # patient 103: the row of the first admission (0-1000) carries the ICU stay of the second (5000-7000)
    op3 = dict(op, op_id=403, subject_id=103, hadm_id=203, cpbon_time=None, cpboff_time=None, orin_time=100,
               orout_time=200, opstart_time=110, opend_time=190, discharge_time=1000, icuin_time=5200, icuout_time=6000)
    op4 = dict(op3, op_id=404, hadm_id=204, admission_time=5000, discharge_time=7000, orin_time=5100, orout_time=5190,
               opstart_time=5110, opend_time=5180, icuin_time=None, icuout_time=None)
    # patient 105: the OR exit is recorded 20 min after the ICU admission; the operation is still the index one
    op5 = dict(op, op_id=405, subject_id=105, hadm_id=205, cpbon_time=None, cpboff_time=None, orin_time=100,
               orout_time=230, opstart_time=110, opend_time=220, discharge_time=1000, icuin_time=210, icuout_time=600)
    _write(src / "operations.parquet", [op, op2, op3, op4, op5])
    _write(src / "ward_vitals.parquet", [
        dict(subject_id=101, chart_time=1000, item_name="hr", value=88.0),
        dict(subject_id=101, chart_time=1000, item_name="nibp_sbp", value=120.0),
        dict(subject_id=101, chart_time=1000, item_name="art_sbp", value=125.0),
        dict(subject_id=101, chart_time=1000, item_name="vent", value=1.0),
        dict(subject_id=101, chart_time=1000, item_name="fio2", value=40.0),
        dict(subject_id=101, chart_time=1060, item_name="vent", value=0.0),
        dict(subject_id=101, chart_time=1060, item_name="fio2", value=21.0),
        dict(subject_id=101, chart_time=1060, item_name="gcs_m", value=6.0),
        dict(subject_id=101, chart_time=1100, item_name="uo", value=50.0),
        dict(subject_id=101, chart_time=1100, item_name="uo", value=50.0),   # INSPIRE repeats rows: one record
        dict(subject_id=101, chart_time=1200, item_name="crrt", value=1.0),
        dict(subject_id=101, chart_time=1200, item_name="crrt", value=1.0),
        dict(subject_id=101, chart_time=99999, item_name="hr", value=70.0),  # outside every admission: dropped
    ])
    _write(src / "vitals.parquet", [
        dict(op_id=401, subject_id=101, chart_time=640, item_name="nepi", value=0.05),
        dict(op_id=401, subject_id=101, chart_time=645, item_name="nepi", value=0.05),  # unchanged rate: dropped
        dict(op_id=401, subject_id=101, chart_time=650, item_name="nepi", value=0.10),
        dict(op_id=401, subject_id=101, chart_time=660, item_name="nepi", value=0.0),
        dict(op_id=401, subject_id=101, chart_time=650, item_name="ftn", value=100.0),
        dict(op_id=401, subject_id=101, chart_time=650, item_name="peep", value=5.0),
        dict(op_id=401, subject_id=101, chart_time=650, item_name="fio2", value=50.0),
        dict(op_id=401, subject_id=101, chart_time=700, item_name="ns", value=500.0),
        dict(op_id=401, subject_id=101, chart_time=700, item_name="ns", value=500.0),   # repeated row
    ])
    _write(src / "labs.parquet", [
        dict(subject_id=101, chart_time=1000, item_name="crp", value=2.5),
        dict(subject_id=101, chart_time=1000, item_name="ica", value=1.1),
        dict(subject_id=101, chart_time=1000, item_name="hba1c", value=6.0),  # unmapped: dropped
    ])
    _write(src / "diagnosis.parquet", [dict(subject_id=101, chart_time=50, icd10_cm="I25.1")])
    med = lambda t, route, name, atc=None: dict(subject_id=101, chart_time=t, route=route, drug_name=name,  # noqa: E731
                                                atc_code=atc)
    _write(src / "medications.parquet", [
        med(1000, "iv", "piperacillin", "J01CA12"), med(1000, "iv", "tazobactam", "J01CG02"),  # one product
        med(1000, "po", "acetaminophen", "N02BE01"),
        med(1000, "po", "acetaminophen", "N02BE01"),        # exact duplicate record: one administration
        med(1000, "ex", "moxifloxacin", "S01AE07"),         # eye drops: no CLIF route category
        med(1100, "iv", "norepinephrine", "C01CA03"),       # infusion-only category: presence event
        med(650, "iv", "fentanyl", "N02AB03"),              # inside the operation: the vitals stream has it
        med(650, "iv", "cefazolin", "J01DB04"),             # inside the operation, not in the vitals stream
        med(1000, "po", "ambroxol", "R05CB06"),             # no CLIF category: audited, not emitted
        med(1200, "po", "levothyroxine sodium", "H03AA01"),  # local intermittent extension: an oral dose
        med(1200, "iv", "esomeprazole", "A02BC05"),          # local intermittent extension of M3: a dose
        med(1300, "iv", "nicardipine", "C08CA04"),           # local intermittent extension, IV: still an infusion
        med(1300, "po", "metoprolol", "C07AB02"),            # ext_ concept: a dose
    ])
    out = tmp_path_factory.mktemp("inspire_clif")
    InspireAdapter(src, out, tmp_path_factory.mktemp("duck")).run()
    return out


def read(clif: Path, table: str) -> pd.DataFrame:
    return pd.read_parquet(clif / f"clif_{table}.parquet")


def test_patient_follows_inspire_meds_conventions(clif: Path) -> None:
    p = read(clif, "patient").set_index("patient_id")
    assert p.loc["101", "birth_date"] == (ORIGIN - timedelta(days=round(70 * 365.25 + 182.625))).date()
    assert p.loc["101", "sex_category"] == "male" and p.loc["101", "race_category"] == "asian"
    assert p.loc["102", "death_dttm"] == at(450)


def test_adt_segments_are_contiguous_and_prioritized(clif: Path) -> None:
    adt = read(clif, "adt")
    a = adt[adt.hospitalization_id == "201"].sort_values("in_dttm")
    assert a["location_category"].tolist() == ["ward", "procedural", "ward", "icu", "ward"]
    assert (a["in_dttm"].iloc[1:].values == a["out_dttm"].iloc[:-1].values).all()  # no gaps or overlaps
    assert a["in_dttm"].iloc[0] == at(0) and a["out_dttm"].iloc[-1] == at(10000)


def test_vitals_sites_units_and_windowing(clif: Path) -> None:
    v = read(clif, "vitals")
    sbp = v[v.vital_category == "sbp"].set_index("meas_site_name")["vital_value"].to_dict()
    assert sbp == {"cuff": 120.0, "arterial": 125.0}
    assert at(99999) not in set(v["recorded_dttm"])
    assert {"height_cm", "weight_kg"} <= set(v["vital_category"])


def test_labs_converted_to_clif_units(clif: Path) -> None:
    lab = read(clif, "labs").set_index("lab_category")
    assert lab.loc["crp", "lab_value_numeric"] == pytest.approx(25.0) and lab.loc["crp", "reference_unit"] == "mg/l"
    assert lab.loc["calcium_ionized", "lab_value_numeric"] == pytest.approx(1.1 * 4.008)
    assert "hba1c" not in set(read(clif, "labs")["lab_name"])


def test_respiratory_support_semantics(clif: Path) -> None:
    rs = read(clif, "respiratory_support").sort_values("recorded_dttm")
    ward = rs[rs.device_name != "anaesthesia ventilator"]
    assert ward["device_category"].tolist() == ["imv", "room_air"]
    assert ward["fio2_set"].tolist() == [pytest.approx(0.40), pytest.approx(0.21)]  # CLIF stores a fraction
    anaes = rs[rs.device_name == "anaesthesia ventilator"].iloc[0]
    assert anaes["device_category"] == "imv" and anaes["peep_obs"] == 5.0


def test_continuous_meds_one_row_per_rate_change(clif: Path) -> None:
    mac = read(clif, "medication_admin_continuous").sort_values("admin_dttm")
    mac = mac[mac.mar_action_category != "going"]                     # ward presence events: see below
    assert mac["med_dose"].tolist() == [0.05, 0.10, 0.0]
    assert mac["mar_action_category"].tolist() == ["start", "dose_change", "stop"]
    assert set(mac["med_group"]) == {"vasoactives"}
    mai = read(clif, "medication_admin_intermittent")
    assert mai.loc[mai.mar_action_name == "intra-operative bolus", "med_category"].tolist() == ["fentanyl"]


def test_ward_medication_records(clif: Path) -> None:
    mai = read(clif, "medication_admin_intermittent")
    ward = mai[mai.mar_action_name == "ward medication record (dose not recorded)"]
    got = sorted(zip(ward["med_category"], ward["med_route_category"]))
    assert got == [("acetaminophen", "enteral"), ("cefazolin", "iv"), ("esomeprazole", "iv"),
                   ("ext_metoprolol", "enteral"), ("levothyroxine", "enteral"), ("piperacillin_tazobactam", "iv")]
    assert ward["med_dose"].isna().all() and set(ward["mar_action_category"]) == {"given"}
    mac = read(clif, "medication_admin_continuous")
    pres = mac[mac.mar_action_category == "going"].sort_values("admin_dttm")
    assert pres[["med_category", "admin_dttm"]].values.tolist() == [["norepinephrine", at(1100)],
                                                                    ["nicardipine", at(1300)]]


def test_icu_stay_context_and_outcome(clif: Path) -> None:
    s = read(clif, "ext_icu_stay").set_index("patient_id")
    assert s.loc["101", "admission_source"] == "or" and s.loc["101", "surgical_status"] == "elective_surgery"
    assert s.loc["101", "icu_outcome"] == "alive" and s.loc["102", "icu_outcome"] == "dead"
    assert s.loc["102", "surgical_status"] == "emergency_surgery"
    assert s.loc[["101", "102"], "adm_dx_group"].tolist() == ["surgical_cardiovascular", "surgical_gastrointestinal"]
    assert s.loc["105", ["admission_source", "surgical_status", "adm_dx_group"]].tolist() == [
        "or", "elective_surgery", "surgical_cardiovascular"]
    assert '"cpb":true' in s.loc["101", "surgery_context"].replace(" ", "")
    assert not s["clock_valid"].any()


def test_io_and_devices(clif: Path) -> None:
    assert read(clif, "input")["input_category"].tolist() == ["iv_crystalloid"]
    assert read(clif, "output")["output_group"].tolist() == ["urine"]
    assert len(read(clif, "crrt_therapy")) == 1
    assert read(clif, "hospital_diagnosis")["diagnosis_code"].tolist() == ["I251"]


def test_overlapping_operations_count_a_shared_record_once(tmp_path: Path) -> None:
    """Two overlapping operations of one admission carrying the same intra-operative records (time, item, value):
    the fluid volume and the bolus are counted once."""
    src = tmp_path / "src"
    src.mkdir()
    base = dict(subject_id=111, hadm_id=311, case_id=None, opdate=0, age=60, sex="F", weight=60.0, height=160.0,
                race="Asian", asa=2, emop=0, department="GS", antype="General", icd10_pcs="0DTJ0", admission_time=0,
                discharge_time=5000, orin_time=100, orout_time=300, opstart_time=110, opend_time=290,
                anstart_time=105, anend_time=295, cpbon_time=None, cpboff_time=None, icuin_time=None,
                icuout_time=None, inhosp_death_time=None, allcause_death_time=None)
    _write(src / "operations.parquet", [dict(base, op_id=501), dict(base, op_id=502, icd10_pcs="0DB60")])
    shared = [dict(subject_id=111, chart_time=200, item_name="ns", value=250.0),
              dict(subject_id=111, chart_time=200, item_name="ftn", value=50.0)]
    _write(src / "vitals.parquet", [dict(r, op_id=op) for op in (501, 502) for r in shared])
    _write(src / "ward_vitals.parquet", [dict(subject_id=111, chart_time=1000, item_name="hr", value=80.0)])
    _write(src / "labs.parquet", [dict(subject_id=111, chart_time=1000, item_name="crp", value=1.0)])
    _write(src / "diagnosis.parquet", [dict(subject_id=111, chart_time=50, icd10_cm="K80.2")])
    _write(src / "medications.parquet", [dict(subject_id=111, chart_time=1000, route="po", drug_name="acetaminophen",
                                              atc_code="N02BE01")])
    out = tmp_path / "out"
    InspireAdapter(src, out, tmp_path / "duck").run()
    assert read(out, "input")["input_volume"].tolist() == [250.0]
    bolus = read(out, "medication_admin_intermittent").query("mar_action_name == 'intra-operative bolus'")
    assert bolus["med_dose"].tolist() == [50.0]


def test_infusion_records_copied_to_two_operations_make_one_stream(tmp_path: Path) -> None:
    """INSPIRE copies an infusion's records to every operation of the admission and keeps some, such as the zeros
    at an operation's end, under one only (#95): the admission's stream has every change and stop."""
    src = tmp_path / "src"
    src.mkdir()
    base = dict(subject_id=112, hadm_id=312, case_id=None, opdate=0, age=60, sex="F", weight=60.0, height=160.0,
                race="Asian", asa=2, emop=0, department="CS", antype="General", icd10_pcs="02703ZZ", admission_time=0,
                discharge_time=5000, orin_time=100, orout_time=300, opstart_time=110, opend_time=290,
                anstart_time=105, anend_time=295, cpbon_time=None, cpboff_time=None, icuin_time=None,
                icuout_time=None, inhosp_death_time=None, allcause_death_time=None)
    _write(src / "operations.parquet", [dict(base, op_id=601),
                                        dict(base, op_id=602, orin_time=600, orout_time=710, opstart_time=610,
                                             opend_time=700, anstart_time=605, anend_time=705)])
    nepi = lambda op, t, v: dict(op_id=op, subject_id=112, chart_time=t, item_name="nepi", value=v)  # noqa: E731
    _write(src / "vitals.parquet", [nepi(op, t, v) for op in (601, 602) for t, v in ((150, 0.05), (200, 0.10), (650, 0.04))]
           + [nepi(602, 300, 0.0), nepi(602, 305, 0.0), nepi(601, 700, 0.0)])
    _write(src / "ward_vitals.parquet", [dict(subject_id=112, chart_time=1000, item_name="hr", value=80.0)])
    _write(src / "labs.parquet", [dict(subject_id=112, chart_time=1000, item_name="crp", value=1.0)])
    _write(src / "diagnosis.parquet", [dict(subject_id=112, chart_time=50, icd10_cm="I25.1")])
    _write(src / "medications.parquet", [dict(subject_id=112, chart_time=1000, route="po", drug_name="acetaminophen",
                                              atc_code="N02BE01")])
    out = tmp_path / "out"
    InspireAdapter(src, out, tmp_path / "duck").run()
    mac = read(out, "medication_admin_continuous").sort_values("admin_dttm")
    assert [(t, a, d) for t, a, d in zip(mac["admin_dttm"], mac["mar_action_category"], mac["med_dose"])] == [
        (at(150), "start", 0.05), (at(200), "dose_change", 0.10), (at(300), "stop", 0.0), (at(650), "start", 0.04),
        (at(700), "stop", 0.0)]


def test_icu_stay_belongs_to_the_admission_containing_it(clif: Path) -> None:
    s = read(clif, "ext_icu_stay")
    s = s[s.patient_id == "103"]
    assert s[["hospitalization_id", "admission_source", "stay_seq"]].values.tolist() == [["204", "or", 1]]
    adt = read(clif, "adt")
    assert (adt.hospitalization_id + ":" + adt.location_category).isin(["204:icu"]).sum() == 1
    assert not (adt.hospitalization_id + ":" + adt.location_category).isin(["203:icu"]).any()
