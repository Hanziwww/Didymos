"""Supplements to reference-tool CLIF output, and how S2b unions and time-shifts their parts."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from icutwin.clif.harmonize import harmonize_dir
from icutwin.clif.supplements import SUPPLEMENT_REPORT
from icutwin.clif.supplements.mimic import MimicSupplement

REFS = Path(__file__).resolve().parents[1] / "refs"
T = lambda h, m=0: datetime(2150, 3, 1, h, m, tzinfo=UTC)  # noqa: E731


def _ts(h: int, m: int = 0, day: int = 1) -> pd.Timestamp:
    return pd.Timestamp(2150, 3, day, h, m)


# -- S2b: parts and wall-clock shift ----------------------------------------------------------------
def _position(path: Path, times: list[datetime]) -> None:
    pq.write_table(pa.table({
        "hospitalization_id": ["H"] * len(times), "recorded_dttm": pa.array(times, pa.timestamp("us", tz="UTC")),
        "position_name": ["prone"] * len(times), "position_category": ["prone"] * len(times)}), path)


def _tool_dir(root: Path, tool_finished: str = "t1", built_for: str = "t1", listed: tuple = ("position",)) -> Path:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    _position(raw / "clif_position.parquet", [T(13, 21)])            # tool output in true UTC (EST + 5 h)
    _position(raw / "clif_position+icutwin.parquet", [T(9, 0)])      # our supplement: wall-clock already
    (raw / "_provenance.json").write_text(json.dumps({"finished_at": tool_finished}))
    (raw / SUPPLEMENT_REPORT).write_text(json.dumps({"tables": {t: {} for t in listed},
                                                     "tool_run": {"finished_at": built_for}}))
    return raw


def test_only_tool_part_is_shifted_and_parts_are_unioned(tmp_path: Path) -> None:
    raw = _tool_dir(tmp_path)
    report = harmonize_dir(raw, tmp_path / "out", tmp_path / "duck", utc_offset_hours=-5)
    got = sorted(pd.read_parquet(tmp_path / "out" / "clif_position.parquet")["recorded_dttm"])
    assert got == [pd.Timestamp(T(8, 21)), pd.Timestamp(T(9, 0))]
    assert report["tables"]["position"]["parts"] == {"clif_position+icutwin.parquet": 1, "clif_position.parquet": 1}
    assert report["time_columns"]["position"] == {"shifted_hours": -5, "shifted": ["recorded_dttm"], "labelled_utc": []}


def test_stale_supplement_is_rejected(tmp_path: Path) -> None:
    raw = _tool_dir(tmp_path, tool_finished="t2", built_for="t1")    # tool re-ran after the supplement
    with pytest.raises(ValueError, match="re-run task=supplement"):
        harmonize_dir(raw, tmp_path / "out", tmp_path / "duck")


def test_part_not_written_by_current_supplement_is_rejected(tmp_path: Path) -> None:
    raw = _tool_dir(tmp_path, listed=())
    with pytest.raises(ValueError, match="not written by the current supplement"):
        harmonize_dir(raw, tmp_path / "out", tmp_path / "duck")


# -- MIMIC supplement on a hand-built fixture (MIMIC-IV layout, no real patients) -------------------
@pytest.fixture(scope="module")
def mimic(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("mimic_sup")
    hosp, icu = d / "src/hosp", d / "src/icu"
    hosp.mkdir(parents=True)
    icu.mkdir(parents=True)
    adm = lambda s, h, typ, loc, death=pd.NaT: {"subject_id": s, "hadm_id": h, "admittime": _ts(0), "dischtime": _ts(23),  # noqa: E731
                                                "deathtime": death, "admission_type": typ, "admission_location": loc,
                                                "edregtime": pd.Timestamp(2150, 2, 28, 20) if h == 10 else pd.NaT}
    pd.DataFrame([adm(1, 10, "EW EMER.", "EMERGENCY ROOM"), adm(1, 11, "EW EMER.", "EMERGENCY ROOM"),
                  adm(2, 12, "SURGICAL SAME DAY ADMISSION", "PHYSICIAN REFERRAL", _ts(20)),
                  adm(3, 13, "URGENT", "TRANSFER FROM HOSPITAL")]).to_parquet(hosp / "admissions.parquet")
    pd.DataFrame({"subject_id": [1, 2, 3], "anchor_year_group": ["2011 - 2013", "2017 - 2019", None]}
                 ).to_parquet(hosp / "patients.parquet")
    tr = lambda h, unit, a, b, ev="transfer": {"subject_id": 0, "hadm_id": h, "transfer_id": 0, "eventtype": ev,  # noqa: E731
                                               "careunit": unit, "intime": a, "outtime": b}
    pd.DataFrame([
        tr(10, "Emergency Department", _ts(1), _ts(2), "ED"), tr(10, "Medical Intensive Care Unit (MICU)", _ts(2), _ts(22)),
        tr(10, "Medicine", _ts(22), _ts(23)),
        tr(11, "Neuro Stepdown", _ts(2), _ts(9)),                                  # icustays row, never an ICU
        tr(12, "Cardiac Vascular Intensive Care Unit (CVICU)", _ts(3), _ts(21)),  # direct admission
        tr(13, "PACU", _ts(4), _ts(6)), tr(13, "Surgical Intensive Care Unit (SICU)", _ts(6), _ts(23)),
    ]).to_parquet(hosp / "transfers.parquet")
    pd.DataFrame([{"subject_id": 1, "hadm_id": 10, "transfertime": _ts(0), "prev_service": None, "curr_service": "MED"},
                  {"subject_id": 2, "hadm_id": 12, "transfertime": _ts(1), "prev_service": None, "curr_service": "CSURG"},
                  {"subject_id": 3, "hadm_id": 13, "transfertime": _ts(0), "prev_service": None, "curr_service": "MED"},
                  {"subject_id": 3, "hadm_id": 13, "transfertime": _ts(23), "prev_service": "MED", "curr_service": "ORTHO"}]
                 ).to_parquet(hosp / "services.parquet")
    st = lambda s, h, sid, unit, a, b: {"subject_id": s, "hadm_id": h, "stay_id": sid, "first_careunit": unit,  # noqa: E731
                                        "last_careunit": unit, "intime": a, "outtime": b, "los": 0.5}
    pd.DataFrame([st(1, 10, 100, "Medical Intensive Care Unit (MICU)", _ts(2), _ts(22)),
                  st(1, 11, 101, "Neuro Stepdown", _ts(2), _ts(9)),
                  st(2, 12, 102, "Cardiac Vascular Intensive Care Unit (CVICU)", _ts(3), _ts(21)),
                  st(3, 13, 103, "Surgical Intensive Care Unit (SICU)", _ts(6), _ts(23))]).to_parquet(icu / "icustays.parquet")
    le = lambda i, h, item, v, unit, store=True, spec=1, comments=None: {  # noqa: E731
        "labevent_id": i, "subject_id": 1, "hadm_id": h, "specimen_id": spec, "itemid": item, "charttime": _ts(5),
        "storetime": _ts(6) if store else pd.NaT, "value": None if v is None else str(v), "valuenum": v,
        "valueuom": unit, "comments": comments}
    pd.DataFrame([
        le(1, 10, 51221, 30.0, "%"), le(2, 10, 51221, 31.0, "%", store=False),      # no result time: dropped
        le(3, 10, 51221, 30.0, "%"),                                                # exact duplicate: dropped
        le(4, None, 51221, 29.0, "%"),                                              # outpatient lab: dropped
        le(5, 10, 50866, None, "umol/L", comments="LESS THAN 10"),                  # value from the comment
        le(6, 10, 50866, None, "umol/L", comments="NOT DONE"),                      # kept, no numeric value
        le(7, 10, 50821, 40.0, "mm Hg", spec=7), {**le(8, 10, 52033, None, None, spec=7), "value": "MIX."},
        le(9, 10, 50821, 95.0, "mm Hg", spec=9), {**le(10, 10, 52033, None, None, spec=9), "value": "ART."},
        le(11, 10, 51196, 1500.0, "ng/mL FEU"), le(12, 10, 51196, 700.0, "ng/mL"),  # DDU/FEU unknown: unmapped
        {**le(13, None, 51221, 28.0, "%"), "charttime": pd.Timestamp(2150, 2, 28, 22),   # in the ED before admission 10
         "storetime": pd.Timestamp(2150, 2, 28, 23)},
    ]).to_parquet(hosp / "labevents.parquet")
    pd.DataFrame([(51221, "Hematocrit"), (50866, "Ammonia"), (50821, "pO2"), (51196, "D-Dimer"), (52033, "Specimen Type")],
                 columns=["itemid", "label"]).to_parquet(hosp / "d_labitems.parquet")
    ce = lambda item, h, v, unit=None: {"subject_id": 1, "hadm_id": 10, "stay_id": 100, "caregiver_id": 1,  # noqa: E731
                                        "charttime": _ts(h), "storetime": _ts(h, 5), "itemid": item, "value": str(v),
                                        "valuenum": v, "valueuom": unit, "warning": 0}
    pd.DataFrame([ce(225664, 7, 140.0), ce(220074, 8, 9.0, "mmHg"), ce(220059, 8, 35.0, "mmHg"),
                  ce(220045, 8, 80.0, "bpm"), ce(226499, 12, 2000.0, "mL")]).to_parquet(icu / "chartevents.parquet")
    pd.DataFrame([{"subject_id": 1, "hadm_id": 10, "stay_id": 100, "caregiver_id": 1, "starttime": _ts(9),
                   "endtime": _ts(12), "storetime": _ts(12), "itemid": 225441, "value": 180.0, "valueuom": "min",
                   "location": None, "locationcategory": None, "orderid": 1, "linkorderid": 1,
                   "ordercategoryname": "Procedures", "ordercategorydescription": "Task", "patientweight": 80.0,
                   "isopenbag": 0, "continueinnextdept": 0, "statusdescription": "FinishedRunning",
                   "originalamount": None, "originalrate": None}]).to_parquet(icu / "procedureevents.parquet")
    pd.DataFrame([(225664, "Glucose finger stick (range 70-100)"), (220074, "Central Venous Pressure"),
                  (220059, "Pulmonary Artery Pressure systolic"), (220045, "Heart Rate"), (227536, "KCl (CRRT)"),
                  (229618, "Calcium Chloride")],
                 columns=["itemid", "label"]).to_parquet(icu / "d_items.parquet")
    out = d / "out"
    out.mkdir()
    (out / "_provenance.json").write_text(json.dumps({"tool": "CLIF-MIMIC", "finished_at": "t1"}))
    mb = lambda i, spec, hadm, spec_desc, test_id, test, org_id=None, org=None, ab=None, interp=None, comments=None: {  # noqa: E731
        "microevent_id": i, "subject_id": 1, "hadm_id": hadm, "micro_specimen_id": spec, "order_provider_id": None,
        "chartdate": _ts(0), "charttime": _ts(4), "spec_itemid": 1, "spec_type_desc": spec_desc, "test_seq": 1,
        "storedate": _ts(0), "storetime": _ts(20), "test_itemid": test_id, "test_name": test, "org_itemid": org_id,
        "org_name": org, "isolate_num": 1 if org else None, "quantity": None, "ab_itemid": None, "ab_name": ab,
        "dilution_text": None, "dilution_comparison": None, "dilution_value": None, "interpretation": interp,
        "comments": comments}
    pd.DataFrame([
        mb(1, 501, 10, "BLOOD CULTURE", 90201, "Blood Culture, Routine", 80002, "ESCHERICHIA COLI"),
        mb(2, 501, 10, "BLOOD CULTURE", 90201, "Blood Culture, Routine", 80002, "ESCHERICHIA COLI", "CEFTRIAXONE", "S"),
        mb(3, 501, 10, "BLOOD CULTURE", 90201, "Blood Culture, Routine", 80002, "ESCHERICHIA COLI", "AMPICILLIN", "R"),
        mb(4, 502, 10, "URINE", 90039, "URINE CULTURE", comments="NO GROWTH."),
        mb(5, 503, 10, "SPUTUM", 90200, "GRAM STAIN",
           comments="2+   (1-5 PER 1000X FIELD):   POLYMORPHONUCLEAR LEUKOCYTES.  NO MICROORGANISMS SEEN."),
        mb(6, 504, 10, "STOOL", 90300, "C. difficile PCR", comments="NEGATIVE"),           # non-culture test
        mb(7, 505, 10, "BLOOD CULTURE", 90201, "Blood Culture, Routine", comments="___"),  # de-identified result
        mb(8, 506, 10, "STOOL", 90050, "FECAL CULTURE", comments="NO SALMONELLA OR SHIGELLA FOUND."),
        mb(9, 507, None, "URINE", 90039, "URINE CULTURE", comments="NO GROWTH."),         # outpatient specimen
        mb(10, 508, 10, "STOOL", 90300, "C. difficile PCR", 80139, "CLOSTRIDIUM DIFFICILE", comments="___"),
        mb(11, 509, 10, "STOOL", 90300, "C. difficile PCR", 90856, "CANCELLED"),         # no result: dropped
        mb(12, 510, 10, "STOOL", 90300, "C. difficile PCR", comments="___"),             # redacted: dropped
        mb(13, 511, 10, "STOOL", 90301, "CLOSTRIDIUM DIFFICILE TOXIN A & B TEST",        # immunoassay, not PCR
           comments="FECES NEGATIVE FOR C. DIFFICILE TOXIN A & B BY EIA."),
    ]).to_parquet(hosp / "microbiologyevents.parquet")
    # eMAR: ICU stay 100 of admission 10 runs 02:00-22:00
    em = lambda i, h, med, ev="Administered", pid=None: {  # noqa: E731
        "subject_id": 1, "hadm_id": 10, "emar_id": f"1-{i}", "emar_seq": i, "poe_id": None, "pharmacy_id": pid,
        "enter_provider_id": None, "charttime": _ts(h), "medication": med, "event_txt": ev, "scheduletime": _ts(h),
        "storetime": _ts(h)}
    pd.DataFrame([
        em(1, 5, "OxyCODONE (Immediate Release)", pid=901),   # ICU, oral (route from the pharmacy order): added
        em(2, 6, "Vancomycin"),                               # ICU, IV, a category CLIF-MIMIC emits: not added
        em(3, 7, "Dexamethasone"),                            # ICU, IV, a category CLIF-MIMIC lacks: added
        em(4, 23, "Vancomycin"),                              # after ICU discharge: added
        em(5, 8, "Albuterol 0.083% Neb Soln"),                # inhaled: local intermittent route extension (M4)
        em(6, 9, "Heparin"),                                  # infusion: not an intermittent dose
        em(7, 10, "Dexamethasone", ev="Not Given"),
        em(8, 1, "Penicillin G Potassium"),                   # before ICU admission; dose in million units
        em(9, 11, "Albumin 25% (12.5g / 50ml)"),              # ICU, IV: CLIF-MIMIC charts albumin as an infusion
        em(10, 0, "Albumin 5% (12.5g / 250ml)"),              # before ICU admission: added (local extension)
    ]).to_parquet(hosp / "emar.parquet")
    ed = lambda i, ordinal, route=None, dose=None, unit=None, typ=None: {  # noqa: E731
        "subject_id": 1, "emar_id": f"1-{i}", "emar_seq": i, "parent_field_ordinal": ordinal,
        "administration_type": typ, "dose_given": dose, "dose_given_unit": unit, "route": route}
    pd.DataFrame([
        ed(1, None, typ="Standard Maintenance Medication"), ed(1, 1.1, dose="5", unit="mg"),
        ed(2, None, "IV"), ed(2, 1.1, dose="1000", unit="mg"),
        ed(3, None, "IV"), ed(3, 1.1, dose="10", unit="mg"),
        ed(4, None, "IV"), ed(4, 1.1, dose="750", unit="mg"), ed(4, 1.2, dose="250", unit="mg"),
        ed(5, None, "IH"), ed(5, 1.1, dose="2.5", unit="mg"),
        ed(6, None, "IV DRIP", typ="IV Infusion"), ed(6, 1.1, dose="1000", unit="UNIT"),
        ed(8, None, "IV"), ed(8, 1.1, dose="3", unit="Million Units"),
        ed(9, None, "IV"), ed(9, 1.1, dose="12.5", unit="g"),
        ed(10, None, "IV"), ed(10, 1.1, dose="12.5", unit="g"),
    ]).to_parquet(hosp / "emar_detail.parquet")
    pd.DataFrame([{"pharmacy_id": 901, "route": "PO"}]).to_parquet(hosp / "pharmacy.parquet")
    ie = lambda item, h, amount, status="FinishedRunning", uom="mL": {  # noqa: E731
        "subject_id": 1, "hadm_id": 10, "stay_id": 100, "caregiver_id": 1, "starttime": _ts(h - 1), "endtime": _ts(h),
        "storetime": _ts(h), "itemid": item, "amount": amount, "amountuom": uom, "rate": None, "rateuom": None,
        "orderid": h, "linkorderid": h, "ordercategoryname": "x", "secondaryordercategoryname": None,
        "ordercomponenttypedescription": None, "ordercategorydescription": None, "patientweight": 80.0,
        "totalamount": amount, "totalamountuom": uom, "isopenbag": 0, "continueinnextdept": 0,
        "statusdescription": status, "originalamount": amount, "originalrate": None}
    drip = lambda h, rate, order=12, status="ChangeDose/Rate": {  # noqa: E731   # CRRT potassium, one order
        **ie(227536, h, rate, status=status, uom="mEq."), "rate": rate, "rateuom": "mEq./hour", "orderid": h,
        "linkorderid": order, "ordercategoryname": "01-Drips", "ordercategorydescription": "Continuous Med"}
    pd.DataFrame([
        ie(225168, 6, 350.0),                         # packed red cells: a transfusion
        ie(225828, 7, 1000.0),                        # lactated Ringer's: no CLIF-MIMIC item
        ie(225158, 8, 500.0),                         # NaCl 0.9%: the tool's item, not re-mapped
        ie(225168, 9, 300.0, status="Rewritten"),     # erroneous record
        ie(225970, 10, 10.0, uom="grams"),            # protein powder: not a volume
        drip(12, 10.0), drip(13, 20.0, status="FinishedRunning"),
        drip(14, 5.0, order=14, status="Rewritten"),  # erroneous record
        {**ie(229618, 15, 1000.0, uom="mg"), "ordercategoryname": "05-Med Bolus",
         "ordercategorydescription": "Drug Push"},     # calcium chloride push: an eMAR dose
    ]).to_parquet(icu / "inputevents.parquet")
    MimicSupplement(d / "src", out, d / "duck").run()
    return out


def _read(out: Path, table: str) -> pd.DataFrame:
    return pd.read_parquet(out / f"clif_{table}+icutwin.parquet")


def test_mimic_icu_stays(mimic: Path) -> None:
    s = _read(mimic, "ext_icu_stay").set_index("stay_id")
    assert list(s.index) == ["mimic:100", "mimic:102", "mimic:103"]              # stepdown-only stay excluded
    assert s.loc["mimic:100", ["admission_source", "surgical_status", "unit_type", "icu_outcome"]].tolist() == [
        "ed", "nonsurgical", "medical_icu", "alive"]
    assert s.loc["mimic:102", ["admission_source", "surgical_status", "unit_type", "icu_outcome"]].tolist() == [
        "direct", "elective_surgery", "mixed_cardiothoracic_icu", "dead"]    # no prior segment: admission location
    assert json.loads(s.loc["mimic:102", "surgery_context"]) == {"service": "CSURG"}
    # PACU before the ICU; the orthopaedic service taken up 17 h after ICU admission is post-admission
    # information (mimic-code OASIS would count it within the first day): not surgical at admission
    assert s.loc["mimic:103", ["admission_source", "surgical_status"]].tolist() == ["or", "nonsurgical"]
    # the service gives the system: general medicine none; the surgical service of an operated stay
    assert s["adm_dx_group"].to_dict() == {"mimic:100": "medical_unspecified", "mimic:102": "surgical_cardiovascular",
                                           "mimic:103": "medical_unspecified"}   # general medicine at admission
    assert s.loc["mimic:100", "icu_in_dttm"] == T(2) and s.loc["mimic:100", "stay_seq"] == 1
    # era: the patient's anchor year group as released (M4 prefix field)
    assert s["era"].tolist() == ["2011-2013", "2017-2019", None]


def test_mimic_labs(mimic: Path) -> None:
    lab = _read(mimic, "labs")
    got = sorted((r.lab_category, r.lab_value_numeric) for r in lab.itertuples() if pd.notna(r.lab_value_numeric))
    assert got == [("d_dimer", 1.5), ("ext_ammonia", 10.0), ("glucose", 140.0), ("hematocrit", 28.0),
                   ("hematocrit", 30.0), ("po2_mixed_venous", 40.0)]
    ed = lab[(lab.lab_category == "hematocrit") & (lab.lab_value_numeric == 28.0)]   # ED lab: its admission
    assert ed["hospitalization_id"].tolist() == ["10"]
    assert lab["lab_value_numeric"].isna().sum() == 1                           # 'NOT DONE' kept without a value
    assert lab.loc[(lab.lab_category == "hematocrit") & (lab.lab_value_numeric == 30.0), "lab_result_dttm"].tolist() == [T(6)]
    report = json.loads((mimic / SUPPLEMENT_REPORT).read_text())
    assert report["audit"]["labs"]["itemid_unit"]["top_unmapped"] == {"labevents:51196 [ng/mL]": 1}
    assert report["tool_run"]["finished_at"] == "t1"


def test_mimic_hemodynamics(mimic: Path) -> None:
    h = _read(mimic, "invasive_hemodynamics").set_index("measurement_category")["measurement_value"]
    assert h.to_dict() == {"cvp": 9.0, "pa_systolic": 35.0}


def test_mimic_supplement_never_remaps_a_tool_item() -> None:
    """Every supplement item is unmapped in CLIF-MIMIC, except pO2 of mixed-venous specimens, which patch
    0002 removes from the tool's po2_arterial."""
    tool = REFS / "CLIF-MIMIC" / "data" / "mappings" / "mimic-to-clif-mappings - labs.csv"
    if not tool.exists():
        pytest.skip("CLIF-MIMIC reference clone not present")
    mapped = {int(float(r["itemid"])) for r in csv.DictReader(tool.open(encoding="utf-8"))
              if r["itemid"] and r["decision"] in ("TO MAP, CONVERT UOM", "TO MAP, AS IS", "UNSURE")}
    ours = csv.DictReader(resources.files("icutwin.clif").joinpath("maps", "mimic", "labs.csv").open(encoding="utf-8"))
    overlap = {(int(r["itemid"]), r["specimen"]) for r in ours if int(r["itemid"]) in mapped}
    assert overlap == {(50821, "MIX.")}
    patch = (REFS.parent / "patches" / "CLIF-MIMIC" / "0002-route-blood-gases-by-specimen.patch").read_text()
    assert "e.itemid = 50821 AND coalesce(s.specimen_type, '') IN ('VEN.', 'MIX.', 'CENTRAL VENOUS.')" in patch


def test_naive_tool_timestamps_are_labelled_utc_without_shift(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pq.write_table(pa.table({"hospitalization_id": ["H"], "recorded_dttm": pa.array([datetime(2010, 1, 1, 7, 30)], pa.timestamp("us")),
                             "position_name": ["prone"], "position_category": ["prone"]}), raw / "clif_position.parquet")
    report = harmonize_dir(raw, tmp_path / "out", tmp_path / "duck")
    out = pq.read_table(tmp_path / "out" / "clif_position.parquet")
    assert str(out.schema.field("recorded_dttm").type) == "timestamp[us, tz=UTC]"
    assert out.column("recorded_dttm").to_pylist() == [datetime(2010, 1, 1, 7, 30, tzinfo=UTC)]
    assert report["time_columns"]["position"]["labelled_utc"] == ["recorded_dttm"]


def test_mimic_microbiology(mimic: Path) -> None:
    c = _read(mimic, "microbiology_culture")
    got = sorted(zip(c["method_category"], c["fluid_category"], c["organism_category"]))
    assert got == [("culture", "blood_buffy", "escherichia_coli"), ("culture", "feces_stool", "no_growth"),
                   ("culture", "genito_urinary_tract", "no_growth"), ("gram_stain", "respiratory_tract_lower", "no_growth")]
    assert set(c["order_id"]) == {"501:90201", "502:90039", "503:90200", "506:90050"}
    assert c.loc[c.organism_category == "escherichia_coli", "result_dttm"].tolist() == [T(20)]
    s = _read(mimic, "microbiology_susceptibility").sort_values("antimicrobial_category")
    assert s[["antimicrobial_category", "susceptibility_category"]].values.tolist() == [
        ["ampicillin", "non_susceptible"], ["ceftriaxone", "susceptible"]]
    assert set(s["organism_id"]) == set(c.loc[c.organism_category == "escherichia_coli", "organism_id"])


def test_mimic_microbiology_nonculture(mimic: Path) -> None:
    nc = _read(mimic, "microbiology_nonculture").sort_values("order_dttm")
    assert sorted(zip(nc["micro_order_name"], nc["result_category"])) == [
        ("C. difficile PCR", "detected"), ("C. difficile PCR", "not_detected")]
    assert set(nc["organism_category"]) == {"clostridium_difficile"} and set(nc["method_category"]) == {"pcr"}
    assert set(nc["fluid_category"]) == {"feces_stool"}
    # PCR tests never appear as cultures
    assert "C. difficile PCR" not in set(_read(mimic, "microbiology_culture")["method_name"])


def test_mimic_emar_doses_complete_the_tool(mimic: Path) -> None:
    m = _read(mimic, "medication_admin_intermittent").sort_values("admin_dttm")
    got = [(t.hour, c, r, d, u) for t, c, r, d, u in zip(m["admin_dttm"], m["med_category"], m["med_route_category"],
                                                       m["med_dose"], m["med_dose_unit"])]
    # the nebulized albuterol and the pre-ICU albumin take local extensions (M4); the ICU albumin is not added
    assert got == [(0, "albumin_5", "iv", 12.5, "g"), (1, "penicillin", "iv", 3.0, "million units"),
                   (5, "oxycodone", "enteral", 5.0, "mg"), (7, "dexamethasone", "iv", 10.0, "mg"),
                   (8, "albuterol", "inhaled", 2.5, "mg"), (23, "vancomycin", "iv", 1000.0, "mg")]


def test_mimic_drips_the_tool_does_not_map(mimic: Path) -> None:
    """One start (rate) and one stop per record and order; rewritten records and bolus pushes are not drips."""
    c = _read(mimic, "medication_admin_continuous").sort_values(["admin_dttm", "mar_action_category"],
                                                                 ascending=[True, False])
    got = [(t.hour, cat, a, d, u, o) for t, cat, a, d, u, o in zip(
        c["admin_dttm"], c["med_category"], c["mar_action_category"], c["med_dose"], c["med_dose_unit"],
        c["med_order_id"])]
    assert got == [(11, "potassium_repletion", "start", 10.0, "mEq./hour", "12"),
                   (12, "potassium_repletion", "stop", 0.0, "mEq./hour", "12"),
                   (12, "potassium_repletion", "start", 20.0, "mEq./hour", "12"),
                   (13, "potassium_repletion", "stop", 0.0, "mEq./hour", "12")]
    assert set(c["med_route_category"]) == {"iv"}


def test_tool_category_list_matches_the_pinned_clif_mimic_mapping() -> None:
    mapping = REFS / "CLIF-MIMIC" / "data" / "mappings" / "mimic-to-clif-mappings - med_category.csv"
    if not mapping.exists():
        pytest.skip("CLIF-MIMIC is not cloned in refs/")
    from icutwin.clif.tools.mimic_tool_categories import tool_categories
    for kind in ("intermittent", "continuous"):
        text = resources.files("icutwin.clif").joinpath("maps", "mimic", f"tool_{kind}_categories.csv").read_text()
        assert {r["med_category"] for r in csv.DictReader(text.splitlines())} == set(tool_categories(mapping, kind))


def test_mimic_intake_the_tool_does_not_map(mimic: Path) -> None:
    i = _read(mimic, "input").sort_values("recorded_dttm")
    assert list(zip([t.hour for t in i["recorded_dttm"]], i["input_category"], i["input_volume"])) == [
        (6, "blood_products", 350.0), (7, "iv_crystalloid", 1000.0)]


def test_mimic_input_items_are_unmapped_in_clif_mimic() -> None:
    tool = REFS / "CLIF-MIMIC" / "data" / "mappings"
    if not tool.exists():
        pytest.skip("CLIF-MIMIC reference clone not present")
    mapped = set()
    for name in ("med_category", "input"):
        for r in csv.DictReader((tool / f"mimic-to-clif-mappings - {name}.csv").open(encoding="utf-8")):
            if r.get("decision") not in ("NO MAPPING", "NOT AVAILABLE", "UNSURE", None, ""):
                mapped.add(int(float(r["itemid"])))
    ours = csv.DictReader(resources.files("icutwin.clif").joinpath("maps", "mimic", "input_items.csv").open(encoding="utf-8"))
    assert {int(r["itemid"]) for r in ours} & mapped == set()


def test_mimic_hemodialysis_sessions(mimic: Path) -> None:
    d = _read(mimic, "intermittent_dialysis").sort_values("recorded_dttm")
    assert [t.hour for t in d["recorded_dttm"]] == [9, 12, 12]      # session start, end, output charting
    assert d["net_ultrafiltration_out"].dropna().tolist() == [2000.0]
