"""eICU adapter on a hand-built fixture (eICU column layout, no real patients)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from icutwin.clif.adapters.eicu import PID_BASE, EicuAdapter

HOSP_DISCH = datetime(2014, 12, 31, 17, 40, tzinfo=UTC)
# eICU documentation example: one hospital stay, unit stays 10 (first) and 2 (second)
ADMIT_10 = HOSP_DISCH - timedelta(minutes=15280)
ADMIT_2 = HOSP_DISCH - timedelta(minutes=10960)


def _write(d: Path, name: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(d / f"{name}.parquet", index=False)


@pytest.fixture(scope="module")
def out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("eicu")
    src = d / "src"
    src.mkdir()
    pat = dict(patienthealthsystemstayid=800, gender="Female", age="> 89", ethnicity="Hispanic", hospitalid=73,
               wardid=1, apacheadmissiondx="Sepsis, pulmonary", admissionheight=160.0, hospitaladmittime24="06:20:00",
               hospitaladmitsource="Emergency Department", hospitaldischargeyear=2014,
               hospitaldischargetime24="17:40:00", hospitaldischargelocation="Death", hospitaldischargestatus="Expired",
               unittype="MICU", unitadmittime24="18:20:00", unitadmitsource="Emergency Department",
               unitstaytype="admit", admissionweight=70.0, dischargeweight=None, unitdischargetime24="06:20:00",
               unitdischargelocation="Floor", unitdischargestatus="Alive", uniquepid="002-34851")
    _write(src, "patient", [
        {**pat, "patientunitstayid": 10, "hospitaladmitoffset": -720, "hospitaldischargeoffset": 15280,
         "unitvisitnumber": 1, "unitdischargeoffset": 2160},
        {**pat, "patientunitstayid": 2, "hospitaladmitoffset": -5040, "hospitaldischargeoffset": 10960,
         "unitvisitnumber": 2, "unitdischargeoffset": 4320, "unitadmitsource": "ICU to SDU",
         "unitdischargestatus": "Expired"},
    ])
    _write(src, "hospital", [{"hospitalid": 73, "numbedscategory": "<100", "teachingstatus": True, "region": "West"}])
    lab = lambda off, name, val, unit, iface=...: {  # noqa: E731
        "labid": off, "patientunitstayid": 10, "labresultoffset": off, "labtypeid": 1, "labname": name,
        "labresult": val, "labresulttext": str(val), "labmeasurenamesystem": unit,
        "labmeasurenameinterface": unit if iface is ... else iface, "labresultrevisedoffset": off}
    _write(src, "lab", [lab(10, "calcium", 2.0, "mmol/L"), lab(10, "troponin - T", 0.05, "ng/mL"),
                        lab(10, "Base Deficit", 3.0, "mEq/L"), lab(10, "glucose", 120.0, "mg/dL"),
                        lab(10, "bedside glucose", 140.0, "mg/dL"), lab(10, "FiO2", 60.0, "%"),
                        # results are in the interface unit; the system unit is a label
                        lab(11, "ammonia", 41.0, "mcg/dL", "umol/L"), lab(12, "ammonia", 62.0, "mcg/dL", "ug/dL"),
                        lab(13, "ammonia", 4.1, "mcg/dL", "mcmol/L"), lab(11, "CRP", 7.2, "mg/dL", "mg/dL"),
                        lab(12, "CRP", 750.0, "mg/dL", "mg/L"), lab(11, "-lymphs", 12.0, "%", "%"),
                        lab(12, "-lymphs", 1.1, "%", None), lab(14, "FiO2", 4.0, "%", "lpm")])
    vp_cols = ["temperature", "sao2", "heartrate", "respiration", "cvp", "etco2", "systemicsystolic",
               "systemicdiastolic", "systemicmean", "pasystolic", "padiastolic", "pamean", "st1", "st2", "st3", "icp"]
    _write(src, "vitalPeriodic", [{"vitalperiodicid": 1, "patientunitstayid": 10, "observationoffset": 5,
                                   **{c: None for c in vp_cols}, "temperature": 99.5, "heartrate": 80.0, "cvp": 9.0}])
    va_cols = ["noninvasivesystolic", "noninvasivediastolic", "noninvasivemean", "paop", "cardiacoutput",
               "cardiacinput", "svr", "svri", "pvr", "pvri"]
    _write(src, "vitalAperiodic", [{"vitalaperiodicid": 1, "patientunitstayid": 10, "observationoffset": 7,
                                    **{c: None for c in va_cols}, "noninvasivesystolic": 120.0}])
    nc = lambda off, cat, label, name, val: {"nursingchartid": off, "patientunitstayid": 10,  # noqa: E731
                                             "nursingchartoffset": off, "nursingchartentryoffset": off,
                                             "nursingchartcelltypecat": cat, "nursingchartcelltypevallabel": label,
                                             "nursingchartcelltypevalname": name, "nursingchartvalue": val}
    _write(src, "nurseCharting", [
        nc(20, "Vital Signs", "Temperature", "Temperature (C)", "37.0"),
        nc(20, "Scores", "Glasgow coma score", "Eyes", "3"),
        nc(20, "Scores", "Glasgow coma score", "GCS Total", "Unable to score due to medication"),
        nc(25, "Scores", "Sedation Scale/Score/Goal", "Sedation Scale", "RASS"),
        nc(25, "Scores", "Sedation Scale/Score/Goal", "Sedation Score", "-2"),
        nc(26, "Scores", "Delirium Scale/Score", "Delirium Scale", "CAM-ICU"),
        nc(26, "Scores", "Delirium Scale/Score", "Delirium Score", "Yes"),
        nc(90, "Vital Signs", "O2 L/%", "O2 L/%", "3"),
        nc(90, "Vital Signs", "O2 Admin Device", "O2 Admin Device", "nasal cannula"),
        nc(150, "Vital Signs", "O2 L/%", "O2 L/%", "40"),
        nc(150, "Vital Signs", "O2 Admin Device", "O2 Admin Device", "HFNC"),
    ])
    rc = lambda i, off, label, val: {"respchartid": i, "patientunitstayid": 10, "respchartoffset": off,  # noqa: E731
                                     "respchartentryoffset": off, "respcharttypecat": "respFlowSettings",
                                     "respchartvaluelabel": label, "respchartvalue": val}
    _write(src, "respiratoryCharting", [rc(1, 30, "FiO2", "40"), rc(2, 30, "O2 Device", "Ventilator"),
                                        rc(3, 30, "PEEP", "8")])
    inf = lambda i, off, name, rate: {"infusiondrugid": i, "patientunitstayid": 10, "infusionoffset": off,  # noqa: E731
                                      "drugname": name, "drugrate": rate, "infusionrate": None, "drugamount": None,
                                      "volumeoffluid": None, "patientweight": None}
    _write(src, "infusionDrug", [
        inf(1, 0, "Norepinephrine (mcg/min)", "5"), inf(2, 60, "Norepinephrine (mcg/min)", "5"),
        inf(3, 120, "Norepinephrine (mcg/min)", "8"), inf(4, 180, "Norepinephrine (mcg/min)", "0"),
        inf(5, 240, "Norepinephrine (mcg/min)", "4"),
        inf(6, 0, "Propofol ()", "20"),                                     # no rate unit: dropped (ricu)
        inf(7, 0, "Heparin (units/hr)", "1000"), inf(8, 500, "Heparin (units/hr)", "1200"),
        inf(9, 0, "Volume-Transfuse RBC (ml/hr)", "100"),                   # blood product: not a medication
    ])
    io = lambda i, path, label, val: {"intakeoutputid": i, "patientunitstayid": 10, "intakeoutputoffset": 60,  # noqa: E731
                                      "intaketotal": None, "outputtotal": None, "dialysistotal": None,
                                      "nettotal": None, "intakeoutputentryoffset": 60, "cellpath": path,
                                      "celllabel": label, "cellvaluenumeric": val, "cellvaluetext": str(val)}
    base = "flowsheet|Flowsheet Cell Labels|I&O|"
    _write(src, "intakeOutput", [
        io(1, base + "Output (ml)|Urine", "Urine", 200.0), io(2, base + "Output (ml)|Stool", "Stool", 1.0),
        io(3, base + "Output (ml)|Urine Count", "Urine Count", 2.0),
        io(4, base + "Intake (ml)|Blood Products (ml)|pRBCs", "pRBCs", 300.0),
        io(5, base + "Weight|Bodyweight (kg)", "Bodyweight (kg)", 72.5),
    ])
    _write(src, "treatment", [
        {"treatmentid": 1, "patientunitstayid": 10, "treatmentoffset": 30, "activeupondischarge": False,
         "treatmentstring": "pulmonary|ventilation and oxygenation|mechanical ventilation"},
        {"treatmentid": 2, "patientunitstayid": 10, "treatmentoffset": 40, "activeupondischarge": False,
         "treatmentstring": "renal|dialysis|C V V H D"},
        {"treatmentid": 3, "patientunitstayid": 10, "treatmentoffset": 41, "activeupondischarge": False,
         "treatmentstring": "renal|dialysis|insertion of venous catheter for hemodialysis"},
    ])
    dx = lambda i, off, code: {"diagnosisid": i, "patientunitstayid": 10, "activeupondischarge": False,  # noqa: E731
                               "diagnosisoffset": off, "diagnosisstring": "x", "icd9code": code,
                               "diagnosispriority": "Major"}
    _write(src, "diagnosis", [dx(1, 50, "518.81, J96.00"), dx(2, 60, "995.92"), dx(3, 70, "E87.1"),
                              dx(4, 80, "E980.2, 965.00, T40.60"), dx(5, 90, "518.81, J96.00")])
    _write(src, "admissionDx", [
        {"admissiondxid": 1, "patientunitstayid": 10, "admitdxenteredoffset": 0, "admitdxname": "Yes", "admitdxtext": "",
         "admitdxpath": "admission diagnosis|Was the patient admitted from the O.R. or went to the O.R. within 4 hours of admission?|Yes"},
        {"admissiondxid": 2, "patientunitstayid": 10, "admitdxenteredoffset": 0, "admitdxname": "No", "admitdxtext": "",
         "admitdxpath": "admission diagnosis|Elective|No"},
        {"admissiondxid": 3, "patientunitstayid": 10, "admitdxenteredoffset": 5, "admitdxtext": "",
         "admitdxname": "GI perforation/rupture, surgery for",
         "admitdxpath": "admission diagnosis|All Diagnosis|Operative|Diagnosis|Gastrointestinal|GI perforation/rupture, surgery for"},
        {"admissiondxid": 4, "patientunitstayid": 2, "admitdxenteredoffset": 5, "admitdxtext": "",
         "admitdxname": "Sepsis, pulmonary",   # APACHE IV files sepsis under cardiovascular: the name decides
         "admitdxpath": "admission diagnosis|All Diagnosis|Non-operative|Diagnosis|Cardiovascular|Sepsis, pulmonary"},
    ])
    _write(src, "apachePatientResult", [{"apachepatientresultsid": 1, "patientunitstayid": 10, "apacheversion": "IVa",
                                         "apachescore": 55}])
    cpl = lambda i, off, grp, val: {"cplgeneralid": i, "patientunitstayid": 10, "activeupondischarge": False,  # noqa: E731
                                    "cplitemoffset": off, "cplgroup": grp, "cplitemvalue": val}
    _write(src, "carePlanGeneral", [
        cpl(1, 100, "Care Limitation", "Do not resuscitate"), cpl(2, 100, "Care Limitation", "No intubation"),
        cpl(3, 300, "Care Limitation", "Do not resuscitate"), cpl(4, 300, "Care Limitation", "No intubation"),
        cpl(5, 400, "Care Limitation", "Comfort measures only"),
        cpl(6, 200, "Ventilation", "Ventilated - with daily extubation evaluation"),
    ])
    med = lambda i, name, hicl, dosage, route, freq, order, start, stop, prn=False, cancelled=False: {  # noqa: E731
        "medicationid": i, "patientunitstayid": 10, "drugorderoffset": order, "drugstartoffset": start,
        "drugivadmixture": False, "drugordercancelled": cancelled, "drugname": name, "drughiclseqno": hicl,
        "dosage": dosage, "routeadmin": route, "frequency": freq, "loadingdose": None, "prn": prn,
        "drugstopoffset": stop, "gtc": None}
    _write(src, "medication", [
        med(1, "piperacillin-tazobactam 3.375 g", 8862, "3.375 g", "IVPB", "Q8H", 30, 30, 1000),
        med(2, None, 1694, "2 mg", "IV", "Q2H PRN", 50, 50, 40, prn=True),     # name withheld: HICL -> morphine
        med(3, "mupirocin 2 % top oint", 7311, "1 Application", "Topical", "BID", 60, 60, 900),
        med(4, "moxifloxacin", 22412, "1 drop", "Both Eyes", "QID", 60, 60, 900),  # ophthalmic route
        med(5, "heparin 25,000 units in d5w 250 ml", 2810, "1,000 units/hr", "IV", "Continuous", 70, 70, 500),
        med(6, "vancomycin 1 gm in ns 250 ml ivpb", 10093, "1 GM", "IV", "Q12H", 80, 80, 800, cancelled=True),
        med(7, None, 999999, "5 mg", "PO", "Daily", 90, 90, 900),             # unknown HICL: unmapped
    ])
    mlab = lambda i, off, site, org, ab=None, level=None: {"microlabid": i, "patientunitstayid": 10,  # noqa: E731
                                                          "culturetakenoffset": off, "culturesite": site,
                                                          "organism": org, "antibiotic": ab, "sensitivitylevel": level}
    _write(src, "microLab", [
        mlab(1, 100, "Blood, Venipuncture", "Escherichia coli", "ciprofloxacin", "Resistant"),
        mlab(2, 100, "Blood, Venipuncture", "Escherichia coli", "meropenam", "Sensitive"),   # eICU spelling
        mlab(3, 100, "Blood, Venipuncture", "Escherichia coli", "Other", "Sensitive"),       # unmapped antibiotic
        mlab(4, 200, "Urine, Catheter Specimen", "no growth"),
    ])
    o = d / "out"
    EicuAdapter(src, o, d / "duck").run()
    return o


def _read(out: Path, table: str) -> pd.DataFrame:
    return pd.read_parquet(out / f"clif_{table}.parquet")


def test_identity_and_eicu_meds_time_anchoring(out: Path) -> None:
    p = _read(out, "patient").iloc[0]
    assert p["patient_id"] == str(2 * PID_BASE + 34851)
    assert p["death_dttm"] == HOSP_DISCH
    assert (p["race_category"], p["ethnicity_category"], p["sex_category"]) == ("other", "hispanic", "female")
    birth = (ADMIT_10 - timedelta(days=90 * 365.25 - 182.625)).date()   # "> 89" -> 90, first unit stay
    assert p["birth_date"] == birth
    h = _read(out, "hospitalization").iloc[0]
    assert h["admission_dttm"] == HOSP_DISCH - timedelta(minutes=16000)   # same for both unit stays
    assert (h["discharge_category"], h["admission_type_category"], h["age_at_admission"]) == ("expired", "ed", 90)
    stays = _read(out, "ext_icu_stay").set_index("stay_id")
    assert stays.loc["eicu:10", "icu_in_dttm"] == ADMIT_10 and stays.loc["eicu:2", "icu_in_dttm"] == ADMIT_2
    assert stays.loc["eicu:10", "stay_seq"] == 1 and stays.loc["eicu:2", "stay_seq"] == 2
    assert stays.loc["eicu:2", "admission_source"] == "other_icu" and stays.loc["eicu:2", "icu_outcome"] == "dead"
    assert stays.loc["eicu:10", "surgical_status"] == "emergency_surgery"
    assert stays.loc["eicu:10", "adm_dx_group"] == "surgical_gastrointestinal"
    assert stays.loc["eicu:2", "adm_dx_group"] == "medical_sepsis"
    assert stays.loc["eicu:10", "severity_value"] == 55 and stays.loc["eicu:10", "site_id"] == "eicu_73"
    assert set(_read(out, "adt")["hospital_type"]) == {"academic"}


def test_vitals_units_and_sites(out: Path) -> None:
    v = _read(out, "vitals")
    temps = sorted(v.loc[v.vital_category == "temp_c", "vital_value"])
    assert temps == [pytest.approx(37.0), pytest.approx(37.5)]          # 99.5 degF recognised and converted
    assert v.loc[v.vital_category == "sbp", "meas_site_name"].tolist() == ["cuff"]
    w = v[v.vital_category == "weight_kg"].sort_values("recorded_dttm")["vital_value"].tolist()
    assert w == [70.0, 72.5, 70.0]          # admission weight of each unit stay + I/O body weight


def test_labs_conversions(out: Path) -> None:
    lab = _read(out, "labs").set_index("lab_name")
    assert lab.loc["calcium", "lab_value_numeric"] == pytest.approx(8.016)
    assert lab.loc["troponin - T", "lab_value_numeric"] == pytest.approx(50.0)
    assert (lab.loc["Base Deficit", "lab_category"], lab.loc["Base Deficit", "lab_value_numeric"]) == ("ext_base_excess", -3.0)
    assert lab.loc["glucose", "lab_category"] == "glucose_serum"
    assert (lab.loc["bedside glucose", "lab_category"], lab.loc["bedside glucose", "lab_order_category"]) == ("glucose", "poc")
    assert "FiO2" not in lab.index                                        # ventilator setting, not a lab


def test_labs_by_interface_unit(out: Path) -> None:
    """Results in the hospital's interface unit: umol/L ammonia kept, ug/dL converted; units that mix scales across
    hospitals (ammonia 'mcmol/L', CRP 'mg/L') and differentials without a percent unit are not mapped."""
    lab = _read(out, "labs")
    lab["min"] = [int((t - ADMIT_10).total_seconds() // 60) for t in lab["lab_collect_dttm"]]
    got = {(r.lab_name, r.min): r.lab_value_numeric for r in lab.itertuples() if r.lab_name in ("ammonia", "CRP", "-lymphs")}
    assert got == {("ammonia", 11): pytest.approx(41.0), ("ammonia", 12): pytest.approx(62.0 * 0.5872),
                   ("CRP", 11): pytest.approx(72.0), ("-lymphs", 11): pytest.approx(12.0)}


def test_lab_fio2_charted_as_flow_is_not_a_fraction(out: Path) -> None:
    r = _read(out, "respiratory_support")
    r["min"] = [int((t - ADMIT_10).total_seconds() // 60) for t in r["recorded_dttm"]]
    assert 14 not in set(r["min"])      # the only record then: a lab 'FiO2' of 4 with interface unit 'lpm'


def test_infusion_rate_changes_and_inferred_stops(out: Path) -> None:
    m = _read(out, "medication_admin_continuous")
    assert set(m["med_category"]) == {"norepinephrine", "heparin"}        # no unit / blood products dropped
    ne = m[m.med_category == "norepinephrine"].sort_values("admin_dttm")
    minutes = [int((t - ADMIT_10).total_seconds() // 60) for t in ne["admin_dttm"]]
    assert list(zip(minutes, ne["mar_action_category"], ne["med_dose"])) == [
        (0, "start", 5.0), (120, "dose_change", 8.0), (180, "stop", 0.0), (240, "start", 4.0), (300, "stop", 0.0)]
    assert ne["mar_action_name"].iloc[-1].startswith("inferred stop") and ne["mar_action_name"].iloc[2] is None
    assert set(ne["med_dose_unit"]) == {"mcg/min"} and set(ne["med_group"]) == {"vasoactives"}
    hep = m[m.med_category == "heparin"].sort_values("admin_dttm")
    minutes = [int((t - ADMIT_10).total_seconds() // 60) for t in hep["admin_dttm"]]
    assert list(zip(minutes, hep["mar_action_category"])) == [(0, "start"), (60, "stop"), (500, "start"), (560, "stop")]


def test_respiratory_support_sources_and_priority(out: Path) -> None:
    r = _read(out, "respiratory_support")
    r["min"] = [int((t - ADMIT_10).total_seconds() // 60) for t in r["recorded_dttm"]]
    r = r.set_index("min")
    assert (r.loc[30, "device_name"], r.loc[30, "device_category"]) == ("Ventilator", "imv")   # charted beats treatment
    assert (r.loc[30, "fio2_set"], r.loc[30, "peep_set"]) == (pytest.approx(0.4), 8.0)
    assert (r.loc[90, "device_category"], r.loc[90, "lpm_set"]) == ("nasal_cannula", 3.0)
    assert r.loc[150, "device_category"] == "hfnc" and pd.isna(r.loc[150, "fio2_set"])         # ambiguous on HFNC
    assert r.loc[200, "device_category"] == "imv" and r.loc[200, "device_name"].startswith("care plan:")
    assert r.loc[10, "fio2_set"] == pytest.approx(0.6)                                         # FiO2 with blood gas


def test_assessments(out: Path) -> None:
    a = _read(out, "patient_assessments").set_index("assessment_category")
    assert a.loc["gcs_eye", "numerical_value"] == 3.0 and "gcs_total" not in a.index      # text result dropped
    assert a.loc["rass", "numerical_value"] == -2.0
    assert a.loc["cam_total", "categorical_value"] == "Positive"


def test_diagnoses_prefer_icd10_and_keep_first_entry(out: Path) -> None:
    d = _read(out, "patient_diagnosis")
    got = {(c, f) for c, f in zip(d["diagnosis_code"], d["diagnosis_code_format"])}
    assert got == {("J9600", "ICD10CM"), ("99592", "ICD9CM"), ("E871", "ICD10CM"), ("T4060", "ICD10CM")}
    j = d[d.diagnosis_code == "J9600"]["start_dttm"].iloc[0]
    assert int((j - ADMIT_10).total_seconds() // 60) == 50                # first of the repeated entries


def test_code_status_combines_limitations_and_keeps_changes(out: Path) -> None:
    c = _read(out, "code_status").sort_values("start_dttm")
    assert c["code_status_category"].tolist() == ["dnr_or_dni", "and"]


def test_io_and_support_devices(out: Path) -> None:
    o = _read(out, "output")
    assert o["output_category"].tolist() == ["urinary_other"]             # stool and counts skipped
    assert _read(out, "input")["input_category"].tolist() == ["blood_products"]
    assert _read(out, "crrt_therapy")["crrt_mode_category"].tolist() == ["cvvhd"]
    assert len(_read(out, "intermittent_dialysis")) == 0                  # catheter insertion is not dialysis
    h = _read(out, "invasive_hemodynamics")
    assert h[["measurement_category", "measurement_value"]].values.tolist() == [["cvp", 9.0]]


def test_medication_orders(out: Path) -> None:
    o = _read(out, "medication_orders").set_index("med_order_id")
    assert sorted(o.index) == ["1", "2", "5", "6"]          # topical / ophthalmic / unmapped orders dropped
    assert (o.loc["1", "med_category"], o.loc["1", "med_dose"], o.loc["1", "med_dose_unit"]) == (
        "piperacillin_tazobactam", 3.375, "g")
    assert o.loc["2", "med_category"] == "morphine" and o.loc["2", "prn"] == 1   # resolved through its HICL code
    assert pd.isna(o.loc["2", "order_end_dttm"])                                 # stop offset before start
    assert (o.loc["5", "med_category"], o.loc["5", "med_dose"], o.loc["5", "med_dose_unit"]) == (
        "heparin", 1000.0, "units/hr")
    assert (o.loc["6", "med_order_status_category"], o.loc["6", "med_dose_unit"]) == ("cancelled", "g")
    assert o.loc["1", "order_start_dttm"] == ADMIT_10 + timedelta(minutes=30)


def test_microbiology(out: Path) -> None:
    c = _read(out, "microbiology_culture").sort_values("collect_dttm")
    assert c[["fluid_category", "organism_category", "method_category"]].values.tolist() == [
        ["blood_buffy", "escherichia_coli", "culture"], ["genito_urinary_tract", "no_growth", "culture"]]
    assert c["collect_dttm"].tolist() == [ADMIT_10 + timedelta(minutes=100), ADMIT_10 + timedelta(minutes=200)]
    s = _read(out, "microbiology_susceptibility").sort_values("antimicrobial_category")
    assert s[["antimicrobial_category", "susceptibility_category"]].values.tolist() == [
        ["ciprofloxacin", "non_susceptible"], ["meropenem", "susceptible"]]
