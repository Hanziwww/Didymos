"""HiRID adapter on a hand-built fixture (HiRID column layout; raw parts typed as text like the mirror)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from icutwin.clif.adapters.hirid import HiridAdapter

T = lambda h, m=0: datetime(2150, 1, 1, h, m, tzinfo=UTC)  # noqa: E731


def _s(h: int, m: int = 0) -> str:
    return f"2150-01-01 {h:02d}:{m:02d}:00"


@pytest.fixture(scope="module")
def out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("hirid")
    src = d / "src"
    for sub in ("reference", "raw_stage/observation_tables/csv", "raw_stage/pharma_records/csv"):
        (src / sub).mkdir(parents=True)
    pd.DataFrame([{"patientid": 1, "admissiontime": pd.Timestamp("2150-01-01 08:00:00"), "sex": "F", "age": 60,
                   "discharge_status": "dead"}]).to_parquet(src / "reference/general_table.parquet")
    pd.DataFrame([
        ("Pharma", 1000657, "Noradrenalin 20 µg/ml Perfusor"), ("Pharma", 1000251, "Fentanyl inj 50 µg/ml"),
        ("Pharma", 251, "Dormicum inj 1 mg/ml"), ("Pharma", 1000100, "Intravenous blood transfusion of packed cells"),
        ("Pharma", 1000724, "Insulin Actrapid inj 100 U/ml"), ("Pharma", 208, "Disoprivan 2%"),
        ("Pharma", 1000727, "Nipruss 60 mg"),
        ("Observation", 200, "Heart rate"),
    ], columns=["Source Table", "ID", "Variable Name"]).assign(**{"Unit": None, "Additional information": None}).to_parquet(
        src / "reference/hirid_variable_reference.parquet")
    pd.DataFrame([(15001525, 8, "2 x 2"), (3845, 12, "NIV DGV"), (15001552, 2, "tracheotomiert")],
                 columns=["variableid", "code", "stringvalue"]).to_parquet(src / "reference/ordinal_vars_ref.parquet")
    obs = [  # (time, variableid, value, status)
        (_s(8, 5), 200, "80", "16"), (_s(20), 200, "70", "16"), (_s(9), 200, "999", "2"),      # invalidated
        (_s(9, 30), 200, "999", "40"),                                                          # notified, not measured
        (_s(8, 10), 4000, "97", "16"), (_s(9), 20000600, "88.42", "8"), (_s(9), 24000585, "5.55", "8"),
        (_s(9), 15001552, "2", "8"), (_s(10), 3845, "12", "8"), (_s(11), 15001525, "8", "8"),
        (_s(12), 15001565, "-1.5", "8"), (_s(21), 20000400, "140", "8"), (_s(9), 9990004, "192", "8"),
        (_s(9), 9990002, "97", "8"),                                     # APACHE II code not in the table: IV used
        (_s(10), 10020000, "60", "8"), (_s(12), 10020000, "30", "8"),   # urine rate (ml/h) at each measurement
        (_s(12), 10010020, "5", "8"),                                    # CSF drainage: a volume (ml)
        (_s(9), 30005075, "100", "8"), (_s(11), 30005075, "400", "8"),   # cumulative crystalloid volume
        (_s(13), 30005075, "50", "8"),                                   # counter reset: 50 ml since
    ]
    rows = [{"datetime": t, "entertime": t, "patientid": "1", "status": st, "stringvalue": None, "type": None,
             "value": v, "variableid": str(vid)} for t, vid, v, st in obs]
    rows.append({**rows[5], "entertime": _s(15)})   # the creatinine result re-sent later: one observation
    pd.DataFrame(rows).to_parquet(src / "raw_stage/observation_tables/csv/part-0.parquet")
    ph = [  # (pharmaid, infusionid, time, dose, unit, route, status, typeid, subtypeid)
        (1000657, 1, _s(8), "0", "µg", "cv-inf", "524", "1", "8.0"),
        (1000657, 1, _s(8, 30), "60", "µg", "cv-inf", "8", "1", "8.0"),
        (1000657, 1, _s(9), "60", "µg", "cv-inf", "520", "1", "8.0"),
        (1000657, 1, _s(9, 30), "120", "µg", "cv-inf", "8", "1", "8.0"),
        (1000657, 1, _s(10), "120", "µg", "cv-inf", "776", "1", "8.0"),
        (1000657, 2, _s(9, 30), "0", "µg", "cv-inf", "524", "1", "8.0"),     # second syringe, overlapping
        (1000657, 2, _s(10, 30), "30", "µg", "cv-inf", "776", "1", "8.0"),
        (1000724, 1, _s(8), "0", "U", "iv-inf", "524", "1", "8.0"),        # another drug under infusion id 1
        (1000724, 1, _s(9), "2", "U", "iv-inf", "776", "1", "8.0"),
        (1000251, 3, _s(12), "50", "µg", "iv-inj", "780", "1", "8.0"),       # bolus
        (251, 4, _s(12), "2", "mg", "iv-inj", "544", "1", "8.0"),            # notified, not administered
        (1000251, 5, _s(13), "50", "µg", "iv-inj", "782", "1", "8.0"),       # invalidated
        (1000100, 6, _s(14), "300", "ml", "iv-inf", "780", "0", "1.0"),      # packed red cells
        # no start record: starts one acting period (5 min) before its first record, so that dose is placed
        (208, 7, _s(14), "30", "mg", "iv-inf", "8", "1", "8.0"),
        (208, 7, _s(14, 30), "30", "mg", "iv-inf", "8", "1", "8.0"),
        # records after the last pump event: their doses count, and the infusion stops at the last record
        (251, 9, _s(16), "0", "mg", "iv-inf", "524", "1", "8.0"),
        (251, 9, _s(16, 30), "3", "mg", "iv-inf", "520", "1", "8.0"),
        (251, 9, _s(17), "3", "mg", "iv-inf", "8", "1", "8.0"),
        (251, 9, _s(17, 30), "3", "mg", "iv-inf", "8", "1", "8.0"),
        # a dose recorded after a stop record was given there; a second stop at the same time without dose is the
        # same record (the benchmark keeps the stop that carries the dose)
        (1000251, 10, _s(18), "0", "µg", "iv-inf", "524", "1", "8.0"),
        (1000251, 10, _s(18, 30), "30", "µg", "iv-inf", "776", "1", "8.0"),
        (1000251, 10, _s(18, 30), "0", "µg", "iv-inf", "776", "1", "8.0"),
        (1000251, 10, _s(19), "15", "µg", "iv-inf", "8", "1", "8.0"),
        # a small rate: 0.0005 mg over an hour
        (1000727, 11, _s(20), "0", "mg", "cv-inf", "524", "1", "8.0"),
        (1000727, 11, _s(21), "0.0005", "mg", "cv-inf", "776", "1", "8.0"),
    ]
    pd.DataFrame([{"patientid": "1", "pharmaid": str(p), "givenat": t, "enteredentryat": t, "givendose": dose,
                   "cumulativedose": dose, "fluidamount_calc": None, "cumulfluidamount_calc": None, "doseunit": u,
                   "route": r, "infusionid": str(i), "typeid": ty, "subtypeid": sub, "recordstatus": st}
                  for p, i, t, dose, u, r, st, ty, sub in ph]).to_parquet(src / "raw_stage/pharma_records/csv/part-0.parquet")
    o = d / "out"
    HiridAdapter(src, o, d / "duck").run()
    return o


def _read(out: Path, table: str) -> pd.DataFrame:
    return pd.read_parquet(out / f"clif_{table}.parquet")


def test_stay_window_death_and_binned_age(out: Path) -> None:
    s = _read(out, "ext_icu_stay").iloc[0]
    assert (s["icu_in_dttm"], s["icu_out_dttm"]) == (T(8), T(21))      # ricu: end = last observation
    assert s["demographics_binned"] and s["icu_outcome"] == "dead"
    assert (s["adm_dx_name"], s["adm_dx_group"]) == ("APACHE IV: Gastrointestinal", "medical_gastrointestinal")
    p = _read(out, "patient").iloc[0]
    assert p["death_dttm"] == T(20)                                      # ricu hirid_death: last HR/MAP
    assert _read(out, "hospitalization").iloc[0]["age_at_admission"] == 62


def test_invalid_observations_dropped_and_units_converted(out: Path) -> None:
    v = _read(out, "vitals")
    assert sorted(v.loc[v.vital_category == "heart_rate", "vital_value"]) == [70.0, 80.0]
    lab = _read(out, "labs").set_index("lab_category")
    assert lab.loc["creatinine", "lab_value_numeric"] == pytest.approx(1.0002, abs=1e-3)
    assert lab.loc["glucose_serum", "lab_value_numeric"] == pytest.approx(100.0, abs=0.1)
    assert _read(out, "patient_assessments").set_index("assessment_category").loc["rass", "numerical_value"] == -2.0


def test_re_entered_observation_counts_once_from_its_first_entry(out: Path) -> None:
    crea = _read(out, "labs").query("lab_category == 'creatinine'")
    assert len(crea) == 1 and crea.iloc[0]["lab_result_dttm"] == T(9)


def test_respiratory_codes(out: Path) -> None:
    r = _read(out, "respiratory_support").set_index("recorded_dttm")
    assert (r.loc[T(9), "device_category"], r.loc[T(9), "tracheostomy"]) == ("imv", 1)
    assert (r.loc[T(10), "device_category"], r.loc[T(10), "mode_category"]) == ("nippv", "other")
    assert (r.loc[T(11), "device_category"], r.loc[T(11), "lpm_set"]) == ("nasal_cannula", 4.0)


def test_infusion_rates_from_dose_records_summed_over_syringes(out: Path) -> None:
    m = _read(out, "medication_admin_continuous").sort_values("admin_dttm")
    steps = lambda d: [(t.hour * 60 + t.minute, a, round(dose, 3)) for t, a, dose in  # noqa: E731
                       zip(d["admin_dttm"], d["mar_action_category"], d["med_dose"])]
    ne = m[m["med_category"] == "norepinephrine"]
    assert steps(ne) == [(480, "start", 2.0), (540, "dose_change", 4.0), (570, "dose_change", 4.5),
                         (600, "dose_change", 0.5), (630, "stop", 0.0)]
    assert set(ne["med_dose_unit"]) == {"mcg/min"}
    # an infusion is a (pharmaid, infusionid) pair: insulin under the same infusion id is its own stream
    ins = m[m["med_category"] == "insulin"]
    assert steps(ins) == [(480, "start", 0.033), (540, "stop", 0.0)] and set(ins["med_dose_unit"]) == {"u/min"}
    assert set(m["med_category"]) == {"norepinephrine", "insulin", "propofol", "midazolam", "fentanyl", "nitroprusside"}


def test_every_infusion_dose_gets_an_interval(out: Path) -> None:
    """As the HiRID benchmark: an infusion without a start record starts one acting period before its first
    record; records after the last pump event and doses recorded after a stop are given; the infusion stops at
    its last record. The stream integrates to every dose recorded."""
    m = _read(out, "medication_admin_continuous").sort_values("admin_dttm")
    steps = lambda d: [(t.hour * 60 + t.minute, a, round(dose, 3)) for t, a, dose in  # noqa: E731
                       zip(d["admin_dttm"], d["mar_action_category"], d["med_dose"])]
    assert steps(m[m["med_category"] == "propofol"]) == [(835, "start", round(60 / 35, 3)), (870, "stop", 0.0)]
    mid = m[m["med_category"] == "midazolam"]
    assert steps(mid) == [(960, "start", 0.1), (1050, "stop", 0.0)]
    assert mid["mar_action_name"].tolist()[-1] == "inferred stop: end of infusion records"
    assert steps(m[m["med_category"] == "fentanyl"]) == [(1080, "start", 1.0), (1110, "dose_change", 0.5),
                                                         (1140, "stop", 0.0)]


def test_small_rates_keep_their_precision(out: Path) -> None:
    # 8.3e-6 mg/min (the running total rounded to 6 decimals gave 8e-6, #98); the stop is exactly 0
    m = _read(out, "medication_admin_continuous")
    n = m[m["med_category"] == "nitroprusside"].sort_values("admin_dttm")
    assert n["mar_action_category"].tolist() == ["start", "stop"]
    assert n["med_dose"].tolist()[0] == pytest.approx(0.0005 / 60, rel=1e-9) and n["med_dose"].tolist()[1] == 0.0


def test_boluses_status_bits_and_blood_products(out: Path) -> None:
    i = _read(out, "medication_admin_intermittent")
    assert i[["med_category", "med_dose", "med_dose_unit", "med_route_category"]].values.tolist() == [
        ["fentanyl", 50.0, "mcg", "iv"]]                                 # 544 (not administered), 782 (invalid) dropped
    inp = _read(out, "input").query("input_category == 'blood_products'")
    assert inp[["input_category", "input_volume"]].values.tolist() == [["blood_products", 300.0]]


def test_urine_rate_becomes_volume(out: Path) -> None:
    o = _read(out, "output")
    assert sorted(zip([t.hour for t in o["recorded_dttm"]], o["output_group"], o["output_volume"])) == [
        (10, "urine", 60.0),     # first report: one hour
        (12, "drains", 5.0),
        (12, "urine", 60.0)]     # 30 ml/h over the 2 h since the previous report


def test_cumulative_crystalloids_become_volumes(out: Path) -> None:
    i = _read(out, "input").query("input_category == 'iv_crystalloid'").sort_values("recorded_dttm")
    assert list(zip([t.hour for t in i["recorded_dttm"]], i["input_volume"])) == [(9, 100.0), (11, 300.0)]
