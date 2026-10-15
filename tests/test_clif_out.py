"""Stays -> canonical CLIF (``generate.clif_out``): codes invert their ELF templates; a stay's tables (wide rows merged
per time, long rows per event, the GCS total, group columns, the lab time, the ICU window, a cut twin); and, on the
2% subset (skipped without it), the derived concepts of MEDS stays written this way against those of the canonical
CLIF."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from icutwin.generate import clif_out as co
from icutwin.tokenization.stays import Stay

R = Path(__file__).resolve().parents[1] / "data"
T0 = datetime(2150, 3, 1, 8, 0)

pytestmark = pytest.mark.skipif(not (co.refs_dir() / "CLIF_MEDS" / "config").exists(), reason="refs/CLIF_MEDS")


def test_codes_invert_their_templates() -> None:
    def cols(code):
        tpl, vals = co.match(code)
        return tpl.table, dict(vals)

    assert cols("VITAL//heart_rate") == ("vitals", {"vital_category": "heart_rate"})
    assert cols("VITAL//sbp//arterial") == ("vitals", {"vital_category": "sbp", "meas_site_name": "arterial"})
    assert cols("MED_CON//norepinephrine//mcg/kg/min//start//iv") == (
        "medication_admin_continuous", {"med_category": "norepinephrine", "med_dose_unit": "mcg/kg/min",
                                        "mar_action_category": "start", "med_route_category": "iv"})
    assert cols("MED_INT//vancomycin//dose//given//UNK")[1]["med_route_category"] is None   # 'UNK' is NULL
    assert cols("RESP//device_category//imv") == ("respiratory_support", {"device_category": "imv"})
    assert cols("RESP//fio2_set") == ("respiratory_support", {})
    assert co.match("RESP//fio2_set")[0].numeric == "fio2_set" and co.match("RESP//fio2_set")[0].wide
    assert cols("LAB//creatinine//mg/dl//bmp") == ("labs", {"lab_category": "creatinine", "reference_unit": "mg/dl",
                                                             "lab_order_category": "bmp"})
    assert co.match("LAB//creatinine//mg/dl//bmp")[0].time == "lab_result_dttm"
    assert cols("MICRO_ORDER//culture//blood_buffy") == (
        "microbiology_culture", {"method_category": "culture", "fluid_category": "blood_buffy"})
    assert co.match("NOPE//x") is None and co.match("VITAL//heart_rate//a//b") is None


def _stay(discharge: bool = True) -> Stay:
    h = lambda x: T0 + timedelta(hours=x)  # noqa: E731
    tl = [(h(-1), 0, "VITAL//heart_rate", 90.0, None), (h(-1), 0, "VITAL//weight_kg", 80.0, None),
          (h(0), 0, co.MARKER, None, None),
          (h(1), 0, "RESP//device_category//imv", None, None), (h(1), 0, "RESP//fio2_set", 0.5, None),
          (h(1), 0, "RESP//peep_set", 8.0, None), (h(1), 0, "VITAL//sbp//arterial", 120.0, None),
          (h(1), 0, "VITAL//map//arterial", 80.0, None),
          (h(2), 0, "PA//gcs_eye", 3.0, None), (h(2), 0, "PA//gcs_motor", 5.0, None),
          (h(2), 0, "PA//gcs_verbal", 1.0, None),
          (h(3), 0, "LAB//creatinine//mg/dl//bmp", 1.2, None),
          (h(4), 0, "OUTPUT//indwelling_urinary_catheter", 100.0, None),
          (h(5), 0, "MED_CON//norepinephrine//mcg/kg/min//start//iv", 0.1, None),
          (h(6), 0, "RESP//fio2_set", 0.4, None), (h(6), 0, "RESP//fio2_set", 0.6, None)]   # a second row at 6 h
    if discharge:
        tl.append((h(30), 0, "ICU//discharge//alive", None, None))
    s = Stay(1, "s1", "mimic", "held_out", h(0), h(30) if discharge else h(6), True,
             prefix={"dataset": "mimic", "sex": "female", "age": 67.4, "site": "mimic"},
             header={"unit_type": "medical_icu", "admission_type": "UNK", "pre_icu_los": 10.0}, timeline=tl,
             end=["HOSP//discharge_category//home"])
    return s


def _read(d: Path, table: str) -> pl.DataFrame:
    return pl.read_parquet(d / f"clif_{table}.parquet")


def test_a_stay_becomes_canonical_tables(tmp_path: Path) -> None:
    rep = co.write([_stay(), _stay(discharge=False)], ["a", "b"], tmp_path)
    assert rep["cut"] == 1 and rep["unmatched"] == {}
    icu = _read(tmp_path, "ext_icu_stay").sort("stay_id")
    assert icu["icu_in_dttm"].dt.replace_time_zone(None).to_list() == [T0, T0]
    assert icu["icu_out_dttm"].dt.replace_time_zone(None).to_list() == [T0 + timedelta(hours=30),
                                                                        T0 + timedelta(hours=6)]
    assert icu["icu_outcome"].to_list() == ["alive", None] and icu["unit_type"].to_list() == ["medical_icu"] * 2
    hosp = _read(tmp_path, "hospitalization").filter(pl.col("hospitalization_id") == "a")
    assert hosp["age_at_admission"][0] == 67 and hosp["admission_type_category"][0] is None    # 'UNK' -> NULL
    assert hosp["admission_dttm"].dt.replace_time_zone(None)[0] == T0 - timedelta(hours=10)
    assert hosp["discharge_category"][0] == "home"
    a = lambda t: _read(tmp_path, t).filter(pl.col("hospitalization_id") == "a")  # noqa: E731
    rs = a("respiratory_support").sort("recorded_dttm", "fio2_set")
    assert rs.height == 3                                           # the 1-h row merged; two rows at 6 h
    first = rs.row(0, named=True)
    assert (first["device_category"], first["fio2_set"], first["peep_set"]) == ("imv", 0.5, 8.0)
    assert rs["fio2_set"].to_list()[1:] == [0.4, 0.6]
    v = a("vitals")
    assert v.height == 4 and set(v["meas_site_name"].drop_nulls()) == {"arterial"}
    gcs = a("patient_assessments").filter(pl.col("assessment_category") == "gcs_total")
    assert gcs["numerical_value"].to_list() == [9.0]
    lab = a("labs")
    assert (lab["lab_collect_dttm"] == lab["lab_result_dttm"]).all() and lab["lab_value_numeric"][0] == 1.2
    assert a("output")["output_group"].to_list() == ["urine"]
    med = a("medication_admin_continuous").sort("admin_dttm")
    assert med.select("med_category", "med_dose", "med_dose_unit", "mar_action_category", "med_route_category",
                      "med_group").row(0) == ("norepinephrine", 0.1, "mcg/kg/min", "start", "iv", "vasoactives")
    assert med.height == 1                       # no stop is invented for an infusion the record leaves open


def test_out_of_vocabulary_events_are_dropped(tmp_path: Path) -> None:
    s = _stay()
    s.timeline.insert(-1, (T0 + timedelta(hours=7), 0, "OUTPUT//OOV", 50.0, None))     # #120
    rep = co.write([s], ["a"], tmp_path)
    assert rep["unmatched"] == {"OOV": 1} and _read(tmp_path, "output").height == 1


def test_derived_concepts_survive_the_meds_path(tmp_path: Path) -> None:
    """MEDS stays -> CLIF -> derive against the canonical CLIF -> derive (AmsterdamUMCdb subset): episodes, the
    24-hour SOFA and the ICU's highest KDIGO stage agree (the residue: labs at result time, the MEDS window)."""
    meds, canon = R / "meds" / "subset" / "aumc", R / "clif_canonical" / "subset" / "aumc"
    if not (meds / "metadata" / "stays.parquet").exists() or not canon.exists():
        pytest.skip("the subset's MEDS and canonical CLIF")
    import importlib.util

    tool = Path(__file__).resolve().parents[1] / "tools" / "clif_roundtrip.py"
    spec = importlib.util.spec_from_file_location("rt", tool)
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    from icutwin.tokenization.stays import iter_stays

    stays = [s for split in ("train", "tuning", "held_out") for s in iter_stays(meds, split)]
    keys = [s.stay_id for s in stays]
    co.write(stays, keys, tmp_path / "meds" / "clif")
    a = rt.concepts(rt.run_derive(canon, tmp_path / "canonical"), rt.icu_windows(canon, set(keys)))
    b = rt.concepts(rt.run_derive(tmp_path / "meds" / "clif", tmp_path / "meds"),
                    rt.icu_windows(tmp_path / "meds" / "clif", set(keys)))
    rep = rt.compare(a, b)
    assert rep["stays"] == len(stays) > 300
    assert all(rep[e]["agreement"] >= 0.99 for e in ("imv", "vasopressor", "rrt"))
    assert rep["sofa_total"]["exact"] >= 0.95 and rep["kdigo_max"]["exact"] >= 0.95
