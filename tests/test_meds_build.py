"""S4 MEDS build: stay windowing, outcome events, splits and MEDS v0.4 metadata on a hand-built stay layout."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import meds
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from icutwin.meds.build import build_dataset, patient_split, stay_subject_id

D = lambda day, h=0: datetime(2100, 1, day, h)  # noqa: E731


def _events(path: Path, rows: list[tuple], hosp: bool = True) -> None:
    cols = {"subject_id": pa.array([r[0] for r in rows], pa.int64()),
            "time": pa.array([r[1] for r in rows], pa.timestamp("us")),
            "code": pa.array([r[2] for r in rows], pa.string()),
            "numeric_value": pa.array([r[3] for r in rows], pa.float32()),
            "text_value": pa.array([None] * len(rows), pa.string())}
    if hosp:
        cols["hospitalization_id"] = pa.array([r[4] for r in rows], pa.int64())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), path)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("meds_build")
    clif, elf = root / "clif", root / "elf"
    clif.mkdir()
    utc = lambda d: d.replace(tzinfo=UTC)  # noqa: E731
    stay = lambda sid, a, b, seq: {  # noqa: E731
        "stay_id": f"demo:{sid}", "patient_id": "7", "hospitalization_id": "70", "dataset": "eicu", "site_id": "73",
        "icu_in_dttm": utc(a), "icu_out_dttm": utc(b), "unit_type": "general_icu", "clock_valid": True,
        "demographics_binned": False, "admission_source": "ed", "surgical_status": "nonsurgical", "stay_seq": seq,
        "icu_outcome": "alive", "surgery_context": None, "adm_dx_name": None, "adm_dx_group": None,
        "severity_name": None, "severity_value": None, "era": "2014"}
    # two ICU stays in one hospitalization (01-01 .. 01-10): a readmission
    pd.DataFrame([stay("s1", D(4), D(5), 1), stay("s2", D(7), D(8), 2),
                  stay("s3", D(8), D(8), 3),   # zero-length unit stay at s2's discharge: not a subject
                  {**stay("s4", D(4), D(6), 1), "patient_id": "9", "hospitalization_id": "90"}]   # a minor: not either
                 ).to_parquet(clif / "clif_ext_icu_stay.parquet")
    pd.DataFrame({"patient_id": ["7", "7", "9"], "hospitalization_id": ["70", "71", "90"],
                  "admission_dttm": [utc(D(1))] * 3, "discharge_dttm": [utc(D(10))] * 3,
                  "age_at_admission": [40, 40, 16]}).to_parquet(clif / "clif_hospitalization.parquet")
    _events(elf / "data" / "PATIENT.parquet", [(7, None, "PATIENT//sex//female", None), (7, D(28), "MEDS_DEATH", None)],
            hosp=False)
    _events(elf / "data" / "HOSP.parquet", [(7, D(1), "HOSP//admission_type//ed", None, 70),
                                            (7, D(10), "HOSP//discharge_category//home", None, 70)])
    # VITAL is a qualified domain: the build runs our engine on canonical vitals with the run's concept config
    vit = lambda t, v, h="70", cat="heart_rate", site=None: {  # noqa: E731
        "hospitalization_id": h, "recorded_dttm": utc(t), "vital_category": cat, "vital_value": v, "meas_site_name": site}
    pd.DataFrame([
        vit(D(1, 12), 70.0),   # 60 h before s1: outside both windows
        vit(D(2, 6), 71.0),    # s1 pre-ICU window
        vit(D(4, 12), 72.0),   # s1
        vit(D(4, 12), 72.0),   # an identical event: kept once
        vit(D(4, 12), 118.0, cat="sbp", site="Arterial"),   # s1, qualified by its site
        vit(D(5, 12), 73.0),   # after s1 discharge, inside s2's 48 h pre-ICU window
        vit(D(7, 12), 74.0),   # s2
        vit(D(9), 75.0),       # after s2: ward
        vit(D(4, 12), 99.0, h="71"),   # another hospitalization: never
    ]).to_parquet(clif / "clif_vitals.parquet")
    _events(elf / "data" / "VITAL.parquet", [(7, D(4, 12), "VITAL//heart_rate", 72.0, 70)])   # replaced by the build
    (elf / "_concept_config").mkdir(parents=True)
    (elf / "_concept_config" / "VITAL.yaml").write_text(   # as CLIF_MEDS's config (outlier shaping disabled)
        "elf_version: '1.0.0'\nsubject_id_col: patient_id\nvitals:\n  vital:\n    code: [VITAL, col(vital_category)]\n"
        "    time: col(recorded_dttm)\n    numeric_value: vital_value\n    hospitalization_id: hospitalization_id\n",
        encoding="utf-8")
    _events(elf / "data" / "ECMO_MCS.parquet", [(7, D(4, 1), "ECMO_MCS//flow", 3.0, 70)])   # 2.1 domain: skipped
    # two 10 mL flushes through two lines at one time: two administrations (amounts are not deduplicated)
    _events(elf / "data" / "INPUT.parquet", [(7, D(4, 13), "INPUT//flush_irrigation_other", 10.0, 70)] * 2)
    out = root / "meds"
    build_dataset("demo", clif, elf, out, pre_icu_hours=48)
    return out


def _subject(out: Path, stay: str) -> pd.DataFrame:
    df = pd.concat(pd.read_parquet(p) for p in sorted((out / "data").rglob("*.parquet")))
    return df[df.subject_id == stay_subject_id(f"demo:{stay}")]


def test_stay_windows(built: Path) -> None:
    hr = lambda s: _subject(built, s).query("code == 'VITAL//heart_rate'")["numeric_value"].tolist()  # noqa: E731
    assert hr("s1") == [71.0, 72.0]
    assert hr("s2") == [73.0, 74.0]
    report = json.loads((built / "metadata" / "_build_report.json").read_text(encoding="utf-8"))
    assert report["identical_events_dropped"] == 1
    flushes = _subject(built, "s1").query("code == 'INPUT//flush_irrigation_other'")
    assert flushes["numeric_value"].tolist() == [10.0, 10.0]


def test_header_outcome_and_patient_events_in_every_stay(built: Path) -> None:
    for s in ("s1", "s2"):
        codes = _subject(built, s)["code"].tolist()
        assert codes.count("ICU//admission//general_icu") == codes.count("ICU//discharge//alive") == 1
        assert codes[0] == "PATIENT//sex//female"                       # static events first
        assert {"HOSP//admission_type//ed", "HOSP//discharge_category//home", "MEDS_DEATH", "DATASET//eicu",
                "SITE//eicu//73", "ICU//admission//general_icu", "ICU//discharge//alive", "ICU//era//2014"} <= set(codes)
        assert not any(c.startswith("ECMO_MCS") for c in codes)


def test_events_are_time_ordered_and_meds_typed(built: Path) -> None:
    for p in sorted((built / "data").rglob("*.parquet")):
        t = pq.read_table(p)
        meds.DataSchema.validate(t)
        df = t.to_pandas()
        for _, g in df.groupby("subject_id", sort=False):
            times = g["time"].dropna()
            assert times.is_monotonic_increasing and g["time"].isna().cumprod().sum() == g["time"].isna().sum()


def test_patient_grouped_splits_and_metadata(built: Path) -> None:
    splits = pq.read_table(built / "metadata" / "subject_splits.parquet").to_pandas()
    assert len(splits) == 2 and splits["split"].nunique() == 1 and splits["split"].iloc[0] == patient_split("eicu", "7")
    codes = pq.read_table(built / "metadata" / "codes.parquet")
    meds.CodeMetadataSchema.validate(codes)
    all_codes = set(pd.concat(pd.read_parquet(p) for p in (built / "data").rglob("*.parquet"))["code"])
    assert set(codes.column("code").to_pylist()) == all_codes
    ds = json.loads((built / "metadata" / "dataset.json").read_text())
    assert ds["dataset_name"] == "icutwin-demo" and ds["meds_version"]
    stays = pq.read_table(built / "metadata" / "stays.parquet").to_pandas()
    assert sorted(stays["stay_id"]) == ["demo:s1", "demo:s2"]
    report = json.loads((built / "metadata" / "_build_report.json").read_text())
    assert report["stays_without_icu_time"] == 1
    assert report["stays_below_min_age"] == 1 and report["stays_age_unknown"] == 0 and report["min_age_years"] == 18
    assert pd.concat(pd.read_parquet(p) for p in (built / "data").rglob("*.parquet"))["text_value"].isna().all()


def test_qualified_codes_extend_their_elf_code(built: Path) -> None:
    """Blood pressure carries its measurement site as a qualifier level; codes.parquet gives the ELF parent."""
    s1 = _subject(built, "s1")
    assert "VITAL//sbp//arterial" in set(s1["code"]) and "VITAL//heart_rate" in set(s1["code"])
    codes = pq.read_table(built / "metadata" / "codes.parquet").to_pandas().set_index("code")
    assert list(codes.loc["VITAL//sbp//arterial", "parent_codes"]) == ["VITAL//sbp"]
    assert codes.loc["VITAL//heart_rate", "parent_codes"] is None


def test_parent_code_levels() -> None:
    from icutwin.meds.build import parent_code
    levels = {"MED_INT": 4, "VITAL": 2, "PA": 2}
    assert parent_code("MED_INT//ext_metoprolol//mg//given//enteral", levels) == "MED_INT//ext_metoprolol//mg//given"
    assert parent_code("MED_INT//ext_metoprolol//mg//given", levels) is None
    assert parent_code("PA//cam_total//positive", levels) == "PA//cam_total"
    assert parent_code("LAB//sodium//mmol/l//bmp", levels) is None


def test_code_descriptions_mark_vocabulary_extensions() -> None:
    from icutwin.meds.build import _description
    assert _description("MED_INT//esomeprazole//mg//iv").endswith("[not CLIF 3.0 vocabulary: local extension]")
    assert "consortium practice" in _description("MED_INT//pantoprazole//mg//iv")
    assert "not CLIF 3.0" not in _description("MED_INT//cefazolin//g//iv")
    assert "not CLIF 3.0" not in _description("MED_CON//esomeprazole//mg/hr//start")   # a CLIF 3.0 infusion category
    # M4 registry: the same term is standard in one table and a local extension in the other
    assert _description("MED_CON//potassium_repletion//mmol/min//start").endswith(
        "[not CLIF 3.0 vocabulary: local extension]")
    assert "not CLIF 3.0" not in _description("MED_INT//potassium_repletion//mmol//given")
    assert _description("MED_INT//ext_metoprolol//mg//given").endswith(
        "[not CLIF 3.0 vocabulary: local concept extension]")
    assert _description("MED_ORD//ext_metoprolol//mg//start").endswith(
        "[not CLIF 3.0 vocabulary: local concept extension]")
    assert "not CLIF 3.0" not in _description("MED_ORD//potassium_repletion//meq//start")   # in the orders list
    assert _description("MED_CON//alteplase//mg/hr//start").endswith("[not CLIF 3.0 vocabulary: local extension]")
    # the route level: 'inhaled' is a declared intermittent route extension, a CLIF 3.0 continuous route
    assert _description("MED_INT//albuterol//mg//given//inhaled").endswith(
        "[route 'inhaled' not CLIF 3.0 vocabulary of the table: local extension]")
    assert "route" not in _description("MED_CON//albuterol//mg/hr//start//inhaled")
    assert "route" not in _description("MED_INT//cefazolin//g//given//iv")


def test_adult_cohort_rule(tmp_path: Path) -> None:
    """Stays of patients younger than min_age_years at hospital admission leave the cohort (MEDS and derived);
    an unknown age stays; None keeps every age."""
    import duckdb

    from icutwin.meds.build import drop_minors
    pd.DataFrame({"hospitalization_id": ["H1", "H2", "0003"], "age_at_admission": [40.0, 17.0, None]}
                 ).to_parquet(tmp_path / "clif_hospitalization.parquet")
    for min_age, left in ((18.0, ["H1", "H4", "3"]), (None, ["H1", "H2", "H4", "3"])):
        con = duckdb.connect()
        con.execute("CREATE TEMP TABLE stays AS SELECT * FROM (VALUES ('H1'), ('H2'), ('H4'), ('3')) t(hospitalization_id)")
        drop_minors(con, tmp_path, min_age)
        assert sorted(r[0] for r in con.execute("SELECT * FROM stays").fetchall()) == sorted(left)
    assert drop_minors(duckdb.connect(), tmp_path / "none", 18.0)["stays_below_min_age"] == 0   # no table: kept
    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE stays AS SELECT * FROM (VALUES ('H1'), ('H2'), ('H4'), ('3')) t(hospitalization_id)")
    assert drop_minors(con, tmp_path, 18.0) == {"min_age_years": 18.0, "stays_below_min_age": 1, "stays_age_unknown": 2}


def test_subject_ids_are_stable_hashes() -> None:
    assert stay_subject_id("mimic:30000123") == stay_subject_id("mimic:30000123") > 0
    assert stay_subject_id("mimic:30000123") != stay_subject_id("eicu:30000123")


def test_infusions_running_at_the_window_start_and_route_basis(tmp_path: Path) -> None:
    """An infusion started before the stay's window is a 'going' event at the window start with the rate then in
    effect (from the events before the window, never from a later one); one stopped before is not. Medication
    events carry the route's basis as the extra column route_basis."""
    clif, elf = tmp_path / "clif", tmp_path / "elf"
    clif.mkdir()
    utc = lambda d: d.replace(tzinfo=UTC)  # noqa: E731
    pd.DataFrame([{"stay_id": "demo:s1", "patient_id": "7", "hospitalization_id": "70", "dataset": "eicu",
                   "site_id": "73", "icu_in_dttm": utc(D(4)), "icu_out_dttm": utc(D(5)), "unit_type": "general_icu",
                   "clock_valid": True, "demographics_binned": False, "admission_source": "ed",
                   "surgical_status": "nonsurgical", "stay_seq": 1, "icu_outcome": "alive", "era": None,
                   "surgery_context": None, "adm_dx_name": None, "adm_dx_group": None, "severity_name": None,
                   "severity_value": None}]
                 ).to_parquet(clif / "clif_ext_icu_stay.parquet")
    pd.DataFrame({"patient_id": ["7"], "hospitalization_id": ["70"], "admission_dttm": [utc(D(1))],
                  "discharge_dttm": [utc(D(10))], "age_at_admission": [60]}).to_parquet(clif / "clif_hospitalization.parquet")
    med = lambda t, cat, dose, action: {"hospitalization_id": "70", "admin_dttm": utc(t), "med_category": cat,  # noqa: E731
                                        "med_dose": dose, "med_dose_unit": "mcg/min", "mar_action_category": action,
                                        "med_route_category": "iv", "med_route_basis": "reported"}
    pd.DataFrame([med(D(1, 12), "norepinephrine", 8.0, "start"),        # before the window (it opens on day 2)
                  med(D(4, 13), "norepinephrine", 12.0, "dose_change"), med(D(4, 20), "norepinephrine", 0.0, "stop"),
                  med(D(1, 0), "propofol", 50.0, "start"), med(D(1, 6), "propofol", 0.0, "stop")]
                 ).to_parquet(clif / "clif_medication_admin_continuous.parquet")
    (elf / "_concept_config").mkdir(parents=True)
    (elf / "_concept_config" / "MED_CON.yaml").write_text(
        "elf_version: '1.0.0'\nsubject_id_col: patient_id\nmedication_admin_continuous:\n  medication:\n"
        "    code: [MED_CON, col(med_category), col(med_dose_unit), col(mar_action_category)]\n"
        "    time: col(admin_dttm)\n    numeric_value: med_dose\n    hospitalization_id: hospitalization_id\n",
        encoding="utf-8")
    _events(elf / "data" / "MED_CON.parquet", [(7, D(4), "MED_CON//x//y//start", 1.0, 70)])   # replaced by the build
    out = tmp_path / "meds"
    report = build_dataset("demo", clif, elf, out, pre_icu_hours=48)
    df = pd.concat(pd.read_parquet(p) for p in sorted((out / "data").rglob("*.parquet")))
    med_events = df[df["code"].str.startswith("MED_CON")]
    assert [(t.day, t.hour, c, v) for t, c, v in zip(med_events["time"], med_events["code"], med_events["numeric_value"])] == [
        (2, 0, "MED_CON//norepinephrine//mcg/min//going//iv", 8.0),
        (4, 13, "MED_CON//norepinephrine//mcg/min//dose_change//iv", 12.0),
        (4, 20, "MED_CON//norepinephrine//mcg/min//stop//iv", 0.0)]
    assert set(med_events["route_basis"]) == {"reported"} and df.loc[~df["code"].str.startswith("MED"), "route_basis"].isna().all()
    assert report["infusions_running_at_window_start"] == 1
