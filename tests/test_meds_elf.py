"""ELF engine (CLIF_MEDS config semantics as SQL) on hand-built canonical tables."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from icutwin.clif.conform import connect
from icutwin.meds.elf import convert_domain

T = lambda h: datetime(2100, 1, 1, h, tzinfo=UTC)  # noqa: E731


@pytest.fixture()
def clif(tmp_path: Path) -> Path:
    d = tmp_path / "clif"
    d.mkdir()
    pd.DataFrame({"patient_id": ["7", "8"], "hospitalization_id": ["70", "80"], "admission_dttm": [T(1), T(2)],
                  "discharge_dttm": [T(20), T(21)]}).to_parquet(d / "clif_hospitalization.parquet")
    pd.DataFrame({"patient_id": ["7", "8"], "sex_category": ["Female", None], "sex_name": ["F", None],
                  "birth_date": [date(2050, 6, 1), None]}).to_parquet(d / "clif_patient.parquet")
    pd.DataFrame({"hospitalization_id": ["70", "70", "80"], "recorded_dttm": [T(3), None, T(4)],
                  "vital_category": [" Heart Rate", "spo2", "sbp"], "vital_value": [80.0, 97.0, None],
                  "meas_site_name": [None, None, "cuff"]}).to_parquet(d / "clif_vitals.parquet")
    pd.DataFrame({"hospitalization_id": ["70"], "diagnosis_code": ["I10"], "diagnosis_code_format": ["ICD10CM"]}
                 ).to_parquet(d / "clif_hospital_diagnosis.parquet")
    return d


def _run(clif: Path, tmp: Path, config: dict, domain: str) -> pd.DataFrame:
    n = convert_domain(connect(), clif, config, domain, tmp / f"{domain}.parquet")
    df = pd.read_parquet(tmp / f"{domain}.parquet")
    assert n == len(df)
    return df


def test_codes_times_values_and_subject_join(clif: Path, tmp_path: Path) -> None:
    cfg = {"elf_version": "1.0.0", "vitals": {"vital": {
        "code": ["VITAL", "col(vital_category)", {"col(meas_site_name)": {"cuff": "nibp"}}],
        "time": "col(recorded_dttm)", "numeric_value": "vital_value", "text_value": "meas_site_name"}}}
    df = _run(clif, tmp_path, cfg, "VITAL").sort_values("time")
    # categories normalised; NULL parts -> UNK; mapped dict part; row with NULL time dropped; subject from
    # hospitalization; times naive wall-clock; an event is kept even without a value
    assert df["code"].tolist() == ["VITAL//heart_rate//UNK", "VITAL//sbp//nibp"]
    assert df["subject_id"].tolist() == [7, 8] and df["hospitalization_id"].tolist() == [70, 80]
    assert df["time"].tolist() == [pd.Timestamp(2100, 1, 1, 3), pd.Timestamp(2100, 1, 1, 4)]
    assert df["numeric_value"].iloc[0] == 80.0 and pd.isna(df["numeric_value"].iloc[1])
    assert str(df["numeric_value"].dtype) == "float32"


def test_static_events_and_patient_domain(clif: Path, tmp_path: Path) -> None:
    cfg = {"patient": {"sex": {"code": ["PATIENT", "sex", "col(sex_category)"], "time": None, "text_value": "sex_name"},
                       "birth": {"code": "MEDS_BIRTH", "time": "col(birth_date)"}}}
    df = _run(clif, tmp_path, cfg, "PATIENT")
    assert "hospitalization_id" not in df.columns
    got = sorted((r.subject_id, r.code, str(r.time), r.text_value) for r in df.itertuples())
    assert got == [(7, "MEDS_BIRTH", "2050-06-01 00:00:00", None), (7, "PATIENT//sex//female", "NaT", "F"),
                   (8, "PATIENT//sex//UNK", "NaT", None)]


def test_time_column_from_hospitalization(clif: Path, tmp_path: Path) -> None:
    cfg = {"hospital_diagnosis": {"diagnosis": {
        "code": ["HOSP_DX", "col(diagnosis_code_format)", "col(diagnosis_code)"], "time": "col(discharge_dttm)"}}}
    df = _run(clif, tmp_path, cfg, "HOSP_DX")
    assert df[["subject_id", "code"]].values.tolist() == [[7, "HOSP_DX//ICD10CM//I10"]]   # *_format is not normalised
    assert df["time"].tolist() == [pd.Timestamp(2100, 1, 1, 20)]


def test_extensions_skip_empty_and_where(clif: Path, tmp_path: Path) -> None:
    cfg = {"vitals": {
        "valued": {"code": "X//valued", "time": "col(recorded_dttm)", "numeric_value": "vital_value", "skip_empty": True},
        "cuff": {"code": "X//cuff", "time": "col(recorded_dttm)", "where": "meas_site_name = 'cuff'"}}}
    df = _run(clif, tmp_path, cfg, "VITAL")
    assert sorted(df["code"]) == ["X//cuff", "X//valued"]


def test_clamping_configs_are_refused(clif: Path, tmp_path: Path) -> None:
    cfg = {"outlier_shaping": {"enabled": True}, "vitals": {"v": {"code": "VITAL//x", "time": "col(recorded_dttm)"}}}
    with pytest.raises(ValueError, match="clamps"):
        _run(clif, tmp_path, cfg, "VITAL")


def test_join_and_distinct_extensions(clif: Path, tmp_path: Path) -> None:
    pd.DataFrame({"patient_id": ["7", "7"], "hospitalization_id": ["70", "70"], "order_id": ["o1", "o1"],
                  "organism_id": ["g1", "g2"], "collect_dttm": [T(5), T(5)], "result_dttm": [T(9), T(10)],
                  "method_category": ["culture", "culture"], "fluid_category": ["blood_buffy", "blood_buffy"],
                  "organism_category": ["escherichia_coli", "enterococcus_faecalis"]}
                 ).to_parquet(clif / "clif_microbiology_culture.parquet")
    pd.DataFrame({"organism_id": ["g1", "g2"], "antimicrobial_category": ["ceftriaxone", "ampicillin"],
                  "susceptibility_category": ["susceptible", "non_susceptible"]}
                 ).to_parquet(clif / "clif_microbiology_susceptibility.parquet")
    collected = {"microbiology_culture": {"collected": {
        "code": ["MICRO", "col(method_category)", "col(fluid_category)", "collected"], "time": "col(collect_dttm)",
        "distinct": True}}}
    df = _run(clif, tmp_path, collected, "MICRO")
    assert df[["subject_id", "code"]].values.tolist() == [[7, "MICRO//culture//blood_buffy//collected"]]  # one order
    sus = {"join": {"table": "microbiology_culture", "key": "organism_id"}, "microbiology_susceptibility": {"ast": {
        "code": ["MICRO_SUS", "col(organism_category)", "col(antimicrobial_category)", "col(susceptibility_category)"],
        "time": "col(result_dttm)"}}}
    df = _run(clif, tmp_path, sus, "MICRO_SUS").sort_values("time")
    assert df["code"].tolist() == ["MICRO_SUS//escherichia_coli//ceftriaxone//susceptible",
                                   "MICRO_SUS//enterococcus_faecalis//ampicillin//non_susceptible"]
    assert df["hospitalization_id"].tolist() == [70, 70] and df["subject_id"].tolist() == [7, 7]


MED_CON = {"elf_version": "1.0.0", "subject_id_col": "patient_id", "medication_admin_continuous": {"medication": {
    "code": ["MED_CON", "col(med_category)", "col(med_dose_unit_converted)", "col(mar_action_category)"],
    "time": "col(admin_dttm)", "numeric_value": "med_dose_converted", "hospitalization_id": "hospitalization_id",
    "unit_conversion": {"enabled": True, "override": True, "allow_other_meds": True,
                        "preferred_units": {"norepinephrine": "mcg/kg/min"}}}}}


def test_medication_doses_convert_as_clif_meds_does(clif: Path, tmp_path: Path) -> None:
    """CLIF_MEDS converts doses with clifpy before coding (weight from vitals); a category without a preferred
    unit gets clifpy's base unit (per minute; intermittent masses in mcg)."""
    pd.DataFrame({"hospitalization_id": ["70"], "recorded_dttm": [T(2)], "vital_category": ["weight_kg"],
                  "vital_value": [80.0], "meas_site_name": [None]}).to_parquet(clif / "clif_vitals.parquet")
    pd.DataFrame({"hospitalization_id": ["70", "70"], "admin_dttm": [T(3), T(4)],
                  "med_category": ["norepinephrine", "ext_clonidine"], "med_dose": [8.0, 30.0],
                  "med_dose_unit": ["mcg/min", "mcg/hr"], "mar_action_category": ["start", "start"],
                  "med_route_category": ["iv", None]}).to_parquet(clif / "clif_medication_admin_continuous.parquet")
    df = _run(clif, tmp_path, MED_CON, "MED_CON").sort_values("time")
    assert df["code"].tolist() == ["MED_CON//norepinephrine//mcg/kg/min//start", "MED_CON//ext_clonidine//mcg/min//start"]
    assert df["numeric_value"].tolist() == pytest.approx([0.1, 0.5])   # 8 mcg/min / 80 kg; 30 mcg/hr
    # canonical wall-clock times, whatever the host's time zone (CLIF_MEDS's own run shifts them to the host's)
    assert [t.to_pydatetime() for t in df["time"]] == [T(3).replace(tzinfo=None), T(4).replace(tzinfo=None)]
    from icutwin.meds.elf import qualified_config
    q = _run(clif, tmp_path, qualified_config(MED_CON, "med_route_category"), "MED_CON").sort_values("time")
    assert q["code"].tolist() == ["MED_CON//norepinephrine//mcg/kg/min//start//iv",
                                  "MED_CON//ext_clonidine//mcg/min//start//UNK"]


def test_a_dose_before_the_first_weight(clif: Path, tmp_path: Path) -> None:
    """Conversion is causal: clifpy looks the weight up backward only (CLIF_MEDS, parity), row by row; the build
    ('segment_start') also keeps an infusion that started before any weight in its unweighted unit to its stop,
    so it never changes code mid-way, and a weight charted later never converts an earlier dose."""
    pd.DataFrame({"hospitalization_id": ["70"], "recorded_dttm": [T(5)], "vital_category": ["weight_kg"],
                  "vital_value": [80.0], "meas_site_name": [None]}).to_parquet(clif / "clif_vitals.parquet")
    pd.DataFrame({"hospitalization_id": ["70", "70"], "admin_dttm": [T(3), T(6)],
                  "med_category": ["norepinephrine"] * 2, "med_dose": [8.0, 16.0], "med_dose_unit": ["mcg/min"] * 2,
                  "mar_action_category": ["start", "dose_change"], "med_route_category": ["iv"] * 2}
                 ).to_parquet(clif / "clif_medication_admin_continuous.parquet")
    for lookup, codes, values in (
            ("backward", ["MED_CON//norepinephrine//mcg/min//start", "MED_CON//norepinephrine//mcg/kg/min//dose_change"],
             [8.0, 0.2]),
            ("segment_start", ["MED_CON//norepinephrine//mcg/min//start",
                               "MED_CON//norepinephrine//mcg/min//dose_change"], [8.0, 16.0])):
        stats: dict = {}
        n = convert_domain(connect(), clif, MED_CON, "MED_CON", tmp_path / f"{lookup}.parquet", weight_lookup=lookup,
                           stats=stats)
        df = pd.read_parquet(tmp_path / f"{lookup}.parquet").sort_values("time")
        assert n == 2 and df["code"].tolist() == codes, lookup
        assert df["numeric_value"].tolist() == pytest.approx(values), lookup
        assert stats.get("rows_kept_in_their_segment_unit", 0) == (1 if lookup == "segment_start" else 0)
    # a segment that starts after the weight is converted throughout
    pd.DataFrame({"hospitalization_id": ["70"] * 3, "admin_dttm": [T(3), T(4), T(6)],
                  "med_category": ["norepinephrine"] * 3, "med_dose": [8.0, 0.0, 16.0], "med_dose_unit": ["mcg/min"] * 3,
                  "mar_action_category": ["start", "stop", "start"], "med_route_category": ["iv"] * 3}
                 ).to_parquet(clif / "clif_medication_admin_continuous.parquet")
    convert_domain(connect(), clif, MED_CON, "MED_CON", tmp_path / "s.parquet", weight_lookup="segment_start")
    df = pd.read_parquet(tmp_path / "s.parquet").sort_values("time")
    assert [c.split("//")[2] for c in df["code"]] == ["mcg/min", "mcg/min", "mcg/kg/min"]


def test_qualifier_with_where_keeps_other_rows_unqualified(clif: Path, tmp_path: Path) -> None:
    from icutwin.meds.elf import qualified_config
    cfg = {"subject_id_col": "patient_id", "vitals": {"vital": {
        "code": ["VITAL", "col(vital_category)"], "time": "col(recorded_dttm)", "numeric_value": "vital_value"}}}
    q = qualified_config(cfg, "meas_site_name", "vital_category IN ('sbp', 'dbp', 'map')")
    df = _run(clif, tmp_path, q, "VITAL").sort_values("time")
    assert df["code"].tolist() == ["VITAL//heart_rate", "VITAL//sbp//cuff"]   # the NULL-time spo2 row is dropped


def test_shipped_extension_configs_are_well_formed() -> None:
    from icutwin.clif.conform import table_schema
    from icutwin.meds.build import extension_configs
    from icutwin.meds.elf import META_KEYS, config_table

    configs = extension_configs()
    assert {"ICU", "HEMO", "MCS", "MICRO", "MICRO_SUS", "MED_ORD"} <= set(configs)
    for domain, cfg in configs.items():
        assert all(isinstance(k, str) for k in cfg), domain                       # YAML 1.1: unquoted on/off/yes
        assert len([k for k in cfg if k not in META_KEYS]) == 1, domain            # one CLIF table per domain
        table, concepts = config_table(cfg)
        if not table.startswith("ext_"):
            table_schema(table)                                                    # a CLIF 3.0 table
        if "join" in cfg:
            assert set(cfg["join"]) == {"table", "key"}, domain
        for name, m in concepts.items():
            assert "code" in m and "time" in m, (domain, name)
            code = m["code"]
            assert (code if isinstance(code, str) else str(code[0])).split("//")[0] == domain or domain == "ICU"
