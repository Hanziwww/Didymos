"""Derived concepts (NEE, episodes, KDIGO, fluid balance) on hand-built canonical stays."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from icutwin.meds.derived import derive, resp_status

D = lambda day, h=0: datetime(2100, 1, day, h, tzinfo=UTC)  # noqa: E731
URINE_HOURS = range(1, 31)   # S2: 20 ml every hour (0.2 ml/kg/h at 100 kg), the last one with bladder irrigation


@pytest.fixture(scope="module")
def derived(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("derived")
    c = root / "clif"
    c.mkdir()
    pd.DataFrame({"stay_id": ["S1", "S2", "S3"], "hospitalization_id": ["H1", "H2", "H3"],
                  "icu_in_dttm": [D(2), D(10), D(20)], "icu_out_dttm": [D(5), D(12, 12), D(22)]}
                 ).to_parquet(c / "clif_ext_icu_stay.parquet")
    pd.DataFrame({"hospitalization_id": ["H1", "H2"], "recorded_dttm": [D(2, 1), D(10, 1)],
                  "vital_category": ["weight_kg"] * 2, "vital_value": [80.0, 100.0]}).to_parquet(c / "clif_vitals.parquet")
    cr = [("H1", D(1, 12), 1.0), ("H1", D(2, 6), 1.2), ("H1", D(3, 6), 2.1), ("H1", D(4, 6), 3.1),  # 2x, then >= 3x
          ("H2", D(9, 12), 4.2), ("H2", D(10, 12), 4.6)]   # >= 4 with a rise of >= 0.3 in 48 h: stage 3
    pd.DataFrame({"hospitalization_id": [h for h, _, _ in cr], "lab_collect_dttm": [t for _, t, _ in cr],
                  "lab_category": ["creatinine"] * len(cr), "lab_value_numeric": [v for _, _, v in cr]}
                 ).to_parquet(c / "clif_labs.parquet")
    meds = [  # norepinephrine 8 mcg/min (80 kg: 0.1 mcg/kg/min); vasopressin 2.4 U/h = 0.04 U/min (NEE 0.1)
        (D(2, 2), "norepinephrine", 8.0, "mcg/min", "start"), (D(2, 10), "norepinephrine", 0.0, "mcg/min", "stop"),
        (D(2, 3), "vasopressin", 2.4, "u/hr", "start"), (D(2, 12), "vasopressin", 0.0, "u/hr", "stop"),
        (D(2, 0), "sodium_chloride", 100.0, "ml/hr", "start"), (D(2, 10), "sodium_chloride", 0.0, "ml/hr", "stop"),
        (D(2, 2), "lactated_ringers_solution", 1.0, "ml/min", "start"),          # HiRID / SICdb chart ml/min
        (D(2, 4), "lactated_ringers_solution", 0.0, "ml/min", "stop")]
    pd.DataFrame({"hospitalization_id": ["H1"] * len(meds), "admin_dttm": [m[0] for m in meds],
                  "med_category": [m[1] for m in meds], "med_dose": [m[2] for m in meds],
                  "med_dose_unit": [m[3] for m in meds], "mar_action_category": [m[4] for m in meds]}
                 ).to_parquet(c / "clif_medication_admin_continuous.parquet")
    rs = [("H1", D(2, 1), "imv"), ("H1", D(2, 2), "imv"),                     # 22 h without IMV: a new episode
          ("H1", D(3, 0), "imv"), ("H1", D(3, 0), "nasal_cannula"),           # one status per time: the most invasive
          ("H1", D(3, 2), "imv"), ("H1", D(3, 5), "nasal_cannula"),           # ends at the next charting
          ("H2", D(10, 2), "imv"),                                            # charted once: no episode
          ("H3", D(20, 1), None, {"mode_category": "ps_or_cpap"}),            # a mode before any device: no status
          ("H3", D(20, 2), "imv"),
          ("H3", D(20, 3), None, {"mode_category": "ps_or_cpap"}),            # a ventilator row: the last device
          ("H3", D(20, 4), None, {"mode_category": "acvc"}),                  # an invasive mode: IMV
          ("H3", D(20, 5), None, {"peep_set": 5.0, "resp_rate_set": 14.0, "tidal_volume_set": 450.0}),
          ("H3", D(20, 6), None, {"fio2_set": 0.4}),                          # FiO2 only: no status
          ("H3", D(20, 7), None, {"peep_set": 5.0}),                          # a ventilator row: the last device
          ("H3", D(20, 9), "nasal_cannula"),
          ("H3", D(20, 10), None, {"peep_set": 5.0})]                         # not after a nasal cannula
    values = ("mode_category", "fio2_set", "peep_set", "resp_rate_set", "tidal_volume_set")
    pd.DataFrame({"hospitalization_id": [r[0] for r in rs], "recorded_dttm": [r[1] for r in rs],
                  "device_category": [r[2] for r in rs],
                  **{k: [(r[3] if len(r) > 3 else {}).get(k) for r in rs] for k in values}}
                 ).astype({k: float for k in values[1:]}).to_parquet(c / "clif_respiratory_support.parquet")
    (root / "rs.txt").write_text(str(c / "clif_respiratory_support.parquet"))
    pd.DataFrame({"hospitalization_id": ["H1", "H2"], "recorded_dttm": [D(2, 6), D(11, 6)],
                  "input_category": ["sodium_chloride", "flush_irrigation_urine"], "input_volume": [500.0, 500.0]}
                 ).to_parquet(c / "clif_input.parquet")
    urine = [("H1", D(2, 8), 300.0)] + [("H2", D(10) + timedelta(hours=h), 520.0 if h == 30 else 20.0)
                                         for h in URINE_HOURS]
    pd.DataFrame({"hospitalization_id": [h for h, _, _ in urine], "recorded_dttm": [t for _, t, _ in urine],
                  "output_group": ["urine"] * len(urine), "output_volume": [v for _, _, v in urine]}
                 ).to_parquet(c / "clif_output.parquet")
    pd.DataFrame({"hospitalization_id": ["H2"], "recorded_dttm": [D(12)], "crrt_mode_category": [None]}
                 ).to_parquet(c / "clif_crrt_therapy.parquet")
    out = root / "derived"
    derive(c, out, root / "work", with_clifpy=False)
    return out


def test_norepinephrine_equivalents(derived: Path) -> None:
    nee = pd.read_parquet(derived / "nee.parquet").sort_values("t")
    assert [(t.hour, v) for t, v in zip(nee["t"], nee["nee_mcg_kg_min"])] == [(2, 0.1), (3, 0.2), (10, 0.1), (12, 0.0)]


def test_episodes(derived: Path) -> None:
    ep = pd.read_parquet(derived / "episodes.parquet").sort_values(["episode", "start_dttm"])
    got = [(s_id, e, s.day, s.hour, t.day, t.hour)
           for s_id, e, s, t in zip(ep["stay_id"], ep["episode"], ep["start_dttm"], ep["end_dttm"])]
    assert got == [("S1", "imv", 2, 1, 2, 2), ("S1", "imv", 3, 0, 3, 5), ("S3", "imv", 20, 2, 20, 9),
                   ("S2", "rrt", 12, 0, 12, 0), ("S1", "vasopressor", 2, 2, 2, 12)]


def test_kdigo_creatinine_stages(derived: Path) -> None:
    k = pd.read_parquet(derived / "kdigo.parquet").dropna(subset=["creat"]).sort_values(["stay_id", "t"])
    assert [(s, t.day, g) for s, t, g in zip(k["stay_id"], k["t"], k["aki_stage_creat"])] == [
        ("S1", 1, 0), ("S1", 2, 0), ("S1", 3, 2), ("S1", 4, 3), ("S2", 9, 0), ("S2", 10, 3)]


def test_kdigo_urine_output_needs_documented_hours(derived: Path) -> None:
    k = pd.read_parquet(derived / "kdigo.parquet")
    k = k[k["stay_id"] == "S2"].set_index("t")
    at = lambda h: k.loc[datetime(2100, 1, 10) + timedelta(hours=h)]  # noqa: E731
    # 0.2 ml/kg/h throughout: stage 1 once 6 h are documented, 2 after 12 h, 3 after 24 h (none in the first 6 h)
    assert [at(h)["aki_stage_uo"] for h in (6, 7, 8, 13, 20, 25)] == [0, 1, 1, 2, 2, 3]
    assert at(7)["uo_rt_6hr"] == 0.2 and pd.isna(at(7)["uo_rt_12hr"]) and at(25)["uo_rt_24hr"] == 0.2
    assert at(30)["aki_stage_uo"] == 3        # 520 ml out with 500 ml bladder irrigation in: 20 ml of urine
    assert at(13)["aki_stage"] == 2 and at(13)["aki_stage_smoothed"] == 3   # creatinine stage 3 at hour 12


def test_kdigo_crrt_is_stage_3(derived: Path) -> None:
    k = pd.read_parquet(derived / "kdigo.parquet")
    row = k[(k["stay_id"] == "S2") & (k["t"] == datetime(2100, 1, 12))].iloc[0]
    assert row["aki_stage_crrt"] == 3 and row["aki_stage"] == 3


def test_fluid_balance(derived: Path) -> None:
    fb = pd.read_parquet(derived / "fluid_balance.parquet")
    fb = fb[fb["stay_id"] == "S1"].set_index("icu_day")
    assert fb.loc[0, "fluid_in_ml"] == 1620.0 and fb.loc[0, "fluid_out_ml"] == 300.0 and fb.loc[0, "net_ml"] == 1320.0


def test_respiratory_status_waterfall(derived: Path) -> None:
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")   # as derive()
    rs = (derived.parent / "rs.txt").read_text()
    con.execute(f"CREATE VIEW respiratory_support AS SELECT * FROM read_parquet('{rs.replace(chr(92), '/')}')")
    resp_status(con)
    got = con.execute("SELECT hour(t), status FROM resp_status WHERE hospitalization_id = 'H3' ORDER BY t").fetchall()
    assert got == [(2, "imv"), (3, "imv"), (4, "imv"), (5, "imv"), (7, "imv"), (9, "nasal_cannula")]


def test_unsupported_pf_below_200_scores_2() -> None:
    """clifpy scores a P/F < 200 without IMV/NIPPV/CPAP as missing (0); SOFA gives it the < 300 score, 2."""
    import polars as pl

    from icutwin.meds.derived import fix_sofa_resp

    rows = [  # p_f, device at that P/F, clifpy's sofa_resp
        (150.0, "Nasal Cannula", 0), (90.0, None, 0), (150.0, "IMV", 3), (80.0, "CPAP", 4), (250.0, "IMV", 2),
        (350.0, "Room Air", 1), (None, None, 0)]
    sofa = pl.DataFrame({"stay_id": [str(i) for i in range(len(rows))], "p_f": [r[0] for r in rows],
                         "device_category": [r[1] for r in rows], "sofa_resp": pl.Series([r[2] for r in rows], dtype=pl.Int32),
                         **{c: pl.Series([1] * len(rows), dtype=pl.Int32)
                            for c in ("sofa_cv_97", "sofa_coag", "sofa_liver", "sofa_cns", "sofa_renal")}})
    out = fix_sofa_resp(sofa)
    assert out["sofa_resp"].to_list() == [2, 2, 3, 4, 2, 1, 0]
    assert out["clifpy_sofa_resp"].to_list() == [r[2] for r in rows]
    assert out["sofa_total"].to_list() == [7, 7, 8, 9, 7, 6, 5]
    assert out.schema["sofa_resp"] == pl.Int32


def test_respiration_is_the_worst_score_of_the_pf_pairs() -> None:
    """SOFA scores the worst value of the window: a P/F of 66 on IMV (4) outweighs the lowest P/F, 56 on high-flow
    oxygen (2), which is all clifpy scores."""
    import polars as pl

    from icutwin.meds.derived import fix_sofa_resp, worst_resp_scores

    pairs = pl.DataFrame({"stay_id": ["a", "a", "b"], "concurrent_pf": [56.0, 66.0, 350.0],
                          "device_category": ["High Flow NC", "IMV", "Room Air"]})
    sofa = pl.DataFrame({"stay_id": ["a", "b", "c"], "p_f": [56.0, 350.0, None],
                         "device_category": ["High Flow NC", "Room Air", None],
                         "sofa_resp": pl.Series([0, 1, 0], dtype=pl.Int32),
                         **{c: pl.Series([0, 0, 0], dtype=pl.Int32)
                            for c in ("sofa_cv_97", "sofa_coag", "sofa_liver", "sofa_cns", "sofa_renal")}})
    out = fix_sofa_resp(sofa, worst_resp_scores(pairs)).sort("stay_id")
    assert out["sofa_resp"].to_list() == [4, 1, 0] and out["clifpy_sofa_resp"].to_list() == [0, 1, 0]
    assert out["sofa_total"].to_list() == [4, 1, 0] and "worst_resp" not in out.columns


def test_weights_charted_later_are_flagged() -> None:
    """mimic-code's weight_durations gives a rate charted before the first weight that weight; the NEE row says so
    (provenance for prospective tasks), and rows after the weight do not."""
    import duckdb

    from icutwin.meds import derived as dv
    con = duckdb.connect()
    con.execute("""CREATE TEMP TABLE vitals AS SELECT * FROM (VALUES
                   ('H1', TIMESTAMP '2100-01-01 02:00', 'weight_kg', 80.0)) t(hospitalization_id, recorded_dttm,
                   vital_category, vital_value)""")
    con.execute("""CREATE TEMP TABLE medication_admin_continuous AS SELECT * FROM (VALUES
                   ('H1', TIMESTAMP '2100-01-01 01:00', 'norepinephrine', 8.0, 'mcg/min'),
                   ('H1', TIMESTAMP '2100-01-01 03:00', 'norepinephrine', 16.0, 'mcg/min'),
                   ('H1', TIMESTAMP '2100-01-01 04:00', 'norepinephrine', 0.0, 'mcg/min'))
                   t(hospitalization_id, admin_dttm, med_category, med_dose, med_dose_unit)""")
    dv._weights(con)
    dv.nee(con)
    got = con.execute("SELECT hour(t), nee_mcg_kg_min, weight_charted_later FROM nee ORDER BY t").fetchall()
    assert got == [(1, 0.1, True), (3, 0.2, False), (4, 0.0, False)]
