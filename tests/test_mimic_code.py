"""mimic-code parity helpers: the ventilation-status extract and the agreement metrics."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from icutwin.meds import mimic_code as mc


def test_ventilation_status_sql_keeps_mimic_code_up_to_the_status(tmp_path: Path) -> None:
    sql = tmp_path / "mimic-code" / mc.CONCEPTS / "treatment" / "ventilation.sql"
    sql.parent.mkdir(parents=True)
    sql.write_text("-- header\nWITH tm AS (SELECT 1 AS stay_id, TIMESTAMP '2150-01-01' AS charttime)\n"
                   ", vs AS (SELECT stay_id, charttime, 'InvasiveVent' AS ventilation_status FROM tm)\n"
                   ", vd0 AS (SELECT * FROM vs)\nSELECT * FROM vd0", encoding="utf-8")
    text = mc.ventilation_status_sql(tmp_path / "mimic-code")
    assert text.startswith("WITH tm AS (") and "vd0" not in text
    assert duckdb.sql(text).fetchall()[0][2] == "InvasiveVent"


def test_binary_agreement_and_kappa() -> None:
    con = duckdb.connect()
    rows = [(True, True)] * 40 + [(True, False)] * 10 + [(False, True)] * 5 + [(False, False)] * 45
    con.execute("CREATE TABLE p (ours BOOLEAN, theirs BOOLEAN)")
    con.executemany("INSERT INTO p VALUES (?, ?)", rows)
    r = mc._binary(con, "SELECT * FROM p")
    assert (r["n"], r["both"], r["only_ours"], r["only_mimic_code"], r["neither"]) == (100, 40, 10, 5, 45)
    assert r["agreement"] == 0.85 and r["kappa"] == pytest.approx(0.7)


def test_ordinal_and_continuous_metrics() -> None:
    con = duckdb.connect()
    con.execute("CREATE TABLE p AS SELECT * FROM (VALUES (0, 0), (1, 1), (2, 3), (3, 3), (NULL, 2)) t(ours, theirs)")
    o = mc._ordinal(con, "SELECT * FROM p")
    assert (o["n"], o["exact"], o["within_1"], o["mean_diff"]) == (4, 0.75, 1.0, -0.25)
    assert o["confusion"] == [[0, 0, 1], [1, 1, 1], [2, 3, 1], [3, 3, 1]]
    c = mc._continuous(con, "SELECT CAST(ours AS DOUBLE) AS ours, CAST(theirs AS DOUBLE) AS theirs FROM p", 0.5)
    assert (c["n"], c["within_0.5"], c["only_mimic_code"], c["only_ours"]) == (4, 0.75, 1, 0)


@pytest.mark.parametrize("vent, novent, first_day, hourly", [
    (90, None, 4, 4), (150, 350, 3, 3), (250, None, 0, 2), (350, None, 0, 1), (275, 347, 1, 2), (None, 250, 2, 2),
    (450, None, 0, 0), (None, None, None, None)])
def test_mimic_code_respiration_rules(vent, novent, first_day, hourly) -> None:
    """``first_day_sofa`` leaves a ventilated P/F of 200-400 unscored; mimic-code's hourly ``sofa`` scores it."""
    v, n = f"CAST({vent if vent is not None else 'NULL'} AS DOUBLE)", f"CAST({novent if novent is not None else 'NULL'} AS DOUBLE)"
    assert duckdb.sql(f"SELECT {mc._mc_resp(v, n)}, {mc._mc_resp_hourly(v, n)}").fetchone() == (first_day, hourly)


def test_renal_scores() -> None:
    cr = [(None, None), (1.1, 0), (1.2, 1), (2.0, 2), (3.5, 3), (5.0, 4)]
    for x, score in cr:
        assert duckdb.sql(f"SELECT {mc._creat_score(f'CAST({x if x is not None else 'NULL'} AS DOUBLE)')}").fetchone()[0] == score
    uo = [(None, None), (199.0, 4), (200.0, 3), (499.0, 3), (500.0, 0)]
    for x, score in uo:
        assert duckdb.sql(f"SELECT {mc._uo_score(f'CAST({x if x is not None else 'NULL'} AS DOUBLE)')}").fetchone()[0] == score


def test_attribute_takes_the_first_cause_that_holds() -> None:
    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE j AS SELECT * FROM (VALUES (1, 1, 0), (2, 0, 1), (3, 1, 1), (4, 0, 0)) t(id, a, b)")
    got = mc._attribute(con, "id <= 4", [("a holds", "a = 1"), ("the 'b' rule", "b = 1")])
    assert got == {"stays": 4, "by_cause": {"a holds": 2, "the 'b' rule": 1, "unexplained": 1}}


def test_first_day_sofa_inputs_sql_stops_at_the_component_inputs(tmp_path: Path) -> None:
    sql = tmp_path / "mimic-code" / mc.CONCEPTS / "firstday" / "first_day_sofa.sql"
    sql.parent.mkdir(parents=True)
    sql.write_text("-- header\nWITH vaso_stg AS (SELECT 1 AS stay_id)\n, scorecomp AS (SELECT stay_id, 3.0 AS creatinine_max "
                   "FROM vaso_stg)\n, scorecalc AS (SELECT stay_id, 2 AS renal FROM scorecomp)\nSELECT * FROM scorecalc",
                   encoding="utf-8")
    text = mc.first_day_sofa_inputs_sql(tmp_path / "mimic-code")
    assert "scorecalc" not in text and duckdb.sql(text).fetchall() == [(1, 3.0)]
