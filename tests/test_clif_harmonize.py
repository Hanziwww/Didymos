"""S2b canonical-CLIF harmonization on clifpy's bundled synthetic demo data (no real patients)."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import clifpy
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from icutwin.clif.conform import connect
from icutwin.clif.harmonize import REPORT, harmonize_dir, hourly_vitals

DEMO = Path(clifpy.__file__).parent / "data" / "clif_demo"


@pytest.fixture(scope="module")
def harmonized(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("h")
    raw = root / "raw"
    shutil.copytree(DEMO, raw)
    # the consortium demo data carries the same off-vocabulary values as CLIF-MIMIC output, so it is
    # harmonized non-strictly here and the violations are asserted to be *reported*
    harmonize_dir(raw, root / "out", temp_dir=root / "duck", strict=False)
    return root / "out"


def test_respiratory_values_follow_clif30(harmonized: Path) -> None:
    rs = pd.read_parquet(harmonized / "clif_respiratory_support.parquet")
    devices = set(rs["device_category"].dropna())
    modes = set(rs["mode_category"].dropna())
    assert {"imv", "nasal_cannula", "hfnc"} <= devices  # 'High Flow NC' -> 'hfnc' (official rename)
    assert {"acvc", "prvc", "ps_or_cpap"} <= modes
    assert all(v == v.lower() for v in devices | modes)


def test_every_table_reported(harmonized: Path) -> None:
    import json

    report = json.loads((harmonized / REPORT).read_text())
    assert set(report["tables"]) == {p.stem.removeprefix("clif_") for p in DEMO.glob("clif_*.parquet")}
    assert report["tables"]["vitals"]["density"]["rows_out"] <= report["tables"]["vitals"]["density"]["rows_in"]
    # off-vocabulary values are surfaced, never silently passed through
    assert "dextrose_in_water_d5w" in report["violations"]["medication_admin_intermittent"]["med_category"]


def test_strict_mode_rejects_undeclared_values(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy2(DEMO / "clif_position.parquet", raw / "clif_position.parquet")
    harmonize_dir(raw, tmp_path / "ok", temp_dir=tmp_path / "duck")  # conformant table passes strictly
    bad = pd.read_parquet(DEMO / "clif_position.parquet")
    bad.loc[bad.index[0], "position_category"] = "sitting"
    bad.to_parquet(raw / "clif_position.parquet")
    with pytest.raises(ValueError, match="position_category"):
        harmonize_dir(raw, tmp_path / "bad", temp_dir=tmp_path / "duck")


def test_respiratory_density_keeps_change_points(tmp_path: Path) -> None:
    from icutwin.clif.harmonize import hourly_respiratory

    t = lambda h, m: datetime(2100, 1, 1, h, m, tzinfo=UTC)  # noqa: E731
    rows = [  # device changes at 10:40 must survive; same-device rows in an hour collapse
        ("H1", t(10, 0), "imv", "acvc", 0.5, 400.0), ("H1", t(10, 20), "imv", "acvc", 0.6, 420.0),
        ("H1", t(10, 40), "hfnc", None, 0.4, None), ("H1", t(10, 50), "hfnc", None, 0.35, None),
        ("H1", t(11, 10), "hfnc", None, 0.3, None),
    ]
    src = tmp_path / "rs.parquet"
    pq.write_table(pa.table({
        "hospitalization_id": [r[0] for r in rows], "recorded_dttm": [r[1] for r in rows],
        "device_category": [r[2] for r in rows], "mode_category": [r[3] for r in rows],
        "fio2_set": [r[4] for r in rows], "tidal_volume_obs": [r[5] for r in rows],
    }), src)
    hourly_respiratory(connect(), src, tmp_path / "o.parquet")
    out = pd.read_parquet(tmp_path / "o.parquet").sort_values("recorded_dttm")
    assert out["device_category"].tolist() == ["imv", "hfnc", "hfnc"]
    assert [ts.minute for ts in out["recorded_dttm"]] == [20, 50, 10]
    assert out["fio2_set"].tolist() == [pytest.approx(0.6), pytest.approx(0.35), pytest.approx(0.3)]  # last setting
    assert out["tidal_volume_obs"].iloc[0] == pytest.approx(410.0)  # median of observations


def test_vitals_density_is_one_row_per_hour(harmonized: Path) -> None:
    con = connect()  # UTC session: local-time date_trunc would merge DST fall-back hours
    dupes = con.execute(
        f"SELECT count(*) FROM (SELECT hospitalization_id, vital_category, meas_site_name, date_trunc('hour', recorded_dttm) h, "
        f"count(*) n FROM read_parquet('{(harmonized / 'clif_vitals.parquet').as_posix()}') GROUP BY ALL HAVING n > 1)"
    ).fetchone()[0]
    assert dupes == 0


def test_hourly_rule_semantics(tmp_path: Path) -> None:
    t = lambda h, m: datetime(2100, 1, 1, h, m, tzinfo=UTC)  # noqa: E731
    rows = [
        ("H1", t(10, 5), "hr", "heart_rate", 80.0, None),
        ("H1", t(10, 20), "hr", "heart_rate", 90.0, None),
        ("H1", t(10, 50), "hr2", "heart_rate", 120.0, None),   # three in hour 10 -> median 90 at 10:50
        ("H1", t(11, 7), "hr", "heart_rate", 70.0, None),     # single -> native time kept
        ("H1", t(10, 30), "abp", "sbp", 130.0, "arterial"),   # site kept separate from cuff
        ("H1", t(10, 40), "nibp", "sbp", 110.0, "cuff"),
    ]
    src = tmp_path / "v.parquet"
    pq.write_table(pa.table({
        "hospitalization_id": [r[0] for r in rows], "recorded_dttm": [r[1] for r in rows],
        "vital_name": [r[2] for r in rows], "vital_category": [r[3] for r in rows],
        "vital_value": [r[4] for r in rows], "meas_site_name": [r[5] for r in rows],
    }), src)
    con = connect()
    hourly_vitals(con, src, tmp_path / "o.parquet")
    out = pd.read_parquet(tmp_path / "o.parquet").sort_values(["vital_category", "recorded_dttm", "meas_site_name"])
    hr = out[out.vital_category == "heart_rate"]
    assert hr["vital_value"].tolist() == [90.0, 70.0]
    assert [ts.minute for ts in hr["recorded_dttm"]] == [50, 7]
    assert hr["vital_name"].tolist() == ["hr2", "hr"]  # name of the last measurement in the hour
    sbp = out[out.vital_category == "sbp"].set_index("meas_site_name")["vital_value"].to_dict()
    assert sbp == {"arterial": 130.0, "cuff": 110.0}


def test_hourly_rule_breaks_ties_the_same_way_in_every_run(tmp_path: Path) -> None:
    """Two items charting one value at the hour's last time (AmsterdamUMCdb monitor and ventilator rates): the
    kept name and site do not depend on the input order."""
    t = datetime(2100, 1, 1, 10, 30, tzinfo=UTC)
    rows = [("H1", t, "8874: Ademfrequentie Monitor", "respiratory_rate", 18.0, "monitor"),
            ("H1", t, "12266: Ademfreq.", "respiratory_rate", 18.0, "ventilator")]
    con, got = connect(), []
    for k, order in enumerate((rows, rows[::-1])):
        src = tmp_path / f"v{k}.parquet"
        pq.write_table(pa.table({
            "hospitalization_id": [r[0] for r in order], "recorded_dttm": [r[1] for r in order],
            "vital_name": [r[2] for r in order], "vital_category": [r[3] for r in order],
            "vital_value": [r[4] for r in order], "meas_site_name": [r[5] for r in order],
        }), src)
        hourly_vitals(con, src, tmp_path / f"o{k}.parquet")
        got.append(pd.read_parquet(tmp_path / f"o{k}.parquet")[["vital_name", "meas_site_name"]].values.tolist())
    assert got[0] == got[1] == [["8874: Ademfrequentie Monitor", "monitor"]]


def test_exact_duplicates_keep_the_first_result(tmp_path: Path) -> None:
    from icutwin.clif.harmonize import drop_exact_duplicates

    t = lambda h, m=0: datetime(2100, 1, 1, h, m, tzinfo=UTC)  # noqa: E731
    rows = [  # one draw entered twice (lab and charted copy): the earlier result is kept
        ("H1", t(10), t(11), "sodium", 140.0, "140", "Sodium (charted)"),
        ("H1", t(10), t(10, 30), "sodium", 140.0, "140", "Sodium"),
        ("H1", t(10), t(10, 30), "sodium", 141.0, "141", "Sodium"),    # another value: kept
        ("H1", t(12), t(12, 30), "sodium", 140.0, "140", "Sodium"),    # another draw: kept
    ]
    cols = ("hospitalization_id", "lab_collect_dttm", "lab_result_dttm", "lab_category", "lab_value_numeric",
            "lab_value", "lab_name")
    src = tmp_path / "labs.parquet"
    pq.write_table(pa.table({c: [r[i] for r in rows] for i, c in enumerate(cols)}), src)
    con = connect()
    report = drop_exact_duplicates(con, src, tmp_path / "o.parquet", "labs")
    out = pd.read_parquet(tmp_path / "o.parquet")
    assert report["rows_dropped"] == 1 and len(out) == 3
    assert out.loc[(out.lab_collect_dttm == t(10)) & (out.lab_value_numeric == 140.0), "lab_name"].tolist() == ["Sodium"]
    assert drop_exact_duplicates(con, src, tmp_path / "v.parquet", "vitals") is None  # vitals: hourly rule instead


def test_conform_derives_groups_and_enforces_the_schema(tmp_path: Path) -> None:
    from icutwin.clif.conform import table_schema
    from icutwin.clif.harmonize import conform_to_schema

    path = tmp_path / "mc.parquet"
    pd.DataFrame({
        "hospitalization_id": ["H1", "H1", "H1"], "patient_id": ["P1"] * 3,      # a tool-specific extra column
        "admin_dttm": pd.to_datetime(["2100-01-01 10:00"] * 3, utc=True),
        "med_category": ["norepinephrine", "propofol", "ext_not_grouped"], "med_group": ["wrong", None, None],
        "mar_action_category": ["start", "stop", "dose_change"], "med_dose": ["0.1", "n/a", "5"],
    }).to_parquet(path)
    report = conform_to_schema(connect(), path, "medication_admin_continuous")
    out = pd.read_parquet(path)
    assert list(out.columns) == [c["name"] for c in table_schema("medication_admin_continuous")["columns"]]
    # groups come from the category, whatever the source wrote
    assert out["med_group"].tolist() == ["vasoactives", "sedation", None]
    assert out["mar_action_group"].tolist() == ["administered", "not_administered", "administered"]
    assert out["med_dose"].tolist()[0] == pytest.approx(0.1) and pd.isna(out["med_dose"].iloc[1])
    assert report["dropped"] == ["patient_id"]
    assert report["uncastable_values_nulled"] == {"med_dose": 1}
    assert report["groups"] == {"med_group": 2, "mar_action_group": 3}
    assert "med_route_category" in report["added"]


def test_ecmo_mcs_to_mcs_reads_official_and_clif_mimic_columns(tmp_path: Path) -> None:
    from icutwin.clif.harmonize import ecmo_mcs_to_mcs

    t = datetime(2100, 1, 1, 10, tzinfo=UTC)
    con = connect()

    def run(rows: list[dict]) -> list[tuple]:
        pd.DataFrame(rows).to_parquet(tmp_path / "src.parquet")
        ecmo_mcs_to_mcs(con, tmp_path / "src.parquet", tmp_path / "dst.parquet")
        out = pd.read_parquet(tmp_path / "dst.parquet").astype(object).where(lambda d: d.notna(), None)
        return sorted(zip(out.hospitalization_id, out.support_category, out.device_category, out.config_category,
                          out.setting_category, out.setting_value), key=str)

    mimic = lambda h, ctx, name, dev, grp, cfg, cpn, cpc, cpv, flow, sweep=None, fdo2=None: {  # noqa: E731
        "hospitalization_id": h, "recorded_dttm": t, "_device_context": ctx, "device_name": name,
        "device_category": dev, "mcs_group": grp, "ecmo_configuration_category": cfg, "control_parameter_name": cpn,
        "control_parameter_category": cpc, "control_parameter_value": cpv, "flow": flow, "sweep_set": sweep,
        "fdo2_set": fdo2}
    assert run([   # CLIF-MIMIC: lowercase values, control parameters, CH item family = Cardiohelp
        mimic("H1", "ecmo_ch", "Circuit Configuration (CH) = VA", "ecmo_other_unspec", "ecmo", "v_a", "Speed (CH)",
              "rpm", 3100.0, 3.2, 2.0, 0.6),
        mimic("H2", "impella_l", "Type of Catheter (Impella) = 2.5 / CP", "impella_lv_other_unspec", "impella_lvad",
              None, "Performance Level", "impella_power", 6.0, 2.4),
    ]) == sorted([("H1", "ecmo", "cardiohelp", "v_a", "fdo2", 0.6), ("H1", "ecmo", "cardiohelp", "v_a", "flow", 3.2),
                  ("H1", "ecmo", "cardiohelp", "v_a", "rpm", 3100.0), ("H1", "ecmo", "cardiohelp", "v_a", "sweep", 2.0),
                  ("H2", "lvad", "impella", "2_5_or_cp", "flow", 2.4),
                  ("H2", "lvad", "impella", "2_5_or_cp", "performance", 6.0)], key=str)
    official = lambda h, dev, grp, rate=None, flow=None: {  # noqa: E731
        "hospitalization_id": h, "recorded_dttm": t, "device_name": dev, "device_category": dev, "mcs_group": grp,
        "device_metric_name": None, "device_rate": rate, "sweep": None, "flow": flow, "fdO2": None}
    assert run([   # official 2.1 columns and spellings (CLIF-AmsterdamUMCdb); Impella device_rate is the P-level
        official("H3", "IABP", "IABP"), official("H4", "Impella_CP", "temporary_LVAD", rate=8.0, flow=3.0),
    ]) == sorted([("H3", "ext_iabp", None, None, None, None), ("H4", "lvad", "impella", "cp", "flow", 3.0),
                  ("H4", "lvad", "impella", "cp", "performance", 8.0)], key=str)


def test_medication_dose_units_are_canonical(tmp_path: Path) -> None:
    from icutwin.clif.harmonize import canonical_medications

    src = tmp_path / "m.parquet"
    pd.DataFrame({"hospitalization_id": ["H"] * 5, "med_category": ["vasopressin", "penicillin", "cefazolin",
                                                                    "vancomycin", "potassium_repletion"],
                  "med_dose": [0.04, 3.0, 2.0, 1.0, 20.0],
                  "med_dose_unit": ["units/hour", "million units", "gm", "grams", "mEq"]}).to_parquet(src)
    report = canonical_medications(connect(), src, tmp_path / "o.parquet", "medication_admin_intermittent")
    out = pd.read_parquet(tmp_path / "o.parquet")
    assert out["med_dose_unit"].tolist() == ["u/hr", "u", "g", "g", "meq"]   # clifpy spelling; mEq kept
    assert out["med_dose"].tolist() == [0.04, 3e6, 2.0, 1.0, 20.0]          # million units scaled
    assert report["dose_units_scaled"] == {"million units": 1e6}


def test_meq_with_a_period_is_meq() -> None:
    """MIMIC charts CRRT potassium in 'mEq./hour': one unit name with eICU's 'mEq/hr'."""
    from icutwin.clif.units import clean_dose_unit
    assert clean_dose_unit("mEq./hour") == ("meq/hr", 1.0) and clean_dose_unit("mEq.") == ("meq", 1.0)
    assert clean_dose_unit("mEq/hr") == ("meq/hr", 1.0)


def test_counting_units_have_one_spelling() -> None:
    """A count of product units is one unit whatever the source's spelling (M4: NWICU 'tablet', eICU 'ea'
    and MIMIC 'TAB' are one code)."""
    from icutwin.clif.units import clean_dose_unit
    for spellings, unit in ((("tablet", "Tablets", "TAB", "tabs", "split tab"), "tab"), (("capsule", "CAP"), "cap"),
                            (("puffs", "Puff"), "puff"), (("inhalation", "INH"), "inh"), (("ea", "each"), "each"),
                            (("vl", "vial"), "vial"), (("drops", "gtt"), "drop")):
        assert {clean_dose_unit(s)[0] for s in spellings} == {unit}, spellings
    assert clean_dose_unit("mg/kg") == ("mg/kg", 1.0) and clean_dose_unit("units/kg") == ("u/kg", 1.0)


def test_official_dose_ranges_apply_to_canonical_units(tmp_path: Path) -> None:
    from icutwin.clif.outliers import apply_outliers, official_outlier_config

    ranges = official_outlier_config()["medication_admin_continuous"]["med_dose"]
    cat, rng = next((c, s["units/hr"]) for c, s in ranges.items() if isinstance(s, dict) and "units/hr" in s)
    src = tmp_path / "m.parquet"
    pd.DataFrame({"med_category": [cat, cat], "med_dose_unit": ["u/hr", "units/hr"],
                  "med_dose": [rng["max"] * 10, rng["max"] * 10]}).to_parquet(src)
    apply_outliers(connect(), src, tmp_path / "o.parquet", "medication_admin_continuous")
    assert pd.read_parquet(tmp_path / "o.parquet")["med_dose"].isna().all()


def _continuous(path: Path, rows: list[tuple]) -> None:
    pd.DataFrame({"hospitalization_id": [r[0] for r in rows],
                  "admin_dttm": [datetime(2100, 1, 1, r[1], r[2], tzinfo=UTC) for r in rows],
                  "med_order_id": [r[3] for r in rows], "med_category": ["norepinephrine"] * len(rows),
                  "med_name": ["NE"] * len(rows), "med_dose": [r[4] for r in rows],
                  "med_dose_unit": ["mcg/kg/min"] * len(rows), "mar_action_category": [r[5] for r in rows],
                  "mar_action_name": [None] * len(rows)}).to_parquet(path)


def _stream(path: Path, hosp: str = "H1") -> list[tuple]:
    df = pd.read_parquet(path).query("hospitalization_id == @hosp").sort_values("admin_dttm")
    return [(t.hour, t.minute, a, round(d, 6) if d == d and d is not None else None)
            for t, a, d in zip(df["admin_dttm"], df["mar_action_category"], df["med_dose"])]


def test_concurrent_orders_are_summed(tmp_path: Path) -> None:
    from icutwin.clif.harmonize import canonical_continuous

    _continuous(tmp_path / "c.parquet", [   # two MetaVision orders of one drug overlap from 11:00 to 12:00
        ("H1", 10, 0, "A", 0.1, "start"), ("H1", 12, 0, "A", 0.0, "stop"),
        ("H1", 11, 0, "B", 0.05, "start"), ("H1", 11, 30, "B", 0.08, "dose_change"), ("H1", 13, 0, "B", 0.0, "stop")])
    report = canonical_continuous(connect(), tmp_path / "c.parquet", tmp_path / "o.parquet")
    assert _stream(tmp_path / "o.parquet") == [(10, 0, "start", 0.1), (11, 0, "dose_change", 0.15),
                                               (11, 30, "dose_change", 0.18), (12, 0, "dose_change", 0.08),
                                               (13, 0, "stop", 0.0)]
    assert pd.read_parquet(tmp_path / "o.parquet")["med_order_id"].isna().all()
    assert report["streams_from_several_orders"] == 1


def test_float32_doses_still_stop(tmp_path: Path) -> None:
    """CLIF-MIMIC writes FLOAT doses: summed in float32, 0.300120 - 0.100040 rounds and the stop keeps a 7e-9
    residue (no stop; the next start becomes a dose change). Streams are summed in double precision."""
    from icutwin.clif.harmonize import canonical_continuous

    _continuous(tmp_path / "c.parquet", [   # MIMIC subset, hospitalization 20897880
        ("H1", 3, 40, "A", 0.1000400260090828, "start"), ("H1", 3, 55, "A", 0.3001200258731842, "dose_change"),
        ("H1", 4, 8, "A", 0.0, "stop"), ("H1", 4, 39, "A", 0.04999999701976776, "start"),
        ("H1", 6, 0, "A", 0.0, "stop")])
    df = pd.read_parquet(tmp_path / "c.parquet")
    df.astype({"med_dose": "float32"}).to_parquet(tmp_path / "c.parquet")
    canonical_continuous(connect(), tmp_path / "c.parquet", tmp_path / "o.parquet")
    assert [a for *_, a, _ in _stream(tmp_path / "o.parquet")] == ["start", "dose_change", "stop", "start", "stop"]


def test_an_order_repeating_another_counts_once(tmp_path: Path) -> None:
    """Three orders with identical events are one infusion documented three times (MIMIC); a fourth, different
    order still adds."""
    from icutwin.clif.harmonize import canonical_continuous

    _continuous(tmp_path / "c.parquet", [
        *[("H1", 10, 0, o, 0.5, "start") for o in "ABC"], *[("H1", 14, 0, o, 0.0, "stop") for o in "ABC"],
        ("H1", 11, 0, "D", 0.5, "start"), ("H1", 15, 0, "D", 0.0, "stop")])
    report = canonical_continuous(connect(), tmp_path / "c.parquet", tmp_path / "o.parquet")
    assert _stream(tmp_path / "o.parquet") == [(10, 0, "start", 0.5), (11, 0, "dose_change", 1.0),
                                               (14, 0, "dose_change", 0.5), (15, 0, "stop", 0.0)]
    assert report["duplicate_orders_dropped"] == 2


def test_exchanges_are_not_stops_and_presence_rows_pass(tmp_path: Path) -> None:
    from icutwin.clif.harmonize import canonical_continuous

    _continuous(tmp_path / "c.parquet", [
        ("H1", 10, 0, None, 5.0, "start"), ("H1", 12, 0, None, 0.0, "stop"),
        ("H1", 12, 5, None, 5.0, "start"),                       # new syringe, same rate: nothing happened
        ("H1", 14, 0, None, 0.0, "stop"), ("H1", 14, 10, None, 7.0, "start"),   # new syringe, new rate
        ("H1", 16, 0, None, 0.0, "stop"), ("H1", 17, 0, None, 5.0, "start"),    # a 60 min pause is a stop
        ("H2", 9, 0, None, None, "start"), ("H2", 10, 0, None, None, "going"),  # presence only (NWICU)
        ("H2", 11, 0, None, 0.0, "stop")])
    report = canonical_continuous(connect(), tmp_path / "c.parquet", tmp_path / "o.parquet")
    assert _stream(tmp_path / "o.parquet") == [(10, 0, "start", 5.0), (14, 10, "dose_change", 7.0),
                                               (16, 0, "stop", 0.0), (17, 0, "start", 5.0)]
    assert _stream(tmp_path / "o.parquet", "H2") == [(9, 0, "start", None), (10, 0, "going", None), (11, 0, "stop", 0.0)]
    assert report["exchanges_bridged"] == 2


def test_route_basis_keeps_source_routes_apart_from_inferred_and_assumed_ones() -> None:
    """med_route_basis: what a route category rests on; no source route is ever 'verified'."""
    from icutwin.clif.harmonize import route_basis_sql
    con = duckdb.connect()
    cases = [("nwicu", "medication_admin_intermittent", "Injection", "im", "inferred"),
             ("nwicu", "medication_admin_intermittent", "Oral", "enteral", "reported"),
             ("aumc", "medication_admin_intermittent", "Injecties Haematologisch", "iv", "reported_class"),
             ("inspire", "medication_admin_intermittent", "iv", "iv", "reported_class"),
             ("mimic", "medication_admin_intermittent", "06-Insulin (Non IV); Main order parameter", "im", "reported_class"),
             ("mimic", "medication_admin_intermittent", "SC", "im", "reported"),
             ("sicdb", "medication_admin_continuous", None, "iv", "assumed"),
             ("sicdb", "medication_admin_intermittent", None, None, None),       # no route at all
             ("aumc", "medication_admin_continuous", "regional", None, None)]    # regional anaesthesia
    for ds, table, name, category, basis in cases:
        sql = route_basis_sql(ds, table, "n", "c")
        got = con.execute(f"SELECT {sql} FROM (SELECT CAST(? AS VARCHAR) AS n, CAST(? AS VARCHAR) AS c)",
                          [name, category]).fetchone()[0]
        assert got == basis, (ds, table, name)
    assert route_basis_sql("hirid", "medication_admin_intermittent", "n", "c").count("verified") == 0
