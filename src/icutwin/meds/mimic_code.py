"""Parity of the MIMIC-IV derived concepts with mimic-code (``refs/mimic-code``, the commit pinned in
``refs/REFS.lock``).

mimic-code's DuckDB concepts (``mimic-iv/concepts_duckdb``, generated from its BigQuery SQL) run unchanged on the
MIMIC parquet mirror (schemas ``mimiciv_hosp`` / ``mimiciv_icu`` as views), in the order of its ``duckdb.sql``.
Two comparisons:

* **logic parity**: our implementations (``icutwin.meds.derived``) run on mimic-code's own inputs, one
  hospitalization per ICU stay: creatinine (``kdigo_creatinine``), urine output (``urine_output``), weights
  (``weight_durations``), CRRT modes (``crrt``) -> KDIGO stages; the charted ventilation status of
  ``ventilation`` (its ``vs`` step) -> ventilation episodes; ``vasoactive_agent`` rates -> norepinephrine
  equivalents. They must reproduce ``kdigo_stages``, ``ventilation`` and ``norepinephrine_equivalent_dose``.
* **end-to-end agreement**: our concepts from canonical CLIF (CLIF-MIMIC + supplements, ``data/derived``)
  against mimic-code's from raw MIMIC, per ICU stay of both: AKI stage, creatinine, first-day urine output and
  weight, vasopressors and maximum norepinephrine equivalent, ventilation, renal replacement, first-day SOFA.

The report (``mimic_code_parity.{json,md}``) holds aggregates only.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import duckdb

from icutwin.clif.conform import connect
from icutwin.meds import derived as dv

log = logging.getLogger(__name__)

CONCEPTS = Path("mimic-iv/concepts_duckdb")
KDIGO_COLS = ("creat", "creat_low_past_48hr", "creat_low_past_7day", "aki_stage_creat", "uo_rt_6hr", "uo_rt_12hr",
              "uo_rt_24hr", "aki_stage_uo", "aki_stage_crrt", "aki_stage", "aki_stage_smoothed")
NEE_DRUGS = {"norepinephrine": "mcg/kg/min", "epinephrine": "mcg/kg/min", "phenylephrine": "mcg/kg/min",
             "dopamine": "mcg/kg/min", "vasopressin": "u/hr"}   # vasoactive_agent units
VENT_STATUS = {"InvasiveVent": "imv", "Tracheostomy": "tracheostomy", "NonInvasiveVent": "nippv", "HFNC": "hfnc",
               "SupplementalOxygen": "supplemental_oxygen", "None": "none"}


def _p(path: Path) -> str:
    return str(path).replace("\\", "/")


# -- mimic-code ---------------------------------------------------------------------------------------
def run_concepts(mirror: Path, mimic_code: Path, db: Path) -> tuple[duckdb.DuckDBPyConnection, dict[str, int]]:
    """Run every mimic-code DuckDB concept on the MIMIC parquet mirror into ``db`` (schema ``mimiciv_derived``)."""
    concepts = mimic_code / CONCEPTS
    db.parent.mkdir(parents=True, exist_ok=True)
    for f in (db, db.with_name(db.name + ".wal")):
        f.unlink(missing_ok=True)
    con = duckdb.connect(str(db))
    for schema, sub in (("mimiciv_hosp", "hosp"), ("mimiciv_icu", "icu")):
        con.execute(f"CREATE SCHEMA {schema}")
        for f in sorted((mirror / sub).glob("*.parquet")):
            con.execute(f"CREATE VIEW {schema}.{f.stem} AS SELECT * FROM read_parquet('{_p(f)}')")
    con.execute("CREATE SCHEMA mimiciv_derived")
    rows = {}
    for rel in re.findall(r"^\.read (\S+)", (concepts / "duckdb.sql").read_text(encoding="utf-8"), flags=re.M):
        con.execute((concepts / rel).read_text(encoding="utf-8"))
        name = Path(rel).stem
        rows[name] = con.execute(f"SELECT count(*) FROM mimiciv_derived.{name}").fetchone()[0]
    # sensitivity: oxygen_delivery filters its FULL JOIN on the O2-flow side (``WHERE ce.rn = 1``), which drops every
    # device charting without an O2 flow at the same time (nearly all endotracheal-tube rows); the same two
    # concepts with only that filter widened, as ``*_fixed``
    od = (concepts / "measurement" / "oxygen_delivery.sql").read_text(encoding="utf-8")
    vent = (concepts / "treatment" / "ventilation.sql").read_text(encoding="utf-8")
    flow_filter = "WHERE\n    ce.rn = 1\n)"
    if od.count(flow_filter) != 1:
        raise ValueError("mimic-code oxygen_delivery changed: review the sensitivity patch")
    con.execute(od.replace(flow_filter, "WHERE\n    ce.rn = 1 OR ce.rn IS NULL\n)")
                  .replace("mimiciv_derived.oxygen_delivery", "mimiciv_derived.oxygen_delivery_fixed"))
    con.execute(vent.replace("mimiciv_derived.oxygen_delivery", "mimiciv_derived.oxygen_delivery_fixed")
                    .replace("mimiciv_derived.ventilation", "mimiciv_derived.ventilation_fixed"))
    for name in ("oxygen_delivery_fixed", "ventilation_fixed"):
        rows[name] = con.execute(f"SELECT count(*) FROM mimiciv_derived.{name}").fetchone()[0]
    return con, rows


def ventilation_status_sql(mimic_code: Path) -> str:
    """mimic-code ``ventilation`` up to its per-time status (``vs``): (stay_id, charttime, ventilation_status)."""
    text = (mimic_code / CONCEPTS / "treatment" / "ventilation.sql").read_text(encoding="utf-8")
    start, end = text.index("WITH tm AS ("), text.index(", vd0 AS (")
    return text[start:end] + "\nSELECT stay_id, charttime, ventilation_status FROM vs WHERE ventilation_status IS NOT NULL"


# -- metrics ------------------------------------------------------------------------------------------
def _binary(con: duckdb.DuckDBPyConnection, sql: str) -> dict:
    """``sql`` yields (ours BOOLEAN, theirs BOOLEAN) per unit: counts, agreement and Cohen's kappa."""
    a, b, c, d = con.execute(f"""SELECT count(*) FILTER (WHERE ours AND theirs), count(*) FILTER (WHERE ours AND NOT theirs),
                                        count(*) FILTER (WHERE NOT ours AND theirs), count(*) FILTER (WHERE NOT ours AND NOT theirs)
                                 FROM ({sql})""").fetchone()
    n = a + b + c + d
    po = (a + d) / n if n else math.nan
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2 if n else math.nan
    kappa = (po - pe) / (1 - pe) if n and pe < 1 else math.nan
    return {"n": n, "both": a, "only_ours": b, "only_mimic_code": c, "neither": d,
            "agreement": round(po, 4), "kappa": round(kappa, 4)}


def _ordinal(con: duckdb.DuckDBPyConnection, sql: str) -> dict:
    """``sql`` yields (ours, theirs) numbers per unit: exact and within-one agreement, mean difference, rank r."""
    r = con.execute(f"""SELECT count(*), avg(CAST(ours = theirs AS DOUBLE)), avg(CAST(abs(ours - theirs) <= 1 AS DOUBLE)),
                               avg(ours - theirs), corr(rk_o, rk_t)
                        FROM (SELECT *, rank() OVER (ORDER BY ours) AS rk_o, rank() OVER (ORDER BY theirs) AS rk_t
                              FROM ({sql}) WHERE ours IS NOT NULL AND theirs IS NOT NULL)""").fetchone()
    confusion = con.execute(f"""SELECT ours, theirs, count(*) FROM ({sql}) WHERE ours IS NOT NULL AND theirs IS NOT NULL
                                GROUP BY 1, 2 ORDER BY 1, 2""").fetchall()
    return {"n": r[0], "exact": _r(r[1]), "within_1": _r(r[2]), "mean_diff": _r(r[3]), "spearman": _r(r[4]),
            "confusion": [list(map(lambda x: x if isinstance(x, int) else float(x), row)) for row in confusion]}


def _continuous(con: duckdb.DuckDBPyConnection, sql: str, tol: float) -> dict:
    """``sql`` yields (ours, theirs) numbers per unit: correlation, median absolute difference, share within ``tol``."""
    r = con.execute(f"""SELECT count(*), corr(ours, theirs), median(abs(ours - theirs)),
                               avg(CAST(abs(ours - theirs) <= {tol} AS DOUBLE)), median(theirs), median(ours), corr(rk_o, rk_t)
                        FROM (SELECT *, rank() OVER (ORDER BY ours) AS rk_o, rank() OVER (ORDER BY theirs) AS rk_t
                              FROM ({sql}) WHERE ours IS NOT NULL AND theirs IS NOT NULL)""").fetchone()
    missing = con.execute(f"""SELECT count(*) FILTER (WHERE ours IS NULL AND theirs IS NOT NULL),
                                     count(*) FILTER (WHERE theirs IS NULL AND ours IS NOT NULL) FROM ({sql})""").fetchone()
    return {"n": r[0], "pearson": _r(r[1]), "spearman": _r(r[6]), "median_abs_diff": _r(r[2]), f"within_{tol}": _r(r[3]),
            "median_mimic_code": _r(r[4]), "median_ours": _r(r[5]), "only_mimic_code": missing[0], "only_ours": missing[1]}


def _r(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


# -- logic parity -------------------------------------------------------------------------------------
def _logic_connection(db: Path) -> duckdb.DuckDBPyConnection:
    con = connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"ATTACH '{_p(db)}' AS mc (READ_ONLY)")
    con.execute("""CREATE OR REPLACE TEMP TABLE stays AS
                   SELECT CAST(stay_id AS VARCHAR) AS stay_id, CAST(stay_id AS VARCHAR) AS hospitalization_id,
                          intime AS icu_in, outtime AS icu_out
                   FROM mc.mimiciv_icu.icustays WHERE outtime > intime""")
    return con


def logic_kdigo(db: Path) -> dict:
    """Our KDIGO on mimic-code's creatinine, urine output, weights and CRRT modes against ``kdigo_stages``."""
    con = _logic_connection(db)
    con.execute("""CREATE OR REPLACE TEMP VIEW labs AS
                   SELECT CAST(stay_id AS VARCHAR) AS hospitalization_id, charttime AS lab_collect_dttm,
                          'creatinine' AS lab_category, creat AS lab_value_numeric
                   FROM mc.mimiciv_derived.kdigo_creatinine WHERE charttime IS NOT NULL""")
    con.execute("""CREATE OR REPLACE TEMP VIEW output AS
                   SELECT CAST(stay_id AS VARCHAR) AS hospitalization_id, charttime AS recorded_dttm,
                          'urine' AS output_group, urineoutput AS output_volume
                   FROM mc.mimiciv_derived.urine_output""")
    con.execute("""CREATE OR REPLACE TEMP VIEW vitals AS
                   SELECT CAST(stay_id AS VARCHAR) AS hospitalization_id, starttime AS recorded_dttm,
                          'weight_kg' AS vital_category, weight AS vital_value
                   FROM mc.mimiciv_derived.weight_durations""")
    con.execute("""CREATE OR REPLACE TEMP VIEW crrt_therapy AS
                   SELECT CAST(stay_id AS VARCHAR) AS hospitalization_id, charttime AS recorded_dttm
                   FROM mc.mimiciv_derived.crrt WHERE crrt_mode IS NOT NULL""")
    dv._weights(con)
    dv.kdigo(con, {"labs", "output", "crrt_therapy"})
    # urine output charted outside the ICU stay window: mimic-code keeps the stay's rows whatever their time
    outside = con.execute("""SELECT count(*) FROM mc.mimiciv_derived.urine_output u JOIN stays s
                             ON s.stay_id = CAST(u.stay_id AS VARCHAR) WHERE u.charttime NOT BETWEEN s.icu_in AND s.icu_out""").fetchone()[0]
    multi = """(SELECT CAST(stay_id AS VARCHAR) FROM mc.mimiciv_icu.icustays WHERE subject_id IN (
                   SELECT subject_id FROM mc.mimiciv_icu.icustays GROUP BY 1 HAVING count(*) > 1))"""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE kj AS
                    SELECT coalesce(m.stay_id, k.stay_id) AS stay_id, coalesce(m.t, k.t) AS t, m.stay_id AS m_stay,
                           k.stay_id AS k_stay, m.* EXCLUDE (stay_id, t), k.* EXCLUDE (stay_id, t)
                    FROM (SELECT CAST(stay_id AS VARCHAR) AS stay_id, charttime AS t, {', '.join(f'{c} AS m_{c}' for c in KDIGO_COLS)}
                          FROM mc.mimiciv_derived.kdigo_stages WHERE charttime IS NOT NULL) m
                    FULL JOIN kdigo k ON k.stay_id = m.stay_id AND k.t = m.t""")
    out: dict = {"mimic_code_rows": con.execute("SELECT count(*) FROM kj WHERE m_stay IS NOT NULL").fetchone()[0],
                 "rows_only_mimic_code": con.execute("SELECT count(*) FROM kj WHERE k_stay IS NULL").fetchone()[0],
                 "rows_only_ours": con.execute("SELECT count(*) FROM kj WHERE m_stay IS NULL").fetchone()[0],
                 "urine_rows_outside_stay_window": outside, "columns": {}}
    outside_stays = """(SELECT DISTINCT CAST(u.stay_id AS VARCHAR) FROM mc.mimiciv_derived.urine_output u JOIN stays s
                         ON s.stay_id = CAST(u.stay_id AS VARCHAR) WHERE u.charttime NOT BETWEEN s.icu_in AND s.icu_out)"""
    out["stays_with_urine_outside_window"] = con.execute(f"SELECT count(*) FROM {outside_stays}").fetchone()[0]
    for c in KDIGO_COLS:
        ours = "coalesce(aki_stage_crrt, 0)" if c == "aki_stage_crrt" else c
        theirs = f"coalesce(m_{c}, 0)" if c == "aki_stage_crrt" else f"m_{c}"
        scope = f"AND stay_id NOT IN {multi}" if c == "aki_stage_smoothed" else ""   # mimic-code smooths per patient
        same = f"({ours}) IS NOT DISTINCT FROM ({theirs}) OR abs(({ours}) - ({theirs})) <= 1e-4"
        n, k, n_in, k_in = con.execute(f"""
            SELECT count(*), count(*) FILTER (WHERE {same}),
                   count(*) FILTER (WHERE stay_id NOT IN {outside_stays}),
                   count(*) FILTER (WHERE ({same}) AND stay_id NOT IN {outside_stays})
            FROM kj WHERE k_stay IS NOT NULL AND m_stay IS NOT NULL {scope}""").fetchone()
        out["columns"][c] = {"compared": n, "identical": k, "share": _r(k / n if n else None),
                             "share_stays_without_outside_urine": _r(k_in / n_in if n_in else None)}
    return out


def logic_ventilation(db: Path, mimic_code: Path) -> dict:
    """Our episode rule on mimic-code's charted statuses against ``ventilation`` (stay, status, start, end)."""
    con = _logic_connection(db)
    status = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in VENT_STATUS.items())
    con.execute(f"""CREATE OR REPLACE TEMP TABLE resp_status AS
                    SELECT CAST(stay_id AS VARCHAR) AS hospitalization_id, charttime AS t,
                           CASE ventilation_status {status} END AS status
                    FROM ({ventilation_status_sql(mimic_code).replace('mimiciv_derived.', 'mc.mimiciv_derived.')})""")
    con.execute(f"CREATE OR REPLACE TEMP TABLE ours AS {dv.respiratory_episodes_sql()}")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE theirs AS
                    SELECT CAST(stay_id AS VARCHAR) AS hospitalization_id, CASE ventilation_status {status} END AS episode,
                           starttime AS start_dttm, endtime AS end_dttm FROM mc.mimiciv_derived.ventilation""")
    key = "hospitalization_id, episode, start_dttm, end_dttm"
    n_t, n_o = (con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("theirs", "ours"))
    only_t = con.execute(f"SELECT count(*) FROM (SELECT {key} FROM theirs EXCEPT ALL SELECT {key} FROM ours)").fetchone()[0]
    only_o = con.execute(f"SELECT count(*) FROM (SELECT {key} FROM ours EXCEPT ALL SELECT {key} FROM theirs)").fetchone()[0]
    return {"mimic_code_episodes": n_t, "our_episodes": n_o, "only_mimic_code": only_t, "only_ours": only_o,
            "identical": only_t == 0 and only_o == 0}


def logic_nee(db: Path) -> dict:
    """Our norepinephrine equivalents on ``vasoactive_agent`` rates against ``norepinephrine_equivalent_dose``."""
    con = _logic_connection(db)
    rows = " UNION ALL ".join(
        f"""SELECT CAST(stay_id AS VARCHAR) AS hospitalization_id, starttime AS admin_dttm, '{d}' AS med_category,
                   coalesce({d}, 0) AS med_dose, '{u}' AS med_dose_unit FROM mc.mimiciv_derived.vasoactive_agent"""
        for d, u in NEE_DRUGS.items())
    con.execute(f"CREATE OR REPLACE TEMP VIEW medication_admin_continuous AS {rows}")
    con.execute("CREATE OR REPLACE TEMP TABLE weights (hospitalization_id VARCHAR, t TIMESTAMP, kg DOUBLE)")
    con.execute("CREATE OR REPLACE TEMP TABLE first_weight (hospitalization_id VARCHAR, kg DOUBLE)")
    dv.nee(con)
    # vasoactive_agent repeats an interval once per concurrent order of a drug (two norepinephrine orders at
    # 0.04 and 0.12 give two rows, not 0.16): those intervals have no single mimic-code value to compare
    dup = """(SELECT stay_id, starttime FROM mc.mimiciv_derived.vasoactive_agent GROUP BY 1, 2 HAVING count(*) > 1)"""
    n, same, worst = con.execute(f"""
        SELECT count(*), count(*) FILTER (WHERE abs(n.nee_mcg_kg_min - m.norepinephrine_equivalent_dose) <= 1e-4),
               max(abs(n.nee_mcg_kg_min - m.norepinephrine_equivalent_dose))
        FROM mc.mimiciv_derived.norepinephrine_equivalent_dose m
        LEFT JOIN nee n ON n.hospitalization_id = CAST(m.stay_id AS VARCHAR) AND n.t = m.starttime
        WHERE (m.stay_id, m.starttime) NOT IN {dup}""").fetchone()
    n_dup = con.execute(f"SELECT count(*) FROM {dup}").fetchone()[0]
    return {"intervals": n, "identical": same, "share": _r(same / n if n else None), "max_abs_diff": _r(worst),
            "intervals_repeated_per_concurrent_order": n_dup}


# -- end-to-end agreement -----------------------------------------------------------------------------
def agreement(db: Path, clif_dir: Path, derived_dir: Path) -> dict:
    """Our concepts from canonical CLIF against mimic-code's from raw MIMIC, per ICU stay present in both."""
    con = connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"ATTACH '{_p(db)}' AS mc (READ_ONLY)")
    d = lambda name: f"read_parquet('{_p(derived_dir / f'{name}.parquet')}')"  # noqa: E731
    c = lambda name: f"read_parquet('{_p(clif_dir / f'clif_{name}.parquet')}')"  # noqa: E731
    con.execute(f"""CREATE OR REPLACE TEMP TABLE st AS
                    SELECT e.stay_id, e.hospitalization_id, CAST(i.stay_id AS BIGINT) AS mc_stay,
                           CAST(e.icu_in_dttm AS TIMESTAMP) AS icu_in, CAST(e.icu_out_dttm AS TIMESTAMP) AS icu_out
                    FROM {c('ext_icu_stay')} e JOIN mc.mimiciv_icu.icustays i ON e.stay_id = 'mimic:' || CAST(i.stay_id AS VARCHAR)
                    WHERE e.icu_out_dttm > e.icu_in_dttm""")
    out: dict = {"stays": con.execute("SELECT count(*) FROM st").fetchone()[0],
                 "mimic_code_icustays": con.execute("SELECT count(*) FROM mc.mimiciv_icu.icustays").fetchone()[0]}
    # AKI: highest stage per stay (all of its times, as both tables hold them), and by criterion
    for col in ("aki_stage", "aki_stage_creat", "aki_stage_uo"):
        out[f"max_{col}"] = _ordinal(con, f"""
            SELECT coalesce(o.v, 0) AS ours, coalesce(m.v, 0) AS theirs FROM st
            LEFT JOIN (SELECT stay_id, max({col}) AS v FROM {d('kdigo')} GROUP BY 1) o USING (stay_id)
            LEFT JOIN (SELECT stay_id, max({col}) AS v FROM mc.mimiciv_derived.kdigo_stages GROUP BY 1) m ON m.stay_id = st.mc_stay""")
    out["creatinine_at_mimic_code_times"] = _continuous(con, f"""
        SELECT o.creat AS ours, m.creat AS theirs FROM mc.mimiciv_derived.kdigo_creatinine m JOIN st ON st.mc_stay = m.stay_id
        LEFT JOIN {d('kdigo')} o ON o.stay_id = st.stay_id AND o.t = m.charttime AND o.creat IS NOT NULL
        WHERE m.charttime IS NOT NULL""", 0.005)
    # first-day urine output (urine minus bladder irrigation inflow, mimic-code urine_output) and weight
    con.execute(f"""CREATE OR REPLACE TEMP TABLE uo AS
                    SELECT st.stay_id, sum(f.ml) AS ml FROM st JOIN (
                        SELECT hospitalization_id, CAST(recorded_dttm AS TIMESTAMP) AS t, output_volume AS ml FROM {c('output')}
                        WHERE output_group = 'urine' AND output_volume IS NOT NULL
                        UNION ALL SELECT hospitalization_id, CAST(recorded_dttm AS TIMESTAMP), -input_volume FROM {c('input')}
                        WHERE input_category = 'flush_irrigation_urine' AND input_volume > 0) f
                      ON f.hospitalization_id = st.hospitalization_id AND f.t BETWEEN st.icu_in AND st.icu_in + INTERVAL 1 DAY
                    GROUP BY 1""")
    out["first_day_urine_output_ml"] = _continuous(con, """
        SELECT u.ml AS ours, m.urineoutput AS theirs FROM st LEFT JOIN uo u USING (stay_id)
        LEFT JOIN mc.mimiciv_derived.first_day_urine_output m ON m.stay_id = st.mc_stay""", 1.0)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE wt AS
                    SELECT hospitalization_id, CAST(recorded_dttm AS TIMESTAMP) AS t, vital_value AS kg
                    FROM {c('vitals')} WHERE vital_category = 'weight_kg' AND vital_value > 0""")
    out["admission_weight_kg"] = _continuous(con, """
        SELECT coalesce((SELECT arg_max(w.kg, w.t) FROM wt w WHERE w.hospitalization_id = st.hospitalization_id AND w.t <= st.icu_in),
                        (SELECT arg_min(w.kg, w.t) FROM wt w WHERE w.hospitalization_id = st.hospitalization_id)) AS ours,
               m.weight_admit AS theirs
        FROM st LEFT JOIN mc.mimiciv_derived.first_day_weight m ON m.stay_id = st.mc_stay""", 1.0)
    # vasopressors: any during the stay and the highest norepinephrine equivalent
    con.execute(f"""CREATE OR REPLACE TEMP TABLE nee_o AS   -- in effect during the stay: changes inside it and
                    WITH n AS (SELECT * FROM {d('nee')}),         -- the rate running at ICU admission
                    inside AS (SELECT st.stay_id, n.nee_mcg_kg_min AS v FROM st JOIN n
                               ON n.hospitalization_id = st.hospitalization_id AND n.t >= st.icu_in AND n.t < st.icu_out),
                    at_in AS (SELECT st.stay_id, arg_max(n.nee_mcg_kg_min, n.t) AS v FROM st JOIN n
                              ON n.hospitalization_id = st.hospitalization_id AND n.t < st.icu_in GROUP BY 1)
                    SELECT stay_id, max(v) AS mx FROM (SELECT * FROM inside UNION ALL SELECT * FROM at_in) GROUP BY 1""")
    con.execute("""CREATE OR REPLACE TEMP TABLE nee_m AS
                   SELECT stay_id, max(norepinephrine_equivalent_dose) AS mx FROM mc.mimiciv_derived.norepinephrine_equivalent_dose
                   GROUP BY 1""")
    out["vasopressor_any"] = _binary(con, """SELECT coalesce(o.mx, 0) > 0 AS ours, coalesce(m.mx, 0) > 0 AS theirs
                                             FROM st LEFT JOIN nee_o o USING (stay_id) LEFT JOIN nee_m m ON m.stay_id = st.mc_stay""")
    out["max_nee_mcg_kg_min"] = _continuous(con, """SELECT o.mx AS ours, m.mx AS theirs FROM st
                                                    JOIN nee_o o USING (stay_id) JOIN nee_m m ON m.stay_id = st.mc_stay""", 0.01)
    # ventilation, renal replacement: episodes overlapping the ICU stay
    ep = d("episodes")
    for name, ours, theirs in (("invasive_ventilation", "'imv'", "('InvasiveVent')"),
                               ("invasive_ventilation_incl_tracheostomy", "'imv'", "('InvasiveVent', 'Tracheostomy')"),
                               ("noninvasive_ventilation", "'nippv', 'cpap'", "('NonInvasiveVent')"),
                               ("high_flow_nasal_cannula", "'hfnc'", "('HFNC')")):
        out[name] = _binary(con, f"""
            SELECT EXISTS (SELECT 1 FROM {ep} e WHERE e.stay_id = st.stay_id AND e.episode IN ({ours})
                           AND e.start_dttm < st.icu_out AND e.end_dttm > st.icu_in) AS ours,
                   EXISTS (SELECT 1 FROM mc.mimiciv_derived.ventilation v WHERE v.stay_id = st.mc_stay
                           AND v.ventilation_status IN {theirs}) AS theirs
            FROM st""")
    out["invasive_ventilation_vs_join_fixed"] = _binary(con, f"""
        SELECT EXISTS (SELECT 1 FROM {ep} e WHERE e.stay_id = st.stay_id AND e.episode = 'imv'
                       AND e.start_dttm < st.icu_out AND e.end_dttm > st.icu_in) AS ours,
               EXISTS (SELECT 1 FROM mc.mimiciv_derived.ventilation_fixed v WHERE v.stay_id = st.mc_stay
                       AND v.ventilation_status = 'InvasiveVent') AS theirs
        FROM st""")
    out["invasive_ventilation_hours_vs_join_fixed"] = _continuous(con, f"""
        SELECT (SELECT sum(epoch(least(e.end_dttm, st.icu_out) - greatest(e.start_dttm, st.icu_in))) / 3600 FROM {ep} e
                WHERE e.stay_id = st.stay_id AND e.episode = 'imv' AND e.start_dttm < st.icu_out AND e.end_dttm > st.icu_in) AS ours,
               (SELECT sum(epoch(v.endtime - v.starttime)) / 3600 FROM mc.mimiciv_derived.ventilation_fixed v
                WHERE v.stay_id = st.mc_stay AND v.ventilation_status = 'InvasiveVent') AS theirs
        FROM st""", 1.0)
    out["noninvasive_ventilation_vs_join_fixed"] = _binary(con, f"""
        SELECT EXISTS (SELECT 1 FROM {ep} e WHERE e.stay_id = st.stay_id AND e.episode IN ('nippv', 'cpap')
                       AND e.start_dttm < st.icu_out AND e.end_dttm > st.icu_in) AS ours,
               EXISTS (SELECT 1 FROM mc.mimiciv_derived.ventilation_fixed v WHERE v.stay_id = st.mc_stay
                       AND v.ventilation_status = 'NonInvasiveVent') AS theirs
        FROM st""")
    out["invasive_ventilation_hours"] = _continuous(con, f"""
        SELECT (SELECT sum(epoch(least(e.end_dttm, st.icu_out) - greatest(e.start_dttm, st.icu_in))) / 3600 FROM {ep} e
                WHERE e.stay_id = st.stay_id AND e.episode = 'imv' AND e.start_dttm < st.icu_out AND e.end_dttm > st.icu_in) AS ours,
               (SELECT sum(epoch(v.endtime - v.starttime)) / 3600 FROM mc.mimiciv_derived.ventilation v
                WHERE v.stay_id = st.mc_stay AND v.ventilation_status = 'InvasiveVent') AS theirs
        FROM st""", 1.0)
    out["notes"] = notes(con, ep)
    out["renal_replacement"] = _binary(con, f"""
        SELECT EXISTS (SELECT 1 FROM {ep} e WHERE e.stay_id = st.stay_id AND e.episode = 'rrt'
                       AND e.start_dttm <= st.icu_out AND e.end_dttm >= st.icu_in) AS ours,
               EXISTS (SELECT 1 FROM mc.mimiciv_derived.rrt r WHERE r.stay_id = st.mc_stay AND r.dialysis_active = 1) AS theirs
        FROM st""")
    # first-day SOFA: clifpy (ours) against mimic-code first_day_sofa
    sofa = {"sofa_total": "sofa", "sofa_resp": "respiration", "sofa_coag": "coagulation", "sofa_liver": "liver",
            "sofa_cv_97": "cardiovascular", "sofa_cns": "cns", "sofa_renal": "renal"}
    for ours, theirs in sofa.items():
        out[f"first_day_{ours}"] = _ordinal(con, f"""
            SELECT o.{ours} AS ours, m.{theirs} AS theirs FROM st JOIN {d('sofa_24h')} o USING (stay_id)
            JOIN mc.mimiciv_derived.first_day_sofa m ON m.stay_id = st.mc_stay""")
    return out


def notes(con: duckdb.DuckDBPyConnection, ep: str) -> dict:
    """The size of the known causes of disagreement (mimic-code limitations and documented deviations)."""
    q = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
    return {
        "endotracheal_tube_chartings": q(
            "SELECT count(*) FROM mc.mimiciv_icu.chartevents WHERE itemid = 226732 AND value = 'Endotracheal tube'"),
        "endotracheal_tube_chartings_kept_by_oxygen_delivery": q("""
            SELECT count(*) FROM mc.mimiciv_derived.oxygen_delivery WHERE 'Endotracheal tube' IN
            (o2_delivery_device_1, o2_delivery_device_2, o2_delivery_device_3, o2_delivery_device_4)"""),
        "hamilton_spont_chartings_not_in_invasive_list": q(
            "SELECT count(*) FROM mc.mimiciv_derived.ventilator_setting WHERE ventilator_mode_hamilton = 'SPONT'"),
        "stays_with_phenylephrine_only_from_items_229630_229632": q("""
            SELECT count(DISTINCT stay_id) FROM mc.mimiciv_icu.inputevents WHERE itemid IN (229630, 229632)
            AND stay_id NOT IN (SELECT stay_id FROM mc.mimiciv_derived.vasoactive_agent
                                WHERE coalesce(norepinephrine, epinephrine, phenylephrine, dopamine, vasopressin) IS NOT NULL)"""),
        "stays_with_mimic_code_nee_above_3": q("""
            SELECT count(DISTINCT stay_id) FROM mc.mimiciv_derived.norepinephrine_equivalent_dose
            WHERE norepinephrine_equivalent_dose > 3"""),
        "stays_with_peritoneal_dialysis": q(
            "SELECT count(DISTINCT stay_id) FROM mc.mimiciv_derived.rrt WHERE dialysis_type = 'Peritoneal'"),
        "our_niv_stays_with_mimic_code_imv_at_those_times": q(f"""
            SELECT count(DISTINCT st.stay_id) FROM st JOIN {ep} e ON e.stay_id = st.stay_id AND e.episode IN ('nippv', 'cpap')
            AND e.start_dttm < st.icu_out AND e.end_dttm > st.icu_in
            WHERE NOT EXISTS (SELECT 1 FROM mc.mimiciv_derived.ventilation_fixed v WHERE v.stay_id = st.mc_stay
                              AND v.ventilation_status = 'NonInvasiveVent')
              AND EXISTS (SELECT 1 FROM mc.mimiciv_derived.ventilation_fixed v WHERE v.stay_id = st.mc_stay
                          AND v.ventilation_status = 'InvasiveVent' AND v.starttime < e.end_dttm AND v.endtime > e.start_dttm)"""),
    }


# -- first-day SOFA, component by component --------------------------------------------------------------
SOFA_COMPONENTS = {"sofa_resp": "respiration", "sofa_coag": "coagulation", "sofa_liver": "liver",
                   "sofa_cv_97": "cardiovascular", "sofa_cns": "cns", "sofa_renal": "renal"}


def first_day_sofa_inputs_sql(mimic_code: Path) -> str:
    """mimic-code ``first_day_sofa`` up to its component inputs (``scorecomp``): stay_id, mbp_min, vasopressor
    rates, creatinine_max, bilirubin_max, platelet_min, pao2fio2_novent_min, pao2fio2_vent_min, urineoutput,
    gcs_min (6 h before to 24 h after ICU admission; urine output from admission)."""
    text = (mimic_code / CONCEPTS / "firstday" / "first_day_sofa.sql").read_text(encoding="utf-8")
    start, end = text.index("WITH vaso_stg AS ("), text.index(", scorecalc AS (")
    return text[start:end] + "\nSELECT * FROM scorecomp"


def _creat_score(x: str) -> str:
    """SOFA renal score of the highest creatinine (mg/dL); clifpy and mimic-code share the thresholds."""
    return f"CASE WHEN {x} >= 5 THEN 4 WHEN {x} >= 3.5 THEN 3 WHEN {x} >= 2 THEN 2 WHEN {x} >= 1.2 THEN 1 " \
           f"WHEN {x} IS NULL THEN NULL ELSE 0 END"


def _uo_score(x: str) -> str:
    """mimic-code's urine-output criterion: first-day urine < 200 mL -> 4, < 500 mL -> 3 (clifpy has none)."""
    return f"CASE WHEN {x} < 200 THEN 4 WHEN {x} < 500 THEN 3 WHEN {x} IS NULL THEN NULL ELSE 0 END"


def _mc_resp(vent: str, novent: str) -> str:
    """mimic-code's respiration rule (``first_day_sofa``): scores 3-4 from P/F gases during invasive ventilation,
    1-2 from gases off it; a ventilated P/F of 200-400 matches no branch and scores 0 (mimic-code's hourly
    ``sofa`` has the two missing branches)."""
    return (f"CASE WHEN {vent} < 100 THEN 4 WHEN {vent} < 200 THEN 3 WHEN {novent} < 300 THEN 2 WHEN {novent} < 400 "
            f"THEN 1 WHEN coalesce({vent}, {novent}) IS NULL THEN NULL ELSE 0 END")


def _mc_resp_hourly(vent: str, novent: str) -> str:
    """mimic-code's hourly ``sofa`` respiration rule, which scores a ventilated P/F of 200-400 as SOFA does."""
    return (f"CASE WHEN {vent} < 100 THEN 4 WHEN {vent} < 200 THEN 3 WHEN {novent} < 300 OR {vent} < 300 THEN 2 "
            f"WHEN {novent} < 400 OR {vent} < 400 THEN 1 WHEN coalesce({vent}, {novent}) IS NULL THEN NULL ELSE 0 END")


def _attribute(con: duckdb.DuckDBPyConnection, where: str, causes: list[tuple[str, str]]) -> dict:
    """Disagreeing stays (``where``) by the first cause (name, SQL predicate) that holds; 'unexplained' else."""
    case = "CASE " + " ".join(f"WHEN {pred} THEN '{name.replace(chr(39), chr(39) * 2)}'" for name, pred in causes) \
           + " ELSE 'unexplained' END"
    rows = con.execute(f"SELECT {case} AS cause, count(*) FROM j WHERE {where} GROUP BY 1 ORDER BY 2 DESC").fetchall()
    return {"stays": sum(n for _, n in rows), "by_cause": {c: n for c, n in rows}}


def sofa_decomposition(db: Path, mimic_code: Path, clif_dir: Path, work_dir: Path) -> dict:
    """First-day SOFA, ours (clifpy on canonical CLIF, respiration scored as SOFA defines it) against mimic-code
    ``first_day_sofa``, per component.

    The two are different definitions, not one score computed twice: windows (ours: ICU admission to +24 h over
    the hospitalization's records, labs by result time; mimic-code: from 6 h before, vitals of this ICU stay,
    labs by specimen time), renal (clifpy: creatinine only; mimic-code: creatinine or first-day urine output),
    respiration (clifpy: worst P/F paired with the FiO2 setting charted within 4 h and the device at that time,
    IMV/NIPPV/CPAP counting as support; mimic-code: P/F of its blood-gas concept, FiO2 of the gas panel first,
    support = invasive ventilation at the gas time), MAP (ours: hourly median per blood-pressure site;
    mimic-code: every reading, arterial and cuff readings of one time averaged) and GCS of intubated patients.
    Each disagreeing stay is attributed to the first cause that explains it: the window, then definition
    differences and implementation defects, then input differences; 'unexplained' when none holds."""
    con = connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"ATTACH '{_p(db)}' AS mc (READ_ONLY)")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE st AS
                    SELECT e.stay_id, e.hospitalization_id, CAST(i.stay_id AS BIGINT) AS mc_stay,
                           CAST(e.icu_in_dttm AS TIMESTAMP) AS icu_in
                    FROM read_parquet('{_p(clif_dir / 'clif_ext_icu_stay.parquet')}') e
                    JOIN mc.mimiciv_icu.icustays i ON e.stay_id = 'mimic:' || CAST(i.stay_id AS VARCHAR)
                    WHERE e.icu_out_dttm > e.icu_in_dttm""")
    inputs = first_day_sofa_inputs_sql(mimic_code).replace("mimiciv_derived.", "mc.mimiciv_derived.") \
                                                   .replace("mimiciv_icu.", "mc.mimiciv_icu.")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE mcs AS
                    SELECT s.stay_id, s.sofa, s.respiration, s.coagulation, s.liver, s.cardiovascular, s.cns, s.renal,
                           i.* EXCLUDE (stay_id)
                    FROM mc.mimiciv_derived.first_day_sofa s JOIN ({inputs}) i USING (stay_id)""")
    for name, pre, collect in (("o0", 0.0, False), ("o6", 6.0, False), ("o6c", 6.0, True)):
        frame = dv.sofa_window(clif_dir, work_dir, pre_hours=pre, keep_inputs=True, labs_at_collection=collect).to_arrow()
        con.register(f"{name}_arrow", frame)
        con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS SELECT * FROM {name}_arrow")
    # the lowest single MAP reading of mimic-code's items and window: of this ICU stay (mimic-code's filter) and of
    # the whole hospitalization (CLIF's range 0-250)
    con.execute("""CREATE OR REPLACE TEMP TABLE map_raw AS
                   SELECT st.stay_id,
                          min(ce.valuenum) FILTER (WHERE ce.stay_id = st.mc_stay AND ce.valuenum > 0 AND ce.valuenum < 300) AS map_stay,
                          min(ce.valuenum) FILTER (WHERE ce.valuenum BETWEEN 0 AND 250) AS map_hosp
                   FROM st JOIN mc.mimiciv_icu.chartevents ce ON ce.hadm_id = CAST(st.hospitalization_id AS BIGINT)
                   WHERE ce.itemid IN (220052, 220181, 225312)
                     AND ce.charttime BETWEEN st.icu_in - INTERVAL 6 HOUR AND st.icu_in + INTERVAL 1 DAY
                   GROUP BY 1""")
    c = lambda name: f"read_parquet('{_p(clif_dir / f'clif_{name}.parquet')}')"  # noqa: E731
    con.execute(f"""CREATE OR REPLACE TEMP TABLE uo AS   -- first-day urine net of bladder irrigation (mimic-code)
                    SELECT st.stay_id, sum(f.ml) AS ml FROM st JOIN (
                        SELECT hospitalization_id, CAST(recorded_dttm AS TIMESTAMP) AS t, output_volume AS ml FROM {c('output')}
                        WHERE output_group = 'urine' AND output_volume IS NOT NULL
                        UNION ALL SELECT hospitalization_id, CAST(recorded_dttm AS TIMESTAMP), -input_volume FROM {c('input')}
                        WHERE input_category = 'flush_irrigation_urine' AND input_volume > 0) f
                      ON f.hospitalization_id = st.hospitalization_id AND f.t BETWEEN st.icu_in AND st.icu_in + INTERVAL 1 DAY
                    GROUP BY 1""")
    comps = ", ".join(f"o0.{o} AS {o}_0, o6.{o} AS {o}_6" for o in (*SOFA_COMPONENTS, "sofa_total"))
    con.execute(f"""CREATE OR REPLACE TEMP TABLE j AS
                    SELECT st.stay_id, m.*, {comps},
                           -- the P/F pair our respiration score comes from (its worst-scoring pair)
                           coalesce(o6.worst_resp_pf, o6.p_f) AS p_f,
                           coalesce(o6.worst_resp_device, o6.device_category) AS dev,
                           coalesce(o6.worst_resp_device, o6.device_category) IN ('IMV', 'NIPPV', 'CPAP') AS support,
                           o6.creatinine AS cr_ours, o6.gcs_total AS gcs_ours, o6.map AS map_ours,
                           o6.platelet_count AS plt_ours, o6.bilirubin_total AS bili_ours, uo.ml AS uo_ours,
                           o0.clifpy_sofa_resp AS clifpy_resp_0, o6.clifpy_sofa_resp AS clifpy_resp_6,
                           o6c.sofa_resp AS sofa_resp_6c, o6c.sofa_coag AS sofa_coag_6c, o6c.sofa_liver AS sofa_liver_6c,
                           o6c.sofa_renal AS sofa_renal_6c, mr.map_stay, mr.map_hosp,
                           least(m.pao2fio2_vent_min, m.pao2fio2_novent_min) AS pf_mc,
                           m.pao2fio2_vent_min IS NOT NULL
                               AND (m.pao2fio2_novent_min IS NULL OR m.pao2fio2_vent_min <= m.pao2fio2_novent_min) AS support_mc
                    FROM st JOIN mcs m ON m.stay_id = st.mc_stay
                    LEFT JOIN o0 ON o0.stay_id = st.stay_id LEFT JOIN o6 ON o6.stay_id = st.stay_id
                    LEFT JOIN o6c ON o6c.stay_id = st.stay_id LEFT JOIN uo ON uo.stay_id = st.stay_id
                    LEFT JOIN map_raw mr ON mr.stay_id = st.stay_id""")
    # definition-harmonized variants on our inputs (mimic-code's window and rules)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE j AS SELECT *,
                        greatest(sofa_renal_6, coalesce({_uo_score('uo_ours')}, 0)) AS renal_mc_rule,
                        greatest(sofa_renal_6c, coalesce({_uo_score('uo_ours')}, 0)) AS renal_mc_rule_c,
                        coalesce({_mc_resp("CASE WHEN dev = 'IMV' THEN p_f END", "CASE WHEN dev IS DISTINCT FROM 'IMV' THEN p_f END")}, 0)
                            AS resp_mc_rule
                    FROM j""")
    out: dict = {"stays": con.execute("SELECT count(*) FROM j").fetchone()[0], "components": {}}
    for ours, theirs in {**SOFA_COMPONENTS, "sofa_total": "sofa"}.items():
        r = con.execute(f"""SELECT count(*), avg(CAST({ours}_0 = {theirs} AS DOUBLE)), avg(CAST(abs({ours}_0 - {theirs}) <= 1 AS DOUBLE)),
                                   avg({ours}_0 - {theirs}), avg(CAST({ours}_6 = {theirs} AS DOUBLE)),
                                   count(*) FILTER (WHERE {ours}_0 > {theirs}), count(*) FILTER (WHERE {ours}_0 < {theirs})
                            FROM j WHERE {theirs} IS NOT NULL""").fetchone()
        dist = con.execute(f"""SELECT {ours}_0 - {theirs} AS d, count(*) FROM j WHERE {theirs} IS NOT NULL
                               GROUP BY 1 ORDER BY 1""").fetchall()
        out["components"][theirs] = {
            "n": r[0], "exact": _r(r[1]), "within_1": _r(r[2]), "mean_diff": _r(r[3]), "exact_mimic_code_window": _r(r[4]),
            "ours_higher": r[5], "ours_lower": r[6], "difference_counts": {int(d): n for d, n in dist if d is not None},
            "mimic_code_missing_ours_positive": con.execute(
                f"SELECT count(*) FROM j WHERE {theirs} IS NULL AND {ours}_0 > 0").fetchone()[0]}
    one = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
    mc_fixed = _mc_resp_hourly("pao2fio2_vent_min", "pao2fio2_novent_min")
    out["components"]["respiration"]["exact_clifpy_as_published"] = _r(one(
        "SELECT avg(CAST(clifpy_resp_0 = respiration AS DOUBLE)) FROM j WHERE respiration IS NOT NULL"))
    out["components"]["sofa"]["exact_clifpy_as_published"] = _r(one(
        "SELECT avg(CAST(sofa_total_0 - sofa_resp_0 + clifpy_resp_0 = sofa AS DOUBLE)) FROM j"))
    window = "definition: window from 6 h before ICU admission (mimic-code) instead of from admission"
    lab_time = "definition: labs by specimen time (mimic-code) instead of result time (clifpy)"
    # renal: the urine-output criterion is a definition difference; creatinine differences are input differences
    out["renal"] = {
        "exact_creatinine_only_vs_mimic_code_creatinine_only": _r(one(f"""
            SELECT avg(CAST(sofa_renal_6 = {_creat_score('creatinine_max')} AS DOUBLE)) FROM j
            WHERE creatinine_max IS NOT NULL""")),
        "exact_with_mimic_code_definition": _r(one(
            "SELECT avg(CAST(renal_mc_rule = renal AS DOUBLE)) FROM j WHERE renal IS NOT NULL")),
        "exact_with_mimic_code_definition_specimen_time": _r(one(
            "SELECT avg(CAST(renal_mc_rule_c = renal AS DOUBLE)) FROM j WHERE renal IS NOT NULL")),
        "mimic_code_stays_scored_by_urine_output": one(f"""
            SELECT count(*) FROM j WHERE renal > coalesce({_creat_score('creatinine_max')}, 0)"""),
        "disagreements": _attribute(con, "renal IS NOT NULL AND sofa_renal_0 <> renal", [
            (window, "sofa_renal_6 = renal"),
            ("definition: urine-output criterion (mimic-code only)", "renal_mc_rule = renal"),
            (lab_time, "renal_mc_rule_c = renal"),
            ("inputs: highest creatinine differs", f"sofa_renal_6c <> coalesce({_creat_score('creatinine_max')}, 0)"),
            ("inputs: first-day urine output differs", f"coalesce({_uo_score('uo_ours')}, 0) <> coalesce({_uo_score('urineoutput')}, 0)"),
        ])}
    out["respiration"] = {
        "raised_by_clifpy_correction": one("SELECT count(*) FROM j WHERE sofa_resp_0 <> clifpy_resp_0"),
        "raised_by_clifpy_correction_now_agreeing": one(
            "SELECT count(*) FROM j WHERE sofa_resp_0 <> clifpy_resp_0 AND sofa_resp_0 = respiration"),
        "raised_by_clifpy_correction_agreeing_before": one(
            "SELECT count(*) FROM j WHERE sofa_resp_0 <> clifpy_resp_0 AND clifpy_resp_0 = respiration"),
        "mimic_code_ventilated_pf_200_400_underscored": one(f"SELECT count(*) FROM j WHERE ({mc_fixed}) <> respiration"),
        "exact_against_mimic_code_hourly_rule": _r(one(
            f"SELECT avg(CAST(sofa_resp_0 = ({mc_fixed}) AS DOUBLE)) FROM j WHERE respiration IS NOT NULL")),
        "exact_with_mimic_code_rule_on_our_inputs": _r(one(
            "SELECT avg(CAST(resp_mc_rule = respiration AS DOUBLE)) FROM j WHERE respiration IS NOT NULL")),
        "disagreements": _attribute(con, "respiration IS NOT NULL AND sofa_resp_0 <> respiration", [
            (window, "sofa_resp_6 = respiration"),
            ("mimic-code defect: a ventilated P/F of 200-400 is not scored (SOFA and mimic-code's hourly sofa: 1-2)",
             f"({mc_fixed}) <> respiration AND ({mc_fixed}) = sofa_resp_6"),
            (lab_time, "sofa_resp_6c = respiration"),
            ("definition: P/F needs an FiO2 setting charted in the 4 h before the gas (clifpy); mimic-code takes "
             "the gas panel's FiO2 first", "p_f IS NULL"),
            ("definition: FiO2 of the charted setting (clifpy) or of the gas panel (mimic-code); P/F differs by > 10%",
             "abs(p_f - pf_mc) > 0.1 * pf_mc"),
            # scores 3-4 need support in both: one side at 3-4 and the other below means a gas's support differs
            ("definition: respiratory support (clifpy: IMV, NIPPV or CPAP charted at the gas; mimic-code: invasive "
             "ventilation by its ventilation concept)",
             "coalesce(support, false) <> support_mc OR (respiration >= 3) <> (sofa_resp_6 >= 3)"),
        ])}
    out["cns"] = {"disagreements": _attribute(con, "cns IS NOT NULL AND sofa_cns_0 <> cns", [
        (window, "sofa_cns_6 = cns"),
        ("definition: mimic-code scores GCS 15 when the verbal response is 'No Response-ETT'", "gcs_min = 15 AND gcs_ours < 15"),
        ("inputs: lowest GCS differs", "gcs_ours IS DISTINCT FROM gcs_min")])}
    out["cardiovascular"] = {"disagreements": _attribute(con, "cardiovascular IS NOT NULL AND sofa_cv_97_0 <> cardiovascular", [
        (window, "sofa_cv_97_6 = cardiovascular"),
        ("cleaning: a recorded MAP of 0 (CLIF's range 0-250 keeps it; mimic-code keeps MAP > 0)", "map_ours <= 0"),
        ("definition: our window covers the hospitalization, here another ICU stay's MAP (mimic-code: this stay)",
         "map_ours < mbp_min AND map_hosp <= map_ours AND coalesce(map_stay, 1e9) > map_ours"),
        ("definition: mimic-code averages arterial and cuff MAP charted at one time", "map_ours < mbp_min AND map_stay <= map_ours"),
        ("definition: our hourly median per blood-pressure site (monitor-density rule) replaces the hour's lowest reading",
         "map_ours > mbp_min AND map_hosp <= mbp_min"),
        ("inputs: vasopressor doses differ (lowest MAP agrees)", "abs(map_ours - mbp_min) < 1"),
    ])}
    out["coagulation"] = {"disagreements": _attribute(con, "coagulation IS NOT NULL AND sofa_coag_0 <> coagulation", [
        (window, "sofa_coag_6 = coagulation"), (lab_time, "sofa_coag_6c = coagulation"),
        ("inputs: lowest platelet count differs", "plt_ours IS DISTINCT FROM platelet_min")])}
    out["liver"] = {"disagreements": _attribute(con, "liver IS NOT NULL AND sofa_liver_0 <> liver", [
        (window, "sofa_liver_6 = liver"), (lab_time, "sofa_liver_6c = liver"),
        ("inputs: highest bilirubin differs", "bili_ours IS DISTINCT FROM bilirubin_max")])}
    # the total under progressively harmonized definitions, then against mimic-code without its respiration defect
    total = "coalesce({r}, 0) + coalesce({c}, 0) + coalesce({l}, 0) + coalesce(sofa_cv_97_6, 0) + coalesce(sofa_cns_6, 0) " \
            "+ coalesce({n}, 0)"
    harmonized = total.format(r="sofa_resp_6c", c="sofa_coag_6c", l="sofa_liver_6c", n="renal_mc_rule_c")
    out["total_exact"] = {
        "clifpy_as_published": out["components"]["sofa"]["exact_clifpy_as_published"],
        "ours": out["components"]["sofa"]["exact"],
        "mimic_code_window": out["components"]["sofa"]["exact_mimic_code_window"],
        "plus_urine_output_criterion": _r(one(f"""SELECT avg(CAST(({total.format(r='sofa_resp_6', c='sofa_coag_6',
            l='sofa_liver_6', n='renal_mc_rule')}) = sofa AS DOUBLE)) FROM j""")),
        "plus_labs_by_specimen_time": _r(one(f"SELECT avg(CAST(({harmonized}) = sofa AS DOUBLE)) FROM j")),
        "against_mimic_code_without_its_respiration_defect": _r(one(f"""
            SELECT avg(CAST(({harmonized}) = sofa - coalesce(respiration, 0) + coalesce(({mc_fixed}), 0) AS DOUBLE)) FROM j""")),
        "with_mimic_code_first_day_respiration_rule_on_our_inputs": _r(one(f"""SELECT avg(CAST(({total.format(
            r='resp_mc_rule', c='sofa_coag_6c', l='sofa_liver_6c', n='renal_mc_rule_c')}) = sofa AS DOUBLE)) FROM j"""))}
    return out


# -- driver -------------------------------------------------------------------------------------------
def parity(mirror: Path, mimic_code: Path, clif_dir: Path, derived_dir: Path, db: Path, out_dir: Path,
           work_dir: Path) -> dict:
    """Run mimic-code, the comparisons and the first-day SOFA decomposition, and write
    ``mimic_code_parity.{json,md}`` to ``out_dir``."""
    con, rows = run_concepts(mirror, mimic_code, db)
    con.close()
    report = {"mimic_code": _p(mimic_code), "concept_rows": rows,
              "logic": {"kdigo": logic_kdigo(db), "ventilation": logic_ventilation(db, mimic_code), "nee": logic_nee(db)},
              "agreement": agreement(db, clif_dir, derived_dir),
              "sofa_first_day": sofa_decomposition(db, mimic_code, clif_dir, work_dir)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mimic_code_parity.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "mimic_code_parity.md").write_text(markdown(report), encoding="utf-8", newline="\n")
    return report


def markdown(report: dict) -> str:
    lines = ["# MIMIC-IV derived concepts against mimic-code", "", "## Logic parity (our code on mimic-code's inputs)", ""]
    k = report["logic"]["kdigo"]
    lines.append(f"KDIGO: {k['mimic_code_rows']} mimic-code rows; only mimic-code {k['rows_only_mimic_code']}, only ours "
                 f"{k['rows_only_ours']}; urine rows outside the stay window {k['urine_rows_outside_stay_window']}.")
    lines += ["", "| Column | Compared | Identical | Identical, stays without urine outside the window |", "|---|---|---|---|"]
    lines += [f"| {c} | {v['compared']} | {v['share']} | {v['share_stays_without_outside_urine']} |"
              for c, v in k["columns"].items()]
    v, n = report["logic"]["ventilation"], report["logic"]["nee"]
    lines += ["", f"Ventilation episodes: mimic-code {v['mimic_code_episodes']}, ours {v['our_episodes']}, only mimic-code "
                  f"{v['only_mimic_code']}, only ours {v['only_ours']}.",
              f"Norepinephrine equivalents: {n['intervals']} intervals, identical {n['share']} (max difference {n['max_abs_diff']}); "
              f"{n['intervals_repeated_per_concurrent_order']} intervals that mimic-code repeats per concurrent order not compared.",
              "", "## End-to-end agreement per ICU stay", "", "| Concept | n | Agreement |", "|---|---|---|"]
    for name, m in report["agreement"].items():
        if not isinstance(m, dict) or name == "notes":
            continue
        if "kappa" in m:
            s = f"agreement {m['agreement']}, kappa {m['kappa']} (both {m['both']}, only ours {m['only_ours']}, only mimic-code {m['only_mimic_code']})"
        elif "exact" in m:
            s = f"exact {m['exact']}, within 1 {m['within_1']}, mean difference {m['mean_diff']}, Spearman {m['spearman']}"
        else:
            tol = next(k for k in m if k.startswith("within_"))
            s = (f"Pearson {m['pearson']}, Spearman {m['spearman']}, median |difference| {m['median_abs_diff']}, {tol} {m[tol]}, medians "
                 f"{m['median_ours']} / {m['median_mimic_code']} (ours / mimic-code), only mimic-code {m['only_mimic_code']}, only ours {m['only_ours']}")
        lines.append(f"| {name} | {m['n']} | {s} |")
    lines += ["", "## Known causes of disagreement (counts on this extract)", ""]
    lines += [f"- {k}: {v}" for k, v in report["agreement"].get("notes", {}).items()]
    if "sofa_first_day" in report:
        lines += sofa_markdown(report["sofa_first_day"])
    return "\n".join(lines) + "\n"


def sofa_markdown(s: dict) -> list[str]:
    lines = ["", "## First-day SOFA per component", "",
             "Not equivalent to mimic-code: the two scores use different definitions (window, renal criteria, P/F "
             "construction, MAP density, GCS of intubated patients), and each carries a reference-implementation "
             f"defect. {s['stays']} ICU stays; ours = clifpy 2.1.0 on canonical CLIF from ICU admission to +24 h, with "
             "the respiration score the worst score of clifpy's P/F pairs (clifpy scores the lowest P/F); mimic-code = `first_day_sofa` (6 h before admission to +24 h).", "",
             "| Component | n | Exact | Within 1 | Mean difference (ours - mimic-code) | Ours higher / lower | "
             "Exact on mimic-code's window |", "|---|---|---|---|---|---|---|"]
    for name, m in s["components"].items():
        lines.append(f"| {name} | {m['n']} | {m['exact']} | {m['within_1']} | {m['mean_diff']} | "
                     f"{m['ours_higher']} / {m['ours_lower']} | {m['exact_mimic_code_window']} |")
    lines += ["", "Disagreeing stays by cause (the first cause that explains the stay):", ""]
    for comp in ("respiration", "coagulation", "liver", "cardiovascular", "cns", "renal"):
        d = s[comp]["disagreements"]
        lines.append(f"- {comp} ({d['stays']}): " + "; ".join(f"{c}: {n}" for c, n in d["by_cause"].items()))
    r, p, t = s["renal"], s["respiration"], s["total_exact"]
    lines += ["", f"Renal: creatinine-only scores agree {r['exact_creatinine_only_vs_mimic_code_creatinine_only']} on "
                  f"mimic-code's window; mimic-code scores {r['mimic_code_stays_scored_by_urine_output']} stays above their "
                  f"creatinine score by urine output; with mimic-code's definition on our inputs {r['exact_with_mimic_code_definition']} "
                  f"({r['exact_with_mimic_code_definition_specimen_time']} with labs by specimen time).",
              f"Respiration: the clifpy correction raises {p['raised_by_clifpy_correction']} stays "
              f"({p['raised_by_clifpy_correction_now_agreeing']} now agree with mimic-code, "
              f"{p['raised_by_clifpy_correction_agreeing_before']} agreed before); mimic-code's first-day rule under-scores "
              f"{p['mimic_code_ventilated_pf_200_400_underscored']} stays (ventilated P/F 200-400); exact against mimic-code's "
              f"hourly rule {p['exact_against_mimic_code_hourly_rule']}.",
              "Total, exact: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in t.items()) + "."]
    return lines
