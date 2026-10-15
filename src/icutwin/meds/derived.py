"""Derived clinical concepts per ICU stay, computed on canonical CLIF (labels and evaluation; never tokenized).

mimic-code ports follow the commit pinned in ``refs/REFS.lock``; deviations are listed with each concept.

* ``sofa_24h``: SOFA over the first 24 h of each ICU stay with clifpy's ``compute_sofa_polars``. clifpy 2.1.0
  ranks respiratory devices by their CLIF 2.1 names ('IMV', 'High Flow NC'), so it reads a view of the
  directory whose ``respiratory_support`` carries those names (every other table is the canonical file).
  Deviation (clifpy defects, :func:`worst_resp_scores`): the respiration score is the worst score of clifpy's
  concurrent P/F pairs, as SOFA defines it, not the score of the lowest P/F.
* ``nee``: norepinephrine-equivalent dose (mcg/kg/min) wherever a vasopressor rate changes, mimic-code
  ``norepinephrine_equivalent_dose`` (Goradia et al. 2020): norepinephrine + epinephrine + phenylephrine/10 +
  dopamine/100 + vasopressin (U/min) x 2.5. MIMIC charts weight-based rates; rates per minute or hour of other
  sources use the weight in effect (below). Rates without a weight or a dose (presence-only sources) give none.
* ``episodes``:
  * respiratory support, mimic-code ``ventilation``: one status per charting time (the CLIF device, a missing
    one inferred with clifpy's waterfall heuristics, see :func:`resp_status`; at one time the most invasive),
    a new episode when the status changes or 14 h after the previous charting of
    the status, ending at the next charting within 14 h; single-charting episodes are dropped. Episode names
    are the CLIF devices (``imv``, ``nippv``, ...). mimic-code's 'Tracheostomy' status (tube or trach mask)
    has no CLIF device: ventilation through a tracheostomy is ``imv``, a trach mask ``trach_collar``.
  * ``vasopressor``: a running vasoactive infusion, from its start to its stop.
  * ``rrt``: CRRT or IHD documentation; a new episode after 24 h without.
* ``kdigo``: AKI stage at every creatinine, urine-output and CRRT time, mimic-code ``kdigo_creatinine``,
  ``kdigo_uo`` and ``kdigo_stages``. Creatinine: baselines are the lowest values of the previous 48 h and 7
  days. Urine output (``urine_output``: urine volumes, bladder irrigation inflow negative) is summed over 6/12/24 h
  windows and divided by the hours of documentation they cover (hours since the previous charting, 1 for the
  first); a rate needs 6 to 12, 12 and 24 h of documentation; the first 6 ICU hours are unstaged. CRRT is stage
  3. Deviations: creatinine of the stay's hospitalization only (mimic-code: any admission of the patient in the
  window; cross-admission clocks are not comparable in every source); CRRT is any CRRT documentation
  (mimic-code: a charted CRRT mode, which HiRID and INSPIRE do not record); ``aki_stage_smoothed`` is
  per stay (mimic-code partitions by patient while ordering by time since each stay's admission).
* ``fluid_balance``: per ICU day (24 h from ICU admission), fluid in (``input`` volumes, infusions charted in
  ml/hr integrated over time, intermittent doses in ml) and out (``output``).
* ``ase``: CDC Adult Sepsis Event per blood culture (clifpy ``compute_ase``), with the stays whose window
  holds the culture; only where blood cultures and administered antimicrobials are both recorded (MIMIC,
  AmsterdamUMCdb: eICU releases antimicrobial orders only, the other sources no cultures).

Weight in effect (mimic-code ``weight_durations``): the last weight charted at or before a time, before the
first weight that weight. That fallback looks ahead: a value it converts carries ``weight_charted_later``
(``nee``, ``kdigo``), and a prospective task drops or recomputes it when the weight was charted after its cutoff.
(The training MEDS converts doses causally: ``icutwin.meds.elf``, ``weight_lookup='segment_start'``.)
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
from pathlib import Path

import duckdb

from icutwin.clif.conform import connect
from icutwin.meds.build import drop_minors

log = logging.getLogger(__name__)

DEVICE_NAMES_2_1 = {"imv": "IMV", "nippv": "NIPPV", "cpap": "CPAP", "hfnc": "High Flow NC", "face_mask": "Face Mask",
                    "trach_collar": "Trach Collar", "nasal_cannula": "Nasal Cannula", "room_air": "Room Air",
                    "other": "Other"}
# norepinephrine equivalents (Goradia et al. 2020, as mimic-code): factor on mcg/kg/min, vasopressin on U/min
NEE_FACTORS = {"norepinephrine": 1.0, "epinephrine": 1.0, "phenylephrine": 0.1, "dopamine": 0.01, "vasopressin": 2.5}
VASOPRESSORS = (*NEE_FACTORS, "angiotensin")
IMV_GAP_HOURS, RRT_GAP_HOURS = 14, 24
# the respiratory status of a charting time with several devices: the most invasive (mimic-code's status order)
DEVICE_SUPPORT_ORDER = ("imv", "nippv", "cpap", "hfnc", "face_mask", "trach_collar", "nasal_cannula", "other", "room_air")
VENTILATOR_DEVICES = ("imv", "nippv", "cpap")
# ventilator settings and measurements (not FiO2, oxygen flow or the patient's respiratory rate)
VENTILATOR_VALUES = ("peep_set", "resp_rate_set", "tidal_volume_set", "peak_inspiratory_pressure_set",
                     "pressure_support_set", "pressure_control_set", "inspiratory_time_set", "tidal_volume_obs",
                     "plateau_pressure_obs", "peak_inspiratory_pressure_obs", "peep_obs", "minute_vent_obs",
                     "mean_airway_pressure_obs")
SOFA_TABLES = ("labs", "vitals", "patient_assessments", "respiratory_support", "medication_admin_continuous")
SOFA_PARTS = ("sofa_cv_97", "sofa_coag", "sofa_liver", "sofa_resp", "sofa_cns", "sofa_renal")
SOFA_RESP_SUPPORT = ("IMV", "NIPPV", "CPAP")   # clifpy's respiratory support for SOFA 3-4 (CLIF 2.1 names)
ASE_TABLES = ("hospitalization", "patient", "adt", "labs", "hospital_diagnosis", "respiratory_support",
              "medication_admin_continuous", "medication_admin_intermittent", "microbiology_culture")


def _p(path: Path) -> str:
    return str(path).replace("\\", "/")


def _stays(con: duckdb.DuckDBPyConnection, clif_dir: Path, min_age_years: float | None = 18.0) -> None:
    """The MEDS cohort (``icutwin.meds.build``): stays with ICU time, of adult patients."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE stays AS
                    SELECT stay_id, hospitalization_id, CAST(icu_in_dttm AS TIMESTAMP) AS icu_in,
                           CAST(icu_out_dttm AS TIMESTAMP) AS icu_out
                    FROM read_parquet('{_p(clif_dir / 'clif_ext_icu_stay.parquet')}')
                    WHERE icu_out_dttm > icu_in_dttm""")
    drop_minors(con, clif_dir, min_age_years)


def _table(con: duckdb.DuckDBPyConnection, clif_dir: Path, table: str) -> bool:
    f = clif_dir / f"clif_{table}.parquet"
    if not f.exists():
        return False
    con.execute(f"CREATE OR REPLACE TEMP VIEW {table} AS SELECT * FROM read_parquet('{_p(f)}')")
    return True


# -- clifpy concepts (SOFA, ASE) ------------------------------------------------------------------------
def _clifpy_view(clif_dir: Path, view: Path, tables: tuple[str, ...], labs_at_collection: bool = False) -> Path:
    """A directory clifpy can read: every canonical file (hard links), ``respiratory_support`` with the CLIF
    2.1 device names clifpy 2.1.0 ranks by ('IMV', 'High Flow NC'), and an empty CLIF 3.0 table for each of
    ``tables`` the dataset lacks: clifpy turns a missing table into an empty frame with naive times, which
    then fails to concatenate with the UTC tables (NWICU has no patient_assessments). ``labs_at_collection``
    stamps each lab at its specimen time (clifpy windows labs by ``lab_result_dttm``)."""
    from icutwin.clif.conform import _DUCK, table_schema

    if view.exists():
        shutil.rmtree(view)
    view.mkdir(parents=True)
    restamped = {"clif_respiratory_support", *(("clif_labs",) if labs_at_collection else ())}
    for f in clif_dir.glob("clif_*.parquet"):
        if f.stem not in restamped:
            try:
                os.link(f, view / f.name)
            except OSError:
                shutil.copy2(f, view / f.name)
    con = connect()
    labs = clif_dir / "clif_labs.parquet"
    if labs_at_collection and labs.exists():
        con.execute(f"""COPY (SELECT * REPLACE (coalesce(lab_collect_dttm, lab_result_dttm) AS lab_result_dttm)
                              FROM read_parquet('{_p(labs)}'))
                        TO '{_p(view / labs.name)}' (FORMAT parquet)""")
    for table in tables:
        if not (clif_dir / f"clif_{table}.parquet").exists():
            cols = ", ".join(f'CAST(NULL AS {_DUCK[c["data_type"].upper()]}) AS "{c["name"]}"'
                             for c in table_schema(table)["columns"])
            con.execute(f"COPY (SELECT {cols} WHERE false) TO '{_p(view / f'clif_{table}.parquet')}' (FORMAT parquet)")
    rs = clif_dir / "clif_respiratory_support.parquet"
    if rs.exists():
        names = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in DEVICE_NAMES_2_1.items())
        con.execute(f"""COPY (SELECT * REPLACE (CASE device_category {names} ELSE device_category END AS device_category)
                              FROM read_parquet('{_p(rs)}'))
                        TO '{_p(view / rs.name)}' (FORMAT parquet)""")
    logging.getLogger("clifpy").setLevel(logging.WARNING)
    return view


def _resp_score(pf, device):
    """SOFA respiration score of one P/F (Vincent et al., Intensive Care Med 1996): < 400 -> 1, < 300 -> 2,
    < 200 with respiratory support -> 3, < 100 with support -> 4; support (IMV, NIPPV, CPAP) is a condition of
    3 and 4 only."""
    import polars as pl

    support = device.is_in(list(SOFA_RESP_SUPPORT)).fill_null(False)
    return (pl.when((pf < 100) & support).then(4).when((pf < 200) & support).then(3).when(pf < 300).then(2)
            .when(pf < 400).then(1).when(pf >= 400).then(0).otherwise(None))


def worst_resp_scores(pairs):
    """Per stay, the worst respiration score over clifpy's concurrent P/F pairs (each PaO2 with the FiO2 setting
    charted within 4 h before it and the device at that time): SOFA scores the worst value of the window.
    clifpy 2.1.0 scores the lowest P/F with the device at its time instead, which misses a supported P/F a little
    above an unsupported one, and it gives a P/F < 200 without support no score (0; SOFA: 2)."""
    import polars as pl

    worst = lambda c: pl.col(c).sort_by(["score", "concurrent_pf"], descending=[True, False]).first()  # noqa: E731
    return (pairs.with_columns(_resp_score(pl.col("concurrent_pf"), pl.col("device_category")).alias("score"))
            .group_by("stay_id").agg(pl.col("score").max().alias("worst_resp"),
                                     worst("concurrent_pf").alias("worst_resp_pf"),
                                     worst("device_category").alias("worst_resp_device")))


def fix_sofa_resp(sofa, worst=None):
    """``sofa_resp`` as SOFA defines it: the worst pair score (``worst_resp_scores``) where clifpy formed P/F
    pairs, else clifpy's lowest P/F with its device, scored by :func:`_resp_score`; ``sofa_total`` follows. Keeps
    clifpy's own score as ``clifpy_sofa_resp`` (not prefixed ``sofa_``: not a component)."""
    import polars as pl

    resp = _resp_score(pl.col("p_f"), pl.col("device_category"))
    if worst is not None:
        sofa = sofa.join(worst, on="stay_id", how="left")
        resp = pl.coalesce(pl.col("worst_resp"), resp)
    dtype = sofa.schema["sofa_resp"]
    out = sofa.with_columns(pl.col("sofa_resp").alias("clifpy_sofa_resp"),
                            resp.fill_null(0).cast(dtype).alias("sofa_resp"))
    if worst is not None:
        out = out.drop("worst_resp")
    return out.with_columns(pl.sum_horizontal([pl.col(c) for c in SOFA_PARTS]).alias("sofa_total"))


@contextlib.contextmanager
def _captured_pf_pairs():
    """Capture the concurrent P/F pairs clifpy forms inside ``compute_sofa_polars`` (it returns only their
    lowest value)."""
    import clifpy.utils.sofa_polars as sp

    captured, original = [], sp._calculate_concurrent_pf_ratios

    def capture(*args, **kwargs):
        captured.append(original(*args, **kwargs))
        return captured[-1]

    sp._calculate_concurrent_pf_ratios = capture
    try:
        yield captured
    finally:
        sp._calculate_concurrent_pf_ratios = original


@contextlib.contextmanager
def _spo2_optional():
    """clifpy imputes PaO2 from an ``spo2`` column of its pivoted values, which is absent when no stay of the call
    has an SpO2 in its window (a small batch of stays, e.g. decoded rollouts): add it empty, so nothing is imputed."""
    import polars as pl
    import clifpy.utils.sofa_polars as sp

    original = sp._impute_pao2_from_spo2

    def tolerant(df):
        if "spo2" not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("spo2"))
        return original(df)

    sp._impute_pao2_from_spo2 = tolerant
    try:
        yield
    finally:
        sp._impute_pao2_from_spo2 = original


def sofa_window(clif_dir: Path, work_dir: Path, pre_hours: float = 0.0, keep_inputs: bool = False,
                labs_at_collection: bool = False):
    """clifpy SOFA (worst values) from ``pre_hours`` before ICU admission to 24 h after it, one row per stay,
    with clifpy's respiration defect corrected (``fix_sofa_resp``). ``keep_inputs`` keeps clifpy's extremal
    inputs (worst concurrent P/F and its device, creatinine, GCS, ...) and ``clifpy_sofa_resp``;
    ``labs_at_collection`` windows labs by specimen time instead of result time."""
    from clifpy.utils.sofa_polars import compute_sofa_polars

    view = _clifpy_view(clif_dir, work_dir / f"clifpy_sofa_{int(pre_hours * 60)}{'_c' if labs_at_collection else ''}",
                        SOFA_TABLES, labs_at_collection)
    cohort = connect().execute(f"""SELECT hospitalization_id, stay_id,
                                          icu_in_dttm - to_microseconds(CAST({pre_hours} * 3600e6 AS BIGINT)) AS start_dttm,
                                          icu_in_dttm + INTERVAL 24 HOUR AS end_dttm
                                   FROM read_parquet('{_p(clif_dir / 'clif_ext_icu_stay.parquet')}')
                                   WHERE icu_out_dttm > icu_in_dttm""").pl()
    with _captured_pf_pairs() as pairs, _spo2_optional():
        out = compute_sofa_polars(str(view), cohort, id_name="stay_id", timezone="UTC")
    out = fix_sofa_resp(out, worst_resp_scores(pairs[0]) if pairs else None)
    shutil.rmtree(view, ignore_errors=True)
    return out if keep_inputs else out.select(["stay_id", *sorted(c for c in out.columns if c.startswith("sofa_"))])


def adult_sepsis_events(clif_dir: Path, work_dir: Path):
    """CDC Adult Sepsis Event per blood culture of every hospitalization (clifpy ``compute_ase``: presumed
    infection = blood culture + >= 4 qualifying antimicrobial days; organ dysfunction within 2 days; 14-day
    repeat-infection timeframe), or None when the dataset records no blood culture or no administered
    antimicrobial (ASE needs both)."""
    from clifpy.utils.ase import compute_ase

    con = connect()
    mc, mi = clif_dir / "clif_microbiology_culture.parquet", clif_dir / "clif_medication_admin_intermittent.parquet"
    if not (mc.exists() and mi.exists()) or not con.execute(f"""
            SELECT count(*) FROM read_parquet('{_p(mc)}') WHERE fluid_category = 'blood_buffy'""").fetchone()[0]:
        return None
    view = _clifpy_view(clif_dir, work_dir / "clifpy_ase", ASE_TABLES)
    try:
        return compute_ase(data_directory=str(view), filetype="parquet", timezone="UTC", verbose=False)
    finally:
        shutil.rmtree(view, ignore_errors=True)


# -- weights and vasopressors -------------------------------------------------------------------------
def _weights(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""CREATE OR REPLACE TEMP TABLE weights AS
                   SELECT hospitalization_id, CAST(recorded_dttm AS TIMESTAMP) AS t, avg(vital_value) AS kg
                   FROM vitals WHERE vital_category = 'weight_kg' AND vital_value > 0 GROUP BY 1, 2""")
    con.execute("""CREATE OR REPLACE TEMP TABLE first_weight AS
                   SELECT hospitalization_id, arg_min(kg, t) AS kg FROM weights GROUP BY 1""")


def _weight_in_effect(src: str, time_col: str) -> str:
    """``src`` (with ``hospitalization_id``) plus ``kg``: the last weight charted at or before ``time_col``,
    before the first weight that weight (mimic-code ``weight_durations``), and ``kg_charted_later``: the weight is
    that fallback, charted after ``time_col`` (provenance: a prospective task must not use such a value before
    the weight's charting)."""
    return f"""
        SELECT s.* EXCLUDE (_w), coalesce(s._w, f.kg) AS kg, (s._w IS NULL AND f.kg IS NOT NULL) AS kg_charted_later
        FROM (
            SELECT s.*, w.kg AS _w FROM ({src}) s
            ASOF LEFT JOIN weights w ON w.hospitalization_id = s.hospitalization_id AND s.{time_col} >= w.t) s
        LEFT JOIN first_weight f ON f.hospitalization_id = s.hospitalization_id"""


def nee(con: duckdb.DuckDBPyConnection) -> None:
    """Table ``nee``: (hospitalization_id, t, nee_mcg_kg_min, incomplete, weight_charted_later) at every vasopressor
    rate change; ``weight_charted_later``: a running rate was converted with a weight charted after it."""
    drugs = ", ".join(f"'{d}'" for d in NEE_FACTORS)
    factor = " ".join(f"WHEN '{d}' THEN {f}" for d, f in NEE_FACTORS.items())
    rates = f"""
        SELECT hospitalization_id, CAST(admin_dttm AS TIMESTAMP) AS t, med_category, med_dose, lower(med_dose_unit) AS unit
        FROM medication_admin_continuous
        WHERE med_category IN ({drugs}) AND med_dose IS NOT NULL"""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE vaso AS
        WITH r AS ({_weight_in_effect(rates, 't')})
        SELECT hospitalization_id, t, med_category,
               (CASE WHEN med_category = 'vasopressin' THEN
                     CASE unit WHEN 'u/min' THEN med_dose WHEN 'u/hr' THEN med_dose / 60 WHEN 'mu/min' THEN med_dose / 1000
                               WHEN 'u/kg/min' THEN med_dose * kg WHEN 'u/kg/hr' THEN med_dose * kg / 60 END
                ELSE CASE unit WHEN 'mcg/kg/min' THEN med_dose WHEN 'mcg/kg/hr' THEN med_dose / 60
                               WHEN 'mg/kg/hr' THEN med_dose * 1000 / 60 WHEN 'mg/kg/min' THEN med_dose * 1000
                               WHEN 'ng/kg/min' THEN med_dose / 1000 WHEN 'mcg/min' THEN med_dose / kg
                               WHEN 'mcg/hr' THEN med_dose / 60 / kg WHEN 'mg/hr' THEN med_dose * 1000 / 60 / kg
                               WHEN 'mg/min' THEN med_dose * 1000 / kg END
                END) * CASE med_category {factor} END AS nee_part,
               kg_charted_later AND CASE WHEN med_category = 'vasopressin' THEN unit IN ('u/kg/min', 'u/kg/hr')
                                         ELSE unit IN ('mcg/min', 'mcg/hr', 'mg/hr', 'mg/min') END AS later_weight
        FROM r""")
    # step functions: at every change time, the current rate of every drug (ASOF: last change at or before t)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE nee AS
        WITH times AS (SELECT DISTINCT hospitalization_id, t FROM vaso),
        cur AS (
            SELECT tm.hospitalization_id, tm.t, d.med_category,
                   (SELECT v.nee_part FROM vaso v WHERE v.hospitalization_id = tm.hospitalization_id
                    AND v.med_category = d.med_category AND v.t <= tm.t ORDER BY v.t DESC LIMIT 1) AS part,
                   (SELECT v.later_weight FROM vaso v WHERE v.hospitalization_id = tm.hospitalization_id
                    AND v.med_category = d.med_category AND v.t <= tm.t ORDER BY v.t DESC LIMIT 1) AS later
            FROM times tm JOIN (SELECT DISTINCT hospitalization_id, med_category FROM vaso) d USING (hospitalization_id))
        SELECT hospitalization_id, t, round(sum(coalesce(part, 0)), 4) AS nee_mcg_kg_min,
               bool_or(part IS NULL AND EXISTS (SELECT 1 FROM vaso v WHERE v.hospitalization_id = cur.hospitalization_id
                                                AND v.med_category = cur.med_category AND v.t <= cur.t)) AS incomplete,
               coalesce(bool_or(coalesce(later, false) AND part > 0), false) AS weight_charted_later
        FROM cur GROUP BY 1, 2""")


# -- episodes -----------------------------------------------------------------------------------------
def _gap_episodes(src: str, gap_hours: float) -> str:
    """Episodes of time-stamped documentation ``src`` (hospitalization_id, t): a new episode after a gap."""
    return f"""
        WITH d AS (SELECT hospitalization_id, t,
                          CASE WHEN lag(t) OVER w IS NULL OR t - lag(t) OVER w > INTERVAL {gap_hours} HOUR THEN 1 ELSE 0 END AS new
                   FROM ({src}) WINDOW w AS (PARTITION BY hospitalization_id ORDER BY t)),
        n AS (SELECT *, sum(new) OVER (PARTITION BY hospitalization_id ORDER BY t ROWS UNBOUNDED PRECEDING) AS ep FROM d)
        SELECT hospitalization_id, ep, min(t) AS start_dttm, max(t) AS end_dttm FROM n GROUP BY 1, 2"""


def resp_status(con: duckdb.DuckDBPyConnection) -> None:
    """Table ``resp_status`` (hospitalization_id, t, status): the respiratory device of every charting time.

    A row without a device takes clifpy's waterfall heuristics (``process_resp_support_waterfall``, phase 1
    and its device forward fill): IMV for an AC/VC, SIMV or pressure-control mode; IMV (NIPPV) next to IMV
    (NIPPV) charting with PEEP, rate and tidal volume (inspiratory and support pressure) set; else, on a
    ventilator row (a mode, or a ventilator setting or measurement), the last documented device when that is a
    ventilator (``VENTILATOR_DEVICES``; clifpy's guard: no PEEP on a nasal cannula). Other rows without a device
    (oxygen flow or FiO2 only) have no status, as mimic-code ``ventilation`` needs a device or a ventilator mode;
    clifpy fills every row, which would carry IMV into nurse-charted oxygen after an undocumented extubation.
    At one time: the most invasive status.
    """
    have = {c for (c,) in con.execute("SELECT column_name FROM (DESCRIBE respiratory_support)").fetchall()}
    col = lambda c: c if c in have else f"NULL::{'VARCHAR' if c == 'mode_category' else 'DOUBLE'} AS {c}"  # noqa: E731
    cols = ", ".join(col(c) for c in ("mode_category", *VENTILATOR_VALUES))
    vent_row = f"(mode_category IS NOT NULL OR coalesce({', '.join(VENTILATOR_VALUES)}) IS NOT NULL)"
    vent_devices = ", ".join(f"'{d}'" for d in VENTILATOR_DEVICES)
    w = "OVER (PARTITION BY hospitalization_id ORDER BY t, _r)"
    rank = " ".join(f"WHEN '{d}' THEN {i}" for i, d in enumerate(DEVICE_SUPPORT_ORDER))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE resp_status AS
        WITH r0 AS (SELECT hospitalization_id, CAST(recorded_dttm AS TIMESTAMP) AS t,
                           row_number() OVER (ORDER BY hospitalization_id, recorded_dttm, device_category NULLS LAST,
                                              {', '.join(c.split(' AS ')[-1] + ' NULLS LAST' for c in cols.split(', '))})
                               AS _r,   -- a deterministic order of rows at one time
                           CASE WHEN device_category IS NULL AND mode_category IN ('acvc', 'simv', 'pressure_control')
                                THEN 'imv' ELSE device_category END AS dev, * EXCLUDE (hospitalization_id, device_category)
                    FROM (SELECT hospitalization_id, recorded_dttm, device_category, {cols} FROM respiratory_support)),
        r1 AS (SELECT * REPLACE (CASE WHEN dev IS NULL AND (lag(dev) {w} = 'imv' OR lead(dev) {w} = 'imv')
                                           AND peep_set > 1 AND resp_rate_set > 1 AND tidal_volume_set > 1
                                      THEN 'imv' ELSE dev END AS dev) FROM r0),
        r2 AS (SELECT * REPLACE (CASE WHEN dev IS NULL AND (lag(dev) {w} = 'nippv' OR lead(dev) {w} = 'nippv')
                                           AND peak_inspiratory_pressure_set > 1 AND pressure_support_set > 1
                                      THEN 'nippv' ELSE dev END AS dev) FROM r1),
        r3 AS (SELECT hospitalization_id, t,
                      CASE WHEN dev IS NOT NULL THEN dev
                           WHEN {vent_row} AND last_dev IN ({vent_devices}) THEN last_dev END AS status
               FROM (SELECT *, last_value(dev IGNORE NULLS) {w} AS last_dev FROM r2))
        SELECT hospitalization_id, t, arg_min(status, CASE status {rank} ELSE {len(DEVICE_SUPPORT_ORDER)} END) AS status
        FROM r3 WHERE status IS NOT NULL GROUP BY 1, 2""")


def respiratory_episodes_sql(status_table: str = "resp_status") -> str:
    """Episodes of ``status_table`` (hospitalization_id, t, status; one status per time), mimic-code
    ``ventilation``: a new episode at a status change or 14 h after the previous charting, ending at the next
    charting within 14 h; single-charting episodes dropped. Columns: episode, hospitalization_id, ep, start, end."""
    return f"""SELECT * FROM (
            WITH d AS (SELECT *, CASE WHEN lag(status) OVER w IS NULL OR lag(status) OVER w <> status
                                      OR date_diff('hour', lag(t) OVER w, t) >= {IMV_GAP_HOURS} THEN 1 ELSE 0 END AS new,
                         lead(t) OVER w AS next_t
                  FROM {status_table} WINDOW w AS (PARTITION BY hospitalization_id ORDER BY t)),
            n AS (SELECT *, sum(new) OVER (PARTITION BY hospitalization_id ORDER BY t ROWS UNBOUNDED PRECEDING) AS ep FROM d)
            SELECT any_value(status) AS episode, hospitalization_id, ep, min(t) AS start_dttm,
                   max(CASE WHEN next_t IS NULL OR date_diff('hour', t, next_t) >= {IMV_GAP_HOURS} THEN t ELSE next_t END)
                       AS end_dttm
            FROM n GROUP BY hospitalization_id, ep HAVING min(t) <> max(t))"""


def episodes(con: duckdb.DuckDBPyConnection, have: set[str]) -> None:
    parts = []
    if "respiratory_support" in have:   # mimic-code ventilation (see module docstring)
        resp_status(con)
        parts.append(respiratory_episodes_sql())
    if "medication_admin_continuous" in have:   # a vasoactive infusion runs from its start to its stop
        drugs = ", ".join(f"'{d}'" for d in VASOPRESSORS)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE vaso_events AS
            SELECT hospitalization_id, CAST(admin_dttm AS TIMESTAMP) AS t, med_category, mar_action_category AS act
            FROM medication_admin_continuous WHERE med_category IN ({drugs})""")
        parts.append("""
            SELECT 'vasopressor' AS episode, hospitalization_id, ep, min(t) AS start_dttm, max(t) AS end_dttm FROM (
                SELECT *, sum(new) OVER (PARTITION BY hospitalization_id ORDER BY t, act <> 'stop' ROWS UNBOUNDED PRECEDING) AS ep
                FROM (SELECT *, CASE WHEN running_before = 0 AND act <> 'stop' THEN 1 ELSE 0 END AS new FROM (
                    SELECT *, coalesce(sum(CASE WHEN act = 'start' THEN 1 WHEN act = 'stop' THEN -1 ELSE 0 END)
                                       OVER (PARTITION BY hospitalization_id ORDER BY t, act <> 'stop'
                                             ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS running_before
                    FROM vaso_events)))
            GROUP BY 1, 2, 3""")
    rrt = [f"SELECT hospitalization_id, CAST(recorded_dttm AS TIMESTAMP) AS t FROM {t}"
           for t in ("crrt_therapy", "intermittent_dialysis") if t in have]
    if rrt:
        parts.append(f"SELECT 'rrt' AS episode, * FROM ({_gap_episodes(' UNION ALL '.join(rrt), RRT_GAP_HOURS)})")
    con.execute("CREATE OR REPLACE TEMP TABLE episodes AS " + (" UNION ALL ".join(parts) if parts else
                "SELECT NULL::VARCHAR AS episode, NULL::VARCHAR AS hospitalization_id, NULL::BIGINT AS ep, "
                "NULL::TIMESTAMP AS start_dttm, NULL::TIMESTAMP AS end_dttm WHERE false"))


# -- KDIGO ------------------------------------------------------------------------------------------
def kdigo(con: duckdb.DuckDBPyConnection, have: set[str]) -> None:
    con.execute("""
        CREATE OR REPLACE TEMP TABLE cr AS
        SELECT s.stay_id, CAST(l.lab_collect_dttm AS TIMESTAMP) AS t, avg(l.lab_value_numeric) AS creat
        FROM stays s JOIN labs l ON l.hospitalization_id = s.hospitalization_id
        WHERE l.lab_category = 'creatinine' AND l.lab_value_numeric IS NOT NULL AND l.lab_value_numeric <= 150
          AND CAST(l.lab_collect_dttm AS TIMESTAMP) BETWEEN s.icu_in - INTERVAL 7 DAY AND s.icu_out
        GROUP BY 1, 2""")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE cr_stg AS
        WITH b AS (
            SELECT c.stay_id, c.t, c.creat,
                   (SELECT min(p.creat) FROM cr p WHERE p.stay_id = c.stay_id AND p.t < c.t AND p.t >= c.t - INTERVAL 48 HOUR) AS low48,
                   (SELECT min(p.creat) FROM cr p WHERE p.stay_id = c.stay_id AND p.t < c.t AND p.t >= c.t - INTERVAL 7 DAY) AS low7
            FROM cr c)
        SELECT *, CASE WHEN creat >= low7 * 3.0 THEN 3
                       WHEN creat >= 4 AND (creat >= low48 + 0.3 OR creat >= 1.5 * low7) THEN 3
                       WHEN creat >= low7 * 2.0 THEN 2
                       WHEN creat >= low48 + 0.3 THEN 1
                       WHEN creat >= low7 * 1.5 THEN 1 ELSE 0 END AS aki_stage_creat
        FROM b""")
    if "output" in have:
        irrigation = ("""UNION ALL SELECT hospitalization_id, CAST(recorded_dttm AS TIMESTAMP), -input_volume FROM input
                         WHERE input_category = 'flush_irrigation_urine' AND input_volume > 0""" if "input" in have else "")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE uo AS
            SELECT s.stay_id, s.hospitalization_id, s.icu_in, f.t, sum(f.ml) AS ml
            FROM stays s JOIN (
                SELECT hospitalization_id, CAST(recorded_dttm AS TIMESTAMP) AS t, output_volume AS ml FROM output
                WHERE output_group = 'urine' AND output_volume IS NOT NULL
                {irrigation}) f ON f.hospitalization_id = s.hospitalization_id AND f.t BETWEEN s.icu_in AND s.icu_out
            GROUP BY 1, 2, 3, 4""")
        windows = ", ".join(f"w{h} AS (PARTITION BY stay_id ORDER BY sec RANGE BETWEEN {h * 3600} PRECEDING AND CURRENT ROW)"
                            for h in (6, 12, 24))
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE uo_stg AS
            WITH s1 AS (
                SELECT *, date_diff('second', icu_in, t) AS sec,
                       coalesce(date_diff('second', lag(t) OVER (PARTITION BY stay_id ORDER BY t), t) / 3600.0, 1) AS hrs
                FROM uo),
            s2 AS (
                SELECT *, sum(ml) OVER w6 AS uo6, sum(ml) OVER w12 AS uo12, sum(ml) OVER w24 AS uo24,
                       round(sum(hrs) OVER w6, 6) AS tm6, round(sum(hrs) OVER w12, 6) AS tm12,
                       round(sum(hrs) OVER w24, 6) AS tm24
                FROM s1 WINDOW {windows}),
            k AS ({_weight_in_effect('SELECT * FROM s2', 't')}),
            r AS (SELECT *, CASE WHEN tm6 >= 6 AND tm6 < 12 THEN round(uo6 / kg / tm6, 4) END AS rt6,
                            CASE WHEN tm12 >= 12 THEN round(uo12 / kg / tm12, 4) END AS rt12,
                            CASE WHEN tm24 >= 24 THEN round(uo24 / kg / tm24, 4) END AS rt24
                  FROM k)
            SELECT stay_id, t, kg AS weight, kg_charted_later AS weight_charted_later, rt6, rt12, rt24,
                   CASE WHEN rt6 IS NULL THEN NULL
                        WHEN t <= icu_in + INTERVAL 6 HOUR THEN 0
                        WHEN tm24 >= 24 AND rt24 < 0.3 THEN 3
                        WHEN tm12 >= 12 AND rt12 = 0 THEN 3
                        WHEN tm12 >= 12 AND rt12 < 0.5 THEN 2
                        WHEN tm6 >= 6 AND rt6 < 0.5 THEN 1 ELSE 0 END AS aki_stage_uo
            FROM r""")
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE uo_stg AS SELECT NULL::VARCHAR AS stay_id, NULL::TIMESTAMP AS t, "
                    "NULL::DOUBLE AS weight, NULL::BOOLEAN AS weight_charted_later, NULL::DOUBLE AS rt6, "
                    "NULL::DOUBLE AS rt12, NULL::DOUBLE AS rt24, "
                    "NULL::INTEGER AS aki_stage_uo WHERE false")
    crrt = ("""SELECT DISTINCT s.stay_id, CAST(c.recorded_dttm AS TIMESTAMP) AS t, 3 AS aki_stage_crrt
               FROM stays s JOIN crrt_therapy c ON c.hospitalization_id = s.hospitalization_id
                AND CAST(c.recorded_dttm AS TIMESTAMP) BETWEEN s.icu_in AND s.icu_out""" if "crrt_therapy" in have else
            "SELECT NULL::VARCHAR AS stay_id, NULL::TIMESTAMP AS t, NULL::INTEGER AS aki_stage_crrt WHERE false")
    con.execute(f"CREATE OR REPLACE TEMP TABLE crrt_stg AS {crrt}")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE kdigo AS
        WITH tm AS (SELECT stay_id, t FROM cr_stg UNION SELECT stay_id, t FROM uo_stg UNION SELECT stay_id, t FROM crrt_stg),
        k AS (
            SELECT tm.stay_id, tm.t, c.creat, c.low48 AS creat_low_past_48hr, c.low7 AS creat_low_past_7day,
                   c.aki_stage_creat, u.weight, u.weight_charted_later, u.rt6 AS uo_rt_6hr, u.rt12 AS uo_rt_12hr,
                   u.rt24 AS uo_rt_24hr,
                   u.aki_stage_uo, r.aki_stage_crrt,
                   greatest(coalesce(c.aki_stage_creat, 0), coalesce(u.aki_stage_uo, 0), coalesce(r.aki_stage_crrt, 0))
                       AS aki_stage
            FROM tm LEFT JOIN cr_stg c USING (stay_id, t) LEFT JOIN uo_stg u USING (stay_id, t)
            LEFT JOIN crrt_stg r USING (stay_id, t))
        SELECT *, max(aki_stage) OVER (PARTITION BY stay_id ORDER BY t
                                       RANGE BETWEEN INTERVAL 6 HOUR PRECEDING AND CURRENT ROW) AS aki_stage_smoothed
        FROM k""")


# -- fluid balance ------------------------------------------------------------------------------------
def fluid_balance(con: duckdb.DuckDBPyConnection, have: set[str]) -> None:
    ins = []
    if "input" in have:
        ins.append("SELECT hospitalization_id, CAST(recorded_dttm AS TIMESTAMP) AS t, input_volume AS ml FROM input")
    if "medication_admin_intermittent" in have:
        ins.append("""SELECT hospitalization_id, CAST(admin_dttm AS TIMESTAMP) AS t, med_dose AS ml
                      FROM medication_admin_intermittent WHERE lower(med_dose_unit) = 'ml'""")
    if "medication_admin_continuous" in have:   # ml/hr steps integrated until the next change of the stream
        ins.append("""
            SELECT hospitalization_id, t, rate * epoch(least(next_t, t + INTERVAL 24 HOUR) - t) / 3600 AS ml FROM (
                SELECT hospitalization_id, CAST(admin_dttm AS TIMESTAMP) AS t,
                       med_dose * CASE lower(med_dose_unit) WHEN 'ml/hr' THEN 1 WHEN 'ml/min' THEN 60
                                                            WHEN 'l/hr' THEN 1000 WHEN 'l/min' THEN 60000 END AS rate,
                       lead(CAST(admin_dttm AS TIMESTAMP)) OVER (PARTITION BY hospitalization_id, med_category,
                                                                 med_dose_unit ORDER BY admin_dttm) AS next_t
                FROM medication_admin_continuous
                WHERE lower(med_dose_unit) IN ('ml/hr', 'ml/min', 'l/hr', 'l/min'))   -- stops end a step
            WHERE next_t IS NOT NULL AND rate > 0""")
    fin = " UNION ALL ".join(ins) or "SELECT NULL::VARCHAR AS hospitalization_id, NULL::TIMESTAMP AS t, NULL::DOUBLE AS ml WHERE false"
    out = ("SELECT hospitalization_id, CAST(recorded_dttm AS TIMESTAMP) AS t, output_volume AS ml FROM output"
           if "output" in have else "SELECT NULL::VARCHAR AS hospitalization_id, NULL::TIMESTAMP AS t, NULL::DOUBLE AS ml WHERE false")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fluid_balance AS
        WITH f AS (SELECT 'in' AS dir, * FROM ({fin}) UNION ALL SELECT 'out', * FROM ({out})),
        d AS (SELECT s.stay_id, CAST(floor(epoch(f.t - s.icu_in) / 86400) AS INTEGER) AS icu_day, f.dir, f.ml
              FROM stays s JOIN f ON f.hospitalization_id = s.hospitalization_id AND f.t >= s.icu_in AND f.t < s.icu_out)
        SELECT stay_id, icu_day, round(coalesce(sum(ml) FILTER (WHERE dir = 'in'), 0), 1) AS fluid_in_ml,
               round(coalesce(sum(ml) FILTER (WHERE dir = 'out'), 0), 1) AS fluid_out_ml,
               round(coalesce(sum(ml) FILTER (WHERE dir = 'in'), 0) - coalesce(sum(ml) FILTER (WHERE dir = 'out'), 0), 1) AS net_ml
        FROM d GROUP BY 1, 2""")


# -- driver -------------------------------------------------------------------------------------------
OUTPUTS = ("nee", "episodes", "kdigo", "fluid_balance", "sofa_24h", "ase")


def derive(clif_dir: Path, out_dir: Path, work_dir: Path, with_clifpy: bool = True,
           min_age_years: float | None = 18.0) -> dict:
    """Write every derived concept of one dataset to ``out_dir``; returns a summary. ``with_clifpy`` = False
    skips the clifpy concepts (SOFA, ASE)."""
    for name in OUTPUTS:   # a concept the inputs no longer support must not survive from an earlier run
        (out_dir / f"{name}.parquet").unlink(missing_ok=True)
    con = connect()
    con.execute("SET TimeZone = 'UTC'")
    _stays(con, clif_dir, min_age_years)
    have = {t for t in ("vitals", "labs", "respiratory_support", "medication_admin_continuous",
                        "medication_admin_intermittent", "crrt_therapy", "intermittent_dialysis", "input", "output")
            if _table(con, clif_dir, t)}
    if "vitals" in have:
        _weights(con)
    else:                                   # no vitals in this batch of stays: no weights (nee() reads both tables)
        con.execute("CREATE OR REPLACE TEMP TABLE weights (hospitalization_id VARCHAR, t TIMESTAMP, kg DOUBLE)")
        con.execute("CREATE OR REPLACE TEMP TABLE first_weight (hospitalization_id VARCHAR, kg DOUBLE)")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"stays": con.execute("SELECT count(*) FROM stays").fetchone()[0]}
    if "medication_admin_continuous" in have:
        nee(con)
        con.execute(f"COPY nee TO '{_p(out_dir / 'nee.parquet')}' (FORMAT parquet)")
        summary["nee_rows"] = con.execute("SELECT count(*) FROM nee").fetchone()[0]
    episodes(con, have)
    con.execute(f"""COPY (SELECT s.stay_id, e.episode, e.start_dttm, e.end_dttm FROM episodes e JOIN stays s
                          ON s.hospitalization_id = e.hospitalization_id
                          AND e.start_dttm <= s.icu_out AND e.end_dttm >= s.icu_in - INTERVAL 48 HOUR)
                    TO '{_p(out_dir / 'episodes.parquet')}' (FORMAT parquet)""")
    summary["stays_with_episode"] = dict(con.execute(f"""
        SELECT episode, count(DISTINCT stay_id) FROM read_parquet('{_p(out_dir / 'episodes.parquet')}') GROUP BY 1""").fetchall())
    if "labs" in have:
        kdigo(con, have)
        con.execute(f"COPY kdigo TO '{_p(out_dir / 'kdigo.parquet')}' (FORMAT parquet)")
        summary["max_aki_stage"] = dict(con.execute("""SELECT stage, count(*) FROM (
            SELECT s.stay_id, coalesce(max(k.aki_stage), 0) AS stage FROM stays s LEFT JOIN kdigo k USING (stay_id)
            GROUP BY 1) GROUP BY 1 ORDER BY 1""").fetchall())
    fluid_balance(con, have)
    con.execute(f"COPY fluid_balance TO '{_p(out_dir / 'fluid_balance.parquet')}' (FORMAT parquet)")
    if with_clifpy:
        sofa = sofa_window(clif_dir, work_dir, keep_inputs=True)
        sofa = sofa.filter(sofa["stay_id"].is_in([r[0] for r in con.execute("SELECT stay_id FROM stays").fetchall()]))
        summary["sofa_24h_resp_raised_over_clifpy"] = int((sofa["sofa_resp"] != sofa["clifpy_sofa_resp"]).sum())
        sofa = sofa.select(["stay_id", *sorted(c for c in sofa.columns if c.startswith("sofa_"))])
        sofa.write_parquet(out_dir / "sofa_24h.parquet")
        summary["sofa_24h_median"] = float(sofa["sofa_total"].median()) if len(sofa) else None
        ase = adult_sepsis_events(clif_dir, work_dir)
        if ase is not None:   # each blood culture with the stays whose window (48 h before ICU admission) holds it
            con.register("ase_raw", ase)
            con.execute(f"""COPY (SELECT s.stay_id, a.* FROM ase_raw a LEFT JOIN stays s
                                  ON s.hospitalization_id = a.hospitalization_id
                                  AND CAST(a.blood_culture_dttm AS TIMESTAMP) BETWEEN s.icu_in - INTERVAL 48 HOUR AND s.icu_out)
                            TO '{_p(out_dir / 'ase.parquet')}' (FORMAT parquet)""")
            summary["stays_with_ase"] = dict(zip(("sepsis", "sepsis_wo_lactate", "presumed_infection"), con.execute(f"""
                SELECT count(DISTINCT stay_id) FILTER (WHERE sepsis = 1), count(DISTINCT stay_id) FILTER (WHERE sepsis_wo_lactate = 1),
                       count(DISTINCT stay_id) FILTER (WHERE presumed_infection = 1)
                FROM read_parquet('{_p(out_dir / 'ase.parquet')}')""").fetchone()))
    (out_dir / "_derived_report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
