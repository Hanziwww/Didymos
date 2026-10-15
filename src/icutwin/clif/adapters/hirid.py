"""HiRID 1.1.1 -> CLIF (our adapter; no published CLIF ETL exists for HiRID).

HiRID covers single ICU admissions of the Bern University Hospital mixed medical-surgical ICU; there is no
hospital-level or cross-admission patient identity, so patient = hospitalization = ICU stay = ``patientid``.

Conventions (each mapping row in ``clif/maps/hirid/`` names its provenance: ricu 0.6.3 concepts and the
HiRID-ICU-Benchmark variable reference by the dataset's authors):
* times are date-shifted but keep the real time of day (``clock_valid``; GCS charting peaks at the 8-hour
  nursing shifts, admissions in the afternoon); stored as UTC wall-clock time;
* ICU admission = ``general_table.admissiontime``; ICU end = last observation of any variable (ricu
  ``id_win_helper.hirid_env``); death time = last heart-rate / MAP observation of patients discharged dead
  (ricu ``hirid_death``);
* age is reported in 5-year bins (``age`` = lower bound, open top bin 90): stored as the bin midpoint
  (lower bound + 2; the open top bin keeps its lower bound, as CLIF-AmsterdamUMCdb does for "80+"),
  flagged ``demographics_binned`` in ``ext_icu_stay``;
* lab values are converted from SI units with ricu's factors; laboratory time = sample time, result time =
  entry time;
* an observation re-entered with the same value (same patient, variable and time; lab results are re-sent with
  later entry times) is kept once, with its first entry time;
* observations flagged invalidated (status bit 2) or notified-not-measured (bit 32) are dropped; values
  flagged "bigger/smaller than" (bits 64/128, censored results such as '< 0.003') keep their bound;
* ordinal variables are decoded with ``ordinal_vars_ref`` (nasal O2 flow codes, ventilator mode codes,
  airway code); the medication part (pharma records) is added with the shared drug lexicon.
"""

from __future__ import annotations

from icutwin.clif.adapters.base import ClifAdapter
from icutwin.clif.dx_groups import group_sql

HOSPITAL_ID = "hirid_bern"
URINE_RATE_MAX_HOURS = 12   # a derived urine rate stands for at most this long before its report
# observation status bits (HiRID schemata.pdf): 2 = invalidated, 32 = notified, not measured
VALID_OBS = "(TRY_CAST(status AS INTEGER) IS NULL OR (TRY_CAST(status AS INTEGER) & 34) = 0)"
RESP_COLS = ("fio2_set", "lpm_set", "tidal_volume_set", "resp_rate_set", "peep_set", "tidal_volume_obs",
             "resp_rate_obs", "plateau_pressure_obs", "peak_inspiratory_pressure_obs", "peep_obs",
             "mean_airway_pressure_obs")


# CLIF route categories (mCIDE): 'im' includes subcutaneous; inhaled doses take the local intermittent route
# extension 'inhaled' (extensions.yaml); rectal, transcutaneous, epidural and paravertebral administrations have no
# CLIF route category and are not CLIF administrations
ROUTE = """CASE WHEN route IN ('iv-inf', 'iv-inj', 'cv-inf', 'cv-inj', 'iv') THEN 'iv'
                WHEN route IN ('im', 'i.m.', 'sc-inj') THEN 'im'
                WHEN route IN ('p os', 'p.os', 'p.o', 'ms', 'p.ms', 'enteral', 'po', 'sonde') THEN 'enteral'
                WHEN route IN ('sl', 'bucc', 'subling') THEN 'buccal_sublingual'
                WHEN route = 'inhal' THEN 'inhaled' END"""
# input categories for HiRID fluid records (typeid 0) by subtype (HiRID schemata.pdf: 1 blood product,
# 2 colloid, 3 enteral); crystalloids (0) and parenteral nutrition (4) are continuous medications
INPUT_SUBTYPES = {1: ("blood_products", "blood_products"), 2: ("iv_colloid", "iv_fluids"),
                  3: ("enteral_nutrition", "nutrition")}


class HiridAdapter(ClifAdapter):
    dataset = "hirid"
    tables = (
        "patient", "hospitalization", "adt", "vitals", "labs", "respiratory_support", "patient_assessments",
        "medication_admin_continuous", "medication_admin_intermittent", "input",
        "invasive_hemodynamics", "output", "crrt_therapy", "ext_icu_stay",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m = self.map("observations")
        raw = self.src("raw_stage/observation_tables/csv/*.parquet")
        # an observation re-entered with the same value (same variable and time, later entertime: lab results are
        # re-sent, one creatinine draw up to seven times over four days) is one observation, available from its
        # first entry
        self.con.execute(f"""
            CREATE OR REPLACE TABLE obs AS
            SELECT pid, t, min(et) AS et, vid, v FROM (
                SELECT CAST(patientid AS BIGINT) AS pid,
                       CAST(CAST(datetime AS TIMESTAMP) AS TIMESTAMPTZ) AS t,
                       CAST(CAST(entertime AS TIMESTAMP) AS TIMESTAMPTZ) AS et,
                       CAST(variableid AS BIGINT) AS vid,
                       coalesce(TRY_CAST(value AS DOUBLE), TRY_CAST(stringvalue AS DOUBLE)) AS v
                FROM {raw}
                WHERE CAST(variableid AS BIGINT) IN (SELECT CAST(variableid AS BIGINT) FROM {m}) AND {VALID_OBS})
            GROUP BY pid, t, vid, v""")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE stays AS
            WITH last_obs AS (
                SELECT CAST(patientid AS BIGINT) AS pid, max(CAST(datetime AS TIMESTAMP)) AS t_last,
                       max(CAST(datetime AS TIMESTAMP)) FILTER (WHERE variableid IN ('200', '110')) AS t_vital
                FROM {raw} WHERE {VALID_OBS} GROUP BY 1)
            SELECT CAST(g.patientid AS BIGINT) AS pid, CAST(g.admissiontime AS TIMESTAMPTZ) AS icu_in,
                   CAST(greatest(l.t_last, g.admissiontime) AS TIMESTAMPTZ) AS icu_out,
                   CASE WHEN g.discharge_status = 'dead' THEN CAST(coalesce(l.t_vital, l.t_last) AS TIMESTAMPTZ) END AS death,
                   g.sex, CAST(g.age AS INTEGER) AS age_bin,
                   CAST(g.age AS INTEGER) + CASE WHEN CAST(g.age AS INTEGER) >= 90 THEN 0 ELSE 2 END AS age_mid,
                   g.discharge_status
            FROM {self.src('reference/general_table.parquet')} g LEFT JOIN last_obs l ON l.pid = CAST(g.patientid AS BIGINT)""")

    def _target(self, target: str) -> str:
        """Observations of one mapping target joined with the map (alias ``m``) and the stay (alias ``s``)."""
        m = self.map("observations")
        return f"""FROM obs o JOIN {m} m ON CAST(m.variableid AS BIGINT) = o.vid AND m.target = '{target}'
                   JOIN stays s ON s.pid = o.pid"""

    # -- core tables --------------------------------------------------------------------------------
    def sql_patient(self):
        sql = """
            SELECT CAST(pid AS VARCHAR) AS patient_id,
                   CAST(icu_in - to_days(CAST(round(age_mid * 365.25) AS INTEGER)) AS DATE) AS birth_date,
                   death AS death_dttm,
                   sex AS sex_name, CASE sex WHEN 'M' THEN 'male' WHEN 'F' THEN 'female' ELSE 'unknown' END AS sex_category,
                   'unknown' AS race_category, 'unknown' AS ethnicity_category, 'unknown' AS language_category
            FROM stays"""
        return sql, ("patient_id",)

    def sql_hospitalization(self):
        # ICU-only dataset: the hospitalization is the ICU admission; the hospital outcome is known only for deaths
        sql = """
            SELECT CAST(pid AS VARCHAR) AS patient_id, CAST(pid AS VARCHAR) AS hospitalization_id,
                   icu_in AS admission_dttm, icu_out AS discharge_dttm, age_mid AS age_at_admission,
                   discharge_status AS discharge_name,
                   CASE WHEN discharge_status = 'dead' THEN 'expired' ELSE 'missing' END AS discharge_category
            FROM stays"""
        return sql, ("hospitalization_id",)

    def sql_adt(self):
        sql = f"""
            SELECT CAST(pid AS VARCHAR) AS hospitalization_id, '{HOSPITAL_ID}' AS hospital_id,
                   'academic' AS hospital_type, icu_in AS in_dttm, icu_out AS out_dttm,
                   'Department of Intensive Care Medicine' AS location_name, 'icu' AS location_category,
                   'general_icu' AS location_type          -- mixed medical-surgical ICU
            FROM stays"""
        return sql, ("hospitalization_id", "in_dttm")

    def sql_vitals(self):
        sql = f"""
            SELECT CAST(o.pid AS VARCHAR) AS hospitalization_id, o.t AS recorded_dttm,
                   CAST(o.vid AS VARCHAR) AS vital_name, m.category AS vital_category,
                   o.v * CAST(m.factor AS DOUBLE) AS vital_value, nullif(m.meas_site_name, '') AS meas_site_name
            {self._target('vitals')} WHERE o.v IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_labs(self):
        sql = f"""
            SELECT CAST(o.pid AS VARCHAR) AS hospitalization_id,
                   o.t AS lab_order_dttm, o.t AS lab_collect_dttm,
                   CASE WHEN o.et >= o.t THEN o.et ELSE o.t END AS lab_result_dttm,
                   m.lab_order_category, CAST(o.vid AS VARCHAR) AS lab_name, m.category AS lab_category,
                   CAST(o.v AS VARCHAR) AS lab_value, o.v * CAST(m.factor AS DOUBLE) AS lab_value_numeric,
                   m.reference_unit, 'plasma_blood' AS lab_specimen_category
            {self._target('labs')} WHERE o.v IS NOT NULL"""
        return sql, ("hospitalization_id", "lab_result_dttm")

    def sql_respiratory_support(self):
        codes = self.map("vent_codes")
        ordinal = self.src("reference/ordinal_vars_ref.parquet")
        cols = ", ".join(f"max(val) FILTER (WHERE col = '{c}') AS {c}" for c in RESP_COLS)
        sql = f"""
            WITH num AS (
                SELECT o.pid, o.t, m.category AS col, o.v * CAST(m.factor AS DOUBLE) AS val
                {self._target('resp')} WHERE o.v IS NOT NULL
                UNION ALL   -- nasal cannula flow: ordinal codes ('2 x 3' = two sources of 3 l/min)
                SELECT o.pid, o.t, 'lpm_set',
                       CASE WHEN r.stringvalue LIKE '% x %'
                            THEN CAST(split_part(r.stringvalue, ' x ', 1) AS DOUBLE) * CAST(split_part(r.stringvalue, ' x ', 2) AS DOUBLE)
                            ELSE TRY_CAST(r.stringvalue AS DOUBLE) END
                {self._target('resp_o2_nasal')}
                JOIN {ordinal} r ON r.variableid = o.vid AND r.code = CAST(o.v AS BIGINT)
                UNION ALL
                SELECT o.pid, o.t, 'lpm_set', o.v {self._target('resp_o2_mask')} WHERE o.v IS NOT NULL),
            wide AS (SELECT pid, t, {cols} FROM num GROUP BY pid, t),
            dev AS (   -- airway code and NIV/CPAP mode codes document the device; O2 flows document their device
                SELECT o.pid, o.t, max(c.device_category) AS device_category,
                       max(CAST(c.tracheostomy AS INTEGER)) AS tracheostomy,
                       max(c.mode_category) AS mode_category,
                       max(CASE WHEN o.vid = 3845 THEN r.stringvalue END) AS mode_name,
                       max(CASE WHEN o.vid = 15001552 THEN r.stringvalue END) AS device_name
                FROM obs o JOIN {codes} c ON CAST(c.variableid AS BIGINT) = o.vid AND CAST(c.code AS BIGINT) = CAST(o.v AS BIGINT)
                LEFT JOIN {ordinal} r ON r.variableid = o.vid AND r.code = CAST(o.v AS BIGINT)
                GROUP BY 1, 2
                UNION ALL
                SELECT o.pid, o.t, CASE m.target WHEN 'resp_o2_nasal' THEN 'nasal_cannula' ELSE 'face_mask' END,
                       NULL, NULL, NULL,
                       CASE m.target WHEN 'resp_o2_nasal' THEN 'O2 nasal cannula' ELSE 'O2 mask' END
                FROM obs o JOIN {self.map('observations')} m ON CAST(m.variableid AS BIGINT) = o.vid
                WHERE m.target IN ('resp_o2_nasal', 'resp_o2_mask')),
            dev_at AS (
                SELECT pid, t, max(device_category) AS device_category, max(tracheostomy) AS tracheostomy,
                       max(mode_category) AS mode_category, max(mode_name) AS mode_name, max(device_name) AS device_name
                FROM dev GROUP BY 1, 2),
            keys AS (SELECT pid, t FROM wide UNION SELECT pid, t FROM dev_at)
            SELECT CAST(k.pid AS VARCHAR) AS hospitalization_id, k.t AS recorded_dttm,
                   d.device_name, d.device_category, d.mode_name, d.mode_category, d.tracheostomy,
                   {', '.join(f'w.{c}' for c in RESP_COLS)}
            FROM keys k LEFT JOIN wide w ON w.pid = k.pid AND w.t = k.t
            LEFT JOIN dev_at d ON d.pid = k.pid AND d.t = k.t"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_patient_assessments(self):
        group = {"gcs_eye": "neurological", "gcs_motor": "neurological", "gcs_verbal": "neurological",
                 "rass": "sedation_or_agitation", "tof": "neurological"}
        grp = "CASE m.category " + " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in group.items()) + " END"
        sql = f"""
            SELECT CAST(o.pid AS VARCHAR) AS hospitalization_id, o.t AS recorded_dttm,
                   CAST(o.vid AS VARCHAR) AS assessment_name, m.category AS assessment_category,
                   {grp} AS assessment_group,
                   CASE WHEN m.category = 'rass' THEN floor(o.v) ELSE o.v END AS numerical_value   -- ricu: floor(RASS)
            {self._target('assessments')} WHERE o.v IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    # -- medications and fluids (pharma records) ----------------------------------------------------
    def _pharma(self) -> str:
        """Typed pharma records with name and lexicon resolution (table ``ph``).

        recordstatus bits (HiRID schemata.pdf): 2 invalidated and 32 notified-not-administered are dropped
        (the HiRID benchmark treats 544 = 512 + 32 as given; the schema says it was not administered);
        4 start, 256 stop (start + stop = instantaneous administration), 8 record, 512 included in reports.
        """
        if self.con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'ph'").fetchone()[0]:
            return "ph"
        ref = self.src("reference/hirid_variable_reference.parquet")
        raw = self.src("raw_stage/pharma_records/csv/*.parquet")
        names = f"""SELECT lower("Variable Name") AS txt FROM {ref} WHERE "Source Table" = 'Pharma'"""
        lk = self.regex_lookup("drugs", names, "drugs", ("ingredient", "continuous", "intermittent"),
                               table=self.lexicon("drugs"))
        self.con.execute(f"""
            CREATE OR REPLACE TABLE ph AS
            SELECT CAST(p.patientid AS BIGINT) AS pid, r."Variable Name" AS name, CAST(p.pharmaid AS BIGINT) AS pharmaid,
                   CAST(p.infusionid AS BIGINT) AS inf,
                   CAST(CAST(p.givenat AS TIMESTAMP) AS TIMESTAMPTZ) AS t, CAST(p.enteredentryat AS TIMESTAMP) AS et,
                   TRY_CAST(p.givendose AS DOUBLE) * CASE WHEN p.doseunit = 'MU' THEN 1e6 ELSE 1 END AS dose,
                   TRY_CAST(p.fluidamount_calc AS DOUBLE) AS fluid,
                   CASE p.doseunit WHEN 'µg' THEN 'mcg' WHEN 'U' THEN 'u' WHEN 'IE' THEN 'u' WHEN 'MU' THEN 'u'
                                   ELSE lower(p.doseunit) END AS unit,
                   p.route, CAST(TRY_CAST(p.typeid AS DOUBLE) AS INTEGER) AS typeid,
                   CAST(TRY_CAST(p.subtypeid AS DOUBLE) AS INTEGER) AS subtypeid,
                   CAST(p.recordstatus AS INTEGER) AS st, l.ingredient,
                   nullif(l.continuous, '') AS continuous, nullif(l.intermittent, '') AS intermittent
            FROM {raw} p
            LEFT JOIN {ref} r ON r."Source Table" = 'Pharma' AND r.ID = CAST(p.pharmaid AS BIGINT)
            LEFT JOIN {lk} l ON l.txt = lower(r."Variable Name")
            WHERE (CAST(p.recordstatus AS INTEGER) & 34) = 0""")
        self.coverage("medication", "pharma_name", """
            SELECT coalesce(name, '(no reference name)') AS txt, count(*) AS n,
                   bool_or(ingredient IS NOT NULL OR typeid = 0) AS mapped
            FROM ph GROUP BY 1""")
        return "ph"

    def sql_medication_admin_intermittent(self):
        ph = self._pharma()
        sql = f"""
            SELECT CAST(pid AS VARCHAR) AS hospitalization_id, t AS admin_dttm, name AS med_name,
                   intermittent AS med_category, {self._med_group('medication_admin_intermittent', 'intermittent')} AS med_group,
                   route AS med_route_name, {ROUTE} AS med_route_category, dose AS med_dose, unit AS med_dose_unit,
                   'given' AS mar_action_category, 'administered' AS mar_action_group
            FROM {ph} WHERE (st & 260) = 260 AND intermittent IS NOT NULL AND dose > 0 AND ({ROUTE}) IS NOT NULL"""
        return sql, ("hospitalization_id", "admin_dttm")

    def sql_medication_admin_continuous(self):
        """Rate steps per infusion, summed over concurrent infusions of the same drug and unit.

        An infusion is a (pharmaid, infusionid) pair, as the HiRID benchmark processes them: one infusion id can
        carry records of several drugs (an insulin infusion and its glucose carrier).

        Within an infusion, events are the records with start/stop/report bits (pump actions) plus its first and
        last records; the dose recorded at a record was given since the previous record, so the rate of the
        segment starting at event k = doses recorded in (T_k, T_k+1] / (T_k+1 - T_k) (HiRID benchmark
        ``process_single_infusion``), also after a stop record (a dose recorded there was given). The infusion
        stops at its last record. As in the benchmark, an infusion without a start record starts one acting
        period (``maps/hirid/acting_periods.csv``) before its first record, without dose, so that record's dose
        has an interval; the dose recorded at a start record has none and is not given a rate. Records of one
        infusion at one time are one record, as the benchmark's ``drop_duplicates_pharma`` keeps them: a stop
        record wins; of several stops, the only one with a dose, else the last entered.
        """
        ph = self._pharma()
        acting = self.map("acting_periods")
        sql = f"""
            WITH r00 AS (
                SELECT * FROM {ph}
                WHERE (st & 260) <> 260 AND continuous IS NOT NULL AND {ROUTE} = 'iv' AND dose IS NOT NULL),
            r0 AS (
                SELECT * EXCLUDE (n_dosed_stops) FROM (
                    SELECT *, count(*) FILTER (WHERE (st & 256) <> 0 AND dose <> 0)
                                  OVER (PARTITION BY pid, pharmaid, inf, t) AS n_dosed_stops
                    FROM r00)
                QUALIFY row_number() OVER (PARTITION BY pid, pharmaid, inf, t
                                           ORDER BY (st & 256) <> 0 DESC,
                                                    CASE WHEN n_dosed_stops = 1 THEN dose <> 0 END DESC NULLS LAST,
                                                    et DESC NULLS LAST, st DESC, dose DESC, fluid DESC) = 1),
            f AS (SELECT *, row_number() OVER (PARTITION BY pid, pharmaid, inf ORDER BY t, st, dose, fluid) AS rn,
                         bool_or((st & 4) <> 0) OVER (PARTITION BY pid, pharmaid, inf) AS has_start
                  FROM r0),
            r AS (
                SELECT * FROM r0
                UNION ALL   -- the benchmark's start record of an infusion without one
                SELECT f.* EXCLUDE (rn, has_start)
                           REPLACE (f.t - to_minutes(CAST(a.minutes AS BIGINT)) AS t, 0.0 AS dose, 0.0 AS fluid, 4 AS st)
                FROM f JOIN {acting} a ON CAST(a.pharmaid AS BIGINT) = f.pharmaid
                WHERE f.rn = 1 AND NOT f.has_start),
            o AS (
                SELECT *, ((st & 772) <> 0 OR row_number() OVER w = 1
                           OR row_number() OVER (PARTITION BY pid, pharmaid, inf ORDER BY t DESC, st DESC, dose DESC, fluid DESC) = 1)
                              AS is_event,
                       row_number() OVER (PARTITION BY pid, pharmaid, inf ORDER BY t DESC, st DESC, dose DESC, fluid DESC) = 1
                           AS is_last
                FROM r WINDOW w AS (PARTITION BY pid, pharmaid, inf ORDER BY t, st, dose, fluid)),   -- only identical rows tie
            s AS (   -- segment index of each record's dose: number of events strictly before it
                SELECT *, sum(CAST(is_event AS INTEGER)) OVER (PARTITION BY pid, pharmaid, inf ORDER BY t, st, dose, fluid
                                                               ROWS UNBOUNDED PRECEDING) AS ev
                FROM o),
            ev AS (SELECT pid, pharmaid, inf, continuous, unit, name, ev AS k, t AS t0, (st & 256) <> 0 AS is_stop, is_last
                   FROM s WHERE is_event),
            seg AS (
                SELECT e.pid, e.pharmaid, e.inf, e.continuous, e.unit, e.name, e.k, e.t0, e.is_stop, e.is_last,
                       lead(e.t0) OVER (PARTITION BY e.pid, e.pharmaid, e.inf ORDER BY e.k) AS t1
                FROM ev e),
            dose AS (SELECT pid, pharmaid, inf, ev - CAST(is_event AS INTEGER) AS k, sum(dose) AS d FROM s GROUP BY ALL),
            rates AS (   -- rate of each segment (a stop's segment has none unless a dose is recorded in it); 0 from the last record
                SELECT g.pid, g.pharmaid, g.inf, g.continuous, g.unit, g.name, g.t0 AS t, g.k,
                       CASE WHEN g.t1 IS NULL THEN 0.0
                            ELSE coalesce(coalesce(d.d, 0) / nullif(date_diff('millisecond', g.t0, g.t1) / 60000.0, 0), 0)
                       END AS rate,
                       (g.t1 IS NULL AND NOT g.is_stop) AS inferred_stop
                FROM seg g LEFT JOIN dose d ON d.pid = g.pid AND d.pharmaid = g.pharmaid AND d.inf = g.inf AND d.k = g.k),
            deltas AS (   -- change of each infusion's rate, summed over infusions changing at the same time, in
                          -- exact decimals: the running total is exactly 0 once every infusion has stopped, and a
                          -- small rate keeps its digits (a floating-point total rounded to 6 decimals lost them, #98)
                SELECT pid, continuous, unit, t, min(name) AS name, bool_or(inferred_stop) AS inferred_stop,
                       sum(CAST(rate AS DECIMAL(38, 18)) - CAST(coalesce(prev, 0) AS DECIMAL(38, 18))) AS delta
                FROM (SELECT *, lag(rate) OVER (PARTITION BY pid, pharmaid, inf ORDER BY t, k) AS prev FROM rates)
                GROUP BY 1, 2, 3, 4),
            total AS (
                SELECT *, CAST(sum(delta) OVER (PARTITION BY pid, continuous, unit ORDER BY t
                                                ROWS UNBOUNDED PRECEDING) AS DOUBLE) AS total
                FROM deltas),
            chg AS (SELECT *, lag(total, 1, 0) OVER (PARTITION BY pid, continuous, unit ORDER BY t) AS prev_total FROM total)
            SELECT CAST(pid AS VARCHAR) AS hospitalization_id, t AS admin_dttm, name AS med_name,
                   continuous AS med_category, {self._med_group('medication_admin_continuous', 'continuous')} AS med_group,
                   'iv' AS med_route_category, greatest(total, 0) AS med_dose, unit || '/min' AS med_dose_unit,
                   CASE WHEN total <= 0 AND inferred_stop THEN 'inferred stop: end of infusion records' END AS mar_action_name,
                   CASE WHEN total <= 0 THEN 'stop' WHEN prev_total <= 0 THEN 'start' ELSE 'dose_change' END AS mar_action_category,
                   'administered' AS mar_action_group
            FROM chg WHERE total IS DISTINCT FROM prev_total"""
        return sql, ("hospitalization_id", "admin_dttm", "med_category")

    def _med_group(self, table: str, column: str) -> str:
        from icutwin.clif.conform import extensions, table_schema
        mapping = dict(table_schema(table).get("med_category_to_group_mapping") or {})
        for cat, spec in ((extensions().get(table) or {}).get("med_category") or {}).items():
            if isinstance(spec, dict) and spec.get("group"):
                mapping.setdefault(cat, spec["group"])
        cases = " ".join(f"WHEN '{k}' THEN '{v if isinstance(v, str) else v[0]}'" for k, v in mapping.items())
        return f"CASE {column} {cases} END"

    def sql_input(self):
        ph = self._pharma()
        cats = " ".join(f"WHEN {k} THEN '{c}'" for k, (c, _) in INPUT_SUBTYPES.items())
        grps = " ".join(f"WHEN {k} THEN '{g}'" for k, (_, g) in INPUT_SUBTYPES.items())
        sql = f"""
            SELECT CAST(pid AS VARCHAR) AS hospitalization_id, t AS recorded_dttm, name AS input_name,
                   CASE subtypeid {cats} END AS input_category, CASE subtypeid {grps} END AS input_group,
                   CASE WHEN unit = 'ml' THEN dose ELSE fluid END AS input_volume
            FROM {ph}
            WHERE typeid = 0 AND subtypeid IN ({', '.join(str(k) for k in INPUT_SUBTYPES)})
              AND continuous IS NULL AND CASE WHEN unit = 'ml' THEN dose ELSE fluid END > 0
            UNION ALL   -- crystalloid and colloid infusions are cumulative observations (not pharma records):
                        -- positive increments, as ricu reads HiRID's cumulative fluid counters
            SELECT CAST(pid AS VARCHAR), t, CAST(vid AS VARCHAR), category, NULL, d FROM (
                SELECT o.pid, o.t, o.vid, m.category,
                       coalesce(o.v - lag(o.v) OVER (PARTITION BY o.pid, o.vid ORDER BY o.t), o.v) AS d
                {self._target('input_cumulative')})
            WHERE d > 0"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_invasive_hemodynamics(self):
        sql = f"""
            SELECT CAST(o.pid AS VARCHAR) AS hospitalization_id, o.t AS recorded_dttm,
                   CAST(o.vid AS VARCHAR) AS measurement_name, m.category AS measurement_category,
                   o.v AS measurement_value
            {self._target('hemodynamics')} WHERE o.v IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_output(self):
        """Output volumes. The urine item 10020000 is a derived rate (ml/h) reported at each urine measurement:
        the volume is the rate x the hours since the previous report (at most 12 h; 1 h for the first). The
        cumulative output counter (30005110, ricu's urine source) also counts drains and resets daily."""
        sql = f"""
            WITH o AS (
                SELECT o.*, m.category, m.reference_unit,
                       epoch(o.t - lag(o.t) OVER (PARTITION BY o.pid, o.vid ORDER BY o.t)) / 3600 AS gap_h
                {self._target('output')})
            SELECT CAST(pid AS VARCHAR) AS hospitalization_id, t AS recorded_dttm,
                   CAST(vid AS VARCHAR) AS output_name, category AS output_category,
                   CASE WHEN category LIKE 'urin%' THEN 'urine' ELSE 'drains' END AS output_group,
                   CASE WHEN reference_unit = 'ml/h' THEN v * least(coalesce(gap_h, 1), {URINE_RATE_MAX_HOURS})
                        ELSE v END AS output_volume
            FROM o WHERE v > 0"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_crrt_therapy(self):
        # Prismaflex haemofiltration documented as running (1); the CRRT mode is not recorded
        sql = f"""
            SELECT CAST(o.pid AS VARCHAR) AS hospitalization_id, o.t AS recorded_dttm,
                   'Haemofiltration (prisma)' AS crrt_mode_name
            {self._target('crrt')} WHERE o.v = 1"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_ext_icu_stay(self):
        """Admission diagnosis: the APACHE II group when it is mapped, else the APACHE IV group, as the
        HiRID-ICU-Benchmark phenotyping task (``maps/hirid/apache_groups.csv``, its appendix table 7), from its
        first entry within 24 h of admission (subset: median 2.4 h). The group gives the operative status of
        ``adm_dx_group``; ``surgical_status`` stays unknown (no urgency)."""
        self.con.execute(f"""
            CREATE OR REPLACE TABLE obs_apache AS
            SELECT CAST(patientid AS BIGINT) AS pid, CAST(CAST(datetime AS TIMESTAMP) AS TIMESTAMPTZ) AS t,
                   CAST(variableid AS BIGINT) AS vid, TRY_CAST(value AS DOUBLE) AS v
            FROM {self.src('raw_stage/observation_tables/csv/*.parquet')} WHERE variableid IN ('9990002', '9990004')""")
        g = self.map("apache_groups")
        self.coverage("ext_icu_stay", "apache_groups", f"""
            SELECT CAST(o.vid AS VARCHAR) || ':' || CAST(CAST(o.v AS INTEGER) AS VARCHAR) AS txt, count(*) AS n,
                   bool_or(g.code IS NOT NULL) AS mapped
            FROM obs_apache o LEFT JOIN {g} g ON CAST(g.variableid AS BIGINT) = o.vid AND CAST(g.code AS DOUBLE) = o.v
            GROUP BY 1""")
        sql = f"""
            WITH first AS (   -- the first entry, within 24 h of admission (admission documentation)
                SELECT o.pid, o.vid, arg_min(o.v, o.t) AS v FROM obs_apache o JOIN stays s ON s.pid = o.pid
                WHERE o.t <= s.icu_in + INTERVAL 24 HOUR GROUP BY 1, 2),
            apache AS (   -- APACHE II first (the benchmark's order), each only when mapped
                SELECT f.pid, arg_min(g.name, f.vid) AS name, arg_min(f.vid, f.vid) AS vid,
                       arg_min(g.surgical, f.vid) AS surgical, arg_min(g.system, f.vid) AS system
                FROM first f JOIN {g} g ON CAST(g.variableid AS BIGINT) = f.vid AND CAST(g.code AS DOUBLE) = f.v
                GROUP BY 1)
            SELECT 'hirid:' || CAST(s.pid AS VARCHAR) AS stay_id, CAST(s.pid AS VARCHAR) AS patient_id,
                   CAST(s.pid AS VARCHAR) AS hospitalization_id, 'hirid' AS dataset, 'hirid' AS site_id,
                   s.icu_in AS icu_in_dttm, s.icu_out AS icu_out_dttm, 'general_icu' AS unit_type,
                   true AS clock_valid, true AS demographics_binned,
                   'unknown' AS admission_source, 'unknown' AS surgical_status,
                   CASE a.vid WHEN 9990002 THEN 'APACHE II: ' WHEN 9990004 THEN 'APACHE IV: ' END || a.name AS adm_dx_name,
                   {group_sql("a.surgical = '1'", 'a.system')} AS adm_dx_group,
                   1 AS stay_seq,
                   CASE s.discharge_status WHEN 'dead' THEN 'dead' WHEN 'alive' THEN 'alive' ELSE 'unknown' END AS icu_outcome
            FROM stays s LEFT JOIN apache a ON a.pid = s.pid"""
        return sql, ("stay_id",)
