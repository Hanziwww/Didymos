"""RICD 3.0 (Russian Intensive Care Dataset: Federal Research and Clinical Center of Intensive Care Medicine and
Rehabilitology, Moscow, 2017-2025) -> CLIF (our adapter). Inference only: Paper P's transfer analysis
(``docs/p1_paper_protocol.md`` amendment 16a); the model never trains on it.

The release (``csv/``, ';'-delimited) has one row per hospitalization in ``all_patients`` (several per patient,
``visit_number``), the department pathway with the ICU periods (``patient_pathway``, hours from admission),
bedside monitoring (``monitoring_data``: minutes from admission and the wall-clock time), laboratory results,
therapy prescriptions (daily orders with method, route, dose and rate), scales (GCS, SOFA, ...), a daily SOFA
record with a mechanical-ventilation flag (``detailed_sofa``) and ICD-10 diagnoses.

Conventions:
* times: monitoring minutes; every other source records whole hours since admission (floor), placed at the end of
  their hour so that no value precedes its measurement; department in and out at their hours. A hospitalization
  starts at the shared origin 2020-01-01 plus its admission's time of day (the most frequent wall-clock time minus
  minutes since admission over its monitoring rows; 10-minute resolution) and, for a patient's later
  hospitalizations, after the earlier ones (their length plus 30 days, in visit order: dates are not released).
  ``clock_valid`` holds for hospitalizations with monitoring; birth date = the first hospitalization's start
  minus (age x 365.25 + 182.625) days;
* ICU stays: ICU periods of the pathway, merged when one starts within 1 h of the previous one's end; the source
  names one ICU without a unit type (``general_icu``). Admission source: the department the patient came from
  (``maps/ricd/departments.csv``); an ICU period that opens the hospitalization is a direct admission or a
  transfer from another hospital (``transfer``);
* death: ``fatal_outcome`` is an in-hospital death, at the hospitalization's last department discharge (the
  patient's first fatal hospitalization); an ICU death when the ICU stay ends within 1 h of it;
* vitals and outputs from monitoring (``maps/ricd/monitoring.csv``); blood pressure has no recorded site and is
  charted hourly (cuff); weight and height at admission;
* labs by parameter, unit and biomaterial (``maps/ricd/labs.csv``, SI units converted); urine, CSF and effusion
  analyses are not mapped;
* ventilation: the daily SOFA record's mechanical-ventilation flag (invasive ventilation at that hour, with its
  FiO2); no other respiratory support is recorded, so endpoints of ventilation onset are not defined here;
* assessments: GCS totals from the scales and the SOFA records;
* medications: prescriptions through the shared lexicon (``inn_eng``) with their recorded route. Infusions are
  intravenous orders by infusion pump or dispenser, and every intravenous order of a vasoactive drug except a
  bolus or jet: an episode starts at the first order, continues while the next follows within 25 h (orders are
  renewed daily: 45% of norepinephrine gaps are exactly 24 h), and stops 12 h after the last (the middle of the
  last order's day). The rate's recorded unit (mg/kg/h for every drug, a median 4 for norepinephrine) is not a
  dose unit, so infusions carry no dose. Other orders with a CLIF intermittent category are intermittent
  administrations at the order, with the prescribed dose and unit;
* diagnoses: the discharge diagnoses' ICD-10 codes (WHO), the main discharge diagnosis primary.
"""

from __future__ import annotations

from icutwin.clif.adapters.base import ClifAdapter, minutes_after
from icutwin.clif.conform import extensions, table_schema

ORIGIN = "TIMESTAMPTZ '2020-01-01 00:00:00+00'"
HOSPITAL_ID = "ricd_fnkc_rr"
MERGE_H, GAP_H, OVERHANG_H, VISIT_GAP_DAYS = 1.0, 25.0, 12.0, 30
VASOACTIVE = ("norepinephrine", "epinephrine", "phenylephrine", "dopamine", "dobutamine", "vasopressin", "milrinone")
BLOOD = ("Venous blood", "Arterial blood", "Capillary blood")
ROUTE = """CASE route_eng WHEN 'iv' THEN 'iv' WHEN 'per os' THEN 'enteral' WHEN 'gastroenteric' THEN 'enteral'
                          WHEN 'gastric' THEN 'enteral' WHEN 's/c' THEN 'im' WHEN 'i/m' THEN 'im'
                          WHEN 'inh' THEN 'inhaled' WHEN 'sl' THEN 'buccal_sublingual'
                          WHEN 'intrapleur' THEN 'intrapleural' END"""
# prescribed dose units: CLIF spellings and the scale to them ('doses', 'drops', 'vial', 'kcal', 'million CFU' and
# 'ophth. sol.' have no CLIF unit)
DOSE_UNITS = {"mg": ("mg", 1), "g": ("g", 1), "mcg": ("mcg", 1), "mL": ("ml", 1), "IU (International Units)": ("u", 1),
              "anti-Xa IU": ("u", 1), "thousand IU": ("u", 1000), "thousand anti-Xa IU": ("u", 1000),
              "million IU": ("u", 1000000)}


def at(start_min: str, minutes_sql: str) -> str:
    """The time ``minutes_sql`` minutes after the hospitalization's start (``start_min`` minutes after the origin)."""
    return minutes_after(ORIGIN, f"({start_min}) + ({minutes_sql})")


def hour_end(col: str) -> str:
    """Minutes of a whole-hour offset placed at the end of its hour."""
    return f"(TRY_CAST({col} AS DOUBLE) + 1) * 60.0"


class RicdAdapter(ClifAdapter):
    dataset = "ricd"
    tables = (
        "patient", "hospitalization", "adt", "vitals", "labs", "respiratory_support", "patient_assessments",
        "medication_admin_continuous", "medication_admin_intermittent", "output", "hospital_diagnosis",
        "ext_icu_stay",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ap, pw, mon = (self.src(f"csv/{t}.parquet") for t in ("all_patients", "patient_pathway", "monitoring_data"))
        self.con.execute(f"""
            CREATE OR REPLACE TABLE base AS
            WITH a AS (SELECT CAST(new_patient_id AS VARCHAR) AS pid, CAST(new_hosp_id AS VARCHAR) AS hid,
                              visit_number, sex, age, body_weight, height, transfer, coalesce(fatal_outcome, false) AS fatal,
                              admission_department AS adm_dept, discharge_department AS dis_dept
                       FROM {ap}),
                 p AS (SELECT CAST(new_hosp_id AS VARCHAR) AS hid, max(actual_department_discharge_hour) AS dis_h
                       FROM {pw} GROUP BY 1),
                 c AS (SELECT CAST(new_hosp_id AS VARCHAR) AS hid,
                              mode(CAST(((date_part('hour', "hour") * 60 + date_part('minute', "hour")
                                          - post_admission_minutes) % 1440 + 1440) % 1440 AS INTEGER)) AS clock_min
                       FROM {mon} WHERE "hour" IS NOT NULL AND post_admission_minutes IS NOT NULL GROUP BY 1),
                 o AS (SELECT a.*, p.dis_h, c.clock_min,
                              coalesce(sum(ceil(p.dis_h / 24.0) + {VISIT_GAP_DAYS}) OVER (
                                  PARTITION BY a.pid ORDER BY a.visit_number, a.hid
                                  ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS day_offset
                       FROM a JOIN p USING (hid) LEFT JOIN c USING (hid))
            SELECT *, day_offset * 1440 + coalesce(clock_min, 0) AS start_min, clock_min IS NOT NULL AS clock_valid
            FROM o""")
        dep = self.map("departments")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE seg AS
            SELECT CAST(x.new_hosp_id AS VARCHAR) AS hid, x.unit_stay_number AS n, x.actual_department AS dept,
                   CAST(x.actual_department_admission_hour AS DOUBLE) AS s,
                   CAST(x.actual_department_discharge_hour AS DOUBLE) AS e, coalesce(x.ICU_period, false) AS icu,
                   d.location_category, d.location_type, d.admission_source
            FROM {pw} x LEFT JOIN {dep} d ON d.department = x.actual_department
            WHERE x.actual_department_admission_hour IS NOT NULL AND x.actual_department_discharge_hour IS NOT NULL""")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE icu_stays AS
            WITH s AS (SELECT hid, s, e FROM seg WHERE icu AND e > s),
                 m AS (SELECT *, CASE WHEN lag(e) OVER w IS NULL OR s > lag(e) OVER w + {MERGE_H} THEN 1 ELSE 0 END AS new
                       FROM s WINDOW w AS (PARTITION BY hid ORDER BY s, e)),
                 g AS (SELECT *, sum(new) OVER (PARTITION BY hid ORDER BY s, e ROWS UNBOUNDED PRECEDING) AS k FROM m)
            SELECT hid, k, min(s) AS s, max(e) AS e FROM g GROUP BY hid, k""")

    def _group(self, table: str, column: str) -> str:
        mapping = dict(table_schema(table).get("med_category_to_group_mapping") or {})
        for cat, spec in ((extensions().get(table) or {}).get("med_category") or {}).items():
            if isinstance(spec, dict) and spec.get("group"):
                mapping.setdefault(cat, spec["group"])
        return "CASE " + column + " " + " ".join(
            f"WHEN '{k}' THEN '{v if isinstance(v, str) else v[0]}'" for k, v in mapping.items()) + " END"

    # -- core tables --------------------------------------------------------------------------------
    def sql_patient(self):
        sql = f"""
            WITH f AS (SELECT pid, min(CASE WHEN fatal THEN {at('start_min', 'dis_h * 60.0')} END) AS death_dttm,
                              arg_min(age, visit_number) AS age, arg_min(start_min, visit_number) AS start_min,
                              arg_min(sex, visit_number) AS sex
                       FROM base GROUP BY pid)
            SELECT pid AS patient_id,
                   CAST({at('start_min', '0')} - to_days(CAST(round(age * 365.25 + 182.625) AS INTEGER)) AS DATE) AS birth_date,
                   death_dttm, 'unknown' AS race_category, 'unknown' AS ethnicity_category, sex AS sex_name,
                   CASE sex WHEN 'male' THEN 'male' WHEN 'female' THEN 'female' ELSE 'unknown' END AS sex_category,
                   'unknown' AS language_category
            FROM f"""
        return sql, ("patient_id",)

    def sql_hospitalization(self):
        sql = f"""
            SELECT pid AS patient_id, hid AS hospitalization_id, {at('start_min', '0')} AS admission_dttm,
                   {at('start_min', 'dis_h * 60.0')} AS discharge_dttm, CAST(age AS INTEGER) AS age_at_admission,
                   adm_dept AS admission_type_name,
                   CASE WHEN transfer THEN 'osh' ELSE 'other' END AS admission_type_category,
                   CASE WHEN fatal THEN 'fatal outcome' ELSE dis_dept END AS discharge_name,
                   CASE WHEN fatal THEN 'expired' ELSE 'other' END AS discharge_category
            FROM base"""
        return sql, ("hospitalization_id",)

    def sql_adt(self):
        sql = f"""
            SELECT g.hid AS hospitalization_id, '{HOSPITAL_ID}' AS hospital_id, 'academic' AS hospital_type,
                   {at('b.start_min', 'g.s * 60.0')} AS in_dttm, {at('b.start_min', 'g.e * 60.0')} AS out_dttm,
                   g.dept AS location_name, coalesce(g.location_category, 'other') AS location_category,
                   g.location_type
            FROM seg g JOIN base b USING (hid)"""
        return sql, ("hospitalization_id", "in_dttm")

    def _monitoring(self, target: str) -> str:
        m = self.map("monitoring")
        return f"""
            SELECT CAST(x.new_hosp_id AS VARCHAR) AS hid, {at('b.start_min', 'x.post_admission_minutes')} AS t,
                   x.parameter, x.value, m.category, nullif(m.meas_site_name, '') AS site, m.output_group
            FROM {self.src('csv/monitoring_data.parquet')} x
            JOIN {m} m ON m.parameter = x.parameter AND m.target = '{target}'
            JOIN base b ON b.hid = CAST(x.new_hosp_id AS VARCHAR)
            WHERE x.value IS NOT NULL AND x.post_admission_minutes IS NOT NULL"""

    def sql_vitals(self):
        sql = f"""
            SELECT hid AS hospitalization_id, t AS recorded_dttm, parameter AS vital_name, category AS vital_category,
                   value AS vital_value, site AS meas_site_name
            FROM ({self._monitoring('vitals')})
            UNION ALL   -- weight and height at admission
            SELECT hid, {at('start_min', '0')}, 'body_weight at admission', 'weight_kg', body_weight, NULL
            FROM base WHERE body_weight IS NOT NULL
            UNION ALL
            SELECT hid, {at('start_min', '0')}, 'height at admission', 'height_cm', CAST(height AS DOUBLE), NULL
            FROM base WHERE height IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_output(self):
        sql = f"""
            SELECT hid AS hospitalization_id, t AS recorded_dttm, parameter AS output_name, category AS output_category,
                   output_group, value AS output_volume
            FROM ({self._monitoring('output')}) WHERE value >= 0"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_labs(self):
        m = self.map("labs")
        lab = self.src("csv/lab_data.parquet")
        blood = ", ".join(f"'{b}'" for b in BLOOD)
        join = f"""m.parameter = x.parameter_eng_short AND m.unit = x.unit_eng
                   AND (m.biomaterial = x.biomaterial_eng OR (m.biomaterial = 'blood' AND x.biomaterial_eng IN ({blood})))"""
        self.coverage("labs", "parameter", f"""
            SELECT x.parameter_eng_short AS txt, count(*) AS n, bool_or(m.lab_category IS NOT NULL) AS mapped
            FROM {lab} x LEFT JOIN {m} m ON {join}
            WHERE x.biomaterial_eng IN ({blood}) AND x.result_num IS NOT NULL GROUP BY 1""")
        sql = f"""
            SELECT CAST(x.new_hosp_id AS VARCHAR) AS hospitalization_id,
                   {at('b.start_min', hour_end('x.post_admission_hours'))} AS lab_order_dttm,
                   {at('b.start_min', hour_end('x.post_admission_hours'))} AS lab_collect_dttm,
                   {at('b.start_min', hour_end('x.post_admission_hours'))} AS lab_result_dttm,
                   x.analysis_eng AS lab_order_name, m.lab_order_category, x.parameter_eng_short AS lab_name,
                   m.lab_category, CAST(x.result_num AS VARCHAR) AS lab_value,
                   x.result_num * CAST(m.factor AS DOUBLE) AS lab_value_numeric, m.reference_unit,
                   'plasma_blood' AS lab_specimen_category
            FROM {lab} x JOIN {m} m ON {join} JOIN base b ON b.hid = CAST(x.new_hosp_id AS VARCHAR)
            WHERE x.result_num IS NOT NULL AND x.post_admission_hours IS NOT NULL"""
        return sql, ("hospitalization_id", "lab_result_dttm")

    def sql_respiratory_support(self):
        sql = f"""
            SELECT CAST(x.new_hosp_id AS VARCHAR) AS hospitalization_id,
                   {at('b.start_min', hour_end('x.post_admission_hours'))} AS recorded_dttm,
                   'SOFA record: mechanical ventilation' AS device_name, 'imv' AS device_category,
                   CASE WHEN x."FiO2 %" BETWEEN 21 AND 100 THEN x."FiO2 %" / 100.0 END AS fio2_set
            FROM {self.src('csv/detailed_sofa.parquet')} x JOIN base b ON b.hid = CAST(x.new_hosp_id AS VARCHAR)
            WHERE x.mechanical_ventilation AND x.post_admission_hours IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_patient_assessments(self):
        sql = f"""
            SELECT CAST(x.new_hosp_id AS VARCHAR) AS hospitalization_id,
                   {at('b.start_min', hour_end('x.post_admission_hours'))} AS recorded_dttm,
                   'GCS (scales)' AS assessment_name, 'gcs_total' AS assessment_category, x.result AS numerical_value
            FROM {self.src('csv/all_scales.parquet')} x JOIN base b ON b.hid = CAST(x.new_hosp_id AS VARCHAR)
            WHERE x.scale_eng = 'GCS' AND x.result BETWEEN 3 AND 15 AND x.post_admission_hours IS NOT NULL
            UNION ALL
            SELECT CAST(x.new_hosp_id AS VARCHAR), {at('b.start_min', hour_end('x.post_admission_hours'))},
                   'GCS (SOFA record)', 'gcs_total', CAST(x.GCS_score AS DOUBLE)
            FROM {self.src('csv/detailed_sofa.parquet')} x JOIN base b ON b.hid = CAST(x.new_hosp_id AS VARCHAR)
            WHERE x.GCS_score BETWEEN 3 AND 15 AND x.post_admission_hours IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    # -- medications ---------------------------------------------------------------------------------
    def _orders(self) -> str:
        if self.con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'orders'").fetchone()[0]:
            return "orders"
        rx = self.src("csv/therapy_prescriptions.parquet")
        lk = self.regex_lookup("drugs", f"SELECT lower(inn_eng) AS txt FROM {rx}", "drugs",
                               ("ingredient", "continuous", "intermittent"), table=self.lexicon("drugs"))
        vaso = ", ".join(f"'{v}'" for v in VASOACTIVE)
        unit = " ".join(f"WHEN '{k}' THEN '{u}'" for k, (u, _) in DOSE_UNITS.items())
        scale = " ".join(f"WHEN '{k}' THEN {f}" for k, (_, f) in DOSE_UNITS.items())
        self.con.execute(f"""
            CREATE OR REPLACE TABLE orders AS
            WITH x AS (
                SELECT CAST(r.new_hosp_id AS VARCHAR) AS hid, (TRY_CAST(r.post_admission_hours AS DOUBLE) + 1) AS h,
                       r.inn_eng AS name, r.route_eng, {ROUTE} AS route, r.method_eng AS method,
                       r.dose * (CASE r.dose_unit_eng {scale} END) AS dose, CASE r.dose_unit_eng {unit} END AS unit,
                       nullif(l.continuous, '') AS continuous, nullif(l.intermittent, '') AS intermittent
                FROM {rx} r LEFT JOIN {lk} l ON l.txt = lower(r.inn_eng)
                WHERE TRY_CAST(r.post_admission_hours AS DOUBLE) IS NOT NULL)
            SELECT x.*, b.start_min,
                   coalesce(route = 'iv' AND continuous IS NOT NULL
                            AND (method IN ('infusion pump', 'dispenser')
                                 OR (continuous IN ({vaso}) AND coalesce(method, '') NOT IN ('bolus', 'jet'))), false) AS infusion
            FROM x JOIN base b USING (hid)""")
        self.coverage("medication", "inn", """
            SELECT name AS txt, count(*) AS n, bool_or(continuous IS NOT NULL OR intermittent IS NOT NULL) AS mapped
            FROM orders GROUP BY 1""")
        return "orders"

    def sql_medication_admin_continuous(self):
        o = self._orders()
        sql = f"""
            WITH r AS (
                SELECT hid, continuous, h, any_value(start_min) AS start_min, min(name) AS name, min(route_eng) AS route_name,
                       lag(h) OVER (PARTITION BY hid, continuous ORDER BY h) AS prev_h,
                       lead(h) OVER (PARTITION BY hid, continuous ORDER BY h) AS next_h
                FROM {o} WHERE infusion GROUP BY hid, continuous, h),
            ev AS (
                SELECT hid, continuous, h, start_min, name, route_name,
                       CASE WHEN prev_h IS NULL OR h - prev_h > {GAP_H} THEN 'start' ELSE 'going' END AS action,
                       NULL AS action_name
                FROM r
                UNION ALL   -- inferred end: not ordered again within the daily cycle
                SELECT hid, continuous, h + {OVERHANG_H}, start_min, name, route_name, 'stop',
                       'inferred stop: not ordered again within {GAP_H:g} h'
                FROM r WHERE next_h IS NULL OR next_h - h > {GAP_H})
            SELECT hid AS hospitalization_id, {at('start_min', 'h * 60.0')} AS admin_dttm, name AS med_name,
                   continuous AS med_category, {self._group('medication_admin_continuous', 'continuous')} AS med_group,
                   route_name AS med_route_name, 'iv' AS med_route_category,
                   CASE WHEN action = 'stop' THEN 0.0 END AS med_dose,
                   action_name AS mar_action_name, action AS mar_action_category, 'administered' AS mar_action_group
            FROM ev"""
        return sql, ("hospitalization_id", "admin_dttm", "med_category")

    def sql_medication_admin_intermittent(self):
        o = self._orders()
        sql = f"""
            SELECT hid AS hospitalization_id, {at('start_min', 'h * 60.0')} AS admin_dttm, name AS med_name,
                   intermittent AS med_category, {self._group('medication_admin_intermittent', 'intermittent')} AS med_group,
                   route_eng AS med_route_name, route AS med_route_category, dose AS med_dose, unit AS med_dose_unit,
                   'given' AS mar_action_category, 'administered' AS mar_action_group
            FROM {o} WHERE intermittent IS NOT NULL AND NOT infusion AND route IS NOT NULL"""
        return sql, ("hospitalization_id", "admin_dttm")

    # -- diagnoses and stay context -------------------------------------------------------------------
    def sql_hospital_diagnosis(self):
        code = "regexp_extract(upper(trim(ICD_10)), '^([A-Z][0-9]{2}(\\.[0-9A-Z]{1,4})?)', 1)"
        sql = f"""
            SELECT CAST(new_hosp_id AS VARCHAR) AS hospitalization_id, replace({code}, '.', '') AS diagnosis_code,
                   'ICD10CM' AS diagnosis_code_format, max(CASE WHEN diagnosis_type = 'discharge diagnosis' THEN 1 ELSE 0 END)
                   AS diagnosis_primary
            FROM {self.src('csv/ICD10_diagnoses.parquet')}
            WHERE diagnosis_type LIKE 'discharge diagnosis%' AND {code} <> ''
            GROUP BY 1, 2"""
        return sql, ("hospitalization_id",)

    def sql_ext_icu_stay(self):
        sql = f"""
            WITH prev AS (   -- the department the patient came from
                SELECT s.hid, s.k, arg_max(g.admission_source, g.e) AS source
                FROM icu_stays s JOIN seg g ON g.hid = s.hid AND NOT g.icu AND g.e <= s.s + {MERGE_H}
                GROUP BY s.hid, s.k)
            SELECT 'ricd:' || s.hid || '_' || CAST(s.k AS VARCHAR) AS stay_id, b.pid AS patient_id,
                   s.hid AS hospitalization_id, 'ricd' AS dataset, 'ricd' AS site_id,
                   {at('b.start_min', 's.s * 60.0')} AS icu_in_dttm, {at('b.start_min', 's.e * 60.0')} AS icu_out_dttm,
                   'general_icu' AS unit_type, b.clock_valid, false AS demographics_binned,
                   CASE WHEN p.source IS NOT NULL THEN p.source
                        WHEN s.s <= {MERGE_H} AND b.transfer THEN 'other_hospital'
                        WHEN s.s <= {MERGE_H} THEN 'direct' ELSE 'unknown' END AS admission_source,
                   'unknown' AS surgical_status,
                   CAST(row_number() OVER (PARTITION BY b.pid ORDER BY b.start_min + s.s * 60) AS INTEGER) AS stay_seq,
                   CASE WHEN b.fatal AND s.e >= b.dis_h - {MERGE_H} THEN 'dead' ELSE 'alive' END AS icu_outcome
            FROM icu_stays s JOIN base b USING (hid) LEFT JOIN prev p ON p.hid = s.hid AND p.k = s.k"""
        return sql, ("stay_id",)
