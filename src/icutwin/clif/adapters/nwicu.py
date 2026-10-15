"""NWICU 0.1.0 (Northwestern Medicine ICU database) -> CLIF (our adapter).

NWICU uses the MIMIC-IV layout (``nw_hosp``, ``nw_icu``) with its own item ids and a reduced content: routine
vital signs in ``chartevents``, laboratory results, medication administrations (``emar``, no doses) with their
prescriptions, and clinical orders in ``procedureevents``. There are no infusion rates, ventilator settings
or I/O volumes; those CLIF tables stay empty or carry presence only, which the dataset support masks of the
token grammar represent.

Conventions:
* timestamps are date-shifted with real time of day (``clock_valid``; ICU admissions peak in the afternoon
  and evening); stored as UTC wall-clock time;
* ages: MIMIC convention, age at admission = ``anchor_age`` + (admission year - ``anchor_year``); birth date at
  1 July of ``anchor_year - anchor_age``; death = ``admissions.deathtime`` or ``patients.dod``;
* ICU stays are ``icustays`` rows whose first care unit is an ICU (intermediate care and ward boarders are ADT
  segments, not ICU stays);
* medications: an intermittent dose is an ``emar`` 'Applied' event and takes the dose and route of its
  prescription: an amount, a count in a counting unit (tablet, puff, packet; as MIMIC's eMAR) or a weight-based
  amount (mg/kg) as prescribed. An order without a dose unit states no administration type (its products
  include infusion bags, drips and PCA syringes as well as bolus vials), so it gives no record. 'Confirmed' events are not administrations of intermittent orders: 89% fall after the
  order stopped and 60% after hospital discharge, 97% on a quarter hour (scheduled times), versus 0%, 0%
  and 8% for 'Applied' (subset). For infusion orders (a rate unit) 'Confirmed' is the periodic rate
  verification while the infusion runs (0.5% after the order stopped, none after discharge). Infusions
  carry no administered rate, so they are presence events (``going``, dose NULL) that stop 60 min after the
  last confirmation when none follows within 6 h (the eICU/ricu interval rule);
* orders in ``procedureevents`` document ventilation (``MECHANICAL VENTILATION``, ``INTUBATION``) and
  non-invasive ventilation, haemodialysis, DNR orders and intubation/extubation (CLIF 3.0 ``airway``).
"""

from __future__ import annotations

from icutwin.clif.adapters.base import ClifAdapter
from icutwin.clif.conform import extensions, table_schema

HOSPITAL_ID = "nwicu_northwestern"
GAP_MIN, OVERHANG_MIN = 360, 60
RATE_UNIT = r"(?i)/(hr|min)$"
# CLIF route categories (mCIDE): 'im' includes subcutaneous. 'Injection' is unspecified: subcutaneous for insulin
# and for heparin up to 7,500 units (prophylactic regimens), intravenous otherwise. Inhaled doses take the local
# intermittent route extension 'inhaled' (extensions.yaml). Topical, ophthalmic, nasal, rectal, transdermal and
# catheter-lock routes have no CLIF route category.
ROUTE = """CASE WHEN route = 'Injection' AND (ingredient = 'insulin'
                     OR (ingredient = 'heparin' AND TRY_CAST(dose_val_rx AS DOUBLE) <= 7500)) THEN 'im'
                WHEN route IN ('Intravenous', 'Injection', 'IV Push', 'Intravenous Piggyback') THEN 'iv'
                WHEN route IN ('Oral', 'Per G Tube', 'Per NG Tube', 'Per J Tube', 'Enteral') THEN 'enteral'
                WHEN route IN ('Intramuscular', 'Subcutaneous') THEN 'im'
                WHEN route IN ('Sublingual', 'Translingual') THEN 'buccal_sublingual'
                WHEN route = 'Intrapleural' THEN 'intrapleural'
                WHEN route = 'Inhalation' THEN 'inhaled' END"""


class NwicuAdapter(ClifAdapter):
    dataset = "nwicu"
    tables = (
        "patient", "hospitalization", "adt", "vitals", "labs", "respiratory_support",
        "medication_admin_continuous", "medication_admin_intermittent", "intermittent_dialysis", "airway",
        "code_status", "hospital_diagnosis", "ext_icu_stay",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.con.execute(f"""
            CREATE OR REPLACE TABLE adm AS
            SELECT a.*, p.gender, p.anchor_age, p.anchor_year, p.anchor_year_group, p.dod,
                   CAST(p.anchor_age + year(a.admittime) - p.anchor_year AS INTEGER) AS age
            FROM {self.src('data/nw_hosp/admissions.parquet')} a
            JOIN {self.src('data/nw_hosp/patients.parquet')} p USING (subject_id)""")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE items AS
            SELECT DISTINCT ON (itemid) itemid, label FROM {self.src('data/nw_icu/d_items.parquet')} ORDER BY itemid, label""")

    def _ts(self, col: str) -> str:
        return f"CAST(CAST({col} AS TIMESTAMP) AS TIMESTAMPTZ)"

    def _group(self, table: str, column: str) -> str:
        mapping = dict(table_schema(table).get("med_category_to_group_mapping") or {})
        for cat, spec in ((extensions().get(table) or {}).get("med_category") or {}).items():
            if isinstance(spec, dict) and spec.get("group"):
                mapping.setdefault(cat, spec["group"])
        return "CASE " + column + " " + " ".join(
            f"WHEN '{k}' THEN '{v if isinstance(v, str) else v[0]}'" for k, v in mapping.items()) + " END"

    # -- core tables --------------------------------------------------------------------------------
    def sql_patient(self):
        race = self.map("race")
        sql = f"""
            WITH p AS (
                SELECT subject_id, any_value(gender) AS gender, any_value(anchor_age) AS anchor_age,
                       any_value(anchor_year) AS anchor_year, any_value(dod) AS dod, max(deathtime) AS deathtime,
                       arg_max(race, admittime) AS race
                FROM adm GROUP BY subject_id)
            SELECT CAST(subject_id AS VARCHAR) AS patient_id,
                   make_date(CAST(anchor_year - anchor_age AS INTEGER), 7, 1) AS birth_date,
                   coalesce({self._ts('deathtime')}, {self._ts('dod')}) AS death_dttm,
                   p.race AS race_name, coalesce(r.race_category, 'unknown') AS race_category,
                   coalesce(r.ethnicity_category, 'unknown') AS ethnicity_category,
                   gender AS sex_name, CASE gender WHEN 'M' THEN 'male' WHEN 'F' THEN 'female' ELSE 'unknown' END AS sex_category,
                   'unknown' AS language_category
            FROM p LEFT JOIN {race} r ON r.race = p.race"""
        return sql, ("patient_id",)

    def sql_hospitalization(self):
        dis = self.map("discharge")
        sql = f"""
            SELECT CAST(a.subject_id AS VARCHAR) AS patient_id, CAST(a.hadm_id AS VARCHAR) AS hospitalization_id,
                   {self._ts('a.admittime')} AS admission_dttm, {self._ts('a.dischtime')} AS discharge_dttm,
                   a.age AS age_at_admission, a.admission_type AS admission_type_name,
                   CASE WHEN a.admission_type IN ('EMERGENCY', 'TRAUMA') THEN 'ed'
                        WHEN a.admission_type LIKE 'ELECTIVE%' THEN 'elective'
                        WHEN a.admission_type IS NOT NULL THEN 'other' END AS admission_type_category,
                   a.discharge_location AS discharge_name,
                   CASE WHEN a.hospital_expire_flag = 1 THEN 'expired'
                        ELSE coalesce(d.discharge_category, CASE WHEN a.discharge_location IS NULL THEN 'missing' ELSE 'other' END)
                   END AS discharge_category
            FROM adm a LEFT JOIN {dis} d ON d.discharge_location = a.discharge_location"""
        return sql, ("hospitalization_id",)

    def sql_adt(self):
        cu = self.map("care_units")
        sql = f"""
            SELECT CAST(s.hadm_id AS VARCHAR) AS hospitalization_id, '{HOSPITAL_ID}' AS hospital_id, 'academic' AS hospital_type,
                   {self._ts('s.intime')} AS in_dttm, {self._ts('s.outtime')} AS out_dttm, s.first_careunit AS location_name,
                   coalesce(c.location_category, 'other') AS location_category, c.location_type
            FROM {self.src('data/nw_icu/icustays.parquet')} s LEFT JOIN {cu} c ON c.careunit = s.first_careunit"""
        return sql, ("hospitalization_id", "in_dttm")

    def sql_vitals(self):
        m = self.map("chart_items")
        sql = f"""
            SELECT CAST(c.hadm_id AS VARCHAR) AS hospitalization_id, {self._ts('c.charttime')} AS recorded_dttm,
                   i.label AS vital_name, m.category AS vital_category,
                   (c.valuenum + CAST(m."offset" AS DOUBLE)) * CAST(m.factor AS DOUBLE) AS vital_value,
                   nullif(m.meas_site_name, '') AS meas_site_name
            FROM {self.src('data/nw_icu/chartevents.parquet')} c
            JOIN {m} m ON CAST(m.itemid AS BIGINT) = c.itemid AND m.target = 'vitals'
            LEFT JOIN items i ON i.itemid = c.itemid
            WHERE c.valuenum IS NOT NULL AND c.hadm_id IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_labs(self):
        m = self.map("labs")
        sql = f"""
            SELECT CAST(l.hadm_id AS VARCHAR) AS hospitalization_id,
                   {self._ts('l.charttime')} AS lab_order_dttm, {self._ts('l.charttime')} AS lab_collect_dttm,
                   {self._ts('coalesce(l.storetime, l.charttime)')} AS lab_result_dttm,
                   m.lab_order_category, d.label AS lab_name, m.lab_category,
                   coalesce(l.value, CAST(l.valuenum AS VARCHAR)) AS lab_value,
                   l.valuenum * CAST(m.factor AS DOUBLE) AS lab_value_numeric,
                   m.reference_unit, 'plasma_blood' AS lab_specimen_category
            FROM {self.src('data/nw_hosp/labevents.parquet')} l
            JOIN {m} m ON CAST(m.itemid AS BIGINT) = l.itemid
            LEFT JOIN (SELECT DISTINCT ON (itemid) itemid, label FROM {self.src('data/nw_hosp/d_labitems.parquet')}) d
                   ON d.itemid = l.itemid
            WHERE l.valuenum IS NOT NULL AND l.valuenum < 9999999 AND l.hadm_id IS NOT NULL"""   # 9999999 = text result
        return sql, ("hospitalization_id", "lab_result_dttm")

    def _orders(self, target: str) -> str:
        p = self.map("procedures")
        return f"""
            SELECT pe.hadm_id, {self._ts('pe.starttime')} AS t, i.label, m.category
            FROM {self.src('data/nw_icu/procedureevents.parquet')} pe JOIN items i ON i.itemid = pe.itemid
            JOIN {p} m ON m.target = '{target}' AND regexp_matches(lower(i.label), m.pattern)
            WHERE pe.hadm_id IS NOT NULL"""

    def sql_respiratory_support(self):
        lab = self.src("data/nw_hosp/labevents.parquet")
        sql = f"""
            SELECT CAST(hadm_id AS VARCHAR) AS hospitalization_id, t AS recorded_dttm,
                   'order: ' || label AS device_name, category AS device_category
            FROM ({self._orders('respiratory_support')})
            UNION ALL BY NAME   -- ventilator values recorded with blood gases
            SELECT CAST(hadm_id AS VARCHAR) AS hospitalization_id, {self._ts('charttime')} AS recorded_dttm,
                   max(CASE WHEN itemid = 100029 THEN valuenum / 100.0 END) AS fio2_set,
                   max(CASE WHEN itemid = 100060 THEN valuenum END) AS peep_set,
                   max(CASE WHEN itemid = 100077 THEN valuenum END) AS tidal_volume_set
            FROM {lab} WHERE itemid IN (100029, 100060, 100077) AND valuenum < 9999999 AND hadm_id IS NOT NULL
            GROUP BY hadm_id, charttime"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_intermittent_dialysis(self):
        sql = f"""SELECT DISTINCT CAST(hadm_id AS VARCHAR) AS hospitalization_id, t AS recorded_dttm
                  FROM ({self._orders('intermittent_dialysis')})"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_airway(self):
        """Intubation orders open an endotracheal airway; the next extubation order closes it."""
        sql = f"""
            WITH i AS (SELECT hadm_id, t FROM ({self._orders('airway_placed')})),
                 e AS (SELECT hadm_id, t FROM ({self._orders('airway_removed')}))
            SELECT CAST(i.hadm_id AS VARCHAR) AS hospitalization_id,
                   CAST(i.hadm_id AS VARCHAR) || '_' || CAST(row_number() OVER (PARTITION BY i.hadm_id ORDER BY i.t) AS VARCHAR) AS airway_id,
                   i.t AS placed_dttm, (SELECT min(e.t) FROM e WHERE e.hadm_id = i.hadm_id AND e.t > i.t) AS removed_dttm,
                   'ett' AS airway_category
            FROM i"""
        return sql, ("hospitalization_id", "placed_dttm")

    def sql_code_status(self):
        sql = f"""
            SELECT CAST(a.subject_id AS VARCHAR) AS patient_id, min(o.t) AS start_dttm,
                   first(o.label ORDER BY o.t, o.label) AS code_status_name, 'dnr' AS code_status_category
            FROM ({self._orders('code_status')}) o JOIN adm a ON a.hadm_id = o.hadm_id
            GROUP BY a.subject_id, a.hadm_id"""
        return sql, ("patient_id", "start_dttm")

    # -- medications ---------------------------------------------------------------------------------
    def _emar(self) -> str:
        if self.con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'emar_rx'").fetchone()[0]:
            return "emar_rx"
        emar = self.src("data/nw_hosp/emar.parquet")
        lk = self.regex_lookup("drugs", f"SELECT lower(medication) AS txt FROM {emar}", "drugs",
                               ("ingredient", "continuous", "intermittent"), table=self.lexicon("drugs"))
        self.con.execute(f"""
            CREATE OR REPLACE TABLE emar_rx AS
            SELECT e.hadm_id, {self._ts('e.charttime')} AS t, e.medication, e.event_txt,
                   p.dose_val_rx, p.dose_unit_rx, p.route,
                   regexp_matches(coalesce(p.dose_unit_rx, ''), '{RATE_UNIT}') AS is_infusion,
                   l.ingredient, nullif(l.continuous, '') AS continuous, nullif(l.intermittent, '') AS intermittent
            FROM {emar} e
            LEFT JOIN (SELECT DISTINCT ON (pharmacy_id) * FROM {self.src('data/nw_hosp/prescriptions.parquet')}) p
                   USING (pharmacy_id)
            LEFT JOIN {lk} l ON l.txt = lower(e.medication)
            WHERE e.event_txt IN ('Confirmed', 'Applied') AND e.hadm_id IS NOT NULL""")
        self.coverage("medication", "emar_name", """
            SELECT medication AS txt, count(*) AS n, bool_or(ingredient IS NOT NULL) AS mapped FROM emar_rx GROUP BY 1""")
        return "emar_rx"

    def sql_medication_admin_intermittent(self):
        e = self._emar()
        route = ROUTE
        sql = f"""
            SELECT CAST(hadm_id AS VARCHAR) AS hospitalization_id, t AS admin_dttm, medication AS med_name,
                   intermittent AS med_category, {self._group('medication_admin_intermittent', 'intermittent')} AS med_group,
                   route AS med_route_name, {route} AS med_route_category,
                   TRY_CAST(dose_val_rx AS DOUBLE) AS med_dose, lower(dose_unit_rx) AS med_dose_unit,
                   'given' AS mar_action_category, 'administered' AS mar_action_group
            FROM {e} WHERE NOT is_infusion AND event_txt = 'Applied' AND intermittent IS NOT NULL
              AND ({route}) IS NOT NULL
              AND dose_unit_rx IS NOT NULL"""
        return sql, ("hospitalization_id", "admin_dttm")

    def sql_medication_admin_continuous(self):
        e = self._emar()
        sql = f"""
            WITH r AS (   -- the ordered rate unit is kept (the administered rate itself is not recorded)
                SELECT hadm_id, continuous, t, min(medication) AS name, min(lower(dose_unit_rx)) AS unit,
                       lag(t) OVER (PARTITION BY hadm_id, continuous ORDER BY t) AS prev_t,
                       lead(t) OVER (PARTITION BY hadm_id, continuous ORDER BY t) AS next_t
                FROM {e} WHERE is_infusion AND continuous IS NOT NULL GROUP BY hadm_id, continuous, t),
            ev AS (
                SELECT hadm_id, continuous, t, name, unit,
                       CASE WHEN prev_t IS NULL OR date_diff('minute', prev_t, t) > {GAP_MIN} THEN 'start' ELSE 'going' END AS action,
                       NULL AS action_name
                FROM r
                UNION ALL   -- inferred end: no further confirmation within 6 h
                SELECT hadm_id, continuous, t + INTERVAL {OVERHANG_MIN} MINUTE, name, unit, 'stop',
                       'inferred stop: not confirmed again within 6 h'
                FROM r WHERE next_t IS NULL OR date_diff('minute', t, next_t) > {GAP_MIN})
            SELECT CAST(hadm_id AS VARCHAR) AS hospitalization_id, t AS admin_dttm, name AS med_name,
                   continuous AS med_category, {self._group('medication_admin_continuous', 'continuous')} AS med_group,
                   'iv' AS med_route_category, unit AS med_dose_unit,
                   CASE WHEN action = 'stop' THEN 0.0 END AS med_dose,
                   action_name AS mar_action_name, action AS mar_action_category, 'administered' AS mar_action_group
            FROM ev"""
        return sql, ("hospitalization_id", "admin_dttm", "med_category")

    # -- diagnoses and stay context -------------------------------------------------------------------
    def sql_hospital_diagnosis(self):
        sql = f"""
            SELECT CAST(hadm_id AS VARCHAR) AS hospitalization_id, replace(icd_code, '.', '') AS diagnosis_code,
                   CASE icd_version WHEN 10 THEN 'ICD10CM' ELSE 'ICD9CM' END AS diagnosis_code_format,
                   CASE WHEN seq_num = 1 THEN 1 ELSE 0 END AS diagnosis_primary
            FROM {self.src('data/nw_hosp/diagnoses_icd.parquet')}"""
        return sql, ("hospitalization_id",)

    def sql_ext_icu_stay(self):
        cu = self.map("care_units")
        sql = f"""
            WITH s AS (   -- an ICU stay when its first or last care unit is an ICU (no transfers table)
                SELECT s.*, coalesce(f.location_type, l.location_type) AS location_type
                FROM {self.src('data/nw_icu/icustays.parquet')} s
                LEFT JOIN {cu} f ON f.careunit = s.first_careunit AND f.is_icu = '1'
                LEFT JOIN {cu} l ON l.careunit = s.last_careunit AND l.is_icu = '1'
                WHERE f.careunit IS NOT NULL OR l.careunit IS NOT NULL)
            SELECT 'nwicu:' || CAST(s.stay_id AS VARCHAR) AS stay_id, CAST(s.subject_id AS VARCHAR) AS patient_id,
                   CAST(s.hadm_id AS VARCHAR) AS hospitalization_id, 'nwicu' AS dataset, 'nwicu' AS site_id,
                   {self._ts('s.intime')} AS icu_in_dttm, {self._ts('s.outtime')} AS icu_out_dttm, s.location_type AS unit_type,
                   true AS clock_valid, false AS demographics_binned,
                   CASE WHEN a.admission_location = 'EMERGENCY MEDICINE' THEN 'ed'
                        WHEN a.admission_location = 'SURGERY' THEN 'or'
                        WHEN a.admission_location IS NULL THEN 'unknown' ELSE 'other' END AS admission_source,
                   'unknown' AS surgical_status,
                   CAST(row_number() OVER (PARTITION BY s.subject_id ORDER BY s.intime, s.stay_id) AS INTEGER) AS stay_seq,
                   CASE WHEN a.deathtime IS NOT NULL AND a.deathtime <= s.outtime THEN 'dead' ELSE 'alive' END AS icu_outcome,
                   replace(a.anchor_year_group, ' ', '') AS era
            FROM s JOIN adm a ON a.hadm_id = s.hadm_id"""
        return sql, ("stay_id",)
