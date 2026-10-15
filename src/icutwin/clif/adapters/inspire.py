"""INSPIRE 1.4.2 -> CLIF (our adapter; no published CLIF ETL exists for INSPIRE).

Time anchoring follows the community INSPIRE_MEDS ETL exactly: every ``*_time`` column is minutes since
the patient's own first admission, placed on the shared origin 2016-01-01 (absolute dates are not real;
``clock_valid`` is false). Birth date = origin - (age at first admission x 365.25 + 182.625) days.

ICU stays are the distinct post-operative ``icuin_time``/``icuout_time`` windows of ``operations``, each in
the admission whose window contains its start (an operation row can carry the ICU stay of the patient's
next admission, so the row's own ``hadm_id`` is not used). ``ward_vitals``
carries ward *and* ICU charting; intra-operative ``vitals`` carries OR monitoring, anaesthesia-machine
ventilation, infusion rates, boluses and fluids.
"""

from __future__ import annotations

import pandas as pd

from icutwin.clif.adapters.base import ClifAdapter, minutes_after
from icutwin.clif.conform import table_schema
from icutwin.clif.dx_groups import group_sql

ORIGIN = "TIMESTAMPTZ '2016-01-01 00:00:00+00'"
HOSPITAL_ID = "inspire_snuh"


def t(col: str) -> str:
    return minutes_after(ORIGIN, col)


class InspireAdapter(ClifAdapter):
    dataset = "inspire"
    tables = (
        "patient", "hospitalization", "adt", "vitals", "labs", "respiratory_support", "patient_assessments",
        "medication_admin_continuous", "medication_admin_intermittent", "crrt_therapy", "mcs",
        "hospital_diagnosis", "patient_procedures", "input", "output", "ext_icu_stay",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ops = self.src("operations.parquet")
        # one row per hospitalization: the window used to attach subject-level events to an admission
        self.con.execute(f"""
            CREATE OR REPLACE TABLE hosp_win AS
            SELECT subject_id, hadm_id, min(admission_time) AS adm_min, max(discharge_time) AS dis_min
            FROM {ops} WHERE hadm_id IS NOT NULL GROUP BY subject_id, hadm_id""")
        # distinct ICU stays of a patient (several operations can share one post-op ICU stay), each in the
        # admission whose window contains its start: an operation row can carry the ICU stay of a later
        # admission of the patient (subset: 102 of 390 rows with ICU times start after their own discharge)
        self.con.execute(f"""
            CREATE OR REPLACE TABLE icu_stays AS
            WITH s AS (SELECT subject_id, icuin_time, max(icuout_time) AS icuout_time
                       FROM {ops} WHERE icuin_time IS NOT NULL GROUP BY subject_id, icuin_time)
            SELECT s.subject_id, last(h.hadm_id ORDER BY h.adm_min, h.hadm_id) AS hadm_id, s.icuin_time, max(s.icuout_time) AS icuout_time
            FROM s JOIN hosp_win h ON h.subject_id = s.subject_id AND s.icuin_time BETWEEN h.adm_min AND h.dis_min
            GROUP BY s.subject_id, s.icuin_time""")
        # intra-operative records of an admission: overlapping operations (two op_ids in one session) can both
        # carry the session's monitoring, fluids and drugs; an identical record (time, item, value) is kept once,
        # under the first op_id, so volumes and doses are not counted twice
        self.con.execute(f"""
            CREATE OR REPLACE TABLE intra AS
            SELECT DISTINCT ON (o.hadm_id, x.chart_time, x.item_name, x.value)
                   x.op_id, o.hadm_id, x.chart_time, x.item_name, x.value
            FROM {self.src('vitals.parquet')} x JOIN {ops} o ON x.op_id = o.op_id WHERE o.hadm_id IS NOT NULL
            ORDER BY o.hadm_id, x.chart_time, x.item_name, x.value, x.op_id""")

    # INSPIRE repeats identical rows in its event tables (subset: ward urine output 81 % of rows, about five
    # copies per measurement; CRRT/ECMO flags 45 %; heart rate 16 %; intra-operative respiratory rate 34 %).
    # One row per identical record is read, the event the community ETL keeps; volumes are not multiplied.
    EVENT_TABLES = ("ward_vitals.parquet", "vitals.parquet", "labs.parquet", "medications.parquet", "diagnosis.parquet")

    def src(self, rel: str) -> str:
        table = super().src(rel)
        return f"(SELECT DISTINCT * FROM {table})" if rel in self.EVENT_TABLES else table

    def _in_hosp(self, table: str, time_col: str = "chart_time") -> str:
        """Attach subject-level rows to the hospitalization whose window contains them."""
        return (f"FROM {table} x JOIN hosp_win h ON x.subject_id = h.subject_id "
                f"AND x.{time_col} BETWEEN h.adm_min AND h.dis_min")

    # -- core tables --------------------------------------------------------------------------------
    def sql_patient(self):
        ops = self.src("operations.parquet")
        sql = f"""
            WITH s AS (
                SELECT subject_id,
                       min(CASE WHEN admission_time = 0 THEN age END) AS birth_age,
                       least(min(inhosp_death_time), min(allcause_death_time)) AS death_min,
                       min(sex) AS sex, min(race) AS race
                FROM {ops} GROUP BY subject_id)
            SELECT CAST(subject_id AS VARCHAR) AS patient_id,
                   CAST({ORIGIN} - to_days(CAST(round(birth_age * 365.25 + 182.625) AS INTEGER)) AS DATE) AS birth_date,
                   {t('death_min')} AS death_dttm,
                   race AS race_name,
                   CASE race WHEN 'Asian' THEN 'asian' ELSE 'unknown' END AS race_category,
                   'unknown' AS ethnicity_category,
                   sex AS sex_name,
                   CASE sex WHEN 'M' THEN 'male' WHEN 'F' THEN 'female' ELSE 'unknown' END AS sex_category,
                   'unknown' AS language_category
            FROM s"""
        return sql, ("patient_id",)

    def sql_hospitalization(self):
        ops = self.src("operations.parquet")
        sql = f"""
            SELECT CAST(subject_id AS VARCHAR) AS patient_id, CAST(hadm_id AS VARCHAR) AS hospitalization_id,
                   {t('min(admission_time)')} AS admission_dttm, {t('max(discharge_time)')} AS discharge_dttm,
                   CAST(min(age) AS INTEGER) AS age_at_admission,
                   CASE WHEN max(emop) = 1 THEN 'emergency surgery' ELSE 'elective surgery' END AS admission_type_name,
                   CASE WHEN max(emop) = 1 THEN 'other' ELSE 'elective' END AS admission_type_category,
                   CASE WHEN min(inhosp_death_time) IS NOT NULL THEN 'in-hospital death' END AS discharge_name,
                   CASE WHEN min(inhosp_death_time) IS NOT NULL THEN 'expired' ELSE 'missing' END AS discharge_category
            FROM {ops} WHERE hadm_id IS NOT NULL
            GROUP BY subject_id, hadm_id"""
        return sql, ("hospitalization_id",)

    def sql_adt(self):
        """Non-overlapping location segments per admission; priority OR > ICU > ward."""
        ops = self.con.execute(f"""
            SELECT hadm_id, admission_time, discharge_time, orin_time, orout_time
            FROM {self.src('operations.parquet')} WHERE hadm_id IS NOT NULL""").df()
        stays = self.con.execute("SELECT hadm_id, icuin_time, icuout_time FROM icu_stays").df()
        icu_by_hadm = {h: list(zip(g["icuin_time"], g["icuout_time"])) for h, g in stays.groupby("hadm_id")}
        rows = []
        for hadm, g in ops.groupby("hadm_id"):
            adm, dis = g["admission_time"].min(), g["discharge_time"].max()
            if pd.isna(adm) or pd.isna(dis) or dis <= adm:
                continue
            ors = [(a, b) for a, b in zip(g["orin_time"], g["orout_time"]) if pd.notna(a) and pd.notna(b) and b > a]
            icus = [(a, b) for a, b in icu_by_hadm.get(hadm, []) if pd.notna(a) and pd.notna(b) and b > a]
            cuts = sorted({adm, dis, *[min(max(x, adm), dis) for iv in ors + icus for x in iv]})
            segs: list[list] = []
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                mid = (lo + hi) / 2
                loc = ("procedural" if any(a <= mid < b for a, b in ors)
                       else "icu" if any(a <= mid < b for a, b in icus) else "ward")
                if segs and segs[-1][2] == loc:
                    segs[-1][1] = hi
                else:
                    segs.append([lo, hi, loc])
            rows += [(str(hadm), lo, hi, loc) for lo, hi, loc in segs]
        self.con.register("adt_segments", pd.DataFrame(rows, columns=["hadm", "lo", "hi", "loc"]))
        sql = f"""
            SELECT hadm AS hospitalization_id, '{HOSPITAL_ID}' AS hospital_id, 'academic' AS hospital_type,
                   {t('lo')} AS in_dttm, {t('hi')} AS out_dttm,
                   CASE loc WHEN 'procedural' THEN 'OR' WHEN 'icu' THEN 'ICU' ELSE 'Ward' END AS location_name,
                   loc AS location_category,
                   CAST(NULL AS VARCHAR) AS location_type   -- ICU type is not recorded in INSPIRE
            FROM adt_segments"""
        return sql, ("hospitalization_id", "in_dttm")

    def sql_vitals(self):
        m = self.map("vitals")
        ward = self.src("ward_vitals.parquet")
        ops = self.src("operations.parquet")
        sql = f"""
            SELECT CAST(h.hadm_id AS VARCHAR) AS hospitalization_id, {t('x.chart_time')} AS recorded_dttm,
                   x.item_name AS vital_name, m.vital_category, x.value * CAST(m.factor AS DOUBLE) AS vital_value,
                   nullif(m.meas_site_name, '') AS meas_site_name
            {self._in_hosp(ward)} JOIN {m} m ON m.source_table = 'ward_vitals' AND m.item_name = x.item_name
            WHERE x.value IS NOT NULL
            UNION ALL
            SELECT CAST(x.hadm_id AS VARCHAR), {t('x.chart_time')}, x.item_name, m.vital_category,
                   x.value * CAST(m.factor AS DOUBLE), nullif(m.meas_site_name, '')
            FROM intra x JOIN {m} m ON m.source_table = 'vitals' AND m.item_name = x.item_name
            WHERE x.value IS NOT NULL
            UNION ALL   -- one height / weight per admission and value (operations repeat them per operation)
            SELECT DISTINCT CAST(hadm_id AS VARCHAR), {t('admission_time')}, 'height', 'height_cm', height, NULL
            FROM {ops} WHERE height IS NOT NULL AND hadm_id IS NOT NULL
            UNION ALL
            SELECT DISTINCT CAST(hadm_id AS VARCHAR), {t('admission_time')}, 'weight', 'weight_kg', weight, NULL
            FROM {ops} WHERE weight IS NOT NULL AND hadm_id IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_labs(self):
        m = self.map("labs")
        sql = f"""
            SELECT CAST(h.hadm_id AS VARCHAR) AS hospitalization_id,
                   {t('x.chart_time')} AS lab_order_dttm, {t('x.chart_time')} AS lab_collect_dttm,
                   {t('x.chart_time')} AS lab_result_dttm,
                   m.lab_order_category, x.item_name AS lab_name, m.lab_category,
                   CAST(x.value AS VARCHAR) AS lab_value,
                   x.value * CAST(m.factor AS DOUBLE) AS lab_value_numeric,
                   m.reference_unit, m.lab_specimen_category
            {self._in_hosp(self.src('labs.parquet'))} JOIN {m} m ON m.item_name = x.item_name
            WHERE x.value IS NOT NULL"""
        return sql, ("hospitalization_id", "lab_result_dttm")

    def sql_respiratory_support(self):
        ward = self.src("ward_vitals.parquet")
        sql = f"""
            WITH w AS (
                SELECT x.subject_id, x.chart_time, h.hadm_id,
                       max(CASE WHEN item_name = 'vent' THEN value END) AS vent,
                       max(CASE WHEN item_name = 'fio2' THEN value END) AS fio2
                {self._in_hosp(ward)} WHERE item_name IN ('vent', 'fio2')
                GROUP BY ALL),
            o AS (
                SELECT x.hadm_id, x.chart_time,
                       max(CASE WHEN item_name = 'fio2' THEN value END) AS fio2,
                       max(CASE WHEN item_name = 'peep' THEN value END) AS peep,
                       max(CASE WHEN item_name = 'vt' THEN value END) AS vt,
                       max(CASE WHEN item_name = 'pip' THEN value END) AS pip,
                       max(CASE WHEN item_name = 'pmean' THEN value END) AS pmean,
                       max(CASE WHEN item_name = 'pplat' THEN value END) AS pplat,
                       max(CASE WHEN item_name = 'minvol' THEN value END) AS minvol
                FROM intra x WHERE item_name IN ('fio2', 'peep', 'vt', 'pip', 'pmean', 'pplat', 'minvol')
                GROUP BY ALL)
            SELECT CAST(hadm_id AS VARCHAR) AS hospitalization_id, {t('chart_time')} AS recorded_dttm,
                   CASE WHEN vent = 1 THEN 'ventilated (ward_vitals)' END AS device_name,
                   -- ward_vitals only distinguishes invasive ventilation; FiO2 21% without ventilation is
                   -- room air, higher FiO2 without ventilation is an unrecorded O2 device ('other')
                   CASE WHEN vent = 1 THEN 'imv' WHEN vent = 0 AND fio2 <= 21 THEN 'room_air'
                        WHEN vent = 0 AND fio2 > 21 THEN 'other' END AS device_category,
                   fio2 / 100.0 AS fio2_set
            FROM w WHERE vent IS NOT NULL OR fio2 IS NOT NULL
            UNION ALL BY NAME
            SELECT CAST(hadm_id AS VARCHAR) AS hospitalization_id, {t('chart_time')} AS recorded_dttm,
                   'anaesthesia ventilator' AS device_name, 'imv' AS device_category,
                   fio2 / 100.0 AS fio2_set, peep AS peep_obs, vt AS tidal_volume_obs,
                   pip AS peak_inspiratory_pressure_obs, pmean AS mean_airway_pressure_obs,
                   pplat AS plateau_pressure_obs, minvol AS minute_vent_obs
            FROM o"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_patient_assessments(self):
        sql = f"""
            SELECT CAST(h.hadm_id AS VARCHAR) AS hospitalization_id, {t('x.chart_time')} AS recorded_dttm,
                   x.item_name AS assessment_name,
                   CASE x.item_name WHEN 'gcs_e' THEN 'gcs_eye' WHEN 'gcs_m' THEN 'gcs_motor'
                                    WHEN 'gcs_v' THEN 'gcs_verbal' END AS assessment_category,
                   'neurological' AS assessment_group, x.value AS numerical_value
            {self._in_hosp(self.src('ward_vitals.parquet'))}
            WHERE x.item_name IN ('gcs_e', 'gcs_m', 'gcs_v') AND x.value IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    # -- medications ---------------------------------------------------------------------------------
    def _med_group(self, table: str) -> str:
        mapping = table_schema(table).get("med_category_to_group_mapping") or {}
        cases = " ".join(f"WHEN '{k}' THEN '{v if isinstance(v, str) else v[0]}'" for k, v in mapping.items())
        return f"CASE med_category {cases} END"

    def _intraop_meds(self, kind: str) -> str:
        m = self.map("meds_intraop")
        return f"""
            SELECT CAST(x.hadm_id AS VARCHAR) AS hospitalization_id, x.op_id, x.chart_time, x.item_name AS med_name,
                   m.med_category, m.med_dose_unit, m.mar_action_category, x.value AS med_dose
            FROM intra x JOIN {m} m ON m.item_name = x.item_name AND m.clif_table = '{kind}'
            WHERE x.value IS NOT NULL"""

    def _ward_meds(self) -> str:
        """``medications``: drug records of the whole stay (time, route po/iv/ex, generic name; no dose) -> ``wmed``.

        * combination products are released as one row per component at the same time and route; listed pairs
          (``maps/inspire/combinations.csv``) are merged into the product before lexicon resolution;
        * route po -> enteral, iv -> iv; 'ex' (inhaled, ophthalmic, topical, transdermal) has no CLIF route
          category and is not a CLIF administration;
        * the source states no administration type: a drug with an intermittent category is a dose, an IV record
          of an infusion-only drug an infusion; a local intermittent extension of a drug CLIF 3.0 gives as an
          infusion (``untyped_iv``) types its oral records only, so its IV records stay infusions;
        * inside an operation, drugs that the intra-operative ``vitals`` stream records with doses are left to it.
        """
        if self.con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'wmed'").fetchone()[0]:
            return "wmed"
        combos, intraop = self.map("combinations"), self.map("meds_intraop")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE wmed_raw AS
            WITH r AS (SELECT DISTINCT subject_id, chart_time, route, lower(trim(drug_name)) AS name
                       FROM {self.src('medications.parquet')} WHERE drug_name IS NOT NULL AND chart_time IS NOT NULL),
            pairs AS (
                SELECT a.subject_id, a.chart_time, a.route, c.product, a.name AS a_name, b.name AS b_name
                FROM r a JOIN {combos} c ON c.component_a = a.name
                JOIN r b ON b.subject_id = a.subject_id AND b.chart_time = a.chart_time AND b.route = a.route
                        AND b.name = c.component_b)
            SELECT subject_id, chart_time, route, product AS name FROM pairs
            UNION ALL
            SELECT r.subject_id, r.chart_time, r.route, r.name FROM r
            WHERE NOT EXISTS (SELECT 1 FROM pairs p WHERE p.subject_id = r.subject_id AND p.chart_time = r.chart_time
                              AND p.route = r.route AND r.name IN (p.a_name, p.b_name))""")
        lk = self.regex_lookup("drugs", "SELECT name AS txt FROM wmed_raw", "drugs",
                               ("ingredient", "continuous", "intermittent", "untyped_iv"), table=self.lexicon("drugs"))
        self.con.execute(f"""
            CREATE OR REPLACE TABLE wmed AS
            SELECT w.subject_id, w.chart_time, w.route, w.name, h.hadm_id, l.ingredient,
                   nullif(l.continuous, '') AS continuous, nullif(l.intermittent, '') AS intermittent,
                   coalesce(l.untyped_iv, '') = 'continuous' AS iv_is_infusion,
                   CASE w.route WHEN 'po' THEN 'enteral' WHEN 'iv' THEN 'iv' END AS route_category,
                   EXISTS (SELECT 1 FROM {self.src('operations.parquet')} o
                           WHERE o.subject_id = w.subject_id AND w.chart_time BETWEEN o.orin_time AND o.orout_time
                             AND coalesce(nullif(l.intermittent, ''), nullif(l.continuous, ''))
                                 IN (SELECT med_category FROM {intraop})) AS intraop_covered
            FROM wmed_raw w
            JOIN hosp_win h ON h.subject_id = w.subject_id AND w.chart_time BETWEEN h.adm_min AND h.dis_min
            LEFT JOIN {lk} l ON l.txt = w.name""")
        self.coverage("medication", "ward_drug_name", """
            SELECT name AS txt, count(*) AS n, bool_or(ingredient IS NOT NULL) AS mapped FROM wmed GROUP BY 1""")
        return "wmed"

    def sql_medication_admin_continuous(self):
        # intra-operative rates: one row per rate change (uniform rule); ward/ICU records of infusion-only drugs
        # carry no rate and are presence events ('going')
        # one stream per admission and item: INSPIRE copies an infusion's records to every operation of the
        # admission and keeps some (the zeros at an operation's end) under one only, so streams built per
        # operation missed rate changes and stops (#95). A row per change; leading zeros start nothing.
        w = self._ward_meds()
        sql = f"""
            WITH r AS ({self._intraop_meds('continuous')}),
            c AS (SELECT *, lag(med_dose) OVER (PARTITION BY hospitalization_id, med_name ORDER BY chart_time, med_dose)
                                AS prev
                  FROM r)
            SELECT hospitalization_id, {t('chart_time')} AS admin_dttm, med_name, med_category,
                   {self._med_group('medication_admin_continuous')} AS med_group,
                   'iv' AS med_route_category, med_dose, med_dose_unit, NULL AS mar_action_name,
                   CASE WHEN med_dose = 0 THEN 'stop' WHEN coalesce(prev, 0) = 0 THEN 'start' ELSE 'dose_change' END
                       AS mar_action_category,
                   'administered' AS mar_action_group
            FROM c WHERE med_dose IS DISTINCT FROM coalesce(prev, 0)
            UNION ALL
            SELECT CAST(hadm_id AS VARCHAR), {t('chart_time')}, name, med_category,
                   {self._med_group('medication_admin_continuous')}, 'iv', NULL, NULL,
                   'ward medication record (rate not recorded)', 'going', 'administered'
            FROM (SELECT *, continuous AS med_category FROM {w}
                  WHERE continuous IS NOT NULL AND (intermittent IS NULL OR iv_is_infusion) AND route_category = 'iv'
                    AND NOT intraop_covered)"""
        return sql, ("hospitalization_id", "admin_dttm")

    def sql_medication_admin_intermittent(self):
        w = self._ward_meds()
        sql = f"""
            WITH r AS ({self._intraop_meds('intermittent')})
            SELECT hospitalization_id, {t('chart_time')} AS admin_dttm, med_name, med_category,
                   {self._med_group('medication_admin_intermittent')} AS med_group,
                   'iv' AS med_route_name, 'iv' AS med_route_category, med_dose, med_dose_unit,
                   'intra-operative bolus' AS mar_action_name, 'given' AS mar_action_category,   -- as CLIF-MIMIC/-AUMC
                   'administered' AS mar_action_group
            FROM r WHERE med_dose > 0
            UNION ALL
            SELECT CAST(hadm_id AS VARCHAR), {t('chart_time')}, name, med_category,
                   {self._med_group('medication_admin_intermittent')}, route, route_category, NULL, NULL,
                   'ward medication record (dose not recorded)', 'given', 'administered'
            FROM (SELECT *, intermittent AS med_category FROM {w}
                  WHERE intermittent IS NOT NULL AND route_category IS NOT NULL AND NOT intraop_covered
                    AND NOT (route_category = 'iv' AND iv_is_infusion AND continuous IS NOT NULL))"""
        return sql, ("hospitalization_id", "admin_dttm")

    # -- support devices ------------------------------------------------------------------------------
    def _ward_flag(self, item: str) -> str:
        return f"""SELECT CAST(h.hadm_id AS VARCHAR) AS hospitalization_id, {t('x.chart_time')} AS recorded_dttm
                   {self._in_hosp(self.src('ward_vitals.parquet'))} WHERE x.item_name = '{item}' AND x.value = 1"""

    def sql_crrt_therapy(self):
        # INSPIRE records CRRT presence only; mode and flows are unknown (left null, never guessed)
        sql = f"SELECT *, 'crrt (ward_vitals flag)' AS crrt_mode_name FROM ({self._ward_flag('crrt')})"
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_mcs(self):
        # CLIF 3.0 long-form MCS; INSPIRE records presence only, so device/config/settings stay null
        sql = f"""
            SELECT *, 'ecmo (ward_vitals flag)' AS support_name, 'ecmo' AS support_category
            FROM ({self._ward_flag('ecmo')})
            UNION ALL
            SELECT *, 'iabp (ward_vitals flag)', 'ext_iabp' FROM ({self._ward_flag('iabp')})"""
        return sql, ("hospitalization_id", "recorded_dttm")

    # -- diagnoses / procedures -----------------------------------------------------------------------
    def sql_hospital_diagnosis(self):
        sql = f"""
            SELECT DISTINCT CAST(h.hadm_id AS VARCHAR) AS hospitalization_id,
                   replace(x.icd10_cm, '.', '') AS diagnosis_code, 'ICD10CM' AS diagnosis_code_format,
                   0 AS diagnosis_primary
            {self._in_hosp(self.src('diagnosis.parquet'))} WHERE x.icd10_cm IS NOT NULL"""
        return sql, ("hospitalization_id",)

    def sql_patient_procedures(self):
        sql = f"""
            SELECT CAST(hadm_id AS VARCHAR) AS hospitalization_id,
                   {t('coalesce(opstart_time, orin_time)')} AS procedure_billed_dttm,
                   icd10_pcs AS procedure_code, 'ICD10PCS' AS procedure_code_format
            FROM {self.src('operations.parquet')} WHERE icd10_pcs IS NOT NULL AND hadm_id IS NOT NULL"""
        return sql, ("hospitalization_id", "procedure_billed_dttm")

    # -- fluid balance (CLIF 3.0 input/output) ---------------------------------------------------------
    def _io(self, direction: str) -> str:
        m = self.map("io")
        return f"""
            SELECT CAST(x.hadm_id AS VARCHAR) AS hospitalization_id, {t('x.chart_time')} AS recorded_dttm,
                   x.item_name AS name, m.category, m.group_name, x.value * CAST(m.factor AS DOUBLE) AS volume
            FROM intra x JOIN {m} m ON m.source_table = 'vitals' AND m.item_name = x.item_name AND m.direction = '{direction}'
            WHERE x.value > 0
            UNION ALL
            SELECT CAST(h.hadm_id AS VARCHAR), {t('x.chart_time')}, x.item_name, m.category, m.group_name,
                   x.value * CAST(m.factor AS DOUBLE)
            {self._in_hosp(self.src('ward_vitals.parquet'))}
            JOIN {m} m ON m.source_table = 'ward_vitals' AND m.item_name = x.item_name AND m.direction = '{direction}'
            WHERE x.value > 0"""

    def sql_input(self):
        sql = f"""SELECT hospitalization_id, recorded_dttm, name AS input_name, category AS input_category,
                         group_name AS input_group, volume AS input_volume FROM ({self._io('input')})"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_output(self):
        sql = f"""SELECT hospitalization_id, recorded_dttm, name AS output_name, category AS output_category,
                         group_name AS output_group, volume AS output_volume FROM ({self._io('output')})"""
        return sql, ("hospitalization_id", "recorded_dttm")

    # -- ICU stay context ----------------------------------------------------------------------------
    def sql_ext_icu_stay(self):
        """Admission diagnosis group: the operated organ system of the index operation, the ICD-10-PCS body system
        (``maps/inspire/pcs_body_systems.csv``; the longest matching prefix; other sections: other)."""
        ops = self.src("operations.parquet")
        pcs = self.map("pcs_body_systems")
        self.coverage("ext_icu_stay", "pcs_body_systems", f"""
            SELECT substr(o.icd10_pcs, 1, 2) AS txt, count(*) AS n, bool_or(m.prefix IS NOT NULL) AS mapped
            FROM {ops} o LEFT JOIN {pcs} m ON starts_with(o.icd10_pcs, m.prefix)
            WHERE o.icd10_pcs IS NOT NULL GROUP BY 1""")
        # index operation of each ICU stay: the latest operation that started at or before ICU admission (its
        # recorded OR exit is up to ~20 min after the ICU admission in a quarter of the subset's stays)
        sql = f"""
            WITH idx AS (
                SELECT s.subject_id, s.hadm_id, s.icuin_time, s.icuout_time,
                       arg_max(o.op_id, coalesce(o.orin_time, o.opstart_time)) AS op_id
                FROM icu_stays s JOIN {ops} o ON o.subject_id = s.subject_id AND o.hadm_id = s.hadm_id
                     AND coalesce(o.orin_time, o.opstart_time, o.orout_time) <= s.icuin_time
                GROUP BY ALL),
            d AS (SELECT subject_id, least(min(inhosp_death_time), min(allcause_death_time)) AS death_min
                  FROM {ops} GROUP BY subject_id),
            body AS (SELECT o.op_id, arg_max(m.system, length(m.prefix)) AS system
                     FROM {ops} o JOIN {pcs} m ON starts_with(o.icd10_pcs, m.prefix) GROUP BY 1)
            SELECT 'inspire:' || CAST(s.subject_id AS VARCHAR) || '_' || CAST(CAST(s.icuin_time AS BIGINT) AS VARCHAR) AS stay_id,
                   CAST(s.subject_id AS VARCHAR) AS patient_id, CAST(s.hadm_id AS VARCHAR) AS hospitalization_id,
                   'inspire' AS dataset, 'inspire' AS site_id,
                   {t('s.icuin_time')} AS icu_in_dttm, {t('s.icuout_time')} AS icu_out_dttm,
                   false AS clock_valid, false AS demographics_binned,
                   CASE WHEN o.op_id IS NOT NULL THEN 'or' ELSE 'unknown' END AS admission_source,
                   CASE WHEN o.emop = 1 THEN 'emergency_surgery' WHEN o.emop = 0 THEN 'elective_surgery'
                        ELSE 'unknown' END AS surgical_status,
                   CASE WHEN o.op_id IS NOT NULL THEN to_json(struct_pack(
                        department := o.department, asa := o.asa, anesthesia := o.antype, icd10_pcs := o.icd10_pcs,
                        op_minutes := o.opend_time - o.opstart_time,
                        cpb := (o.cpbon_time IS NOT NULL)))::VARCHAR END AS surgery_context,
                   {group_sql('o.op_id IS NOT NULL OR NULL', "coalesce(b.system, CASE WHEN o.icd10_pcs IS NOT NULL THEN 'other' END)")}
                       AS adm_dx_group,
                   CAST(row_number() OVER (PARTITION BY s.subject_id ORDER BY s.icuin_time) AS INTEGER) AS stay_seq,
                   CASE WHEN s.icuout_time IS NULL THEN 'unknown'
                        WHEN d.death_min IS NOT NULL AND d.death_min <= s.icuout_time THEN 'dead'
                        ELSE 'alive' END AS icu_outcome,
                   CASE WHEN o.asa IS NOT NULL THEN 'asa' END AS severity_name,
                   CAST(o.asa AS DOUBLE) AS severity_value
            FROM icu_stays s LEFT JOIN idx i USING (subject_id, hadm_id, icuin_time)
            LEFT JOIN {ops} o ON o.op_id = i.op_id
            LEFT JOIN body b ON b.op_id = o.op_id
            LEFT JOIN d ON d.subject_id = s.subject_id
            WHERE s.icuout_time IS NOT NULL"""
        return sql, ("stay_id",)
