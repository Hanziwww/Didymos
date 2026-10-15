"""zigongICU 1.1 (PhysioNet: Critical care database comprising patients with infection at Zigong Fourth People's
Hospital, China) -> CLIF (our adapter). Inference only: Paper P's transfer analysis (``docs/p1_paper_protocol.md``
amendment 16a); the model never trains on it.

The release has eight tables, one hospitalization per patient (``PATIENT_ID``, ``INP_NO``): ``dtBaseline`` (age,
sex, admitting department, infection site, ICU and hospital discharge times), ``dtTransfer`` (department segments),
``dtNursingChart`` (vitals, ventilation, consciousness, urine volume), ``dtLab``, ``dtDrugs`` (prescription times
without doses, rates or stops), ``dtICD`` and ``dtOutCome`` (vital status and time of death at follow-up).

Conventions:
* times are hours from hospital admission, placed on the shared origin 2019-01-01 (dates are not real; ``clock_valid``
  is false); rows outside a hospitalization's window are dropped by the ICU-stay window downstream. Birth date =
  origin - (age x 365.25 + 182.625) days. The nursing chart's ``ChartTime`` is a date, not an hour (one time of day
  per stay; a median 26 rows per stay-day): its rows are spread in their recorded order over the day
  (:meth:`ZigongAdapter._nursing`), an approximation, so endpoints defined from the nursing chart have no recorded
  hourly timing in this database;
* ICU stays: runs of ICU-department transfer segments (``maps/zigong/care_units.csv``: the EICU, the ICU wards, the
  neurosurgical and neurology NICUs, the respiratory RICU, the vascular-surgery ICU), merged when one starts within
  1 h of the previous one's end;
* death: the follow-up's time of death when it falls at most 24 h after hospital discharge, or a discharge status
  'Dead' in ``dtICD``, is an in-hospital death, placed at the hospital discharge (as HiRID and AmsterdamUMCdb record
  mortality): the follow-up records the date of death (deaths fall up to a day before the recorded discharge), and
  patients discharged to die at home die within the day. An in-hospital death whose ICU stay ends at the hospital
  discharge (within 1 h) is an ICU death. A later death stays on the patient;
* vitals: the nursing chart's columns (``maps/zigong/vitals.csv``); MAP is not charted, so MAP = (SBP + 2 DBP)/3 from
  the same row's systolic and diastolic pressures (cuff and arterial separately);
* ventilation: the ventilator mode (``Breathing_pattern``) with the oxygen-delivery field (``Oxygen_inhalation_mode``,
  Chinese terms) and the endotracheal tube's size (``Endotracheal_intubation``): a ventilator mode with an airway
  (endotracheal or tracheostomy tube) or with an invasive mode is invasive ventilation; a bilevel or CPAP mode without
  an airway is non-invasive; without a mode, the oxygen device. FiO2 is recorded only as its unit and is not mapped;
* drugs: prescription times only, through the shared lexicon (three product names aliased first,
  ``maps/zigong/drug_aliases.csv``). Infusions are the vasoactive drugs (norepinephrine, epinephrine, dopamine,
  dobutamine, phenylephrine, vasopressin, milrinone, isoproterenol, nitroglycerin, nitroprusside), the infusion
  sedatives and opioids (propofol, dexmedetomidine, remifentanil, sufentanil) and every injectable product whose
  lexicon category is continuous only: an episode starts at the first record, continues while the next record follows
  within 25 h (the prescription cycle: gaps' 90th percentile 24.4 h for norepinephrine), and stops 6 h after its last
  record; doses are not recorded (unit unknown). Other drugs with a CLIF intermittent category are intermittent
  administrations at their record, route from the product's form (injection, tablet, inhalation; subcutaneous for
  insulin and the low-molecular-weight heparins), dose not recorded. Metaraminol has no lexicon entry;
* CRRT: a 'Hemofiltration replacement base fluid' record is CRRT documentation (haemofiltration, CVVH), the only
  renal replacement source.
"""

from __future__ import annotations

from icutwin.clif.adapters.base import ClifAdapter, minutes_after
from icutwin.clif.conform import extensions, table_schema

ORIGIN = "TIMESTAMPTZ '2019-01-01 00:00:00+00'"
HOSPITAL_ID = "zigong_fourth"
MERGE_H, GAP_H, OVERHANG_H, DEATH_AFTER_DISCHARGE_H = 1.0, 25.0, 6.0, 24.0
INFUSIONS = ("norepinephrine", "epinephrine", "dopamine", "dobutamine", "phenylephrine", "vasopressin", "milrinone",
             "isoproterenol", "nitroglycerin", "nitroprusside", "propofol", "dexmedetomidine", "remifentanil",
             "ext_sufentanil")
NUMERIC = r"^\s*-?[0-9]+(\.[0-9]+)?\s*$"
INJECTABLE = r"inject|\binj\b"


def t(hours_sql: str) -> str:
    """A time ``hours_sql`` hours after the hospitalization's admission, on the shared origin."""
    return minutes_after(ORIGIN, f"({hours_sql}) * 60.0")


def num(col: str) -> str:
    return f"TRY_CAST(nullif(trim(CAST({col} AS VARCHAR)), 'NA') AS DOUBLE)"


class ZigongAdapter(ClifAdapter):
    dataset = "zigong"
    tables = (
        "patient", "hospitalization", "adt", "vitals", "labs", "respiratory_support", "patient_assessments",
        "medication_admin_continuous", "medication_admin_intermittent", "crrt_therapy", "output",
        "hospital_diagnosis", "ext_icu_stay",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cu = self.map("care_units")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE base AS
            SELECT CAST(PATIENT_ID AS VARCHAR) AS pid, CAST(INP_NO AS VARCHAR) AS hid,
                   {num('Age')} AS age, SEX AS sex, AdmitDept AS admit_dept,
                   {num('DISCHARGE_DATE_TIME')} AS dis_h
            FROM {self.src('dtBaseline.parquet')}""")
        # the follow-up's death: in hospital when at most 24 h after discharge, or with a discharge status 'Dead'
        self.con.execute(f"""
            CREATE OR REPLACE TABLE deaths AS
            WITH o AS (SELECT CAST(PATIENT_ID AS VARCHAR) AS pid, {num('Death_Date')} AS death_h
                       FROM {self.src('dtOutCome.parquet')} WHERE Follow_Vital = 'Death'),
                 st AS (SELECT CAST(INP_NO AS VARCHAR) AS hid, bool_or(Status_Discharge = 'Dead') AS dead
                        FROM {self.src('dtICD.parquet')} GROUP BY 1)
            SELECT b.pid, b.hid, o.death_h,
                   coalesce(o.death_h <= b.dis_h + {DEATH_AFTER_DISCHARGE_H}, false) OR coalesce(st.dead, false) AS in_hosp
            FROM base b LEFT JOIN o USING (pid) LEFT JOIN st USING (hid)""")
        lk = self.regex_lookup("dept", f"""SELECT TransferDept AS txt FROM {self.src('dtTransfer.parquet')}
                                          UNION SELECT AdmitDept FROM {self.src('dtBaseline.parquet')}""",
                               "care_units", ("location_category", "location_type", "is_icu"), table=cu)
        self.con.execute(f"""
            CREATE OR REPLACE TABLE seg AS
            SELECT CAST(x.INP_NO AS VARCHAR) AS hid, {num('x.StartTime')} AS s, {num('x.StopTime')} AS e,
                   x.TransferDept AS dept, d.location_category, d.location_type, d.is_icu = '1' AS icu
            FROM {self.src('dtTransfer.parquet')} x JOIN {lk} d ON d.txt = x.TransferDept
            WHERE {num('x.StartTime')} IS NOT NULL AND {num('x.StopTime')} > {num('x.StartTime')}""")
        # ICU stays: runs of ICU segments, merged across gaps of at most 1 h
        self.con.execute(f"""
            CREATE OR REPLACE TABLE icu_stays AS
            WITH s AS (SELECT hid, s, e, location_type FROM seg WHERE icu),
                 m AS (SELECT *, CASE WHEN lag(e) OVER w IS NULL OR s > lag(e) OVER w + {MERGE_H} THEN 1 ELSE 0 END AS new
                       FROM s WINDOW w AS (PARTITION BY hid ORDER BY s, e)),
                 g AS (SELECT *, sum(new) OVER (PARTITION BY hid ORDER BY s, e ROWS UNBOUNDED PRECEDING) AS k FROM m)
            SELECT hid, k, min(s) AS s, max(e) AS e, arg_min(location_type, s) AS unit_type FROM g GROUP BY hid, k""")

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
            SELECT b.pid AS patient_id,
                   CAST({ORIGIN} - to_days(CAST(round(b.age * 365.25 + 182.625) AS INTEGER)) AS DATE) AS birth_date,
                   CASE WHEN d.in_hosp THEN {t('b.dis_h')} WHEN d.death_h IS NOT NULL THEN {t('d.death_h')} END AS death_dttm,
                   'unknown' AS race_category, 'unknown' AS ethnicity_category,
                   b.sex AS sex_name,
                   CASE b.sex WHEN 'Male' THEN 'male' WHEN 'Female' THEN 'female' ELSE 'unknown' END AS sex_category,
                   'unknown' AS language_category
            FROM base b JOIN deaths d USING (pid, hid)"""
        return sql, ("patient_id",)

    def sql_hospitalization(self):
        sql = f"""
            SELECT b.pid AS patient_id, b.hid AS hospitalization_id, {t('0')} AS admission_dttm, {t('b.dis_h')} AS discharge_dttm,
                   CAST(b.age AS INTEGER) AS age_at_admission, b.admit_dept AS admission_type_name,
                   CASE WHEN regexp_matches(lower(b.admit_dept), 'emergency|eicu') THEN 'ed' ELSE 'other' END AS admission_type_category,
                   CASE WHEN d.in_hosp THEN 'in-hospital death' END AS discharge_name,
                   CASE WHEN d.in_hosp THEN 'expired' ELSE 'other' END AS discharge_category
            FROM base b JOIN deaths d USING (pid, hid)
            WHERE b.dis_h IS NOT NULL"""
        return sql, ("hospitalization_id",)

    def sql_adt(self):
        sql = f"""
            SELECT hid AS hospitalization_id, '{HOSPITAL_ID}' AS hospital_id, 'academic' AS hospital_type,
                   {t('s')} AS in_dttm, {t('e')} AS out_dttm, dept AS location_name,
                   location_category, location_type
            FROM seg"""
        return sql, ("hospitalization_id", "in_dttm")

    def _nursing(self, cols: tuple[str, ...]) -> str:
        """The nursing chart with reconstructed times ``h``: ``ChartTime`` is the record's date (each stay's chart
        times share one time of day, 00:00, while its labs, drugs and transfers carry hours), and the rows of a day
        are in chronological order (lag-1 heart-rate differences in row order are well below random pairs). The
        k-th of a day's n rows is placed at the end of the k-th of n equal slots over the day's charted interval
        (admission or 00:00 to 24:00 or discharge), so no value precedes its measurement for evenly spaced charting."""
        if not self.con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'nursing'").fetchone()[0]:
            self.con.execute(f"""
                CREATE OR REPLACE TABLE nursing AS
                WITH n AS (SELECT *, CAST(INP_NO AS VARCHAR) AS hid, {num('ChartTime')} AS day_h
                           FROM {self.src('dtNursingChart.parquet')} WHERE {num('ChartTime')} IS NOT NULL),
                     w AS (SELECT n.*, greatest(n.day_h, 0.0) AS s, least(n.day_h + 24.0, b.dis_h) AS e,
                                  row_number() OVER (PARTITION BY n.hid, n.day_h ORDER BY n.column00) AS k,
                                  count(*) OVER (PARTITION BY n.hid, n.day_h) AS m
                           FROM n JOIN base b USING (hid))
                SELECT * EXCLUDE (s, e, k, m), s + k * (e - s) / m AS h FROM w WHERE e > s""")
        sel = ", ".join(f'"{c}"' for c in cols)
        return f"SELECT hid, h, {sel} FROM nursing"

    def sql_vitals(self):
        m = self.map("vitals")
        cols = ("temperature", "heart_rate", "breathing", "Blood_oxygen_saturation", "Blood_pressure_high",
                "Blood_pressure_low", "Invasive_SBP", "Invasive_diastolic_blood_pressure")
        unpivot = " UNION ALL ".join(f"SELECT hid, h, '{c}' AS col, {num(chr(34) + c + chr(34))} AS v FROM nc" for c in cols)
        sql = f"""
            WITH nc AS ({self._nursing(cols)}),
                 u AS ({unpivot})
            SELECT u.hid AS hospitalization_id, {t('u.h')} AS recorded_dttm, u.col AS vital_name, m.vital_category,
                   u.v * CAST(m.factor AS DOUBLE) AS vital_value, nullif(m.meas_site_name, '') AS meas_site_name
            FROM u JOIN {m} m ON m."column" = u.col WHERE u.v IS NOT NULL
            UNION ALL   -- MAP is not charted: (SBP + 2 DBP) / 3 from the same row, cuff and arterial apart
            SELECT hid, {t('h')}, 'MAP from cuff SBP and DBP', 'map',
                   ({num('Blood_pressure_high')} + 2 * {num('Blood_pressure_low')}) / 3.0, 'cuff'
            FROM nc WHERE {num('Blood_pressure_high')} IS NOT NULL AND {num('Blood_pressure_low')} IS NOT NULL
            UNION ALL
            SELECT hid, {t('h')}, 'MAP from invasive SBP and DBP', 'map',
                   ({num('Invasive_SBP')} + 2 * {num('Invasive_diastolic_blood_pressure')}) / 3.0, 'arterial'
            FROM nc WHERE {num('Invasive_SBP')} IS NOT NULL AND {num('Invasive_diastolic_blood_pressure')} IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_labs(self):
        m = self.map("labs")
        lab = self.src("dtLab.parquet")
        self.coverage("labs", "item", f"""
            SELECT x.Item AS txt, count(*) AS n, bool_or(m.lab_category IS NOT NULL) AS mapped
            FROM {lab} x LEFT JOIN {m} m ON m.item = x.Item AND coalesce(m.unit, '') = coalesce(x.Unit, '')
            WHERE NOT regexp_matches(lower(x.Category), 'urin') GROUP BY 1""")
        sql = f"""
            SELECT CAST(x.INP_NO AS VARCHAR) AS hospitalization_id,
                   {t(num('x.LabTime'))} AS lab_order_dttm, {t(num('x.LabTime'))} AS lab_collect_dttm,
                   {t(num('x.LabTime'))} AS lab_result_dttm,
                   x.Category AS lab_order_name, m.lab_order_category, x.Item AS lab_name, m.lab_category,
                   CAST(x.LabValue AS VARCHAR) AS lab_value,
                   {num('x.LabValue')} * CAST(m.factor AS DOUBLE) AS lab_value_numeric,
                   m.reference_unit, m.lab_specimen_category
            FROM {lab} x JOIN {m} m ON m.item = x.Item AND coalesce(m.unit, '') = coalesce(x.Unit, '')
            WHERE NOT regexp_matches(lower(x.Category), 'urin') AND {num('x.LabValue')} IS NOT NULL
              AND {num('x.LabTime')} IS NOT NULL
            UNION ALL   -- point-of-care glucose charted by the nurses (mmol/L)
            SELECT hid, {t('h')}, {t('h')}, {t('h')},
                   'nursing chart', 'poc', 'blood_sugar', 'glucose', CAST(blood_sugar AS VARCHAR),
                   {num('blood_sugar')} * 18.016, 'mg/dl', 'plasma_blood'
            FROM ({self._nursing(('blood_sugar',))}) WHERE {num('blood_sugar')} IS NOT NULL"""
        return sql, ("hospitalization_id", "lab_result_dttm")

    def sql_respiratory_support(self):
        cols = ("Oxygen_inhalation_mode", "Breathing_pattern", "Endotracheal_intubation", "Vt_setting",
                "Vt_Supervisor", "Oxygen_flow")
        o2 = self.regex_lookup("o2_mode", f"SELECT trim(Oxygen_inhalation_mode) AS txt FROM ({self._nursing(cols)})",
                               "o2_modes", ("device", "airway"))
        lk = self.regex_lookup("vent_mode", f"SELECT upper(trim(Breathing_pattern)) AS txt FROM ({self._nursing(cols)})",
                               "vent_modes", ("mode_category", "niv"))
        self.coverage("respiratory_support", "breathing_pattern", f"""
            SELECT x.txt, count(*) AS n, bool_or(v.mode_category IS NOT NULL) AS mapped
            FROM (SELECT upper(trim(Breathing_pattern)) AS txt FROM ({self._nursing(cols)})) x
            JOIN {lk} v USING (txt) WHERE x.txt <> '' GROUP BY 1""")
        sql = f"""
            WITH r AS (
                SELECT x.hid, x.h, trim(x.Oxygen_inhalation_mode) AS o2_txt, trim(x.Breathing_pattern) AS mode_txt,
                       o.device AS o2_device, o.airway AS o2_airway, v.mode_category, v.niv,
                       regexp_matches(coalesce(x.Endotracheal_intubation, ''), '{NUMERIC}') AS tube,
                       {num('x.Vt_setting')} AS vt_set, {num('x.Vt_Supervisor')} AS vt_obs, {num('x.Oxygen_flow')} AS lpm
                FROM ({self._nursing(cols)}) x
                LEFT JOIN {o2} o ON o.txt = trim(x.Oxygen_inhalation_mode)
                LEFT JOIN {lk} v ON v.txt = upper(trim(x.Breathing_pattern))),
            d AS (
                SELECT *, coalesce(o2_airway, CASE WHEN tube THEN 'ett' END) AS airway,
                       CASE WHEN mode_category IS NOT NULL AND (coalesce(o2_airway, CASE WHEN tube THEN 'ett' END) IS NOT NULL
                                                                OR niv = '0') THEN 'imv'
                            WHEN mode_category IS NOT NULL AND niv = '2' THEN 'cpap'
                            WHEN mode_category IS NOT NULL THEN 'nippv'
                            WHEN coalesce(o2_airway, CASE WHEN tube THEN 'ett' END) = 'trach' THEN 'trach_collar'
                            WHEN coalesce(o2_airway, CASE WHEN tube THEN 'ett' END) = 'ett' THEN 'imv'
                            WHEN o2_device IN ('nasal_cannula', 'face_mask', 'hfnc', 'other') THEN o2_device END AS device
                FROM r)
            SELECT hid AS hospitalization_id, {t('h')} AS recorded_dttm,
                   concat_ws(' / ', nullif(o2_txt, ''), nullif(mode_txt, '')) AS device_name, device AS device_category,
                   nullif(mode_txt, '') AS mode_name, CASE WHEN device IN ('imv', 'nippv', 'cpap') THEN mode_category END AS mode_category,
                   CASE WHEN airway = 'trach' THEN 1 ELSE 0 END AS tracheostomy,
                   CASE WHEN device IN ('imv', 'nippv', 'cpap') THEN vt_set END AS tidal_volume_set,
                   CASE WHEN device IN ('imv', 'nippv', 'cpap') THEN vt_obs END AS tidal_volume_obs,
                   CASE WHEN device IN ('nasal_cannula', 'face_mask', 'hfnc') THEN lpm END AS lpm_set
            FROM d WHERE device IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_patient_assessments(self):
        cols = ("open_one's_eyes", "motion", "language", "RASS_sedation_score")
        q = '"'
        parts = []
        for col, cat, top in (("open_one's_eyes", "gcs_eye", 4), ("motion", "gcs_motor", 6), ("language", "gcs_verbal", 5)):
            score = f"TRY_CAST(regexp_extract({q}{col}{q}, '^\\s*([1-6])', 1) AS DOUBLE)"
            parts.append(f"""SELECT hid, h, '{col.replace("'", "''")}' AS name, '{cat}' AS cat, {score} AS v
                             FROM nc WHERE {score} BETWEEN 1 AND {top}""")
        rass = f"TRY_CAST(replace(trim({q}RASS_sedation_score{q}), '+', '') AS DOUBLE)"
        parts.append(f"SELECT hid, h, 'RASS_sedation_score', 'rass', {rass} FROM nc WHERE {rass} BETWEEN -5 AND 5")
        sql = f"""
            WITH nc AS ({self._nursing(cols)})
            SELECT hid AS hospitalization_id, {t('h')} AS recorded_dttm, name AS assessment_name,
                   cat AS assessment_category, v AS numerical_value
            FROM ({' UNION ALL '.join(parts)})"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_output(self):
        sql = f"""
            SELECT hid AS hospitalization_id, {t('h')} AS recorded_dttm, 'Urine_volume' AS output_name,
                   'urinary_other' AS output_category, 'urine' AS output_group, {num('Urine_volume')} AS output_volume
            FROM ({self._nursing(('Urine_volume',))}) WHERE {num('Urine_volume')} > 0"""
        return sql, ("hospitalization_id", "recorded_dttm")

    # -- medications ---------------------------------------------------------------------------------
    def _drugs(self) -> str:
        if self.con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'drug_rx'").fetchone()[0]:
            return "drug_rx"
        drugs = self.src("dtDrugs.parquet")
        aliases = self.con.execute(f"SELECT pattern, replacement FROM {self.map('drug_aliases')} ORDER BY _row").fetchall()
        txt = "lower(d.DrugName)"
        for pattern, replacement in aliases:
            txt = f"regexp_replace({txt}, '{pattern}', '{replacement}')"
        lk = self.regex_lookup("drugs", f"SELECT {txt} AS txt FROM {drugs} d", "drugs",
                               ("ingredient", "continuous", "intermittent"), table=self.lexicon("drugs"))
        inf = ", ".join(f"'{x}'" for x in INFUSIONS)
        self.con.execute(f"""
            CREATE OR REPLACE TABLE drug_rx AS
            SELECT b.hid, {num('d.Drug_time')} AS h, d.DrugName AS name, l.ingredient,
                   nullif(l.continuous, '') AS continuous, nullif(l.intermittent, '') AS intermittent,
                   coalesce(nullif(l.continuous, '') IN ({inf})
                            OR (nullif(l.intermittent, '') IS NULL AND nullif(l.continuous, '') IS NOT NULL
                                AND regexp_matches(lower(d.DrugName), '{INJECTABLE}')), false) AS infusion
            FROM {drugs} d JOIN base b ON b.pid = CAST(d.PATIENT_ID AS VARCHAR)
            LEFT JOIN {lk} l ON l.txt = {txt}
            WHERE {num('d.Drug_time')} IS NOT NULL""")
        self.coverage("medication", "drug_name", """
            SELECT name AS txt, count(*) AS n, bool_or(continuous IS NOT NULL OR intermittent IS NOT NULL) AS mapped
            FROM drug_rx GROUP BY 1""")
        return "drug_rx"

    def sql_medication_admin_continuous(self):
        d = self._drugs()
        sql = f"""
            WITH r AS (
                SELECT hid, continuous, h, min(name) AS name,
                       lag(h) OVER (PARTITION BY hid, continuous ORDER BY h) AS prev_h,
                       lead(h) OVER (PARTITION BY hid, continuous ORDER BY h) AS next_h
                FROM {d} WHERE infusion GROUP BY hid, continuous, h),
            ev AS (
                SELECT hid, continuous, h, name,
                       CASE WHEN prev_h IS NULL OR h - prev_h > {GAP_H} THEN 'start' ELSE 'going' END AS action,
                       NULL AS action_name
                FROM r
                UNION ALL   -- inferred end: no further prescription within the cycle
                SELECT hid, continuous, h + {OVERHANG_H}, name, 'stop',
                       'inferred stop: no further prescription within {GAP_H:g} h'
                FROM r WHERE next_h IS NULL OR next_h - h > {GAP_H})
            SELECT hid AS hospitalization_id, {t('h')} AS admin_dttm, name AS med_name,
                   continuous AS med_category, {self._group('medication_admin_continuous', 'continuous')} AS med_group,
                   'iv' AS med_route_category, CASE WHEN action = 'stop' THEN 0.0 END AS med_dose,
                   action_name AS mar_action_name, action AS mar_action_category, 'administered' AS mar_action_group
            FROM ev"""
        return sql, ("hospitalization_id", "admin_dttm", "med_category")

    def sql_medication_admin_intermittent(self):
        d = self._drugs()
        route = """CASE WHEN regexp_matches(lower(name), 'enoxaparin|low molecular weight heparin|insulin|subcutaneous') THEN 'im'
                        WHEN regexp_matches(lower(name), 'inhal|nebuli|aerosol') THEN 'inhaled'
                        WHEN regexp_matches(lower(name), 'tablet|capsule|granule|oral|suspension|syrup|dry powder|pill') THEN 'enteral'
                        WHEN regexp_matches(lower(name), 'injection|for injection|infusion') THEN 'iv' END"""
        sql = f"""
            SELECT hid AS hospitalization_id, {t('h')} AS admin_dttm, name AS med_name,
                   intermittent AS med_category, {self._group('medication_admin_intermittent', 'intermittent')} AS med_group,
                   {route} AS med_route_category, 'given' AS mar_action_category, 'administered' AS mar_action_group
            FROM {d}
            WHERE intermittent IS NOT NULL AND NOT infusion AND ({route}) IS NOT NULL"""
        return sql, ("hospitalization_id", "admin_dttm")

    def sql_crrt_therapy(self):
        sql = f"""
            SELECT b.hid AS hospitalization_id, {t(num('d.Drug_time'))} AS recorded_dttm,
                   d.DrugName AS crrt_mode_name, 'cvvh' AS crrt_mode_category
            FROM {self.src('dtDrugs.parquet')} d JOIN base b ON b.pid = CAST(d.PATIENT_ID AS VARCHAR)
            WHERE d.DrugName ILIKE 'hemofiltration replacement%' AND {num('d.Drug_time')} IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    # -- diagnoses and stay context -------------------------------------------------------------------
    def sql_hospital_diagnosis(self):
        """ICD-10 codes of the Chinese national extension (e.g. J32.000X001), cut to the category and its first
        decimal (J320); morphology codes (M80700/3) are not diagnoses and are left out."""
        code = "regexp_extract(upper(ICD_Code), '^([A-Z][0-9]{2})(\\.([0-9]))?', 0)"
        sql = f"""
            SELECT DISTINCT CAST(INP_NO AS VARCHAR) AS hospitalization_id, replace({code}, '.', '') AS diagnosis_code,
                   'ICD10CM' AS diagnosis_code_format,
                   CASE WHEN TRY_CAST(Diagnosis_seq AS INTEGER) = 1 THEN 1 ELSE 0 END AS diagnosis_primary
            FROM {self.src('dtICD.parquet')}
            WHERE NOT contains(ICD_Code, '/') AND {code} <> ''"""
        return sql, ("hospitalization_id",)

    def sql_ext_icu_stay(self):
        sql = f"""
            WITH prev AS (   -- the department segment that ends where the ICU stay starts
                SELECT s.hid, s.k, arg_max(g.location_category, g.e) AS before
                FROM icu_stays s JOIN seg g ON g.hid = s.hid AND NOT g.icu AND g.e <= s.s + {MERGE_H}
                GROUP BY s.hid, s.k)
            SELECT 'zigong:' || s.hid || '_' || CAST(s.k AS VARCHAR) AS stay_id, b.pid AS patient_id,
                   s.hid AS hospitalization_id, 'zigong' AS dataset, 'zigong' AS site_id,
                   {t('s.s')} AS icu_in_dttm, {t('s.e')} AS icu_out_dttm, s.unit_type,
                   false AS clock_valid, false AS demographics_binned,
                   CASE WHEN p.before = 'ed' THEN 'ed' WHEN p.before = 'ward' THEN 'ward' WHEN p.before IS NOT NULL THEN 'other'
                        WHEN s.s <= {MERGE_H} AND regexp_matches(lower(b.admit_dept), 'emergency|eicu') THEN 'ed'
                        ELSE 'unknown' END AS admission_source,
                   'unknown' AS surgical_status,
                   CAST(row_number() OVER (PARTITION BY b.pid ORDER BY s.s) AS INTEGER) AS stay_seq,
                   CASE WHEN d.in_hosp AND s.e >= b.dis_h - {MERGE_H} THEN 'dead' ELSE 'alive' END AS icu_outcome
            FROM icu_stays s JOIN base b USING (hid) JOIN deaths d USING (pid, hid)
            LEFT JOIN prev p ON p.hid = s.hid AND p.k = s.k"""
        return sql, ("stay_id",)
