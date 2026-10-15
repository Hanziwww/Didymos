"""SICdb 1.0.8 -> CLIF (our adapter; no published CLIF ETL exists for the Salzburg Intensive Care database).

Identity: a case (``CaseID``) is one ICU admission within a hospital stay; ``PatientID`` links cases.
CLIF hospitalization = case; ``ext_icu_stay`` = case.

Time: every ``Offset`` is seconds from the case start; the case start is placed, as in the community
SICdb_MEDS ETL, at 1 July 00:00 of the admission year of the patient's first case plus
``OffsetAfterFirstAdmission``. Time of day is not recoverable (offsets relative to the case start show no
circadian structure), so ``clock_valid`` is false. ICU admission = ``ICUOffset``; ICU end = ``TimeOfStay``
(ricu ``id_cfg``); hospital discharge = ``HospitalDischargeDay`` at midday (SICdb_MEDS); death =
``OffsetOfDeath`` (may follow discharge: registry follow-up).

Signals (``data_float_h``) are hourly aggregates starting at ``Offset``: each value is stamped at the end of
its hour (Offset + 59 min), which is where S2b's hourly rule stamps aggregated monitor values and avoids
placing a value before the measurements it summarises. Age, weight and height are binned (5 years, 5 kg,
5 cm): age is stored as bin lower bound + 2 (as for HiRID), weight and height as given, all flagged
``demographics_binned``.

Medication rows are administration periods with an amount per minute (drug amounts in g, IE, ml or mol):
continuous periods become rate steps summed over overlapping periods of the same drug (as for HiRID
syringes); single doses become intermittent administrations. Interval documentation (``data_range``:
airways, lines, drains) fills the CLIF 3.0 ``airway``/``line``/``drain`` tables; ventilator values recorded
while an endotracheal tube or tracheal cannula is in place are documented invasive ventilation.
"""

from __future__ import annotations

from icutwin.clif.adapters.base import ClifAdapter
from icutwin.clif.conform import extensions, table_schema
from icutwin.clif.dx_groups import group_sql

HOSPITAL_ID = "sicdb_salzburg"
HOUR_END = 3540  # seconds: stamp hourly aggregates at minute 59 of their hour
RESP_COLS = ("fio2_set", "lpm_set", "resp_rate_set", "pressure_support_set", "pressure_control_set", "peep_set",
             "tidal_volume_obs", "resp_rate_obs", "peak_inspiratory_pressure_obs", "minute_vent_obs",
             "mean_airway_pressure_obs")
RATE_UNIT = {"g": ("mcg/min", 1e6), "IE": ("u/min", 1.0), "ml": ("ml/min", 1.0), "mol": ("mmol/min", 1e3)}
DOSE_UNIT = {"g": ("mg", 1e3), "IE": ("u", 1.0), "ml": ("ml", 1.0), "mol": ("mmol", 1e3)}


def t(seconds_sql: str, alias: str = "s") -> str:
    """Pseudotime of an offset in seconds from the case start of stay ``alias``."""
    return f"({alias}.case_start + to_microseconds(CAST(round(CAST(({seconds_sql}) AS DOUBLE) * 1e6) AS BIGINT)))"


def _unit_case(mapping: dict, unit_col: str, value_col: str) -> tuple[str, str]:
    unit = "CASE " + " ".join(f"WHEN {unit_col} = '{k}' THEN '{u}'" for k, (u, _) in mapping.items()) + " END"
    value = "CASE " + " ".join(f"WHEN {unit_col} = '{k}' THEN {value_col} * {f}" for k, (_, f) in mapping.items()) + " END"
    return unit, value


class SicdbAdapter(ClifAdapter):
    dataset = "sicdb"
    tables = (
        "patient", "hospitalization", "adt", "vitals", "labs", "respiratory_support", "patient_assessments",
        "medication_admin_continuous", "medication_admin_intermittent", "input", "output", "invasive_hemodynamics",
        "crrt_therapy", "mcs", "airway", "line", "drain", "hospital_diagnosis", "ext_icu_stay",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ref = self.src("d_references.parquet")
        self.con.execute(f"CREATE OR REPLACE TABLE ref AS SELECT * FROM {ref}")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE stays AS
            WITH c AS (
                SELECT *, arg_min(AdmissionYear, coalesce(OffsetAfterFirstAdmission, 0))
                              OVER (PARTITION BY PatientID) AS first_year
                FROM {self.src('cases.parquet')}),
            a AS (
                SELECT *, CAST(make_date(CAST(first_year AS INTEGER), 7, 1) AS TIMESTAMPTZ)
                          + to_seconds(CAST(coalesce(OffsetAfterFirstAdmission, 0) AS BIGINT)) AS case_start
                FROM c)
            SELECT CaseID AS cid, PatientID AS pid, case_start,
                   case_start + to_seconds(CAST(ICUOffset AS BIGINT)) AS icu_in,
                   case_start + to_seconds(CAST(greatest(TimeOfStay, ICUOffset) AS BIGINT)) AS icu_out,
                   greatest(case_start + to_days(CAST(HospitalDischargeDay AS INTEGER)) + INTERVAL 12 HOUR,
                            case_start + to_seconds(CAST(greatest(TimeOfStay, ICUOffset) AS BIGINT))) AS hosp_out,
                   case_start + to_seconds(CAST(OffsetOfDeath AS BIGINT)) AS death,
                   CAST(AgeOnAdmission AS INTEGER) + CASE WHEN AgeOnAdmission >= 90 THEN 0 ELSE 2 END AS age_mid,
                   a.* EXCLUDE (CaseID, PatientID, case_start, first_year)
            FROM a""")

    def _name(self, col: str) -> str:
        return f"(SELECT ReferenceValue FROM ref WHERE ReferenceGlobalID = {col})"

    def _signals(self, target: str) -> str:
        m = self.map("signals")
        return f"""FROM {self.src('data_float_h.parquet')} x
                   JOIN {m} m ON CAST(m.dataid AS BIGINT) = x.DataID AND m.target = '{target}'
                   JOIN stays s ON s.cid = x.CaseID"""

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
            WITH p AS (SELECT pid, arg_min(case_start - to_days(CAST(round(age_mid * 365.25) AS INTEGER)), case_start) AS birth,
                              min(death) AS death, arg_min(Sex, case_start) AS sex FROM stays GROUP BY pid)
            SELECT CAST(pid AS VARCHAR) AS patient_id, CAST(birth AS DATE) AS birth_date, death AS death_dttm,
                   {self._name('sex')} AS sex_name,
                   CASE {self._name('sex')} WHEN 'male' THEN 'male' WHEN 'female' THEN 'female' ELSE 'unknown' END AS sex_category,
                   'unknown' AS race_category, 'unknown' AS ethnicity_category, 'unknown' AS language_category
            FROM p"""
        return sql, ("patient_id",)

    def sql_hospitalization(self):
        dtype = self._name("HospitalDischargeType")
        sql = f"""
            SELECT CAST(pid AS VARCHAR) AS patient_id, CAST(cid AS VARCHAR) AS hospitalization_id,
                   case_start AS admission_dttm, hosp_out AS discharge_dttm, age_mid AS age_at_admission,
                   {self._name('AdmissionUrgency')} AS admission_type_name,
                   CASE {self._name('AdmissionUrgency')} WHEN 'Elective' THEN 'elective' WHEN 'Urgent' THEN 'other' END
                       AS admission_type_category,
                   {dtype} AS discharge_name,
                   CASE WHEN {dtype} IN ('Sterbefall', 'Deceased') THEN 'expired'
                        WHEN {dtype} = 'Transfer in ein anderes Krankenhaus' THEN 'acute_care_hosp'
                        WHEN {dtype} = 'Entlassung gegen Revers' THEN 'ama'
                        WHEN {dtype} IS NULL THEN 'missing'
                        ELSE 'other' END AS discharge_category   -- regular discharge: destination not recorded
            FROM stays"""
        return sql, ("hospitalization_id",)

    def sql_adt(self):
        sql = f"""
            SELECT CAST(cid AS VARCHAR) AS hospitalization_id, '{HOSPITAL_ID}' AS hospital_id, 'academic' AS hospital_type,
                   icu_in AS in_dttm, icu_out AS out_dttm, {self._name('HospitalUnit')} AS location_name,
                   'icu' AS location_category, 'general_icu' AS location_type
            FROM stays"""
        return sql, ("hospitalization_id", "in_dttm")

    def sql_vitals(self):
        sql = f"""
            SELECT CAST(s.cid AS VARCHAR) AS hospitalization_id, {t(f'x.Offset + {HOUR_END}')} AS recorded_dttm,
                   {self._name('x.DataID')} AS vital_name, m.category AS vital_category,
                   x.Val * CAST(m.factor AS DOUBLE) AS vital_value, nullif(m.meas_site_name, '') AS meas_site_name
            {self._signals('vitals')} WHERE x.Val IS NOT NULL
            UNION ALL
            SELECT CAST(cid AS VARCHAR), icu_in, 'WeightOnAdmission', 'weight_kg', WeightOnAdmission / 1000.0, NULL
            FROM stays WHERE WeightOnAdmission > 0
            UNION ALL
            SELECT CAST(cid AS VARCHAR), icu_in, 'HeightOnAdmission', 'height_cm', HeightOnAdmission, NULL
            FROM stays WHERE HeightOnAdmission > 0"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_labs(self):
        m = self.map("labs")
        sql = f"""
            SELECT CAST(s.cid AS VARCHAR) AS hospitalization_id,
                   {t('x.Offset')} AS lab_order_dttm, {t('x.Offset')} AS lab_collect_dttm, {t('x.Offset')} AS lab_result_dttm,
                   m.lab_order_category, {self._name('x.LaboratoryID')} AS lab_name, m.lab_category,
                   CAST(x.LaboratoryValue AS VARCHAR) AS lab_value,
                   x.LaboratoryValue * CAST(m.factor AS DOUBLE) AS lab_value_numeric,
                   m.reference_unit, 'plasma_blood' AS lab_specimen_category
            FROM {self.src('laboratory.parquet')} x JOIN {m} m ON CAST(m.laboratoryid AS BIGINT) = x.LaboratoryID
            JOIN stays s ON s.cid = x.CaseID WHERE x.LaboratoryValue IS NOT NULL"""
        return sql, ("hospitalization_id", "lab_result_dttm")

    def sql_respiratory_support(self):
        rng = self.map("ranges")
        cols = ", ".join(f"max(v) FILTER (WHERE col = '{c}') AS {c}" for c in RESP_COLS)
        sql = f"""
            WITH v AS (   -- (t0, t1) = time span the value describes: an hourly aggregate, or a blood-gas instant
                SELECT x.CaseID AS cid, x.Offset + {HOUR_END} AS off, x.Offset AS t0, x.Offset + 3600 AS t1,
                       m.category AS col, x.Val * CAST(m.factor AS DOUBLE) AS v, m.target
                {self._signals('resp')} WHERE x.Val IS NOT NULL
                UNION ALL
                SELECT x.CaseID, x.Offset + {HOUR_END}, x.Offset, x.Offset + 3600, m.category,
                       x.Val * CAST(m.factor AS DOUBLE), m.target
                {self._signals('resp_hfnc')} WHERE x.Val IS NOT NULL
                UNION ALL
                SELECT x.CaseID, x.Offset + {HOUR_END}, x.Offset, x.Offset + 3600, m.category,
                       x.Val * CAST(m.factor AS DOUBLE), m.target
                {self._signals('resp_o2')} WHERE x.Val IS NOT NULL
                UNION ALL   -- FiO2 recorded with the blood gas
                SELECT CaseID, "Offset", "Offset", "Offset", 'fio2_set', LaboratoryValue / 100.0, 'resp'
                FROM {self.src('laboratory.parquet')} WHERE LaboratoryID = 684 AND LaboratoryValue IS NOT NULL),
            w AS (SELECT cid, off, min(t0) AS t0, max(t1) AS t1, bool_or(target = 'resp') AS vent,
                         bool_or(target = 'resp_hfnc') AS hfnc, {cols}
                  FROM v GROUP BY cid, off),
            airway AS (
                SELECT r.CaseID AS cid, r.Offset AS a0, r.OffsetEnd AS a1, m.category
                FROM {self.src('data_range.parquet')} r JOIN {rng} m ON CAST(m.dataid AS BIGINT) = r.DataID
                WHERE m."table" = 'airway'),
            on_airway AS (   -- ventilator values whose time span overlaps an artificial-airway interval
                SELECT DISTINCT ON (w.cid, w.off) w.cid, w.off, a.category
                FROM w JOIN airway a ON a.cid = w.cid AND w.t0 <= a.a1 AND w.t1 >= a.a0
                ORDER BY w.cid, w.off, a.category = 'trach' DESC)
            SELECT CAST(s.cid AS VARCHAR) AS hospitalization_id, {t('w.off')} AS recorded_dttm,
                   CASE WHEN w.vent AND a.category IS NOT NULL THEN 'ventilator (' || a.category || ')'
                        WHEN w.hfnc THEN 'Optiflow' END AS device_name,
                   CASE WHEN w.vent AND a.category IS NOT NULL THEN 'imv' WHEN w.hfnc THEN 'hfnc' END AS device_category,
                   CASE WHEN a.category = 'trach' THEN 1 WHEN a.category = 'ett' THEN 0 END AS tracheostomy,
                   {', '.join(f'w.{c}' for c in RESP_COLS)}
            FROM w JOIN stays s ON s.cid = w.cid
            LEFT JOIN on_airway a ON a.cid = w.cid AND a.off = w.off"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_patient_assessments(self):
        sql = f"""
            SELECT CAST(s.cid AS VARCHAR) AS hospitalization_id, {t(f'x.Offset + {HOUR_END}')} AS recorded_dttm,
                   {self._name('x.DataID')} AS assessment_name, m.category AS assessment_category,
                   CASE m.category WHEN 'rass' THEN 'sedation_or_agitation' ELSE 'pain' END AS assessment_group,
                   x.Val AS numerical_value
            {self._signals('assessments')} WHERE x.Val IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_invasive_hemodynamics(self):
        sql = f"""
            SELECT CAST(s.cid AS VARCHAR) AS hospitalization_id, {t(f'x.Offset + {HOUR_END}')} AS recorded_dttm,
                   {self._name('x.DataID')} AS measurement_name, m.category AS measurement_category,
                   x.Val AS measurement_value
            {self._signals('hemodynamics')} WHERE x.Val IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_output(self):
        sql = f"""
            SELECT CAST(s.cid AS VARCHAR) AS hospitalization_id, {t(f'x.Offset + {HOUR_END}')} AS recorded_dttm,
                   {self._name('x.DataID')} AS output_name, m.category AS output_category,
                   CASE WHEN m.category = 'urinary_other' THEN 'urine' WHEN m.category LIKE 'orogastric%' THEN 'gi'
                        ELSE 'drains' END AS output_group,
                   x.Val AS output_volume
            {self._signals('output')} WHERE x.Val > 0"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_crrt_therapy(self):
        cols = ("blood_flow_rate", "dialysate_flow_rate", "pre_filter_replacement_fluid_rate",
                "post_filter_replacement_fluid_rate", "ultrafiltration_out")
        pivot = ", ".join(f"max(x.Val) FILTER (WHERE m.category = '{c}') AS {c}" for c in cols)
        sql = f"""
            WITH c AS (SELECT s.cid, x.Offset, {pivot} {self._signals('crrt')} GROUP BY s.cid, x.Offset)
            SELECT CAST(c.cid AS VARCHAR) AS hospitalization_id, {t(f'c.Offset + {HOUR_END}')} AS recorded_dttm,
                   'Prismaflex' AS crrt_mode_name,
                   -- mode from the configured flows: dialysate and replacement -> CVVHDF, one of them -> CVVHD/CVVH
                   CASE WHEN coalesce(dialysate_flow_rate, 0) > 0 AND coalesce(pre_filter_replacement_fluid_rate, 0)
                                 + coalesce(post_filter_replacement_fluid_rate, 0) > 0 THEN 'cvvhdf'
                        WHEN coalesce(dialysate_flow_rate, 0) > 0 THEN 'cvvhd'
                        WHEN coalesce(pre_filter_replacement_fluid_rate, 0) + coalesce(post_filter_replacement_fluid_rate, 0) > 0
                             THEN 'cvvh'
                        WHEN coalesce(ultrafiltration_out, 0) > 0 THEN 'scuf' END AS crrt_mode_category,
                   {', '.join(cols)}
            FROM c JOIN stays s ON s.cid = c.cid WHERE coalesce(blood_flow_rate, 0) > 0"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_mcs(self):
        sql = f"""
            SELECT CAST(s.cid AS VARCHAR) AS hospitalization_id, {t(f'x.Offset + {HOUR_END}')} AS recorded_dttm,
                   {self._name('x.DataID')} AS support_name, 'ecmo' AS support_category,
                   m.category AS setting_category, x.Val * CAST(m.factor AS DOUBLE) AS setting_value
            {self._signals('ecmo')} WHERE x.Val IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    # -- devices with placement/removal ----------------------------------------------------------------
    def _ranges(self, table: str, category_col: str, extra: str = "") -> tuple[str, tuple[str, ...]]:
        rng = self.map("ranges")
        sql = f"""
            SELECT CAST(s.cid AS VARCHAR) AS hospitalization_id, CAST(r.id AS VARCHAR) AS {table}_id,
                   {t('r.Offset')} AS placed_dttm, {t('r.OffsetEnd')} AS removed_dttm, m.category AS {category_col}{extra}
            FROM {self.src('data_range.parquet')} r JOIN {rng} m ON CAST(m.dataid AS BIGINT) = r.DataID AND m."table" = '{table}'
            JOIN stays s ON s.cid = r.CaseID"""
        return sql, ("hospitalization_id", "placed_dttm")

    def sql_airway(self):
        return self._ranges("airway", "airway_category")

    def sql_line(self):
        return self._ranges("line", "line_category", ", 'other' AS line_site")   # insertion site not recorded

    def sql_drain(self):
        return self._ranges("drain", "drain_category", ", 'unknown' AS anatomical_location")

    # -- medications and fluids -------------------------------------------------------------------------
    def _meds(self) -> str:
        if self.con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'med'").fetchone()[0]:
            return "med"
        names = "SELECT lower(ReferenceValue) AS txt FROM ref WHERE ReferenceName IN ('Drug', 'Fluid', 'Food')"
        drug = self.regex_lookup("drugs", names, "drugs", ("ingredient", "continuous", "intermittent"),
                                 table=self.lexicon("drugs"))
        fluid = self.regex_lookup("fluids", names, "fluids", ("category", "group_name"))
        self.con.execute(f"""
            CREATE OR REPLACE TABLE med AS
            SELECT x.CaseID AS cid, x.Offset AS off, x.OffsetDrugEnd AS off_end, x.IsSingleDose = 1 AS single,
                   x.Amount AS amount, x.AmountPerMinute AS apm, r.ReferenceValue AS name, r.ReferenceName AS kind,
                   r.ReferenceUnit AS unit, d.ingredient,
                   -- a fluid vehicle (diluent, propofol lipid) is never a medication of its own
                   CASE WHEN f.category = '__skip__' THEN NULL ELSE nullif(d.continuous, '') END AS continuous,
                   CASE WHEN f.category = '__skip__' THEN NULL ELSE nullif(d.intermittent, '') END AS intermittent,
                   CASE WHEN r.ReferenceName IN ('Fluid', 'Food') THEN f.category END AS fluid_category,
                   CASE WHEN r.ReferenceName IN ('Fluid', 'Food') THEN f.group_name END AS fluid_group
            FROM {self.src('medication.parquet')} x
            JOIN ref r ON r.ReferenceGlobalID = x.DrugID
            LEFT JOIN {drug} d ON d.txt = lower(r.ReferenceValue)
            LEFT JOIN {fluid} f ON f.txt = lower(r.ReferenceValue)""")
        self.coverage("medication", "drug_name", """
            SELECT name AS txt, count(*) AS n,
                   bool_or(ingredient IS NOT NULL OR fluid_category IS NOT NULL OR kind = 'Food') AS mapped
            FROM med GROUP BY 1""")
        return "med"

    def sql_medication_admin_continuous(self):
        med = self._meds()
        unit, rate = _unit_case(RATE_UNIT, "unit", "apm")
        sql = f"""
            WITH p AS (
                SELECT cid, continuous, {unit} AS unit, {rate} AS rate, off, off_end, name
                FROM {med} WHERE NOT single AND continuous IS NOT NULL AND apm > 0 AND off_end > off),
            deltas AS (   -- rate steps of all periods; overlapping periods of one drug add up
                SELECT cid, continuous, unit, t, min(name) AS name, sum(d) AS delta
                FROM (SELECT cid, continuous, unit, off AS t, rate AS d, name FROM p
                      UNION ALL SELECT cid, continuous, unit, off_end, -rate, name FROM p)
                WHERE unit IS NOT NULL GROUP BY 1, 2, 3, 4),
            total AS (SELECT *, round(sum(delta) OVER (PARTITION BY cid, continuous, unit ORDER BY t
                                                       ROWS UNBOUNDED PRECEDING), 9) AS total FROM deltas),
            chg AS (SELECT *, lag(total, 1, 0) OVER (PARTITION BY cid, continuous, unit ORDER BY t) AS prev FROM total)
            SELECT CAST(c.cid AS VARCHAR) AS hospitalization_id, {t('c.t')} AS admin_dttm, c.name AS med_name,
                   c.continuous AS med_category, {self._group('medication_admin_continuous', 'c.continuous')} AS med_group,
                   'iv' AS med_route_category, greatest(c.total, 0) AS med_dose, c.unit AS med_dose_unit,
                   CASE WHEN c.total <= 0 THEN 'stop' WHEN c.prev <= 0 THEN 'start' ELSE 'dose_change' END AS mar_action_category,
                   'administered' AS mar_action_group
            FROM chg c JOIN stays s ON s.cid = c.cid WHERE c.total IS DISTINCT FROM c.prev"""
        return sql, ("hospitalization_id", "admin_dttm", "med_category")

    def sql_medication_admin_intermittent(self):
        med = self._meds()
        unit, dose = _unit_case(DOSE_UNIT, "m.unit", "m.amount")
        sql = f"""
            SELECT CAST(s.cid AS VARCHAR) AS hospitalization_id, {t('m.off')} AS admin_dttm, m.name AS med_name,
                   m.intermittent AS med_category, {self._group('medication_admin_intermittent', 'm.intermittent')} AS med_group,
                   {dose} AS med_dose, {unit} AS med_dose_unit,
                   'given' AS mar_action_category, 'administered' AS mar_action_group
            FROM {med} m JOIN stays s ON s.cid = m.cid
            WHERE m.single AND m.intermittent IS NOT NULL AND m.amount > 0 AND {unit} IS NOT NULL"""
        return sql, ("hospitalization_id", "admin_dttm")

    def sql_input(self):
        med = self._meds()
        sql = f"""
            SELECT CAST(s.cid AS VARCHAR) AS hospitalization_id,
                   {t('CASE WHEN m.single THEN m.off ELSE m.off_end END')} AS recorded_dttm,   -- volume given by then
                   m.name AS input_name, m.fluid_category AS input_category, m.fluid_group AS input_group,
                   m.amount AS input_volume
            FROM {med} m JOIN stays s ON s.cid = m.cid
            WHERE m.fluid_category IS NOT NULL AND m.fluid_category <> '__skip__' AND m.unit = 'ml' AND m.amount > 0
              AND NOT (m.single AND m.intermittent IS NOT NULL) AND NOT (NOT m.single AND m.continuous IS NOT NULL)"""
        return sql, ("hospitalization_id", "recorded_dttm")

    # -- diagnoses and stay context -------------------------------------------------------------------
    def sql_hospital_diagnosis(self):
        sql = """
            SELECT CAST(cid AS VARCHAR) AS hospitalization_id, replace(trim(ICD10Main), '.', '') AS diagnosis_code,
                   'ICD10CM' AS diagnosis_code_format, 1 AS diagnosis_primary
            FROM stays WHERE nullif(trim(ICD10Main), '') IS NOT NULL"""
        return sql, ("hospitalization_id",)

    def sql_ext_icu_stay(self):
        """Admission diagnosis group from the admission form (``ICD10Main`` is the case's main diagnosis, a
        discharge-time code): operated (surgery type, else a recorded surgery site or cardiac-surgery times) -> the
        site's system (``maps/sicdb/surgical_sites.csv``; cardiac surgery times: cardiovascular); not operated ->
        sepsis when the form records it, else the referring department's system
        (``maps/sicdb/referring_unit_systems.csv``)."""
        ru = self.map("referring_units")
        ss, us = self.map("surgical_sites"), self.map("referring_unit_systems")
        site, kind, unit = self._name("s.SurgicalSite"), self._name("s.SurgicalAdmissionType"), self._name("s.ReferringUnit")
        for m, col, expr in ((ss, "site", site), (us, "referring_unit", unit)):
            self.coverage("ext_icu_stay", m.removeprefix("map_"), f"""
                SELECT {expr} AS txt, count(*) AS n, bool_or(m.{col} IS NOT NULL) AS mapped
                FROM stays s LEFT JOIN {m} m ON m.{col} = {expr} GROUP BY 1""")
        heart = "(s.HeartSurgeryBeginOffset IS NOT NULL OR s.HeartSurgeryCPBTime IS NOT NULL)"   # cardiac surgery times
        surgical = f"""(CASE WHEN {kind} IN ('Elective Surgery', 'Urgent Surgery') THEN true WHEN {kind} = 'No Surgery' THEN false
                             WHEN {site} NOT IN ('Unknown', '.') OR {heart} THEN true END)"""
        system = f"""(CASE WHEN {kind} = 'No Surgery' AND {self._name('s.AdmissionFormHasSepsis')} = 'Yes' THEN 'sepsis'
                           WHEN {kind} = 'No Surgery' THEN nullif(us.system, '')
                           ELSE coalesce(nullif(st.system, ''), CASE WHEN {heart} THEN 'cardiovascular' END) END)"""
        sql = f"""
            SELECT 'sicdb:' || CAST(s.cid AS VARCHAR) AS stay_id, CAST(s.pid AS VARCHAR) AS patient_id,
                   CAST(s.cid AS VARCHAR) AS hospitalization_id, 'sicdb' AS dataset, 'sicdb' AS site_id,
                   s.icu_in AS icu_in_dttm, s.icu_out AS icu_out_dttm, 'general_icu' AS unit_type,
                   false AS clock_valid, true AS demographics_binned,
                   coalesce(u.admission_source,
                            CASE WHEN {self._name('s.SurgicalAdmissionType')} IN ('Elective Surgery', 'Urgent Surgery')
                                 THEN 'or' ELSE 'ward' END) AS admission_source,
                   CASE {self._name('s.SurgicalAdmissionType')} WHEN 'Elective Surgery' THEN 'elective_surgery'
                        WHEN 'Urgent Surgery' THEN 'emergency_surgery' WHEN 'No Surgery' THEN 'nonsurgical'
                        ELSE 'unknown' END AS surgical_status,
                   CASE WHEN {self._name('s.SurgicalSite')} NOT IN ('Unknown', '.') OR s.HeartSurgeryCPBTime IS NOT NULL
                        THEN to_json(struct_pack(site := {self._name('s.SurgicalSite')},
                                                 cpb_minutes := s.HeartSurgeryCPBTime,
                                                 cross_clamp_minutes := s.HeartSurgeryCrossClampTime))::VARCHAR END
                       AS surgery_context,
                   s.ICD10MainText AS adm_dx_name,
                   {group_sql(surgical, system)} AS adm_dx_group,
                   CAST(row_number() OVER (PARTITION BY s.pid ORDER BY s.case_start, s.cid) AS INTEGER) AS stay_seq,
                   CASE {self._name('s.DischargeState')} WHEN 'verstorben' THEN 'dead' WHEN 'lebend' THEN 'alive'
                        ELSE CASE WHEN s.death IS NOT NULL AND s.death <= s.icu_out THEN 'dead' ELSE 'unknown' END
                   END AS icu_outcome,
                   CASE WHEN s.saps3 IS NOT NULL THEN 'saps3' END AS severity_name, CAST(s.saps3 AS DOUBLE) AS severity_value,
                   CAST(TRY_CAST(s.AdmissionYear AS INTEGER) AS VARCHAR) AS era
            FROM stays s LEFT JOIN {ru} u ON u.referring_unit = {self._name('s.ReferringUnit')}
            LEFT JOIN {ss} st ON st.site = {site} LEFT JOIN {us} us ON us.referring_unit = {unit}"""
        return sql, ("stay_id",)
