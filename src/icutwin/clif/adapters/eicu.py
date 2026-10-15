"""eICU-CRD 2.0 -> CLIF (our adapter; no published CLIF ETL exists for eICU).

Identity and time follow the community eICU_MEDS ETL:
* hospital discharge pseudotime = 31 Dec of ``hospitaldischargeyear`` + ``hospitaldischargetime24``; every
  offset is minutes from its own unit admission, and unit admission = hospital discharge -
  ``hospitaldischargeoffset``. Time of day is real (``clock_valid``), calendar dates are not. eICU cannot
  order health-system stays of one patient within a year, so their pseudo-timelines may overlap:
  downstream stay windowing must key events by hospitalization, never by patient time alone;
* birth date = unit admission - (age x 365.25 - 182.625) days, with "> 89" read as 90;
* ``patient_id`` = ``uniquepid`` ``AAA-B...`` encoded as the integer AAA x 10^8 + B (CLIF_MEDS needs integer
  subject ids); the encoding is checked to be injective.

Source semantics (each mapping row in ``clif/maps/eicu/`` names its provenance):
* vitals combine the monitor streams (vitalPeriodic 5-min medians, vitalAperiodic) and nurse-validated
  charting (nurseCharting); S2b's hourly rule merges them per clock hour and blood-pressure site;
* infusionDrug: the rate unit is the trailing "(...)" of the drug name (ricu ``eicu_extract_unit``); rows
  without a rate unit are dropped, as in ricu. Rates stay in their charted unit (volume rates cannot be
  converted without a concentration, which eICU records only on mass-rate rows). eICU charts no end of an
  infusion: a running infusion not charted again for > 6 h is stopped 60 min after its last chart (ricu
  ``expand_intervals``: max_len 6 h, overhang 1 h), flagged in ``mar_action_name``;
* respiratory support merges charted settings/observations (respiratoryCharting, ventilator settings
  recorded with blood gases), charted O2 devices (respiratoryCharting, nurseCharting), and documented
  ventilation states (treatment, care plan); at one time point the charted device wins over documented
  states. Nurse-charted "O2 L/%" is a flow when <= 15 and an FiO2 when 21-100 (dropped when ambiguous);
* the medication table holds orders, not administrations: it is left for ``medication_orders`` (shared
  drug lexicon, M2 follow-up); microLab likewise waits for the shared organism lexicon.
"""

from __future__ import annotations

from icutwin.clif.adapters.base import ClifAdapter, minutes_after
from icutwin.clif.conform import extensions, table_schema
from icutwin.clif.dx_groups import group_sql

PID_BASE = 100_000_000
STOP_GAP_MIN = 360       # ricu expand_intervals max_len
STOP_OVERHANG_MIN = 60   # ricu expand_intervals overhang
DX_WITHIN_MIN = 1440     # header fields: APACHE admission documentation entered within 24 h of unit admission
RATE_UNIT_RE = r"(mcg|mg|units|ml|nanograms)(/kg)?/(min|hr)"
INFUSION_BASE = r"lower(trim(regexp_replace({}, '\s*\([^()]*\)\s*$', '')))"   # infusion name without its '(unit)'
INFUSION_UNIT = r"lower(trim(regexp_extract({}, '\(([^()]*)\)\s*$', 1)))"      # the trailing '(unit)'
IFACE_UNIT = "lower(trim(coalesce(x.labmeasurenameinterface, '')))"   # the unit labresult is in (lab.parquet x)
FLOW_UNITS = ("lpm", "l/min")   # a lab 'FiO2' charted as an oxygen flow (L/min), not a fraction
RESP_COLS = ("fio2_set", "lpm_set", "tidal_volume_set", "resp_rate_set", "pressure_control_set",
             "pressure_support_set", "flow_rate_set", "peep_set", "tidal_volume_obs", "resp_rate_obs",
             "plateau_pressure_obs", "peak_inspiratory_pressure_obs", "minute_vent_obs", "mean_airway_pressure_obs")
DEVICE_LABELS = ("O2 Device", "Oxygen Delivery Method", "RETIRED O2 Device", "Non-invasive Ventilation Mode",
                 "Bipap Delivery Mode")
MODE_LABELS = ("Mechanical Ventilator Mode", "Ventilator Support Mode")


def t(offset_sql: str, alias: str = "s") -> str:
    """Pseudotime of an offset (minutes from the unit admission of stay ``alias``)."""
    return minutes_after(f"{alias}.unit_admit", offset_sql)


def _in(values) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


class EicuAdapter(ClifAdapter):
    dataset = "eicu"
    tables = (
        "patient", "hospitalization", "adt", "vitals", "labs", "respiratory_support", "patient_assessments",
        "medication_admin_continuous", "medication_orders", "crrt_therapy", "intermittent_dialysis", "mcs",
        "patient_diagnosis", "microbiology_culture", "microbiology_susceptibility",
        "input", "output", "invasive_hemodynamics", "code_status", "ext_icu_stay",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p = self.src("patient.parquet")
        bad = self.con.execute(f"""
            SELECT count(*) FILTER (WHERE uniquepid IS NULL OR NOT regexp_full_match(uniquepid, '[0-9]{{3}}-[0-9]{{1,8}}')),
                   count(DISTINCT uniquepid)
            FROM {p}""").fetchone()
        if bad[0]:
            raise ValueError(f"eicu: {bad[0]} uniquepid values outside the AAA-B format; patient_id encoding undefined")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE stays AS
            WITH b AS (
                SELECT *, CAST(make_date(CAST(hospitaldischargeyear AS INTEGER), 12, 31)
                               + CAST(hospitaldischargetime24 AS TIME) AS TIMESTAMPTZ) AS hosp_disch,
                       CASE WHEN trim(age) = '> 89' THEN 90 ELSE TRY_CAST(age AS INTEGER) END AS age_years
                FROM {p}),
            a AS (SELECT *, {minutes_after('hosp_disch', '-hospitaldischargeoffset')} AS unit_admit FROM b)
            SELECT patientunitstayid AS stay, patienthealthsystemstayid AS hosp,
                   CAST(split_part(uniquepid, '-', 1) AS BIGINT) * {PID_BASE}
                       + CAST(split_part(uniquepid, '-', 2) AS BIGINT) AS pid,
                   unit_admit, hosp_disch,
                   {minutes_after('unit_admit', 'unitdischargeoffset')} AS unit_disch,
                   {minutes_after('unit_admit', 'hospitaladmitoffset')} AS hosp_admit,
                   CAST({minutes_after('unit_admit', '-(age_years * 365.25 - 182.625) * 1440')} AS DATE) AS birth,
                   a.* EXCLUDE (patientunitstayid, patienthealthsystemstayid, unit_admit, hosp_disch)
            FROM a""")
        n_pid = self.con.execute("SELECT count(DISTINCT pid) FROM stays").fetchone()[0]
        if n_pid != bad[1]:
            raise ValueError(f"eicu: patient_id encoding is not injective ({bad[1]} uniquepid -> {n_pid} ids)")

    # -- helpers ------------------------------------------------------------------------------------
    def _join_stay(self, table: str, alias: str = "x") -> str:
        return f"FROM {self.src(table)} {alias} JOIN stays s ON s.stay = {alias}.patientunitstayid"

    def _monitor_long(self, table: str, items: list[str]) -> str:
        """(stay, off, item, value) from a wide monitor table (vitalPeriodic / vitalAperiodic)."""
        if not items:
            return "SELECT NULL::BIGINT AS stay, NULL::BIGINT AS off, NULL AS item, NULL::DOUBLE AS value WHERE false"
        cols = ", ".join(f'CAST("{c}" AS DOUBLE) AS "{c}"' for c in items)
        return f"""
            SELECT patientunitstayid AS stay, observationoffset AS off, item, value FROM (
                UNPIVOT (SELECT patientunitstayid, observationoffset, {cols} FROM {self.src(table)})
                ON {', '.join(f'"{c}"' for c in items)} INTO NAME item VALUE value)"""

    def _nurse_long(self, keys: list[str]) -> str:
        """(stay, off, item, value, text) from nurseCharting for ``label|name`` keys."""
        labels = sorted({k.split("|", 1)[0] for k in keys}) or ["__none__"]
        return f"""
            SELECT patientunitstayid AS stay, nursingchartoffset AS off,
                   nursingchartcelltypevallabel || '|' || nursingchartcelltypevalname AS item,
                   TRY_CAST(nursingchartvalue AS DOUBLE) AS value, nursingchartvalue AS text
            FROM {self.src('nurseCharting.parquet')}
            WHERE nursingchartcelltypevallabel IN ({_in(labels)})
              AND nursingchartcelltypevallabel || '|' || nursingchartcelltypevalname IN ({_in(keys) or "''"})"""

    def _items(self, map_name: str, source: str, col: str = "item") -> list[str]:
        m = self.map(map_name)
        return [r[0] for r in self.con.execute(f"SELECT {col} FROM {m} WHERE source_table = ?", [source]).fetchall()]

    def _med_group(self, table: str) -> str:
        mapping = dict(table_schema(table).get("med_category_to_group_mapping") or {})
        for cat, spec in ((extensions().get(table) or {}).get("med_category") or {}).items():
            if isinstance(spec, dict) and spec.get("group"):
                mapping.setdefault(cat, spec["group"])
        cases = " ".join(f"WHEN '{k}' THEN '{v if isinstance(v, str) else v[0]}'" for k, v in mapping.items())
        return f"CASE med_category {cases} END"

    # -- core tables --------------------------------------------------------------------------------
    def sql_patient(self):
        m = self.map("ethnicity")
        sql = f"""
            WITH p AS (
                SELECT pid, arg_min(birth, unit_admit) AS birth_date,
                       max(hosp_disch) FILTER (WHERE hospitaldischargestatus = 'Expired') AS death_dttm,
                       arg_min(ethnicity, unit_admit) AS ethnicity, arg_min(gender, unit_admit) AS gender
                FROM stays GROUP BY pid)
            SELECT CAST(pid AS VARCHAR) AS patient_id, birth_date, death_dttm,
                   p.ethnicity AS race_name, coalesce(m.race_category, 'unknown') AS race_category,
                   p.ethnicity AS ethnicity_name, coalesce(m.ethnicity_category, 'unknown') AS ethnicity_category,
                   gender AS sex_name,
                   CASE gender WHEN 'Male' THEN 'male' WHEN 'Female' THEN 'female' ELSE 'unknown' END AS sex_category,
                   'unknown' AS language_category
            FROM p LEFT JOIN {m} m ON m.ethnicity = p.ethnicity"""
        return sql, ("patient_id",)

    def sql_hospitalization(self):
        src = self.map("admit_sources")
        dis = self.map("discharge")
        sql = f"""
            WITH h AS (
                SELECT hosp, any_value(pid) AS pid, min(hosp_admit) AS admission_dttm, any_value(hosp_disch) AS discharge_dttm,
                       arg_min(age_years, unit_admit) AS age, arg_min(hospitaladmitsource, unit_admit) AS adm_src,
                       any_value(hospitaldischargelocation) AS dis_loc,
                       bool_or(hospitaldischargestatus = 'Expired') AS died
                FROM stays GROUP BY hosp)
            SELECT CAST(h.pid AS VARCHAR) AS patient_id, CAST(h.hosp AS VARCHAR) AS hospitalization_id,
                   admission_dttm, discharge_dttm, age AS age_at_admission,
                   adm_src AS admission_type_name, a.hospital_admission_type AS admission_type_category,
                   dis_loc AS discharge_name,
                   CASE WHEN died THEN 'expired' WHEN dis_loc IS NULL THEN 'missing'
                        ELSE coalesce(d.discharge_category, 'other') END AS discharge_category
            FROM h LEFT JOIN {src} a ON a.source = h.adm_src
            LEFT JOIN {dis} d ON d.hospitaldischargelocation = h.dis_loc"""
        return sql, ("hospitalization_id",)

    def sql_adt(self):
        u = self.map("unit_types")
        sql = f"""
            SELECT CAST(s.hosp AS VARCHAR) AS hospitalization_id, 'eicu_' || CAST(s.hospitalid AS VARCHAR) AS hospital_id,
                   CASE h.teachingstatus WHEN true THEN 'academic' WHEN false THEN 'community' END AS hospital_type,
                   s.unit_admit AS in_dttm, s.unit_disch AS out_dttm, s.unittype AS location_name,
                   'icu' AS location_category, u.location_type
            FROM stays s LEFT JOIN {self.src('hospital.parquet')} h ON h.hospitalid = s.hospitalid
            LEFT JOIN {u} u ON u.unittype = s.unittype"""
        return sql, ("hospitalization_id", "in_dttm")

    def sql_vitals(self):
        m = self.map("vitals")
        long = " UNION ALL ".join([
            self._monitor_long("vitalPeriodic.parquet", self._items("vitals", "vitalPeriodic")),
            self._monitor_long("vitalAperiodic.parquet", self._items("vitals", "vitalAperiodic")),
            f"SELECT stay, off, item, value FROM ({self._nurse_long(self._items('vitals', 'nurseCharting'))})",
        ])
        io = self.src("intakeOutput.parquet")
        value = ("CASE WHEN m.transform = 'temp_c_or_f' AND v.value BETWEEN 80 AND 113 "
                 "THEN (v.value - 32) / 1.8 ELSE v.value END")
        sql = f"""
            SELECT CAST(s.hosp AS VARCHAR) AS hospitalization_id, {t('v.off')} AS recorded_dttm,
                   v.item AS vital_name, m.vital_category, {value} AS vital_value,
                   nullif(m.meas_site_name, '') AS meas_site_name
            FROM ({long}) v JOIN {m} m ON m.item = v.item JOIN stays s ON s.stay = v.stay
            WHERE v.value IS NOT NULL
            UNION ALL
            SELECT CAST(hosp AS VARCHAR), unit_admit, 'admissionheight', 'height_cm', admissionheight, NULL
            FROM stays WHERE admissionheight IS NOT NULL
            UNION ALL
            SELECT CAST(hosp AS VARCHAR), unit_admit, 'admissionweight', 'weight_kg', admissionweight, NULL
            FROM stays WHERE admissionweight IS NOT NULL
            UNION ALL
            SELECT CAST(hosp AS VARCHAR), unit_disch, 'dischargeweight', 'weight_kg', dischargeweight, NULL
            FROM stays WHERE dischargeweight IS NOT NULL
            UNION ALL
            SELECT CAST(s.hosp AS VARCHAR), {t('x.intakeoutputoffset')}, x.celllabel, 'weight_kg', x.cellvaluenumeric, NULL
            FROM {io} x JOIN stays s ON s.stay = x.patientunitstayid
            WHERE x.cellpath = 'flowsheet|Flowsheet Cell Labels|I&O|Weight|Bodyweight (kg)' AND x.cellvaluenumeric > 0"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_labs(self):
        """``labresult`` is in the unit the hospital's interface sends (``labmeasurenameinterface``); the system unit
        is a label per lab name and is not converted to. Map rows match any unit ('*'), the system unit, or, as
        ``iface:<unit>``, the interface unit (lower-cased): for the labs whose hospitals report different scales
        (ammonia, CRP, differentials with absolute counts), where a unit that mixes scales across hospitals stays
        unmapped (counted in the coverage report)."""
        m = self.map("labs")
        lab = self.src("lab.parquet")
        join = (f"{m} m ON m.labname = x.labname AND (m.unit_in = '*' "
                f"OR (starts_with(m.unit_in, 'iface:') AND substr(m.unit_in, 7) = {IFACE_UNIT}) "
                f"OR (NOT starts_with(m.unit_in, 'iface:') AND lower(m.unit_in) = lower(trim(x.labmeasurenamesystem))))")
        self.coverage("labs", "labname_unit", f"""
            SELECT x.labname || ' [' || coalesce(x.labmeasurenamesystem, 'NULL') || ' | ' || {IFACE_UNIT} || ']' AS txt,
                   count(*) AS n, bool_or(m.lab_category IS NOT NULL) AS mapped
            FROM {lab} x LEFT JOIN {join}
            WHERE x.labname IN (SELECT labname FROM {m}) GROUP BY 1""")
        sql = f"""
            SELECT CAST(s.hosp AS VARCHAR) AS hospitalization_id,
                   {t('x.labresultoffset')} AS lab_order_dttm, {t('x.labresultoffset')} AS lab_collect_dttm,
                   {t('x.labresultoffset')} AS lab_result_dttm,
                   m.lab_order_category, x.labname AS lab_name, m.lab_category,
                   coalesce(x.labresulttext, CAST(x.labresult AS VARCHAR)) AS lab_value,
                   CAST(x.labresult AS DOUBLE) * CAST(m.factor AS DOUBLE) AS lab_value_numeric,
                   m.reference_unit, m.lab_specimen_category
            FROM {lab} x JOIN stays s ON s.stay = x.patientunitstayid JOIN {join}
            WHERE x.labresult IS NOT NULL"""
        return sql, ("hospitalization_id", "lab_result_dttm")

    # -- respiratory support --------------------------------------------------------------------------
    def sql_respiratory_support(self):
        num = self.map("resp_numeric")
        rc = self.src("respiratoryCharting.parquet")
        nc = self.src("nurseCharting.parquet")
        dev_texts = f"""
            SELECT lower(trim(respchartvalue)) AS txt FROM {rc} WHERE respchartvaluelabel IN ({_in(DEVICE_LABELS)})
            UNION ALL
            SELECT lower(trim(nursingchartvalue)) FROM {nc} WHERE nursingchartcelltypevallabel = 'O2 Admin Device'"""
        dev = self.regex_lookup("o2_devices", dev_texts, "o2_devices", ("device_category",))
        modes = self.regex_lookup("vent_modes", f"""
            SELECT lower(trim(respchartvalue)) AS txt FROM {rc} WHERE respchartvaluelabel IN ({_in(MODE_LABELS)})""",
            "vent_modes", ("mode_category",))
        trt = self._treatments()
        cp = self._care_plan()
        self.coverage("respiratory_support", "device_text", f"""
            SELECT d.txt, count(*) AS n, bool_or(l.device_category IS NOT NULL) AS mapped
            FROM ({dev_texts}) d LEFT JOIN {dev} l USING (txt) GROUP BY 1""")
        fio2 = "CASE WHEN v > 1 THEN v / 100.0 ELSE v END"
        cols = ", ".join(f"max(v) FILTER (WHERE col = '{c}') AS {c}" for c in RESP_COLS)
        sql = f"""
            WITH num AS (
                SELECT x.patientunitstayid AS stay, x.respchartoffset AS off, m.clif_column AS col, m.transform,
                       TRY_CAST(replace(x.respchartvalue, '%', '') AS DOUBLE) AS v
                FROM {rc} x JOIN {num} m ON m.source_table = 'respiratoryCharting' AND m.label = x.respchartvaluelabel
                UNION ALL
                SELECT x.patientunitstayid, x.labresultoffset, m.clif_column, m.transform, CAST(x.labresult AS DOUBLE)
                FROM {self.src('lab.parquet')} x JOIN {num} m ON m.source_table = 'lab' AND m.label = x.labname
                WHERE NOT (m.transform = 'fio2' AND {IFACE_UNIT} IN ({_in(FLOW_UNITS)}))),
            devices AS (   -- charted devices (priority 1)
                SELECT x.patientunitstayid AS stay, x.respchartoffset AS off, x.respchartvalue AS device_name,
                       l.device_category, 1 AS prio
                FROM {rc} x JOIN {dev} l ON l.txt = lower(trim(x.respchartvalue))
                WHERE x.respchartvaluelabel IN ({_in(DEVICE_LABELS)})
                UNION ALL
                SELECT x.patientunitstayid, x.nursingchartoffset, x.nursingchartvalue, l.device_category, 1
                FROM {nc} x JOIN {dev} l ON l.txt = lower(trim(x.nursingchartvalue))
                WHERE x.nursingchartcelltypevallabel = 'O2 Admin Device'
                UNION ALL  -- documented states: active treatments (2), care-plan ventilation status (3)
                SELECT stay, off, 'treatment: ' || name, category, 2 FROM {trt} WHERE target = 'respiratory_support'
                UNION ALL
                SELECT stay, off, 'care plan: ' || name, category, 3 FROM {cp} WHERE target = 'respiratory_support'),
            dev_at AS (
                SELECT stay, off, arg_min(device_category, prio) AS device_category,
                       coalesce(arg_min(device_name, prio) FILTER (WHERE device_category IS NOT NULL),
                                arg_min(device_name, prio)) AS device_name
                FROM devices GROUP BY stay, off),
            o2 AS (   -- nurse-charted "O2 L/%": flow (<= 15) or FiO2 (21-100); ambiguous on high-flow devices
                SELECT x.patientunitstayid AS stay, x.nursingchartoffset AS off, TRY_CAST(x.nursingchartvalue AS DOUBLE) AS v,
                       d.device_category AS dev
                FROM {nc} x LEFT JOIN dev_at d ON d.stay = x.patientunitstayid AND d.off = x.nursingchartoffset
                WHERE x.nursingchartcelltypevallabel = 'O2 L/%'),
            vals AS (
                SELECT stay, off, col, CASE WHEN transform = 'fio2' THEN {fio2} ELSE v END AS v FROM num WHERE v IS NOT NULL
                UNION ALL
                SELECT stay, off, CASE WHEN v <= 15 THEN 'lpm_set' ELSE 'fio2_set' END,
                       CASE WHEN v <= 15 THEN v ELSE v / 100.0 END
                FROM o2 WHERE v >= 0 AND (v <= 15 OR (v >= 21 AND v <= 100 AND dev IS DISTINCT FROM 'hfnc'))),
            wide AS (SELECT stay, off, {cols} FROM vals GROUP BY stay, off),
            mode_at AS (
                SELECT x.patientunitstayid AS stay, x.respchartoffset AS off,
                       arg_min(l.mode_category, x.respchartid) AS mode_category,
                       coalesce(arg_min(x.respchartvalue, x.respchartid) FILTER (WHERE l.mode_category IS NOT NULL),
                                arg_min(x.respchartvalue, x.respchartid)) AS mode_name
                FROM {rc} x JOIN {modes} l ON l.txt = lower(trim(x.respchartvalue))
                WHERE x.respchartvaluelabel IN ({_in(MODE_LABELS)}) GROUP BY 1, 2),
            keys AS (SELECT stay, off FROM wide UNION SELECT stay, off FROM dev_at UNION SELECT stay, off FROM mode_at)
            SELECT CAST(s.hosp AS VARCHAR) AS hospitalization_id, {t('k.off')} AS recorded_dttm,
                   d.device_name, d.device_category, md.mode_name, md.mode_category,
                   {', '.join(f'w.{c}' for c in RESP_COLS)}
            FROM keys k JOIN stays s ON s.stay = k.stay
            LEFT JOIN wide w ON w.stay = k.stay AND w.off = k.off
            LEFT JOIN dev_at d ON d.stay = k.stay AND d.off = k.off
            LEFT JOIN mode_at md ON md.stay = k.stay AND md.off = k.off
            WHERE d.device_category IS NOT NULL OR d.device_name IS NOT NULL OR md.mode_category IS NOT NULL
               OR {' OR '.join(f'w.{c} IS NOT NULL' for c in RESP_COLS)}"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def _treatments(self) -> str:
        """Materialise mapped treatment rows: lk_treatments + table ``trt(stay, off, name, target, category)``."""
        tr = self.src("treatment.parquet")
        lk = self.regex_lookup("treatments", f"SELECT lower(treatmentstring) AS txt FROM {tr}", "treatments",
                               ("target", "category"))
        self.con.execute(f"""
            CREATE OR REPLACE TABLE trt AS
            SELECT x.patientunitstayid AS stay, x.treatmentoffset AS off, x.treatmentstring AS name,
                   l.target, nullif(l.category, '') AS category
            FROM {tr} x JOIN {lk} l ON l.txt = lower(x.treatmentstring)
            WHERE l.target IS NOT NULL AND l.target <> '__skip__'""")
        return "trt"

    def _care_plan(self) -> str:
        """Materialise mapped care-plan rows: table ``cp(stay, off, name, target, category)``."""
        m = self.map("care_plan")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE cp AS
            SELECT x.patientunitstayid AS stay, x.cplitemoffset AS off, x.cplitemvalue AS name, m.target, m.category
            FROM {self.src('carePlanGeneral.parquet')} x
            JOIN {m} m ON m.cplgroup = x.cplgroup AND regexp_matches(lower(trim(x.cplitemvalue)), m.pattern)""")
        return "cp"

    # -- assessments --------------------------------------------------------------------------------
    def sql_patient_assessments(self):
        m = self.map("assessments")
        keys = [r[0] for r in self.con.execute(f"SELECT item FROM {m}").fetchall()]
        nc = self.src("nurseCharting.parquet")
        group = table_schema("patient_assessments")["assessment_category_to_group_mapping"]
        grp = "CASE assessment_category " + " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in group.items()) + " END"
        pair = """
            SELECT patientunitstayid AS stay, nursingchartoffset AS off,
                   max(nursingchartvalue) FILTER (WHERE nursingchartcelltypevalname = '{scale}') AS scale,
                   max(nursingchartvalue) FILTER (WHERE nursingchartcelltypevalname = '{score}') AS score
            FROM {src} WHERE nursingchartcelltypevalname IN ('{scale}', '{score}') GROUP BY 1, 2"""
        sql = f"""
            WITH a AS (
                SELECT v.stay, v.off, v.item AS assessment_name, m.assessment_category, v.value AS numerical_value,
                       NULL AS categorical_value
                FROM ({self._nurse_long(keys)}) v JOIN {m} m ON m.item = v.item WHERE v.value IS NOT NULL
                UNION ALL   -- 'Sedation Score' is meaningful only with its paired 'Sedation Scale'
                SELECT stay, off, 'Sedation Score (' || scale || ')', lower(scale), TRY_CAST(score AS DOUBLE), NULL
                FROM ({pair.format(scale='Sedation Scale', score='Sedation Score', src=nc)})
                WHERE scale IN ('RASS', 'SAS') AND TRY_CAST(score AS DOUBLE) IS NOT NULL
                UNION ALL   -- delirium screens: CAM-ICU / ICDSC result Yes/No
                SELECT stay, off, 'Delirium Score (' || scale || ')',
                       CASE scale WHEN 'CAM-ICU' THEN 'cam_total' ELSE 'icsdc' END, NULL,
                       CASE upper(trim(score)) WHEN 'YES' THEN 'Positive' ELSE 'Negative' END
                FROM ({pair.format(scale='Delirium Scale', score='Delirium Score', src=nc)})
                WHERE scale IN ('CAM-ICU', 'ICDSC') AND upper(trim(score)) IN ('YES', 'NO'))
            SELECT CAST(s.hosp AS VARCHAR) AS hospitalization_id, {t('a.off')} AS recorded_dttm,
                   a.assessment_name, a.assessment_category, {grp} AS assessment_group,
                   a.numerical_value, a.categorical_value
            FROM a JOIN stays s ON s.stay = a.stay"""
        return sql, ("hospitalization_id", "recorded_dttm")

    # -- medications ---------------------------------------------------------------------------------
    def sql_medication_orders(self):
        """``medication`` (orders) -> CLIF 3.0 ``medication_orders``; eICU records no intermittent doses given.

        * names resolve with the shared lexicon; orders without a usable name (about a third carry only the
          First Databank HICL ingredient code) resolve through ``maps/eicu/hicl.csv``, built from the orders that
          carry both (``icutwin.clif.tools.eicu_hicl``);
        * orders on routes without a CLIF route category (topical, ophthalmic, rectal, epidural, line locks;
          ``maps/eicu/routes.csv``) are not CLIF medications; the others carry their route category in the
          declared extension column ``med_route_category`` (``extension_columns.yaml``; NULL where unknown);
        * the category follows the order type: continuous orders (frequency 'continuous' or a rate dosage) take
          the drug's continuous category, others its intermittent one (each falls back to the other);
        * dosage text '<amount> <unit>' gives dose and unit; ranges and placeholders ('1-4', 'PYXIS') give none;
          a stop offset before the start offset is unknown.
        """
        med = self.src("medication.parquet")
        lex = self.lexicon("drugs")
        lk = self.regex_lookup("drugs", f"SELECT lower(trim(drugname)) AS txt FROM {med}", "drugs",
                               ("ingredient", "continuous", "intermittent"), table=lex)
        routes = self.regex_lookup("routes", f"SELECT lower(trim(routeadmin)) AS txt FROM {med}", "routes",
                                   ("route_class",))
        hicl = self.map("hicl")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE ing_cat AS   -- categories of an ingredient reached through its HICL code
            SELECT ingredient, max(nullif(continuous, '')) AS continuous, max(nullif(intermittent, '')) AS intermittent
            FROM {lex} GROUP BY ingredient""")
        dose_re = r"^([0-9][0-9,]*(\.[0-9]+)?)\s*([a-zµ%][a-z0-9µ%/.]*)\s*$"
        self.con.execute(f"""
            CREATE OR REPLACE TABLE ord AS
            SELECT m.*, nullif(trim(m.drugname), '') AS name,
                   coalesce(l.ingredient, h.ingredient) AS ingredient,
                   CASE WHEN l.ingredient IS NOT NULL THEN 'name' WHEN h.ingredient IS NOT NULL THEN 'hicl' END AS resolved_by,
                   CASE WHEN l.ingredient IS NOT NULL THEN nullif(l.continuous, '') ELSE c.continuous END AS cont_cat,
                   CASE WHEN l.ingredient IS NOT NULL THEN nullif(l.intermittent, '') ELSE c.intermittent END AS int_cat,
                   coalesce(r.route_class, 'unknown') AS route_class,
                   (lower(coalesce(m.frequency, '')) LIKE '%continuous%'
                    OR regexp_matches(lower(coalesce(m.dosage, '')), '/(hr|h|min)\\b')) AS is_continuous,
                   TRY_CAST(replace(regexp_extract(lower(trim(m.dosage)), '{dose_re}', 1), ',', '') AS DOUBLE) AS dose,
                   nullif(regexp_extract(lower(trim(m.dosage)), '{dose_re}', 3), '') AS dose_unit
            FROM {med} m
            LEFT JOIN {lk} l ON l.txt = lower(trim(m.drugname))
            LEFT JOIN {hicl} h ON l.ingredient IS NULL AND h.drughiclseqno = CAST(m.drughiclseqno AS VARCHAR)
            LEFT JOIN ing_cat c ON c.ingredient = h.ingredient
            LEFT JOIN {routes} r ON r.txt = lower(trim(m.routeadmin))""")
        self.coverage("medication_orders", "drug_name", """
            SELECT coalesce(lower(name), '(no name: HICL ' || CAST(drughiclseqno AS VARCHAR) || ')') AS txt,
                   count(*) AS n, bool_or(ingredient IS NOT NULL) AS mapped FROM ord GROUP BY 1""")
        self.coverage("medication_orders", "route", """
            SELECT coalesce(lower(trim(routeadmin)), '(none)') AS txt, count(*) AS n,
                   bool_or(route_class <> 'unknown') AS mapped FROM ord GROUP BY 1""")
        sql = f"""
            SELECT CAST(s.hosp AS VARCHAR) AS hospitalization_id, CAST(o.medicationid AS VARCHAR) AS med_order_id,
                   {t('o.drugorderoffset')} AS ordered_dttm, {t('o.drugstartoffset')} AS order_start_dttm,
                   CASE WHEN o.drugstopoffset >= o.drugstartoffset THEN {t('o.drugstopoffset')} END AS order_end_dttm,
                   o.name AS med_name,
                   CASE WHEN o.is_continuous THEN coalesce(o.cont_cat, o.int_cat) ELSE coalesce(o.int_cat, o.cont_cat) END
                       AS med_category,
                   CASE WHEN o.drugordercancelled THEN 'cancelled' ELSE 'ordered' END AS med_order_status_name,
                   CASE WHEN o.drugordercancelled THEN 'cancelled' ELSE 'ordered' END AS med_order_status_category,
                   o.routeadmin AS med_route_name,
                   CASE WHEN o.route_class IN ('iv', 'im', 'enteral', 'buccal_sublingual', 'intrapleural', 'inhaled')
                        THEN o.route_class END AS med_route_category,   -- declared extension column
                   o.dose AS med_dose,
                   CASE WHEN o.dose_unit IN ('gm', 'gram', 'grams') THEN 'g'
                        WHEN o.dose_unit IN ('unit', 'u', 'iu', 'units') THEN 'units'
                        WHEN o.dose_unit IN ('tablet', 'tablets', 'tabs') THEN 'tab'
                        WHEN o.dose_unit IN ('capsule', 'capsules', 'caps') THEN 'cap'
                        ELSE o.dose_unit END AS med_dose_unit,
                   o.frequency AS med_frequency, CAST(o.prn AS INTEGER) AS prn
            FROM ord o JOIN stays s ON s.stay = o.patientunitstayid
            WHERE o.route_class <> 'non_systemic' AND coalesce(o.cont_cat, o.int_cat) IS NOT NULL"""
        return sql, ("hospitalization_id", "order_start_dttm", "med_order_id")

    # -- microbiology (microLab: positive and 'no growth' culture results with susceptibilities) -------------
    def _micro(self) -> str:
        if self.con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'micro'").fetchone()[0]:
            return "micro"
        ml = self.src("microLab.parquet")
        specs = self.regex_lookup("micro_specs", f"SELECT lower(culturesite) AS txt FROM {ml}", "micro_specimens",
                                  ("fluid_category",), table=self.lexicon("micro_specimens"))
        orgs = self.regex_lookup("organisms", f"SELECT lower(organism) AS txt FROM {ml}", "organisms",
                                 ("organism_category", "organism_group"), table=self.lexicon("organisms"))
        # an order = one culture of one site at one time; an organism = one organism grown in it
        self.con.execute(f"""
            CREATE OR REPLACE TABLE micro AS
            SELECT DISTINCT x.patientunitstayid AS stay, x.culturetakenoffset AS off, x.culturesite, x.organism,
                   s.fluid_category, g.organism_category, g.organism_group,
                   CAST(x.patientunitstayid AS VARCHAR) || ':' || CAST(x.culturetakenoffset AS VARCHAR) || ':'
                       || coalesce(x.culturesite, '') AS order_id,
                   CAST(x.patientunitstayid AS VARCHAR) || ':' || CAST(x.culturetakenoffset AS VARCHAR) || ':'
                       || coalesce(x.culturesite, '') || ':' || coalesce(x.organism, '') AS organism_id
            FROM {ml} x
            LEFT JOIN {specs} s ON s.txt = lower(x.culturesite)
            LEFT JOIN {orgs} g ON g.txt = lower(x.organism)""")
        self.coverage("microbiology_culture", "organism", """
            SELECT coalesce(lower(organism), '(none)') AS txt, count(*) AS n, bool_or(organism_category IS NOT NULL) AS mapped
            FROM micro GROUP BY 1""")
        return "micro"

    def sql_microbiology_culture(self):
        mi = self._micro()
        sql = f"""
            SELECT CAST(s.pid AS VARCHAR) AS patient_id, CAST(s.hosp AS VARCHAR) AS hospitalization_id,
                   m.order_id, m.organism_id, {t('m.off')} AS order_dttm, {t('m.off')} AS collect_dttm,
                   {t('m.off')} AS result_dttm,   -- eICU records the culture time only
                   m.culturesite AS fluid_name, m.fluid_category, 'microLab culture' AS method_name,
                   'culture' AS method_category, m.organism AS organism_name, m.organism_category, m.organism_group
            FROM {mi} m JOIN stays s ON s.stay = m.stay
            WHERE m.organism_category IS NOT NULL AND m.organism_category <> '__skip__'"""
        return sql, ("hospitalization_id", "collect_dttm", "organism_id")

    def sql_microbiology_susceptibility(self):
        mi = self._micro()
        ml = self.src("microLab.parquet")
        ab = self.regex_lookup("antimicrobials", f"SELECT lower(antibiotic) AS txt FROM {ml}", "antimicrobials",
                               ("antimicrobial_category",), table=self.lexicon("antimicrobials"))
        self.coverage("microbiology_susceptibility", "antibiotic", f"""
            SELECT lower(x.antibiotic) AS txt, count(*) AS n, bool_or(a.antimicrobial_category IS NOT NULL) AS mapped
            FROM {ml} x LEFT JOIN {ab} a ON a.txt = lower(x.antibiotic) WHERE x.antibiotic IS NOT NULL GROUP BY 1""")
        sql = f"""
            SELECT m.organism_id, x.antibiotic AS antimicrobial_name, a.antimicrobial_category,
                   x.sensitivitylevel AS susceptibility_name,
                   CASE x.sensitivitylevel WHEN 'Sensitive' THEN 'susceptible' WHEN 'Resistant' THEN 'non_susceptible'
                        WHEN 'Intermediate' THEN 'non_susceptible' ELSE 'na' END AS susceptibility_category
            FROM {ml} x
            JOIN {mi} m ON m.stay = x.patientunitstayid AND m.off = x.culturetakenoffset
                       AND m.culturesite IS NOT DISTINCT FROM x.culturesite AND m.organism IS NOT DISTINCT FROM x.organism
            JOIN {ab} a ON a.txt = lower(x.antibiotic)
            WHERE x.antibiotic IS NOT NULL AND a.antimicrobial_category <> '__skip__'
              AND m.organism_category IS NOT NULL AND m.organism_category NOT IN ('__skip__', 'no_growth')"""
        return sql, ("organism_id", "antimicrobial_category")

    def sql_medication_admin_continuous(self):
        inf = self.src("infusionDrug.parquet")
        base, unit = INFUSION_BASE.format("drugname"), INFUSION_UNIT.format("drugname")
        lk = self.regex_lookup("infusions", f"SELECT {base} AS txt FROM {inf}", "infusions", ("med_category",))
        self.coverage("medication_admin_continuous", "drug_name", f"""
            SELECT {base} AS txt, count(*) AS n, bool_or(l.med_category IS NOT NULL) AS mapped
            FROM {inf} x LEFT JOIN {lk} l ON l.txt = {base} GROUP BY 1""")
        self.coverage("medication_admin_continuous", "rate_unit", f"""
            SELECT {unit} AS txt, count(*) AS n, bool_or(regexp_full_match({unit}, '{RATE_UNIT_RE}')) AS mapped
            FROM {inf} x JOIN {lk} l ON l.txt = {base} WHERE l.med_category <> '__skip__' GROUP BY 1""")
        sql = f"""
            WITH r AS (
                SELECT x.patientunitstayid AS stay, x.infusionoffset AS off, x.drugname, l.med_category,
                       replace({unit}, 'nanograms', 'ng') AS unit,
                       TRY_CAST(x.drugrate AS DOUBLE) AS rate, TRY_CAST(x.infusionrate AS DOUBLE) AS vol_rate
                FROM {inf} x JOIN {lk} l ON l.txt = {base}
                WHERE l.med_category <> '__skip__' AND regexp_full_match({unit}, '{RATE_UNIT_RE}')),
            g AS (   -- one rate per (stay, drug, unit, time)
                SELECT stay, med_category, unit, off, max(rate) AS rate, max(vol_rate) AS vol_rate,
                       arg_max(drugname, rate) AS med_name
                FROM r WHERE rate >= 0 GROUP BY ALL),
            w AS (
                SELECT *, lag(rate) OVER k AS prev_rate, lag(off) OVER k AS prev_off, lead(off) OVER k AS next_off
                FROM g WINDOW k AS (PARTITION BY stay, med_category, unit ORDER BY off)),
            e AS (SELECT *, (prev_off IS NULL OR off - prev_off > {STOP_GAP_MIN}) AS new_run FROM w),
            ev AS (
                SELECT stay, med_category, unit, off, rate, vol_rate, med_name, NULL AS mar_action_name,
                       CASE WHEN rate > 0 AND (new_run OR prev_rate = 0) THEN 'start'
                            WHEN rate > 0 AND rate <> prev_rate THEN 'dose_change'
                            WHEN rate = 0 AND NOT new_run AND prev_rate > 0 THEN 'stop' END AS mar_action_category
                FROM e
                UNION ALL   -- inferred end of a running infusion that is not charted again within 6 h
                SELECT e.stay, e.med_category, e.unit,
                       CASE WHEN s.unitdischargeoffset >= e.off THEN least(e.off + {STOP_OVERHANG_MIN}, s.unitdischargeoffset)
                            ELSE e.off + {STOP_OVERHANG_MIN} END,
                       0, NULL, e.med_name, 'inferred stop: not charted again within 6 h (ricu interval rule)', 'stop'
                FROM e JOIN stays s ON s.stay = e.stay
                WHERE e.rate > 0 AND (e.next_off IS NULL OR e.next_off - e.off > {STOP_GAP_MIN}))
            SELECT CAST(s.hosp AS VARCHAR) AS hospitalization_id, {t('ev.off')} AS admin_dttm, ev.med_name,
                   ev.med_category, {self._med_group('medication_admin_continuous')} AS med_group,
                   'iv' AS med_route_category, ev.rate AS med_dose, ev.unit AS med_dose_unit,
                   CASE WHEN ev.unit NOT LIKE 'ml/%' THEN ev.vol_rate END AS volume_infusion_rate,
                   CASE WHEN ev.unit NOT LIKE 'ml/%' AND ev.vol_rate IS NOT NULL THEN 'ml/hr' END AS volume_infusion_rate_unit,
                   ev.mar_action_name, ev.mar_action_category, 'administered' AS mar_action_group
            FROM ev JOIN stays s ON s.stay = ev.stay
            WHERE ev.mar_action_category IS NOT NULL"""
        return sql, ("hospitalization_id", "admin_dttm", "med_category")

    # -- renal / circulatory support -------------------------------------------------------------------
    def sql_crrt_therapy(self):
        trt = self._treatments()
        sql = f"""
            SELECT CAST(s.hosp AS VARCHAR) AS hospitalization_id, {t('x.off')} AS recorded_dttm,
                   x.name AS crrt_mode_name, x.category AS crrt_mode_category
            FROM {trt} x JOIN stays s ON s.stay = x.stay WHERE x.target = 'crrt_therapy'"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_intermittent_dialysis(self):
        trt = self._treatments()
        sql = f"""
            SELECT DISTINCT CAST(s.hosp AS VARCHAR) AS hospitalization_id, {t('x.off')} AS recorded_dttm
            FROM {trt} x JOIN stays s ON s.stay = x.stay WHERE x.target = 'intermittent_dialysis'"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_mcs(self):
        trt = self._treatments()
        sql = f"""
            SELECT CAST(s.hosp AS VARCHAR) AS hospitalization_id, {t('x.off')} AS recorded_dttm,
                   x.name AS support_name, x.category AS support_category
            FROM {trt} x JOIN stays s ON s.stay = x.stay WHERE x.target = 'mcs'"""
        return sql, ("hospitalization_id", "recorded_dttm")

    # -- diagnoses --------------------------------------------------------------------------------------
    def sql_patient_diagnosis(self):
        """Time-stamped problem-list codes; an entry's ICD-10 code is used when present, else its ICD-9 codes."""
        sql = f"""
            WITH d AS (
                SELECT x.patientunitstayid AS stay, x.diagnosisoffset AS off, x.diagnosisid,
                       list_transform(string_split(x.icd9code, ','), c -> trim(c)) AS codes
                FROM {self.src('diagnosis.parquet')} x WHERE x.icd9code IS NOT NULL),
            c AS (
                SELECT stay, off, diagnosisid, codes[i] AS code, i, len(codes) AS n
                FROM (SELECT *, generate_subscripts(codes, 1) AS i FROM d)),
            k AS (   -- ICD-10 = the last code of a multi-code entry (eICU order: ICD-9 first), or a single
                     -- letter-initial code that is not an ICD-9 E/V code (E + 3 digits, V + 2 digits)
                SELECT *, CASE WHEN regexp_full_match(code, '[A-Z][0-9]{{2}}[0-9A-Z.]*')
                                    AND NOT regexp_full_match(code, 'E[0-9]{{3}}(\\.[0-9]+)?')
                                    AND (i = n AND n > 1 OR n = 1 AND NOT regexp_full_match(code, 'V[0-9]{{2}}(\\.[0-9]+)?'))
                               THEN 'ICD10CM' ELSE 'ICD9CM' END AS fmt
                FROM c WHERE code <> ''),
            pick AS (
                SELECT *, bool_or(fmt = 'ICD10CM') OVER (PARTITION BY diagnosisid) AS has10 FROM k)
            SELECT CAST(any_value(s.pid) AS VARCHAR) AS patient_id, CAST(s.hosp AS VARCHAR) AS hospitalization_id,
                   replace(p.code, '.', '') AS diagnosis_code, p.fmt AS diagnosis_code_format,
                   'problem_list' AS source_type, min({t('p.off')}) AS start_dttm
            FROM pick p JOIN stays s ON s.stay = p.stay
            WHERE p.fmt = 'ICD10CM' OR NOT p.has10
            GROUP BY s.hosp, replace(p.code, '.', ''), p.fmt"""
        return sql, ("hospitalization_id", "start_dttm")

    # -- fluid balance ----------------------------------------------------------------------------------
    def _io(self, direction: str) -> str:
        io = self.src("intakeOutput.parquet")
        lk = self.regex_lookup("io", f"SELECT lower(cellpath) AS txt FROM {io}", "io",
                               ("direction", "category", "group_name"))
        if direction == "output":
            self.coverage("output", "cellpath", f"""
                SELECT x.celllabel AS txt, count(*) AS n, bool_or(l.category IS NOT NULL) AS mapped
                FROM {io} x JOIN {lk} l ON l.txt = lower(x.cellpath)
                WHERE lower(x.cellpath) LIKE '%|output (ml)|%' GROUP BY 1""")
        return f"""
            SELECT DISTINCT CAST(s.hosp AS VARCHAR) AS hospitalization_id, {t('x.intakeoutputoffset')} AS recorded_dttm,
                   x.celllabel AS name, l.category, l.group_name, x.cellvaluenumeric AS volume, x.cellpath
            FROM {io} x JOIN stays s ON s.stay = x.patientunitstayid JOIN {lk} l ON l.txt = lower(x.cellpath)
            WHERE l.direction = '{direction}' AND l.category IS NOT NULL AND l.category <> '__skip__'
              AND x.cellvaluenumeric > 0"""

    def sql_input(self):
        sql = f"""SELECT hospitalization_id, recorded_dttm, name AS input_name, category AS input_category,
                         group_name AS input_group, volume AS input_volume FROM ({self._io('input')})"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_output(self):
        sql = f"""SELECT hospitalization_id, recorded_dttm, name AS output_name, category AS output_category,
                         group_name AS output_group, volume AS output_volume FROM ({self._io('output')})"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_invasive_hemodynamics(self):
        m = self.map("hemodynamics")
        long = " UNION ALL ".join([
            self._monitor_long("vitalPeriodic.parquet", self._items("hemodynamics", "vitalPeriodic")),
            self._monitor_long("vitalAperiodic.parquet", self._items("hemodynamics", "vitalAperiodic")),
            f"SELECT stay, off, item, value FROM ({self._nurse_long(self._items('hemodynamics', 'nurseCharting'))})",
        ])
        sql = f"""
            SELECT CAST(s.hosp AS VARCHAR) AS hospitalization_id, {t('v.off')} AS recorded_dttm,
                   v.item AS measurement_name, m.measurement_category, v.value AS measurement_value
            FROM ({long}) v JOIN {m} m ON m.item = v.item JOIN stays s ON s.stay = v.stay
            WHERE v.value IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_code_status(self):
        """Care-plan limitations per time point -> one category; only changes per patient are kept."""
        cp = self._care_plan()
        sql = f"""
            WITH f AS (
                SELECT stay, off, string_agg(DISTINCT name, '; ' ORDER BY name) AS name,
                       bool_or(category = 'and') AS cmo, bool_or(category = 'dnr') AS dnr,
                       bool_or(category = 'dni_only') AS dni, bool_or(category = 'other') AS partial,
                       bool_or(category = 'full') AS full_
                FROM {cp} WHERE target = 'code_status' GROUP BY 1, 2),
            c AS (
                SELECT * FROM (
                    SELECT s.pid, {t('f.off')} AS start_dttm, f.name,
                           CASE WHEN cmo THEN 'and' WHEN dnr AND dni THEN 'dnr_or_dni' WHEN dnr THEN 'dnr'
                                WHEN dni THEN 'dni_only' WHEN partial THEN 'other' WHEN full_ THEN 'full' END AS category
                    FROM f JOIN stays s ON s.stay = f.stay)
                WHERE category IS NOT NULL),
            o AS (SELECT *, lag(category) OVER (PARTITION BY pid ORDER BY start_dttm, category) AS prev FROM c)
            SELECT CAST(pid AS VARCHAR) AS patient_id, start_dttm, name AS code_status_name,
                   category AS code_status_category
            FROM o WHERE category IS DISTINCT FROM prev"""
        return sql, ("patient_id", "start_dttm")

    # -- ICU stay context ----------------------------------------------------------------------------
    def sql_ext_icu_stay(self):
        """Admission diagnosis group: the APACHE IV admission diagnosis (``admissionDx`` 'All Diagnosis' path:
        operative branch, organ system, diagnosis), its system resolved by the shared ``dx_systems`` lexicon;
        sepsis and overdose diagnoses (filed under cardiovascular and neurology) by their names. Only entries
        made within 24 h of unit admission count (subset: median 35 min, 99th percentile 25 h), so the header
        is admission documentation; APACHE's operative status by definition includes going to the OR within 4 h."""
        u = self.map("unit_types")
        src = self.map("admit_sources")
        lex = self.lexicon("dx_systems")
        apx = f"""SELECT patientunitstayid AS stay, admitdxenteredoffset AS off, split_part(admitdxpath, '|', 3) AS branch,
                         split_part(admitdxpath, '|', 5) AS system_txt, admitdxname AS dx_name
                  FROM {self.src('admissionDx.parquet')} WHERE split_part(admitdxpath, '|', 2) = 'All Diagnosis'
                    AND admitdxenteredoffset <= {DX_WITHIN_MIN}"""
        sys_lk = self.regex_lookup("dx_system", f"SELECT lower(system_txt) AS txt FROM ({apx})", "dx_systems",
                                   ("system",), table=lex)
        name_lk = self.regex_lookup("dx_name", f"SELECT lower(dx_name) AS txt FROM ({apx})", "dx_systems",
                                    ("system",), table=lex)
        self.coverage("ext_icu_stay", "dx_systems", f"""
            SELECT a.system_txt AS txt, count(*) AS n, bool_or(l.system IS NOT NULL) AS mapped
            FROM ({apx}) a LEFT JOIN {sys_lk} l ON l.txt = lower(a.system_txt) GROUP BY 1""")
        sql = f"""
            WITH apache_dx AS (
                SELECT a.stay, arg_min(a.branch, a.off) AS branch,
                       arg_min(coalesce(CASE WHEN a.branch <> 'Operative' AND n.system IN ('sepsis', 'intoxication') THEN n.system END,
                                       l.system), a.off)
                           AS system
                FROM ({apx}) a LEFT JOIN {sys_lk} l ON l.txt = lower(a.system_txt)
                LEFT JOIN {name_lk} n ON n.txt = lower(a.dx_name)
                GROUP BY 1),
            dx AS (
                SELECT patientunitstayid AS stay,
                       max(split_part(admitdxpath, '|', 3)) FILTER (WHERE split_part(admitdxpath, '|', 2)
                           = 'Was the patient admitted from the O.R. or went to the O.R. within 4 hours of admission?') AS op,
                       max(split_part(admitdxpath, '|', 3)) FILTER (WHERE split_part(admitdxpath, '|', 2) = 'Elective') AS elective
                FROM {self.src('admissionDx.parquet')} WHERE admitdxenteredoffset <= {DX_WITHIN_MIN} GROUP BY 1),
            sev AS (
                SELECT patientunitstayid AS stay, max(apachescore) AS apache
                FROM {self.src('apachePatientResult.parquet')} WHERE apacheversion = 'IVa' AND apachescore >= 0 GROUP BY 1)
            SELECT 'eicu:' || CAST(s.stay AS VARCHAR) AS stay_id, CAST(s.pid AS VARCHAR) AS patient_id,
                   CAST(s.hosp AS VARCHAR) AS hospitalization_id, 'eicu' AS dataset,
                   'eicu_' || CAST(s.hospitalid AS VARCHAR) AS site_id,
                   s.unit_admit AS icu_in_dttm, s.unit_disch AS icu_out_dttm, u.location_type AS unit_type,
                   true AS clock_valid, false AS demographics_binned,
                   coalesce(a.unit_admission_source, 'unknown') AS admission_source,
                   CASE WHEN dx.op = 'No' THEN 'nonsurgical'
                        WHEN dx.op = 'Yes' AND dx.elective = 'Yes' THEN 'elective_surgery'
                        WHEN dx.op = 'Yes' AND dx.elective = 'No' THEN 'emergency_surgery'
                        ELSE 'unknown' END AS surgical_status,
                   s.apacheadmissiondx AS adm_dx_name,
                   {group_sql("ad.branch = 'Operative'", 'ad.system')} AS adm_dx_group,
                   CAST(row_number() OVER (PARTITION BY s.pid ORDER BY s.unit_admit, s.stay) AS INTEGER) AS stay_seq,
                   CASE s.unitdischargestatus WHEN 'Expired' THEN 'dead' WHEN 'Alive' THEN 'alive' ELSE 'unknown' END AS icu_outcome,
                   CASE WHEN sev.apache IS NOT NULL THEN 'apache_iva' END AS severity_name,
                   CAST(sev.apache AS DOUBLE) AS severity_value,
                   CAST(TRY_CAST(s.hospitaldischargeyear AS INTEGER) AS VARCHAR) AS era
            FROM stays s LEFT JOIN {u} u ON u.unittype = s.unittype
            LEFT JOIN {src} a ON a.source = s.unitadmitsource
            LEFT JOIN dx ON dx.stay = s.stay LEFT JOIN sev ON sev.stay = s.stay
            LEFT JOIN apache_dx ad ON ad.stay = s.stay"""
        return sql, ("stay_id",)
