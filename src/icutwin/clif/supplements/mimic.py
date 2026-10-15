"""MIMIC-IV supplement to CLIF-MIMIC.

* ``ext_icu_stay`` from ``icustays``. A stay counts as an ICU stay when one of its ADT segments is an ICU
  per CLIF-MIMIC's care-unit mapping: ``icustays`` also lists stays spent only in Neuro Intermediate, Neuro
  Stepdown or PACU, which CLIF (and our NWICU adapter) treat as stepdown or procedural locations.
  - admission source: the care unit of the last ADT segment before ICU admission, otherwise the hospital
    admission location;
  - surgical status: a surgical or orthopaedic service at or before ICU admission (mimic-code OASIS also
    reads the first day after it; a service taken up then is post-admission information, which a header must
    not carry: 92 of 1,917 subset stays), elective when the admission type is ELECTIVE or SURGICAL SAME DAY
    ADMISSION (MIMIC-IV's type for planned surgery; mimic-code checks ELECTIVE only, a MIMIC-III carry-over);
  - admission diagnosis group (``icutwin.clif.dx_groups``): MIMIC records no admission diagnosis, so the
    service gives the organ system (``maps/mimic/services.csv``): the surgical service of an operated stay,
    else the service at ICU admission; general medicine and surgery give none ('unspecified');
  - ICU death: hospital death time at or before ICU discharge.
* ``labs``: CLIF 3.0 categories CLIF-MIMIC does not map (``maps/mimic/labs.csv``), with CLIF-MIMIC's
  conventions: hospital labs (``hadm_id`` set, or drawn in the emergency department before the admission, as
  patch 0003 gives CLIF-MIMIC), order and collect time = ``charttime``, result time =
  ``storetime`` (rows without it dropped), values recovered from comments, exact duplicates dropped.
* ``invasive_hemodynamics`` from chartevents (``maps/mimic/hemodynamics.csv``; CLIF-MIMIC has no such table)
  and the CLIF 3.0 intracranial-pressure vital (``maps/mimic/vitals.csv``).
* ``medication_admin_intermittent`` from the hospital eMAR (``emar``, ``emar_detail``): CLIF-MIMIC reads only
  ICU ``inputevents``, so it has no oral drugs and none of the categories its mapping omits (steroids,
  potassium, antipsychotics, ...); parenteral ICU doses it already emits are not added again.
* ``medication_admin_continuous`` from ICU ``inputevents`` drips CLIF-MIMIC's mapping leaves out
  (``maps/mimic/continuous_items.csv``: CRRT potassium, calcium and citrate, IVIG, isoproterenol).
* ``input`` from ICU ``inputevents`` (``maps/mimic/input_items.csv``): CLIF-MIMIC maps one irrigation item, so
  transfusions, enteral formulas, oral intake, lactated Ringer's and other unmapped fluids were missing.
* ``intermittent_dialysis`` (CLIF-MIMIC builds none; mimic-code ``rrt`` types these sessions IHD): the start and
  end of each 'Hemodialysis' procedure (procedureevents 225441) and each 'Hemodialysis Output' (chartevents
  226499, mL) as the net ultrafiltration. Peritoneal dialysis has no CLIF 3.0 table.
* ``microbiology_culture`` / ``microbiology_susceptibility`` from ``microbiologyevents`` (CLIF-MIMIC builds
  neither), resolved with the shared microbiology lexicons; ``microbiology_nonculture`` for the PCR tests of
  CLIF 3.0's organism panel (``maps/mimic/nonculture_tests.csv``: the C. difficile PCR).

Times are MIMIC wall-clock times (our canonical convention; CLIF-MIMIC's UTC output is shifted back in S2b).
"""

from __future__ import annotations

from icutwin.clif.conform import medication_extensions
from icutwin.clif.dx_groups import group_sql
from icutwin.clif.supplements.base import ClifSupplement


def _ts(col: str) -> str:
    return f"CAST({col} AS TIMESTAMPTZ)"  # naive MIMIC times are wall-clock; the session zone is UTC


class MimicSupplement(ClifSupplement):
    dataset = "mimic"
    tables = ("ext_icu_stay", "labs", "vitals", "invasive_hemodynamics", "medication_admin_intermittent",
              "medication_admin_continuous", "input", "microbiology_culture", "microbiology_nonculture",
              "microbiology_susceptibility", "intermittent_dialysis")

    # -- ext_icu_stay -------------------------------------------------------------------------------
    def sql_ext_icu_stay(self):
        cu, al, sv = self.map("care_units"), self.map("admission_locations"), self.map("services")
        tr = self.src("hosp/transfers.parquet")
        self.coverage("ext_icu_stay", "services", f"""
            SELECT v.curr_service AS txt, count(*) AS n, bool_or(m.service IS NOT NULL) AS mapped
            FROM {self.src('hosp/services.parquet')} v LEFT JOIN {sv} m ON m.service = v.curr_service GROUP BY 1""")
        self.coverage("ext_icu_stay", "care_units", f"""
            SELECT t.careunit AS txt, count(*) AS n, bool_or(c.careunit IS NOT NULL) AS mapped
            FROM {tr} t LEFT JOIN {cu} c ON c.careunit = t.careunit WHERE t.careunit IS NOT NULL GROUP BY 1""")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE seg AS
            SELECT t.hadm_id, {_ts('t.intime')} AS seg_in, {_ts('t.outtime')} AS seg_out, t.eventtype, t.careunit,
                   c.location_category, c.location_type, c.admission_source
            FROM {tr} t LEFT JOIN {cu} c ON c.careunit = t.careunit
            WHERE t.hadm_id IS NOT NULL AND t.careunit IS NOT NULL""")
        sql = f"""
            WITH stays AS (
                SELECT i.subject_id, i.hadm_id, i.stay_id, {_ts('i.intime')} AS icu_in, {_ts('i.outtime')} AS icu_out
                FROM {self.src('icu/icustays.parquet')} i
                WHERE i.hadm_id IS NOT NULL AND i.intime IS NOT NULL AND i.outtime IS NOT NULL),
            icu_seg AS (     -- ICU segments overlapping each stay
                SELECT s.stay_id, g.seg_in, g.location_type
                FROM stays s JOIN seg g ON g.hadm_id = s.hadm_id AND g.location_category = 'icu'
                 AND g.seg_in < s.icu_out AND coalesce(g.seg_out, s.icu_out) > s.icu_in),
            unit AS (SELECT stay_id, arg_min(location_type, seg_in) AS unit_type FROM icu_seg GROUP BY 1),
            prev AS (        -- where the patient came from: the last ADT segment starting before ICU admission
                SELECT s.stay_id, arg_max(coalesce(g.admission_source, 'unknown'), g.seg_in) AS source
                FROM stays s JOIN seg g ON g.hadm_id = s.hadm_id AND g.seg_in < s.icu_in AND g.eventtype <> 'discharge'
                GROUP BY 1),
            surg AS (        -- mimic-code OASIS surgflag, on the services up to ICU admission
                SELECT s.stay_id,
                       max(CASE WHEN lower(v.curr_service) LIKE '%surg%' OR v.curr_service = 'ORTHO' THEN 1 ELSE 0 END) AS surgical,
                       arg_min(v.curr_service, v.transfertime) FILTER (
                           WHERE lower(v.curr_service) LIKE '%surg%' OR v.curr_service = 'ORTHO') AS surg_service,
                       arg_max(v.curr_service, v.transfertime) AS icu_service
                FROM stays s LEFT JOIN {self.src('hosp/services.parquet')} v
                  ON v.hadm_id = s.hadm_id AND {_ts('v.transfertime')} <= s.icu_in
                GROUP BY 1)
            SELECT 'mimic:' || CAST(s.stay_id AS VARCHAR) AS stay_id, CAST(s.subject_id AS VARCHAR) AS patient_id,
                   CAST(s.hadm_id AS VARCHAR) AS hospitalization_id, 'mimic' AS dataset, 'mimic' AS site_id,
                   s.icu_in AS icu_in_dttm, s.icu_out AS icu_out_dttm, u.unit_type,
                   true AS clock_valid, false AS demographics_binned,
                   coalesce(p.source, l.admission_source, 'unknown') AS admission_source,
                   CASE WHEN a.admission_type IS NULL THEN 'unknown'
                        WHEN g.surgical = 1 AND a.admission_type IN ('ELECTIVE', 'SURGICAL SAME DAY ADMISSION')
                            THEN 'elective_surgery'
                        WHEN g.surgical = 1 THEN 'emergency_surgery'
                        ELSE 'nonsurgical' END AS surgical_status,
                   CASE WHEN g.surgical = 1 THEN json_object('service', g.surg_service) END AS surgery_context,
                   {group_sql('g.surgical = 1', "nullif(CASE WHEN g.surgical = 1 THEN vs.system ELSE vm.system END, '')")}
                       AS adm_dx_group,
                   CAST(row_number() OVER (PARTITION BY s.subject_id ORDER BY s.icu_in, s.stay_id) AS INTEGER) AS stay_seq,
                   CASE WHEN a.deathtime IS NOT NULL AND {_ts('a.deathtime')} <= s.icu_out THEN 'dead' ELSE 'alive' END
                       AS icu_outcome,
                   replace(pt.anchor_year_group, ' ', '') AS era   -- the patient's anchor year group
            FROM stays s
            JOIN unit u USING (stay_id)                    -- ICU stays only (see module docstring)
            JOIN {self.src('hosp/admissions.parquet')} a ON a.hadm_id = s.hadm_id
            LEFT JOIN {self.src('hosp/patients.parquet')} pt ON pt.subject_id = s.subject_id
            LEFT JOIN prev p USING (stay_id)
            LEFT JOIN surg g USING (stay_id)
            LEFT JOIN {sv} vs ON vs.service = g.surg_service LEFT JOIN {sv} vm ON vm.service = g.icu_service
            LEFT JOIN {al} l ON l.admission_location = a.admission_location"""
        return sql, ("stay_id",)

    # -- intermittent_dialysis ------------------------------------------------------------------------
    def sql_intermittent_dialysis(self):
        pe, ce = self.src("icu/procedureevents.parquet"), self.src("icu/chartevents.parquet")
        sql = f"""
            SELECT CAST(hadm_id AS VARCHAR) AS hospitalization_id, {_ts('t')} AS recorded_dttm,
                   CAST(NULL AS DOUBLE) AS blood_flow_rate, CAST(NULL AS DOUBLE) AS dialysate_flow_rate,
                   CAST(NULL AS DOUBLE) AS net_ultrafiltration_out
            FROM (SELECT hadm_id, starttime AS t FROM {pe} WHERE itemid = 225441 AND value IS NOT NULL
                  UNION SELECT hadm_id, endtime FROM {pe} WHERE itemid = 225441 AND value IS NOT NULL)
            WHERE hadm_id IS NOT NULL AND t IS NOT NULL
            UNION ALL
            SELECT CAST(hadm_id AS VARCHAR), {_ts('charttime')}, NULL, NULL, valuenum
            FROM {ce} WHERE itemid = 226499 AND valuenum IS NOT NULL AND hadm_id IS NOT NULL"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def _ed_admission(self, table: str, time: str, key: str, where: str = "TRUE") -> str:
        """The rows of ``table`` (those meeting ``where``) that belong to a hospital admission: their own ``hadm_id``,
        or, for rows drawn in the emergency department before the admission (they carry none), the admission whose
        ED stay (``edregtime``..``admittime``) holds ``time``, the earliest if several (CLIF-MIMIC patch 0003 does
        the same for its labs). Rows of no admission are left out (every caller drops them). ``key`` identifies a
        row. Only the rows without ``hadm_id`` are joined and ranked: at full scale a window over every row of
        ``labevents`` (158M) dominated the whole MIMIC supplement."""
        return f"""(SELECT x.* REPLACE (CAST(x.hadm_id AS BIGINT) AS hadm_id) FROM {table} x
                    WHERE x.hadm_id IS NOT NULL AND ({where})
                    UNION ALL
                    SELECT x.* REPLACE (CAST(a.hadm_id AS BIGINT) AS hadm_id)
                    FROM {table} x JOIN {self.src('hosp/admissions.parquet')} a
                      ON a.subject_id = x.subject_id AND {time} BETWEEN a.edregtime AND a.admittime
                    WHERE x.hadm_id IS NULL AND ({where})
                    QUALIFY row_number() OVER (PARTITION BY x.{key} ORDER BY a.admittime) = 1)"""

    # -- labs ---------------------------------------------------------------------------------------
    def sql_labs(self):
        m = self.map("labs")
        le = self.src("hosp/labevents.parquet")
        ids = lambda source: (f"SELECT CAST(itemid AS BIGINT) FROM {m} WHERE source = '{source}'")  # noqa: E731
        self.con.execute(f"""
            CREATE OR REPLACE TABLE lab_raw AS
            WITH spec AS (SELECT specimen_id, any_value(value) AS specimen FROM {le} WHERE itemid = 52033 GROUP BY 1)
            SELECT 'labevents' AS source, l.hadm_id, l.charttime, l.storetime, l.itemid, l.value, l.valuenum,
                   l.valueuom, l.comments, s.specimen
            FROM {self._ed_admission(le, 'x.charttime', 'labevent_id', f"x.itemid IN ({ids('labevents')})")} l
                 LEFT JOIN spec s USING (specimen_id)
            WHERE l.hadm_id IS NOT NULL AND l.itemid IN ({ids('labevents')})
            UNION ALL
            SELECT 'chartevents', c.hadm_id, c.charttime, c.storetime, c.itemid, c.value, c.valuenum, c.valueuom,
                   NULL, NULL
            FROM {self.src('icu/chartevents.parquet')} c
            WHERE c.hadm_id IS NOT NULL AND c.itemid IN ({ids('chartevents')})""")
        join = (f"JOIN {m} m ON m.source = r.source AND CAST(m.itemid AS BIGINT) = r.itemid "
                f"AND (m.specimen IS NULL OR m.specimen = r.specimen)")
        unit_ok = "(m.unit_in = '*' OR lower(trim(r.valueuom)) = lower(m.unit_in))"
        self.coverage("labs", "itemid_unit", f"""
            SELECT r.source || ':' || CAST(r.itemid AS VARCHAR) || ' [' || coalesce(r.valueuom, 'NULL') || ']' AS txt,
                   count(*) AS n, bool_or({unit_ok}) AS mapped
            FROM lab_raw r {join} GROUP BY 1""")
        # CLIF-MIMIC: a missing valuenum is recovered from the comment unless it says 'not done' / 'unable'
        num = (r"coalesce(r.valuenum, CASE WHEN NOT regexp_matches(lower(r.comments), 'not done|unable') "
               r"THEN TRY_CAST(nullif(regexp_extract(r.comments, '\d+\.\d+|\d+'), '') AS DOUBLE) END)")
        sql = f"""
            SELECT DISTINCT ON (hospitalization_id, lab_collect_dttm, lab_result_dttm, lab_category, lab_value_numeric) *
            FROM (
                SELECT CAST(r.hadm_id AS VARCHAR) AS hospitalization_id,
                       {_ts('r.charttime')} AS lab_order_dttm, {_ts('r.charttime')} AS lab_collect_dttm,
                       {_ts('r.storetime')} AS lab_result_dttm,
                       m.lab_order_category, coalesce(d.label, CAST(r.itemid AS VARCHAR)) AS lab_name,
                       m.lab_category,
                       CASE WHEN CAST(m.factor AS DOUBLE) = 1 THEN coalesce(r.value, r.comments)
                            ELSE CAST({num} * CAST(m.factor AS DOUBLE) AS VARCHAR) END AS lab_value,
                       {num} * CAST(m.factor AS DOUBLE) AS lab_value_numeric,
                       m.reference_unit, r.specimen AS lab_specimen_name, m.lab_specimen_category
                FROM lab_raw r {join}
                LEFT JOIN (SELECT itemid, label FROM {self.src('hosp/d_labitems.parquet')}
                           UNION ALL SELECT itemid, label FROM {self.src('icu/d_items.parquet')}) d ON d.itemid = r.itemid
                WHERE {unit_ok} AND r.storetime IS NOT NULL)"""
        return sql, ("hospitalization_id", "lab_collect_dttm", "lab_category")

    # -- microbiology -------------------------------------------------------------------------------
    def _micro(self) -> str:
        """Organism-level rows of culture, Gram-stain and smear tests (table ``micro``).

        An order is one test on one specimen (``micro_specimen_id`` x ``test_itemid``). Tests that isolated
        no organism carry their result in the comment ('no growth', Gram-stain findings); comments that state
        no result (e.g. 'pending') are dropped. Non-culture tests (PCR, antigen, serology, viral load) are not
        cultures; the PCR tests of CLIF 3.0's panel go to ``microbiology_nonculture``. Hospital specimens only
        (``hadm_id``).
        """
        if self.con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'micro'").fetchone()[0]:
            return "micro"
        me = self.src("hosp/microbiologyevents.parquet")
        tests = self.regex_lookup("micro_tests", f"SELECT lower(test_name) AS txt FROM {me}", "micro_tests",
                                  ("method_category",), table=self.lexicon("micro_tests"))
        specs = self.regex_lookup("micro_specs", f"SELECT lower(spec_type_desc) AS txt FROM {me}", "micro_specimens",
                                  ("fluid_category",), table=self.lexicon("micro_specimens"))
        self.con.execute(f"""
            CREATE OR REPLACE TABLE micro_raw AS
            WITH m AS (SELECT * FROM {self._ed_admission(me, 'coalesce(x.charttime, x.chartdate)', 'microevent_id')}
                       WHERE hadm_id IS NOT NULL),
            org AS (     -- one row per isolated organism
                SELECT micro_specimen_id, test_itemid, org_itemid, isolate_num, any_value(org_name) AS name, 'org' AS src
                FROM m WHERE org_name IS NOT NULL GROUP BY 1, 2, 3, 4),
            neg AS (     -- tests without an isolate: the comment states the result; Gram-stain sentences
                         -- about leukocytes or epithelial cells are removed so 'no PMNs seen' is not read
                SELECT micro_specimen_id, test_itemid, NULL AS org_itemid, NULL AS isolate_num,
                       trim(regexp_replace(lower(first(comments ORDER BY coalesce(storetime, storedate) DESC NULLS LAST,
                                                       comments)),
                                           '[^.]*(leukocyte|epithelial|squamous|\\bcells?\\b|not the optimal specimen'
                                           '|interpreted with caution|whenever possible)[^.]*\\.?', '', 'g')) AS name,
                       'comment' AS src
                FROM m GROUP BY 1, 2 HAVING count(org_name) = 0)
            SELECT o.*, h.* FROM (SELECT * FROM org UNION ALL SELECT * FROM neg) o
            JOIN (SELECT micro_specimen_id, test_itemid, any_value(subject_id) AS subject_id,
                         any_value(hadm_id) AS hadm_id, any_value(spec_type_desc) AS spec_type_desc,
                         any_value(test_name) AS test_name,
                         min(coalesce(charttime, chartdate)) AS t_collect, min(coalesce(storetime, storedate)) AS t_result
                  FROM m GROUP BY 1, 2) h USING (micro_specimen_id, test_itemid)""")
        orgs = self.regex_lookup("organisms", "SELECT lower(name) AS txt FROM micro_raw", "organisms",
                                 ("organism_category", "organism_group"), table=self.lexicon("organisms"))
        self.con.execute(f"""
            CREATE OR REPLACE TABLE micro AS
            SELECT r.*, t.method_category, s.fluid_category, g.organism_category, g.organism_group,
                   CAST(r.micro_specimen_id AS VARCHAR) || ':' || CAST(r.test_itemid AS VARCHAR) AS order_id,
                   CAST(r.micro_specimen_id AS VARCHAR) || ':' || CAST(r.test_itemid AS VARCHAR) || ':'
                       || coalesce(CAST(r.org_itemid AS VARCHAR), 'none') || ':' || coalesce(CAST(r.isolate_num AS VARCHAR), '0')
                       AS organism_id
            FROM micro_raw r
            LEFT JOIN {tests} t ON t.txt = lower(r.test_name)
            LEFT JOIN {specs} s ON s.txt = lower(r.spec_type_desc)
            LEFT JOIN {orgs} g ON g.txt = lower(r.name)""")
        self.coverage("microbiology_culture", "test", """
            SELECT lower(test_name) AS txt, count(*) AS n, bool_or(method_category IS NOT NULL) AS mapped
            FROM micro GROUP BY 1""")
        self.coverage("microbiology_culture", "organism", """
            SELECT coalesce(lower(name), '(no result)') AS txt, count(*) AS n, bool_or(organism_category IS NOT NULL) AS mapped
            FROM micro WHERE method_category IN ('culture', 'gram_stain', 'smear') GROUP BY 1""")
        return "micro"

    def sql_microbiology_culture(self):
        mi = self._micro()
        sql = f"""
            SELECT CAST(subject_id AS VARCHAR) AS patient_id, CAST(hadm_id AS VARCHAR) AS hospitalization_id,
                   order_id, organism_id, {_ts('t_collect')} AS order_dttm, {_ts('t_collect')} AS collect_dttm,
                   {_ts('t_result')} AS result_dttm, spec_type_desc AS fluid_name, fluid_category,
                   test_name AS method_name, method_category, name AS organism_name, organism_category, organism_group
            FROM {mi}
            WHERE method_category IN ('culture', 'gram_stain', 'smear') AND organism_category IS NOT NULL
              AND organism_category <> '__skip__'"""
        return sql, ("hospitalization_id", "collect_dttm", "organism_id")

    # -- medication_admin_intermittent (emar) ------------------------------------------------------
    def sql_medication_admin_intermittent(self):
        """Doses given per the hospital eMAR, which CLIF-MIMIC does not read (it uses ICU ``inputevents``).

        An administration is an ``emar`` 'Administered' / 'Delayed Administered' event; its dose is the sum of
        the product rows of ``emar_detail`` (one unit), its route the administration row's, else the pharmacy
        order's (``maps/mimic/emar_routes.csv``; non-systemic routes and infusions are not intermittent doses).
        Inside an ICU stay (``icustays``, MetaVision charting) the tool already emits parenteral doses of the
        categories its ``inputevents`` mapping covers (``tool_intermittent_categories.csv``: 60-94% of such
        eMAR doses have a tool row within 60 min), and the drugs it maps as infusions whose intermittent category
        is a local extension (``tool_continuous_categories.csv`` x ``medication_extensions.csv`` rows with
        ``untyped_iv``: albumin, vasoactive and vasodilator boluses, which MetaVision charts in ``inputevents``),
        so only enteral, sublingual and inhaled doses and parenteral doses of other categories are added there;
        outside ICU stays every dose is added.
        """
        h, i = "hosp", "icu"
        routes, tool = self.map("emar_routes"), self.map("tool_intermittent_categories")
        infused = sorted({r["med_category"] for r in medication_extensions()
                          if r["table"] == "medication_admin_intermittent" and r["untyped_iv"] == "continuous"})
        tool_cont = self.map("tool_continuous_categories")
        icu_charted = f"""SELECT med_category FROM {tool} UNION
                          SELECT med_category FROM {tool_cont} WHERE med_category IN ({', '.join(f"'{c}'" for c in infused)})"""
        emar = self.src(f"{h}/emar.parquet")
        lk = self.regex_lookup("drugs", f"SELECT lower(medication) AS txt FROM {emar}", "drugs",
                               ("ingredient", "continuous", "intermittent"), table=self.lexicon("drugs"))
        self.con.execute(f"""
            CREATE OR REPLACE TABLE emar_dose AS
            WITH d AS (
                SELECT emar_id,
                       sum(TRY_CAST(dose_given AS DOUBLE)) FILTER (WHERE parent_field_ordinal IS NOT NULL) AS dose,
                       count(DISTINCT lower(dose_given_unit)) FILTER (WHERE parent_field_ordinal IS NOT NULL) AS n_units,
                       any_value(lower(dose_given_unit)) FILTER (WHERE parent_field_ordinal IS NOT NULL) AS unit,
                       any_value(route) FILTER (WHERE parent_field_ordinal IS NULL) AS route,
                       any_value(administration_type) FILTER (WHERE parent_field_ordinal IS NULL) AS admin_type
                FROM {self.src(f'{h}/emar_detail.parquet')} GROUP BY 1)
            SELECT e.hadm_id, e.charttime, e.medication, e.event_txt, upper(trim(coalesce(d.route, ph.route))) AS route,
                   CASE WHEN d.n_units = 1 THEN d.dose END AS dose, CASE WHEN d.n_units = 1 THEN d.unit END AS unit,
                   d.admin_type, nullif(l.intermittent, '') AS intermittent,
                   EXISTS (SELECT 1 FROM {self.src(f'{i}/icustays.parquet')} s
                           WHERE s.hadm_id = e.hadm_id AND e.charttime BETWEEN s.intime AND s.outtime) AS in_icu
            FROM {emar} e
            LEFT JOIN d USING (emar_id)
            LEFT JOIN (SELECT DISTINCT ON (pharmacy_id) pharmacy_id, route FROM {self.src(f'{h}/pharmacy.parquet')}) ph
                   ON ph.pharmacy_id = e.pharmacy_id
            LEFT JOIN {lk} l ON l.txt = lower(e.medication)
            WHERE e.event_txt IN ('Administered', 'Delayed Administered') AND e.hadm_id IS NOT NULL""")
        self.coverage("medication_admin_intermittent", "emar_name", """
            SELECT lower(medication) AS txt, count(*) AS n, bool_or(intermittent IS NOT NULL) AS mapped
            FROM emar_dose GROUP BY 1""")
        self.coverage("medication_admin_intermittent", "emar_route", f"""
            SELECT coalesce(e.route, '(none)') AS txt, count(*) AS n, bool_or(r.route IS NOT NULL) AS mapped
            FROM emar_dose e LEFT JOIN {routes} r ON r.route = e.route GROUP BY 1""")
        sql = f"""
            SELECT CAST(e.hadm_id AS VARCHAR) AS hospitalization_id, {_ts('e.charttime')} AS admin_dttm,
                   e.medication AS med_name, e.intermittent AS med_category, e.route AS med_route_name,
                   r.med_route_category, e.dose AS med_dose, e.unit AS med_dose_unit,
                   e.event_txt AS mar_action_name, 'given' AS mar_action_category
            FROM emar_dose e JOIN {routes} r ON r.route = e.route
            WHERE e.intermittent IS NOT NULL AND r.med_route_category <> ''
              AND coalesce(e.admin_type, '') NOT ILIKE '%infusion%'
              AND (NOT e.in_icu OR r.med_route_category IN ('enteral', 'buccal_sublingual', 'inhaled')
                   OR e.intermittent NOT IN ({icu_charted}))"""
        return sql, ("hospitalization_id", "admin_dttm", "med_category")

    # -- medication_admin_continuous (inputevents items CLIF-MIMIC leaves out) ----------------------
    def sql_medication_admin_continuous(self):
        """Infusions of ``inputevents`` items CLIF-MIMIC's mapping leaves out (``maps/mimic/continuous_items.csv``:
        CRRT potassium, calcium and citrate, IVIG, isoproterenol), as CLIF-MIMIC turns its own drips into rows:
        each record a start at ``starttime`` with its rate (amount / duration where none is charted) and a stop
        at ``endtime``, per order (``linkorderid``); S2b sums the orders into one rate stream per drug and unit.
        Bolus records (order category '05-Med Bolus', 'Drug Push' / 'Bolus') are eMAR doses; rewritten
        (erroneous) records are excluded, as in mimic-code."""
        items = self.map("continuous_items")
        bolus = """i.ordercategoryname = '05-Med Bolus' OR i.ordercategorydescription IN ('Drug Push', 'Bolus')
                   OR i.statusdescription = 'Bolus'"""
        sql = f"""
            WITH r AS (
                SELECT i.hadm_id, CAST(i.linkorderid AS VARCHAR) AS med_order_id, i.starttime, i.endtime,
                       d.label AS med_name, m.med_category, i.statusdescription,
                       coalesce(i.rate, i.amount / (epoch(i.endtime - i.starttime) / 60)) AS rate,
                       coalesce(i.rateuom, i.amountuom || '/min') AS unit,
                       concat_ws('; ', i.ordercategoryname, i.secondaryordercategoryname, i.ordercomponenttypedescription,
                                 i.ordercategorydescription) AS route_name
                FROM {self.src('icu/inputevents.parquet')} i
                JOIN {items} m ON CAST(m.itemid AS BIGINT) = i.itemid
                JOIN {self.src('icu/d_items.parquet')} d ON d.itemid = i.itemid
                WHERE i.hadm_id IS NOT NULL AND i.endtime > i.starttime AND NOT ({bolus})
                  AND coalesce(i.statusdescription, '') <> 'Rewritten'
                QUALIFY row_number() OVER (PARTITION BY i.hadm_id, i.linkorderid, i.itemid, i.starttime, i.endtime
                                           ORDER BY i.orderid) = 1)
            SELECT CAST(hadm_id AS VARCHAR) AS hospitalization_id, med_order_id, {_ts('starttime')} AS admin_dttm,
                   med_name, med_category, route_name AS med_route_name, 'iv' AS med_route_category,
                   rate AS med_dose, unit AS med_dose_unit, '[Started]' AS mar_action_name,
                   'start' AS mar_action_category
            FROM r WHERE rate IS NOT NULL
            UNION ALL
            SELECT CAST(hadm_id AS VARCHAR), med_order_id, {_ts('endtime')}, med_name, med_category, route_name, 'iv',
                   0.0, unit, statusdescription, 'stop'
            FROM r WHERE rate IS NOT NULL"""
        return sql, ("hospitalization_id", "admin_dttm", "med_category")

    def sql_input(self):
        """Intake items CLIF-MIMIC does not map (``maps/mimic/input_items.csv``, mL items only): the volume of each
        inputevents record at its end; rewritten (erroneous) records are excluded, as in mimic-code."""
        items = self.map("input_items")
        sql = f"""
            SELECT CAST(i.hadm_id AS VARCHAR) AS hospitalization_id, {_ts('i.endtime')} AS recorded_dttm,
                   m.label AS input_name, m.input_category, i.amount AS input_volume
            FROM {self.src('icu/inputevents.parquet')} i JOIN {items} m ON CAST(m.itemid AS BIGINT) = i.itemid
            WHERE i.hadm_id IS NOT NULL AND i.amount > 0 AND lower(i.amountuom) = 'ml'
              AND coalesce(i.statusdescription, '') <> 'Rewritten'"""
        return sql, ("hospitalization_id", "recorded_dttm")

    def sql_microbiology_nonculture(self):
        """PCR tests of the CLIF 3.0 organism panel (``maps/mimic/nonculture_tests.csv``): the panel organism is
        given by the test; the result by the reported organism or comment (the test's regexes). Tests without a
        result (cancelled, redacted) are dropped."""
        mi, nc = self._micro(), self.map("nonculture_tests")
        self.coverage("microbiology_nonculture", "result", f"""
            SELECT m.test_name || ': ' || coalesce(lower(m.name), '(no result)') AS txt, count(*) AS n,
                   bool_or(regexp_matches(coalesce(lower(m.name), ''), n.detected_regex)
                           OR regexp_matches(coalesce(lower(m.name), ''), n.not_detected_regex)) AS mapped
            FROM {mi} m JOIN {nc} n ON n.test_name = m.test_name GROUP BY 1""")
        sql = f"""
            SELECT * FROM (
                SELECT CAST(m.subject_id AS VARCHAR) AS patient_id, CAST(m.hadm_id AS VARCHAR) AS hospitalization_id,
                       {_ts('m.t_collect')} AS order_dttm, {_ts('m.t_collect')} AS collect_dttm,
                       {_ts('m.t_result')} AS result_dttm, m.spec_type_desc AS fluid_name, m.fluid_category,
                       m.test_name AS method_name, n.method_category, m.test_name AS micro_order_name,
                       n.organism_category, m.name AS result_name,
                       CASE WHEN regexp_matches(lower(m.name), n.detected_regex) THEN 'detected'
                            WHEN regexp_matches(lower(m.name), n.not_detected_regex) THEN 'not_detected'
                       END AS result_category
                FROM {mi} m JOIN {nc} n ON n.test_name = m.test_name)
            WHERE result_category IS NOT NULL"""
        return sql, ("hospitalization_id", "collect_dttm", "organism_category")

    def sql_microbiology_susceptibility(self):
        mi = self._micro()
        me = self.src("hosp/microbiologyevents.parquet")
        ab = self.regex_lookup("antimicrobials", f"SELECT lower(ab_name) AS txt FROM {me}", "antimicrobials",
                               ("antimicrobial_category",), table=self.lexicon("antimicrobials"))
        self.coverage("microbiology_susceptibility", "antibiotic", f"""
            SELECT lower(ab_name) AS txt, count(*) AS n, bool_or(a.antimicrobial_category IS NOT NULL) AS mapped
            FROM {me} e LEFT JOIN {ab} a ON a.txt = lower(e.ab_name) WHERE e.ab_name IS NOT NULL GROUP BY 1""")
        sql = f"""
            SELECT m.organism_id, e.ab_name AS antimicrobial_name, a.antimicrobial_category,
                   e.dilution_text AS sensitivity_name, e.interpretation AS susceptibility_name,
                   CASE e.interpretation WHEN 'S' THEN 'susceptible' WHEN 'R' THEN 'non_susceptible'
                        WHEN 'I' THEN 'non_susceptible' WHEN 'D' THEN 'indeterminate' ELSE 'na' END
                       AS susceptibility_category
            FROM {me} e
            JOIN {mi} m ON m.micro_specimen_id = e.micro_specimen_id AND m.test_itemid = e.test_itemid
                       AND m.org_itemid = e.org_itemid AND m.isolate_num IS NOT DISTINCT FROM e.isolate_num
            JOIN {ab} a ON a.txt = lower(e.ab_name)
            WHERE e.ab_name IS NOT NULL AND a.antimicrobial_category <> '__skip__'
              AND m.method_category = 'culture' AND m.organism_category IS NOT NULL"""
        return sql, ("organism_id", "antimicrobial_category")

    # -- vitals / invasive_hemodynamics (chartevents) -----------------------------------------------
    def _chart(self, map_name: str, category: str, name: str, value: str) -> str:
        m = self.map(map_name)
        return f"""
            SELECT CAST(c.hadm_id AS VARCHAR) AS hospitalization_id, {_ts('c.charttime')} AS recorded_dttm,
                   d.label AS {name}, m.{category}, c.valuenum AS {value}
            FROM {self.src('icu/chartevents.parquet')} c
            JOIN {m} m ON CAST(m.itemid AS BIGINT) = c.itemid
            LEFT JOIN {self.src('icu/d_items.parquet')} d ON d.itemid = c.itemid
            WHERE c.hadm_id IS NOT NULL AND c.valuenum IS NOT NULL"""

    def sql_vitals(self):
        return (self._chart("vitals", "vital_category", "vital_name", "vital_value"),
                ("hospitalization_id", "recorded_dttm", "vital_category"))

    def sql_invasive_hemodynamics(self):
        return (self._chart("hemodynamics", "measurement_category", "measurement_name", "measurement_value"),
                ("hospitalization_id", "recorded_dttm", "measurement_category"))
