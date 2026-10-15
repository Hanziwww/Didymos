"""AmsterdamUMCdb supplement to CLIF-AmsterdamUMCdb.

* ``ext_icu_stay`` from ``admissions``: an admission counts as an ICU stay when its location includes the
  ICU (IC, IC&MC, MC&IC; medium-care-only admissions are stepdown, as in the tool's ADT).
  - unit type: general_icu (Amsterdam UMC's ICU is a mixed medical-surgical ICU);
  - admission source: ``origin`` (``maps/aumc/origins.csv``; NULL before 2010);
  - admission diagnosis: AmsterdamUMCdb's official reason-for-admission concept (group | subgroup | diagnosis);
  - surgical status: the concept's diagnosis-based surgical flag, else ricu's AUMC ``adm`` specialty map;
    elective when ``urgency`` = 0;
  - ICU death: ``destination`` = 'Overleden'.
* ``labs``: items the tool does not map, mostly from ricu 0.6.3's AUMC concepts (``maps/aumc/labs.csv``). The
  tool's blood gases are re-emitted here with specimen routing (freetextitems 11646 / listitems 7432 via
  ``specimens.csv`` and ``gas_routing.csv``); the tool's own rows for those items are retracted in S2b.
  Base excess carries its sign in ``tag`` (ricu ``aumc_bxs``).
* ``vitals``: intracranial pressure; ``invasive_hemodynamics``: CVP, PA pressures, wedge, cardiac output, PiCCO.
* ``medication_admin_continuous`` rebuilt from drugitems (the tool drops the dose-rate time unit and emits no
  stops; its rows are retracted), ``medication_admin_intermittent`` for the items the tool does not map, and
  ``input`` (blood products, colloids, enteral nutrition, oral fluids), which the tool does not build.
* ``microbiology_culture``: culture orders as collected specimens without results (the source has none).

Times follow the tool exactly: epoch of the patient's first admission year group (2003-01-01 / 2010-01-01;
2000-01-01 otherwise) plus the source's milliseconds. AmsterdamUMCdb has no real time of day.
"""

from __future__ import annotations

import csv
import re
from importlib import resources

from icutwin.clif.dx_groups import group_sql
from icutwin.clif.supplements.base import ClifSupplement

EPOCHS = {"2003-2009": "2003-01-01", "2010-2016": "2010-01-01"}  # CLIF-AmsterdamUMCdb TimestampConfig.epoch_map
FALLBACK_EPOCH = "2000-01-01"
ICU_LOCATIONS = ("IC", "IC&MC", "MC&IC")


def _ts(ms: str) -> str:
    return f"(a.epoch + to_milliseconds(CAST({ms} AS BIGINT)))"


DAY_MS = 24 * 3600 * 1000   # admission documentation: entries within the first 24 h (header fields)


class AumcSupplement(ClifSupplement):
    dataset = "aumc"
    tables = ("ext_icu_stay", "labs", "vitals", "invasive_hemodynamics", "medication_admin_continuous",
              "medication_admin_intermittent", "input", "microbiology_culture")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        whens = " ".join(f"WHEN '{g}' THEN TIMESTAMPTZ '{d} 00:00:00+00'" for g, d in EPOCHS.items())
        self.con.execute(f"""
            CREATE OR REPLACE TABLE adm AS
            WITH first AS (SELECT patientid, any_value(admissionyeargroup) AS yeargroup
                           FROM {self.src('admissions.parquet')} WHERE admissioncount = 1 GROUP BY 1)
            SELECT x.*, CASE f.yeargroup {whens} ELSE TIMESTAMPTZ '{FALLBACK_EPOCH} 00:00:00+00' END AS epoch
            FROM {self.src('admissions.parquet')} x LEFT JOIN first f USING (patientid)""")

    def retractions(self) -> dict:
        """The tool's rows this supplement replaces: blood gases re-emitted with specimen routing (matched by
        its 'itemid: item' lab_name) and the whole continuous-medication table (rebuilt with rate units).

        Also dropped: the tool's second row of every blood-gas actual bicarbonate (9992), filed as BMP total CO2
        (``carbon_dioxide``, "Total CO2 ~ serum bicarbonate on BMP"). Total CO2 is a different quantity (HCO3
        plus dissolved CO2, about 1-2 mmol/L higher) that AmsterdamUMCdb does not measure; the proxy duplicates
        each bicarbonate value under a second category. The original row stays ``bicarbonate``."""
        text = resources.files("icutwin.clif").joinpath("maps", "aumc", "labs.csv").read_text(encoding="utf-8")
        ids = sorted({r["itemid"] for r in csv.DictReader(text.splitlines()) if r["gas"] and "tool item" in r["provenance"]})
        return {"labs": [{"when_regex": {"lab_name": f"^({'|'.join(map(re.escape, ids))}): "}},
                         {"when_regex": {"lab_name": "^9992: ", "lab_category": "^carbon_dioxide$"}}],
                "medication_admin_continuous": [{"when_regex": {}}]}

    def text_repairs(self) -> dict[str, str]:
        """CLIF-AmsterdamUMCdb reads the Latin-1 source CSVs as lossy UTF-8, so every non-ASCII character of its
        text (item names, list values, units: 'Geïntubeerd', 'µg') arrives as U+FFFD. The parquet mirror is
        decoded correctly: each source token with non-ASCII characters is mapped from its lossy form (the
        Latin-1 bytes decoded as UTF-8 with replacement) back to itself, when that form is unambiguous."""
        texts = {
            "numericitems.parquet": ("item", "unit"), "listitems.parquet": ("item", "value"),
            "drugitems.parquet": ("item", "doseunit", "rateunit", "administeredunit", "ordercategory"),
            "processitems.parquet": ("item",), "procedureorderitems.parquet": ("item",), "freetextitems.parquet": ("item",),
        }
        tokens: dict[str, set[str]] = {}
        for table, cols in texts.items():
            try:
                names = {r[0] for r in self.con.execute(f"DESCRIBE SELECT * FROM {self.src(table)}").fetchall()}
            except Exception:   # noqa: BLE001 - a table missing from an extract has nothing to repair
                continue
            for col in (c for c in cols if c in names):
                for (text,) in self.con.execute(f"SELECT DISTINCT {col} FROM {self.src(table)} "
                                                f"WHERE regexp_matches({col}, '[^\\x00-\\x7F]')").fetchall():
                    for tok in re.findall(r"\w+|[^\x00-\x7F]", text):
                        if any(ord(ch) > 127 for ch in tok):
                            try:
                                lossy = tok.encode("latin-1").decode("utf-8", errors="replace")
                            except UnicodeEncodeError:
                                continue
                            tokens.setdefault(lossy, set()).add(tok)
                            tokens.setdefault(lossy.lower(), set()).add(tok.lower())
        return {lossy: next(iter(orig)) for lossy, orig in sorted(tokens.items()) if len(orig) == 1}

    # -- ext_icu_stay -------------------------------------------------------------------------------
    def _reason_for_admission(self) -> str:
        """AmsterdamUMCdb's official reason-for-admission concept (sql/diagnosis/legacy/reason_for_admission.sql,
        MIT): per admission the diagnosis group (level 0), subgroup (1) and diagnosis (2), each the entry of the
        most specific source (NICE APACHE IV > NICE APACHE II > APACHE IV > APACHE II > ICU > medium care) last
        updated; NICE entries 'Group - Diagnosis' are split; level-2 items carry the surgical flag. Only entries
        measured and updated within 24 h of admission count (the official concept reads every entry: a median
        67 h after admission in the subset, revised as the stay evolves), so the header describes the admission."""
        dx = self.map("diagnosis_items")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE rfa AS
            WITH li AS (
                SELECT l.admissionid, l.itemid, l.item, l.value, l.valueid, l.updatedat, l.measuredat, m.level,
                       m.split_part AS part, CAST(m.priority AS INTEGER) AS prio, m.surgical,
                       CASE m.split_part WHEN '1' THEN split_part(l.value, ' - ', 1)
                                         WHEN '2' THEN split_part(l.value, ' - ', 2) ELSE l.value END AS txt
                FROM {self.src('listitems.parquet')} l JOIN {dx} m ON CAST(m.itemid AS BIGINT) = l.itemid
                JOIN adm a ON a.admissionid = l.admissionid
                WHERE l.measuredat <= a.admittedat + {DAY_MS} AND l.updatedat <= a.admittedat + {DAY_MS}),
            r AS (SELECT *, row_number() OVER (PARTITION BY admissionid, level
                                               ORDER BY prio DESC, updatedat DESC, measuredat DESC, valueid, itemid) AS rk
                  FROM li)
            SELECT admissionid,
                   any_value(txt) FILTER (WHERE level = '0' AND rk = 1) AS dx_group,
                   any_value(txt) FILTER (WHERE level = '1' AND rk = 1) AS dx_subgroup,
                   any_value(txt) FILTER (WHERE level = '2' AND rk = 1) AS dx,
                   -- organ-system text: a NICE entry's group, else the item ('APACHEIV Post-operative
                   -- cardiovascular', 'D_Interne Geneeskunde_Pulmonaal'); a subgroup with its specialty item
                   any_value(CASE WHEN part = '2' THEN split_part(value, ' - ', 1) ELSE item END)
                       FILTER (WHERE level = '2' AND rk = 1) AS dx_system_txt,
                   any_value(item || ' ' || txt) FILTER (WHERE level = '1' AND rk = 1) AS dx_subgroup_txt,
                   any_value(CASE WHEN surgical = '1' THEN 1 WHEN surgical = '0' THEN 0
                                  WHEN surgical = 'value:1-26' THEN CAST(valueid BETWEEN 1 AND 26 AS INTEGER)
                                  WHEN surgical = 'value:222-452' THEN CAST(valueid BETWEEN 222 AND 452 AS INTEGER) END)
                       FILTER (WHERE level = '2' AND rk = 1) AS dx_surgical
            FROM r GROUP BY 1""")
        return "rfa"

    def sql_ext_icu_stay(self):
        org, spec = self.map("origins"), self.map("specialties")
        rfa = self._reason_for_admission()
        locs = ", ".join(f"'{x}'" for x in ICU_LOCATIONS)
        self.coverage("ext_icu_stay", "origins", f"""
            SELECT a.origin AS txt, count(*) AS n, bool_or(o.origin IS NOT NULL) AS mapped
            FROM adm a LEFT JOIN {org} o ON o.origin = a.origin WHERE a.origin IS NOT NULL GROUP BY 1""")
        self.coverage("ext_icu_stay", "specialties", f"""
            SELECT a.specialty AS txt, count(*) AS n, bool_or(s.specialty IS NOT NULL) AS mapped
            FROM adm a LEFT JOIN {spec} s ON s.specialty = a.specialty WHERE a.specialty IS NOT NULL GROUP BY 1""")
        # admission diagnosis group: the system of the chosen diagnosis (else of its subgroup) by the shared
        # lexicon; a medical admission with sepsis at admission (items 15808 / 16675, as the official concept) or a
        # sepsis or intoxication diagnosis is that group
        lex = self.lexicon("dx_systems")
        lk = self.regex_lookup("dx_system", f"""
            SELECT lower(dx_system_txt) AS txt FROM {rfa} UNION ALL SELECT lower(dx_subgroup_txt) FROM {rfa}
            UNION ALL SELECT lower(dx) FROM {rfa}""", "dx_systems", ("system",), table=lex)
        self.coverage("ext_icu_stay", "dx_systems", f"""
            SELECT coalesce(x.dx_system_txt, x.dx_subgroup_txt) AS txt, count(*) AS n,
                   bool_or(coalesce(l2.system, l1.system) IS NOT NULL) AS mapped
            FROM {rfa} x LEFT JOIN {lk} l2 ON l2.txt = lower(x.dx_system_txt)
            LEFT JOIN {lk} l1 ON l1.txt = lower(x.dx_subgroup_txt)
            WHERE coalesce(x.dx_system_txt, x.dx_subgroup_txt) IS NOT NULL GROUP BY 1""")
        surgical = "coalesce(x.dx_surgical, CASE s.adm WHEN 'surg' THEN 1 WHEN 'med' THEN 0 WHEN 'other' THEN 0 END)"
        system = f"""CASE WHEN {surgical} = 0 AND (sp.sepsis OR nm.system = 'sepsis') THEN 'sepsis'
                          WHEN {surgical} = 0 AND nm.system = 'intoxication' THEN 'intoxication'
                          ELSE coalesce(l2.system, l1.system) END"""
        sql = f"""
            WITH sepsis AS (   -- sepsis at admission as documented 24 h after admission (the last update then)
                SELECT admissionid, valueid = 1 AS sepsis FROM (
                    SELECT l.*, row_number() OVER (PARTITION BY l.admissionid
                                                   ORDER BY l.updatedat DESC, l.measuredat DESC, l.valueid) AS k
                    FROM {self.src('listitems.parquet')} l JOIN adm x ON x.admissionid = l.admissionid
                    WHERE l.itemid IN (15808, 16675) AND l.measuredat <= x.admittedat + {DAY_MS}
                      AND l.updatedat <= x.admittedat + {DAY_MS})
                WHERE k = 1)
            SELECT 'aumc:' || CAST(a.admissionid AS VARCHAR) AS stay_id, CAST(a.patientid AS VARCHAR) AS patient_id,
                   CAST(a.admissionid AS VARCHAR) AS hospitalization_id, 'aumc' AS dataset, 'aumc' AS site_id,
                   {_ts('a.admittedat')} AS icu_in_dttm, {_ts('a.dischargedat')} AS icu_out_dttm,
                   'general_icu' AS unit_type, false AS clock_valid, true AS demographics_binned,
                   coalesce(o.admission_source, 'unknown') AS admission_source,
                   -- surgical: the official diagnosis flag when a diagnosis is recorded, else ricu's specialty map
                   CASE WHEN coalesce(x.dx_surgical, CASE s.adm WHEN 'surg' THEN 1 WHEN 'med' THEN 0
                                                                 WHEN 'other' THEN 0 END) IS NULL THEN 'unknown'
                        WHEN coalesce(x.dx_surgical, CASE WHEN s.adm = 'surg' THEN 1 ELSE 0 END) = 0 THEN 'nonsurgical'
                        WHEN a.urgency = 0 THEN 'elective_surgery'
                        ELSE 'emergency_surgery' END AS surgical_status,
                   CASE WHEN coalesce(x.dx_surgical, CASE WHEN s.adm = 'surg' THEN 1 ELSE 0 END) = 1
                        THEN json_object('specialty', a.specialty) END AS surgery_context,
                   nullif(concat_ws(' | ', x.dx_group, x.dx_subgroup, x.dx), '') AS adm_dx_name,
                   {group_sql(f"{surgical} = 1", system)} AS adm_dx_group,
                   CAST(row_number() OVER (PARTITION BY a.patientid ORDER BY a.admittedat, a.admissionid) AS INTEGER)
                       AS stay_seq,
                   CASE WHEN a.destination = 'Overleden' THEN 'dead' ELSE 'alive' END AS icu_outcome,
                   replace(a.admissionyeargroup, ' ', '') AS era
            FROM adm a
            LEFT JOIN {org} o ON o.origin = a.origin
            LEFT JOIN {spec} s ON s.specialty = a.specialty
            LEFT JOIN {rfa} x ON x.admissionid = a.admissionid
            LEFT JOIN {lk} l2 ON l2.txt = lower(x.dx_system_txt) LEFT JOIN {lk} l1 ON l1.txt = lower(x.dx_subgroup_txt)
            LEFT JOIN {lk} nm ON nm.txt = lower(x.dx)
            LEFT JOIN sepsis sp ON sp.admissionid = a.admissionid
            WHERE a.location IN ({locs})"""
        return sql, ("stay_id",)

    # -- labs ---------------------------------------------------------------------------------------
    def sql_labs(self):
        m, sp, gr = self.map("labs"), self.map("specimens"), self.map("gas_routing")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE lab_raw AS
            WITH sf AS (SELECT admissionid, measuredat, min(value) AS specimen
                        FROM {self.src('freetextitems.parquet')} WHERE itemid = 11646 GROUP BY 1, 2),
                 sl AS (SELECT admissionid, measuredat, min(value) AS specimen
                        FROM {self.src('listitems.parquet')} WHERE itemid = 7432 GROUP BY 1, 2)
            SELECT n.admissionid, n.itemid, n.item, n.tag, n.value, n.unit, n.measuredat, m.unit_in, m.lab_category,
                   CAST(m.factor AS DOUBLE) AS factor, m.reference_unit, m.lab_order_category, m.gas, m.sign_tag,
                   CASE m.specimen_source WHEN 'freetext' THEN sf.specimen WHEN 'list' THEN sl.specimen END AS specimen
            FROM {self.src('numericitems.parquet')} n
            JOIN {m} m ON CAST(m.itemid AS BIGINT) = n.itemid
            LEFT JOIN sf ON m.specimen_source = 'freetext' AND sf.admissionid = n.admissionid AND sf.measuredat = n.measuredat
            LEFT JOIN sl ON m.specimen_source = 'list' AND sl.admissionid = n.admissionid AND sl.measuredat = n.measuredat
            WHERE n.value IS NOT NULL""")
        unit_ok = "(r.unit_in = '*' OR r.unit = r.unit_in)"
        self.coverage("labs", "itemid_unit", f"""
            SELECT CAST(r.itemid AS VARCHAR) || ' [' || coalesce(r.unit, 'NULL') || ']' AS txt, count(*) AS n,
                   bool_or({unit_ok}) AS mapped FROM lab_raw r GROUP BY 1""")
        self.coverage("labs", "specimens", f"""
            SELECT r.specimen AS txt, count(*) AS n, bool_or(s.specimen IS NOT NULL) AS mapped
            FROM lab_raw r LEFT JOIN {sp} s ON s.specimen = r.specimen WHERE r.specimen IS NOT NULL GROUP BY 1""")
        value = "(CASE WHEN r.sign_tag = 'tag' AND r.tag = '-' THEN -r.value ELSE r.value END * r.factor)"
        sql = f"""
            SELECT CAST(r.admissionid AS VARCHAR) AS hospitalization_id,
                   {_ts('r.measuredat')} AS lab_order_dttm, {_ts('r.measuredat')} AS lab_collect_dttm,
                   {_ts('r.measuredat')} AS lab_result_dttm,
                   r.lab_order_category, CAST(r.itemid AS VARCHAR) || ': ' || r.item AS lab_name,
                   CASE WHEN r.gas IS NULL THEN r.lab_category ELSE g.lab_category END AS lab_category,
                   CAST({value} AS VARCHAR) AS lab_value, {value} AS lab_value_numeric,
                   r.reference_unit, r.specimen AS lab_specimen_name, 'plasma_blood' AS lab_specimen_category
            FROM lab_raw r
            JOIN adm a USING (admissionid)
            LEFT JOIN {sp} s ON s.specimen = r.specimen
            LEFT JOIN {gr} g ON g.gas = r.gas AND g.specimen_class = coalesce(s.specimen_class, 'unknown')
            WHERE {unit_ok} AND (r.gas IS NULL OR g.lab_category IS NOT NULL)"""
        return sql, ("hospitalization_id", "lab_collect_dttm", "lab_category")

    # -- medications and fluids (drugitems) ---------------------------------------------------------
    def _drugs(self) -> str:
        """drugitems with their order-category target and lexicon resolution (table ``drug``).

        CLIF-AmsterdamUMCdb writes the continuous dose without its time unit (0.1 'mg' for 0.1 mg per hour)
        and emits no stops. Each AmsterdamUMCdb record is a segment [start, stop) at a constant rate: ``dose``
        per ``doserateunit`` (uur / min / dag), per kg when ``doserateperkg`` = 1, as ricu's ``aumc_rate``
        reads it; records without a dose rate keep their volume rate (ml/uur).
        """
        if self.con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'drug'").fetchone()[0]:
            return "drug"
        di, oc = self.src("drugitems.parquet"), self.map("order_categories")
        lk = self.regex_lookup("drugs", f"SELECT lower(item) AS txt FROM {di}", "drugs",
                               ("ingredient", "continuous", "intermittent"), table=self.lexicon("drugs"))
        self.con.execute(f"""
            CREATE OR REPLACE TABLE drug AS
            SELECT d.*, o.target, o.input_category, l.ingredient, nullif(l.continuous, '') AS continuous,
                   CASE WHEN d.doserateunit IS NOT NULL AND d.dose IS NOT NULL THEN d.dose ELSE d.rate END AS seg_rate,
                   CASE WHEN d.doserateunit IS NOT NULL AND d.dose IS NOT NULL THEN
                            CASE d.doseunit WHEN 'µg' THEN 'mcg' WHEN 'IE' THEN 'u' WHEN 'E' THEN 'u' ELSE d.doseunit END
                            || CASE WHEN d.doserateperkg = 1 THEN '/kg' ELSE '' END || '/'
                            || CASE d.doserateunit WHEN 'uur' THEN 'hr' WHEN 'min' THEN 'min' WHEN 'dag' THEN 'day' END
                        WHEN d.rateunit = 'ml/uur' THEN 'ml/hr' END AS seg_unit
            FROM {di} d
            JOIN {oc} o ON o.ordercategory = d.ordercategory
            LEFT JOIN {lk} l ON l.txt = lower(d.item)""")
        self.coverage("medication_admin_continuous", "drug_name", """
            SELECT lower(item) AS txt, count(*) AS n, bool_or(continuous IS NOT NULL) AS mapped
            FROM drug WHERE target = 'continuous' AND iscontinuous = 1 GROUP BY 1""")
        return "drug"

    def sql_medication_admin_continuous(self):
        """Canonical start / dose_change / stop events: segment rates summed over concurrent syringes or bags of
        the same drug and unit (flushes are line care, not a rate); a stop when the summed rate falls to 0."""
        dr = self._drugs()
        sql = f"""
            WITH seg AS (
                SELECT d.admissionid, d.continuous, d.seg_unit AS unit, d.item, d.start, d.stop, d.seg_rate AS rate
                FROM {dr} d
                WHERE d.target = 'continuous' AND d.iscontinuous = 1 AND d.continuous IS NOT NULL
                  AND coalesce(d.action, '') <> 'Flush' AND d.seg_unit IS NOT NULL AND d.seg_rate > 0
                  AND d.stop > d.start),
            delta AS (
                SELECT admissionid, continuous, unit, start AS t, rate AS d, item FROM seg
                UNION ALL
                SELECT admissionid, continuous, unit, stop AS t, -rate AS d, item FROM seg),
            tot AS (
                SELECT admissionid, continuous, unit, t, min(item) AS item,
                       round(sum(sum(d)) OVER (PARTITION BY admissionid, continuous, unit ORDER BY t
                                               ROWS UNBOUNDED PRECEDING), 9) AS total
                FROM delta GROUP BY 1, 2, 3, 4),
            chg AS (SELECT *, lag(total, 1, 0) OVER (PARTITION BY admissionid, continuous, unit ORDER BY t) AS prev FROM tot)
            SELECT CAST(c.admissionid AS VARCHAR) AS hospitalization_id, {_ts('c.t')} AS admin_dttm,
                   c.item AS med_name, c.continuous AS med_category, 'iv' AS med_route_category,
                   greatest(c.total, 0) AS med_dose, c.unit AS med_dose_unit,
                   'rate summed over concurrent records' AS mar_action_name,
                   CASE WHEN c.total <= 0 THEN 'stop' WHEN c.prev <= 0 THEN 'start' ELSE 'dose_change' END
                       AS mar_action_category
            FROM chg c JOIN adm a USING (admissionid)
            WHERE c.total IS DISTINCT FROM c.prev"""
        return sql, ("hospitalization_id", "admin_dttm", "med_category")

    def sql_medication_admin_intermittent(self):
        """Non-continuous administrations of the items CLIF-AmsterdamUMCdb does not map (it maps 62 items,
        ``tool_intermittent_items.csv``; pantoprazole, electrolyte repletion, cefazolin, furosemide boluses, ...
        are missing), resolved with the lexicon. Route from the order category as the tool maps it
        (``intermittent_routes.csv``), except topical and nutrition categories, which have no CLIF intermittent
        route; doses in counting units keep their count in the English spelling of the other sources (Sachet ->
        sachet, druppel -> drop, Puff -> puff, Aantal -> each); a dose without a unit ('Geen') has no amount.
        A record given at a dose rate (``doserateunit`` set: phosphate over hours, a mannitol or amiodarone
        loading infusion) has ``dose`` per time unit (AmsterdamUMCdb drugitems documentation): its amount is
        ``administered``, in ``administeredunit`` (the tool's own 62 items keep ``dose``: 185 such records)."""
        di = self.src("drugitems.parquet")
        self._drugs()   # creates the lexicon lookup lk_drugs over every drugitems name
        routes, tool = self.map("intermittent_routes"), self.map("tool_intermittent_items")
        self.coverage("medication_admin_intermittent", "drug_name", f"""
            SELECT lower(d.item) AS txt, count(*) AS n, bool_or(nullif(l.intermittent, '') IS NOT NULL) AS mapped
            FROM {di} d LEFT JOIN lk_drugs l ON l.txt = lower(d.item)
            WHERE d.iscontinuous = 0 AND d.itemid NOT IN (SELECT CAST(itemid AS BIGINT) FROM {tool}) GROUP BY 1""")
        sql = f"""
            SELECT CAST(d.admissionid AS VARCHAR) AS hospitalization_id, {_ts('d.start')} AS admin_dttm,
                   CAST(d.itemid AS VARCHAR) || ': ' || d.item AS med_name, l.intermittent AS med_category,
                   d.ordercategory AS med_route_name, r.med_route_category,
                   CASE WHEN u.unit IS NOT NULL THEN CASE WHEN d.doserateunit IS NULL THEN d.dose ELSE d.administered END
                        END AS med_dose, u.unit AS med_dose_unit,
                   d.action AS mar_action_name, 'given' AS mar_action_category
            FROM {di} d
            JOIN adm a USING (admissionid)
            JOIN {routes} r ON CAST(r.ordercategoryid AS BIGINT) = d.ordercategoryid AND r.med_route_category <> ''
            JOIN lk_drugs l ON l.txt = lower(d.item) AND nullif(l.intermittent, '') IS NOT NULL
            LEFT JOIN (VALUES ('mg', 'mg'), ('g', 'g'), ('µg', 'mcg'), ('IE', 'u'), ('mmol', 'mmol'), ('ml', 'ml'),
                              ('Sachet', 'sachet'), ('druppel', 'drop'), ('Puff', 'puff'), ('Aantal', 'each'),
                              ('Flacon', 'vial'), ('fl', 'vial'), ('Dragee', 'tab'))
                 AS u(src, unit) ON u.src = CASE WHEN d.doserateunit IS NULL THEN d.doseunit ELSE d.administeredunit END
            WHERE d.iscontinuous = 0
              AND d.action IN ('Nieuwe toediening', 'Nieuwe portie', 'Herstart', 'Nieuwe zak', 'Nieuwe spuit')
              AND d.itemid NOT IN (SELECT CAST(itemid AS BIGINT) FROM {tool})"""
        return sql, ("hospitalization_id", "admin_dttm", "med_category")

    def sql_input(self):
        """Blood products, colloids, enteral nutrition and oral fluids: the volume given (``fluidin``) counted
        at the end of each record (crystalloids and parenteral nutrition are continuous medications)."""
        dr = self._drugs()
        sql = f"""
            SELECT CAST(d.admissionid AS VARCHAR) AS hospitalization_id, {_ts('coalesce(d.stop, d.start)')} AS recorded_dttm,
                   CAST(d.itemid AS VARCHAR) || ': ' || d.item AS input_name, d.input_category,
                   d.fluidin AS input_volume
            FROM {dr} d JOIN adm a USING (admissionid)
            WHERE d.target = 'input' AND d.fluidin > 0"""
        return sql, ("hospitalization_id", "recorded_dttm")

    # -- vitals / invasive_hemodynamics -------------------------------------------------------------
    def _numeric(self, map_name: str, category: str, name: str, value: str) -> str:
        m = self.map(map_name)
        self.coverage(map_name, "itemid_unit", f"""
            SELECT CAST(n.itemid AS VARCHAR) || ' [' || coalesce(n.unit, 'NULL') || ']' AS txt, count(*) AS n,
                   bool_or(m.unit_in = '*' OR n.unit = m.unit_in) AS mapped
            FROM {self.src('numericitems.parquet')} n JOIN {m} m ON CAST(m.itemid AS BIGINT) = n.itemid GROUP BY 1""")
        return f"""
            SELECT CAST(n.admissionid AS VARCHAR) AS hospitalization_id, {_ts('n.measuredat')} AS recorded_dttm,
                   CAST(n.itemid AS VARCHAR) || ': ' || n.item AS {name}, m.{category}, n.value AS {value}
            FROM {self.src('numericitems.parquet')} n
            JOIN {m} m ON CAST(m.itemid AS BIGINT) = n.itemid
            JOIN adm a USING (admissionid)
            WHERE n.value IS NOT NULL AND (m.unit_in = '*' OR n.unit = m.unit_in)"""

    def sql_vitals(self):
        return (self._numeric("vitals", "vital_category", "vital_name", "vital_value"),
                ("hospitalization_id", "recorded_dttm", "vital_category"))

    # -- microbiology_culture ------------------------------------------------------------------------
    def sql_microbiology_culture(self):
        """Culture orders (``procedureorderitems``; ``maps/aumc/culture_orders.csv``, ricu's AUMC ``samp``
        items): AmsterdamUMCdb releases no culture results, so an order is a collected specimen whose result is
        unknown (organism NULL, no result time), collected at the order's registration. Research cultures
        (19663, 19664) are study samples, not care."""
        m, po = self.map("culture_orders"), self.src("procedureorderitems.parquet")
        self.coverage("microbiology_culture", "culture_orders", f"""
            SELECT CAST(p.itemid AS VARCHAR) || ': ' || p.item AS txt, count(*) AS n, bool_or(m.itemid IS NOT NULL) AS mapped
            FROM {po} p LEFT JOIN {m} m ON CAST(m.itemid AS BIGINT) = p.itemid
            WHERE lower(p.item) LIKE '%kweek%' GROUP BY 1""")
        sql = f"""
            SELECT CAST(a.patientid AS VARCHAR) AS patient_id, CAST(p.admissionid AS VARCHAR) AS hospitalization_id,
                   CAST(p.orderid AS VARCHAR) AS order_id, CAST(p.orderid AS VARCHAR) AS organism_id,
                   {_ts('p.registeredat')} AS order_dttm, {_ts('p.registeredat')} AS collect_dttm,
                   CAST(NULL AS TIMESTAMPTZ) AS result_dttm, p.item AS fluid_name, m.fluid_category,
                   p.item AS method_name, 'culture' AS method_category, CAST(NULL AS VARCHAR) AS organism_name,
                   CAST(NULL AS VARCHAR) AS organism_category, CAST(NULL AS VARCHAR) AS organism_group
            FROM {po} p JOIN {m} m ON CAST(m.itemid AS BIGINT) = p.itemid JOIN adm a USING (admissionid)"""
        return sql, ("hospitalization_id", "collect_dttm", "organism_id")

    def sql_invasive_hemodynamics(self):
        return (self._numeric("hemodynamics", "measurement_category", "measurement_name", "measurement_value"),
                ("hospitalization_id", "recorded_dttm", "measurement_category"))
