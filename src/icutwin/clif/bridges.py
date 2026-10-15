"""Per-dataset bridges for the S3 reconciliation (``icutwin.clif.reconcile``).

A bridge reads the same raw rows twice: as the community MEDS ETL coded them (``ev``: subject_id, time, code,
numeric_value, text_value, ...) and as our CLIF tables hold them (``t_<table>``, before harmonization). Items are
the source items (MIMIC/NWICU item ids, eICU/HiRID/SICdb/INSPIRE/AmsterdamUMCdb item names). Measurement bridges
restrict the community side to the items the CLIF mapping selects; medication bridges take every record and
explain the rest by the adapters' rules and the shared drug lexicon. So every unpaired row is a row one pipeline
kept and the other did not. The reasons name the documented rules that drop or add rows; each is a SQL
predicate over the unpaired row ``u`` (columns of the bridge's SELECT).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from icutwin.clif.adapters.eicu import INFUSION_BASE, INFUSION_UNIT, RATE_UNIT_RE, STOP_GAP_MIN
from icutwin.clif.adapters.hirid import ROUTE as HIRID_ROUTE
from icutwin.clif.adapters.nwicu import RATE_UNIT as NWICU_RATE_UNIT
from icutwin.clif.adapters.nwicu import ROUTE as NWICU_ROUTE
from icutwin.clif.adapters.sicdb import DOSE_UNIT as SICDB_DOSE_UNIT
from icutwin.clif.adapters.sicdb import RATE_UNIT as SICDB_RATE_UNIT
from icutwin.clif.conform import medication_extensions
from icutwin.clif.lexicon import rows as lexicon_rows
from icutwin.clif.reconcile import _F32, _SIG5, SCOPE_BEFORE_ADMISSION, Bridge, Streams


@dataclass(frozen=True)
class DatasetBridges:
    community: str                       # MEDS root under the community output dir
    bridges: list[Bridge]
    setup: tuple[str, ...] = ()
    scope: str | None = None             # SQL (pid, t0, t1) when the bridges pair on another time axis
    streams: tuple[Streams, ...] = ()    # rate streams compared per (pid, drug/unit) where rows cannot pair


def _q(p: Path) -> str:
    return str(p).replace("\\", "/")


def _pq(p: Path) -> str:
    return f"read_parquet('{_q(p)}')"


def _csv(p: Path) -> str:
    # explicit RFC 4180 dialect, non-strict: DuckDB's strict sniffing rejects some maps whose quoted text fields
    # hold commas (Python's csv module, which the adapters use, reads them unchanged)
    return (f"""read_csv('{_q(p)}', header = true, all_varchar = true, delim = ',', quote = '"', escape = '"', """
            "strict_mode = false)")


def _med_name(code: str, trailing: int = 1) -> str:
    """The drug of a community ``MEDICATION//<name>//<field>...`` code with ``trailing`` fields after the name
    (NWICU: the event; MIMIC: the event and a route), split from the right: a name can end in '/' ('INV
    enoxaparin 75 mg/'), which a split at every ``//`` cuts short."""
    return f"regexp_extract({code}, '^MEDICATION//(.*)" + "//[^/]*" * trailing + "$', 1)"


def _med_event(code: str, trailing: int = 1) -> str:
    """The first field after the drug name (the event) of such a code."""
    return f"regexp_extract({code}, '//([^/]*)" + "//[^/]*" * (trailing - 1) + "$', 1)"


def _hosp(table: str) -> str:
    return f"t_{table} x JOIN t_hospitalization h ON h.hospitalization_id = x.hospitalization_id"


def _same(a: str, b: str, rtol: float = 1e-4) -> str:
    return f"abs({a} - {b}) <= {rtol} * greatest(1, abs({a}))"


def _sq(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def lexicon_setup(names_sql: str) -> tuple[str, str]:
    """``lk(txt, ingredient, continuous, intermittent)``: the shared drug lexicon resolved for the names of
    ``names_sql`` (column ``txt``, lower-cased) exactly as the adapters resolve them (lowest priority, then file
    order, RE2 partial match)."""
    values = ", ".join(f"({int(r['priority'])}, {i}, {_sq(r['pattern'])}, {_sq(r['ingredient'])}, "
                       f"{_sq(r['continuous'] or '')}, {_sq(r['intermittent'] or '')})"
                       for i, r in enumerate(lexicon_rows("drugs")))
    return (f"""CREATE OR REPLACE TEMP TABLE lex AS SELECT * FROM (VALUES {values})
                    t(priority, idx, pattern, ingredient, continuous, intermittent)""",
            f"""CREATE OR REPLACE TEMP TABLE lk AS
                SELECT txt, ingredient, nullif(continuous, '') AS continuous, nullif(intermittent, '') AS intermittent
                FROM (SELECT d.txt, l.*, row_number() OVER (PARTITION BY d.txt ORDER BY l.priority, l.idx) AS k
                      FROM (SELECT DISTINCT txt FROM ({names_sql}) WHERE txt IS NOT NULL) d
                      LEFT JOIN lex l ON regexp_matches(d.txt, l.pattern))
                WHERE k = 1""")


def regex_map_setup(table: str, path: Path, cols: tuple[str, ...], names_sql: str) -> tuple[str, str]:
    """``<table>(txt, *cols)``: a dataset's ordered regex map (``priority``, ``pattern``, ``cols``) resolved for
    the names of ``names_sql`` (column ``txt``) as the adapters' ``regex_lookup`` resolves them (lowest
    priority, then file order, RE2 partial match; unmatched names get NULLs)."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    values = ", ".join(f"({int(r['priority']) if r['priority'] else 'NULL'}, {i}, {_sq(r['pattern'])}, "
                       + ", ".join(_sq(r[c]) for c in cols) + ")" for i, r in enumerate(rows))
    picked = ", ".join(f"m.{c}" for c in cols)
    return (f"""CREATE OR REPLACE TEMP TABLE {table}_map AS SELECT * FROM (VALUES {values})
                    t(priority, idx, pattern, {', '.join(cols)})""",
            f"""CREATE OR REPLACE TEMP TABLE {table} AS
                SELECT txt, {', '.join(cols)} FROM (
                    SELECT d.txt, {picked}, row_number() OVER (PARTITION BY d.txt ORDER BY m.priority NULLS LAST, m.idx) AS k
                    FROM (SELECT DISTINCT txt FROM ({names_sql}) WHERE txt IS NOT NULL) d
                    LEFT JOIN {table}_map m ON regexp_matches(d.txt, m.pattern))
                WHERE k = 1""")


# a community record of a drug without a CLIF category for this administration type (lexicon)
NO_INTERMITTENT_CATEGORY = ("no_intermittent_category",
                            "(SELECT intermittent FROM lk WHERE lk.txt = lower(u.item)) IS NULL")

def _merged_records(table: str) -> tuple[str, str]:
    """A rate-stream key where ours exceeds the community by the integral of the raw records the community
    merged into an identical event (``table``: pid, key, t0, t1, rate of those records), within 0.1%. The integral
    is taken inside the comparison's windows (``scope_u``), as the difference is: a record can run past a discharge."""
    return ("community_merges_identical_records",
            f"""k.o_total > k.c_total AND abs(k.diff - coalesce((
                    SELECT sum(m.rate * epoch(least(m.t1, w.t1) - greatest(m.t0, w.t0)) / 60)
                    FROM {table} m JOIN scope_u w ON w.pid = m.pid AND w.t0 < m.t1 AND w.t1 > m.t0
                    WHERE m.pid = k.pid AND m.key = k.key), 0))
                <= 1e-3 * greatest(k.c_total, k.o_total)""")


# a community event and our row that hold the same value at the same key, where the community merged identical
# events (MEDS-Extract and MEDS-Transforms keep one row per distinct event; we keep both raw rows)
MERGED_IDENTICAL = ("community_merges_identical_events",
                    """EXISTS (SELECT 1 FROM o_all o2 JOIN pairs p ON p.o_rid = o2.rid
                               WHERE p.c_rid IS NOT NULL AND o2.pid = u.pid AND o2.t = u.t AND o2.item = u.item
                                 AND o2.value IS NOT DISTINCT FROM u.value)""")


# -- MIMIC-IV (CLIF-MIMIC + our supplement) ----------------------------------------------------------------
# a source row whose time lies outside its own admission (MIMIC labevents and eMAR rows with a hadm_id years from that
# admission: one patient at full scale): its community copy is out of every window; the MEDS stay windows leave ours out
SOURCE_TIME_OUTSIDE_ITS_ADMISSION = ("source_time_outside_its_admission",
                                     f"u.t < u.admission_dttm - {SCOPE_BEFORE_ADMISSION} OR u.t > u.discharge_dttm")
def mimic(mirror: Path, refs: Path, maps: Path) -> DatasetBridges:
    """MIMIC_IV_MEDS: ``LAB//RESULT//<itemid>//<unit>`` (labevents, at storetime), ``LAB//<itemid>//<unit>``
    (chartevents, at charttime), ``SUBJECT_FLUID_OUTPUT//<itemid>//<unit>`` (outputevents). The item universes are
    CLIF-MIMIC's mapping CSVs (the decisions its builders select) plus our supplement maps; our rows carry the
    item label, so community item ids are matched by label (labels shared by several selected items pool)."""
    def tool(t: str) -> str:
        return _csv(refs / "CLIF-MIMIC" / "data" / "mappings" / f"mimic-to-clif-mappings - {t}.csv")

    def ours(t: str) -> str:
        return _csv(maps / f"{t}.csv")

    setup = (
        f"""CREATE OR REPLACE TEMP VIEW labels AS
            SELECT itemid, label FROM {_pq(mirror / 'hosp/d_labitems.parquet')}
            UNION ALL SELECT itemid, label FROM {_pq(mirror / 'icu/d_items.parquet')}""",
        """CREATE OR REPLACE TEMP TABLE ev_i AS
           SELECT *, CASE WHEN code LIKE 'LAB//RESULT//%' THEN TRY_CAST(split_part(code, '//', 3) AS BIGINT)
                          WHEN code LIKE 'LAB//SPECIMEN%' THEN NULL
                          ELSE TRY_CAST(split_part(code, '//', 2) AS BIGINT) END AS itemid
           FROM ev""",
        # labs: CLIF-MIMIC's selected decisions (5-digit ids from labevents, 6-digit from chartevents) + ours
        f"""CREATE OR REPLACE TEMP TABLE lab_items AS
            SELECT DISTINCT itemid, CASE WHEN itemid < 100000 THEN 'le' ELSE 'ce' END AS src FROM (
                SELECT TRY_CAST(itemid AS BIGINT) AS itemid FROM {tool('labs')}
                WHERE decision IN ('TO MAP, CONVERT UOM', 'TO MAP, AS IS', 'UNSURE')
                UNION SELECT TRY_CAST(itemid AS BIGINT) FROM {ours('labs')}) WHERE itemid IS NOT NULL""",
        """CREATE OR REPLACE TEMP TABLE lab_src AS
           SELECT DISTINCT l.label, i.src FROM lab_items i JOIN labels l USING (itemid)""",
        # the CLIF categories (and unit factors) an item can land in: the mappings plus patch 0002's specimen routing
        f"""CREATE OR REPLACE TEMP TABLE lab_item_cats AS
            SELECT TRY_CAST(itemid AS BIGINT) AS itemid, lab_category,
                   CASE WHEN decision = 'TO MAP, CONVERT UOM' THEN CAST(conversion_multiplier AS DOUBLE) ELSE 1 END
                       AS factor
            FROM {tool('labs')} WHERE decision IN ('TO MAP, CONVERT UOM', 'TO MAP, AS IS', 'UNSURE') AND itemid IS NOT NULL
            UNION ALL SELECT * FROM (VALUES (50818, 'pco2_venous', 1.0), (50820, 'ph_venous', 1.0),
                                            (50817, 'so2_mixed_venous', 1.0), (50817, 'so2_central_venous', 1.0))
            UNION ALL SELECT TRY_CAST(itemid AS BIGINT), lab_category, CAST(factor AS DOUBLE) FROM {ours('labs')}""",
        # labs without hadm_id that the ED rule (CLIF-MIMIC patch 0003, our supplement) assigns to no admission: the
        # draw is outside every ED stay of the patient. Keyed as the community events are (patient, item, store
        # time), since two draws of one item can share a store time (one in the ED stay, one before it)
        f"""CREATE OR REPLACE TEMP TABLE lab_outside_ed AS
            SELECT DISTINCT CAST(l.subject_id AS VARCHAR) AS pid, l.itemid, l.storetime
            FROM {_pq(mirror / 'hosp/labevents.parquet')} l
            WHERE l.hadm_id IS NULL
              AND NOT EXISTS (SELECT 1 FROM {_pq(mirror / 'hosp/admissions.parquet')} a
                              WHERE a.subject_id = l.subject_id AND l.charttime BETWEEN a.edregtime AND a.admittime)""",
        f"""CREATE OR REPLACE TEMP TABLE bg_specimen AS
            WITH le AS (SELECT * FROM {_pq(mirror / 'hosp/labevents.parquet')} WHERE itemid IN (50817, 50821, 52033)),
                 sp AS (SELECT specimen_id, any_value(value) AS specimen_type FROM le WHERE itemid = 52033 GROUP BY 1)
            SELECT DISTINCT CAST(l.subject_id AS VARCHAR) AS pid, l.itemid, l.storetime, sp.specimen_type
            FROM le l JOIN sp USING (specimen_id) WHERE l.itemid IN (50817, 50821)""",
        f"""CREATE OR REPLACE TEMP TABLE vital_items AS
            SELECT DISTINCT TRY_CAST(itemid AS BIGINT) AS itemid FROM {tool('vitals')}
            WHERE vital_category NOT IN ('NO MAPPING', 'UNSURE', 'MAPPED ELSEWHERE', 'SPECIAL CASE', 'ALREADY MAPPED',
                                         'temp_c')
            UNION SELECT * FROM (VALUES (223761), (223762))
            UNION SELECT TRY_CAST(itemid AS BIGINT) FROM {ours('vitals')}""",
        f"""CREATE OR REPLACE TEMP TABLE vital_item_cats AS
            SELECT TRY_CAST(itemid AS BIGINT) AS itemid, vital_category FROM {tool('vitals')}
            WHERE vital_category NOT IN ('NO MAPPING', 'UNSURE', 'MAPPED ELSEWHERE', 'SPECIAL CASE', 'ALREADY MAPPED')
            UNION ALL SELECT TRY_CAST(itemid AS BIGINT), vital_category FROM {ours('vitals')}""",
        f"""CREATE OR REPLACE TEMP TABLE output_items AS
            SELECT TRY_CAST(itemid AS BIGINT) AS itemid FROM {tool('output')} WHERE decision = 'TO MAP'""",
    )
    labs = Bridge(
        "labs",
        """SELECT e.pid, e.time AS t, l.label AS item, e.numeric_value AS value, e.code AS unit_key, e.itemid, i.src,
                  e.hadm_id
           FROM ev_i e JOIN lab_items i USING (itemid) JOIN labels l USING (itemid) WHERE e.code LIKE 'LAB//%'""",
        f"""SELECT h.patient_id AS pid, CASE WHEN s.src = 'ce' THEN x.lab_collect_dttm ELSE x.lab_result_dttm END AS t,
                   x.lab_name AS item, x.lab_value_numeric AS value, x.lab_category, x.lab_collect_dttm,
                   x.lab_result_dttm, h.admission_dttm, h.discharge_dttm
            FROM {_hosp('labs')} LEFT JOIN lab_src s ON s.label = x.lab_name""",
        explain_community=(
            # a draw left out (outside every ED stay, no hadm_id) shares the row's item and store time; text results
            # (no value) of one store time cannot be told apart, so the pair may have gone to the outpatient row
            ("outpatient_no_admission",
             "EXISTS (SELECT 1 FROM lab_outside_ed e WHERE e.pid = u.pid AND e.itemid = u.itemid AND e.storetime = u.t)"),
            # CLIF-MIMIC keeps one row per (hospitalization, collect, result, category, value): a labevents result
            # and its ICU chartevents copy (or two items of one category) collapse
            ("same_category_value_and_time",
             f"""EXISTS (SELECT 1 FROM o_all o JOIN lab_item_cats k ON k.lab_category = o.lab_category
                        WHERE k.itemid = u.itemid AND o.pid = u.pid
                          AND (CASE WHEN u.src = 'ce' THEN o.lab_collect_dttm ELSE o.lab_result_dttm END) = u.t
                          AND {_same('o.value', 'u.value * k.factor')})"""),
            # patch 0002: venous / central-venous pO2 and peripheral-venous SO2 have no CLIF 3.0 category
            ("venous_gas_without_clif_category",
             """EXISTS (SELECT 1 FROM bg_specimen s WHERE s.pid = u.pid AND s.itemid = u.itemid AND s.storetime = u.t
                        AND ((u.itemid = 50821 AND s.specimen_type IN ('VEN.', 'CENTRAL VENOUS.'))
                             OR (u.itemid = 50817 AND s.specimen_type = 'VEN.')))"""),
            ("unit_not_in_map",
             f"""u.itemid IN (SELECT TRY_CAST(itemid AS BIGINT) FROM {ours('labs')} WHERE unit_in <> '*')
                 AND NOT EXISTS (SELECT 1 FROM {ours('labs')} m WHERE TRY_CAST(m.itemid AS BIGINT) = u.itemid
                                 AND lower(trim(split_part(u.unit_key, '//', 4))) = lower(m.unit_in))"""),
        ),
        explain_ours=(
            MERGED_IDENTICAL,
            # a source row whose time lies outside its own admission (labevents with a hadm_id years from that
            # admission: 240 rows of one patient at full scale); its community copy is out of every window, and
            # the MEDS stay windows leave it out
            SOURCE_TIME_OUTSIDE_ITS_ADMISSION,
            # draws of one item stored at one time without a `value` (the result in the comment, or the comment
            # masked) are identical community events, of which MEDS keeps one; ours keeps each draw's result, one
            # paired with that event (#97: ammonia '<10.' and '___', both stored 13:05)
            ("community_merges_valueless_results_of_one_store_time",
             """u.t = u.lab_result_dttm AND EXISTS (SELECT 1 FROM o_all o2 JOIN pairs p ON p.o_rid = o2.rid
                                                    JOIN c_all c ON c.rid = p.c_rid
                                                    WHERE o2.pid = u.pid AND o2.t = u.t AND o2.item = u.item
                                                      AND o2.rid <> u.rid AND c.value IS NULL)"""),
        ))
    vitals = Bridge(
        "vitals",
        """SELECT e.pid, e.time AS t, l.label AS item,
                  CASE e.itemid WHEN 223761 THEN round((e.numeric_value - 32) * 5 / 9, 1)
                                WHEN 226531 THEN round(e.numeric_value / 2.205, 1) ELSE e.numeric_value END AS value,
                  e.itemid
           FROM ev_i e JOIN vital_items USING (itemid) JOIN labels l USING (itemid) WHERE e.code LIKE 'LAB//%'""",
        f"""SELECT h.patient_id AS pid, x.recorded_dttm AS t, x.vital_name AS item, x.vital_value AS value,
                   x.vital_category
            FROM {_hosp('vitals')}""",
        explain_community=(
            # CLIF-MIMIC: Celsius charted at the same time wins over Fahrenheit
            ("celsius_charted_at_same_time",
             "u.itemid = 223761 AND EXISTS (SELECT 1 FROM o_all o WHERE o.pid = u.pid AND o.t = u.t "
             "AND o.item = 'Temperature Celsius')"),
            # CLIF-MIMIC drops duplicates per (hospitalization, time, category, value) across items
            ("same_category_value_and_time",
             f"""EXISTS (SELECT 1 FROM o_all o JOIN vital_item_cats k ON k.vital_category = o.vital_category
                        WHERE k.itemid = u.itemid AND o.pid = u.pid AND o.t = u.t AND {_same('o.value', 'u.value')})"""),
        ))
    output = Bridge(
        "output",
        """SELECT e.pid, e.time AS t, l.label AS item, e.numeric_value AS value
           FROM ev_i e JOIN output_items USING (itemid) JOIN labels l USING (itemid)
           WHERE e.code LIKE 'SUBJECT_FLUID_OUTPUT//%'""",
        f"""SELECT h.patient_id AS pid, x.recorded_dttm AS t, x.output_name AS item, x.output_volume AS value
            FROM {_hosp('output')}""")
    setup += (
        # one community record per inputevents row: its INFUSION_START (rate) and INFUSION_END (amount) events share
        # the item and order id; routed as CLIF-MIMIC routes the raw row (mapping decision; for 'BOTH' items the
        # bolus order categories; five items over 1 min to the continuous table)
        f"""CREATE OR REPLACE TEMP TABLE med_items AS
            SELECT TRY_CAST(itemid AS BIGINT) AS itemid, decision FROM {tool('med_category')}
            WHERE decision IN ('BOTH', 'CONTINUOUS', 'INTERMITTENT')""",
        f"""CREATE OR REPLACE TEMP TABLE ie AS
            WITH s AS (SELECT pid, CAST(split_part(code, '//', 2) AS BIGINT) AS itemid, order_id, link_order_id,
                              time AS starttime, numeric_value AS rate
                       FROM ev WHERE code LIKE 'INFUSION_START//%'),
                 e AS (SELECT pid, CAST(split_part(code, '//', 2) AS BIGINT) AS itemid, order_id, time AS endtime,
                              numeric_value AS amount
                       FROM ev WHERE code LIKE 'INFUSION_END//%'),
                 r AS (SELECT CAST(subject_id AS VARCHAR) AS pid, itemid, orderid AS order_id, ordercategoryname,
                              ordercategorydescription, statusdescription
                       FROM {_pq(mirror / 'icu/inputevents.parquet')})
            SELECT s.*, e.endtime, e.amount, l.label, m.decision, epoch(e.endtime - s.starttime) / 60 AS minutes,
                   r.statusdescription, (r.ordercategoryname = '05-Med Bolus' OR r.statusdescription = 'Bolus'
                                         OR r.ordercategorydescription IN ('Drug Push', 'Bolus')) AS bolus,
                   s.itemid IN (SELECT TRY_CAST(itemid AS BIGINT) FROM {ours('continuous_items')}) AS supplement_item,
                   CASE WHEN m.decision IS NULL THEN NULL
                        WHEN m.decision = 'INTERMITTENT' OR (m.decision = 'BOTH' AND (r.ordercategoryname = '05-Med Bolus'
                             OR r.ordercategorydescription IN ('Drug Push', 'Bolus') OR r.statusdescription = 'Bolus'))
                        THEN CASE WHEN epoch(e.endtime - s.starttime) / 60 > 1
                                       AND s.itemid IN (222168, 225158, 220949, 221668, 225942) THEN 'cont' ELSE 'intm' END
                        ELSE 'cont' END AS to_table,
                   s.itemid IN (SELECT TRY_CAST(itemid AS BIGINT) FROM {ours('input_items')}) AS input_item
            FROM s JOIN e USING (pid, itemid, order_id)
            LEFT JOIN r USING (pid, itemid, order_id)
            LEFT JOIN (SELECT DISTINCT itemid, label FROM labels) l USING (itemid)
            LEFT JOIN med_items m USING (itemid)""",
        # eMAR: the inputs of our supplement's rules per event (route of the administration row, else the pharmacy
        # order's; administration type; inside an ICU stay)
        *lexicon_setup(f"SELECT lower({_med_name('code', 2)}) AS txt FROM ev WHERE emar_id IS NOT NULL"),
        f"CREATE OR REPLACE TEMP TABLE emar_routes AS SELECT * FROM {ours('emar_routes')}",
        # categories whose ICU parenteral doses MetaVision charts in inputevents (the supplement's rule)
        f"""CREATE OR REPLACE TEMP TABLE tool_intm_cats AS
            SELECT med_category FROM {ours('tool_intermittent_categories')} UNION
            SELECT med_category FROM {ours('tool_continuous_categories')}
            WHERE med_category IN ({', '.join(_sq(c) for c in _infused_local_intermittent())})""",
        f"""CREATE OR REPLACE TEMP TABLE emar_x AS
            WITH d AS (SELECT emar_id, any_value(route) FILTER (WHERE parent_field_ordinal IS NULL) AS route,
                              any_value(administration_type) FILTER (WHERE parent_field_ordinal IS NULL) AS admin_type
                       FROM {_pq(mirror / 'hosp/emar_detail.parquet')} GROUP BY 1)
            SELECT e.emar_id, upper(trim(coalesce(d.route, ph.route))) AS route, d.admin_type,
                   EXISTS (SELECT 1 FROM {_pq(mirror / 'icu/icustays.parquet')} s
                           WHERE s.hadm_id = e.hadm_id AND e.charttime BETWEEN s.intime AND s.outtime) AS in_icu
            FROM {_pq(mirror / 'hosp/emar.parquet')} e LEFT JOIN d USING (emar_id)
            LEFT JOIN (SELECT DISTINCT ON (pharmacy_id) pharmacy_id, route FROM {_pq(mirror / 'hosp/pharmacy.parquet')}) ph
                   ON ph.pharmacy_id = e.pharmacy_id""",
    )
    # CLIF-MIMIC: one intermittent row per routed record (start time, amount)
    med_intermittent = Bridge(
        "medication_admin_intermittent",
        """SELECT pid, starttime AS t, label AS item, amount AS value, link_order_id AS pair_key, to_table, minutes,
                  decision, input_item FROM ie""",
        f"""SELECT h.patient_id AS pid, x.admin_dttm AS t, x.med_name AS item, x.med_dose AS value,
                   x.med_order_id AS pair_key
            FROM {_hosp('medication_admin_intermittent')} WHERE x._src = 'tool'""",
        explain_community=(
            ("fluid_or_nutrition_item_input_table", "u.decision IS NULL AND u.input_item"),
            ("item_outside_clif_mimic_medication_mapping", "u.decision IS NULL"),
            ("duration_not_positive_dropped_by_clif_mimic", "u.minutes <= 0"),
            ("continuous_record_compared_in_the_continuous_bridge", "u.to_table = 'cont'"),
        ),
        name="medication_admin_intermittent (inputevents)")
    # CLIF-MIMIC: each continuous record flattened to its start (rate, or amount / duration without one) and its end
    # (stop rows carry 0: compared by presence); one row per (order, time), so a record's end at the order's next
    # start collapses into that start
    med_continuous = Bridge(
        "medication_admin_continuous",
        """SELECT pid, starttime AS t, label AS item, coalesce(rate, amount / minutes) AS value, 'start' AS role,
                  link_order_id AS pair_key, link_order_id, order_id, starttime, endtime
           FROM ie WHERE to_table = 'cont' AND minutes > 0
           UNION ALL
           SELECT pid, endtime, label, NULL, 'end', link_order_id, link_order_id, order_id, starttime, endtime
           FROM ie WHERE to_table = 'cont' AND minutes > 0""",
        f"""SELECT h.patient_id AS pid, x.admin_dttm AS t, x.med_name AS item,
                   CASE WHEN x.mar_action_category = 'stop' THEN NULL ELSE x.med_dose END AS value,
                   x.med_order_id AS pair_key
            FROM {_hosp('medication_admin_continuous')} WHERE x._src = 'tool'""",
        explain_community=(
            ("record_end_at_the_orders_next_start",
             """u.role = 'end' AND EXISTS (SELECT 1 FROM c_all c WHERE c.pid = u.pid AND c.item = u.item AND c.t = u.t
                                           AND c.role = 'start' AND c.link_order_id = u.link_order_id)"""),
            ("duplicate_record_same_order_start_and_end",
             """EXISTS (SELECT 1 FROM c_all c WHERE c.pid = u.pid AND c.item = u.item AND c.link_order_id = u.link_order_id
                        AND c.starttime = u.starttime AND c.endtime = u.endtime AND c.order_id <> u.order_id)"""),
            # the other way round: CLIF-MIMIC keeps one action per (order, category, time), chosen by its
            # mar_action_dedup mapping, and where a record starts as another of its order ends (a 'Bolus' record
            # among them, mostly) the mapping can keep the end's action or neither (108 starts at full scale)
            ("record_start_at_the_orders_previous_end",
             """u.role = 'start' AND EXISTS (SELECT 1 FROM c_all c WHERE c.pid = u.pid AND c.item = u.item AND c.t = u.t
                                             AND c.role = 'end' AND c.link_order_id = u.link_order_id)"""),
        ),
        name="medication_admin_continuous (inputevents)")
    # our supplement: eMAR 'Administered' / 'Delayed Administered' events (no dose on the community side)
    emar = Bridge(
        "medication_admin_intermittent",
        f"""SELECT e.pid, e.time AS t, {_med_name('e.code', 2)} AS item, CAST(NULL AS DOUBLE) AS value,
                  {_med_event('e.code', 2)} AS event_txt, {_med_event('e.code', 2)} || '|' || coalesce(r.med_route_category, '')
                      -- an infusion's administration is never a dose of ours: it must not take a dose's pair (no
                      -- dose on the community side; a heparin infusion start and bolus at one minute, full scale)
                      || CASE WHEN coalesce(x.admin_type, '') ILIKE '%infusion%' THEN '|infusion' ELSE '' END
                      AS pair_key, e.hadm_id, x.admin_type, x.in_icu, r.med_route_category
           FROM ev e LEFT JOIN emar_x x USING (emar_id) LEFT JOIN emar_routes r ON r.route = x.route
           WHERE e.emar_id IS NOT NULL""",
        f"""SELECT h.patient_id AS pid, x.admin_dttm AS t, x.med_name AS item, CAST(NULL AS DOUBLE) AS value,
                   x.mar_action_name || '|' || coalesce(x.med_route_category, '') AS pair_key,
                   h.admission_dttm, h.discharge_dttm
            FROM {_hosp('medication_admin_intermittent')} WHERE x._src = 'icutwin'""",
        explain_community=(
            ("not_an_administration_event", "u.event_txt NOT IN ('Administered', 'Delayed Administered')"),
            ("no_hospital_admission", "u.hadm_id IS NULL"),
            NO_INTERMITTENT_CATEGORY,
            ("route_not_systemic_or_not_mapped", "coalesce(u.med_route_category, '') = ''"),
            ("infusion_administration", "coalesce(u.admin_type, '') ILIKE '%infusion%'"),
            ("icu_parenteral_dose_inputevents_charts",
             """u.in_icu AND u.med_route_category NOT IN ('enteral', 'buccal_sublingual', 'inhaled')
                AND (SELECT intermittent FROM lk WHERE lk.txt = lower(u.item)) IN (SELECT med_category FROM tool_intm_cats)"""),
        ),
        explain_ours=(MERGED_IDENTICAL, SOURCE_TIME_OUTSIDE_ITS_ADMISSION),
        name="medication_admin_intermittent (eMAR)")
    # our supplement: drips of the inputevents items CLIF-MIMIC leaves out, one start (rate) and one stop per record
    supplement_continuous = Bridge(
        "medication_admin_continuous",
        """SELECT pid, starttime AS t, label AS item, coalesce(rate, amount / minutes) AS value, 'start' AS role,
                  link_order_id AS pair_key, link_order_id, order_id, starttime, endtime, statusdescription, bolus
           FROM ie WHERE supplement_item
           UNION ALL
           SELECT pid, endtime, label, NULL, 'end', link_order_id, link_order_id, order_id, starttime, endtime,
                  statusdescription, bolus
           FROM ie WHERE supplement_item""",
        f"""SELECT h.patient_id AS pid, x.admin_dttm AS t, x.med_name AS item,
                   CASE WHEN x.mar_action_category = 'stop' THEN NULL ELSE x.med_dose END AS value,
                   x.med_order_id AS pair_key
            FROM {_hosp('medication_admin_continuous')} WHERE x._src = 'icutwin'""",
        explain_community=(
            ("bolus_record_given_as_emar_dose", "u.bolus"),
            ("rewritten_record_excluded_as_in_mimic_code", "u.statusdescription = 'Rewritten'"),
            ("duration_not_positive", "u.endtime <= u.starttime"),
            ("duplicate_record_same_order_start_and_end",
             """EXISTS (SELECT 1 FROM c_all c WHERE c.pid = u.pid AND c.item = u.item AND c.link_order_id = u.link_order_id
                        AND c.starttime = u.starttime AND c.endtime = u.endtime AND c.order_id <> u.order_id)"""),
        ),
        name="medication_admin_continuous (inputevents, supplement)")
    return DatasetBridges("mimic", [labs, vitals, output, med_intermittent, med_continuous, emar, supplement_continuous],
                          setup)


def _infused_local_intermittent() -> list[str]:
    """Local intermittent extensions of drugs CLIF 3.0 gives as infusions (``untyped_iv``)."""
    return sorted({r["med_category"] for r in medication_extensions()
                   if r["table"] == "medication_admin_intermittent" and r["untyped_iv"] == "continuous"})


# -- NWICU (our adapter) -----------------------------------------------------------------------------------
def nwicu(mirror: Path, refs: Path, maps: Path) -> DatasetBridges:
    """NWICU_MEDS: labevents and chartevents both as ``LAB//<itemid>//<unit>`` at charttime (lab item ids
    1000xx, chart item ids 3200xx). Our labs pair on the collection time."""
    d = mirror / "data"
    setup = (
        f"""CREATE OR REPLACE TEMP VIEW labels AS
            SELECT DISTINCT ON (itemid) itemid, label FROM (
                SELECT itemid, label FROM {_pq(d / 'nw_hosp/d_labitems.parquet')}
                UNION ALL SELECT itemid, label FROM {_pq(d / 'nw_icu/d_items.parquet')}) ORDER BY itemid, label""",
        """CREATE OR REPLACE TEMP TABLE ev_i AS
           SELECT *, TRY_CAST(split_part(code, '//', 2) AS BIGINT) AS itemid FROM ev WHERE code LIKE 'LAB//%'""",
        f"""CREATE OR REPLACE TEMP TABLE lab_items AS
            SELECT CAST(itemid AS BIGINT) AS itemid, CAST(factor AS DOUBLE) AS factor FROM {_csv(maps / 'labs.csv')}""",
        f"""CREATE OR REPLACE TEMP TABLE chart_items AS
            SELECT CAST(itemid AS BIGINT) AS itemid, target, CAST(factor AS DOUBLE) AS factor,
                   CAST("offset" AS DOUBLE) AS "offset"
            FROM {_csv(maps / 'chart_items.csv')}""",
    )
    no_value = ("no_numeric_value", "u.value IS NULL")
    labs = Bridge(
        "labs",
        """SELECT e.pid, e.time AS t, l.label AS item, e.numeric_value AS value, e.code AS unit_key, e.hadm_id
           FROM ev_i e JOIN lab_items USING (itemid) JOIN labels l USING (itemid)""",
        f"""SELECT h.patient_id AS pid, x.lab_collect_dttm AS t, x.lab_name AS item, x.lab_value_numeric AS value
            FROM {_hosp('labs')}""",
        explain_community=(
            no_value,
            ("text_result_sentinel", "u.value >= 9999999"),
            ("outpatient_no_admission", "u.hadm_id IS NULL"),
        ),
        # a result whose source row has no charttime (storetime only; 15 at full scale): the community has no time
        # to place it at, and the MEDS build leaves it out
        explain_ours=(MERGED_IDENTICAL, ("no_collection_time", "u.t IS NULL")))
    vitals = Bridge(
        "vitals",
        # the map's affine conversion (degF -> degC) applied before comparing
        """SELECT e.pid, e.time AS t, l.label AS item, (e.numeric_value + c."offset") * c.factor AS value,
                  e.code AS unit_key, e.hadm_id
           FROM ev_i e JOIN chart_items c USING (itemid) JOIN labels l USING (itemid) WHERE c.target = 'vitals'""",
        f"""SELECT h.patient_id AS pid, x.recorded_dttm AS t, x.vital_name AS item, x.vital_value AS value
            FROM {_hosp('vitals')}""",
        explain_community=(no_value, ("no_admission", "u.hadm_id IS NULL")),
        explain_ours=(MERGED_IDENTICAL,))
    # eMAR 'Applied' events (``MEDICATION//<medication>//<event_txt>``, with the emar_id): why the adapter does not
    # emit one as an intermittent dose, by its own rules in order
    setup += (
        f"""CREATE OR REPLACE TEMP TABLE emar_why AS
            SELECT emar_id, CASE
                       WHEN hadm_id IS NULL THEN 'no_admission'
                       WHEN regexp_matches(coalesce(dose_unit_rx, ''), '{NWICU_RATE_UNIT}') THEN 'infusion_order'
                       WHEN ({NWICU_ROUTE}) IS NULL THEN 'route_without_clif_category'
                       WHEN dose_unit_rx IS NULL THEN 'no_dose_unit_administration_type_unknown' END AS why
            FROM (SELECT e.emar_id, e.hadm_id, p.dose_unit_rx, p.dose_val_rx, p.route,
                         CAST(NULL AS VARCHAR) AS ingredient   -- decides only between two route categories
                  FROM {_pq(d / 'nw_hosp/emar.parquet')} e
                  LEFT JOIN (SELECT DISTINCT ON (pharmacy_id) * FROM {_pq(d / 'nw_hosp/prescriptions.parquet')}) p
                         USING (pharmacy_id))""",
    )
    setup += lexicon_setup(f"""SELECT lower({_med_name('code')}) AS txt FROM ev
                               WHERE code LIKE 'MEDICATION//%//Applied' OR code LIKE 'MEDICATION//%//Confirmed'""")

    def emar_reason(why: str) -> tuple[str, str]:
        return (why, f"EXISTS (SELECT 1 FROM emar_why w WHERE w.emar_id = u.emar_id AND w.why = '{why}')")

    # continuous: 'Confirmed' / 'Applied' events of orders whose dose unit is a rate (no rate is recorded: presence);
    # ours keep one row per (admission, drug category, time), named by the first medication, plus inferred stops
    setup += (
        f"""CREATE OR REPLACE TEMP TABLE emar_cont AS
            SELECT e.emar_id, e.hadm_id, regexp_matches(coalesce(p.dose_unit_rx, ''), '{NWICU_RATE_UNIT}') AS is_infusion
            FROM {_pq(d / 'nw_hosp/emar.parquet')} e
            LEFT JOIN (SELECT DISTINCT ON (pharmacy_id) * FROM {_pq(d / 'nw_hosp/prescriptions.parquet')}) p
                   USING (pharmacy_id)""",
    )
    infusions = Bridge(
        "medication_admin_continuous",
        f"""SELECT e.pid, e.time AS t, {_med_name('e.code')} AS item, CAST(NULL AS DOUBLE) AS value, e.hadm_id,
                  x.is_infusion, l.continuous,
                  CASE WHEN x.is_infusion THEN 'infusion order' ELSE 'other order' END AS pair_key
           FROM ev e LEFT JOIN emar_cont x USING (emar_id) LEFT JOIN lk l ON l.txt = lower({_med_name('e.code')})
           WHERE e.code LIKE 'MEDICATION//%//Confirmed' OR e.code LIKE 'MEDICATION//%//Applied'""",
        f"""SELECT h.patient_id AS pid, x.admin_dttm AS t, CAST(NULL AS DOUBLE) AS value,
                   CASE WHEN x.mar_action_name LIKE 'inferred stop%' THEN x.med_name || ' [inferred stop]'
                        ELSE x.med_name END AS item, x.mar_action_name, 'infusion order' AS pair_key
            FROM {_hosp('medication_admin_continuous')}""",
        explain_community=(
            ("no_admission", "u.hadm_id IS NULL"),
            ("not_an_infusion_order", "NOT coalesce(u.is_infusion, false)"),
            ("no_continuous_category", "u.continuous IS NULL"),
            ("one_row_per_drug_category_and_time",
             """EXISTS (SELECT 1 FROM c_all c WHERE c.pid = u.pid AND c.t = u.t AND c.continuous = u.continuous
                        AND c.is_infusion AND c.rid <> u.rid AND c.item <= u.item)"""),
        ),
        explain_ours=(("inferred_stop_not_confirmed_within_6_h", "u.mar_action_name LIKE 'inferred stop%'"),
                      MERGED_IDENTICAL),
        name="medication_admin_continuous (eMAR)")

    medication = Bridge(
        "medication_admin_intermittent",
        # the community events carry no dose, so events of one drug at one time are interchangeable in pairing: an
        # event the adapter's rules leave out (pair key 'not_emitted') must not take the pair of the one it emits
        # (an order without a dose unit charted at the minute of a dosed one, full scale)
        f"""SELECT e.pid, e.time AS t, {_med_name('e.code')} AS item, CAST(NULL AS DOUBLE) AS value, e.emar_id,
                  CASE WHEN w.why IS NULL AND (SELECT intermittent FROM lk WHERE lk.txt = lower({_med_name('e.code')}))
                            IS NOT NULL THEN '' ELSE 'not_emitted' END AS pair_key
           FROM ev e LEFT JOIN emar_why w USING (emar_id) WHERE e.code LIKE 'MEDICATION//%//Applied'""",
        f"""SELECT h.patient_id AS pid, x.admin_dttm AS t, x.med_name AS item, CAST(NULL AS DOUBLE) AS value,
                   '' AS pair_key
            FROM {_hosp('medication_admin_intermittent')}""",
        explain_community=(emar_reason("no_admission"), emar_reason("infusion_order"), NO_INTERMITTENT_CATEGORY,
                           emar_reason("route_without_clif_category"),
                           emar_reason("no_dose_unit_administration_type_unknown")),
        explain_ours=(MERGED_IDENTICAL,))
    return DatasetBridges("nwicu", [labs, vitals, medication, infusions], setup)


# -- INSPIRE (our adapter) ---------------------------------------------------------------------------------
# INSPIRE dates admissions and discharges to the day, so a same-day discharge and readmission gives two
# hospitalization windows that share that day; a ward record of the day belongs to both (in MEDS each copy falls in
# a different stay's window, as the pre-ICU window allows): the copy beyond the paired one is explained
IN_TWO_HOSPITALIZATIONS = (
    "record_in_two_overlapping_hospitalizations",
    """(SELECT count(*) FROM hwin w WHERE w.pid = u.pid AND u.t BETWEEN w.t0 AND w.t1) > 1
       AND EXISTS (SELECT 1 FROM o_all o2 JOIN pairs p ON p.o_rid = o2.rid
                   WHERE p.c_rid IS NOT NULL AND o2.pid = u.pid AND o2.t = u.t AND o2.item = u.item
                     AND o2.value IS NOT DISTINCT FROM u.value)""")


def inspire(mirror: Path, refs: Path, maps: Path) -> DatasetBridges:
    """INSPIRE_MEDS: ``VITAL//<item>//<unit>`` (intra-operative ``vitals``), ``WARD_VITAL//<item>//<unit>``,
    ``LAB//<item>//<unit>``, ``WEIGHT`` / ``HEIGHT`` (operations, at admission); times on the shared origin
    2016-01-01 + minutes, as ours. Ward and lab rows attach to the hospitalization whose window holds them."""
    setup = (
        """CREATE OR REPLACE TEMP TABLE ev_i AS
           SELECT *, split_part(code, '//', 1) AS fam, split_part(code, '//', 2) AS item_name,
                  CASE split_part(code, '//', 1) WHEN 'VITAL' THEN 'vitals' WHEN 'WARD_VITAL' THEN 'ward_vitals' END
                      AS source_table
           FROM ev""",
        f"CREATE OR REPLACE TEMP TABLE vmap AS SELECT * FROM {_csv(maps / 'vitals.csv')}",
        f"CREATE OR REPLACE TEMP TABLE lmap AS SELECT * FROM {_csv(maps / 'labs.csv')}",
        f"CREATE OR REPLACE TEMP TABLE iomap AS SELECT * FROM {_csv(maps / 'io.csv')}",
        """CREATE OR REPLACE TEMP TABLE hwin AS
           SELECT CAST(patient_id AS VARCHAR) AS pid, admission_dttm AS t0, discharge_dttm AS t1 FROM t_hospitalization""",
    )
    no_value = ("no_numeric_value", "u.value IS NULL")
    outside = ("outside_hospital_window",
               "u.fam <> 'VITAL' AND NOT EXISTS (SELECT 1 FROM hwin w WHERE w.pid = u.pid AND u.t BETWEEN w.t0 AND w.t1)")
    vitals = Bridge(
        "vitals",
        """SELECT e.pid, e.time AS t, e.item_name AS item, e.numeric_value * CAST(m.factor AS DOUBLE) AS value, e.fam
           FROM ev_i e JOIN vmap m USING (source_table, item_name)
           UNION ALL
           SELECT pid, time, lower(code), numeric_value, 'OPERATIONS' FROM ev_i WHERE code IN ('WEIGHT', 'HEIGHT')""",
        f"""SELECT h.patient_id AS pid, x.recorded_dttm AS t, x.vital_name AS item, x.vital_value AS value
            FROM {_hosp('vitals')}""",
        explain_community=(no_value, outside),
        explain_ours=(MERGED_IDENTICAL,))
    labs = Bridge(
        "labs",
        """SELECT e.pid, e.time AS t, e.item_name AS item, e.numeric_value * CAST(m.factor AS DOUBLE) AS value, e.fam
           FROM ev_i e JOIN lmap m USING (item_name) WHERE e.fam = 'LAB'""",
        f"""SELECT h.patient_id AS pid, x.lab_collect_dttm AS t, x.lab_name AS item, x.lab_value_numeric AS value
            FROM {_hosp('labs')}""",
        explain_community=(no_value, outside),
        explain_ours=(MERGED_IDENTICAL,))

    def io(table: str, direction: str) -> Bridge:
        return Bridge(
            table,
            f"""SELECT e.pid, e.time AS t, e.item_name AS item, e.numeric_value * CAST(m.factor AS DOUBLE) AS value, e.fam
                FROM ev_i e JOIN iomap m USING (source_table, item_name) WHERE m.direction = '{direction}'""",
            f"""SELECT h.patient_id AS pid, x.recorded_dttm AS t, x.{table}_name AS item, x.{table}_volume AS value
                FROM {_hosp(table)}""",
            explain_community=(("no_positive_volume", "u.value IS NULL OR u.value <= 0"), outside),
            explain_ours=(MERGED_IDENTICAL,))

    resp_items = {"fio2": "fio2_set * 100", "peep": "peep_obs", "vt": "tidal_volume_obs",
                  "pip": "peak_inspiratory_pressure_obs", "pmean": "mean_airway_pressure_obs",
                  "pplat": "plateau_pressure_obs", "minvol": "minute_vent_obs"}
    resp = Bridge(
        "respiratory_support",
        f"""SELECT e.pid, e.time AS t, e.item_name AS item, e.numeric_value AS value, e.fam FROM ev_i e
            WHERE (e.fam = 'VITAL' AND e.item_name IN ({", ".join(f"'{k}'" for k in resp_items)}))
               OR (e.fam = 'WARD_VITAL' AND e.item_name = 'fio2')""",
        f"""UNPIVOT (SELECT h.patient_id AS pid, x.recorded_dttm AS t,
                            {", ".join(f"{v} AS {k}" for k, v in resp_items.items())}
                     FROM {_hosp('respiratory_support')})
            ON {", ".join(resp_items)} INTO NAME item VALUE value""",
        explain_community=(
            no_value, outside,
            # the adapter keeps one value per item and time (max) when an item is recorded twice at one time
            ("max_per_item_and_time",
             "EXISTS (SELECT 1 FROM o_all o WHERE o.pid = u.pid AND o.t = u.t AND o.item = u.item AND o.value >= u.value)"),
        ),
        explain_ours=(IN_TWO_HOSPITALIZATIONS,))
    # ward and ICU medication records (``MEDICATION//<atc>//<route>``, drug name as text; no dose): our ward records
    # are the intermittent doses and the presence events of infusion-only drugs, per (time, drug, route)
    setup += (
        f"CREATE OR REPLACE TEMP TABLE combos AS SELECT * FROM {_csv(maps / 'combinations.csv')}",
        """CREATE OR REPLACE TEMP TABLE wmed AS
           SELECT h.patient_id AS pid, x.admin_dttm AS t, x.med_name AS nm, x.med_route_name AS route, x.med_category
           FROM t_medication_admin_intermittent x JOIN t_hospitalization h USING (hospitalization_id)
           WHERE x.mar_action_name = 'ward medication record (dose not recorded)'
           UNION ALL
           SELECT h.patient_id, x.admin_dttm, x.med_name, 'iv', x.med_category
           FROM t_medication_admin_continuous x JOIN t_hospitalization h USING (hospitalization_id)
           WHERE x.mar_action_name = 'ward medication record (rate not recorded)'""",
        f"""CREATE OR REPLACE TEMP TABLE op_win AS
            SELECT CAST(subject_id AS VARCHAR) AS pid, TIMESTAMP '2016-01-01' + to_minutes(CAST(orin_time AS BIGINT)) AS t0,
                   TIMESTAMP '2016-01-01' + to_minutes(CAST(orout_time AS BIGINT)) AS t1
            FROM {_pq(mirror / 'operations.parquet')} WHERE orin_time IS NOT NULL AND orout_time IS NOT NULL""",
    )
    setup += lexicon_setup("SELECT lower(trim(text_value)) AS txt FROM ev_i WHERE fam = 'MEDICATION'")
    ward_meds = Bridge(
        "medication_admin_intermittent",
        """SELECT e.pid, e.time AS t, lower(trim(e.text_value)) || ' (' || split_part(e.code, '//', 3) || ')' AS item,
                  CAST(NULL AS DOUBLE) AS value, e.fam, lower(trim(e.text_value)) AS nm, split_part(e.code, '//', 3) AS route
           FROM ev_i e WHERE e.fam = 'MEDICATION'""",
        "SELECT pid, t, nm || ' (' || route || ')' AS item, CAST(NULL AS DOUBLE) AS value, nm, route FROM wmed",
        explain_community=(
            ("route_without_clif_category", "u.route = 'ex'"),
            outside,
            # a combination product is released as one row per component: the pair is one administration
            ("combination_component",
             """EXISTS (SELECT 1 FROM combos c JOIN o_all o ON o.item = c.product || ' (' || u.route || ')'
                        WHERE u.nm IN (c.component_a, c.component_b) AND o.pid = u.pid AND o.t = u.t)"""),
            ("no_clif_category",
             "(SELECT coalesce(intermittent, continuous) FROM lk WHERE lk.txt = u.nm) IS NULL"),
            # the source repeats a drug at one time and route under name variants ('metformin ', 'metformin'): the
            # adapter keeps one administration, paired with one of them
            ("same_drug_recorded_twice_at_one_time",
             "EXISTS (SELECT 1 FROM o_all o WHERE o.pid = u.pid AND o.t = u.t AND o.item = u.item)"),
            # a drug CLIF lists as continuous only is an infusion presence when intravenous, nothing otherwise
            ("continuous_only_drug_not_iv",
             "u.route <> 'iv' AND (SELECT intermittent IS NULL AND continuous IS NOT NULL FROM lk WHERE lk.txt = u.nm)"),
            # inside an operation, drugs the anaesthesia record (intra-operative vitals) doses are left to it
            ("dosed_by_the_anaesthesia_record",
             f"""EXISTS (SELECT 1 FROM op_win w WHERE w.pid = u.pid AND u.t BETWEEN w.t0 AND w.t1)
                 AND (SELECT coalesce(intermittent, continuous) FROM lk WHERE lk.txt = u.nm)
                     IN (SELECT med_category FROM {_csv(maps / 'meds_intraop.csv')})"""),
        ),
        explain_ours=(
            IN_TWO_HOSPITALIZATIONS,
            ("combination_of_community_components",
             """EXISTS (SELECT 1 FROM combos c WHERE u.nm = c.product
                        AND EXISTS (SELECT 1 FROM c_all a WHERE a.pid = u.pid AND a.t = u.t AND a.route = u.route
                                    AND a.nm = c.component_a)
                        AND EXISTS (SELECT 1 FROM c_all b WHERE b.pid = u.pid AND b.t = u.t AND b.route = u.route
                                    AND b.nm = c.component_b))"""),
        ))
    # intra-operative drugs are items of the vitals stream: rates (ours: one row per rate change within an admission)
    # and injected doses (ours: doses above 0)
    setup += (f"CREATE OR REPLACE TEMP TABLE intra_meds AS SELECT * FROM {_csv(maps / 'meds_intraop.csv')}",)

    def intraop(kind: str, explain: tuple[tuple[str, str], ...]) -> Bridge:
        table = f"medication_admin_{kind}"
        return Bridge(
            table,
            f"""SELECT e.pid, e.time AS t, e.item_name AS item, e.numeric_value AS value,
                       lag(e.numeric_value) OVER (PARTITION BY e.pid, e.item_name ORDER BY e.time) AS prev
                FROM ev_i e WHERE e.fam = 'VITAL'
                  AND e.item_name IN (SELECT item_name FROM intra_meds WHERE clif_table = '{kind}')""",
            f"""SELECT h.patient_id AS pid, x.admin_dttm AS t, x.med_name AS item, x.med_dose AS value
                FROM {_hosp(table)} WHERE x.med_name IN (SELECT item_name FROM intra_meds WHERE clif_table = '{kind}')
                  AND coalesce(x.mar_action_name, '') NOT LIKE 'ward medication record%'""",
            explain_community=(no_value, *explain),
            explain_ours=(MERGED_IDENTICAL,),
            name=f"{table} (intra-operative)")

    intra_rates = intraop("continuous", (
        ("rate_unchanged_since_the_previous_charting", "u.value = u.prev"),
        # our stream is the admission's (#95): a zero charted before anything ran in its admission starts nothing
        ("zero_before_any_start_in_its_admission",
         """u.value = 0 AND NOT EXISTS (
                SELECT 1 FROM c_all c2 JOIN t_hospitalization h ON CAST(h.patient_id AS VARCHAR) = c2.pid
                WHERE c2.pid = u.pid AND c2.item = u.item AND c2.value > 0 AND c2.t < u.t
                  AND c2.t >= h.admission_dttm AND u.t <= h.discharge_dttm)"""),
    ))
    intra_doses = intraop("intermittent", (("no_positive_dose", "NOT u.value > 0"),))
    return DatasetBridges("inspire/MEDS_cohort",
                          [vitals, labs, io("input", "input"), io("output", "output"), resp, ward_meds, intra_rates,
                           intra_doses], setup)


# -- SICdb (our adapter) -----------------------------------------------------------------------------------
REL0 = "TIMESTAMP '2000-01-01'"


def sicdb(mirror: Path, refs: Path, maps: Path) -> DatasetBridges:
    """SICdb_MEDS: ``VITAL//<ReferenceValue>//<unit>`` (``data_float_h``), ``LAB//LOINC//<code>//<unit>``. It
    places every row at the patient's anchor (1 July of the first case's admission year) + ``Offset``, without
    the case's ``OffsetAfterFirstAdmission``: a later case of a patient lands on the first case's timeline.
    Rows therefore pair on the time since their case start (ours: since ``admission_dttm``; community: since the
    anchor). Our hourly signals are stamped at the end of their hour (Offset + 59 min), so community signal times
    move by the same 3540 s. Admission weight and height are not paired: the community stamps them at the
    anchor + ``AdmissionYear`` seconds (its ``cases`` pseudotime columns include the year)."""
    ref = _pq(mirror / "d_references.parquet")
    signals = _csv(maps / "signals.csv")
    setup = (
        f"""CREATE OR REPLACE TEMP TABLE anchor AS
            SELECT CAST(PatientID AS VARCHAR) AS pid,
                   make_timestamp(CAST(min(AdmissionYear) AS BIGINT), 7, 1, 0, 0, 0) AS a
            FROM {_pq(mirror / 'cases.parquet')} GROUP BY 1""",
        f"""CREATE OR REPLACE TEMP TABLE sig AS
            SELECT DISTINCT m.target, m.category, r.ReferenceValue AS nm
            FROM {signals} m JOIN {ref} r ON r.ReferenceGlobalID = CAST(m.dataid AS BIGINT)""",
        # the community keys lab rows by (LOINC, unit), 'UNK' for a missing part; our mapped items pool per key
        # (two keys hold a central-lab and a blood-gas item), and a key can also hold items we do not map
        f"""CREATE OR REPLACE TEMP TABLE lab_ref AS
            SELECT ReferenceGlobalID AS id, ReferenceValue AS nm, coalesce(LOINC_code, 'UNK') AS loinc,
                   coalesce(ReferenceUnit, 'UNK') AS unit,
                   ReferenceGlobalID IN (SELECT CAST(laboratoryid AS BIGINT) FROM {_csv(maps / 'labs.csv')}) AS mapped
            FROM {ref}""",
        """CREATE OR REPLACE TEMP TABLE lab_key AS
           SELECT nm, loinc, unit, string_agg(nm, ' | ' ORDER BY nm) OVER (PARTITION BY loinc, unit) AS item
           FROM lab_ref WHERE mapped""",
        f"""CREATE OR REPLACE TEMP TABLE lab_raw AS
            SELECT CAST(c.PatientID AS VARCHAR) AS pid, {REL0} + to_seconds(CAST(x.Offset AS BIGINT)) AS t,
                   r.id, r.loinc, r.unit, r.mapped, x.LaboratoryValue AS v
            FROM {_pq(mirror / 'laboratory.parquet')} x JOIN {_pq(mirror / 'cases.parquet')} c USING (CaseID)
            JOIN lab_ref r ON r.id = x.LaboratoryID""",
        # signal names: 'RespRate' names both the set (2282) and the measured (719) rate; columns sharing a
        # name pool into one item
        """CREATE OR REPLACE TEMP TABLE resp_pool AS
           SELECT a.category, string_agg(DISTINCT b.category, ' | ' ORDER BY b.category) AS item
           FROM sig a JOIN sig b ON a.nm = b.nm WHERE a.target LIKE 'resp%' AND b.target LIKE 'resp%'
           GROUP BY a.category""",
        f"""CREATE OR REPLACE TEMP TABLE ev_i AS
            SELECT e.*, split_part(e.code, '//', 1) AS fam, split_part(e.code, '//', 2) AS nm,
                   {REL0} + (e.time - a.a) + CASE WHEN e.code LIKE 'VITAL//%' THEN INTERVAL 3540 SECOND
                                                  ELSE INTERVAL 0 SECOND END AS t_rel
            FROM ev e JOIN anchor a USING (pid)""",
    )
    scope = f"""SELECT CAST(patient_id AS VARCHAR) AS pid, {REL0} - {SCOPE_BEFORE_ADMISSION} AS t0,
                       {REL0} + (discharge_dttm - admission_dttm) AS t1 FROM t_hospitalization"""
    no_value = ("no_numeric_value", "u.value IS NULL")

    def rel(col: str) -> str:
        return f"{REL0} + (x.{col} - h.admission_dttm)"

    def signal(table: str, target: str, name_col: str, value_col: str, explain=(), where: str = "") -> Bridge:
        return Bridge(
            table,
            f"""SELECT e.pid, e.t_rel AS t, e.nm AS item, e.numeric_value AS value FROM ev_i e
                WHERE e.fam = 'VITAL' AND e.nm IN (SELECT nm FROM sig WHERE target = '{target}')""",
            f"""SELECT h.patient_id AS pid, {rel('recorded_dttm')} AS t, x.{name_col} AS item, x.{value_col} AS value
                FROM {_hosp(table)} {where}""",
            explain_community=(no_value, *explain),
            explain_ours=(MERGED_IDENTICAL,))

    labs = Bridge(
        "labs",
        """SELECT e.pid, e.t_rel AS t, k.item, e.numeric_value AS value, e.code AS unit_key, k.loinc, k.unit FROM ev_i e
           JOIN (SELECT DISTINCT loinc, unit, item FROM lab_key) k
             ON e.fam = 'LAB' AND k.loinc = split_part(e.code, '//', 3) AND k.unit = split_part(e.code, '//', 4)""",
        f"""SELECT h.patient_id AS pid, {rel('lab_collect_dttm')} AS t, k.item, x.lab_value_numeric AS value
            FROM {_hosp('labs')} JOIN lab_key k ON k.nm = x.lab_name""",
        explain_community=(
            no_value,
            ("lab_key_of_unmapped_item",
             f"""EXISTS (SELECT 1 FROM lab_raw r WHERE NOT r.mapped AND r.pid = u.pid AND r.t = u.t
                        AND r.loinc = u.loinc AND r.unit = u.unit AND {_same('r.v', 'u.value')})"""),
        ),
        explain_ours=(MERGED_IDENTICAL,))
    resp_cols = ("fio2_set", "lpm_set", "resp_rate_set", "pressure_support_set", "pressure_control_set", "peep_set",
                 "tidal_volume_obs", "resp_rate_obs", "peak_inspiratory_pressure_obs", "minute_vent_obs",
                 "mean_airway_pressure_obs")
    resp = Bridge(
        "respiratory_support",
        f"""SELECT DISTINCT e.pid, e.t_rel AS t, p.item, e.numeric_value AS value, e.code AS unit_key FROM ev_i e
            JOIN sig s ON s.nm = e.nm AND s.target LIKE 'resp%' JOIN resp_pool p USING (category)
            WHERE e.fam = 'VITAL'
            UNION ALL   -- FiO2 recorded with the blood gas (LaboratoryID 684; its key 'UNK'/% holds other items)
            SELECT e.pid, e.t_rel, p.item, e.numeric_value, 'lab 684' FROM ev_i e
            JOIN resp_pool p ON p.category = 'fio2_set'
            WHERE e.code = 'LAB//LOINC//UNK//%'
              AND EXISTS (SELECT 1 FROM lab_raw r WHERE r.id = 684 AND r.pid = e.pid AND r.t = e.t_rel
                          AND {_same('r.v', 'e.numeric_value')})""",
        f"""SELECT u.pid, u.t, p.item, u.value
            FROM (UNPIVOT (SELECT h.patient_id AS pid, {rel('recorded_dttm')} AS t,
                                  {", ".join(f"x.{c}" for c in resp_cols)}
                           FROM {_hosp('respiratory_support')})
                  ON {", ".join(resp_cols)} INTO NAME category VALUE value) u
            JOIN resp_pool p USING (category)""",
        explain_community=(
            no_value,
            # one value per column and time: several items of one column (e.g. two O2 flows) keep the maximum
            ("max_per_item_and_time",
             "EXISTS (SELECT 1 FROM o_all o WHERE o.pid = u.pid AND o.t = u.t AND o.item = u.item)"),
        ),
        explain_ours=(MERGED_IDENTICAL,))
    bridges = [
        signal("vitals", "vitals", "vital_name", "vital_value",
               where="WHERE x.vital_name NOT IN ('WeightOnAdmission', 'HeightOnAdmission')"),
        labs, resp,
        signal("patient_assessments", "assessments", "assessment_name", "numerical_value"),
        signal("invasive_hemodynamics", "hemodynamics", "measurement_name", "measurement_value"),
        signal("output", "output", "output_name", "output_volume", explain=(("no_positive_volume", "u.value <= 0"),)),
    ]
    # medication: ``MEDICATION//<kind>`` at the start (no drug name; Amount, AmountPerMinute as text) and
    # ``MEDICATION_END//<kind>//<drug>//<unit>`` at the end. Raw records (the adapter's resolution) classify what
    # the community cannot: single dose or infusion, and the drug of a nameless start
    med_names = f"SELECT lower(ReferenceValue) AS txt FROM {ref} WHERE ReferenceName IN ('Drug', 'Fluid', 'Food')"
    dose_back = " ".join(f"WHEN '{u}' THEN {f}" for u, f in SICDB_DOSE_UNIT.values())
    rate_unit = " ".join(f"WHEN '{k}' THEN '{u}'" for k, (u, _) in SICDB_RATE_UNIT.items())
    rate_factor = " ".join(f"WHEN '{k}' THEN {f}" for k, (_, f) in SICDB_RATE_UNIT.items())
    setup += (
        *lexicon_setup(med_names),
        *regex_map_setup("lk_fluids", maps / "fluids.csv", ("category", "group_name"), med_names),
        f"""CREATE OR REPLACE TEMP TABLE med_kind AS
            SELECT ReferenceValue AS name, min(ReferenceName) AS kind FROM {ref}
            WHERE ReferenceName IN ('Drug', 'Fluid', 'Food') GROUP BY 1""",
        f"""CREATE OR REPLACE TEMP TABLE med_raw AS
            SELECT row_number() OVER () AS rid, CAST(c.PatientID AS VARCHAR) AS pid,
                   {REL0} + to_seconds(CAST(x."Offset" AS BIGINT)) AS t0,
                   {REL0} + to_seconds(CAST(x.OffsetDrugEnd AS BIGINT)) AS t1, x.AmountPerMinute AS apm,
                   x.IsSingleDose = 1 AS single, x.Amount AS amount, r.ReferenceName AS kind, r.ReferenceValue AS name,
                   r.ReferenceUnit AS unit,
                   CASE WHEN f.category = '__skip__' THEN NULL ELSE l.continuous END AS continuous,
                   CASE WHEN f.category = '__skip__' THEN NULL ELSE l.intermittent END AS intermittent
            FROM {_pq(mirror / 'medication.parquet')} x JOIN {_pq(mirror / 'cases.parquet')} c USING (CaseID)
            JOIN {ref} r ON r.ReferenceGlobalID = x.DrugID
            LEFT JOIN lk l ON l.txt = lower(r.ReferenceValue) LEFT JOIN lk_fluids f ON f.txt = lower(r.ReferenceValue)""",
        f"""CREATE OR REPLACE TEMP TABLE single_case AS
            SELECT CAST(PatientID AS VARCHAR) AS pid FROM {_pq(mirror / 'cases.parquet')} GROUP BY 1 HAVING count(*) = 1""",
        # community infusion records: a start (no drug name: the drug, and whether it is a single dose, come from
        # its raw record with the same start, kind, amount and rate) matched k-th to k-th in time with an end of
        # the same drug, amount and rate. Records whose start events are identical have one start event, which the
        # first record (by drug, single dose, end) takes; the others' end events can pair with no start and are
        # left out, or they would take the next start of their drug (an end event that is also a first record's
        # stays). The rate test filters inner joins: DuckDB runs an outer join whose condition is not a
        # conjunction of comparisons as a blockwise nested loop (starts x records)
        """CREATE OR REPLACE TEMP TABLE sic_rec AS
           WITH s0 AS (SELECT row_number() OVER () AS sid, pid, t_rel AS t0, nm AS kind, coalesce(numeric_value, -1) AS amount,
                              coalesce(text_value, '') AS apm_txt
                       FROM ev_i WHERE fam = 'MEDICATION'),
                cand AS (SELECT s0.sid, r.pid, r.t1, r.kind, r.name, r.amount, r.apm, r.single,
                                row_number() OVER (PARTITION BY s0.sid ORDER BY r.name, r.single, r.t1, r.rid) AS j
                         FROM s0 JOIN med_raw r
                           ON r.pid = s0.pid AND r.t0 = s0.t0 AND r.kind = s0.kind AND CAST(r.amount AS FLOAT) = CAST(s0.amount AS FLOAT)
                         WHERE (s0.apm_txt = '' AND (r.apm IS NULL OR r.single))   -- single doses carry no rate text
                            OR abs(r.apm - TRY_CAST(s0.apm_txt AS DOUBLE)) <= 1e-5 * greatest(abs(r.apm), 1e-12)),
                s1 AS (SELECT s0.*, c.name, c.single FROM s0 LEFT JOIN (SELECT sid, name, single FROM cand WHERE j = 1) c USING (sid)),
                s AS (SELECT *, row_number() OVER (PARTITION BY pid, kind, name, amount, apm_txt ORDER BY t0, sid) AS k FROM s1),
                e0 AS (SELECT row_number() OVER () AS eid, pid, t_rel AS t1, split_part(code, '//', 2) AS kind,
                              split_part(code, '//', 3) AS name, split_part(code, '//', 4) AS unit,
                              coalesce(numeric_value, -1) AS amount, coalesce(text_value, '') AS apm_txt, numeric_value, text_value
                       FROM ev_i WHERE fam = 'MEDICATION_END'),
                ends AS (SELECT e0.eid, c.j FROM cand c JOIN e0
                           ON e0.pid = c.pid AND e0.t1 = c.t1 AND e0.kind = c.kind AND e0.name = c.name
                          AND CAST(e0.amount AS FLOAT) = CAST(c.amount AS FLOAT)
                         WHERE (e0.apm_txt = '' AND (c.apm IS NULL OR c.single))
                            OR abs(c.apm - TRY_CAST(e0.apm_txt AS DOUBLE)) <= 1e-5 * greatest(abs(c.apm), 1e-12)),
                e AS (SELECT *, row_number() OVER (PARTITION BY pid, kind, name, numeric_value, text_value ORDER BY t1, unit) AS k
                      FROM e0 WHERE eid NOT IN (SELECT eid FROM ends GROUP BY eid HAVING min(j) > 1))
           SELECT s.pid, s.t0, e.t1, s.name, e.unit, s.kind, s.amount, s.single, TRY_CAST(s.apm_txt AS DOUBLE) AS apm
           FROM s JOIN e USING (pid, kind, name, amount, apm_txt, k)""",
        # streamed raw infusions without a community record: one of two records whose start events (time, kind,
        # amount, rate) or end events (time, drug, unit, amount, rate) are identical; MEDS-Transforms keeps one
        # row per distinct event
        f"""CREATE OR REPLACE TEMP TABLE sic_merged AS
            WITH r AS (SELECT *, continuous || ' ' || CASE unit {rate_unit} END AS key, apm * CASE unit {rate_factor} END AS rate,
                              row_number() OVER (PARTITION BY pid, t0, name, CAST(amount AS FLOAT) ORDER BY t1) AS k
                       FROM med_raw WHERE NOT single AND continuous IS NOT NULL AND apm > 0 AND t1 > t0
                         AND CASE unit {rate_unit} END IS NOT NULL AND pid IN (SELECT pid FROM single_case)),
                 c AS (SELECT *, row_number() OVER (PARTITION BY pid, t0, name, CAST(amount AS FLOAT) ORDER BY t1) AS k FROM sic_rec)
            SELECT r.pid, r.key, r.t0, r.t1, r.rate
            FROM r ANTI JOIN c ON c.pid = r.pid AND c.t0 = r.t0 AND c.name = r.name
                                  AND CAST(c.amount AS FLOAT) = CAST(r.amount AS FLOAT) AND c.k = r.k
            WHERE EXISTS (SELECT 1 FROM med_raw r2 WHERE r2.pid = r.pid AND r2.rid <> r.rid AND r2.amount = r.amount
                          AND r2.apm IS NOT DISTINCT FROM r.apm
                          AND ((r2.t0 = r.t0 AND r2.kind = r.kind)
                               OR (r2.t1 = r.t1 AND r2.name = r.name AND r2.unit IS NOT DISTINCT FROM r.unit)))""",
    )
    fate = ("EXISTS (SELECT 1 FROM med_raw r WHERE r.pid = u.pid AND r.t0 = u.t AND r.kind = u.item "
            "AND CAST(r.amount AS FLOAT) = CAST(u.value AS FLOAT) AND ({}))")
    # the starts name no drug: doses pair within (patient, time, kind) only when their amounts are equal
    amount_key = _SIG5.format(v=_F32.format(v="{x}"))
    medication = Bridge(
        "medication_admin_intermittent",
        f"""SELECT e.pid, e.t_rel AS t, e.nm AS item, e.numeric_value AS value,
                   CAST({amount_key.format(x="e.numeric_value")} AS VARCHAR) AS pair_key
            FROM ev_i e WHERE e.fam = 'MEDICATION'""",
        # our single doses on the community's terms: the drug's kind, the amount in the source unit
        f"""SELECT h.patient_id AS pid, {rel('admin_dttm')} AS t, k.kind AS item,
                   x.med_dose / CASE x.med_dose_unit {dose_back} END AS value,
                   CAST({amount_key.format(x=f'x.med_dose / CASE x.med_dose_unit {dose_back} END')} AS VARCHAR) AS pair_key
            FROM {_hosp('medication_admin_intermittent')} JOIN med_kind k ON k.name = x.med_name""",
        explain_community=(
            ("food_energy_record", "u.item = 'FoodKcal'"),
            ("infusion_compared_as_rate_stream", fate.format("NOT r.single")),
            ("no_intermittent_category", fate.format("r.intermittent IS NULL")),
            ("no_positive_amount", "NOT u.value > 0"),
            ("unit_not_mapped", fate.format(f"coalesce(r.unit, '') NOT IN ({', '.join(_sq(k) for k in SICDB_DOSE_UNIT)})")),
        ),
        explain_ours=(MERGED_IDENTICAL,),
        name="medication_admin_intermittent (medication)")
    bridges.append(medication)
    # infusions (``sic_rec``). Patients with several cases are left out: the community lays every case on the
    # first case's timeline
    stream = Streams(
        "medication_admin_continuous (medication, rate streams)", "medication_admin_continuous",
        f"""SELECT c.pid, l.continuous || ' ' || CASE c.unit {rate_unit} END AS key, c.t0, c.t1,
                   c.apm * CASE c.unit {rate_factor} END AS rate,
                   CASE WHEN c.pid NOT IN (SELECT pid FROM single_case) THEN 'patient_with_several_cases'
                        WHEN c.single THEN 'single_dose'
                        WHEN f.category = '__skip__' OR l.continuous IS NULL THEN 'no_continuous_category'
                        WHEN CASE c.unit {rate_unit} END IS NULL THEN 'rate_unit_not_mapped' END AS reason
            FROM sic_rec c LEFT JOIN lk l ON l.txt = lower(c.name) LEFT JOIN lk_fluids f ON f.txt = lower(c.name)""",
        f"""SELECT h.patient_id AS pid, x.med_category || ' ' || x.med_dose_unit AS key, {rel('admin_dttm')} AS t,
                   x.med_dose AS rate
            FROM {_hosp('medication_admin_continuous')} WHERE CAST(h.patient_id AS VARCHAR) IN (SELECT pid FROM single_case)""",
        explain=(_merged_records("sic_merged"),))
    return DatasetBridges("sicdb/MEDS_cohort", bridges, setup, scope=scope, streams=(stream,))


# -- eICU (our adapter) ------------------------------------------------------------------------------------
def eicu(mirror: Path, refs: Path, maps: Path) -> DatasetBridges:
    """eICU_MEDS (subject = ``patienthealthsystemstayid``, our hospitalization; times = unit admission + offset,
    as ours): ``VITALS//PERIODIC|APERIODIC//[BP//]<COLUMN>`` (one event per column of the wide monitor rows,
    also when the column is empty), ``ICU_ADMISSION_WEIGHT`` / ``_HEIGHT``, ``ICU_DISCHARGE_WEIGHT``,
    ``LAB//<labname>//<unit>``. Rows pair per hospitalization (health-system stays of one patient can overlap
    on the pseudo-timeline). nurseCharting, intakeOutput and respiratoryCharting are not in the community ETL;
    our rows from them are left out."""
    setup = (
        f"CREATE OR REPLACE TEMP TABLE vmap AS SELECT * FROM {_csv(maps / 'vitals.csv')}",
        f"CREATE OR REPLACE TEMP TABLE lmap AS SELECT * FROM {_csv(maps / 'labs.csv')}",
        # labs mapped by the interface unit (map rows 'iface:<unit>'): the community codes carry the system unit
        # only, so the raw results whose interface unit no row maps are listed by stay, lab and float32 value
        f"""CREATE OR REPLACE TEMP TABLE lab_iface_unmapped AS
            SELECT DISTINCT CAST(p.patienthealthsystemstayid AS VARCHAR) AS pid, l.labname AS item,
                   {_F32.format(v='l.labresult')} AS value
            FROM {_pq(mirror / 'lab.parquet')} l JOIN {_pq(mirror / 'patient.parquet')} p USING (patientunitstayid)
            WHERE l.labname IN (SELECT labname FROM lmap WHERE starts_with(unit_in, 'iface:'))
              AND NOT EXISTS (SELECT 1 FROM lmap m WHERE m.labname = l.labname AND starts_with(m.unit_in, 'iface:')
                              AND substr(m.unit_in, 7) = lower(trim(coalesce(l.labmeasurenameinterface, ''))))""",
        """CREATE OR REPLACE TEMP TABLE ev_v AS
           SELECT pid, time, numeric_value,
                  lower(replace(regexp_replace(code, '^VITALS//(PERIODIC|APERIODIC)//(BP//)?', ''), '_', '')) AS item
           FROM ev WHERE code LIKE 'VITALS//%' AND numeric_value IS NOT NULL""",
    )
    scope = f"""SELECT hospitalization_id AS pid, admission_dttm - {SCOPE_BEFORE_ADMISSION} AS t0, discharge_dttm AS t1
                FROM t_hospitalization"""
    weights = {"ICU_ADMISSION_WEIGHT": "admissionweight", "ICU_ADMISSION_HEIGHT": "admissionheight",
               "ICU_DISCHARGE_WEIGHT": "dischargeweight"}
    vitals = Bridge(
        "vitals",
        """SELECT v.pid, v.time AS t, v.item,
                  CASE WHEN m.transform = 'temp_c_or_f' AND v.numeric_value BETWEEN 80 AND 113
                       THEN (v.numeric_value - 32) / 1.8 ELSE v.numeric_value END AS value
           FROM ev_v v JOIN vmap m ON m.item = v.item AND m.source_table IN ('vitalPeriodic', 'vitalAperiodic')
           UNION ALL
           SELECT pid, time, CASE code """ + " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in weights.items()) + """ END,
                  numeric_value
           FROM ev WHERE code IN (""" + ", ".join(f"'{k}'" for k in weights) + ") AND numeric_value IS NOT NULL",
        """SELECT x.hospitalization_id AS pid, x.recorded_dttm AS t, x.vital_name AS item, x.vital_value AS value
           FROM t_vitals x
           WHERE x.vital_name IN (SELECT item FROM vmap WHERE source_table IN ('vitalPeriodic', 'vitalAperiodic'))
              OR x.vital_name IN (""" + ", ".join(f"'{v}'" for v in weights.values()) + ")",
        explain_ours=(MERGED_IDENTICAL,))
    labs = Bridge(
        "labs",
        """SELECT e.pid, e.time AS t, split_part(e.code, '//', 2) AS item, e.numeric_value AS value, e.code AS unit_key,
                  split_part(e.code, '//', 3) AS unit
           FROM ev e WHERE e.code LIKE 'LAB//%' AND split_part(e.code, '//', 2) IN (SELECT labname FROM lmap)""",
        """SELECT x.hospitalization_id AS pid, x.lab_collect_dttm AS t, x.lab_name AS item, x.lab_value_numeric AS value
           FROM t_labs x""",
        explain_community=(
            ("no_numeric_value", "u.value IS NULL"),
            # a lab mapped by interface unit (ammonia, CRP, differentials): a result in a unit no row maps
            ("interface_unit_not_in_map",
             f"""EXISTS (SELECT 1 FROM lab_iface_unmapped r WHERE r.pid = u.pid AND r.item = u.item
                        AND r.value = {_F32.format(v='u.value')})"""),
            ("unit_not_in_map",
             """u.item NOT IN (SELECT labname FROM lmap WHERE starts_with(unit_in, 'iface:'))
                AND NOT EXISTS (SELECT 1 FROM lmap m WHERE m.labname = u.item
                                AND (m.unit_in = '*' OR lower(m.unit_in) = lower(trim(u.unit))))"""),
        ),
        explain_ours=(MERGED_IDENTICAL,))
    # infusionDrug: one INFUSION_DRUG event per charted row (``text_value`` = drug name with its '(unit)',
    # ``drug_rate``); our adapter keeps rate changes: one rate per (drug category, unit, time), the highest;
    # repeated rates dropped within a run (gaps up to 6 h); a stop where the rate falls to 0; inferred stops
    inf_name, inf_unit = INFUSION_BASE.format("e.text_value"), INFUSION_UNIT.format("e.text_value")
    setup += (
        *regex_map_setup("lk_inf", maps / "infusions.csv", ("med_category",),
                         f"SELECT {inf_name} AS txt FROM ev e WHERE e.code = 'INFUSION_DRUG'"),
        # runs are per ICU unit stay (the event's infusiondrugid gives it)
        f"""CREATE OR REPLACE TEMP TABLE inf_c AS
            WITH r AS (
                SELECT e.pid, i.patientunitstayid AS stay, e.time AS t, e.text_value AS item,
                       TRY_CAST(e.drug_rate AS DOUBLE) AS value, l.med_category, replace({inf_unit}, 'nanograms', 'ng') AS unit,
                       regexp_full_match({inf_unit}, '{RATE_UNIT_RE}') AS unit_ok
                FROM ev e LEFT JOIN lk_inf l ON l.txt = {inf_name}
                LEFT JOIN {_pq(mirror / 'infusionDrug.parquet')} i ON i.infusiondrugid = TRY_CAST(e.medication_id AS BIGINT)
                WHERE e.code = 'INFUSION_DRUG'),
                 g AS (SELECT pid, stay, med_category, unit, t, max(value) AS max_rate FROM r
                       WHERE med_category <> '__skip__' AND unit_ok AND value >= 0 GROUP BY ALL),
                 w AS (SELECT *, lag(max_rate) OVER k AS prev_rate, epoch(t - lag(t) OVER k) / 60 AS gap_min
                       FROM g WINDOW k AS (PARTITION BY pid, stay, med_category, unit ORDER BY t))
            SELECT r.*, w.max_rate, w.prev_rate, w.gap_min FROM r LEFT JOIN w USING (pid, stay, med_category, unit, t)""",
    )
    infusions = Bridge(
        "medication_admin_continuous",
        "SELECT pid, t, item, value, stay, med_category, unit, unit_ok, max_rate, prev_rate, gap_min FROM inf_c",
        # an inferred stop is not a charting: its own item, so it never pairs with a charted rate at that time
        """SELECT x.hospitalization_id AS pid, x.admin_dttm AS t, x.med_dose AS value, x.mar_action_name,
                  CASE WHEN x.mar_action_name LIKE 'inferred stop%' THEN x.med_name || ' [inferred stop]'
                       ELSE x.med_name END AS item
           FROM t_medication_admin_continuous x""",
        explain_community=(
            ("no_clif_continuous_category", "u.med_category IS NULL OR u.med_category = '__skip__'"),
            ("unit_not_a_rate", "NOT coalesce(u.unit_ok, false)"),
            ("rate_not_a_number", "u.value IS NULL OR u.value < 0"),
            ("another_charting_of_the_drug_at_that_time_kept",
             """u.value < u.max_rate OR EXISTS (SELECT 1 FROM c_all c WHERE c.pid = u.pid AND c.t = u.t AND c.rid <> u.rid
                                                AND c.stay = u.stay AND c.med_category = u.med_category AND c.unit = u.unit
                                                AND c.value = u.value)"""),
            ("rate_unchanged_within_a_run", f"u.value = u.prev_rate AND u.gap_min <= {STOP_GAP_MIN}"),
            ("zero_rate_without_a_running_infusion",
             f"u.value = 0 AND (u.prev_rate IS NULL OR u.prev_rate = 0 OR u.gap_min > {STOP_GAP_MIN})"),
        ),
        explain_ours=(("inferred_stop_ricu_interval_rule", "u.mar_action_name LIKE 'inferred stop%'"), MERGED_IDENTICAL),
        name="medication_admin_continuous (infusionDrug)")
    # medication (orders): ``MEDICATION//STARTED|STOPPED//<drug name>`` with ``medication_id``; ours are CLIF 3.0
    # medication_orders (order start, end) keyed by that id
    med = _pq(mirror / "medication.parquet")
    setup += (
        *lexicon_setup(f"SELECT lower(trim(drugname)) AS txt FROM {med}"),
        *regex_map_setup("lk_route", maps / "routes.csv", ("route_class",), f"SELECT lower(trim(routeadmin)) AS txt FROM {med}"),
        f"""CREATE OR REPLACE TEMP TABLE ord_raw AS
            WITH ing_cat AS (SELECT ingredient, max(nullif(continuous, '')) AS continuous,
                                    max(nullif(intermittent, '')) AS intermittent FROM lex GROUP BY 1)
            SELECT m.medicationid, m.drugstopoffset < m.drugstartoffset AS stop_before_start,
                   coalesce(r.route_class, 'unknown') AS route_class,
                   coalesce(CASE WHEN l.ingredient IS NOT NULL THEN coalesce(l.continuous, l.intermittent)
                                 ELSE coalesce(c.continuous, c.intermittent) END, '') <> '' AS has_category
            FROM {med} m
            LEFT JOIN lk l ON l.txt = lower(trim(m.drugname))
            LEFT JOIN {_csv(maps / 'hicl.csv')} h ON l.ingredient IS NULL AND h.drughiclseqno = CAST(m.drughiclseqno AS VARCHAR)
            LEFT JOIN ing_cat c ON c.ingredient = h.ingredient
            LEFT JOIN lk_route r ON r.txt = lower(trim(m.routeadmin))""",
    )
    orders = Bridge(
        "medication_orders",
        """SELECT e.pid, e.time AS t, 'medication order' AS item, CAST(NULL AS DOUBLE) AS value,
                  CAST(e.medication_id AS VARCHAR) || '|' || split_part(e.code, '//', 2) AS pair_key,
                  split_part(e.code, '//', 2) AS kind, o.has_category, o.route_class, o.stop_before_start
           FROM ev e LEFT JOIN ord_raw o ON o.medicationid = TRY_CAST(e.medication_id AS BIGINT)
           WHERE e.code LIKE 'MEDICATION//%'""",
        """SELECT x.hospitalization_id AS pid, x.order_start_dttm AS t, 'medication order' AS item,
                  CAST(NULL AS DOUBLE) AS value, x.med_order_id || '|STARTED' AS pair_key
           FROM t_medication_orders x
           UNION ALL
           SELECT x.hospitalization_id, x.order_end_dttm, 'medication order', NULL, x.med_order_id || '|STOPPED'
           FROM t_medication_orders x WHERE x.order_end_dttm IS NOT NULL""",
        explain_community=(
            ("route_without_clif_route_category", "u.route_class = 'non_systemic'"),
            ("drug_without_clif_category", "NOT u.has_category"),
            ("stop_before_start_unknown", "u.kind = 'STOPPED' AND u.stop_before_start"),
        ),
        name="medication_orders (medication)")
    return DatasetBridges("eicu/MEDS_cohort", [vitals, labs, infusions, orders], setup, scope=scope)


# -- HiRID (our adapter) -----------------------------------------------------------------------------------
def hirid(mirror: Path, refs: Path, maps: Path) -> DatasetBridges:
    """HiRID_MEDS: ``OBSERVATION//<Variable Name>//<unit>`` at ``datetime`` (value; ``stringvalue`` as text),
    on the same date-shifted clock as ours. Several variable ids share a name (four potassium ids, two SpO2
    sensors), so items pool by name. The derived urine rate (ml/h) becomes a volume in our output (rate x hours
    since the previous report): such items pair on presence only."""
    ref = _pq(mirror / "reference/hirid_variable_reference.parquet")
    obs_map = _csv(maps / "observations.csv")
    setup = (
        f"""CREATE OR REPLACE TEMP TABLE vname AS
            SELECT DISTINCT CAST(m.variableid AS BIGINT) AS vid, m.target, m.reference_unit, r."Variable Name" AS nm
            FROM {obs_map} m JOIN {ref} r ON r.ID = CAST(m.variableid AS BIGINT)
            WHERE r."Source Table" = 'Observation'""",
        """CREATE OR REPLACE TEMP TABLE ev_o AS
           SELECT pid, time, numeric_value, code, split_part(code, '//', 2) AS nm FROM ev WHERE code LIKE 'OBSERVATION//%'""",
        # observations our adapter drops: status bit 2 (invalidated) or 32 (notified, not measured)
        f"""CREATE OR REPLACE TEMP TABLE obs_dropped AS
            SELECT DISTINCT CAST(x.patientid AS VARCHAR) AS pid, CAST(x.datetime AS TIMESTAMP) AS t, v.nm
            FROM {_pq(mirror / 'raw_stage/observation_tables/csv/*.parquet')} x
            JOIN vname v ON v.vid = CAST(x.variableid AS BIGINT)
            WHERE (TRY_CAST(x.status AS INTEGER) & 34) <> 0""",
        # raw rows of names that several variables share, with the variable's target (NULL: not mapped)
        f"""CREATE OR REPLACE TEMP TABLE obs_shared AS
            WITH r AS (SELECT ID AS vid, "Variable Name" AS nm FROM {ref} WHERE "Source Table" = 'Observation'),
                 s AS (SELECT nm FROM r GROUP BY nm HAVING count(*) > 1)
            SELECT CAST(x.patientid AS VARCHAR) AS pid, CAST(x.datetime AS TIMESTAMP) AS t, r.nm, r.vid,
                   (SELECT any_value(target) FROM vname v WHERE v.vid = r.vid) AS target, TRY_CAST(x.value AS DOUBLE) AS v
            FROM {_pq(mirror / 'raw_stage/observation_tables/csv/*.parquet')} x
            JOIN r ON r.vid = CAST(x.variableid AS BIGINT) WHERE r.nm IN (SELECT nm FROM s)""",
    )
    dropped = ("invalidated_or_not_measured",
               "EXISTS (SELECT 1 FROM obs_dropped d WHERE d.pid = u.pid AND d.t = u.t AND d.nm = u.item)")

    def obs(table: str, target: str, name_col: str, value_col: str, time_col: str = "recorded_dttm",
            explain=(), presence_only: bool = False) -> Bridge:
        value = ("CASE WHEN e.nm IN (SELECT nm FROM vname WHERE reference_unit = 'ml/h') THEN NULL "
                 "ELSE e.numeric_value END") if presence_only else "e.numeric_value"
        return Bridge(
            table,
            f"""SELECT e.pid, e.time AS t, e.nm AS item, {value} AS value, e.code AS unit_key
                FROM ev_o e WHERE e.nm IN (SELECT nm FROM vname WHERE target = '{target}')""",
            f"""SELECT h.patient_id AS pid, x.{time_col} AS t, v.nm AS item, x.{value_col} AS value
                FROM {_hosp(table)} JOIN (SELECT DISTINCT vid, nm FROM vname WHERE target = '{target}') v
                  ON CAST(v.vid AS VARCHAR) = x.{name_col}""",
            explain_community=(
                dropped, ("no_numeric_value", "u.value IS NULL"),
                # e.g. 'Respiratory rate' names the monitor (300, vitals), ventilator (310, respiratory support)
                # and an unmapped variable (5685): the row is another variable's
                ("name_of_another_variable",
                 f"""EXISTS (SELECT 1 FROM obs_shared s WHERE s.pid = u.pid AND s.t = u.t AND s.nm = u.item
                            AND s.target IS DISTINCT FROM '{target}'
                            AND (s.v IS NULL AND u.value IS NULL OR {_same('s.v', 'u.value')}))"""),
                # an observation re-entered with the same value is one observation of ours; the community keeps
                # the entries apart when their text differs (one entry carries the value as text: troponin
                # 558 and '0.558')
                ("re_entered_observation",
                 """EXISTS (SELECT 1 FROM c_all c2 JOIN pairs p ON p.c_rid = c2.rid
                            WHERE p.o_rid IS NOT NULL AND c2.rid <> u.rid AND c2.pid = u.pid AND c2.t = u.t
                              AND c2.item = u.item AND c2.value IS NOT DISTINCT FROM u.value)"""),
                *explain),
            explain_ours=(MERGED_IDENTICAL,))

    # pharma records: why the adapter does not emit a record as an intermittent dose (its own rules, in order)
    setup += (
        f"""CREATE OR REPLACE TEMP TABLE ph_why AS
            SELECT pid, t, nm, dose, CASE
                       WHEN (st & 34) <> 0 THEN 'invalidated_or_not_administered'
                       WHEN (st & 260) <> 260 THEN 'infusion_record'
                       WHEN NOT coalesce(dose > 0, false) THEN 'no_positive_dose'
                       WHEN ({HIRID_ROUTE}) IS NULL THEN 'route_without_clif_category' END AS why
            FROM (SELECT CAST(p.patientid AS VARCHAR) AS pid, CAST(p.givenat AS TIMESTAMP) AS t, r."Variable Name" AS nm,
                         TRY_CAST(p.givendose AS DOUBLE) AS dose, CAST(p.recordstatus AS INTEGER) AS st, p.route
                  FROM {_pq(mirror / 'raw_stage/pharma_records/csv/*.parquet')} p
                  JOIN {ref} r ON r."Source Table" = 'Pharma' AND r.ID = CAST(p.pharmaid AS BIGINT))""",
    )

    def ph_reason(why: str) -> tuple[str, str]:
        return (why, f"""EXISTS (SELECT 1 FROM ph_why w WHERE w.why = '{why}' AND w.pid = u.pid AND w.t = u.t
                                AND w.nm = u.item AND (w.dose IS NULL AND u.value IS NULL OR {_same('w.dose', 'u.value')}))""")

    setup += lexicon_setup("SELECT lower(split_part(code, '//', 2)) AS txt FROM ev WHERE code LIKE 'MEDICATION//%'")
    medication = Bridge(
        "medication_admin_intermittent",
        """SELECT e.pid, e.time AS t, split_part(e.code, '//', 2) AS item, e.numeric_value AS value, e.code AS unit_key
           FROM ev e WHERE e.code LIKE 'MEDICATION//%'""",
        f"""SELECT h.patient_id AS pid, x.admin_dttm AS t, x.med_name AS item, x.med_dose AS value
            FROM {_hosp('medication_admin_intermittent')}""",
        explain_community=(*(ph_reason(w) for w in ("invalidated_or_not_administered", "infusion_record")),
                           NO_INTERMITTENT_CATEGORY,
                           *(ph_reason(w) for w in ("no_positive_dose", "route_without_clif_category"))),
        explain_ours=(MERGED_IDENTICAL,))
    bridges = [
        obs("vitals", "vitals", "vital_name", "vital_value"),
        obs("labs", "labs", "lab_name", "lab_value_numeric", time_col="lab_collect_dttm"),
        obs("patient_assessments", "assessments", "assessment_name", "numerical_value"),
        obs("invasive_hemodynamics", "hemodynamics", "measurement_name", "measurement_value"),
        obs("output", "output", "output_name", "output_volume", presence_only=True,
            explain=(("no_positive_volume", "u.value <= 0"),)),
        medication,
    ]
    # infusions: a pharma record's dose was given since the infusion's previous record. The community has the
    # doses but no infusion id, so the rate steps cannot be rebuilt: total dose per (patient, drug and unit),
    # community doses against the integral of our rate stream. Each community dose is matched (k-th to k-th) to
    # its raw record for the unit and status bits the event lacks
    setup += (
        f"""CREATE OR REPLACE TEMP TABLE ph_raw AS
            SELECT row_number() OVER () AS rid, pid, t, et, nm, dose, st, inf, pharmaid, ({HIRID_ROUTE}) AS route_cat,
                   CASE WHEN doseunit = 'MU' THEN 1e6 ELSE 1 END AS factor,
                   CASE doseunit WHEN 'µg' THEN 'mcg' WHEN 'U' THEN 'u' WHEN 'IE' THEN 'u' WHEN 'MU' THEN 'u'
                                 ELSE lower(doseunit) END AS unit
            FROM (SELECT CAST(p.patientid AS VARCHAR) AS pid, CAST(p.givenat AS TIMESTAMP) AS t,
                         CAST(p.enteredentryat AS TIMESTAMP) AS et, r."Variable Name" AS nm,
                         TRY_CAST(p.givendose AS DOUBLE) AS dose, CAST(p.recordstatus AS INTEGER) AS st, p.route, p.doseunit,
                         CAST(p.infusionid AS BIGINT) AS inf, CAST(p.pharmaid AS BIGINT) AS pharmaid
                  FROM {_pq(mirror / 'raw_stage/pharma_records/csv/*.parquet')} p
                  JOIN {ref} r ON r."Source Table" = 'Pharma' AND r.ID = CAST(p.pharmaid AS BIGINT))""",
        # each community dose event with its raw record (k-th to k-th within patient, time, name and dose)
        """CREATE OR REPLACE TEMP TABLE ph_ev AS
           WITH c AS (SELECT pid, time AS t, split_part(code, '//', 2) AS nm, numeric_value AS dose,
                             row_number() OVER (PARTITION BY pid, time, code, CAST(numeric_value AS FLOAT) ORDER BY numeric_value) AS k
                      FROM ev WHERE code LIKE 'MEDICATION//%'),
                r AS (SELECT *, row_number() OVER (PARTITION BY pid, t, nm, CAST(dose AS FLOAT) ORDER BY st, inf, rid) AS k
                      FROM ph_raw)
           SELECT c.pid, c.t, c.nm, c.dose, r.rid, r.st, r.route_cat, r.factor, r.unit FROM c LEFT JOIN r
             ON r.pid = c.pid AND r.t = c.t AND r.nm = c.nm AND CAST(r.dose AS FLOAT) = CAST(c.dose AS FLOAT) AND r.k = c.k""",
        # the raw infusion records our stream reads, with the two documented reasons a total can differ: a record
        # identical to another has no community event (MEDS-Transforms keeps one row per distinct event: ours is
        # higher), and records of one infusion at one time are one record (the benchmark's drop_duplicates_pharma:
        # the community is higher by the doses of the others)
        """CREATE OR REPLACE TEMP TABLE ph_adj AS
           WITH s AS (SELECT p.*, l.continuous || ' ' || p.unit || '/min' AS key,
                             count(*) FILTER (WHERE (p.st & 256) <> 0 AND p.dose <> 0)
                                 OVER (PARTITION BY p.pid, p.pharmaid, p.inf, p.t) AS n_dosed_stops
                      FROM ph_raw p JOIN lk l ON l.txt = lower(p.nm)
                      WHERE (p.st & 34) = 0 AND (p.st & 260) <> 260 AND l.continuous IS NOT NULL AND p.route_cat = 'iv'
                        AND p.dose IS NOT NULL),
                k AS (SELECT *, row_number() OVER (PARTITION BY pid, pharmaid, inf, t
                                                   ORDER BY (st & 256) <> 0 DESC,
                                                            CASE WHEN n_dosed_stops = 1 THEN dose <> 0 END DESC NULLS LAST,
                                                            et DESC NULLS LAST, st DESC, dose DESC) AS same_time_rank
                      FROM s)
           SELECT pid, key, coalesce(sum(dose * factor) FILTER (WHERE rid NOT IN (SELECT rid FROM ph_ev WHERE rid IS NOT NULL)), 0)
                                AS merged,
                  coalesce(sum(dose * factor) FILTER (WHERE same_time_rank > 1), 0) AS same_time
           FROM k GROUP BY 1, 2""",
    )
    stream = Streams(
        "medication_admin_continuous (pharma, total dose)", "medication_admin_continuous",
        """SELECT m.pid, l.continuous || ' ' || m.unit || '/min' AS key, m.t - INTERVAL 1 MINUTE AS t0, m.t AS t1,
                  m.dose * m.factor AS rate,
                  CASE WHEN m.dose IS NULL THEN 'no_dose'
                       WHEN (m.st & 34) <> 0 THEN 'invalidated_or_not_administered'
                       WHEN (m.st & 260) = 260 THEN 'single_administration'
                       WHEN l.continuous IS NULL THEN 'no_continuous_category'
                       WHEN m.route_cat IS DISTINCT FROM 'iv' THEN 'route_not_iv' END AS reason
           FROM ph_ev m LEFT JOIN lk l ON l.txt = lower(m.nm)""",
        f"""SELECT h.patient_id AS pid, x.med_category || ' ' || x.med_dose_unit AS key, x.admin_dttm AS t, x.med_dose AS rate
            FROM {_hosp('medication_admin_continuous')}""",
        explain=(("merged_identical_or_same_time_records",
                  """abs((k.o_total - k.c_total) - coalesce((SELECT a.merged - a.same_time FROM ph_adj a
                                                             WHERE a.pid = k.pid AND a.key = k.key), 0))
                     <= 1e-3 * greatest(k.c_total, k.o_total)"""),),
        totals_only=True)
    return DatasetBridges("hirid/MEDS_cohort", bridges, setup, streams=(stream,))


# -- AmsterdamUMCdb (CLIF-AmsterdamUMCdb + our supplement) -------------------------------------------------
def aumc(mirror: Path, refs: Path, maps: Path) -> DatasetBridges:
    """AUMCdb_MEDS: ``MEASURE//<item>//<unit>`` (numericitems, at ``measuredat``), times = 1 January of the
    patient's first admission year group + offset, the tool's convention. Our rows name items
    ``<itemid>: <item>``; items pair by name (the community codes carry no item id). The community reads the
    Latin-1 source as lossy UTF-8 ('Intracrani\\ufffdle druk'): names compare with every non-ASCII character
    as U+FFFD. Admission weight and height groups (rows named by their group) are not paired."""
    lossy = "regexp_replace({}, '[^\\x00-\\x7F]', chr(65533), 'g')"
    adm = _pq(mirror / "admissions.parquet")
    setup = (
        f"""CREATE OR REPLACE TEMP TABLE ev_m AS
            SELECT pid, time, numeric_value, code, {lossy.format("split_part(code, '//', 2)")} AS nm,
                   split_part(code, '//', 3) AS unit
            FROM ev WHERE code LIKE 'MEASURE//%'""",
        f"CREATE OR REPLACE TEMP TABLE lmap AS SELECT * FROM {_csv(maps / 'labs.csv')}",
        # blood gases our supplement drops: the gas of a specimen class without a CLIF category (venous pO2
        # / SO2), by the supplement's own specimen lookup and routing maps; times on the patient's epoch
        f"""CREATE OR REPLACE TEMP TABLE gas_dropped AS
            WITH m AS (SELECT * FROM lmap WHERE coalesce(gas, '') <> ''),
                 sf AS (SELECT admissionid, measuredat, min(value) AS specimen
                        FROM {_pq(mirror / 'freetextitems.parquet')} WHERE itemid = 11646 GROUP BY 1, 2),
                 sl AS (SELECT admissionid, measuredat, min(value) AS specimen
                        FROM {_pq(mirror / 'listitems.parquet')} WHERE itemid = 7432 GROUP BY 1, 2),
                 ep AS (SELECT patientid, make_timestamp(CAST(regexp_extract(any_value(admissionyeargroup),
                                                              '(2003|2010)') AS BIGINT), 1, 1, 0, 0, 0) AS epoch
                        FROM {adm} WHERE admissioncount = 1 GROUP BY 1)
            SELECT CAST(a.patientid AS VARCHAR) AS pid, ep.epoch + to_milliseconds(CAST(n.measuredat AS BIGINT)) AS t,
                   n.item AS nm, n.value
            FROM {_pq(mirror / 'numericitems.parquet')} n JOIN m ON CAST(m.itemid AS BIGINT) = n.itemid
            JOIN {adm} a USING (admissionid) JOIN ep USING (patientid)
            LEFT JOIN sf ON m.specimen_source = 'freetext' AND sf.admissionid = n.admissionid AND sf.measuredat = n.measuredat
            LEFT JOIN sl ON m.specimen_source = 'list' AND sl.admissionid = n.admissionid AND sl.measuredat = n.measuredat
            LEFT JOIN {_csv(maps / 'specimens.csv')} s
                   ON s.specimen = CASE m.specimen_source WHEN 'freetext' THEN sf.specimen WHEN 'list' THEN sl.specimen END
            LEFT JOIN {_csv(maps / 'gas_routing.csv')} g ON g.gas = m.gas AND g.specimen_class = coalesce(s.specimen_class, 'unknown')
            WHERE g.lab_category IS NULL""",
        f"""CREATE OR REPLACE TEMP TABLE item_units AS
            SELECT DISTINCT {lossy.format('n.item')} AS nm, m.unit_in
            FROM (SELECT DISTINCT itemid, item FROM {_pq(mirror / 'numericitems.parquet')}) n
            JOIN lmap m ON CAST(m.itemid AS BIGINT) = n.itemid""",
    )
    gas = ("venous_gas_without_clif_category",
           f"""EXISTS (SELECT 1 FROM gas_dropped d WHERE d.pid = u.pid AND d.t = u.t AND d.nm = u.item
                      AND {_same('d.value', 'u.value')})""")
    unit = ("unit_not_in_map",
            """u.item IN (SELECT nm FROM item_units)
               AND NOT EXISTS (SELECT 1 FROM item_units i WHERE i.nm = u.item AND (i.unit_in = '*' OR i.unit_in = u.unit))""")

    def measure(table: str, name_col: str, value_col: str, time_col: str = "recorded_dttm", explain=()) -> Bridge:
        name = lossy.format(f"regexp_replace(x.{name_col}, '^[0-9]+: ', '')")
        return Bridge(
            table,
            f"""SELECT e.pid, e.time AS t, e.nm AS item, e.numeric_value AS value, e.code AS unit_key, e.unit
                FROM ev_m e WHERE e.nm IN (SELECT DISTINCT {name} FROM t_{table} x)""",
            f"""SELECT h.patient_id AS pid, x.{time_col} AS t, {name} AS item, x.{value_col} AS value
                FROM {_hosp(table)} WHERE regexp_matches(x.{name_col}, '^[0-9]+: ')""",
            explain_community=(("no_numeric_value", "u.value IS NULL"), *explain),
            # AmsterdamUMCdb repeats a patient's measurements under each of the patient's admissions (times are on
            # the patient's axis); the community, keyed by patient, keeps one
            explain_ours=(MERGED_IDENTICAL,))

    bridges = [measure("vitals", "vital_name", "vital_value"),
               measure("labs", "lab_name", "lab_value_numeric", time_col="lab_collect_dttm", explain=(gas, unit))]
    # drugitems (``DRUG//{START,RATE,DOSE}//<order category>//<item>//...`` at the start, ``DRUG//END//...`` at the
    # stop; no record id). Each raw record with the fate our pipeline gives it: the tool's intermittent items
    # (non-continuous records of 62 items, every action), our supplement's intermittent doses (other items: an
    # administration action, a mapped route, a lexicon category), continuous records (a rate stream)
    tool_items = f"SELECT CAST(itemid AS BIGINT) FROM {_csv(maps / 'tool_intermittent_items.csv')}"
    setup += (
        *lexicon_setup(f"SELECT lower(item) AS txt FROM {_pq(mirror / 'drugitems.parquet')}"),
        f"""CREATE OR REPLACE TEMP TABLE drug_raw AS
            WITH ep AS (SELECT patientid, coalesce(CASE regexp_extract(any_value(admissionyeargroup), '(2003|2010)')
                                                       WHEN '2003' THEN TIMESTAMP '2003-01-01' WHEN '2010' THEN TIMESTAMP '2010-01-01' END,
                                                  TIMESTAMP '2000-01-01') AS epoch
                        FROM {adm} WHERE admissioncount = 1 GROUP BY 1)
            SELECT CAST(a.patientid AS VARCHAR) AS pid, ep.epoch + to_milliseconds(CAST(d.start AS BIGINT)) AS t0,
                   ep.epoch + to_milliseconds(CAST(d.stop AS BIGINT)) AS t1, {lossy.format('d.item')} AS nm, d.itemid,
                   d.dose, d.rate, d.iscontinuous, d.action, o.target, l.continuous, l.intermittent,
                   coalesce(r.med_route_category, '') <> '' AS route_ok, d.itemid IN ({tool_items}) AS tool_item,
                   CASE WHEN d.doserateunit IS NOT NULL AND d.dose IS NOT NULL THEN d.dose ELSE d.rate END AS seg_rate,
                   d.doserateunit IS NOT NULL AND d.dose IS NOT NULL AS by_dose,
                   CASE WHEN d.doserateunit IS NOT NULL AND d.dose IS NOT NULL THEN
                            CASE d.doseunit WHEN 'µg' THEN 'mcg' WHEN 'IE' THEN 'u' WHEN 'E' THEN 'u' ELSE d.doseunit END
                            || CASE WHEN d.doserateperkg = 1 THEN '/kg' ELSE '' END || '/'
                            || CASE d.doserateunit WHEN 'uur' THEN 'hr' WHEN 'min' THEN 'min' WHEN 'dag' THEN 'day' END
                        WHEN d.rateunit = 'ml/uur' THEN 'ml/hr' END AS seg_unit
            FROM {_pq(mirror / 'drugitems.parquet')} d JOIN {adm} a USING (admissionid) JOIN ep USING (patientid)
            LEFT JOIN {_csv(maps / 'order_categories.csv')} o ON o.ordercategory = d.ordercategory
            LEFT JOIN {_csv(maps / 'intermittent_routes.csv')} r ON CAST(r.ordercategoryid AS BIGINT) = d.ordercategoryid
            LEFT JOIN lk l ON l.txt = lower(d.item)""",
        f"""CREATE OR REPLACE TEMP TABLE ev_drug AS
            SELECT pid, time AS t, split_part(code, '//', 2) AS kind, {lossy.format("split_part(code, '//', 4)")} AS nm,
                   numeric_value AS value
            FROM ev WHERE code LIKE 'DRUG//%'""",
        # the tool names items from its mapping, not the source text ('Voriconazol (VFEND)'): our rows pair by id
        f"""CREATE OR REPLACE TEMP TABLE drug_names AS
            SELECT itemid, any_value({lossy.format('item')}) AS nm FROM {_pq(mirror / 'drugitems.parquet')} GROUP BY 1""",
        # community continuous records: a DOSE (or, without a dose rate, RATE) event matched to its raw record by
        # patient, start, item and value (k-th to k-th), for the stop time and the dose-rate unit the codes omit;
        # the record keeps its stop only when the community holds an END event there. Raw records left without an
        # event are identical to a matched one (same start, item and rate): MEDS-Transforms keeps one of them
        """CREATE OR REPLACE TEMP TABLE drug_ck AS
           SELECT *, row_number() OVER (PARTITION BY pid, t, nm, kind, CAST(value AS FLOAT) ORDER BY t) AS k
           FROM ev_drug WHERE kind IN ('DOSE', 'RATE')""",
        """CREATE OR REPLACE TEMP TABLE drug_rk AS
           SELECT *, CASE WHEN by_dose THEN 'DOSE' ELSE 'RATE' END AS kind,
                  row_number() OVER (PARTITION BY pid, t0, nm, by_dose, CAST(seg_rate AS FLOAT) ORDER BY t1) AS k,
                  coalesce(continuous, '') || ' ' || coalesce(seg_unit, '') AS key,
                  target = 'continuous' AND continuous IS NOT NULL AND coalesce(action, '') <> 'Flush'
                      AND seg_unit IS NOT NULL AND seg_rate > 0 AND t1 > t0 AS streamed
           FROM drug_raw WHERE iscontinuous = 1""",
        """CREATE OR REPLACE TEMP TABLE drug_rec AS
           SELECT r.*, c.value AS c_rate,
                  EXISTS (SELECT 1 FROM ev_drug e WHERE e.kind = 'END' AND e.pid = r.pid AND e.t = r.t1 AND e.nm = r.nm) AS c_end
           FROM drug_rk r JOIN drug_ck c ON c.pid = r.pid AND c.t = r.t0 AND c.nm = r.nm AND c.kind = r.kind
                        AND CAST(c.value AS FLOAT) = CAST(r.seg_rate AS FLOAT) AND c.k = r.k""",
        """CREATE OR REPLACE TEMP TABLE drug_merged AS
           SELECT r.pid, r.key, r.t0, r.t1, r.seg_rate AS rate
           FROM drug_rk r ANTI JOIN drug_rec m ON m.pid = r.pid AND m.t0 = r.t0 AND m.nm = r.nm AND m.kind = r.kind AND m.k = r.k
           WHERE r.streamed""",
    )
    fate = "EXISTS (SELECT 1 FROM drug_raw r WHERE r.pid = u.pid AND r.t0 = u.t AND r.nm = u.item AND ({}))"
    medication = Bridge(
        "medication_admin_intermittent",
        "SELECT pid, t, nm AS item, value FROM ev_drug WHERE kind = 'DOSE'",
        f"""SELECT h.patient_id AS pid, x.admin_dttm AS t,
                   coalesce(n.nm, {lossy.format("regexp_replace(x.med_name, '^[0-9]+: ', '')")}) AS item, x.med_dose AS value
            FROM {_hosp('medication_admin_intermittent')}
            LEFT JOIN drug_names n ON n.itemid = TRY_CAST(regexp_extract(x.med_name, '^([0-9]+): ', 1) AS BIGINT)""",
        explain_community=(
            ("continuous_record_compared_as_rate_stream", fate.format("r.iscontinuous = 1")),
            ("fluid_item_input_table", fate.format("NOT r.tool_item AND r.target = 'input'")),
            ("not_an_administration_action",
             fate.format("NOT r.tool_item AND coalesce(r.action, '') NOT IN "
                         "('Nieuwe toediening', 'Nieuwe portie', 'Herstart', 'Nieuwe zak', 'Nieuwe spuit')")),
            ("no_intermittent_category", fate.format("NOT r.tool_item AND r.intermittent IS NULL")),
            ("no_clif_intermittent_route", fate.format("NOT r.tool_item AND NOT r.route_ok")),
        ),
        explain_ours=(
            MERGED_IDENTICAL,
            # records of one item and start with one dose are one community event (MEDS keeps one of identical
            # events); ours keep each record, whose amount (administered, #91) can differ
            ("community_merges_records_of_one_dose",
             """(SELECT count(*) FROM drug_raw r WHERE r.pid = u.pid AND r.t0 = u.t AND r.nm = u.item AND r.iscontinuous = 0)
                > (SELECT count(DISTINCT CAST(r.dose AS FLOAT)) FROM drug_raw r
                   WHERE r.pid = u.pid AND r.t0 = u.t AND r.nm = u.item AND r.iscontinuous = 0)"""),
        ),
        name="medication_admin_intermittent (drugitems)")
    bridges.append(medication)
    # our continuous table: concurrent records of one drug and unit summed (the tool's rows are retracted)
    stream = Streams(
        "medication_admin_continuous (drugitems, rate streams)", "medication_admin_continuous",
        """SELECT pid, coalesce(continuous, '') || ' ' || coalesce(seg_unit, '') AS key, t0,
                  CASE WHEN c_end THEN t1 END AS t1, c_rate AS rate,
                  CASE WHEN target IS DISTINCT FROM 'continuous' THEN 'order_category_not_a_continuous_medication'
                       WHEN continuous IS NULL THEN 'no_continuous_category'
                       WHEN action = 'Flush' THEN 'flush_line_care'
                       WHEN seg_unit IS NULL THEN 'no_rate_unit'
                       WHEN NOT c_end THEN 'community_end_event_missing' END AS reason
           FROM drug_rec""",
        # our streams are per hospitalization; a patient's two admissions can overlap at a transfer (an infusion of
        # the first stops after one of the next began), so the patient's stream is the sum of the hospitalizations'
        # step functions, as the community's records are summed (#96)
        """WITH r AS (SELECT h.patient_id AS pid, x.hospitalization_id AS hid, x.med_category || ' ' || x.med_dose_unit AS key,
                             x.admin_dttm AS t, x.med_dose AS rate
                      FROM t_medication_admin_continuous x JOIN t_hospitalization h USING (hospitalization_id)),
                d AS (SELECT pid, key, t, rate - coalesce(lag(rate) OVER (PARTITION BY pid, hid, key ORDER BY t), 0) AS delta
                      FROM r)
           SELECT pid, key, t, round(sum(sum(delta)) OVER (PARTITION BY pid, key ORDER BY t ROWS UNBOUNDED PRECEDING), 9)
                      AS rate
           FROM d GROUP BY pid, key, t""",
        explain=(_merged_records("drug_merged"),))
    return DatasetBridges("aumc/MEDS_cohort", bridges, setup, streams=(stream,))


BUILDERS = {"mimic": mimic, "nwicu": nwicu, "inspire": inspire, "sicdb": sicdb, "eicu": eicu, "hirid": hirid,
            "aumc": aumc}
