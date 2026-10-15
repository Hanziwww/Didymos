"""S2b: raw CLIF (reference ETLs or our adapters) -> canonical CLIF.

Canonical CLIF = the CLIF 3.0 schemas shipped with clifpy, reached per table in this order:
0. canonical times: site-local wall-clock time as TIMESTAMPTZ labelled UTC (a reference ETL that emits true
   UTC is shifted back by the fixed offset it applied; naive timestamps are labelled UTC); the reference
   tool's rows that a supplement replaces are dropped; our supplement parts
   ``clif_<table>+<part>.parquet`` are unioned into their table;
1. clifpy's official 2.1->3.0 value crosswalk (beta tables) or its value normalizer (other tables);
2. ``value_fixes.yaml``: documented fixes for reference-ETL values the crosswalk cannot resolve (renames,
   plus table-level ``_drop_rows`` / ``_scale`` rules for duplicated or mis-scaled rows);
3. canonical conventions shared by all datasets: lab reference units and order categories
   (``lab_conventions.yaml``; a unit that is not a spelling variant of the CLIF 3.0 reference unit fails),
   blood-pressure measurement site (``vital_conventions.yaml``), medication dose units in clifpy's cleaned
   spelling, the regional (non-intravenous) route of continuous local-anaesthetic infusions, and one summed
   rate stream per continuous drug (``canonical_continuous``: start / dose_change / stop; exchanges are not
   stops);
4. exact duplicates dropped (one event recorded twice; ``DEDUP``);
5. ``ecmo_mcs`` (2.1, wide; official or CLIF-MIMIC columns) -> ``mcs`` (3.0, long form) via
   ``ecmo_mcs_to_mcs.csv``;
6. clifpy's official outlier ranges plus ``outlier_extensions.yaml`` (out of range -> NULL, never clamped;
   ``outliers.py``);
7. uniform resolution rules applied identically to every dataset:
   * vitals and invasive hemodynamics: per (hospitalization, category, blood-pressure site, clock hour) a
     single value keeps its native time; several become their median stamped at the last measurement time
     (ricu/YAIB/ICareFM hourly semantics; removes dataset-specific monitor sampling rates);
   * respiratory_support (and crrt_therapy, by CRRT mode): a row documenting a device/mode different from
     the last documented one starts a new segment, so a change is never merged with the previous state;
     within (segment, clock hour) settings keep their last value, observed values their median and
     device/mode/names their last documented value, stamped at the bucket's last time;
8. schema conformance: group columns derived from their category (``group_mappings.yaml``), every CLIF 3.0
   schema column present and cast to its type, nothing else; then a strict vocabulary check: every
   standardized column must hold CLIF 3.0 values or declared extensions.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from functools import lru_cache
from importlib import resources
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from clifpy.utils.crosswalk import BETA_TABLES, crosswalk_file_2_1_to_3_0, normalize_category_value

from icutwin.clif.conform import allowed_values, connect, extensions, table_schema
from icutwin.clif.outliers import apply_outliers
from icutwin.clif.supplements import check_supplement_fresh
from icutwin.clif.units import clean_dose_unit

log = logging.getLogger(__name__)

REPORT = "_harmonize_report.json"

_EXTRA_CATEGORY_COLS = {
    "ecmo_mcs": ("device_category", "mcs_group"),
    "input": ("input_category", "input_group"),
    "output": ("output_category", "output_group"),
}
_RESP_SETTINGS = ("fio2_set", "lpm_set", "tidal_volume_set", "resp_rate_set", "pressure_control_set",
                  "pressure_support_set", "flow_rate_set", "peak_inspiratory_pressure_set", "inspiratory_time_set",
                  "peep_set")
_RESP_OBS = ("tidal_volume_obs", "resp_rate_obs", "plateau_pressure_obs", "peak_inspiratory_pressure_obs",
             "peep_obs", "minute_vent_obs", "mean_airway_pressure_obs")


def _sql(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _yaml(name: str) -> dict:
    return yaml.safe_load(resources.files("icutwin.clif").joinpath(name).read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=1)
def value_fixes() -> dict:
    return _yaml("value_fixes.yaml")


@lru_cache(maxsize=1)
def lab_conventions() -> dict:
    return _yaml("lab_conventions.yaml")


@lru_cache(maxsize=1)
def vital_conventions() -> dict:
    return _yaml("vital_conventions.yaml")


def _case(col: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return f'"{col}"'
    whens = " ".join(f"WHEN {_lit(k)} THEN {_lit(v)}" for k, v in mapping.items())
    return f'CASE "{col}" {whens} ELSE "{col}" END AS "{col}"'


def _rewrite(con: duckdb.DuckDBPyConnection, src: Path, dst: Path, maps: dict[str, dict[str, str]]) -> None:
    names = pq.read_schema(src).names
    select = ", ".join(_case(c, maps.get(c, {})) if c in maps else f'"{c}"' for c in names)
    con.execute(f"COPY (SELECT {select} FROM read_parquet('{_sql(src)}')) TO '{_sql(dst)}' (FORMAT parquet, COMPRESSION zstd)")


def _when(regexes: dict | None) -> str:
    """Conjunction of RE2 conditions {column: regex} (NULL columns match as '')."""
    return " AND ".join(f"regexp_matches(coalesce(CAST(\"{c}\" AS VARCHAR), ''), {_lit(rx)})"
                        for c, rx in (regexes or {}).items()) or "true"


def _fix_case(col: str, fixes: dict) -> str:
    """CASE expression for value fixes; a fix is {to} or an ordered list of {to, when_regex: {col: re}}."""
    whens = []
    for src_val, spec in fixes.items():
        for fix in spec if isinstance(spec, list) else [spec]:
            cond = f'"{col}" = {_lit(src_val)} AND {_when(fix.get("when_regex"))}'
            whens.append(f"WHEN {cond} THEN {_lit(fix['to'])}")
    return f'CASE {" ".join(whens)} ELSE "{col}" END AS "{col}"'


def _apply_fixes(con: duckdb.DuckDBPyConnection, src: Path, dst: Path, table_fixes: dict) -> dict:
    """Column renames plus table-level rules ``_drop_rows`` (list of {when_regex}) and ``_scale`` (list of
    {column, multiply, when_regex, value_max}); returns the number of rows each table-level rule touched."""
    names = pq.read_schema(src).names
    scale = {}
    for rule in table_fixes.get("_scale") or []:
        cond = _when(rule.get("when_regex"))
        if rule.get("value_max") is not None:
            cond += f' AND "{rule["column"]}" <= {float(rule["value_max"])}'
        scale.setdefault(rule["column"], []).append((cond, float(rule["multiply"])))
    select = []
    for c in names:
        if c in scale:
            whens = " ".join(f'WHEN {cond} THEN "{c}" * {k}' for cond, k in scale[c])
            select.append(f'CASE {whens} ELSE "{c}" END AS "{c}"')
        elif c in table_fixes:
            select.append(_fix_case(c, table_fixes[c]))
        else:
            select.append(f'"{c}"')
    drops = [_when(rule.get("when_regex")) for rule in table_fixes.get("_drop_rows") or []]
    where = f"WHERE NOT ({' OR '.join(drops)})" if drops else ""
    counts = {}
    for i, cond in enumerate(drops):
        counts[f"_drop_rows[{i}]"] = con.execute(f"SELECT count(*) FROM read_parquet('{_sql(src)}') WHERE {cond}").fetchone()[0]
    for c, rules in scale.items():
        for i, (cond, _) in enumerate(rules):
            counts[f"_scale[{c}][{i}]"] = con.execute(f"SELECT count(*) FROM read_parquet('{_sql(src)}') WHERE {cond}").fetchone()[0]
    con.execute(f"COPY (SELECT {', '.join(select)} FROM read_parquet('{_sql(src)}') {where}) TO '{_sql(dst)}' "
                f"(FORMAT parquet, COMPRESSION zstd)")
    return counts


def _lab_reference_units() -> dict[str, str]:
    units = dict(table_schema("labs").get("lab_reference_units") or {})
    for cat, spec in ((extensions().get("labs") or {}).get("lab_category") or {}).items():
        if isinstance(spec, dict) and spec.get("unit"):
            units[cat] = spec["unit"]
    return units


def canonical_labs(con: duckdb.DuckDBPyConnection, src: Path, dst: Path) -> dict:
    """Rewrite reference_unit / lab_order_category to the canonical conventions (``lab_conventions.yaml``).

    The source unit must be the reference unit or one of its spellings (clifpy ``allowed_unit_variants`` plus
    declared equivalences); a missing unit is accepted. Anything else is a unit error of the source ETL and is
    returned under ``violations`` (strict harmonization fails on it).
    """
    conv = lab_conventions()
    ref = _lab_reference_units()
    variants = table_schema("labs").get("allowed_unit_variants") or {}
    rows = []
    for cat, unit in ref.items():
        ok = {unit.lower(), *(v.lower() for v in variants.get(unit.lower(), []))}
        for eq_unit, spec in (conv.get("equivalent_units") or {}).items():
            if eq_unit == unit.lower() and cat in spec["categories"]:
                ok |= {u.lower() for u in spec["units"]}
        rows += [(cat, u) for u in sorted(ok)]
    con.execute("CREATE OR REPLACE TEMP TABLE lab_unit_ok (lab_category VARCHAR, unit VARCHAR)")
    con.executemany("INSERT INTO lab_unit_ok VALUES (?, ?)", rows)
    order = conv["order_category"]
    keep = conv.get("keep_order_categories") or {}
    con.execute("CREATE OR REPLACE TEMP TABLE lab_canon (lab_category VARCHAR, ref_unit VARCHAR, "
                "order_category VARCHAR, keep VARCHAR[])")
    con.executemany("INSERT INTO lab_canon VALUES (?, ?, ?, ?)",
                    [(c, ref.get(c), order.get(c), keep.get(c, [])) for c in sorted(set(ref) | set(order))])
    # whitespace (incl. stray tabs, as in CLIF-MIMIC '10^3/µL\t') collapses to single spaces, as in the variants
    spelled = "lower(trim(regexp_replace(l.reference_unit, '\\s+', ' ', 'g')))"
    norm = f"nullif(nullif({spelled}, ''), 'unk')"
    base = f"FROM read_parquet('{_sql(src)}') l LEFT JOIN lab_canon c ON c.lab_category = l.lab_category"
    bad_units = con.execute(f"""
        SELECT l.lab_category || ' [' || coalesce(l.reference_unit, 'NULL') || ']', count(*) {base}
        WHERE {norm} IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM lab_unit_ok u WHERE u.lab_category = l.lab_category AND u.unit = {spelled})
        GROUP BY 1""").fetchall()
    no_order = con.execute(f"SELECT l.lab_category, count(*) {base} WHERE c.order_category IS NULL GROUP BY 1").fetchall()
    missing_unit = con.execute(f"SELECT count(*) {base} WHERE {norm} IS NULL").fetchone()[0]
    names = pq.read_schema(src).names
    select = []
    for col in names:
        if col == "reference_unit":
            select.append('coalesce(c.ref_unit, l.reference_unit) AS reference_unit')
        elif col == "lab_order_category":
            select.append("CASE WHEN list_contains(c.keep, l.lab_order_category) THEN l.lab_order_category "
                          "ELSE coalesce(c.order_category, l.lab_order_category) END AS lab_order_category")
        else:
            select.append(f'l."{col}"')
    con.execute(f"COPY (SELECT {', '.join(select)} {base}) TO '{_sql(dst)}' (FORMAT parquet, COMPRESSION zstd)")
    violations = {}
    if bad_units:
        violations["reference_unit"] = {k: n for k, n in bad_units}
    if no_order:
        violations["lab_order_category"] = {f"no canonical order category for {k}": n for k, n in no_order}
    return {"rule": "lab_conventions", "rows_missing_unit": missing_unit, "violations": violations}


def canonical_vitals(con: duckdb.DuckDBPyConnection, src: Path, dst: Path) -> dict:
    """Resolve the blood-pressure site (``vital_conventions.yaml``) into meas_site_name of sbp/dbp/map."""
    conv = vital_conventions()
    names = pq.read_schema(src).names
    bp = ", ".join(_lit(c) for c in conv["bp_categories"])
    whens = " ".join(f"WHEN regexp_matches(lower(coalesce(\"{r['column']}\", '')), {_lit(r['regex'])}) "
                     f"THEN {_lit(r['site'])}" for r in conv["bp_site_rules"] if r["column"] in names)
    native = "meas_site_name" if "meas_site_name" in names else "NULL"
    site = f"CASE WHEN vital_category IN ({bp}) THEN (CASE {whens} END) ELSE {native} END AS meas_site_name"
    select = ", ".join([f'"{c}"' for c in names if c != "meas_site_name"] + [site])
    con.execute(f"COPY (SELECT {select} FROM read_parquet('{_sql(src)}')) TO '{_sql(dst)}' (FORMAT parquet, COMPRESSION zstd)")
    sites = con.execute(f"SELECT coalesce(meas_site_name, 'unresolved'), count(*) FROM read_parquet('{_sql(dst)}') "
                        f"WHERE vital_category IN ({bp}) GROUP BY 1").fetchall()
    return {"rule": "bp_site", "bp_sites": dict(sites)}


# continuous local-anaesthetic infusions are regional (epidural, perineural, wound); CLIF has no such route
REGIONAL_INFUSIONS = ("bupivacaine", "ropivacaine")
ROUTE_BASES = ("reported", "reported_class", "inferred", "assumed", "verified")


@lru_cache(maxsize=1)
def route_basis_rules() -> tuple[dict, ...]:
    """``route_basis.csv``: per (dataset, table, source route name) the basis of the route category, first match
    wins. Name patterns: ``*`` any name, ``<none>`` no name, ``re:<regex>``, else the exact name."""
    text = resources.files("icutwin.clif").joinpath("route_basis.csv").read_text(encoding="utf-8")
    rules = tuple(csv.DictReader(text.splitlines()))
    bad = [r for r in rules if r["basis"] not in ROUTE_BASES]
    if bad:
        raise ValueError(f"route_basis.csv: unknown basis in {bad}")
    return rules


def route_basis_sql(dataset: str, table: str, name: str, category: str) -> str:
    """SQL of ``med_route_basis`` for a dataset's table: NULL without a route category, else the first rule's basis
    (NULL when none matches; ``canonical_medications`` refuses that)."""
    whens = [f"WHEN ({category}) IS NULL THEN NULL"]
    for r in route_basis_rules():
        if r["dataset"] not in (dataset, "*") or r["table"] not in (table, "*"):
            continue
        pat = r["med_route_name"]
        cond = (f"({name}) IS NULL" if pat == "<none>" else f"({name}) IS NOT NULL" if pat == "*"
                else f"regexp_matches({name}, {_lit(pat[3:])})" if pat.startswith("re:") else f"({name}) = {_lit(pat)}")
        whens.append(f"WHEN {cond} THEN {_lit(r['basis'])}")
    return "CASE " + " ".join(whens) + " END"


def canonical_medications(con: duckdb.DuckDBPyConnection, src: Path, dst: Path, table: str,
                          dataset: str | None = None, strict: bool = True) -> dict:
    """Medication conventions for every dataset:
    * dose units in clifpy's cleaned spelling (``clean_dose_unit``), so one unit has one name across
      datasets and the official (category, unit) outlier ranges apply to all of them;
    * continuous local-anaesthetic infusions get no route category: they are regional, never intravenous,
      and sources without a route (SICdb, INSPIRE) would otherwise say 'iv';
    * ``med_route_basis`` (a declared extension column) says what the route category rests on: the source's
      route (``reported``), a route class or order category it maps from (``reported_class``), a rule on drug
      and dose where the source names no route (``inferred``), or nothing but the record type (``assumed``: an
      infusion). ``verified`` is reserved for clinically verified routes: no source supplies one, so source
      routes are never presented as verified.
    (med_group is derived from med_category in ``conform_to_schema``.)"""
    names = pq.read_schema(src).names
    regional = ", ".join(f"'{c}'" for c in REGIONAL_INFUSIONS)
    is_regional = f"m.med_category IN ({regional})" if table == "medication_admin_continuous" else "false"
    units, factors = {}, {}
    if "med_dose_unit" in names:
        for (u,) in con.execute(f"SELECT DISTINCT med_dose_unit FROM read_parquet('{_sql(src)}') "
                                f"WHERE med_dose_unit IS NOT NULL").fetchall():
            clean, factor = clean_dose_unit(u)
            if clean != u:
                units[u] = clean
            if factor != 1.0:
                factors[u] = factor
    scale = " ".join(f"WHEN m.med_dose_unit = {_lit(u)} THEN m.med_dose * {f}" for u, f in factors.items())
    category = (f"CASE WHEN {is_regional} THEN NULL ELSE m.med_route_category END"
                if "med_route_category" in names else "CAST(NULL AS VARCHAR)")
    route_name = "m.med_route_name" if "med_route_name" in names else "CAST(NULL AS VARCHAR)"
    basis = route_basis_sql(dataset or "", table, route_name, category)
    cols = []
    for c in names:
        if c == "med_route_category":
            cols.append(f"{category} AS med_route_category")
        elif c == "med_route_name":
            cols.append(f"CASE WHEN {is_regional} THEN 'regional' ELSE m.med_route_name END AS med_route_name")
        elif c == "med_dose_unit" and units:
            cols.append(_case("med_dose_unit", units))
        elif c == "med_dose" and factors:
            cols.append(f"CASE {scale} ELSE m.med_dose END AS med_dose")
        elif c == "med_route_basis":
            continue
        else:
            cols.append(f'm."{c}"')
    cols.append(f"{basis} AS med_route_basis")
    base = f"FROM read_parquet('{_sql(src)}') m"
    regional_rows = con.execute(f"SELECT count(*) {base} WHERE {is_regional}").fetchone()[0]
    unmatched = con.execute(f"""SELECT {route_name}, count(*) {base}
                                WHERE ({category}) IS NOT NULL AND ({basis}) IS NULL GROUP BY 1 ORDER BY 1""").fetchall()
    if unmatched and strict:
        raise ValueError(f"{dataset}.{table}: routes without a declared basis (route_basis.csv): {unmatched[:5]}")
    bases = dict(con.execute(f"SELECT {basis}, count(*) {base} GROUP BY 1").fetchall())
    con.execute(f"COPY (SELECT {', '.join(cols)} {base}) TO '{_sql(dst)}' (FORMAT parquet, COMPRESSION zstd)")
    return {"rule": "medication_conventions", "regional_infusion_rows": regional_rows,
            "dose_units_renamed": units, "dose_units_scaled": factors,
            "route_basis": {str(k): v for k, v in sorted(bases.items(), key=lambda kv: str(kv[0]))},
            **({"route_basis_undeclared": dict(unmatched)} if unmatched else {})}


CONTINUOUS_EXCHANGE_MINUTES = 15
_C_KEY = ("hospitalization_id", "med_category", "med_dose_unit")


def canonical_continuous(con: duckdb.DuckDBPyConnection, src: Path, dst: Path) -> dict:
    """One rate stream per (hospitalization, drug, dose unit), for every dataset.

    * Concurrent streams are summed: orders where the source identifies them (``med_order_id``: CLIF-MIMIC
      emits each MetaVision order separately, 10% of its starts begin while another order of the drug runs);
      our adapters already sum concurrent syringes. An order whose events repeat another order's exactly
      (same times, doses and actions) is one infusion documented twice and counts once (MIMIC: three
      norepinephrine orders with identical start, end and rate). A stream's rate is its last
      start/dose_change/going dose, 0 after a stop; events are emitted where the summed rate changes: start (from
      0), dose_change, stop (to 0).
    * A restart within ``CONTINUOUS_EXCHANGE_MINUTES`` of a stop is a bag or syringe exchange, not a stop:
      both events go, and the restart becomes a dose_change when its rate differs (HiRID 78% and MIMIC 49% of
      restarts follow their stop within 15 min, most within seconds).
    * Presence-only streams (a non-stop row without dose: NWICU confirmations, INSPIRE ward records) pass
      through unchanged.
    * Rates are summed in double precision whatever the source type: in float32 (CLIF-MIMIC writes FLOAT doses)
      a rate change leaves a 1e-9 residue, so a later stop is no stop (a 'dose_change' to 0) and the next
      start no start.
    The summed stream has no single order: ``med_order_id`` is NULL on rated rows.
    """
    key = ", ".join(_C_KEY)
    names = pq.read_schema(src).names
    order_col = "med_order_id" if "med_order_id" in names else "NULL"
    # a total row order (only identical rows tie), so the row carried into each event is the same in every run
    rid_order = ", ".join([*_C_KEY, "admin_dttm", *(f'"{c}"' for c in names if c not in (*_C_KEY, "admin_dttm"))])
    con.execute(f"""CREATE OR REPLACE TEMP TABLE c_in AS
                    SELECT *, row_number() OVER (ORDER BY {rid_order}) AS _rid FROM read_parquet('{_sql(src)}')""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE c_presence AS
                    SELECT DISTINCT {key} FROM c_in WHERE med_dose IS NULL AND mar_action_category IS DISTINCT FROM 'stop'""")
    w_key = f"PARTITION BY {key} ORDER BY admin_dttm"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE c_events AS
        WITH r0 AS (
            SELECT c.*, coalesce(CAST({order_col} AS VARCHAR), '') AS _stream,
                   CASE WHEN mar_action_category = 'stop' THEN 0.0 ELSE CAST(med_dose AS DOUBLE) END AS _rate,
                   CASE WHEN mar_action_category = 'stop' THEN 0 ELSE 1 END AS _act
            FROM c_in c ANTI JOIN c_presence USING ({key})),
        sig AS (   -- an order's events; an order repeating another's exactly is the same infusion
            SELECT {key}, _stream,
                   string_agg(CAST(admin_dttm AS VARCHAR) || '|' || CAST(_rate AS VARCHAR) || '|' || CAST(_act AS VARCHAR),
                              ';' ORDER BY admin_dttm, _act, _rate) AS _sig
            FROM r0 GROUP BY {key}, _stream),
        dup AS (SELECT {key}, _stream FROM (
                    SELECT *, row_number() OVER (PARTITION BY {key}, _sig ORDER BY _stream) AS _k FROM sig) WHERE _k > 1),
        r AS (SELECT * FROM r0 ANTI JOIN dup USING ({key}, _stream)),
        d AS (SELECT *, _rate - coalesce(lag(_rate) OVER (PARTITION BY {key}, _stream ORDER BY admin_dttm, _act, _rid), 0)
                        AS _delta FROM r),
        t AS (SELECT {key}, admin_dttm, sum(_delta) AS _delta, min(_rid) AS _rid0, count(*) AS _n
              FROM d GROUP BY {key}, admin_dttm),
        run AS (SELECT *, round(sum(_delta) OVER ({w_key} ROWS UNBOUNDED PRECEDING), 9) AS _total FROM t),
        chg AS (SELECT *, lag(_total, 1, 0) OVER ({w_key}) AS _prev FROM run),
        ev AS (SELECT * FROM chg WHERE _total IS DISTINCT FROM _prev),
        ev2 AS (
            SELECT *, lag(_total, 1, 0) OVER ({w_key}) AS _p_total, lag(_total, 2, 0) OVER ({w_key}) AS _p2_total,
                   lag(admin_dttm) OVER ({w_key}) AS _p_t, lead(admin_dttm) OVER ({w_key}) AS _n_t
            FROM ev),
        ev3 AS (
            SELECT *, (_total <= 0 AND _n_t IS NOT NULL
                       AND _n_t - admin_dttm <= INTERVAL {CONTINUOUS_EXCHANGE_MINUTES} MINUTE) AS _bridged_stop,
                      (_p_total <= 0 AND _p_t IS NOT NULL
                       AND admin_dttm - _p_t <= INTERVAL {CONTINUOUS_EXCHANGE_MINUTES} MINUTE) AS _bridged_start
            FROM ev2)
        SELECT * FROM ev3""")
    bridged = con.execute("SELECT count(*) FILTER (WHERE _bridged_stop) FROM c_events").fetchone()[0]
    duplicates = con.execute(f"""
        WITH r0 AS (SELECT c.*, coalesce(CAST({order_col} AS VARCHAR), '') AS _stream,
                           CASE WHEN mar_action_category = 'stop' THEN 0.0 ELSE CAST(med_dose AS DOUBLE) END AS _rate,
                           CASE WHEN mar_action_category = 'stop' THEN 0 ELSE 1 END AS _act
                    FROM c_in c ANTI JOIN c_presence USING ({key})),
        sig AS (SELECT {key}, _stream, string_agg(CAST(admin_dttm AS VARCHAR) || '|' || CAST(_rate AS VARCHAR) || '|'
                                                  || CAST(_act AS VARCHAR), ';' ORDER BY admin_dttm, _act, _rate) AS _sig
                FROM r0 GROUP BY {key}, _stream)
        SELECT count(*) - count(DISTINCT ({key}, _sig)) FROM sig""").fetchone()[0]
    multi = con.execute(f"""SELECT count(*) FROM (SELECT {key} FROM c_in ANTI JOIN c_presence USING ({key})
                            GROUP BY {key} HAVING count(DISTINCT coalesce(CAST({order_col} AS VARCHAR), '')) > 1)""").fetchone()[0]
    other = [c for c in names if c not in (*_C_KEY, "admin_dttm", "med_dose", "mar_action_category",
                                           "mar_action_name", "med_order_id")]
    cols = ", ".join([*(f"e.{c}" for c in _C_KEY), "e.admin_dttm", *(f'i."{c}"' for c in other)])
    rated = f"""
        SELECT {cols}, greatest(e._total, 0) AS med_dose,
               CASE WHEN e._total <= 0 THEN 'stop' WHEN e._p_total <= 0 AND NOT e._bridged_start THEN 'start'
                    ELSE 'dose_change' END AS mar_action_category,
               CASE WHEN e._bridged_start THEN 'restart within {CONTINUOUS_EXCHANGE_MINUTES} min: bag or syringe exchange'
                    ELSE CAST(i.mar_action_name AS VARCHAR) END AS mar_action_name,
               CAST(NULL AS VARCHAR) AS med_order_id
        FROM c_events e JOIN c_in i ON i._rid = e._rid0
        WHERE NOT e._bridged_stop AND NOT (e._bridged_start AND e._total = e._p2_total)"""
    passthrough = f"SELECT * EXCLUDE (_rid) FROM c_in SEMI JOIN c_presence USING ({key})"
    sel = ", ".join(f'"{c}"' for c in names)
    con.execute(f"""COPY (SELECT {sel} FROM ({rated}) UNION ALL BY NAME SELECT {sel} FROM ({passthrough}))
                    TO '{_sql(dst)}' (FORMAT parquet, COMPRESSION zstd)""")
    n_in, n_out = pq.ParquetFile(src).metadata.num_rows, pq.ParquetFile(dst).metadata.num_rows
    return {"rule": "continuous_rate_stream", "rows_in": n_in, "rows_out": n_out, "exchanges_bridged": bridged,
            "streams_from_several_orders": multi, "duplicate_orders_dropped": duplicates}


def _normalize_extra(con: duckdb.DuckDBPyConnection, src: Path, table: str) -> dict[str, dict[str, str]]:
    maps = {}
    names = pq.read_schema(src).names
    for col in _EXTRA_CATEGORY_COLS.get(table, ()):
        if col not in names:
            continue
        distinct = [r[0] for r in con.execute(f'SELECT DISTINCT "{col}" FROM read_parquet(\'{_sql(src)}\')').fetchall()]
        maps[col] = {v: normalize_category_value(v) for v in distinct if isinstance(v, str) and normalize_category_value(v) != v}
    return maps


def _hourly_median(con: duckdb.DuckDBPyConnection, src: Path, dst: Path, category: str, value: str, name: str,
                   site: str | None) -> dict:
    """Median per (hospitalization, category[, site], clock hour), stamped at the last measurement time; the
    other columns come from the hour's last row (rows recorded at the same last time ordered by name and value,
    so two items charting one value at one time give the same row in every run)."""
    cols = pq.read_schema(src).names
    passthrough = [c for c in cols if c not in ("hospitalization_id", "recorded_dttm", category, value, name)
                   and c != site]
    key = ", ".join(["recorded_dttm", name, value, *(f'"{c}"' for c in passthrough)])
    extra = "".join(f', last("{c}" ORDER BY {key}) AS "{c}"' for c in passthrough)
    group_site = f", {site}" if site else ""
    out_site = f", {site}" if site else ""
    con.execute(f"""
        COPY (
            SELECT hospitalization_id,
                   max(recorded_dttm)            AS recorded_dttm,
                   last({name} ORDER BY {key}) AS {name},
                   {category},
                   median({value})               AS {value}{out_site}{extra}
            FROM read_parquet('{_sql(src)}')
            WHERE {value} IS NOT NULL
            GROUP BY hospitalization_id, {category}{group_site}, date_trunc('hour', recorded_dttm)
            ORDER BY hospitalization_id, recorded_dttm, {category}
        ) TO '{_sql(dst)}' (FORMAT parquet, COMPRESSION zstd)""")
    return {"rule": "hourly_median", "rows_in": pq.ParquetFile(src).metadata.num_rows,
            "rows_out": pq.ParquetFile(dst).metadata.num_rows}


def hourly_vitals(con: duckdb.DuckDBPyConnection, src: Path, dst: Path) -> dict:
    """Monitor-density rule on ``clif_vitals``; blood pressures are kept apart by canonical site.

    Expects ``canonical_vitals`` to have resolved the BP site; other categories merge across sites and keep
    the site of the hour's last measurement.
    """
    stage = dst.with_name(dst.stem + "_site.parquet")
    bp = ", ".join(_lit(c) for c in vital_conventions()["bp_categories"])
    con.execute(f"""COPY (SELECT *, CASE WHEN vital_category IN ({bp}) THEN meas_site_name END AS _bp_site
                          FROM read_parquet('{_sql(src)}')) TO '{_sql(stage)}' (FORMAT parquet, COMPRESSION zstd)""")
    tmp = dst.with_name(dst.stem + "_agg.parquet")
    stats = _hourly_median(con, stage, tmp, "vital_category", "vital_value", "vital_name", "_bp_site")
    con.execute(f"""COPY (SELECT * EXCLUDE (_bp_site) FROM read_parquet('{_sql(tmp)}'))
                    TO '{_sql(dst)}' (FORMAT parquet, COMPRESSION zstd)""")
    stage.unlink()
    tmp.unlink()
    stats["rows_in"] = pq.ParquetFile(src).metadata.num_rows
    return stats


def hourly_hemodynamics(con: duckdb.DuckDBPyConnection, src: Path, dst: Path) -> dict:
    """Monitor-density rule on ``clif_invasive_hemodynamics`` (same semantics as vitals)."""
    return _hourly_median(con, src, dst, "measurement_category", "measurement_value", "measurement_name", None)


def _hourly_change_points(con: duckdb.DuckDBPyConnection, src: Path, dst: Path, state_cols: tuple[str, ...],
                          median_cols: tuple[str, ...]) -> dict:
    """Keep state change points; collapse the rest of each clock hour (observations: median, others: last).

    A change point is a row documenting a state (e.g. device or mode) different from the last documented
    one; rows that document no state (settings or observations only) never start a segment. Rows sharing a
    timestamp are ordered with state-less rows first, so a same-time state row decides the state.
    """
    cols = pq.read_schema(src).names
    states = [c for c in state_cols if c in cols]
    keep = [c for c in cols if c not in ("hospitalization_id", "recorded_dttm")]
    agg = [f'median("{c}") AS "{c}"' if c in median_cols
           else f'last("{c}" ORDER BY recorded_dttm, "{c}") FILTER (WHERE "{c}" IS NOT NULL) AS "{c}"' for c in keep]
    order = ", ".join(["recorded_dttm", *(f'"{c}" NULLS FIRST' for c in states)])
    prev = ", ".join(f'last_value("{c}" IGNORE NULLS) OVER w AS "_prev_{c}"' for c in states) or "NULL AS _none"
    changed = " OR ".join(f'("{c}" IS NOT NULL AND "{c}" IS DISTINCT FROM "_prev_{c}")' for c in states) or "false"
    con.execute(f"""
        COPY (
            WITH f AS (   -- last documented state before each row
                SELECT *, {prev}
                FROM read_parquet('{_sql(src)}')
                WINDOW w AS (PARTITION BY hospitalization_id ORDER BY {order}
                             ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)),
            b AS (
                SELECT *, sum(CAST(({changed}) AS INTEGER)) OVER (PARTITION BY hospitalization_id ORDER BY {order}
                          ROWS UNBOUNDED PRECEDING) AS seg
                FROM f)
            SELECT hospitalization_id, max(recorded_dttm) AS recorded_dttm, {', '.join(agg)}
            FROM b
            GROUP BY hospitalization_id, seg, date_trunc('hour', recorded_dttm)
            ORDER BY hospitalization_id, recorded_dttm
        ) TO '{_sql(dst)}' (FORMAT parquet, COMPRESSION zstd)""")
    return {"rule": "change_points_plus_hourly", "rows_in": pq.ParquetFile(src).metadata.num_rows,
            "rows_out": pq.ParquetFile(dst).metadata.num_rows}


def hourly_respiratory(con: duckdb.DuckDBPyConnection, src: Path, dst: Path) -> dict:
    """Device/mode change points kept; per clock hour settings keep their last value, observations their median."""
    return _hourly_change_points(con, src, dst, ("device_category", "mode_category"), _RESP_OBS)


def hourly_crrt(con: duckdb.DuckDBPyConnection, src: Path, dst: Path) -> dict:
    """CRRT mode change points kept; per clock hour the configured flows keep their last value."""
    return _hourly_change_points(con, src, dst, ("crrt_mode_category",), ())


# 2.1 setting columns (official wide table, CLIF-MIMIC's variant) -> CLIF 3.0 setting_category; the 2.1
# device_rate depends on the device (see ecmo_mcs_to_mcs)
_MCS_SETTINGS = (("device_rate", None), ("flow", "flow"), ("sweep", "sweep"), ("sweep_set", "sweep"),
                 ("fdO2", "fdo2"), ("fdo2_set", "fdo2"))
_MCS_LOOKUPS = ("mcs_group", "device_category", "_device_context", "ecmo_configuration_category",
                "control_parameter_category")


@lru_cache(maxsize=1)
def _mcs_crosswalk() -> tuple[dict, ...]:
    text = resources.files("icutwin.clif").joinpath("ecmo_mcs_to_mcs.csv").read_text(encoding="utf-8")
    return tuple(csv.DictReader(text.splitlines()))


def _mcs_lookup(con: duckdb.DuckDBPyConnection, name: str, rows: list[tuple]) -> None:
    con.execute(f"CREATE OR REPLACE TEMP TABLE {name} (v VARCHAR, support VARCHAR, device VARCHAR, config VARCHAR, "
                f"setting VARCHAR)")
    if rows:
        con.executemany(f"INSERT INTO {name} VALUES (?, ?, ?, ?, ?)", rows)


def ecmo_mcs_to_mcs(con: duckdb.DuckDBPyConnection, src: Path, dst: Path) -> dict:
    """CLIF 2.1 wide ``ecmo_mcs`` -> CLIF 3.0 long ``mcs``: one row per recorded setting, else one presence row.

    Reads the official 2.1 columns and CLIF-MIMIC's variant (``control_parameter_*``, ``sweep_set``,
    ``fdo2_set``, ``ecmo_configuration_category``, ``_device_context``). Categories come from
    ``ecmo_mcs_to_mcs.csv``, matched case-insensitively (``device_name`` rows are regexes; first match wins):
    support from the group, else the device, the item context or an ECMO configuration; device from the
    device category, else the item context; configuration from the ECMO configuration column, else the
    device category, else the device name. The 2.1 ``device_rate`` is the pump speed, except the Impella
    P-level (``performance``) and the IABP assist ratio (``frequency``).
    """
    names = pq.read_schema(src).names
    xw = _mcs_crosswalk()
    for col in _MCS_LOOKUPS:
        _mcs_lookup(con, f"xw_{col.strip('_')}", [
            (r["value_21"].lower(), r["support_category"] or None, r["device_category"] or None,
             r["config_category"] or None, r["setting_category"] or None) for r in xw if r["column"] == col])
    resolved = []
    if "device_name" in names:
        patterns = [(re.compile(r["value_21"], re.I), r["config_category"]) for r in xw if r["column"] == "device_name"]
        for (v,) in con.execute(f"SELECT DISTINCT lower(device_name) FROM read_parquet('{_sql(src)}') "
                                f"WHERE device_name IS NOT NULL").fetchall():
            hit = next((cfg for p, cfg in patterns if p.search(v)), None)
            if hit:
                resolved.append((v, None, None, hit, None))
    _mcs_lookup(con, "xw_device_name", resolved)
    joins, sel = [], {}
    for col in (*_MCS_LOOKUPS[:4], "device_name"):
        alias = f"x_{col.strip('_')}"
        if col in names:
            joins.append(f'LEFT JOIN xw_{col.strip("_")} {alias} ON {alias}.v = lower(trim(s."{col}"))')
        sel[col] = alias if col in names else None
    a = {k: (lambda f, al=al: f"{al}.{f}" if al else "NULL") for k, al in sel.items()}
    support = (f"coalesce({a['mcs_group']('support')}, {a['device_category']('support')}, "
               f"{a['_device_context']('support')}, {a['ecmo_configuration_category']('support')})")
    device = f"coalesce({a['device_category']('device')}, {a['_device_context']('device')})"
    config = (f"coalesce({a['ecmo_configuration_category']('config')}, {a['device_category']('config')}, "
              f"{a['device_name']('config')})")
    col = lambda c: f's."{c}"' if c in names else "NULL"  # noqa: E731
    settings = [(c, cat) for c, cat in _MCS_SETTINGS if c in names]
    keep = ", ".join(f's."{c}"' for c, _ in settings)
    control = "control_parameter_value" in names
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE mcs_base AS
        SELECT s.hospitalization_id, s.recorded_dttm, CAST({col('mcs_group')} AS VARCHAR) AS support_name,
               CAST({support} AS VARCHAR) AS support_category, CAST({col('device_name')} AS VARCHAR) AS device_name,
               CAST({device} AS VARCHAR) AS device_category,
               CAST({col('ecmo_configuration_category')} AS VARCHAR) AS config_name,
               CAST({config} AS VARCHAR) AS config_category,
               CAST({col('control_parameter_name')} AS VARCHAR) AS control_name,
               CAST({col('control_parameter_category')} AS VARCHAR) AS control_category,
               CAST({col('control_parameter_value')} AS DOUBLE) AS control_value{', ' + keep if keep else ''}
        FROM read_parquet('{_sql(src)}') s {' '.join(joins)}""")
    common = ("hospitalization_id, recorded_dttm, support_name, support_category, device_name, device_category, "
              "config_name, config_category")
    parts = []
    for c, cat in settings:
        cat_sql = (f"'{cat}'" if cat else "CASE WHEN device_category = 'impella' THEN 'performance' "
                                          "WHEN support_category = 'ext_iabp' THEN 'frequency' ELSE 'rpm' END")
        parts.append(f"SELECT {common}, '{c}' AS setting_name, {cat_sql} AS setting_category, "
                     f'CAST("{c}" AS DOUBLE) AS setting_value FROM mcs_base WHERE "{c}" IS NOT NULL')
    if control:
        parts.append(f"SELECT {common}, b.control_name AS setting_name, "
                     f"coalesce(p.setting, lower(b.control_category)) AS setting_category, b.control_value AS setting_value "
                     f"FROM mcs_base b LEFT JOIN xw_control_parameter_category p ON p.v = lower(b.control_category) "
                     f"WHERE b.control_value IS NOT NULL")
    empty = " AND ".join([f'"{c}" IS NULL' for c, _ in settings] + (["control_value IS NULL"] if control else [])) or "true"
    parts.append(f"SELECT {common}, NULL AS setting_name, NULL AS setting_category, NULL AS setting_value "
                 f"FROM mcs_base WHERE {empty}")
    con.execute(f"""COPY (SELECT * FROM ({' UNION ALL '.join(parts)}) ORDER BY hospitalization_id, recorded_dttm,
                                 setting_category NULLS FIRST)
                    TO '{_sql(dst)}' (FORMAT parquet, COMPRESSION zstd)""")
    unresolved = con.execute(f"""SELECT count(*) FILTER (WHERE support_category IS NULL),
                                        count(*) FILTER (WHERE device_category IS NULL),
                                        count(*) FILTER (WHERE config_category IS NULL)
                                 FROM read_parquet('{_sql(dst)}')""").fetchone()
    return {"rule": "ecmo_mcs_2.1_to_mcs_3.0", "rows_in": pq.ParquetFile(src).metadata.num_rows,
            "rows_out": pq.ParquetFile(dst).metadata.num_rows,
            "rows_without": dict(zip(("support_category", "device_category", "config_category"), unresolved))}


# Exact duplicates: rows that agree on every clinically meaningful column are one event recorded twice
# (HiRID re-entries with a new entry time, INSPIRE ward rows repeated up to 5x, CLIF-MIMIC lab copies from
# labevents and chartevents, repeated AUMC diagnosis items). Names are not part of the identity except where
# they separate sources (two drains, two infusion lines); the kept row is the first by ``order``.
DEDUP = {
    "labs": (("hospitalization_id", "lab_collect_dttm", "lab_category", "lab_value_numeric", "lab_value"),
             "lab_result_dttm NULLS LAST, lab_order_dttm NULLS LAST, lab_name"),
    "patient_assessments": (("hospitalization_id", "recorded_dttm", "assessment_category", "numerical_value",
                             "categorical_value", "text_value"), "assessment_name"),
    "medication_admin_intermittent": (("hospitalization_id", "admin_dttm", "med_category", "med_dose", "med_dose_unit",
                                       "med_route_category", "mar_action_category"), "med_name"),
    "medication_admin_continuous": (("hospitalization_id", "admin_dttm", "med_category", "med_dose", "med_dose_unit",
                                     "mar_action_category"), "med_name"),
    "position": (("hospitalization_id", "recorded_dttm", "position_category"), "position_name"),
    "hospital_diagnosis": (("hospitalization_id", "diagnosis_code", "diagnosis_code_format"),
                           "diagnosis_primary DESC NULLS LAST"),
    "patient_procedures": (("hospitalization_id", "procedure_billed_dttm", "procedure_code", "procedure_code_format"),
                           "procedure_code"),
    "code_status": (("patient_id", "start_dttm", "code_status_category"), "code_status_name"),
    "input": (("hospitalization_id", "recorded_dttm", "input_category", "input_name", "input_volume"), "input_group"),
    "output": (("hospitalization_id", "recorded_dttm", "output_category", "output_name", "output_volume"), "output_group"),
}


def drop_exact_duplicates(con: duckdb.DuckDBPyConnection, src: Path, dst: Path, table: str) -> dict | None:
    if table not in DEDUP:
        return None
    keys, order = DEDUP[table]
    names = pq.read_schema(src).names
    keys = [k for k in keys if k in names]
    order = ", ".join(o for o in order.split(", ") if o.split()[0] in names) or "1"
    part = ", ".join(f'"{k}"' for k in keys)
    con.execute(f"""COPY (SELECT * EXCLUDE (_k) FROM (
                        SELECT *, row_number() OVER (PARTITION BY {part} ORDER BY {order}) AS _k
                        FROM read_parquet('{_sql(src)}')) WHERE _k = 1)
                    TO '{_sql(dst)}' (FORMAT parquet, COMPRESSION zstd)""")
    n_in, n_out = pq.ParquetFile(src).metadata.num_rows, pq.ParquetFile(dst).metadata.num_rows
    return {"rule": "exact_duplicates", "keys": keys, "rows_dropped": n_in - n_out}


@lru_cache(maxsize=1)
def group_mappings() -> dict:
    """category -> group per (table, group column): CLIF 3.0 schema mappings, mCIDE action-group mappings and
    the groups of declared extension categories (``group_mappings.yaml``)."""
    spec = _yaml("group_mappings.yaml")
    out: dict[tuple[str, str], tuple[str, dict[str, str]]] = {}
    for table, rules in spec.items():
        for group_col, rule in rules.items():
            mapping = {}
            if rule.get("schema_mapping"):
                for k, v in (table_schema(table).get(rule["schema_mapping"]) or {}).items():
                    if v:
                        mapping[k.lower()] = v if isinstance(v, str) else v[0]
            for k, v in (rule.get("values") or {}).items():
                mapping[k.lower()] = v
            cat_col = rule["category_column"]
            for cat, ext in ((extensions().get(table) or {}).get(cat_col) or {}).items():
                if isinstance(ext, dict) and ext.get("group"):
                    mapping.setdefault(cat.lower(), ext["group"])
            out[(table, group_col)] = (cat_col, mapping)
    return out


def conform_to_schema(con: duckdb.DuckDBPyConnection, path: Path, table: str) -> dict:
    """Make the table exactly its CLIF 3.0 schema: group columns derived from their category column (one
    mapping for every dataset), every schema column present (NULL when the source has none), each cast to its
    schema type, columns outside the schema dropped (e.g. CLIF-AmsterdamUMCdb's per-row ``patient_id``)."""
    from icutwin.clif.conform import _DUCK
    names = pq.read_schema(path).names
    cols = table_schema(table)["columns"]
    schema_names = [c["name"] for c in cols]
    groups = {g: spec for (t, g), spec in group_mappings().items() if t == table}
    report: dict = {"dropped": [c for c in names if c not in schema_names],
                    "added": [c for c in schema_names if c not in names and c not in groups], "groups": {}}
    for g, (cat_col, mapping) in groups.items():
        if cat_col in names:
            con.execute(f"CREATE OR REPLACE TEMP TABLE grp_{g} (k VARCHAR, v VARCHAR)")
            con.executemany(f"INSERT INTO grp_{g} VALUES (?, ?)", sorted(mapping.items()))
    select, joins = [], []
    for c in cols:
        n, typ = c["name"], _DUCK[c["data_type"].upper()]
        if n in groups and groups[n][0] in names:
            alias = f"g_{n}"
            joins.append(f'LEFT JOIN grp_{n} {alias} ON {alias}.k = lower(t."{groups[n][0]}")')
            expr = f"{alias}.v"
        elif n in names:
            expr = f't."{n}"'
        else:
            expr = "NULL"
        select.append(f'TRY_CAST({expr} AS {typ}) AS "{n}"')
    for n in [c["name"] for c in cols if c["name"] in names and c["name"] not in groups]:
        typ = _DUCK[next(c for c in cols if c["name"] == n)["data_type"].upper()]
        lost = con.execute(f'SELECT count(*) FROM read_parquet(\'{_sql(path)}\') WHERE "{n}" IS NOT NULL '
                           f'AND TRY_CAST("{n}" AS {typ}) IS NULL').fetchone()[0]
        if lost:
            report.setdefault("uncastable_values_nulled", {})[n] = lost
    for g in groups:
        if groups[g][0] in names:
            report["groups"][g] = con.execute(
                f'SELECT count(*) FROM read_parquet(\'{_sql(path)}\') t LEFT JOIN grp_{g} m ON m.k = lower(t."{groups[g][0]}") '
                f'WHERE m.v IS DISTINCT FROM {"t." + chr(34) + g + chr(34) if g in names else "NULL"}').fetchone()[0]
    tmp = path.with_name(path.stem + "_conform.parquet")
    con.execute(f"COPY (SELECT {', '.join(select)} FROM read_parquet('{_sql(path)}') t {' '.join(joins)}) "
                f"TO '{_sql(tmp)}' (FORMAT parquet, COMPRESSION zstd)")
    tmp.replace(path)
    return report


def vocabulary_violations(con: duckdb.DuckDBPyConnection, path: Path, table: str) -> dict:
    out = {}
    names = pq.read_schema(path).names
    for col in table_schema(table)["columns"]:
        name = col["name"]
        ok = allowed_values(table, name)
        if ok is None or name not in names:
            continue
        rows = con.execute(f'SELECT "{name}", count(*) FROM read_parquet(\'{_sql(path)}\') '
                           f'WHERE "{name}" IS NOT NULL GROUP BY 1').fetchall()
        bad = {str(v): n for v, n in rows if v not in ok and str(v) not in ok}
        if bad:
            out[name] = bad
    return out


def _parts(raw_dir: Path) -> dict[str, list[Path]]:
    """Group ``clif_<table>.parquet`` with its supplement parts ``clif_<table>+<part>.parquet`` by table."""
    tables: dict[str, list[Path]] = {}
    for f in sorted(raw_dir.glob("clif_*.parquet")):
        tables.setdefault(f.stem.removeprefix("clif_").split("+", 1)[0], []).append(f)
    return tables


def _union_parts(con: duckdb.DuckDBPyConnection, parts: list[Path], dst: Path) -> Path:
    if len(parts) == 1:
        return parts[0]
    files = ", ".join(f"'{_sql(p)}'" for p in parts)
    con.execute(f"COPY (SELECT * FROM read_parquet([{files}], union_by_name=true)) TO '{_sql(dst)}' "
                f"(FORMAT parquet, COMPRESSION zstd)")
    return dst


LOSSY_TOKEN = re.compile("[\\w\ufffd]*\ufffd[\\w\ufffd]*")


def repair_lossy_text(con: duckdb.DuckDBPyConnection, src: Path, dst: Path, token_map: dict[str, str]) -> dict:
    """Repair text a reference tool decoded with the wrong encoding: every token holding U+FFFD is replaced by
    the source token it stands for (``token_map``, from the supplement's correctly decoded source vocabulary).
    Text columns only; tokens without a unique source token stay as they are (and are reported)."""
    types = {n: t for n, t, *_ in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{_sql(src)}')").fetchall()}
    maps, unresolved = {}, {}
    for col, typ in types.items():
        if typ != "VARCHAR":
            continue
        values = [r[0] for r in con.execute(f"SELECT DISTINCT \"{col}\" FROM read_parquet('{_sql(src)}') "
                                            f"WHERE strpos(\"{col}\", chr(65533)) > 0").fetchall()]
        for v in values:
            fixed = LOSSY_TOKEN.sub(lambda m: token_map.get(m.group(0), m.group(0)), v)
            if fixed != v:
                maps.setdefault(col, {})[v] = fixed
            if "\ufffd" in fixed:
                unresolved.setdefault(col, []).append(fixed)
    _rewrite(con, src, dst, maps)
    return {"rule": "lossy_text_repair", "values_repaired": {c: len(m) for c, m in maps.items()},
            "unresolved": {c: sorted(v)[:20] for c, v in unresolved.items()}}


def _retract(con: duckdb.DuckDBPyConnection, src: Path, dst: Path, rules: list[dict]) -> dict:
    """Drop the rows of a reference tool's table that a supplement replaces; returns rows dropped per rule."""
    conds = [_when(r.get("when_regex")) for r in rules]
    counts = {f"retract[{i}]": con.execute(f"SELECT count(*) FROM read_parquet('{_sql(src)}') WHERE {c}").fetchone()[0]
              for i, c in enumerate(conds)}
    con.execute(f"COPY (SELECT * FROM read_parquet('{_sql(src)}') WHERE NOT ({' OR '.join(conds)})) "
                f"TO '{_sql(dst)}' (FORMAT parquet, COMPRESSION zstd)")
    return counts


def canonical_times(con: duckdb.DuckDBPyConnection, src: Path, dst: Path, utc_offset_hours: float | None) -> dict | None:
    """Canonical time columns: site-local wall-clock time as TIMESTAMPTZ labelled UTC.

    Time-zone-aware columns of a tool that emits true UTC are shifted back by the fixed offset it applied;
    naive timestamps (CLIF-AmsterdamUMCdb) are wall-clock already and are only labelled UTC. Dates (e.g.
    ``birth_date``) are kept. Returns the converted columns, or None when nothing needs converting.
    """
    schema = pq.read_schema(src)
    aware = [f.name for f in schema if pa.types.is_timestamp(f.type) and f.type.tz is not None] if utc_offset_hours else []
    naive = [f.name for f in schema if pa.types.is_timestamp(f.type) and f.type.tz is None]
    if not aware and not naive:
        return None
    shift = f"to_minutes(CAST({round((utc_offset_hours or 0) * 60)} AS BIGINT))"
    select = ", ".join(f'("{c}" + {shift}) AS "{c}"' if c in aware
                       else f'CAST("{c}" AS TIMESTAMPTZ) AS "{c}"' if c in naive  # session time zone is UTC
                       else f'"{c}"' for c in schema.names)
    con.execute(f"COPY (SELECT {select} FROM read_parquet('{_sql(src)}')) TO '{_sql(dst)}' (FORMAT parquet, COMPRESSION zstd)")
    return {"shifted_hours": utc_offset_hours, "shifted": aware, "labelled_utc": naive}


def harmonize_dir(raw_dir: Path, out_dir: Path, temp_dir: Path, strict: bool = True,
                  utc_offset_hours: float | None = None) -> dict:
    """Harmonize one dataset's raw CLIF directory into canonical CLIF 3.0.

    Canonical timestamps are site-local wall-clock times stored as UTC (the convention of every dataset whose
    source gives local or pseudo times). A reference ETL that emits true UTC is shifted back by
    ``utc_offset_hours``, the fixed offset it applied (see ``configs/clif.yaml``). Supplement files
    ``clif_<table>+<part>.parquet`` (our own, already wall-clock) are unioned into their table.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in [*out_dir.glob("clif_*.parquet"), out_dir / REPORT]:  # never leave tables of an earlier run behind
        old.unlink(missing_ok=True)
    stage = out_dir / "_stage"
    stage.mkdir(exist_ok=True)
    con = connect(temp_dir)
    supplement = check_supplement_fresh(raw_dir)
    retract = (supplement or {}).get("retract") or {}
    text_repairs = (supplement or {}).get("text_repairs") or {}
    report: dict = {"raw_dir": str(raw_dir), "utc_offset_hours": utc_offset_hours, "tables": {}, "violations": {},
                    "supplement": supplement}
    fixes = value_fixes()
    for table, parts in _parts(raw_dir).items():
        try:
            table_schema("mcs" if table == "ecmo_mcs" else table)
        except KeyError:  # tool-specific side tables (e.g. CLIF-MIMIC patient_assessments_raw_gcs)
            report["tables"][table] = {"skipped": "no CLIF schema; not part of canonical CLIF"}
            log.info("skipped %s (no CLIF schema)", table)
            continue
        name = f"clif_{table}.parquet"
        parts_info = {p.name: pq.ParquetFile(p).metadata.num_rows for p in parts} if len(parts) > 1 else None
        # the reference tool's own file: shift true UTC back to wall-clock and drop the rows a supplement
        # replaces; supplement parts are ours (wall-clock already) and are never touched
        primary, tool_file = raw_dir / name, raw_dir / name
        retracted, repaired = None, None
        if primary in parts and not table.startswith("ext_"):
            local = stage / f"wc_{name}"
            times = canonical_times(con, tool_file, local, utc_offset_hours)
            if times:
                report.setdefault("time_columns", {})[table] = times
                tool_file = local
            if retract.get(table):
                kept = stage / f"rt_{name}"
                retracted = _retract(con, tool_file, kept, retract[table])
                tool_file = kept
            if text_repairs:
                fixed_text = stage / f"tx_{name}"
                repaired = repair_lossy_text(con, tool_file, fixed_text, text_repairs)
                tool_file = fixed_text
            parts = [tool_file if p == primary else p for p in parts]
        src = _union_parts(con, parts, stage / f"parts_{name}")
        if table.startswith("ext_"):  # our own extension tables are already canonical
            shutil.copy2(src, out_dir / name)
            report["tables"][table] = {"crosswalk": None, "parts": parts_info, "rows": pq.ParquetFile(src).metadata.num_rows}
            continue
        crosswalked = stage / f"xw_{name}"
        if table in BETA_TABLES:
            xw = crosswalk_file_2_1_to_3_0(str(src), str(crosswalked), table, backend="duckdb")
            entry = {"crosswalk": "clifpy_2.1_to_3.0", "is_complete": xw.get("is_complete"), "columns": xw.get("columns")}
        else:
            maps = _normalize_extra(con, src, table)
            _rewrite(con, src, crosswalked, maps)
            entry = {"crosswalk": "clifpy normalize_category_value", "columns": maps}
        fixed = stage / f"fx_{name}"
        table_fixes = fixes.get(table) or {}
        vf = stage / f"vf_{name}"
        entry["value_fixes"] = {col: (sorted(v) if isinstance(v, dict) else len(v)) for col, v in table_fixes.items()}
        entry["table_rules"] = _apply_fixes(con, crosswalked, vf, table_fixes)
        if parts_info:
            entry["parts"] = parts_info
        if retracted is not None:
            entry["retracted_from_tool"] = retracted
        if repaired is not None and repaired["values_repaired"]:
            entry["text_repaired"] = repaired
        canonical = stage / f"cn_{name}"
        conv_violations = {}
        if table == "labs":
            entry["conventions"] = canonical_labs(con, vf, canonical)
            conv_violations = entry["conventions"].pop("violations")
        elif table == "vitals":
            entry["conventions"] = canonical_vitals(con, vf, canonical)
        elif table in ("medication_admin_continuous", "medication_admin_intermittent", "medication_orders"):
            entry["conventions"] = canonical_medications(con, vf, canonical, table, raw_dir.name, strict)
            if table == "medication_admin_continuous":
                streamed = stage / f"rs_{name}"
                entry["rate_stream"] = canonical_continuous(con, canonical, streamed)
                canonical = streamed
        else:
            canonical = vf
        deduped = stage / f"dd_{name}"
        dedup = drop_exact_duplicates(con, canonical, deduped, table)
        if dedup is not None:
            entry["duplicates"] = dedup
            canonical = deduped
        out_table = "mcs" if table == "ecmo_mcs" else table
        if table == "ecmo_mcs":   # 2.1 wide -> 3.0 long first: the ranges are keyed by 3.0 setting categories
            converted = stage / f"mcs_{name}"
            entry["structure"] = ecmo_mcs_to_mcs(con, canonical, converted)
            canonical = converted
        entry["outliers"] = apply_outliers(con, canonical, fixed, out_table)
        final = out_dir / f"clif_{out_table}.parquet"
        if table == "vitals":
            entry["density"] = hourly_vitals(con, fixed, final)
        elif table == "invasive_hemodynamics":
            entry["density"] = hourly_hemodynamics(con, fixed, final)
        elif table == "respiratory_support":
            entry["density"] = hourly_respiratory(con, fixed, final)
        elif table == "crrt_therapy":
            entry["density"] = hourly_crrt(con, fixed, final)
        else:
            shutil.copy2(fixed, final)
        entry["schema"] = conform_to_schema(con, final, out_table)
        entry["rows"] = pq.ParquetFile(final).metadata.num_rows
        violations = {**vocabulary_violations(con, final, out_table), **conv_violations}
        if violations:
            report["violations"][out_table] = violations
        report["tables"][out_table] = entry
        log.info("harmonized %s rows=%d%s", out_table, entry["rows"], f" VIOLATIONS {violations}" if violations else "")
    shutil.rmtree(stage)
    (out_dir / REPORT).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if strict and report["violations"]:
        raise ValueError(f"{raw_dir}: values outside CLIF 3.0 + extensions after harmonization: "
                         f"{json.dumps(report['violations'])[:3000]}")
    return report
