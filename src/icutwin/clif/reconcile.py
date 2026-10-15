"""S3 reconciliation: our CLIF output against the community MEDS ETLs, event by event, per source item.

Both pipelines read the same raw rows. For each dataset a *bridge* turns
* the community events (``data/community_meds/<mode>/<dataset>``, MEDS) into (patient, time, source item, value),
* our CLIF rows (``data/clif/<mode>/<dataset>``: reference-tool output on site wall-clock time plus our
  supplement or adapter output, before harmonization) into the same keys.
Rows are paired per (patient, time, item) in order of value; a pair's values agree when ours equals the
community value times the item's unit factor (the median ratio of the item's pairs).

An unpaired row is then classified, in order: community rows outside every hospitalization of the patient
(our CLIF scope: admission - 1 day .. discharge) are out of scope; the bridge's documented reasons (a filter of
the reference tool, a deliberate rule of ours) explain the rest, each a SQL predicate over the unpaired row
``u``. What no reason explains is a defect to fix or a finding to document.

``reconciliation.{json,md}`` holds aggregates only (counts per item and reason).
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from icutwin.clif.conform import connect
from icutwin.clif.harmonize import _retract, repair_lossy_text
from icutwin.clif.supplements import check_supplement_fresh

log = logging.getLogger(__name__)

REPORT = "reconciliation"
SCOPE_BEFORE_ADMISSION = "INTERVAL 1 DAY"   # ED and pre-admission time belongs to the encounter
VALUE_RTOL = 1e-4
_SIG5 = ("(CASE WHEN {v} IS NULL OR isnan({v}) OR isinf({v}) THEN NULL WHEN {v} = 0 THEN 0 "
         "ELSE round({v}, 4 - CAST(floor(log10(abs({v}))) AS INTEGER)) END)")
# the 5 significant digits truncated: two values less than half a unit of the 5th digit apart share the rounded
# or the truncated key (a pair straddling a rounding boundary, e.g. 0.3846149 and 0.3846150, shares the truncation)
_FLOOR5 = ("(CASE WHEN {v} IS NULL OR isnan({v}) OR isinf({v}) THEN NULL WHEN {v} = 0 THEN 0 "
           "ELSE floor({v} * pow(10, 4 - CAST(floor(log10(abs({v}))) AS INTEGER))) END)")
# community values are float32: both sides are keyed on the float32 value (a double and its float32 copy can
# fall on either side of a 5-digit rounding boundary)
_F32 = "CAST(CAST({v} AS FLOAT) AS DOUBLE)"


@dataclass(frozen=True)
class Bridge:
    """One table family: SQL over ``ev`` (community events) and over ``t_<table>`` (our CLIF rows), each yielding
    ``pid`` (our ``patient_id``), ``t``, ``item``, ``value`` and any extra columns the reasons use; a
    ``pair_key`` column on both sides (e.g. an order id both keep) narrows pairing to (pid, t, item, pair_key).
    A reason is (name, SQL predicate over the unpaired row ``u``); the first that holds classifies the row.
    ``name`` keys the report when several bridges read one table (default: the table)."""
    table: str
    community: str
    ours: str
    explain_community: tuple[tuple[str, str], ...] = ()
    explain_ours: tuple[tuple[str, str], ...] = ()
    name: str = ""


@dataclass(frozen=True)
class Streams:
    """Rate streams where records cannot pair one to one (ours sums concurrent records of a drug, or charts a
    rate the community records as amounts): per (pid, key) the community's piecewise-constant rate (records
    ``t0, t1, rate``, summed where they overlap) against ours (rows ``t, rate``, each held until the key's next
    row), evaluated on the union of both breakpoints. SQL over ``ev`` / the ``t_<table>`` views yielding
    ``pid, key, t0, t1, rate, reason`` (community; a record with a reason, a documented rule of ours, is counted
    by reason and left out) and ``pid, key, t, rate`` (ours; one row per key and time). A key's streams agree
    when the time integral of their absolute difference is within ``rtol`` of the community's integral (then
    totals and running minutes agree too). Both streams are integrated inside the scope windows only (the
    patient's hospitalizations, admission - 1 day to discharge, merged where they overlap): a record running past
    a discharge counts up to it, one wholly outside is out of scope. Totals (``totals_only``) are compared over
    the whole stream of every patient with a window. ``explain`` holds (name, SQL predicate over the key row ``k``: pid,
    key, side, c_total, o_total, diff, c_minutes, o_minutes) for keys only one side has or whose streams differ."""
    name: str
    table: str
    community: str
    ours: str
    explain: tuple[tuple[str, str], ...] = ()
    rtol: float = 1e-3
    totals_only: bool = False   # the community records doses, not rate intervals: compare per-key totals only


def _p(path: Path) -> str:
    return str(path).replace("\\", "/")


def _clif_views(con: duckdb.DuckDBPyConnection, clif_dir: Path, utc_offset_hours: float | None,
                work_dir: Path) -> set[str]:
    """``t_<table>``: the tool file as harmonize reads it (on site wall-clock time, shifted back by the tool's
    UTC offset; without the rows a supplement replaces; with its lossy text repaired) unioned with our parts
    ``clif_<table>+*.parquet``; every timestamp naive; ``_src`` names the file ('tool' or the part's suffix)."""
    supplement = check_supplement_fresh(clif_dir) or {}
    retract, repairs = supplement.get("retract") or {}, supplement.get("text_repairs") or {}
    tables: dict[str, list[Path]] = {}
    for f in sorted(clif_dir.glob("clif_*.parquet")):
        tables.setdefault(f.stem.removeprefix("clif_").split("+")[0], []).append(f)
    for table, files in tables.items():
        parts = []
        for f in files:
            tool = "+" not in f.stem
            if tool and not table.startswith("ext_") and (retract.get(table) or repairs):
                work_dir.mkdir(parents=True, exist_ok=True)
                if retract.get(table):
                    _retract(con, f, work_dir / f"rt_{f.name}", retract[table])
                    f = work_dir / f"rt_{f.name}"
                if repairs:
                    repair_lossy_text(con, f, work_dir / f"tx_{f.name}", repairs)
                    f = work_dir / f"tx_{f.name}"
            schema = pq.read_schema(f)
            cols = []
            for fld in schema:
                if pa.types.is_timestamp(fld.type):
                    shift = (f" + to_minutes(CAST({round(utc_offset_hours * 60)} AS BIGINT))"
                             if tool and utc_offset_hours and fld.type.tz is not None else "")
                    cols.append(f'CAST("{fld.name}"{shift} AS TIMESTAMP) AS "{fld.name}"')
                else:
                    cols.append(f'"{fld.name}"')
            src = "tool" if tool else f.stem.split("+", 1)[1]
            parts.append(f"SELECT {', '.join(cols)}, '{src}' AS _src FROM read_parquet('{_p(f)}')")
        con.execute(f"CREATE OR REPLACE TEMP VIEW t_{table} AS {' UNION ALL BY NAME '.join(parts)}")
    return set(tables)


def reconcile(name: str, community_dir: Path, clif_dir: Path, bridges: list[Bridge],
              utc_offset_hours: float | None = None, setup: tuple[str, ...] = (), scope: str | None = None,
              work_dir: Path | None = None, streams: list[Streams] = ()) -> dict:
    """Pair community events with our CLIF rows per bridge, and compare rate streams per ``streams``. A
    community subject is our ``pid`` (the patient, or the hospitalization where the scope says so); ``setup``
    statements run first (helper views the bridges share); ``scope`` (SQL yielding pid, t0, t1) replaces the
    default scope when the bridges pair per hospitalization or on another time axis (e.g. time since case
    start); ``work_dir`` holds the tool files as harmonize reads them (default: a temporary directory)."""
    with tempfile.TemporaryDirectory() as tmp:
        return _reconcile(name, community_dir, clif_dir, bridges, utc_offset_hours, setup, scope, work_dir or Path(tmp),
                          streams)


def _reconcile(name: str, community_dir: Path, clif_dir: Path, bridges: list[Bridge], utc_offset_hours: float | None,
               setup: tuple[str, ...], scope: str | None, work_dir: Path, streams: list[Streams] = ()) -> dict:
    con = connect()
    con.execute("SET TimeZone = 'UTC'")
    have = _clif_views(con, clif_dir, utc_offset_hours, work_dir)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE ev AS
                    SELECT * REPLACE (CAST(time AS TIMESTAMP) AS time), CAST(subject_id AS VARCHAR) AS pid
                    FROM read_parquet('{_p(community_dir)}/data/**/*.parquet', union_by_name = true)""")
    for i, s in enumerate(setup):
        _timed(con, s, f"{name} setup {i}")
    default_scope = f"""SELECT CAST(patient_id AS VARCHAR) AS pid, admission_dttm - {SCOPE_BEFORE_ADMISSION} AS t0,
                               discharge_dttm AS t1 FROM t_hospitalization"""
    con.execute(f"CREATE OR REPLACE TEMP TABLE scope AS {scope or default_scope}")
    report: dict = {"dataset": name, "community_dir": str(community_dir),
                    "community_events": con.execute("SELECT count(*) FROM ev").fetchone()[0], "tables": {}}
    for b in bridges:
        key = b.name or b.table
        if b.table not in have:
            report["tables"][key] = {"skipped": "no CLIF table"}
            continue
        report["tables"][key] = _pair(con, b, work_dir / "unexplained")
        log.info("%s %s: %s", name, key, {k: v for k, v in report["tables"][key].items() if k != "items"})
    for s in streams:
        if s.table not in have:
            report.setdefault("streams", {})[s.name] = {"skipped": "no CLIF table"}
            continue
        report.setdefault("streams", {})[s.name] = _streams(con, s, work_dir / "unexplained")
        log.info("%s %s: %s", name, s.name, {k: v for k, v in report["streams"][s.name].items() if k != "keys_by_item"})
    return report


def _timed(con: duckdb.DuckDBPyConnection, sql: str, label: str, slow: float = 30.0) -> None:
    """Run ``sql``; log its duration when it is slow (a full-scale stall is otherwise silent until a bridge ends)."""
    t0 = time.time()
    con.execute(sql)
    took = time.time() - t0
    if took >= slow:
        log.info("%s took %.0f s: %s", label, took, " ".join(sql.split())[:160])


def _streams(con: duckdb.DuckDBPyConnection, s: Streams, unexplained_dir: Path | None = None) -> dict:
    # both streams are compared inside the scope windows only (a record running past a discharge counts up to it)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE sc_all AS
                    SELECT CAST(pid AS VARCHAR) AS pid, CAST(key AS VARCHAR) AS key, CAST(t0 AS TIMESTAMP) AS t0,
                           CAST(t1 AS TIMESTAMP) AS t1, CAST(rate AS DOUBLE) AS rate, CAST(reason AS VARCHAR) AS reason,
                           EXISTS (SELECT 1 FROM scope p WHERE p.pid = CAST(c.pid AS VARCHAR)
                                   AND p.t0 < coalesce(CAST(c.t1 AS TIMESTAMP), CAST(c.t0 AS TIMESTAMP))
                                   AND p.t1 >= CAST(c.t0 AS TIMESTAMP)) AS in_scope
                    FROM ({s.community}) c""")
    # totals are compared over a patient's whole stream: a community dose is a point in time, ours is spread over
    # the segment since the infusion's previous record, and a segment crossing a window's edge would split it
    inside = "pid IN (SELECT pid FROM scope)" if s.totals_only else "in_scope"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE sc AS
                    SELECT * FROM sc_all WHERE {inside} AND reason IS NULL AND t1 > t0 AND rate IS NOT NULL AND rate <> 0""")
    excluded = dict(con.execute(f"""SELECT CASE WHEN NOT ({inside}) THEN 'out_of_scope' WHEN reason IS NOT NULL THEN reason
                                                WHEN t1 IS NULL OR t1 <= t0 THEN 'no_positive_duration'
                                                ELSE 'no_rate' END, count(*)
                                    FROM sc_all WHERE NOT ({inside}) OR reason IS NOT NULL OR t1 IS NULL OR t1 <= t0
                                                      OR rate IS NULL OR rate = 0
                                    GROUP BY 1 ORDER BY 2 DESC""").fetchall())
    con.execute(f"""CREATE OR REPLACE TEMP TABLE so AS
                    SELECT CAST(pid AS VARCHAR) AS pid, CAST(key AS VARCHAR) AS key, CAST(t AS TIMESTAMP) AS t,
                           CAST(rate AS DOUBLE) AS rate FROM ({s.ours})""")
    dup = con.execute("SELECT count(*) FROM (SELECT pid, key, t FROM so GROUP BY ALL HAVING count(*) > 1)").fetchone()[0]
    if dup:
        raise ValueError(f"{s.name}: our stream has {dup} (pid, key, time) with several rows")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE c_steps AS
        SELECT pid, key, t, sum(sum(d)) OVER (PARTITION BY pid, key ORDER BY t ROWS UNBOUNDED PRECEDING) AS rate
        FROM (SELECT pid, key, t0 AS t, rate AS d FROM sc UNION ALL SELECT pid, key, t1, -rate FROM sc)
        GROUP BY pid, key, t""")
    # a patient's windows merged where they overlap (two hospitalizations' windows overlap within the time before
    # admission): a minute inside two windows counts once
    con.execute("""
        CREATE OR REPLACE TEMP TABLE scope_u AS
        WITH s AS (SELECT pid, t0, t1, max(t1) OVER (PARTITION BY pid ORDER BY t0, t1
                                                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_end
                   FROM scope),
             g AS (SELECT *, sum(CASE WHEN prev_end IS NULL OR t0 > prev_end THEN 1 ELSE 0 END)
                                 OVER (PARTITION BY pid ORDER BY t0, t1 ROWS UNBOUNDED PRECEDING) AS island FROM s)
        SELECT pid, min(t0) AS t0, max(t1) AS t1 FROM g GROUP BY pid, island""")
    if s.totals_only:
        con.execute("""CREATE OR REPLACE TEMP TABLE scope_u AS
                       SELECT DISTINCT pid, TIMESTAMP '0001-01-01' AS t0, TIMESTAMP '9999-12-31' AS t1 FROM scope""")
    # both step functions on the union of breakpoints; an interval's value is each side's last step at its start
    con.execute("""
        CREATE OR REPLACE TEMP TABLE iv AS
        WITH bp AS (SELECT DISTINCT pid, key, t FROM (SELECT pid, key, t FROM c_steps UNION ALL SELECT pid, key, t FROM so)),
             b AS (SELECT *, lead(t) OVER (PARTITION BY pid, key ORDER BY t) AS t_next FROM bp),
             v AS (SELECT b.pid, b.key, b.t, b.t_next, coalesce(round(c.rate, 9), 0) AS rc, coalesce(o.rate, 0) AS ro
                   FROM b
                   ASOF LEFT JOIN c_steps c ON c.pid = b.pid AND c.key = b.key AND b.t >= c.t
                   ASOF LEFT JOIN so o ON o.pid = b.pid AND o.key = b.key AND b.t >= o.t
                   WHERE b.t_next IS NOT NULL)
        SELECT v.pid, v.key, v.rc, v.ro, sum(epoch(least(v.t_next, w.t1) - greatest(v.t, w.t0)) / 60) AS minutes
        FROM v JOIN scope_u w ON w.pid = v.pid AND w.t0 < v.t_next AND w.t1 > v.t
        GROUP BY v.pid, v.key, v.t, v.rc, v.ro""")
    # a side holds a key when its stream integrates to more than 0 inside the windows; with ``totals_only`` the
    # difference is that of the totals (the community's records are doses at points in time)
    diff = "abs(sum(ro * minutes) - sum(rc * minutes))" if s.totals_only else "sum(abs(rc - ro) * minutes)"
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE kk AS
        WITH m AS (SELECT pid, key, sum(rc * minutes) AS c_total, sum(ro * minutes) AS o_total,
                          {diff} AS diff, coalesce(sum(minutes) FILTER (WHERE rc > 0), 0) AS c_minutes,
                          coalesce(sum(minutes) FILTER (WHERE ro > 0), 0) AS o_minutes
                   FROM iv GROUP BY 1, 2)
        SELECT pid, key, CASE WHEN c_total > 0 AND o_total > 0 THEN 'both' WHEN c_total > 0 THEN 'community' ELSE 'ours' END
                   AS side,
               c_total, o_total, diff, c_minutes, o_minutes,
               diff <= {s.rtol} * greatest(c_total, 1e-9) AND c_total > 0 AND o_total > 0 AS agree
        FROM m WHERE c_total > 0 OR o_total > 0""")
    whens = " ".join(f"WHEN ({pred}) THEN '{reason}'" for reason, pred in s.explain)
    case = f"CASE WHEN k.agree THEN NULL {whens} ELSE 'unexplained' END" if whens else \
        "CASE WHEN k.agree THEN NULL ELSE 'unexplained' END"
    con.execute(f"CREATE OR REPLACE TEMP TABLE kr AS SELECT k.*, {case} AS reason FROM kk k")
    if unexplained_dir is not None:   # the keys no reason explains (row-level, in the work directory)
        unexplained_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", s.name).strip("_")
        con.execute(f"""COPY (SELECT * FROM kr WHERE reason = 'unexplained' ORDER BY pid, key LIMIT 5000)
                        TO '{_p(unexplained_dir / f"{safe}_keys.parquet")}' (FORMAT parquet)""")
    q = lambda sql: con.execute(sql).fetchone()  # noqa: E731
    keys, both, agree = q("SELECT count(*), count(*) FILTER (WHERE side = 'both'), count(*) FILTER (WHERE agree) FROM kr")
    tot = q("""SELECT sum(c_total), sum(o_total), sum(diff), sum(c_minutes), sum(o_minutes) FROM kr WHERE side = 'both'""")
    reasons = dict(con.execute("SELECT side || ': ' || reason, count(*) FROM kr WHERE reason IS NOT NULL "
                               "GROUP BY 1 ORDER BY 2 DESC").fetchall())
    by_item = con.execute("""
        SELECT key, count(*) AS keys, count(*) FILTER (WHERE agree) AS agree,
               count(*) FILTER (WHERE side = 'community') AS only_community, count(*) FILTER (WHERE side = 'ours') AS only_ours,
               round(sum(c_total), 3) AS c_total, round(sum(o_total), 3) AS o_total, round(sum(diff), 3) AS diff,
               count(*) FILTER (WHERE reason = 'unexplained') AS unexplained
        FROM kr GROUP BY 1 ORDER BY unexplained DESC, keys DESC, key""").fetchall()
    cols = ("item", "keys", "agree", "only_community", "only_ours", "community_total", "our_total", "abs_difference",
            "unexplained")
    return {"community_records": q("SELECT count(*) FROM sc_all")[0],
            "community_out_of_scope": excluded.pop("out_of_scope", 0), "community_records_excluded": excluded,
            "community_records_in_streams": q("SELECT count(*) FROM sc")[0],
            "our_rows": q("SELECT count(*) FROM so")[0],
            "comparison": "total per key" if s.totals_only else "rate steps", "keys": keys, "keys_both": both,
            "keys_agree": agree,
            "share_of_shared_keys_agreeing": round(agree / both, 4) if both else None,
            "integral_community": tot[0], "integral_ours": tot[1], "integral_abs_difference": tot[2],
            "minutes_community": tot[3], "minutes_ours": tot[4], "rtol": s.rtol,
            "explained": {k: v for k, v in reasons.items() if not k.endswith(": unexplained")},
            "unexplained": sum(v for k, v in reasons.items() if k.endswith(": unexplained")),
            "keys_by_item": [dict(zip(cols, r)) for r in by_item]}


def _classify(side: str, reasons: tuple[tuple[str, str], ...], scoped: bool) -> str:
    whens = (["WHEN NOT EXISTS (SELECT 1 FROM scope s WHERE s.pid = u.pid AND u.t BETWEEN s.t0 AND s.t1) "
              "THEN 'out_of_scope'"] if scoped else [])
    whens += [f"WHEN ({pred}) THEN '{reason}'" for reason, pred in reasons]
    case = f"CASE {' '.join(whens)} END" if whens else "CAST(NULL AS VARCHAR)"
    rid = "c_rid" if side == "c" else "o_rid"
    other = "o_rid" if side == "c" else "c_rid"
    return f"""CREATE OR REPLACE TEMP TABLE u_{side} AS
               SELECT u.rid, u.item, {case} AS reason
               FROM {side}_all u JOIN pairs p ON p.{rid} = u.rid WHERE p.{other} IS NULL"""


def _write_unexplained(con: duckdb.DuckDBPyConnection, out: Path, key: str, limit: int = 5000) -> None:
    """Up to ``limit`` unexplained rows of each side, every bridge column (``<key>_community|ours.parquet`` in
    the work directory, next to the data: row-level, never part of the report)."""
    out.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", key).strip("_")
    for side, label in (("c", "community"), ("o", "ours")):
        con.execute(f"""COPY (SELECT a.* FROM u_{side} u JOIN {side}_all a ON a.rid = u.rid WHERE u.reason IS NULL
                              ORDER BY a.rid LIMIT {limit})
                        TO '{_p(out / f"{safe}_{label}.parquet")}' (FORMAT parquet)""")


def _pair(con: duckdb.DuckDBPyConnection, b: Bridge, unexplained_dir: Path | None = None) -> dict:
    for side, sql in (("c", b.community), ("o", b.ours)):
        _timed(con, f"""CREATE OR REPLACE TEMP TABLE {side}_all AS
                        SELECT row_number() OVER () AS rid, * REPLACE (CAST(pid AS VARCHAR) AS pid, CAST(t AS TIMESTAMP) AS t,
                               CAST(item AS VARCHAR) AS item, CAST(value AS DOUBLE) AS value)
                        FROM ({sql})""", f"{b.name or b.table} {side}_all")
    c_cols = {d[0] for d in con.execute("SELECT * FROM c_all LIMIT 0").description}
    o_cols = {d[0] for d in con.execute("SELECT * FROM o_all LIMIT 0").description}
    unit_key = "CAST(unit_key AS VARCHAR)" if "unit_key" in c_cols else "CAST(NULL AS VARCHAR)"
    keyed = "pair_key" in c_cols and "pair_key" in o_cols
    pk = "coalesce(CAST(pair_key AS VARCHAR), '')" if keyed else "''"
    # Two passes per (pid, t, item, pair key): rows holding the same value pair first (community value x the item's
    # unit factor, estimated on keys with one row per side; 5 significant digits absorb float32), the rest by value
    # order. Order alone would misalign a key where one side merged or dropped a duplicate.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE c_k AS
        WITH c AS (SELECT rid, pid, t, item, {pk} AS pk, value, {unit_key} AS unit_key FROM c_all),
             cs AS (SELECT pid, t, item, pk, any_value(value) AS cv, any_value(unit_key) AS unit_key
                    FROM c GROUP BY pid, t, item, pk HAVING count(*) = 1),
             os AS (SELECT pid, t, item, {pk} AS pk, any_value(value) AS ov FROM o_all GROUP BY pid, t, item, {pk}
                    HAVING count(*) = 1),
             f0 AS (SELECT item, unit_key, median(ov / cv) AS f FROM cs JOIN os USING (pid, t, item, pk)
                    WHERE cv <> 0 GROUP BY item, unit_key)
        SELECT c.*, {_SIG5.format(v=_F32.format(v='c.value * coalesce(f0.f, 1)'))} AS vk,
               {_FLOOR5.format(v=_F32.format(v='c.value * coalesce(f0.f, 1)'))} AS vf
        FROM c LEFT JOIN f0 ON f0.item = c.item AND f0.unit_key IS NOT DISTINCT FROM c.unit_key""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE o_k AS
                    SELECT rid, pid, t, item, {pk} AS pk, value, {_SIG5.format(v=_F32.format(v='value'))} AS vk,
                           {_FLOOR5.format(v=_F32.format(v='value'))} AS vf FROM o_all""")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE p1 AS
        WITH c AS (SELECT rid, pid, t, item, pk, vk, row_number() OVER (PARTITION BY pid, t, item, pk, vk ORDER BY rid) AS k
                   FROM c_k WHERE vk IS NOT NULL),
             o AS (SELECT rid, pid, t, item, pk, vk, row_number() OVER (PARTITION BY pid, t, item, pk, vk ORDER BY rid) AS k
                   FROM o_k WHERE vk IS NOT NULL)
        SELECT c.rid AS c_rid, o.rid AS o_rid FROM c JOIN o USING (pid, t, item, pk, vk, k)""")
    # the rows the rounded key left: by the truncated key (a pair straddling a rounding boundary)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE p1f AS
        WITH c AS (SELECT rid, pid, t, item, pk, vf, row_number() OVER (PARTITION BY pid, t, item, pk, vf ORDER BY rid) AS k
                   FROM c_k ANTI JOIN p1 ON p1.c_rid = c_k.rid WHERE vf IS NOT NULL),
             o AS (SELECT rid, pid, t, item, pk, vf, row_number() OVER (PARTITION BY pid, t, item, pk, vf ORDER BY rid) AS k
                   FROM o_k ANTI JOIN p1 ON p1.o_rid = o_k.rid WHERE vf IS NOT NULL)
        SELECT c.rid AS c_rid, o.rid AS o_rid FROM c JOIN o USING (pid, t, item, pk, vf, k)""")
    con.execute("INSERT INTO p1 SELECT c_rid, o_rid FROM p1f")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pairs AS
        WITH c AS (SELECT rid, pid, t, item, pk, value, unit_key,
                          row_number() OVER (PARTITION BY pid, t, item, pk ORDER BY value NULLS LAST, rid) AS k
                   FROM c_k ANTI JOIN p1 ON p1.c_rid = c_k.rid),
             o AS (SELECT rid, pid, t, item, pk, value,
                          row_number() OVER (PARTITION BY pid, t, item, pk ORDER BY value NULLS LAST, rid) AS k
                   FROM o_k ANTI JOIN p1 ON p1.o_rid = o_k.rid)
        SELECT coalesce(c.item, o.item) AS item, c.rid AS c_rid, o.rid AS o_rid, c.value AS cv, o.value AS ov, c.unit_key
        FROM c FULL JOIN o ON o.pid = c.pid AND o.t = c.t AND o.item = c.item AND o.pk = c.pk AND o.k = c.k
        UNION ALL
        SELECT c.item, c.rid, o.rid, c.value, o.value, c.unit_key
        FROM p1 JOIN c_k c ON c.rid = p1.c_rid JOIN o_k o ON o.rid = p1.o_rid""")
    _timed(con, _classify("c", b.explain_community, scoped=True), f"{b.name or b.table} classify community")
    _timed(con, _classify("o", b.explain_ours, scoped=False), f"{b.name or b.table} classify ours")
    if unexplained_dir is not None:
        _write_unexplained(con, unexplained_dir, b.name or b.table)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE items AS
        WITH f AS (SELECT item, unit_key,
                          median(ov / cv) FILTER (WHERE c_rid IS NOT NULL AND o_rid IS NOT NULL AND cv <> 0) AS factor
                   FROM pairs GROUP BY 1, 2),
             v AS (SELECT p.item, count(*) FILTER (WHERE c_rid IS NOT NULL) AS community,
                          count(*) FILTER (WHERE o_rid IS NOT NULL) AS ours,
                          count(*) FILTER (WHERE c_rid IS NOT NULL AND o_rid IS NOT NULL) AS paired,
                          list(DISTINCT round(f.factor, 6)) FILTER (WHERE f.factor IS NOT NULL) AS factor,
                          count(*) FILTER (WHERE cv IS NOT NULL AND ov IS NOT NULL
                                           AND abs(ov - coalesce(f.factor, 1) * cv) <= {VALUE_RTOL} * greatest(1, abs(ov)))
                              AS values_agree,
                          count(*) FILTER (WHERE cv IS NOT NULL AND ov IS NOT NULL) AS values_compared
                   FROM pairs p LEFT JOIN f ON f.item = p.item AND f.unit_key IS NOT DISTINCT FROM p.unit_key
                   GROUP BY p.item),
             uc AS (SELECT item, count(*) FILTER (WHERE reason = 'out_of_scope') AS community_out_of_scope,
                           count(*) FILTER (WHERE reason IS NOT NULL AND reason <> 'out_of_scope') AS only_community_explained,
                           count(*) FILTER (WHERE reason IS NULL) AS only_community_unexplained FROM u_c GROUP BY 1),
             uo AS (SELECT item, count(*) FILTER (WHERE reason IS NOT NULL) AS only_ours_explained,
                           count(*) FILTER (WHERE reason IS NULL) AS only_ours_unexplained FROM u_o GROUP BY 1)
        SELECT v.item, community, ours, paired, coalesce(community_out_of_scope, 0) AS community_out_of_scope,
               coalesce(only_community_explained, 0) AS only_community_explained,
               coalesce(only_community_unexplained, 0) AS only_community_unexplained,
               coalesce(only_ours_explained, 0) AS only_ours_explained,
               coalesce(only_ours_unexplained, 0) AS only_ours_unexplained, factor, values_agree, values_compared
        FROM v LEFT JOIN uc USING (item) LEFT JOIN uo USING (item)""")
    cols = [d[0] for d in con.execute("SELECT * FROM items LIMIT 0").description]
    items = [dict(zip(cols, r)) for r in con.execute(
        "SELECT * FROM items ORDER BY only_community_unexplained + only_ours_unexplained DESC, community DESC, item").fetchall()]
    tot = {k: sum(i[k] for i in items) for k in cols if k not in ("item", "factor")}
    in_scope = tot["community"] - tot["community_out_of_scope"]
    reasons = {side: dict(con.execute(f"SELECT reason, count(*) FROM u_{side} WHERE reason IS NOT NULL "
                                      "GROUP BY 1 ORDER BY 2 DESC, 1").fetchall()) for side in ("c", "o")}
    reasons["c"].pop("out_of_scope", None)
    return {"community": tot["community"], "ours": tot["ours"], "paired": tot["paired"],
            "community_out_of_scope": tot["community_out_of_scope"],
            "only_community_explained": reasons["c"], "only_community_unexplained": tot["only_community_unexplained"],
            "only_ours_explained": reasons["o"], "only_ours_unexplained": tot["only_ours_unexplained"],
            "share_of_community_paired": round(tot["paired"] / in_scope, 4) if in_scope else None,
            "share_of_community_accounted": (round((in_scope - tot["only_community_unexplained"]) / in_scope, 4)
                                             if in_scope else None),
            "share_of_ours_accounted": (round((tot["ours"] - tot["only_ours_unexplained"]) / tot["ours"], 4)
                                        if tot["ours"] else None),
            "values_agree": round(tot["values_agree"] / tot["values_compared"], 4) if tot["values_compared"] else None,
            "items": [{**i, "factor": sorted(float(x) for x in i["factor"] or [])} for i in items]}


def _g(x: float | None) -> str:
    return "-" if x is None else f"{x:.6g}"


def write(reports: list[dict], out_dir: Path, top: int = 15) -> None:
    """Write ``reconciliation.{json,md}``; datasets reconciled earlier and not in ``reports`` keep their entry."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{REPORT}.json"
    earlier = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    new = {r["dataset"]: r for r in reports}
    reports = [new.pop(r["dataset"], r) for r in earlier] + list(new.values())
    path.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    lines = ["# Reconciliation with the community MEDS ETLs", "",
             "Paired: same patient, time and source item. *Accounted*: paired, or unpaired for a documented reason. "
             "Values agree: ours equals community x the item's unit factor (relative tolerance "
             f"{VALUE_RTOL:g}).", ""]
    for r in reports:
        lines += [f"## {r['dataset']}", "", f"Community events: {r['community_events']}", "",
                  "| Table | Community | Ours | Paired | Out of scope | Community accounted | Ours accounted | "
                  "Unexplained (community / ours) | Values agree |", "|---|---|---|---|---|---|---|---|---|"]
        for t, m in r["tables"].items():
            if "skipped" in m:
                lines.append(f"| {t} | skipped: {m['skipped']} | | | | | | | |")
                continue
            lines.append(f"| {t} | {m['community']} | {m['ours']} | {m['paired']} | {m['community_out_of_scope']} | "
                         f"{m['share_of_community_accounted']} | {m['share_of_ours_accounted']} | "
                         f"{m['only_community_unexplained']} / {m['only_ours_unexplained']} | {m['values_agree']} |")
        for t, m in r["tables"].items():
            if "skipped" in m:
                continue
            for side, label in (("only_community_explained", "community rows explained"),
                                ("only_ours_explained", "our rows explained")):
                if m[side]:
                    lines += ["", f"{t}, {label}: " + ", ".join(f"{k} {v}" for k, v in m[side].items())]
            worst = [i for i in m["items"] if i["only_community_unexplained"] or i["only_ours_unexplained"]][:top]
            if worst:
                lines += ["", f"{t}: items with unexplained rows (community / ours / unexplained community / "
                              "unexplained ours)", ""]
                lines += [f"- {i['item']}: {i['community']} / {i['ours']} / {i['only_community_unexplained']} / "
                          f"{i['only_ours_unexplained']}" for i in worst]
            off = [i for i in m["items"] if i["values_compared"] and i["values_agree"] < i["values_compared"]][:top]
            if off:
                lines += ["", f"{t}: items whose values disagree (factor, agree / compared)", ""]
                lines += [f"- {i['item']}: {i['factor']}, {i['values_agree']} / {i['values_compared']}" for i in off]
        if r.get("streams"):
            lines += ["", "Rate streams per (patient, drug and unit), inside the hospitalization windows: a key agrees "
                          "when the integral of the absolute rate difference ('rate steps') or the difference of "
                          "the totals ('total per key') is within rtol of the community's integral.", "",
                      "| Stream | Comparison | Community records (out of scope) | Our rows | Keys | Both sides | Agree | "
                      "Integral community / ours / abs. difference | Unexplained keys |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for sname, m in r["streams"].items():
                if "skipped" in m:
                    lines.append(f"| {sname} | skipped: {m['skipped']} | | | | | | | |")
                    continue
                lines.append(f"| {sname} | {m['comparison']} | {m['community_records']} ({m['community_out_of_scope']}) | "
                             f"{m['our_rows']} | {m['keys']} | {m['keys_both']} | {m['keys_agree']} | "
                             f"{_g(m['integral_community'])} / {_g(m['integral_ours'])} / "
                             f"{_g(m['integral_abs_difference'])} | {m['unexplained']} |")
            for sname, m in r["streams"].items():
                if m.get("community_records_excluded"):
                    lines += ["", f"{sname}, community records not in a stream (our documented rules): "
                              + ", ".join(f"{k} {v}" for k, v in m["community_records_excluded"].items())]
                if m.get("explained"):
                    lines += ["", f"{sname}, keys explained: " + ", ".join(f"{k} {v}" for k, v in m["explained"].items())]
                worst = [i for i in m.get("keys_by_item", []) if i["unexplained"]][:top]
                if worst:
                    lines += ["", f"{sname}: drugs with unexplained keys (keys / agree / unexplained)", ""]
                    lines += [f"- {i['item']}: {i['keys']} / {i['agree']} / {i['unexplained']}" for i in worst]
        lines.append("")
    (out_dir / f"{REPORT}.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
