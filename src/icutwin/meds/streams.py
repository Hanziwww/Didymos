"""One infusion stream per MEDS code (S4).

Canonical CLIF has one rate stream per (hospitalization, drug, dose unit) (``harmonize.canonical_continuous``).
CLIF_MEDS's unit conversion codes ``MED_CON//<drug>//<preferred unit>//<action>``, so the streams of a drug charted
in two units (MIMIC insulin in u/hr and, from amount over duration, u/min) land on one code, and a stream whose
weight-based conversion has no weight yet, or whose route changes, is split across codes. The code's events then
stop forming one stream: a dose change after the code's stop, two starts. A hospitalization's drug with more than
one canonical stream or code is therefore summed again per code, as ``canonical_continuous`` sums concurrent
orders:
* each canonical stream (identified by its source unit, carried in ``text_value`` by the build's MED_CON config)
  contributes its last rate, converted, to the code of its last event, until its stop;
* an event is written where a code's total changes: ``start`` (from not running), ``stop`` (to not running, value
  0), ``dose_change`` (a new total; NULL while a contributing rate is unknown: a conversion without a weight);
  a presence confirmation (``going``, ``verify``, ``other``) of a running code is kept with the total.
Every other drug of a hospitalization (one stream, one code) passes unchanged. ``text_value`` is blanked.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

STREAM_COLUMN = "med_dose_unit"   # the canonical stream's dose unit, carried as text_value
_TOL = 1e-9


def _p(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= _TOL * max(1.0, abs(a), abs(b))


def _total(subs: dict, key: tuple) -> tuple[bool, float | None]:
    """(running, total rate; None when a running contribution is unknown) of a code."""
    rates = [r for k, run, r in subs.values() if run and k == key]
    if not rates:
        return False, None
    return True, None if any(r is None for r in rates) else round(sum(rates), 9)


def restream(rows: list[tuple]) -> list[tuple]:
    """One hospitalization's drug: ``rows`` (time, (unit, action, route...), value, stream, row) -> the summed
    events (time, (unit, route...), action, value, row of the code at that time)."""
    subs: dict = {}                    # stream -> (code key, running, rate)
    out = []
    by_time: dict = defaultdict(list)
    for r in rows:
        by_time[r[0]].append(r)
    for t in sorted(by_time):
        batch = by_time[t]
        key_of = lambda parts: (parts[0], *parts[2:])  # noqa: E731
        keys = {k for k, _run, _r in subs.values()} | {key_of(r[1]) for r in batch}
        before = {k: _total(subs, k) for k in keys}
        confirmed = {}
        for _t, parts, value, stream, extra in batch:
            key, action = key_of(parts), parts[1]
            if action == "stop":
                subs[stream] = (key, False, 0.0)
            elif action in ("start", "dose_change") or value is not None:
                subs[stream] = (key, True, None if value is None else float(value))
            else:                       # a presence confirmation: running, rate unchanged
                prev = subs.get(stream)
                subs[stream] = (key, True, prev[2] if prev and prev[1] and prev[0] == key else None)
                confirmed.setdefault(key, (action, extra))
        for key in sorted(keys, key=str):
            (was, a), (now, b) = before[key], _total(subs, key)
            extra = next((r[4] for r in batch if key_of(r[1]) == key), batch[0][4])
            if not was and now:
                out.append((t, key, "start", b, extra))
            elif was and not now:
                out.append((t, key, "stop", 0.0, extra))
            elif now and not _same(a, b):
                out.append((t, key, "dose_change", b, extra))
            elif now and key in confirmed:
                out.append((t, key, confirmed[key][0], b, confirmed[key][1]))
    return out


def restream_infusions(con, path: Path) -> dict:
    """Rewrite a MED_CON event file (``MED_CON//drug//unit//action//route``, the canonical stream in
    ``text_value``) with one stream per code; returns counts."""
    src = f"read_parquet('{_p(path)}')"
    # a hospitalization's drug with several canonical streams (source units) or several codes (unit, route)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _merged AS
                    SELECT subject_id, hospitalization_id, split_part(code, '//', 2) AS drug FROM {src} GROUP BY ALL
                    HAVING count(DISTINCT coalesce(text_value, '')) > 1
                        OR count(DISTINCT array_to_string(list_concat([split_part(code, '//', 3)],
                                                                      string_split(code, '//')[5:]), '//')) > 1""")
    on = """ON e.subject_id = m.subject_id AND e.hospitalization_id IS NOT DISTINCT FROM m.hospitalization_id
            AND split_part(e.code, '//', 2) = m.drug"""
    schema = pq.read_schema(path)
    rows = con.execute(f"""SELECT e.* FROM {src} e SEMI JOIN _merged m {on}
                           ORDER BY e.subject_id, e.hospitalization_id, e.time, e.code""").arrow().read_all().to_pylist()
    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["subject_id"], r["hospitalization_id"], r["code"].split("//")[1])].append(r)
    rebuilt = []
    for (_sid, _hid, drug), part in groups.items():
        events = restream([(r["time"], tuple(r["code"].split("//")[2:]), r["numeric_value"], r["text_value"], r)
                           for r in part])
        for t, key, action, value, extra in events:
            rebuilt.append({**extra, "time": t, "code": "//".join(["MED_CON", drug, key[0], action, *key[1:]]),
                            "numeric_value": value, "text_value": None})
    kept = con.execute(f"""SELECT e.* REPLACE (CAST(NULL AS VARCHAR) AS text_value) FROM {src} e
                           ANTI JOIN _merged m {on}""").arrow().read_all()
    out = pa.concat_tables([kept.cast(schema), pa.Table.from_pylist(rebuilt, schema=schema)])
    tmp = path.with_suffix(".restreamed.parquet")
    pq.write_table(out, tmp, compression="zstd")
    tmp.replace(path)
    con.execute("DROP TABLE _merged")
    return {"drugs_summed": len(groups), "events_in": len(rows), "events_out": len(rebuilt)}
