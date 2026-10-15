"""Weight causality of the training MEDS (S4 gate; M5 preflight).

A medication value in a weight-normalized unit (``.../kg...``) either is the source's own (the dose was charted
per kg at the bedside) or was converted from an unweighted dose with a charted weight. The conversion must be
causal: no weight charted after the dose (``meds.elf``: the weight in effect at the dose, and for an infusion at
the start of its segment). A first-weight fallback, the usual non-causal shortcut, would show here as a converted
per-kg value before the hospitalization's first charted weight. The gate counts, per dataset:

* per-kg medication events before the first weight (CLIF ``vitals``, ``weight_kg``), split into reported (a
  source row of the drug in the same table at the event's time is per kg; for a window-start ``going`` event,
  any of the drug's infusion rows before it: the running stream may be an older one, e.g. NWICU charting a
  drug per kg, then the same infusion in mcg/min without stopping the first stream) and converted;
* per-kg infusion events whose segment (the stream's last ``start`` or ``going`` at or before them) began before
  the first weight, split the same way by the segment's first event.

Converted events in either group are violations: a weight charted later was used.
"""

from __future__ import annotations

from pathlib import Path

from icutwin.clif.conform import connect

PER_KG = "lower(replace({col}, ' ', '')) LIKE '%/kg%'"


def weight_causality(clif_dir: Path, meds_dir: Path) -> dict:
    con = connect()
    p = lambda f: f.as_posix().replace("'", "''")  # noqa: E731
    sources = [f"SELECT '{domain}' AS domain, TRY_CAST(hospitalization_id AS BIGINT) AS h, "
               f"CAST(admin_dttm AS TIMESTAMP) AS t, med_category AS cat, {PER_KG.format(col='med_dose_unit')} AS per_kg "
               f"FROM read_parquet('{p(f)}')"
               for domain, f in (("MED_CON", clif_dir / "clif_medication_admin_continuous.parquet"),
                                 ("MED_INT", clif_dir / "clif_medication_admin_intermittent.parquet")) if f.exists()]
    if not sources:
        return {"weighted_events": 0, "violations": 0}
    con.execute(f"CREATE TEMP TABLE src AS {' UNION ALL '.join(sources)}")
    con.execute(f"""CREATE TEMP TABLE w AS
        SELECT TRY_CAST(hospitalization_id AS BIGINT) AS h, min(CAST(recorded_dttm AS TIMESTAMP)) AS t0
        FROM read_parquet('{p(clif_dir / "clif_vitals.parquet")}')
        WHERE vital_category = 'weight_kg' AND vital_value IS NOT NULL GROUP BY 1""")
    con.execute(f"""CREATE TEMP TABLE ev AS
        SELECT subject_id, hospitalization_id AS h, time, split_part(code, '//', 1) AS domain,
               split_part(code, '//', 2) AS cat, split_part(code, '//', 4) AS action,
               concat_ws('//', split_part(code, '//', 2), split_part(code, '//', 3), split_part(code, '//', 5)) AS stream
        FROM read_parquet('{p(meds_dir / "data")}/*/*.parquet')
        WHERE split_part(code, '//', 1) IN ('MED_CON', 'MED_INT') AND {PER_KG.format(col="split_part(code, '//', 3)")}""")
    con.execute("""CREATE TEMP TABLE ev2 AS
        WITH r AS (
            SELECT ev.*, w.t0,
                   CASE WHEN action = 'going'
                        THEN (SELECT bool_or(s.per_kg) FROM src s
                              WHERE s.domain = ev.domain AND s.h = ev.h AND s.cat = ev.cat AND s.t < ev.time)
                        ELSE (SELECT bool_or(s.per_kg) FROM src s
                              WHERE s.domain = ev.domain AND s.h = ev.h AND s.cat = ev.cat AND s.t = ev.time)
                   END AS reported0
            FROM ev LEFT JOIN w USING (h)),
        r2 AS (SELECT * EXCLUDE (reported0), coalesce(reported0, false) AS reported FROM r)
        SELECT *,
               max(CASE WHEN domain = 'MED_CON' AND action IN ('start', 'going') THEN time END) OVER seg AS seg_start,
               arg_max(CASE WHEN domain = 'MED_CON' AND action IN ('start', 'going') THEN reported END,
                       CASE WHEN domain = 'MED_CON' AND action IN ('start', 'going') THEN time END) OVER seg AS seg_reported
        FROM r2
        WINDOW seg AS (PARTITION BY subject_id, stream ORDER BY time, action NOT IN ('start', 'going')
                       ROWS UNBOUNDED PRECEDING)""")
    row = con.execute("""
        SELECT count(*),
               count(*) FILTER (WHERE (t0 IS NULL OR time < t0) AND coalesce(reported, false)),
               count(*) FILTER (WHERE (t0 IS NULL OR time < t0) AND NOT coalesce(reported, false)),
               count(*) FILTER (WHERE domain = 'MED_CON' AND (t0 IS NULL OR seg_start IS NULL OR seg_start < t0)
                                AND coalesce(seg_reported, false)),
               count(*) FILTER (WHERE domain = 'MED_CON' AND (t0 IS NULL OR seg_start IS NULL OR seg_start < t0)
                                AND NOT coalesce(seg_reported, false))
        FROM ev2""").fetchone()
    keys = ["weighted_events", "before_first_weight_reported", "before_first_weight_converted",
            "segment_before_first_weight_reported", "segment_before_first_weight_converted"]
    out = dict(zip(keys, (int(x) for x in row)))
    out["violations"] = out["before_first_weight_converted"] + out["segment_before_first_weight_converted"]
    return out
