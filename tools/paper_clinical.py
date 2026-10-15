"""Paper P amendment 10 (``docs/p1_paper_protocol.md``): the clinical scores and bedside indicators of A1's held-out
anchors (``eval.clinical``), from the canonical CLIF tables: the harmonized raw values, not the tokens. Only data
recorded or resulted by each anchor's prediction time count, from the 24 hours before it. The window ends at the
stay's ICU admission plus ``t_min`` + 1 minute (exclusive; the store's minutes are floors).

Per anchor:
* vitals (mimic-code ``vitalsign``'s plausibility ranges), labs (at their result time), the GCS (the total, else eye +
  verbal + motor charted together), respiratory support (device, FiO2), urine output: the latest value, minimum and
  maximum in the window; MAP from systolic and diastolic pressure (charted together) where no MAP is charted;
* SOFA over the window (clifpy ``compute_sofa_polars``, with ``meds.derived``'s respiration correction);
* the lowest PaO2/FiO2 under ventilation or CPAP (the FiO2 and device charted within 4 h before the PaO2);
* invasive ventilation in the window (``derived/<dataset>/episodes.parquet``);
* the KDIGO stage: the creatinine rules on the latest creatinine of the last 48 h against its 48-hour and 7-day minima
  (result times), the urine-output stage of ``derived/<dataset>/kdigo.parquet`` (its latest in the window), CRRT;
* the datasets' own scores (``clif_ext_icu_stay`` severity: eICU APACHE IVa points, SICdb SAPS 3, INSPIRE ASA) and
  eICU's APACHE IVa predictions (``apachePatientResult``, version IVa; -1 is missing).

Scores: an unobserved component scores 0 (``eval.clinical``). Bedside indicators: a missing value is set to normal
(``NORMAL``); the ROX index's FiO2 is the charted FiO2, else the device's (``fio2_estimate``).

Writes ``<out>/variables.parquet`` and ``<out>/scores.parquet`` (per design and anchor; host only) and
``<out>/clinical.json`` (coverage per design and dataset, score medians).

    python tools/paper_clinical.py --designs runs/paper/a1/test_L24 runs/paper/a1/test_L48 runs/paper/a1/test_R \\
        --clif data/clif_canonical/full --derived data/derived/full --raw data/raw/full --out runs/paper/f10/clinical
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

VITALS = {"heart_rate": (0, 300), "sbp": (0, 400), "dbp": (0, 300), "map": (0, 300), "respiratory_rate": (0, 70),
          "spo2": (0, 100.001), "temp_c": (10, 50)}
LABS = {"platelet_count": (0, 10000), "bilirubin_total": (0, 150), "creatinine": (0, 150), "bun": (0, 300),
        "wbc": (0, 1000), "potassium": (0, 30), "sodium": (0, 250), "bicarbonate": (0, 100), "lactate": (0, 50),
        "pco2_arterial": (0, 250), "po2_arterial": (0, 800)}
NORMAL = {"heart_rate": 80.0, "sbp": 120.0, "map": 80.0, "respiratory_rate": 16.0, "spo2": 97.0, "lactate": 1.0,
          "creatinine": 0.9, "potassium": 4.0, "platelet_count": 200.0}
DEVICE_FIO2 = {"room_air": 0.21, "face_mask": 0.40, "trach_collar": 0.35}
COVERAGE = ("heart_rate_last", "sbp_last", "map_last", "respiratory_rate_last", "spo2_last", "temp_c_last",
            "gcs_last", "platelet_count_last", "bilirubin_total_last", "creatinine_last", "lactate_last", "wbc_last",
            "pco2_arterial_last", "urine_ml", "pf_vent_min", "dev_last", "sofa_total")


def _p(path: Path) -> str:
    return str(path).replace("\\", "/")


def _in_range(col: str, val: str, ranges: dict) -> str:
    return " OR ".join(f"({col} = '{c}' AND {val} > {lo} AND {val} < {hi})" for c, (lo, hi) in ranges.items())


def fio2_estimate(fio2, dev, lpm) -> np.ndarray:
    """The charted FiO2, else the device's usual one: nasal cannula 0.21 + 0.04 per L/min (2 L/min when not
    charted, at most 0.45), ``DEVICE_FIO2``, 0.21 without a device."""
    out = np.asarray(fio2, dtype=np.float64).copy()
    for i in np.flatnonzero(~np.isfinite(out)):
        d = dev[i] if isinstance(dev[i], str) else None
        if d == "nasal_cannula":
            flow = lpm[i] if np.isfinite(lpm[i]) else 2.0
            out[i] = min(0.21 + 0.04 * flow, 0.45)
        else:
            out[i] = DEVICE_FIO2.get(d, 0.21 if d is None else 0.40)
    return out


def extract(g, clif: Path, derived: Path, raw: Path, work: Path):
    """The variables of one dataset's anchors (``g``: key, stay_id, t_min), one row per key."""
    import pandas as pd

    from icutwin.clif.conform import connect

    con = connect()
    con.register("anc_in", g[["key", "stay_id", "t_min"]])
    con.execute(f"""CREATE TEMP TABLE anc AS
        SELECT a.key, a.stay_id, e.hospitalization_id, e.icu_in_dttm AS icu_in, e.surgical_status, e.severity_name,
               e.severity_value,
               e.icu_in_dttm + to_microseconds(CAST((a.t_min + 1) * 60000000 AS BIGINT)) AS t_end,
               e.icu_in_dttm + to_microseconds(CAST((a.t_min + 1) * 60000000 AS BIGINT)) - INTERVAL 24 HOUR AS t_start
        FROM anc_in a JOIN read_parquet('{_p(clif / 'clif_ext_icu_stay.parquet')}') e USING (stay_id)""")
    frames = [con.execute("SELECT key, stay_id, hospitalization_id, surgical_status, severity_name, severity_value "
                          "FROM anc").df().set_index("key")]

    def window_join(alias: str) -> str:
        return (f"ON {alias}.hospitalization_id = a.hospitalization_id AND {alias}.t >= a.t_start "
                f"AND {alias}.t < a.t_end")

    def wide(table: str) -> pd.DataFrame:
        d = con.execute(f"SELECT key, cat, arg_max(val, t) AS last, min(val) AS mn, max(val) AS mx "
                        f"FROM {table} GROUP BY 1, 2").df()
        if not len(d):
            return pd.DataFrame(index=pd.Index([], name="key"))
        w = d.pivot(index="key", columns="cat", values=["last", "mn", "mx"])
        w.columns = [f"{c}_{dict(last='last', mn='min', mx='max')[s]}" for s, c in w.columns]
        return w

    # vitals, and MAP from systolic and diastolic pressure charted together
    con.execute(f"""CREATE TEMP TABLE vit AS
        SELECT a.key, v.cat, v.val, v.t FROM anc a JOIN (
            SELECT hospitalization_id, recorded_dttm AS t, vital_category AS cat, vital_value AS val
            FROM read_parquet('{_p(clif / 'clif_vitals.parquet')}')
            WHERE {_in_range('vital_category', 'vital_value', VITALS)}) v {window_join('v')}""")
    con.execute("""INSERT INTO vit SELECT s.key, 'map_calc', (s.val + 2 * d.val) / 3, s.t
                   FROM vit s JOIN vit d ON d.key = s.key AND d.t = s.t AND d.cat = 'dbp' WHERE s.cat = 'sbp'""")
    frames.append(wide("vit"))
    # labs at their result time
    labs = _p(clif / "clif_labs.parquet")
    con.execute(f"""CREATE TEMP TABLE lab AS
        SELECT a.key, l.cat, l.val, l.t FROM anc a JOIN (
            SELECT hospitalization_id, coalesce(lab_result_dttm, lab_collect_dttm) AS t, lab_category AS cat,
                   lab_value_numeric AS val
            FROM read_parquet('{labs}') WHERE {_in_range('lab_category', 'lab_value_numeric', LABS)}) l
            {window_join('l')}""")
    frames.append(wide("lab"))
    # GCS: the total, else eye + verbal + motor charted together
    if (clif / "clif_patient_assessments.parquet").exists():
        con.execute(f"""CREATE TEMP TABLE gcs_raw AS
            SELECT a.key, p.t, p.cat, p.val FROM anc a JOIN (
                SELECT hospitalization_id, recorded_dttm AS t, assessment_category AS cat, numerical_value AS val
                FROM read_parquet('{_p(clif / 'clif_patient_assessments.parquet')}')
                WHERE assessment_category IN ('gcs_total', 'gcs_eye', 'gcs_verbal', 'gcs_motor')
                  AND numerical_value IS NOT NULL) p {window_join('p')}""")
        con.execute("""CREATE TEMP TABLE gcs AS SELECT key, 'gcs' AS cat, t, val FROM (
            SELECT key, t, coalesce(max(CASE WHEN cat = 'gcs_total' THEN val END),
                CASE WHEN count(DISTINCT CASE WHEN cat <> 'gcs_total' THEN cat END) = 3
                     THEN max(CASE WHEN cat = 'gcs_eye' THEN val END) + max(CASE WHEN cat = 'gcs_verbal' THEN val END)
                          + max(CASE WHEN cat = 'gcs_motor' THEN val END) END) AS val
            FROM gcs_raw GROUP BY key, t) WHERE val BETWEEN 3 AND 15""")
        frames.append(wide("gcs"))
    # respiratory support: the latest device, FiO2 and flow; P/F under ventilation or CPAP
    con.execute(f"""CREATE TEMP TABLE rs AS
        SELECT a.key, r.t, r.dev, r.fio2, r.lpm FROM anc a JOIN (
            SELECT hospitalization_id, recorded_dttm AS t, device_category AS dev,
                   CASE WHEN fio2_set > 1 AND fio2_set <= 100 THEN fio2_set / 100 ELSE fio2_set END AS fio2,
                   lpm_set AS lpm
            FROM read_parquet('{_p(clif / 'clif_respiratory_support.parquet')}')) r {window_join('r')}""")
    frames.append(con.execute("""
        SELECT key, arg_max(dev, t) FILTER (WHERE dev IS NOT NULL) AS dev_last,
               arg_max(lpm, t) FILTER (WHERE dev IS NOT NULL) AS lpm_last,
               arg_max(fio2, t) FILTER (WHERE fio2 BETWEEN 0.21 AND 1.0) AS fio2_last
        FROM rs GROUP BY key""").df().set_index("key"))
    frames.append(con.execute("""
        WITH po2 AS (SELECT key, t, val FROM lab WHERE cat = 'po2_arterial'),
             f AS (SELECT key, t, fio2 FROM rs WHERE fio2 BETWEEN 0.21 AND 1.0),
             d AS (SELECT key, t, dev FROM rs WHERE dev IS NOT NULL),
             pf AS (SELECT po2.key, po2.t, po2.val / f.fio2 AS pf, f.t AS tf
                    FROM po2 ASOF JOIN f ON po2.key = f.key AND po2.t >= f.t),
             pfd AS (SELECT pf.*, d.dev, d.t AS td FROM pf ASOF JOIN d ON pf.key = d.key AND pf.t >= d.t)
        SELECT key, min(pf) AS pf_vent_min FROM pfd
        WHERE tf >= t - INTERVAL 4 HOUR AND td >= t - INTERVAL 4 HOUR AND dev IN ('imv', 'nippv', 'cpap')
        GROUP BY key""").df().set_index("key"))
    # urine output over the window
    if (clif / "clif_output.parquet").exists():
        frames.append(con.execute(f"""
            SELECT a.key, sum(o.val) AS urine_ml FROM anc a JOIN (
                SELECT hospitalization_id, recorded_dttm AS t, output_volume AS val
                FROM read_parquet('{_p(clif / 'clif_output.parquet')}')
                WHERE output_group = 'urine' AND output_volume >= 0) o {window_join('o')}
            GROUP BY 1""").df().set_index("key"))
    # invasive ventilation in the window (the derived episodes)
    ep = derived / "episodes.parquet"
    if ep.exists():
        frames.append(con.execute(f"""
            SELECT a.key, 1 AS ventilated FROM anc a JOIN read_parquet('{_p(ep)}') e ON e.stay_id = a.stay_id
            WHERE e.episode = 'imv' AND e.start_dttm < CAST(a.t_end AS TIMESTAMP)
              AND e.end_dttm > CAST(a.t_start AS TIMESTAMP)
            GROUP BY 1""").df().set_index("key"))
    # KDIGO: creatinine (result time) against its 48-hour and 7-day minima; the urine-output stage; CRRT
    con.execute(f"""CREATE TEMP TABLE cr AS
        SELECT a.key, a.t_end, l.t, l.val FROM anc a JOIN (
            SELECT hospitalization_id, coalesce(lab_result_dttm, lab_collect_dttm) AS t, lab_value_numeric AS val
            FROM read_parquet('{labs}') WHERE lab_category = 'creatinine' AND lab_value_numeric > 0
              AND lab_value_numeric <= 150) l
        ON l.hospitalization_id = a.hospitalization_id AND l.t < a.t_end AND l.t >= a.t_end - INTERVAL 9 DAY""")
    frames.append(con.execute("""
        WITH cur AS (SELECT key, arg_max(val, t) AS creat, max(t) AS tc FROM cr
                     WHERE t >= t_end - INTERVAL 48 HOUR GROUP BY key)
        SELECT cur.key, cur.creat AS creat_48h,
               (SELECT min(p.val) FROM cr p WHERE p.key = cur.key AND p.t < cur.tc
                  AND p.t >= cur.tc - INTERVAL 48 HOUR) AS creat_low48,
               (SELECT min(p.val) FROM cr p WHERE p.key = cur.key AND p.t < cur.tc
                  AND p.t >= cur.tc - INTERVAL 7 DAY) AS creat_low7
        FROM cur""").df().set_index("key"))
    kd = derived / "kdigo.parquet"
    if kd.exists():
        frames.append(con.execute(f"""
            SELECT a.key, arg_max(k.aki_stage_uo, k.t) FILTER (WHERE k.aki_stage_uo IS NOT NULL) AS kdigo_uo,
                   max(k.aki_stage_crrt) AS kdigo_crrt
            FROM anc a JOIN read_parquet('{_p(kd)}') k ON k.stay_id = a.stay_id
              AND k.t < CAST(a.t_end AS TIMESTAMP) AND k.t >= CAST(a.t_start AS TIMESTAMP)
            GROUP BY 1""").df().set_index("key"))
    # age and the hospital time before the ICU
    frames.append(con.execute(f"""
        SELECT a.key, h.age_at_admission AS age,
               CASE WHEN h.admission_dttm <= a.icu_in THEN date_diff('minute', h.admission_dttm, a.icu_in) END
                   AS pre_icu_minutes
        FROM anc a JOIN read_parquet('{_p(clif / 'clif_hospitalization.parquet')}') h USING (hospitalization_id)
        """).df().set_index("key"))
    # eICU's APACHE IVa predictions
    apr = raw / "apachePatientResult.csv.gz"
    if apr.exists():
        frames.append(con.execute(f"""
            SELECT a.key, CASE WHEN r.predictedicumortality >= 0 THEN r.predictedicumortality END AS apache_icu_mort,
                   CASE WHEN r.predictedhospitalmortality >= 0 THEN r.predictedhospitalmortality END
                       AS apache_hosp_mort,
                   CASE WHEN r.predictediculos >= 0 THEN r.predictediculos END AS apache_icu_los
            FROM anc a JOIN (SELECT * FROM read_csv_auto('{_p(apr)}') WHERE apacheversion = 'IVa') r
              ON a.stay_id = 'eicu:' || CAST(r.patientunitstayid AS VARCHAR)""").df().set_index("key"))
    # SOFA over each window (clifpy, the respiration correction of meds.derived)
    from clifpy.utils.sofa_polars import compute_sofa_polars

    from icutwin.meds import derived as dv

    cohort = con.execute("SELECT hospitalization_id, key AS stay_id, t_start AS start_dttm, t_end AS end_dttm "
                         "FROM anc").pl()
    view = dv._clifpy_view(clif, work / f"clifpy_{clif.name}", dv.SOFA_TABLES)
    try:
        with dv._captured_pf_pairs() as pairs, dv._spo2_optional():
            sofa = compute_sofa_polars(str(view), cohort, id_name="stay_id", timezone="UTC")
        sofa = dv.fix_sofa_resp(sofa, dv.worst_resp_scores(pairs[0]) if pairs else None)
    finally:
        shutil.rmtree(view, ignore_errors=True)
    frames.append(sofa.select(["stay_id", *sorted(c for c in sofa.columns if c.startswith("sofa_"))])
                  .to_pandas().rename(columns={"stay_id": "key"}).set_index("key"))
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="left")
    return out


def fill_map(v):
    """MAP: charted, else from systolic and diastolic pressure charted together (in place)."""
    for s in ("last", "min", "max"):
        calc = v[f"map_calc_{s}"] if f"map_calc_{s}" in v else None
        if f"map_{s}" in v and calc is not None:
            v[f"map_{s}"] = v[f"map_{s}"].fillna(calc)
        elif calc is not None:
            v[f"map_{s}"] = calc
    return v


def scores(v):
    """The scores and bedside indicators (``eval.clinical``) of the variables (``extract``, ``fill_map``)."""
    import pandas as pd

    from icutwin.eval import clinical as cl

    def col(c):
        return v[c].to_numpy(dtype=np.float64) if c in v else np.full(len(v), np.nan)

    dev = v["dev_last"].to_numpy(dtype=object) if "dev_last" in v else np.full(len(v), None, dtype=object)
    fio2 = col("fio2_last")
    supp = np.array([(isinstance(d, str) and d != "room_air") or (np.isfinite(f) and f > 0.21)
                     for d, f in zip(dev, fio2, strict=True)], dtype=np.float64)
    hr, sbp, rr, spo2, temp, gcs = (col("heart_rate_last"), col("sbp_last"), col("respiratory_rate_last"),
                                    col("spo2_last"), col("temp_c_last"), col("gcs_last"))
    surg = v["surgical_status"].astype(str).to_numpy() if "surgical_status" in v else np.full(len(v), "unknown")
    sev_name = v["severity_name"].astype(str).to_numpy() if "severity_name" in v else np.full(len(v), "")
    sev = col("severity_value")
    vent = np.nan_to_num(col("ventilated"), nan=0.0)
    out = pd.DataFrame(index=v.index)
    out["sofa"] = col("sofa_total")
    out["news"] = cl.news(rr, spo2, supp, temp, sbp, hr, gcs)
    out["mews"] = cl.mews(sbp, hr, rr, temp, gcs)
    out["qsofa"] = cl.qsofa(rr, sbp, gcs)
    out["sirs"] = cl.sirs(temp, hr, rr, col("pco2_arterial_last"), col("wbc_last"))
    out["sapsii"] = cl.sapsii(col("age"), col("heart_rate_min"), col("heart_rate_max"), col("sbp_min"), col("sbp_max"),
                              col("temp_c_max"), col("pf_vent_min"), col("urine_ml"), col("bun_max"), col("wbc_min"),
                              col("wbc_max"), col("potassium_min"), col("potassium_max"), col("sodium_min"),
                              col("sodium_max"), col("bicarbonate_min"), col("bilirubin_total_max"), col("gcs_min"),
                              surg)
    out["sapsii_prob"] = cl.sapsii_prob(out["sapsii"])
    out["oasis"] = cl.oasis(col("pre_icu_minutes"), col("age"), col("gcs_min"), col("heart_rate_min"),
                            col("heart_rate_max"), col("map_min"), col("map_max"), col("respiratory_rate_min"),
                            col("respiratory_rate_max"), col("temp_c_min"), col("temp_c_max"), col("urine_ml"), vent, surg)
    out["oasis_prob"] = cl.oasis_prob(out["oasis"])
    out["apache_iva"] = np.where(sev_name == "apache_iva", sev, np.nan)
    out["apache_icu_mort"], out["apache_hosp_mort"] = col("apache_icu_mort"), col("apache_hosp_mort")
    out["apache_icu_los"] = col("apache_icu_los")
    out["saps3"] = np.where(sev_name == "saps3", sev, np.nan)
    out["saps3_prob"] = cl.saps3_prob(out["saps3"])
    out["asa"] = np.where(sev_name == "asa", sev, np.nan)

    def normal(c, name):
        x = col(c)
        return np.where(np.isfinite(x), x, NORMAL[name])

    hr_n, sbp_n = normal("heart_rate_last", "heart_rate"), normal("sbp_last", "sbp")
    out["shock_index"] = cl.shock_index(hr_n, sbp_n)
    out["lactate"] = normal("lactate_last", "lactate")
    out["neg_rox"] = -cl.rox(normal("spo2_last", "spo2"), fio2_estimate(fio2, dev, col("lpm_last")),
                             normal("respiratory_rate_last", "respiratory_rate"))
    out["neg_map"] = -normal("map_last", "map")
    out["kdigo"] = np.maximum.reduce([cl.kdigo_creatinine(col("creat_48h"), col("creat_low48"), col("creat_low7")),
                                      np.nan_to_num(col("kdigo_uo")), np.nan_to_num(col("kdigo_crrt"))])
    out["creatinine"] = np.where(np.isfinite(col("creat_48h")), col("creat_48h"), NORMAL["creatinine"])
    out["potassium"] = normal("potassium_last", "potassium")
    out["neg_platelets"] = -normal("platelet_count_last", "platelet_count")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", required=True)
    for k in ("clif", "derived", "raw", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    a = ap.parse_args(argv)
    import pandas as pd

    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for dd in map(Path, a.designs):
        design = json.loads((dd / "design.json").read_text())["design"]
        an = pd.read_parquet(dd / "anchors.parquet").drop_duplicates("anchor")
        frames.append(pd.DataFrame({"key": design + "|" + an["anchor"].astype(str), "design": design,
                                    "anchor": an["anchor"].astype(str).to_numpy(),
                                    "stay_id": an["twin"].astype(str).to_numpy(),
                                    "dataset": an["dataset"].astype(str).to_numpy(),
                                    "t_min": an["t_min"].to_numpy(dtype=np.int64)}))
    anchors = pd.concat(frames, ignore_index=True)
    parts = []
    for ds, g in anchors.groupby("dataset"):
        if a.datasets and ds not in a.datasets:
            continue
        t1 = time.time()
        v = extract(g, Path(a.clif) / ds, Path(a.derived) / ds, Path(a.raw) / ds, out / "work")
        v = v.join(g.set_index("key")[["design", "anchor", "dataset", "t_min"]])
        parts.append(v)
        print(json.dumps({"dataset": ds, "anchors": len(g), "rows": len(v), "seconds": round(time.time() - t1, 1)}),
              flush=True)
    shutil.rmtree(out / "work", ignore_errors=True)
    var = fill_map(pd.concat(parts))
    sc = scores(var.copy())
    sc = sc.join(var[["design", "anchor", "dataset", "stay_id"]])
    var.reset_index().to_parquet(out / "variables.parquet", index=False)
    sc.reset_index().to_parquet(out / "scores.parquet", index=False)
    rep = {"anchors": int(len(sc)), "coverage": {}, "medians": {}, "seconds": None}
    for (design, ds), g in var.groupby(["design", "dataset"]):
        rep["coverage"].setdefault(design, {})[ds] = {c: float(g[c].notna().mean()) for c in COVERAGE if c in g}
    for (design, ds), g in sc.groupby(["design", "dataset"]):
        rep["medians"].setdefault(design, {})[ds] = {c: (float(g[c].median()) if g[c].notna().any() else None)
                                                     for c in sc.columns if c not in ("design", "anchor", "dataset",
                                                                                      "stay_id")}
    rep["seconds"] = round(time.time() - t0, 1)
    (out / "clinical.json").write_text(json.dumps(rep, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({"anchors": rep["anchors"], "seconds": rep["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
