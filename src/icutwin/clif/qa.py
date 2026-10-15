"""S3 cross-dataset QA: per category, coverage and value distribution in every dataset, with automatic flags.

For every numeric measure of canonical CLIF (labs, vitals, hemodynamics, assessments, medication doses per
unit, fluids, respiratory settings, CRRT and MCS settings) and every dataset:
* coverage: share of the dataset's ICU hospitalizations with at least one value;
* n, quantiles (1, 25, 50, 75, 99%).
Flags, for categories present (n >= ``min_n``) in at least ``min_datasets`` datasets:
* ``median_ratio``: the dataset median differs from the median of the dataset medians by more than
  ``ratio`` (either way) - the signature of a unit or scale error;
* ``iqr_disjoint``: the dataset's IQR does not overlap the IQR of the other datasets' medians-range
  (pooled 25th-75th percentile of their medians);
* ``coverage_gap``: coverage below ``gap`` x the median coverage of the other datasets, when that median is
  at least ``min_coverage``.
A flag is a question, not a verdict: practice and case mix differ between sites. Every flag is reviewed and
explained. Output: ``qa_report.json`` (all statistics) and ``qa_report.md`` (flags).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from icutwin.clif.conform import connect

# table -> (category expression, value column, unit column)
MEASURES = {
    "labs": ("lab_category", "lab_value_numeric", None),
    # blood pressure by site (arterial and cuff are separate canonical measurements); other vitals across sites
    "vitals": ("vital_category || CASE WHEN vital_category IN ('sbp', 'dbp', 'map') "
               "THEN coalesce('/' || meas_site_name, '') ELSE '' END", "vital_value", None),
    "invasive_hemodynamics": ("measurement_category", "measurement_value", None),
    "patient_assessments": ("assessment_category", "numerical_value", None),
    # doses: rates and amounts given (stop rows carry 0); intermittent doses per route (oral and IV doses of
    # one drug differ by design: vancomycin 125 mg PO vs 1 g IV)
    "medication_admin_continuous": ("med_category", "CASE WHEN med_dose > 0 THEN med_dose END", "med_dose_unit"),
    "medication_admin_intermittent": ("med_category || '/' || coalesce(med_route_category, '?')",
                                      "CASE WHEN med_dose > 0 THEN med_dose END", "med_dose_unit"),
    "input": ("input_category", "input_volume", None),
    "output": ("output_category", "output_volume", None),
    "mcs": ("support_category || '/' || setting_category", "setting_value", None),
}
WIDE = {  # wide tables: each numeric column is a category
    "respiratory_support": ("fio2_set", "lpm_set", "tidal_volume_set", "resp_rate_set", "pressure_control_set",
                            "pressure_support_set", "flow_rate_set", "peak_inspiratory_pressure_set",
                            "inspiratory_time_set", "peep_set", "tidal_volume_obs", "resp_rate_obs",
                            "plateau_pressure_obs", "peak_inspiratory_pressure_obs", "peep_obs", "minute_vent_obs",
                            "mean_airway_pressure_obs"),
    "crrt_therapy": ("blood_flow_rate", "pre_filter_replacement_fluid_rate", "post_filter_replacement_fluid_rate",
                     "dialysate_flow_rate", "ultrafiltration_out"),
}


def _p(path: Path) -> str:
    return str(path).replace("\\", "/")


def _stats_sql(src: str, cat: str, val: str, unit: str | None) -> str:
    key = f"({cat})" + (f" || ' [' || coalesce({unit}, '?') || ']'" if unit else "")
    return f"""
        SELECT {key} AS category, count({val}) AS n, count(DISTINCT CASE WHEN {val} IS NOT NULL THEN hospitalization_id END) AS hosp,
               quantile_cont({val}, [0.01, 0.25, 0.5, 0.75, 0.99]) AS q
        FROM {src} WHERE {cat} IS NOT NULL GROUP BY 1"""


def dataset_stats(clif_dir: Path) -> dict:
    con = connect()
    stays = clif_dir / "clif_ext_icu_stay.parquet"
    hosps = con.execute(f"SELECT count(DISTINCT hospitalization_id) FROM read_parquet('{_p(stays)}')").fetchone()[0]
    out: dict = {"icu_hospitalizations": hosps, "tables": {}, "coverage": {}}
    # only hospitalizations with an ICU stay count (adapters may emit ward-only admissions)
    icu = f"(SELECT DISTINCT hospitalization_id FROM read_parquet('{_p(stays)}'))"
    for table, (cat, val, unit) in MEASURES.items():
        f = clif_dir / f"clif_{table}.parquet"
        if not f.exists():
            continue
        src = f"(SELECT * FROM read_parquet('{_p(f)}') WHERE hospitalization_id IN {icu})"
        rows = con.execute(_stats_sql(src, cat, val, unit)).fetchall()
        out["tables"][table] = {r[0]: {"n": r[1], "coverage": round(r[2] / max(hosps, 1), 4),
                                       "q": [None if x is None else round(x, 4) for x in (r[3] or [None] * 5)]}
                                for r in rows if r[1]}
        # coverage of the concept itself, whatever its unit or route (a drug charted in mcg/min in one dataset
        # and mcg/kg/min in another is covered in both)
        base = cat.split(" || '/' ||")[0] if table.startswith("medication") else cat
        out["coverage"][table] = dict(con.execute(f"""
            SELECT {base}, round(count(DISTINCT hospitalization_id) / {max(hosps, 1)}, 4)
            FROM {src} WHERE {base} IS NOT NULL GROUP BY 1""").fetchall())
    for table, cols in WIDE.items():
        f = clif_dir / f"clif_{table}.parquet"
        if not f.exists():
            continue
        names = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{_p(f)}')").fetchall()}
        src = f"(SELECT * FROM read_parquet('{_p(f)}') WHERE hospitalization_id IN {icu})"
        res = {}
        for col in (c for c in cols if c in names):
            n, hosp, q = con.execute(f"""SELECT count({col}), count(DISTINCT CASE WHEN {col} IS NOT NULL THEN hospitalization_id END),
                                               quantile_cont({col}, [0.01, 0.25, 0.5, 0.75, 0.99]) FROM {src}""").fetchone()
            if n:
                res[col] = {"n": n, "coverage": round(hosp / max(hosps, 1), 4), "q": [round(x, 4) for x in q]}
        out["tables"][table] = res
        out["coverage"][table] = {c: v["coverage"] for c, v in res.items()}
    return out


def _flags(stats: dict[str, dict], min_n: int, min_datasets: int, ratio: float, gap: float,
           min_coverage: float) -> list[dict]:
    flags = []
    tables = sorted({t for s in stats.values() for t in s["tables"]})
    for table in tables:
        cats = sorted({c for s in stats.values() for c in s["tables"].get(table, {})})
        for cat in cats:
            have = {ds: s["tables"][table][cat] for ds, s in stats.items()
                    if cat in s["tables"].get(table, {}) and s["tables"][table][cat]["n"] >= min_n}
            if len(have) >= min_datasets:
                medians = {ds: v["q"][2] for ds, v in have.items()}
                for ds, v in have.items():
                    others = [m for d, m in medians.items() if d != ds]
                    ref = statistics.median(others)
                    med, lo, hi = v["q"][2], v["q"][1], v["q"][3]
                    if ref and med and ref > 0 and med > 0 and not (1 / ratio <= med / ref <= ratio):
                        flags.append({"table": table, "category": cat, "dataset": ds, "flag": "median_ratio",
                                      "median": med, "others_median": ref, "ratio": round(med / ref, 3)})
                    q = statistics.quantiles(others, n=4) if len(others) >= 2 else [others[0]] * 3
                    if hi < q[0] or lo > q[2]:
                        flags.append({"table": table, "category": cat, "dataset": ds, "flag": "iqr_disjoint",
                                      "iqr": [lo, hi], "others_medians_iqr": [round(q[0], 4), round(q[2], 4)]})
        concepts = sorted({c for s in stats.values() for c in s.get("coverage", {}).get(table, {})})
        for cat in concepts:
            covs = {ds: s["coverage"].get(table, {}).get(cat, 0.0) for ds, s in stats.items()
                    if table in s.get("coverage", {})}
            for ds, cov in covs.items():
                others = [c for d, c in covs.items() if d != ds and c > 0]
                if len(others) >= min_datasets - 1:
                    ref = statistics.median(others)
                    if ref >= min_coverage and cov < gap * ref:
                        flags.append({"table": table, "category": cat, "dataset": ds, "flag": "coverage_gap",
                                      "coverage": cov, "others_coverage": round(ref, 4)})
    return flags


def qa_report(clif_root: Path, datasets: list[str], out_dir: Path, min_n: int = 20, min_datasets: int = 3,
              ratio: float = 3.0, gap: float = 0.2, min_coverage: float = 0.3) -> dict:
    stats = {ds: dataset_stats(clif_root / ds) for ds in datasets}
    flags = _flags(stats, min_n, min_datasets, ratio, gap, min_coverage)
    report = {"datasets": datasets, "thresholds": {"min_n": min_n, "min_datasets": min_datasets, "ratio": ratio,
                                                    "gap": gap, "min_coverage": min_coverage},
              "validation": {}, "stats": stats, "flags": flags}
    for ds in datasets:
        v = clif_root / ds / "_clifpy_validation.json"
        if v.exists():
            r = json.loads(v.read_text(encoding="utf-8"))
            report["validation"][ds] = {"passed": r.get("passed"), "hard_errors": r.get("n_hard_errors"),
                                        "flagged_tables": sorted(r.get("flags") or {})}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qa_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "qa_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict) -> str:
    lines = ["# S3 cross-dataset QA report", "", "## CLIF 3.0 validation", "", "| dataset | passed | hard errors | flagged tables |",
             "|---|---|---|---|"]
    for ds, v in report["validation"].items():
        lines.append(f"| {ds} | {v['passed']} | {v['hard_errors']} | {', '.join(v['flagged_tables']) or '-'} |")
    lines += ["", f"## Distribution and coverage flags ({len(report['flags'])})", "",
              "| table | category | dataset | flag | detail |", "|---|---|---|---|---|"]
    for f in sorted(report["flags"], key=lambda x: (x["flag"], x["table"], x["category"], x["dataset"])):
        detail = {k: v for k, v in f.items() if k not in ("table", "category", "dataset", "flag")}
        lines.append(f"| {f['table']} | {f['category']} | {f['dataset']} | {f['flag']} | {detail} |")
    return "\n".join(lines) + "\n"
