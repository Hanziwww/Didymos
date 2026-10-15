"""S4: canonical CLIF + ELF events -> one MEDS v0.4 dataset per source, with the ICU stay as subject.

Events come from CLIF_MEDS (its ELF domains, ``<elf_dir>/data``) and from our ELF extension domains
(``meds/elf_ext``, converted here into ``<elf_dir>/ext`` by ``icutwin.meds.elf``). Both key events by patient
(``subject_id``) and hospitalization. Each ICU stay of ``ext_icu_stay`` becomes one MEDS subject holding
* static events of its patient (time NULL: sex, race, ethnicity, dataset, site) and the patient's birth and
  death events;
* its hospitalization's events from ``pre_icu_hours`` before ICU admission to ICU discharge, except ICU-stay
  events (header, outcome) of other stays: an earlier stay inside the window is history carried by ADT;
  stays without ICU time (discharge not after admission; eICU zero-length unit stays) and stays of patients
  younger than ``min_age_years`` at hospital admission (the adult cohort) are not subjects;
* the hospitalization's header and outcome events whatever their time: admission type and age at hospital
  admission, discharge disposition and billing diagnoses at hospital discharge (HOSP, HOSP_DX).
Events are keyed by hospitalization, never by patient time alone: eICU health-system stays of one patient can
overlap on the pseudo-timeline. Identical events (subject, time, code, value, ids) are kept once, as in
MEDS-Transforms (some sources repeat raw rows, and CLIF keeps every row), except doses and volumes
(``REPEATABLE_DOMAINS``): ELF codes carry the category, so two administrations of one category with the same
amount at the same time are two events.

Subject ids are a 63-bit hash of the stay id (stable between subsets and full data, and across datasets);
``metadata/stays.parquet`` maps them back. Splits are by patient (all stays of a patient share one), from a
hash of dataset and patient id, so a patient keeps its split in every extract. Within a subject, events are
ordered by time (static first), then by a fixed domain order, then code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import UTC, datetime
from functools import lru_cache
from importlib import resources
from pathlib import Path

import meds
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from icutwin.clif.conform import connect, extensions
from icutwin.meds.domains import DOMAIN_ORDER  # noqa: F401  (re-exported)
from icutwin.meds.elf import config_table, convert_domain, load_config, qualified_config
from icutwin.meds.streams import STREAM_COLUMN, restream_infusions

log = logging.getLogger(__name__)

# tie order at one timestamp (the plan's fixed CLIF-table order); unknown domains sort last
HEADER_OUTCOME_DOMAINS = ("HOSP", "HOSP_DX")
# MEDS domain -> (CLIF table, category column) whose declared extensions (``extensions.yaml``) code descriptions mark
EXTENSION_COLUMNS = {"MED_INT": ("medication_admin_intermittent", "med_category"),
                     "MED_CON": ("medication_admin_continuous", "med_category"),
                     "MED_ORD": ("medication_orders", "med_category"), "LAB": ("labs", "lab_category")}
REPEATABLE_DOMAINS = ("MED_INT", "INPUT", "OUTPUT")   # amounts: identical events are separate administrations
SKIP_CLIF_MEDS = ("ECMO_MCS",)   # reads the CLIF 2.1 table; canonical mcs is our MCS domain
SPLITS = (("train", 0.80), ("tuning", 0.05), ("held_out", 0.15))
DOMAIN_LABELS = {  # descriptions (CLIF_MEDS code/metadata.py labels plus our domains)
    "VITAL": "Vital sign", "LAB": "Laboratory", "MED_CON": "Continuous medication",
    "MED_INT": "Intermittent medication", "RESP": "Respiratory support", "PA": "Patient assessment",
    "CODE_STATUS": "Code status", "HOSP": "Hospitalization", "PATIENT": "Patient", "ADT": "ADT",
    "POS": "Position", "CRRT": "CRRT", "PROC": "Procedure", "HOSP_DX": "Hospital diagnosis",
    "ICU": "ICU stay", "DATASET": "Source dataset", "SITE": "Site", "HEMO": "Invasive hemodynamics",
    "INPUT": "Fluid input", "OUTPUT": "Fluid output", "IHD": "Intermittent hemodialysis",
    "MCS": "Mechanical circulatory support", "AIRWAY": "Artificial airway", "LINE": "Vascular line",
    "DRAIN": "Drain or tube", "MED_ORD": "Medication order", "MICRO_ORDER": "Microbiology specimen",
    "MICRO": "Microbiology culture", "MICRO_NC": "Microbiology PCR", "MICRO_SUS": "Antimicrobial susceptibility",
    "PT_DX": "Problem-list diagnosis",
}


def _p(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def drop_minors(con, clif_dir: Path, min_age_years: float | None) -> dict:
    """Cohort rule (plan S2: adult ICU stays): delete from the temp table ``stays`` (a ``hospitalization_id``
    column) the stays whose hospitalization's ``age_at_admission`` is below ``min_age_years``; None keeps every
    age, and a stay of unknown age is kept. Binned ages (AmsterdamUMCdb, HiRID, SICdb) are compared at their
    stored bin midpoint, so SICdb's 15-19 bin (midpoint 17) is excluded. Returns the counts."""
    hosp = clif_dir / "clif_hospitalization.parquet"
    if min_age_years is None or not hosp.exists() or "age_at_admission" not in pq.read_schema(hosp).names:
        return {"min_age_years": min_age_years, "stays_below_min_age": 0, "stays_age_unknown": None}
    key = lambda c: f"coalesce(CAST(TRY_CAST({c} AS BIGINT) AS VARCHAR), CAST({c} AS VARCHAR))"  # noqa: E731
    ages = f"SELECT {key('hospitalization_id')} AS h, CAST(age_at_admission AS DOUBLE) AS age FROM read_parquet('{_p(hosp)}')"
    below, unknown = con.execute(f"""
        SELECT count(*) FILTER (WHERE a.age < $1), count(*) FILTER (WHERE a.age IS NULL)
        FROM stays s LEFT JOIN ({ages}) a ON a.h = {key('s.hospitalization_id')}""", [float(min_age_years)]).fetchone()
    con.execute(f"DELETE FROM stays WHERE {key('hospitalization_id')} IN (SELECT h FROM ({ages}) WHERE age < $1)",
                [float(min_age_years)])
    return {"min_age_years": min_age_years, "stays_below_min_age": below, "stays_age_unknown": unknown}


def stay_subject_id(stay_id: str) -> int:
    """63-bit MEDS subject id of an ICU stay (blake2b of the stay id; stays ids carry their dataset)."""
    return int.from_bytes(hashlib.blake2b(stay_id.encode(), digest_size=8).digest(), "big") & (2**63 - 1)


def patient_split(dataset: str, patient_id: str) -> str:
    u = int.from_bytes(hashlib.blake2b(f"{dataset}:{patient_id}".encode(), digest_size=8).digest(), "big") / 2**64
    acc = 0.0
    for name, frac in SPLITS:
        acc += frac
        if u < acc:
            return name
    return SPLITS[-1][0]


def extension_configs() -> dict[str, dict]:
    root = resources.files("icutwin.meds").joinpath("elf_ext")
    return {Path(f.name).stem: load_config(Path(str(f))) for f in root.iterdir() if f.name.endswith(".yaml")}


@lru_cache(maxsize=1)
def qualifiers() -> dict[str, dict]:
    """``meds/qualifiers.yaml``: domain -> {level, where}: the extension level appended to its codes."""
    text = resources.files("icutwin.meds").joinpath("qualifiers.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


MEDICATION_DOMAINS = ("MED_CON", "MED_INT", "MED_ORD")


def _qualify(domain: str, cfg: dict) -> dict:
    """The domain's config with its qualifier level (``meds/qualifiers.yaml``); medication events also carry the
    route's basis (``med_route_basis``) as the extra column ``route_basis``."""
    q = qualifiers().get(domain)
    out = qualified_config(cfg, q["level"], q.get("where")) if q else cfg
    if domain in MEDICATION_DOMAINS:
        table, concepts = config_table(out)
        out = {**out, table: {k: {**m, "extra": {"route_basis": "med_route_basis"}} for k, m in concepts.items()}}
    return out


def base_code_levels(cfg: dict) -> int:
    """Parts of the domain's unqualified code (its first concept): a qualified code has one more."""
    _, concepts = config_table(cfg)
    code = next(iter(concepts.values()))["code"]
    return len(code) if isinstance(code, list) else 1


def convert_extensions(clif_dir: Path, elf_dir: Path) -> dict[str, int]:
    """Our ELF extension domains (qualified per ``meds/qualifiers.yaml``) -> ``<elf_dir>/ext/<DOMAIN>.parquet``."""
    out_dir = elf_dir / "ext"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    con, counts = connect(), {}
    for domain, cfg in sorted(extension_configs().items()):
        n = convert_domain(con, clif_dir, _qualify(domain, cfg), domain, out_dir / f"{domain}.parquet")
        if n is not None:
            counts[domain] = n
    return counts


def convert_qualified(clif_dir: Path, elf_dir: Path) -> dict[str, int]:
    """CLIF_MEDS domains with a qualifier (``meds/qualifiers.yaml``), run by our engine from the exact concept
    configs of the CLIF_MEDS run (``<elf_dir>/_concept_config``) with the level appended ->
    ``<elf_dir>/qualified/<DOMAIN>.parquet``; they replace CLIF_MEDS's unqualified files in the build."""
    out_dir = elf_dir / "qualified"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    con, counts = connect(), {}
    provenance = elf_dir / "_provenance.json"   # clif.external.PROVENANCE: domains CLIF_MEDS left to our engine
    engine_only = set(json.loads(provenance.read_text(encoding="utf-8")).get("engine_only", [])) \
        if provenance.exists() else set()
    for domain in sorted(qualifiers()):
        cfg_path = elf_dir / "_concept_config" / f"{domain}.yaml"
        if domain in extension_configs() or not ((elf_dir / "data" / f"{domain}.parquet").exists()
                                                 or domain in engine_only):
            continue
        cfg = _qualify(domain, load_config(cfg_path))
        if domain == "MED_CON":   # each event names its canonical stream, so one stream per code can be restored
            table, concepts = config_table(cfg)
            cfg = {**cfg, table: {k: {**m, "text_value": STREAM_COLUMN} for k, m in concepts.items()}}
        path = out_dir / f"{domain}.parquet"
        # causal conversion: only a weight charted at or before a dose converts it, and an infusion keeps the unit
        # its segment started with (CLIF_MEDS converts row by row, so an infusion would change code mid-way)
        stats: dict = {}
        n = convert_domain(con, clif_dir, cfg, domain, path, weight_lookup="segment_start", stats=stats)
        if n is not None and domain == "MED_CON":
            report = {**stats, **restream_infusions(con, path)}
            (out_dir / "_restream.json").write_text(json.dumps(report), encoding="utf-8")
            n = pq.ParquetFile(path).metadata.num_rows
        if n is not None:
            counts[domain] = n
    return counts


def code_levels(elf_dir: Path) -> dict[str, int]:
    """Qualified domain -> parts of its unqualified (ELF) code."""
    out = {}
    for domain in qualifiers():
        cfg = extension_configs().get(domain)
        if cfg is None and (path := elf_dir / "_concept_config" / f"{domain}.yaml").exists():
            cfg = load_config(path)
        if cfg is not None:
            out[domain] = base_code_levels(cfg)
    return out


def parent_code(code: str, levels: dict[str, int]) -> str | None:
    """The ELF code a qualified code extends (None for unqualified codes)."""
    parts = code.split("//")
    n = levels.get(parts[0])
    return "//".join(parts[:n]) if n is not None and len(parts) == n + 1 else None


def _event_files(elf_dir: Path) -> dict[str, Path]:
    qualified = {p.stem: p for p in sorted((elf_dir / "qualified").glob("*.parquet"))}
    files = {p.stem: p for p in sorted((elf_dir / "data").glob("*.parquet"))
             if p.stem not in SKIP_CLIF_MEDS and p.stem not in qualified}
    for p in [*sorted((elf_dir / "ext").glob("*.parquet")), *qualified.values()]:
        if p.stem in files:
            raise ValueError(f"extension domain {p.stem} collides with a CLIF_MEDS domain")
        files[p.stem] = p
    return files


@lru_cache(maxsize=None)
def _extension_origins(domain: str) -> dict[str, str]:
    """Declared extensions of a domain's category vocabulary -> where they come from (not CLIF 3.0)."""
    table, column = EXTENSION_COLUMNS[domain]
    declared = (extensions().get(table) or {}).get(column) or {}
    return {cat: (spec or {}).get("origin") or ("local concept" if cat.startswith("ext_") else "consortium practice")
            for cat, spec in declared.items()}


def _description(code: str) -> str:
    if code in (meds.birth_code, meds.death_code):
        return {meds.birth_code: "Patient birth event", meds.death_code: "Patient death event"}[code]
    parts = code.split("//")
    label = DOMAIN_LABELS.get(parts[0], parts[0])
    concept = parts[1].replace("_", " ") if len(parts) > 1 else ""
    levels = [p for p in parts[2:] if p not in ("NA", "UNK")]
    text = f"{label}: {concept}" + (f" ({', '.join(levels)})" if levels else "")
    if parts[0] in EXTENSION_COLUMNS and len(parts) > 1 and parts[1] in _extension_origins(parts[0]):
        text += f" [not CLIF 3.0 vocabulary: {_extension_origins(parts[0])[parts[1]]} extension]"
    route = _extension_routes(parts[0]).get(parts[-1]) if len(parts) > 2 else None
    if route:   # a route outside the table's CLIF 3.0 route list (qualifier level)
        text += f" [route '{parts[-1]}' not CLIF 3.0 vocabulary of the table: {route} extension]"
    return text


@lru_cache(maxsize=None)
def _extension_routes(domain: str) -> dict[str, str]:
    """Declared route extensions of the medication table a domain's qualifier level comes from."""
    if (qualifiers().get(domain) or {}).get("level") != "med_route_category" or domain not in EXTENSION_COLUMNS:
        return {}
    table = EXTENSION_COLUMNS[domain][0]
    declared = (extensions().get(table) or {}).get("med_route_category") or {}
    return {r: (spec or {}).get("origin") or "local" for r, spec in declared.items()}


def build_dataset(name: str, clif_dir: Path, elf_dir: Path, out_dir: Path, pre_icu_hours: float = 48.0,
                  subjects_per_shard: int = 50_000, dataset_version: str = "",
                  min_age_years: float | None = 18.0) -> dict:
    """Write the MEDS v0.4 dataset of one source to ``out_dir``; returns a report."""
    ext = {**convert_extensions(clif_dir, elf_dir), **convert_qualified(clif_dir, elf_dir)}
    files = _event_files(elf_dir)
    levels = code_levels(elf_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "metadata").mkdir(parents=True)
    con = connect()
    con.create_function("stay_subject_id", stay_subject_id, ["VARCHAR"], "BIGINT")
    con.create_function("patient_split", patient_split, ["VARCHAR", "VARCHAR"], "VARCHAR")
    con.execute(f"""
        CREATE TEMP TABLE stays AS
        SELECT stay_subject_id(stay_id) AS subject_id, stay_id, dataset,
               CAST(patient_id AS BIGINT) AS patient_id, CAST(hospitalization_id AS BIGINT) AS hospitalization_id,
               CAST(icu_in_dttm AS TIMESTAMP) AS icu_in, CAST(icu_out_dttm AS TIMESTAMP) AS icu_out,
               CAST(icu_in_dttm AS TIMESTAMP) - to_microseconds(CAST({pre_icu_hours} * 3600e6 AS BIGINT)) AS window_start,
               patient_split(dataset, patient_id) AS split, coalesce(clock_valid, false) AS clock_valid
        FROM read_parquet('{_p(clif_dir / 'clif_ext_icu_stay.parquet')}')""")
    # cohort: a stay needs ICU time (eICU registers zero-length unit stays at transfers) and an adult patient
    excluded = con.execute("SELECT count(*) FROM stays WHERE NOT icu_out > icu_in").fetchone()[0]
    con.execute("DELETE FROM stays WHERE NOT icu_out > icu_in OR icu_out IS NULL OR icu_in IS NULL")
    ages = drop_minors(con, clif_dir, min_age_years)
    dup = con.execute("SELECT count(*) - count(DISTINCT subject_id) FROM stays").fetchone()[0]
    if dup:
        raise ValueError(f"{name}: {dup} subject-id collisions")
    ranks = " ".join(f"WHEN '{d}' THEN {i}" for i, d in enumerate(DOMAIN_ORDER))
    union = " UNION ALL ".join(
        f"SELECT '{d}' AS domain, subject_id AS patient_id, time, code, numeric_value, text_value, "
        + ("hospitalization_id" if "hospitalization_id" in pq.read_schema(f).names else "CAST(NULL AS BIGINT)")
        + " AS hospitalization_id, "
        + ("route_basis" if "route_basis" in pq.read_schema(f).names else "CAST(NULL AS VARCHAR)")
        + f" AS route_basis FROM read_parquet('{_p(f)}')" for d, f in files.items())
    con.execute(f"CREATE TEMP VIEW events AS {union}")
    con.execute(f"""
        CREATE TEMP TABLE windowed_all AS
        WITH patient_level AS (      -- the patient's statics, birth and death (PATIENT has no hospitalization)
            SELECT s.subject_id, e.* FROM events e JOIN stays s ON s.patient_id = e.patient_id
            WHERE e.domain = 'PATIENT'),
        hosp_level AS (
            SELECT s.subject_id, e.* FROM events e JOIN stays s ON s.hospitalization_id = e.hospitalization_id
            WHERE e.domain <> 'PATIENT' AND (e.time IS NULL OR e.domain IN {HEADER_OUTCOME_DOMAINS}
               OR e.time BETWEEN s.window_start AND s.icu_out)
              -- ICU header and outcome events (text value: their stay id) belong to their own stay only:
              -- an earlier stay of the hospitalization inside the pre-ICU window is history, carried by ADT
              AND (e.domain <> 'ICU' OR e.time IS NULL OR e.text_value = s.stay_id))
        SELECT subject_id, time, code, numeric_value,
               CASE WHEN domain = 'ICU' THEN NULL ELSE text_value END AS text_value, patient_id, hospitalization_id,
               route_basis, CASE domain {ranks} ELSE {len(DOMAIN_ORDER)} END AS domain_rank
        FROM (SELECT * FROM patient_level UNION ALL SELECT * FROM hosp_level)""")
    # infusions running at the window start: a 'going' event at the window start with the rate in effect, from the
    # code's last event before the window (causal: the encoder never infers a running infusion from a later event)
    running = con.execute(f"""
        INSERT INTO windowed_all BY NAME
        WITH pre AS (
            SELECT s.subject_id, s.window_start, e.*,
                   split_part(e.code, '//', 2) AS drug, split_part(e.code, '//', 3) AS unit,
                   array_to_string(string_split(e.code, '//')[5:], '//') AS route,
                   row_number() OVER (PARTITION BY s.subject_id, split_part(e.code, '//', 2), split_part(e.code, '//', 3),
                                                   array_to_string(string_split(e.code, '//')[5:], '//')
                                      ORDER BY e.time DESC, e.code) AS k
            FROM events e JOIN stays s ON s.hospitalization_id = e.hospitalization_id
            WHERE e.domain = 'MED_CON' AND e.time < s.window_start)
        SELECT subject_id, window_start AS time,
               'MED_CON//' || drug || '//' || unit || '//going' || CASE WHEN route = '' THEN '' ELSE '//' || route END
                   AS code,
               numeric_value, CAST(NULL AS VARCHAR) AS text_value, patient_id, hospitalization_id, route_basis,
               {DOMAIN_ORDER.index('MED_CON')} AS domain_rank
        FROM pre WHERE k = 1 AND split_part(code, '//', 4) <> 'stop'""").fetchone()[0]
    # amounts can repeat for real (two flushes of 10 mL through two lines; two tablets of one drug): ELF codes
    # carry the category, not the item, so identical dose and volume events stay
    keep = ", ".join(str(DOMAIN_ORDER.index(d)) for d in REPEATABLE_DOMAINS)
    con.execute(f"""CREATE TEMP TABLE windowed AS
                    SELECT DISTINCT * FROM windowed_all WHERE domain_rank NOT IN ({keep})
                    UNION ALL SELECT * FROM windowed_all WHERE domain_rank IN ({keep})""")
    repeated = con.execute("SELECT (SELECT count(*) FROM windowed_all) - (SELECT count(*) FROM windowed)").fetchone()[0]
    con.execute("DROP TABLE windowed_all")
    restreamed = elf_dir / "qualified" / "_restream.json"
    report = {"dataset": name, "pre_icu_hours": pre_icu_hours, "extension_domains": ext, "splits": {},
              "infusion_streams": json.loads(restreamed.read_text(encoding="utf-8")) if restreamed.exists() else None,
              "stays_without_icu_time": excluded, **ages, "identical_events_dropped": repeated,
              "infusions_running_at_window_start": running}
    for split, _ in SPLITS:
        subjects = [r[0] for r in con.execute(
            f"SELECT subject_id FROM stays WHERE split = '{split}' ORDER BY subject_id").fetchall()]
        for k in range(0, max(len(subjects), 1), subjects_per_shard):
            chunk = subjects[k:k + subjects_per_shard]
            if not chunk:
                continue
            path = out_dir / "data" / split / f"{k // subjects_per_shard}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            con.execute("CREATE OR REPLACE TEMP TABLE shard_subjects AS SELECT unnest(?) AS subject_id", [chunk])
            batch = con.execute("""
                SELECT w.subject_id, w.time, w.code, CAST(w.numeric_value AS FLOAT) AS numeric_value, w.text_value,
                       w.patient_id, w.hospitalization_id, w.route_basis
                FROM windowed w JOIN shard_subjects USING (subject_id)
                ORDER BY w.subject_id, w.time NULLS FIRST, w.domain_rank, w.code, w.numeric_value, w.text_value,
                         w.route_basis
                """).arrow().read_all()
            table = meds.DataSchema.align(batch)   # MEDS dtypes (text_value large_string)
            meds.DataSchema.validate(table)
            pq.write_table(table, path, compression="zstd")
        report["splits"][split] = {"subjects": len(subjects)}
    _write_metadata(con, name, out_dir, files, dataset_version, levels)
    report["events"] = con.execute("SELECT count(*) FROM windowed").fetchone()[0]
    report["subjects"] = con.execute("SELECT count(*) FROM stays").fetchone()[0]
    (out_dir / "metadata" / "_build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _write_metadata(con, name: str, out_dir: Path, files: dict[str, Path], dataset_version: str,
                    levels: dict[str, int] | None = None) -> None:
    meta = out_dir / "metadata"
    con.execute(f"""COPY (SELECT subject_id, split FROM stays ORDER BY subject_id)
                    TO '{_p(meta / 'subject_splits.parquet')}' (FORMAT parquet)""")
    con.execute(f"""COPY (SELECT subject_id, stay_id, dataset, patient_id, hospitalization_id, icu_in, icu_out,
                                 window_start, split, clock_valid FROM stays ORDER BY subject_id)
                    TO '{_p(meta / 'stays.parquet')}' (FORMAT parquet)""")
    codes = [r[0] for r in con.execute("SELECT DISTINCT code FROM windowed ORDER BY 1").fetchall()]
    parents = [parent_code(c, levels or {}) for c in codes]   # a qualified code extends its ELF code
    code_table = meds.CodeMetadataSchema.align(pa.table({
        "code": pa.array(codes, pa.string()), "description": pa.array([_description(c) for c in codes], pa.string()),
        "parent_codes": pa.array([[p] if p else None for p in parents], pa.list_(pa.string()))}))
    meds.CodeMetadataSchema.validate(code_table)
    pq.write_table(code_table, meta / "codes.parquet")
    splits = meds.SubjectSplitSchema.align(pq.read_table(meta / "subject_splits.parquet"))
    meds.SubjectSplitSchema.validate(splits)
    pq.write_table(splits, meta / "subject_splits.parquet")
    from importlib.metadata import PackageNotFoundError, version
    try:
        etl_version = version("icutwin")
    except PackageNotFoundError:
        etl_version = "0+source"
    dataset = {"dataset_name": f"icutwin-{name}", "dataset_version": dataset_version,
               "etl_name": "icutwin", "etl_version": etl_version, "meds_version": version("meds"),
               "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
               "raw_source_id_columns": ["patient_id", "hospitalization_id"],
               "other_extension_columns": [],
               "data_harmonization_format": "ELF 1.0.0-beta + icutwin extension domains",
               "event_domains": sorted(files)}
    (meta / "dataset.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
