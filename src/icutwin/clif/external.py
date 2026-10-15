"""Run the published CLIF reference ETLs (pinned in ``refs/REFS.lock``) unchanged, in their own uv envs.

* CLIF-MIMIC            — MIMIC-IV 3.1 → CLIF 2.1 (official CLIF consortium ETL)
* CLIF-AmsterdamUMCdb   — AmsterdamUMCdb 1.0.2 → CLIF 2.1

Each run writes a provenance record (tool commit, generated config, timing, output row counts) next
to the outputs, so every CLIF table can be traced to the exact reference implementation that built it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import yaml

log = logging.getLogger(__name__)

PROVENANCE = "_provenance.json"

MIMIC_TABLES = (
    "patient", "hospitalization", "adt", "vitals", "labs", "patient_assessments", "respiratory_support",
    "medication_admin_continuous", "medication_admin_intermittent", "position", "crrt_therapy", "ecmo_mcs",
    "code_status", "hospital_diagnosis", "patient_procedures", "input", "output",
)


def find_uv() -> str:
    for candidate in (os.environ.get("ICUTWIN_UV"), shutil.which("uv"), r"D:\pymol\Scripts\uv.exe"):
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("uv executable not found; set ICUTWIN_UV")


def tool_env(cache_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)  # the tool's own .venv must be used, never ours
    env.setdefault("UV_CACHE_DIR", str(cache_root / "uv"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(cache_root / "python"))
    env.setdefault("UV_PYTHON_PREFERENCE", "only-managed")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args],
                          capture_output=True, text=True, check=check)


def git_commit(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def ensure_patches(tool_dir: Path, patch_dir: Path) -> list[dict]:
    """Apply our versioned patches (``patches/<tool>/*.patch``) idempotently; return their hashes."""
    applied = []
    for patch in sorted(patch_dir.glob("*.patch")) if patch_dir.exists() else []:
        if _git(tool_dir, "apply", "--reverse", "--check", str(patch), check=False).returncode != 0:
            _git(tool_dir, "apply", str(patch))
            log.info("applied %s to %s", patch.name, tool_dir.name)
        applied.append({"patch": patch.name, "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()})
    return applied


def _run(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"{cmd} failed in {cwd} (exit {proc.returncode}); log tail:\n{tail}")
    return time.time() - t0


def _row_counts(out_dir: Path) -> dict[str, int]:
    """Row counts of the tool's own tables (our supplement parts ``clif_<table>+<part>`` are not tool output)."""
    return {p.stem: pq.ParquetFile(p).metadata.num_rows for p in sorted(out_dir.glob("clif_*.parquet"))
            if "+" not in p.stem}


def _write_provenance(out_dir: Path, record: dict) -> dict:
    record["tables"] = _row_counts(out_dir)
    record["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    (out_dir / PROVENANCE).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def run_clif_mimic(tool_dir: Path, raw_dir: Path, work_dir: Path, out_dir: Path, cache_root: Path,
                   run_name: str, patch_dir: Path, tables: tuple[str, ...] = MIMIC_TABLES) -> dict:
    """CLIF-MIMIC reads ``config/config.json`` and writes to ``output/<run_name>/`` inside its repo."""
    patches = ensure_patches(tool_dir, patch_dir)
    parquet_dir = work_dir / "mimic_parquet"
    for module in ("hosp", "icu"):  # the tool writes {hosp,icu}/<table>.parquet when these exist
        (parquet_dir / module).mkdir(parents=True, exist_ok=True)
    config = {
        "current_workspace": "icutwin",
        "icutwin": {"mimic_csv_dir": raw_dir.as_posix(), "mimic_parquet_dir": parquet_dir.as_posix()},
        "create_mimic_parquet_from_csv": 1,
        "overwrite_existing_mimic_parquet": 0,
        "clif_output_dir_name": run_name,
        "mimic_version": "3.1",
        "clif_version": "2.1",
        "clif_tables": {t: int(t in tables) for t in MIMIC_TABLES},
    }
    (tool_dir / "config" / "config.json").write_text(json.dumps(config, indent=4), encoding="utf-8")
    started = datetime.now(UTC).isoformat(timespec="seconds")
    seconds = _run([find_uv(), "run", "--frozen", "python", "main.py"], tool_dir, tool_env(cache_root),
                   out_dir / "_tool.log")
    produced = tool_dir / "output" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in produced.glob("clif_*.parquet"):
        shutil.copy2(p, out_dir / p.name)
    return _write_provenance(out_dir, {
        "tool": "CLIF-MIMIC", "commit": git_commit(tool_dir), "patches": patches, "config": config,
        "started_at": started, "seconds": round(seconds, 1),
    })


AMSTEL_FILES = ("data/mappings/procedureorderitems_item.usagi.csv",)   # what the tool reads from ref/AMSTEL


def stage_amstel(tool_dir: Path, amstel_dir: Path) -> list[dict]:
    """CLIF-AmsterdamUMCdb reads AMSTEL's procedure mapping from ``ref/AMSTEL`` inside its own checkout (git
    ignored there); it comes from the AMSTEL checkout pinned in ``refs/REFS.lock``. Returns the files' sha256."""
    placed = []
    for rel in AMSTEL_FILES:
        src, dst = amstel_dir / rel, tool_dir / "ref" / "AMSTEL" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        placed.append({"file": rel, "amstel_commit": git_commit(amstel_dir),
                       "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()})
    return placed


def run_clif_aumc(tool_dir: Path, raw_dir: Path, out_dir: Path, cache_root: Path, patch_dir: Path) -> dict:
    """CLIF-AmsterdamUMCdb takes ``--config``; we pass a copy of its default config with our paths."""
    patches = ensure_patches(tool_dir, patch_dir)
    amstel = stage_amstel(tool_dir, tool_dir.parent / "AMSTEL")
    config = yaml.safe_load((tool_dir / "config.yaml").read_text(encoding="utf-8"))
    config.update({
        "input_path": raw_dir.as_posix(),
        "output_path": out_dir.as_posix(),
        "mappings_path": (tool_dir / "mappings").as_posix(),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "_tool_config.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = datetime.now(UTC).isoformat(timespec="seconds")
    seconds = _run([find_uv(), "run", "--frozen", "python", "main.py", "--config", str(cfg_path)], tool_dir,
                   tool_env(cache_root), out_dir / "_tool.log")
    return _write_provenance(out_dir, {
        "tool": "CLIF-AmsterdamUMCdb", "commit": git_commit(tool_dir), "patches": patches, "amstel": amstel,
        "config": config, "started_at": started, "seconds": round(seconds, 1),
    })


CLIF_MEDS_DOMAINS = ("VITAL", "LAB", "MED_CON", "MED_INT", "RESP", "PA", "CODE_STATUS", "HOSP", "PATIENT", "ADT",
                     "POS", "CRRT", "ECMO_MCS", "PROC", "HOSP_DX")
CLIF_MEDS_DOMAIN_TABLE = {
    "VITAL": "vitals", "LAB": "labs", "MED_CON": "medication_admin_continuous",
    "MED_INT": "medication_admin_intermittent", "RESP": "respiratory_support", "PA": "patient_assessments",
    "CODE_STATUS": "code_status", "HOSP": "hospitalization", "PATIENT": "patient", "ADT": "adt", "POS": "position",
    "CRRT": "crrt_therapy", "ECMO_MCS": "ecmo_mcs", "PROC": "patient_procedures", "HOSP_DX": "hospital_diagnosis",
}


def _all_null(path: Path, column: str) -> bool:
    """True when a parquet table has rows and ``column``, and every row is NULL there."""
    f = pq.ParquetFile(path)
    if column not in f.schema_arrow.names or not f.metadata.num_rows:
        return False
    return f.read(columns=[column]).column(column).null_count == f.metadata.num_rows


def run_clif_meds(tool_dir: Path, clif_dir: Path, out_dir: Path, cache_root: Path, patch_dir: Path, site: str,
                  domains: tuple[str, ...] = CLIF_MEDS_DOMAINS) -> dict:
    """CLIF -> MEDS with ELF codes (official CLIF consortium ETL), run unchanged in its own env."""
    from icutwin.meds.clif_meds_config import build_config_dir

    patches = ensure_patches(tool_dir, patch_dir)
    # a domain is enabled only if its CLIF table exists (datasets legitimately lack e.g. code_status)
    domains = tuple(d for d in domains if (clif_dir / f"clif_{CLIF_MEDS_DOMAIN_TABLE[d]}.parquet").exists())
    # a medication table without any dose unit (zigongICU: prescription times only) stops CLIF_MEDS's clifpy unit
    # conversion (an all-NULL column is read as INTEGER); our engine converts these domains in every build anyway
    # (meds.build.convert_qualified), so CLIF_MEDS skips them and the provenance names them
    engine_only = tuple(d for d in domains if d in ("MED_CON", "MED_INT") and _all_null(
        clif_dir / f"clif_{CLIF_MEDS_DOMAIN_TABLE[d]}.parquet", "med_dose_unit"))
    domains = tuple(d for d in domains if d not in engine_only)
    concepts = build_config_dir(tool_dir / "config", out_dir / "_concept_config")
    config = yaml.safe_load((tool_dir / "clif_config_template.yaml").read_text(encoding="utf-8"))
    config.update({
        "site": site,
        "data_directory": clif_dir.as_posix(),
        "output_directory": out_dir.as_posix(),
        "timezone": "UTC",
        "concept_config_directory": Path(concepts["config_dir"]).as_posix(),
        "domains": {d: int(d in domains) for d in CLIF_MEDS_DOMAINS},
    })
    cfg_path = out_dir / "_tool_config.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = datetime.now(UTC).isoformat(timespec="seconds")
    seconds = _run([find_uv(), "run", "--frozen", "python", "main.py", "--config", str(cfg_path)], tool_dir,
                   tool_env(cache_root), out_dir / "_tool.log")
    record = {"tool": "CLIF_MEDS", "commit": git_commit(tool_dir), "patches": patches,
              "concept_overlay_sha256": concepts["overlay_sha256"], "config": config,
              **({"engine_only": list(engine_only)} if engine_only else {}),
              "started_at": started, "seconds": round(seconds, 1),
              "data_files": {p.relative_to(out_dir).as_posix(): pq.ParquetFile(p).metadata.num_rows
                             for p in sorted((out_dir / "data").rglob("*.parquet"))},
              "finished_at": datetime.now(UTC).isoformat(timespec="seconds")}
    (out_dir / PROVENANCE).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
