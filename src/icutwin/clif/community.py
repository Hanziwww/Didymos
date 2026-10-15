"""Community MEDS ETLs (pinned in ``refs/REFS.lock``), run unchanged for the S3 item reconciliation
(``icutwin.clif.reconcile``).

Their stage runner (MEDS-Transforms) launches each stage as a shell command with POSIX quoting, so they run
under bash on Linux: on a Linux host (the server) directly, on Windows in WSL2. Each ETL has its own
environment under the Linux home (``ICUTWIN_COMMUNITY_CACHE``, default ``$HOME/.cache/icutwin``). Every
environment is locked by a file in ``community_envs/`` (the validated run's environment, exact versions):

* MEDS-Extract 0.7 ETLs (MIMIC-IV, INSPIRE, NWICU) install from their own ``uv.lock`` (wheels from PyPI's file
  host); the lock file records the environment that produces;
* MEDS-Transforms 0.2 ETLs (eICU, SICdb, HiRID, AmsterdamUMCdb) ship no lock file: their dependencies are
  installed exactly from the lock file (``uv pip sync``, from ``ICUTWIN_PYPI_INDEX``, by default PyPI; set it to
  a mirror where pypi.org is not reachable, as from the study's WSL network), then the ETL itself without
  dependencies.

A run fails when its installed environment differs from the lock file.

Inputs are the staged raw subsets (``data/raw/<mode>``, native layouts; MIMIC also needs the mimic-code v2.4.0
concept maps its spec pins by checksum); outputs go to ``data/community_meds/<mode>/<dataset>``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from icutwin.clif.conform import connect
from icutwin.clif.external import git_commit

NATIVE = os.name != "nt"   # a Linux host runs the scripts with its own bash; Windows runs them in WSL2
WSL_DISTRO = "Ubuntu-24.04"
HOME_CACHE = os.environ.get("ICUTWIN_COMMUNITY_CACHE", "$HOME/.cache/icutwin")   # expanded by bash
MIRROR = os.environ.get("ICUTWIN_PYPI_INDEX") or "https://pypi.org/simple"
LOCKS = Path(__file__).parent / "community_envs"   # <repo>.txt: exact environment of each validated ETL


@dataclass(frozen=True)
class Etl:
    repo: str
    runner: str            # "extract" (MEDS-Extract 0.7, uv.lock) | "transforms" (MEDS-Transforms 0.2)
    spec: str = ""         # MEDS-Extract dataset spec name
    command: str = ""      # the ETL's own entry point (MEDS-Transforms ETLs; INSPIRE's pre-MEDS wrapper)
    cohort: str = "MEDS_cohort"   # output sub-directory holding data/ and metadata/
    pins: tuple[str, ...] = ()    # versions decided against the ETL's declared range (deviation; see ETLS)


ETLS = {
    "mimic": Etl("MIMIC_IV_MEDS", "extract", spec="MIMIC-IV", cohort=""),
    "inspire": Etl("INSPIRE_MEDS", "extract", command="MEDS_extract-INSPIRE"),
    "nwicu": Etl("NWICU_MEDS", "extract", spec="NWICU", cohort=""),
    "eicu": Etl("eICU_MEDS", "transforms", command="MEDS_extract-eICU"),
    "sicdb": Etl("SICdb_MEDS", "transforms", command="MEDS_extract-SICdb"),
    # HiRID_MEDS allows polars<1.30; 1.29.0 panics in parquet predicate pushdown while sharding events
    # (polars-io predicates.rs unwrap on None), 1.26.0 (the version eICU_MEDS pins) does not.
    "hirid": Etl("HiRID_MEDS", "transforms", command="MEDS_extract-HIRID", pins=("polars==1.26.0",)),
    # AUMCdb_MEDS declares meds-transforms>=0.1, which now resolves to 0.7 (no MEDS_transforms.utils.get_shard_prefix);
    # 0.2.4 is the release contemporary with the pinned commit (2025-03-05) and the one the other 0.2-era ETLs pin.
    "aumc": Etl("AUMCdb_MEDS", "transforms", command="MEDS_extract-AUMCdb", pins=("meds-transforms==0.2.4",)),
}


def lock_file(etl: Etl) -> Path:
    """The ETL's lock file; its deliberate pins must be in it."""
    path = LOCKS / f"{etl.repo}.txt"
    locked = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")}
    missing = [p for p in etl.pins if p not in locked]
    if missing:
        raise ValueError(f"{path.name} does not hold the pins {missing}")
    return path


def wsl_path(p: Path) -> str:
    """``D:\\project\\x`` -> ``/mnt/d/project/x``."""
    p = p.resolve()
    return f"/mnt/{p.drive[0].lower()}{p.as_posix()[2:]}"


def shell_path(p: Path) -> str:
    """A path as the scripts' bash sees it: itself on a Linux host, its WSL mount on Windows."""
    return p.resolve().as_posix() if NATIVE else wsl_path(p)


def run_script(script: str, script_path: Path, log_path: Path, distro: str = WSL_DISTRO) -> int:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_bytes(script.encode("utf-8"))   # LF only: bash rejects CR
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if NATIVE:
        cmd, env = ["bash", shell_path(script_path)], dict(os.environ)
    else:
        cmd, env = ["wsl", "-d", distro, "--", "bash", wsl_path(script_path)], dict(os.environ, MSYS_NO_PATHCONV="1")
    with log_path.open("wb") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def stage_mimic(raw: Path, stage: Path, cache: Path, tool_dir: Path, proxy: str | None) -> None:
    """The raw subset plus the concept maps the MIMIC spec pins (hard links; maps verified by sha256)."""
    spec = yaml.safe_load((tool_dir / "src" / "MIMIC_IV_MEDS" / "event_configs.yaml").read_text(encoding="utf-8"))
    cache.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": proxy, "http": proxy} if proxy else {}))
    for src in spec["sources"]["common"]:
        for u in src["urls"]:
            f = cache / u["rel_path"]
            if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != u["sha256"]:
                f.write_bytes(opener.open(u["url"], timeout=60).read())
            if hashlib.sha256(f.read_bytes()).hexdigest() != u["sha256"]:
                raise ValueError(f"checksum mismatch: {u['rel_path']}")
    for sub in ("hosp", "icu"):
        (stage / sub).mkdir(parents=True, exist_ok=True)
        for f in (raw / sub).glob("*.csv.gz"):
            _link(f, stage / sub / f.name)
    for f in cache.glob("*.csv"):
        _link(f, stage / f.name)


# HiRID 1.1.1 ships each raw table as CSV and as typed parquet; HiRID_MEDS reads the parquet variant (it casts
# ``datetime`` to a timestamp, which a CSV string column does not allow). Ids stay int64, the type the ETL's
# general-table join keys get.
HIRID_PARQUET = {
    ("raw_stage/observation_tables", "raw_stage/observation_tables"): {
        "datetime": "TIMESTAMP", "entertime": "TIMESTAMP", "patientid": "BIGINT", "status": "BIGINT",
        "stringvalue": "VARCHAR", "type": "VARCHAR", "value": "DOUBLE", "variableid": "BIGINT"},
    ("raw_stage/pharma_records", "raw_stage/pharma_records_parquet"): {
        "patientid": "BIGINT", "pharmaid": "BIGINT", "givenat": "TIMESTAMP", "enteredentryat": "TIMESTAMP",
        "givendose": "DOUBLE", "cumulativedose": "DOUBLE", "fluidamount_calc": "DOUBLE",
        "cumulfluidamount_calc": "DOUBLE", "doseunit": "VARCHAR", "route": "VARCHAR", "infusionid": "BIGINT",
        "typeid": "BIGINT", "subtypeid": "DOUBLE", "recordstatus": "BIGINT"},
}


def stage_hirid(raw: Path, stage: Path) -> None:
    """A fresh input tree in the layout HiRID_MEDS keys its table processors on (rebuilt per run, since the ETL
    unpacks archives inside its input directory):

    * ``raw_stage/observation_tables/parquet`` and ``raw_stage/pharma_records_parquet/parquet``: the subset's CSV
      parts as the release's typed parquet variant (``HIRID_PARQUET``), under the processor names of the
      PhysioNet archives; header-only parts (a 2 % subset leaves some of the 250 empty) are skipped;
    * ``reference_data.tar.gz`` rebuilt from the subset's ``reference/`` tables, packed as the release ships it:
      the ETL unpacks it for the patient table; an unpacked ``reference_data/general_table.csv`` would instead be
      read as a data table whose ``admissiontime`` (a datetime) is taken for a seconds offset.
    """
    if stage.exists():
        shutil.rmtree(stage)
    con = connect()
    for (sub, dst_sub), types in HIRID_PARQUET.items():
        out = stage / dst_sub / "parquet"
        out.mkdir(parents=True, exist_ok=True)
        cols = ", ".join(f"'{c}': '{t}'" for c, t in types.items())
        for f in sorted((raw / sub / "csv").glob("*.csv")):
            with f.open("rb") as fh:
                fh.readline()
                if not fh.readline():
                    continue
            src, dst = f.as_posix(), (out / f"{f.stem}.parquet").as_posix()
            # RFC 4180 quoting stated, not sniffed: a quoted comma ("i.ventr, EVD") can first appear after the
            # sniffer's sample, which then splits it
            con.execute(f"""COPY (SELECT * FROM read_csv('{src}', header = true, quote = '"', escape = '"',
                                                         columns = {{{cols}}}))
                            TO '{dst}' (FORMAT parquet)""")
    con.close()
    with tarfile.open(stage / "reference_data.tar.gz", "w:gz") as tar:
        for f in sorted((raw / "reference").glob("*.csv")):
            tar.add(f, arcname=f.name)


def _link(src: Path, dst: Path) -> None:
    if not dst.exists():
        os.link(src, dst)


def script(name: str, etl: Etl, tool_dir: Path, input_dir: Path, out_dir: Path) -> str:
    venv = f"{HOME_CACHE}/venvs/{etl.repo}"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail",
             f'export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="{HOME_CACHE}/uv" '
             'UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-only-system}"',
             f'VENV="{venv}" LOCK="{shell_path(lock_file(etl))}"']
    if etl.runner == "extract":
        lines.append(f'UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen --project "{shell_path(tool_dir)}"')
    else:   # the locked dependencies exactly, then the ETL itself
        lines += ['[ -x "$VENV/bin/python" ] || uv venv "$VENV" --python 3.12',
                  f'uv pip sync --python "$VENV/bin/python" --index-url {MIRROR} "$LOCK"',
                  f'uv pip install --python "$VENV/bin/python" --index-url {MIRROR} --no-deps "{shell_path(tool_dir)}"']
    lines += ['export PATH="$VENV/bin:$PATH" PYTHONIOENCODING=utf-8', f'mkdir -p "{shell_path(out_dir)}"',
              f'uv pip freeze --python "$VENV/bin/python" > "{shell_path(out_dir)}/_environment.txt"',
              # the run fails when the installed environment is not the locked one
              f"""diff <(grep -v '^#' "$LOCK" | LC_ALL=C sort) """
              f"""<(grep -v -e '^-e ' -e ' @ file://' "{shell_path(out_dir)}/_environment.txt" | LC_ALL=C sort)"""]
    if etl.runner == "extract" and etl.command:   # wrapper: stages a derived side table, then meds-extract-run
        lines.append(f'{etl.command} "{shell_path(out_dir)}" --raw-input-dir "{shell_path(input_dir)}" --do-download false')
    elif etl.runner == "extract":
        lines.append(f'meds-extract-run spec={etl.spec} output_dir="{shell_path(out_dir)}" do_download=false '
                     f'input_dir="{shell_path(input_dir)}"')
    elif name == "aumc":
        lines.append(f'cd "{shell_path(tool_dir)}" && {etl.command} input_dir="{shell_path(input_dir)}" '
                     f'output_dir="{shell_path(out_dir)}" do_download=False do_overwrite=True')
    else:
        lines.append(f'{etl.command} root_output_dir="{shell_path(out_dir)}" raw_input_dir="{shell_path(input_dir)}" '
                     f'do_download=False do_overwrite=True')
    return "\n".join(lines) + "\n"


def run(name: str, refs: Path, raw_root: Path, out_root: Path, work_root: Path, cache_root: Path,
        proxy: str | None = None, distro: str = WSL_DISTRO) -> dict:
    """Run the community ETL of ``name`` on its raw subset; returns the provenance record."""
    etl = ETLS[name]
    tool_dir, out_dir = refs / etl.repo, out_root / name
    input_dir = raw_root / name
    if name == "mimic":
        input_dir = work_root / "community" / "mimic_input"
        stage_mimic(raw_root / name, input_dir, cache_root / "community" / "mimic_concept_maps", tool_dir, proxy)
    elif name == "hirid":
        input_dir = work_root / "community" / "hirid_input"
        stage_hirid(raw_root / name, input_dir)
    if out_dir.exists():   # MEDS-Extract skips stages whose outputs exist: every run starts empty
        shutil.rmtree(out_dir)
    started, t0 = datetime.now(UTC).isoformat(timespec="seconds"), time.time()
    code = run_script(script(name, etl, tool_dir, input_dir, out_dir), work_root / "community" / f"{name}.sh",
                      out_root / f"{name}.log", distro)
    lock = lock_file(etl)
    record = {"etl": etl.repo, "commit": git_commit(tool_dir), "runner": etl.runner,
              "host": "linux" if NATIVE else f"wsl:{distro}", "package_index": MIRROR,
              "lock": lock.name, "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(), "pins": list(etl.pins),
              "input": str(input_dir),
              "cohort_dir": str(out_dir / etl.cohort) if etl.cohort else str(out_dir), "started_at": started,
              "seconds": round(time.time() - t0, 1), "exit_code": code}
    (out_root / f"{name}_provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    if code:
        raise RuntimeError(f"{etl.repo} failed (exit {code}); log: {out_root / f'{name}.log'}")
    return record
