"""Check that S2 is deterministic: run each dataset's S2 step (our adapter, or the supplement for datasets built by
a reference tool, which is not re-run) and the harmonize step twice and compare every CLIF table row for row.

Ties that a query resolves arbitrarily (``any_value``, ``arg_max`` on equal times, window orders with equal keys)
show up as rows that differ between the runs. Output locations come from the ``clif`` config; the last run's
tables stay in place, snapshots go to ``--work``. Exits 1 when a table differs.

    python -m icutwin.clif.tools.determinism --mode subset mimic hirid ...
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb
from hydra import compose, initialize_config_module

from icutwin.clif.cli import REFERENCE_TOOLS

DATASETS = ("mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire")


def _run(task: str, name: str, mode: str) -> None:
    subprocess.run([sys.executable, "-m", "icutwin.clif.cli", f"task={task}", f"datasets=[{name}]", f"mode={mode}"],
                   check=True, capture_output=True, text=True)


def _snapshot(src: Path, dst: Path) -> None:
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    for f in src.glob("clif_*.parquet"):
        shutil.copy2(f, dst / f.name)


def differing_rows(a: Path, b: Path) -> dict[str, tuple[int, int]]:
    """Tables of directory ``a`` whose rows differ from ``b``: (rows only in a, rows only in b)."""
    con, out = duckdb.connect(), {}
    for f in sorted(a.glob("clif_*.parquet")):
        p, q = f.as_posix(), (b / f.name).as_posix()
        x, y = (con.execute(f"SELECT count(*) FROM (SELECT * FROM '{u}' EXCEPT ALL SELECT * FROM '{v}')").fetchone()[0]
                for u, v in ((p, q), (q, p)))
        if x or y:
            out[f.name] = (x, y)
    return out


def check(name: str, mode: str, work: Path) -> dict[str, dict[str, tuple[int, int]]]:
    """Run the S2 step twice, then harmonize twice (both on the second S2 output); differing tables per task."""
    with initialize_config_module(config_module="icutwin.configs", version_base="1.3"):
        cfg = compose("clif", overrides=[f"mode={mode}"])
    step = "supplement" if name in REFERENCE_TOOLS else "adapter"
    found = {}
    for task, out in ((step, Path(cfg.clif_root) / name), ("harmonize", Path(cfg.harmonized_root) / name)):
        _run(task, name, mode)
        _snapshot(out, work / f"{name}_{task}")
        _run(task, name, mode)
        if diff := differing_rows(work / f"{name}_{task}", out):
            found[task] = diff
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--mode", default="subset")
    ap.add_argument("--work", type=Path, default=Path(".tmp/determinism"))
    args = ap.parse_args()
    failed = False
    for name in args.datasets:
        found = check(name, args.mode, args.work)
        for task, tables in found.items():
            for table, (x, y) in tables.items():
                print(f"{name} {task} {table}: {x} / {y} rows differ between runs")
        failed |= bool(found)
        print(f"{name}: {'NOT deterministic' if found else 'deterministic'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
