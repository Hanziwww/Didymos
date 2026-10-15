"""``icutwin_stage``: stage raw archives into native layouts (S1)."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import hydra
import psutil
import pyarrow as pa
from omegaconf import DictConfig

from icutwin.stage.mirror import mirror_dataset
from icutwin.stage.spec import load_spec
from icutwin.stage.staging import stage_dataset

log = logging.getLogger(__name__)


def _run_one(name: str, cfg: dict) -> dict:
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [{name}] %(message)s", force=True)
    # Bound pyarrow's thread pools: CSV readahead scales with them, and several stagings run at once.
    pa.set_cpu_count(cfg["threads_per_job"])
    pa.set_io_thread_count(cfg["threads_per_job"])
    manifest = stage_dataset(
        load_spec(name),
        archive_root=Path(cfg["raw_archives"]),
        out_root=Path(cfg["out_root"]),
        refs_root=Path(cfg["refs"]),
        mode=cfg["mode"],
        frac=cfg["frac"],
        salt=cfg["salt"],
        compression=cfg["compression"],
        include=tuple(cfg["include"]),
    )
    peak = psutil.Process().memory_info()
    peak_gb = getattr(peak, "peak_wset", getattr(peak, "rss", 0)) / 2**30
    log.info("%s: peak memory %.2f GB", name, peak_gb)
    return {"dataset": name, "seconds": manifest["seconds"], "selection": manifest["selection"], "peak_gb": round(peak_gb, 2)}


@hydra.main(version_base="1.3", config_path="../configs", config_name="stage")
def main(cfg: DictConfig) -> None:
    job = {
        "raw_archives": cfg.paths.raw_archives,
        "out_root": cfg.out_root,
        "refs": cfg.paths.refs,
        "mode": cfg.mode,
        "frac": float(cfg.frac),
        "salt": cfg.salt,
        "compression": cfg.compression,
        "include": list(cfg.include),
        "threads_per_job": int(cfg.threads_per_job),
    }
    names = list(cfg.datasets)
    if cfg.task == "mirror":
        for name in names:
            m = mirror_dataset(
                native_dir=Path(cfg.out_root) / name,
                out_dir=Path(cfg.mirror_root) / name,
                temp_dir=Path(cfg.paths.tmp) / "duckdb",
                threads=int(cfg.duckdb.threads),
                memory_limit=str(cfg.duckdb.memory_limit),
                sample_size=int(cfg.duckdb.sample_size),
            )
            log.info("mirrored %s: %d tables in %ss", name, len(m["tables"]), m["seconds"])
        return
    if cfg.task != "stage":
        raise ValueError(f"unknown task {cfg.task!r}")
    if cfg.jobs > 1 and len(names) > 1:
        with ProcessPoolExecutor(max_workers=min(cfg.jobs, len(names))) as pool:
            results = list(pool.map(_run_one, names, [job] * len(names)))
    else:
        results = [_run_one(name, job) for name in names]
    for r in results:
        log.info("staged %s in %ss: %s", r["dataset"], r["seconds"], r["selection"])


if __name__ == "__main__":
    main()
