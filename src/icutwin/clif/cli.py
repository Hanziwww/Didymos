"""``icutwin_clif``: build, harmonize, validate and convert CLIF per dataset (S2 / S2b / S3 / S4).

Tasks: ``mimic_tool`` / ``aumc_tool`` (reference CLIF ETLs), ``supplement`` (what they do not build),
``adapter`` (our raw -> CLIF adapters), ``harmonize`` (canonical CLIF 3.0), ``validate`` (CLIFpy gate),
``meds_tool`` (CLIF_MEDS, ELF codes), ``pipeline``, which runs every step in order for each dataset, and
``qa`` (S3 cross-dataset distribution and coverage report over all datasets). S3 reconciliation:
``community`` runs the pinned community MEDS ETLs (WSL2) on the raw subsets, ``reconcile`` pairs their events
with our CLIF rows (``reconciliation.{json,md}`` in ``qa_root``) and fails on rows no documented reason explains.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from icutwin.clif import community, reconcile
from icutwin.clif.adapters import ADAPTERS
from icutwin.clif.bridges import BUILDERS
from icutwin.clif.external import CLIF_MEDS_DOMAINS, run_clif_aumc, run_clif_meds, run_clif_mimic
from icutwin.clif.harmonize import harmonize_dir
from icutwin.clif.qa import qa_report
from icutwin.clif.supplements import SUPPLEMENTS
from icutwin.clif.validate import validate_canonical_dir

log = logging.getLogger(__name__)

REFERENCE_TOOLS = {"mimic": "mimic_tool", "aumc": "aumc_tool"}
PIPELINE = ("build", "harmonize", "validate", "meds_tool")


def _tool(cfg: DictConfig, name: str) -> None:
    refs, cache, patches = Path(cfg.paths.refs), Path(cfg.paths.root) / ".cache", Path(cfg.paths.root) / "patches"
    raw_root, clif_root, work_root = Path(cfg.raw_root), Path(cfg.clif_root), Path(cfg.work_root)
    if name == "mimic":
        rec = run_clif_mimic(refs / "CLIF-MIMIC", raw_root / "mimic", work_root / "mimic", clif_root / "mimic",
                             cache, run_name=f"icutwin-{cfg.mode}", patch_dir=patches / "CLIF-MIMIC")
    else:
        rec = run_clif_aumc(refs / "CLIF-AmsterdamUMCdb", raw_root / "aumc", clif_root / "aumc", cache,
                            patch_dir=patches / "CLIF-AmsterdamUMCdb")
    log.info("%s reference tool done in %ss: %s", name, rec["seconds"], rec["tables"])


def _supplement(cfg: DictConfig, name: str) -> None:
    sup = SUPPLEMENTS[name](Path(cfg.parquet_root) / name, Path(cfg.clif_root) / name, Path(cfg.paths.tmp) / "duckdb")
    report = sup.run()
    log.info("%s supplement: %s", name, {k: v["rows"] for k, v in report["tables"].items()})


def _adapter(cfg: DictConfig, name: str) -> None:
    adapter = ADAPTERS[name](Path(cfg.parquet_root) / name, Path(cfg.clif_root) / name, Path(cfg.paths.tmp) / "duckdb")
    report = adapter.run()
    log.info("%s adapter: %s", name, {k: v["rows"] for k, v in report["tables"].items()})


def _harmonize(cfg: DictConfig, name: str) -> None:
    report = harmonize_dir(Path(cfg.clif_root) / name, Path(cfg.harmonized_root) / name, Path(cfg.paths.tmp) / "duckdb",
                           strict=bool(cfg.strict), utc_offset_hours=(cfg.utc_offset_hours or {}).get(name))
    log.info("%s harmonized: %s", name, {k: v.get("rows") for k, v in report["tables"].items()})


def _validate(cfg: DictConfig, name: str) -> None:
    report = validate_canonical_dir(Path(cfg.harmonized_root) / name, Path(cfg.paths.tmp) / "clifpy" / name)
    log.info("%s: passed=%s hard_errors=%d flagged tables=%s", name, report["passed"], report["n_hard_errors"],
             sorted(report["flags"]))
    if not report["passed"]:
        raise ValueError(f"{name}: CLIF 3.0 validation failed; see {cfg.harmonized_root}/{name}/_clifpy_validation.json")


def _meds_tool(cfg: DictConfig, name: str) -> None:
    refs, cache, patches = Path(cfg.paths.refs), Path(cfg.paths.root) / ".cache", Path(cfg.paths.root) / "patches"
    # ECMO_MCS reads the CLIF 2.1 wide table; canonical CLIF 3.0 `mcs` is converted by icutwin.meds instead
    domains = tuple(d for d in CLIF_MEDS_DOMAINS if d != "ECMO_MCS")
    rec = run_clif_meds(refs / "CLIF_MEDS", Path(cfg.harmonized_root) / name, Path(cfg.meds_root) / name,
                        cache, patches / "CLIF_MEDS", site=name, domains=domains)
    log.info("CLIF_MEDS %s done in %ss: %s", name, rec["seconds"], rec["data_files"])


def _build(cfg: DictConfig, name: str) -> None:
    """Raw -> CLIF: the reference tool plus our supplement, or our adapter."""
    if name in REFERENCE_TOOLS:
        _tool(cfg, name)
        _supplement(cfg, name)
    else:
        _adapter(cfg, name)


STEPS = {"mimic_tool": _tool, "aumc_tool": _tool, "supplement": _supplement, "adapter": _adapter, "build": _build,
         "harmonize": _harmonize, "validate": _validate, "meds_tool": _meds_tool}


def _qa(cfg: DictConfig) -> None:
    report = qa_report(Path(cfg.harmonized_root), list(cfg.datasets), Path(cfg.qa_root))
    log.info("S3 QA: %d flags (%s); report in %s", len(report["flags"]),
             sorted({f["flag"] for f in report["flags"]}), cfg.qa_root)


def _community(cfg: DictConfig) -> None:
    for name in cfg.datasets:
        rec = community.run(name, Path(cfg.paths.refs), Path(cfg.raw_root), Path(cfg.community_root),
                            Path(cfg.work_root), Path(cfg.paths.root) / ".cache", proxy=cfg.community_proxy,
                            distro=cfg.community_wsl_distro)
        log.info("%s: %s at %s done in %ss", name, rec["etl"], rec["commit"][:8], rec["seconds"])


def _reconcile(cfg: DictConfig) -> None:
    reports, open_rows = [], {}
    for name in cfg.datasets:
        db = BUILDERS[name](Path(cfg.parquet_root) / name, Path(cfg.paths.refs),
                            Path(__file__).parent / "maps" / name)
        rep = reconcile.reconcile(name, Path(cfg.community_root) / db.community, Path(cfg.clif_root) / name,
                                  db.bridges, utc_offset_hours=(cfg.utc_offset_hours or {}).get(name),
                                  setup=db.setup, scope=db.scope, work_dir=Path(cfg.work_root) / "reconcile" / name,
                                  streams=list(db.streams))
        reports.append(rep)
        for table, m in rep["tables"].items():
            if m.get("only_community_unexplained") or m.get("only_ours_unexplained"):
                open_rows[f"{name}.{table}"] = (m["only_community_unexplained"], m["only_ours_unexplained"])
        for stream, m in rep.get("streams", {}).items():
            if m.get("unexplained"):
                open_rows[f"{name}.{stream}"] = (m["unexplained"], "keys")
    reconcile.write(reports, Path(cfg.qa_root))
    if open_rows:
        raise ValueError(f"rows no documented reason explains (community, ours): {open_rows}; "
                         f"see {cfg.qa_root}/{reconcile.REPORT}.md")
    log.info("reconciliation: every row paired or explained; report in %s", cfg.qa_root)


def run(cfg: DictConfig) -> None:
    """Run ``cfg.task`` for every dataset in ``cfg.datasets`` (``pipeline``: all steps, one dataset at a time)."""
    if cfg.task in ("qa", "community", "reconcile"):
        {"qa": _qa, "community": _community, "reconcile": _reconcile}[cfg.task](cfg)
        return
    if cfg.task != "pipeline" and cfg.task not in STEPS:
        raise ValueError(f"unknown task {cfg.task!r}")
    for name in cfg.datasets:
        if cfg.task == "pipeline":
            for step in PIPELINE:
                STEPS[step](cfg, name)
        elif cfg.task not in REFERENCE_TOOLS.values() or REFERENCE_TOOLS.get(name) == cfg.task:
            STEPS[cfg.task](cfg, name)


@hydra.main(version_base="1.3", config_path="../configs", config_name="clif")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
