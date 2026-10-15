"""``icutwin_meds``: S4 per dataset.

Tasks: ``build`` (our ELF extension domains, then one MEDS v0.4 dataset with the ICU stay as subject),
``parity`` (our ELF engine against the CLIF_MEDS run on its generic domains; fails on any difference),
``derived`` (SOFA, ASE, norepinephrine equivalents, episodes, KDIGO, fluid balance per ICU stay),
``mimic_code`` (MIMIC only: the derived concepts against mimic-code at its pinned commit; fails when our code on
mimic-code's own inputs does not reproduce its outputs) and ``weights`` (weight causality of the training MEDS,
``meds.weights``: fails when a per-kg medication value was converted with a weight charted later).
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig

from icutwin.meds.build import build_dataset
from icutwin.meds.derived import derive
from icutwin.meds.mimic_code import parity as mimic_code_parity
from icutwin.meds.parity import check
from icutwin.meds.weights import weight_causality

log = logging.getLogger(__name__)


def _build(cfg: DictConfig, name: str) -> None:
    report = build_dataset(name, Path(cfg.clif_root) / name, Path(cfg.elf_root) / name, Path(cfg.meds_root) / name,
                           pre_icu_hours=float(cfg.pre_icu_hours), subjects_per_shard=int(cfg.subjects_per_shard),
                           dataset_version=str((cfg.dataset_versions or {}).get(name, "")),
                           min_age_years=cfg.min_age_years)
    log.info("%s MEDS: %d subjects, %d events, splits %s, extension domains %s", name, report["subjects"],
             report["events"], {k: v["subjects"] for k, v in report["splits"].items()}, report["extension_domains"])


def _parity(cfg: DictConfig, name: str) -> None:
    Path(cfg.paths.tmp).mkdir(parents=True, exist_ok=True)   # e.g. a per-process ICUTWIN_TMP not made yet
    with tempfile.TemporaryDirectory(dir=Path(cfg.paths.tmp)) as tmp:
        report = check(Path(cfg.clif_root) / name, Path(cfg.elf_root) / name, Path(tmp))
    log.info("%s ELF parity with CLIF_MEDS: %s", name,
             {k: (v["clif_meds"], v["only_clif_meds"], v["only_icutwin"]) for k, v in report.items() if isinstance(v, dict)})
    if not report["identical"]:
        raise ValueError(f"{name}: ELF engine differs from CLIF_MEDS: {json.dumps(report)[:2000]}")


def _derived(cfg: DictConfig, name: str) -> None:
    summary = derive(Path(cfg.clif_root) / name, Path(cfg.derived_root) / name, Path(cfg.paths.tmp) / "derived" / name,
                     min_age_years=cfg.min_age_years)
    log.info("%s derived concepts: %s", name, summary)


def _mimic_code(cfg: DictConfig, name: str) -> None:
    if name != "mimic":
        return
    report = mimic_code_parity(Path(cfg.parquet_root) / name, Path(cfg.paths.refs) / "mimic-code", Path(cfg.clif_root) / name,
                               Path(cfg.derived_root) / name, Path(cfg.work_root) / "mimic_code.duckdb", Path(cfg.qa_root),
                               Path(cfg.work_root) / "sofa_decomposition")
    logic = report["logic"]
    kd = {c: v["share_stays_without_outside_urine"] for c, v in logic["kdigo"]["columns"].items()}
    log.info("mimic-code parity: KDIGO %s, ventilation identical %s, NEE identical %s; report in %s", kd,
             logic["ventilation"]["identical"], logic["nee"]["share"], cfg.qa_root)
    if not (logic["ventilation"]["identical"] and logic["nee"]["share"] == 1.0 and all(v == 1.0 for v in kd.values())):
        raise ValueError(f"our derived concepts do not reproduce mimic-code on its own inputs: {json.dumps(logic)[:2000]}")


def _weights(cfg: DictConfig, name: str) -> None:
    report = weight_causality(Path(cfg.clif_root) / name, Path(cfg.meds_root) / name)
    out = Path(cfg.qa_root) / "weight_causality"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("%s weight causality: %s", name, report)
    if report["violations"]:
        raise ValueError(f"{name}: per-kg medication values converted with a weight charted later: {report}")


STEPS = {"build": _build, "parity": _parity, "derived": _derived, "mimic_code": _mimic_code, "weights": _weights}


def run(cfg: DictConfig) -> None:
    if cfg.task not in STEPS:
        raise ValueError(f"unknown task {cfg.task!r}")
    for name in cfg.datasets:
        STEPS[cfg.task](cfg, name)


@hydra.main(version_base="1.3", config_path="../configs", config_name="meds")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
