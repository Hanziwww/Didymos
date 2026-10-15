"""``icutwin_train``: S6 per the plan (M5).

Tasks:
* ``pilot``: the bounded local pilot (``icutwin.model.pilot``); ``overfit``: one batch, a tiny model.
* Server preflight, in order: ``resources`` (GPUs, kernels, disk), ``exposure`` (train tokens per dataset and the
  configured run's sampling exposure), ``parity`` (padding-free packing keeps windows apart under
  ``train.attn_implementation``), ``smoke`` (the configured run's path for 50 steps with save and resume,
  ``icutwin.model.smoke``; under the run's launcher).
* ``train``: a training run, on the server under
  ``accelerate launch --config_file configs/accelerate/<ddp|fsdp2>.yaml``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from icutwin.model.data import TransformConfig, exposure, split_sizes
from icutwin.model.pilot import attention_parity, overfit, resources, run
from icutwin.model.smoke import smoke
from icutwin.model.train import TrainConfig, train

log = logging.getLogger(__name__)


def _train_config(cfg: DictConfig) -> TrainConfig:
    t = OmegaConf.to_container(cfg.train, resolve=True)
    transforms = TransformConfig(**t.pop("transforms"))
    return TrainConfig(tokens_root=str(cfg.tokens_root), run_dir=str(Path(cfg.run_root) / t.pop("name")),
                       datasets=list(cfg.datasets), transforms=transforms, seed=int(cfg.seed), **t)


def _exposure(cfg: DictConfig) -> dict:
    tokens = Path(cfg.tokens_root)
    splits = {s: split_sizes(tokens, list(cfg.datasets), s) for s in ("train", "tuning", "held_out")}
    t = cfg.train
    planned = int(t.max_steps) * int(t.tokens_per_batch) * int(t.grad_accum) * int(cfg.exposure.gpus)
    budgets = sorted({float(planned), *(float(b) for b in cfg.exposure.budgets)})
    passes = dict(t.passes) if t.get("passes") else None
    out = {"splits": splits, "train_tokens": sum(d["tokens"] for d in splits["train"].values()),
           "alpha": float(t.alpha), "passes": passes, "configured_run_tokens_at_most": planned,
           "exposure": {f"{b:.3g}": exposure(splits["train"], float(t.alpha), b, passes) for b in budgets}}
    path = Path(cfg.qa_root) / "exposure.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def run_task(cfg: DictConfig) -> dict:
    tokens, datasets = Path(cfg.tokens_root), list(cfg.datasets)
    if cfg.task == "pilot":
        p = cfg.pilot
        runs = [dict(r) for r in OmegaConf.to_container(p.runs, resolve=True)]
        return run(tokens, Path(cfg.qa_root), Path(cfg.run_root) / "pilot", datasets, runs, int(p.steps),
                   int(p.tokens_per_batch), list(p.probes), float(p.gpu_memory_fraction), int(cfg.seed), int(p.n_check),
                   int(p.grad_accum))
    if cfg.task == "overfit":
        return overfit(tokens, datasets, int(cfg.overfit_steps), int(cfg.seed))
    if cfg.task == "parity":
        out = attention_parity(tokens, datasets, str(cfg.train.attn_implementation), seed=int(cfg.seed),
                               preset=str(cfg.train.preset))
        if not out["passed"]:
            raise RuntimeError(f"packed batches are not equivalent to separate windows: {out}")
        return out
    if cfg.task == "resources":
        return resources(Path(cfg.paths.data), Path(cfg.paths.runs))
    if cfg.task == "exposure":
        return _exposure(cfg)
    if cfg.task == "smoke":
        out = smoke(_train_config(cfg), int(cfg.smoke.steps), int(cfg.smoke.resume_at))
        if out["process_index"] == 0:
            path = Path(cfg.run_root) / cfg.train.name / "smoke" / "smoke_report.json"
            path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        if not out["passed"]:
            raise RuntimeError(f"smoke test failed: {json.dumps(out['checks'], default=str)[:3000]}")
        return out
    if cfg.task == "train":
        tc = _train_config(cfg)
        out = train(tc)
        if out["process_index"] == 0:
            (Path(tc.run_dir) / "train_report.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        return out
    raise ValueError(f"unknown task {cfg.task!r}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    out = run_task(cfg)
    log.info("%s: %s", cfg.task, json.dumps(out, default=str)[:4000])


if __name__ == "__main__":
    main()
