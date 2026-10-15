"""One process of the two-process data-parallel test (``test_model.test_data_parallel_smoke``): CPU, gloo.

Runs the smoke test (``icutwin.model.smoke``: token-normalized gradient through the all-reduce, first-step
loss, learning rate, data partitioning, save and resume, checkpoint) on the fixture shards and writes
``<out>/rank<k>.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from icutwin.model.smoke import smoke
from icutwin.model.train import TrainConfig


def main(tokens_root: str, out: str) -> None:
    cfg = TrainConfig(tokens_root=tokens_root, run_dir=str(Path(out) / "run"), datasets=["mimic"], preset="tiny",
                      max_steps=100, tokens_per_batch=512, grad_accum=2, warmup_steps=1, lr=1e-3)
    report = smoke(cfg, steps=6, resume_at=3)
    (Path(out) / f"rank{report['process_index']}.json").write_text(json.dumps(report, default=str), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
