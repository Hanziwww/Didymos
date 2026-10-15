"""icutwin on the AutoDL host: freeze a trained candidate (run name, token root): sha256 of its weights and HF config, of
the token snapshot it was trained and evaluated on (every shard and the vocabulary), the code commit, then the
checkpoint made read-only. Writes <run>/FROZEN.json.
    python freeze_run.py <run name> <token root> "<instruction>"
"""
import hashlib
import json
import os
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

R = Path("/root/autodl-tmp/icutwin")
name, tokens, instruction = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
RUN = R / "runs/train/full" / name


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


files = {str(p.relative_to(RUN)): sha256(p) for p in sorted((RUN / "final").rglob("*")) if p.is_file()}
snapshot = {str(p.relative_to(tokens)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(tokens.rglob("*")) if p.is_file() and p.suffix in (".parquet", ".json")}
out = {
    "model": name, "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"), "instruction": instruction,
    "code_commit": subprocess.run(["git", "-C", str(R), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "token_root": str(tokens),
    "train_config": json.loads((RUN / "config.json").read_text()),
    "checkpoint_files_sha256": files,
    "token_snapshot": snapshot,
}
(RUN / "FROZEN.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for p in [*(RUN / "final").rglob("*"), RUN / "final", RUN / "FROZEN.json", RUN / "config.json", RUN / "train_report.json",
          RUN / "metrics.jsonl"]:
    mode = os.stat(p).st_mode
    os.chmod(p, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
print(json.dumps({k: v for k, v in out.items() if k not in ("train_config", "token_snapshot")}, indent=1)[:3000])
