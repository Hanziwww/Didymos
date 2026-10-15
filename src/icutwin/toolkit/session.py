"""The toolkit's session (``docs/toolkit.md``): one model with its tokens, its privacy gate, the token caps and the
sampling backend; every tool takes a session.

* ``Paths``: every location from the repository's path configuration (``configs/paths.yaml``: the ``ICUTWIN_ROOT``,
  ``ICUTWIN_DATA`` and ``ICUTWIN_RUNS`` environment variables, else the repository), so no host path is written in
  the code. Outputs go to ``runs/toolkit`` (git-ignored, as all of ``runs/``).
* ``gate``: ``generate.producer.require_gate`` once per session (the M7 report must be PASS for these weights; it
  hashes the weights), with the report's scope (``not_in_v1``: what gate v1 did not assess, such as conditional and
  counterfactual sampling) recorded in every result's provenance. Sampling refuses without it.
* ``caps``: per dataset, the 95% quantile of the tokens in ``hours`` after an ICU time on a split's stays
  (``factory.oracle.token_caps``, the paper's caps rule), computed once and cached as JSON (an aggregate); a rollout
  that reaches its cap ends on the censored path.
* ``sample``: a job's sampling stage. ``backend="vllm"`` runs ``toolkit.stage`` in vLLM's environment as a subprocess
  (``vllm_python``, default ``ICUTWIN_VLLM_PYTHON`` or this interpreter; ``engines`` lanes started one after another,
  as vLLM sizes its cache from the free memory at start); ``backend="hf"`` samples in process with transformers and
  the grammar's reference processor (``backends.HfSampler``: tests and small jobs).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path

DATASETS = ("mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire")
REPO = Path(__file__).resolve().parents[3]


@dataclass
class Paths:
    """Where a session reads and writes (``default``: the ``ICUTWIN_*`` variables, else the repository's layout)."""
    root: Path
    tokens: Path
    model: Path
    audit: Path
    out: Path
    paper: Path
    bank: Path
    bank_store: Path
    stores: dict = field(default_factory=dict)      # split -> its real store (labels of real stays)

    @classmethod
    def default(cls, root=None, run: str = "MUC32_main", tokens: str = "full_r4") -> Paths:
        r = Path(root) if root else Path(os.environ.get("ICUTWIN_ROOT") or REPO)
        data = Path(os.environ.get("ICUTWIN_DATA") or r / "data")
        runs = Path(os.environ.get("ICUTWIN_RUNS") or r / "runs")
        st = data / "factory" / "stores"
        return cls(r, data / "tokens" / tokens, runs / "train" / "full" / run / "final" / "hf", runs / "audit" / f"m7_{run}",
                   runs / "toolkit", runs / "paper", data / "bank" / f"{run}_bank", st / "bank_dev",
                   {"held_out": st / "real_test", "tuning": st / "real_dev", "train": st / "real_train"})

    @classmethod
    def from_config(cls, cfg) -> Paths:
        """From the toolkit's hydra configuration (``configs/toolkit.yaml``)."""
        g = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))
        p = cls.default(g("root"), g("run", "MUC32_main"))
        for k in ("tokens", "model", "audit", "out", "paper", "bank", "bank_store"):
            if g(k):
                setattr(p, k, Path(str(g(k))))
        stores = g("stores")
        if stores:
            p.stores = {k: Path(str(x)) for k, x in dict(stores).items()}
        return p


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Session:
    """One model's toolkit session (module docstring)."""

    def __init__(self, paths: Paths | None = None, *, run: str = "MUC32_main", backend: str = "vllm",
                 vllm_python: str | None = None, gpu_memory_utilization: float = 0.85, engines: int = 1,
                 device: str = "cuda", workers: int = 2, seed: int = 0):
        if backend not in ("vllm", "hf"):
            raise ValueError(f"unknown backend {backend}")
        self.paths = paths or Paths.default(run=run)
        self.run, self.backend, self.device, self.workers, self.seed = run, backend, device, workers, seed
        self.vllm_python = vllm_python or os.environ.get("ICUTWIN_VLLM_PYTHON") or sys.executable
        self.gpu_memory_utilization, self.engines = float(gpu_memory_utilization), int(engines)

    @classmethod
    def from_config(cls, cfg) -> Session:
        g = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))
        return cls(Paths.from_config(cfg), run=g("run", "MUC32_main"), backend=g("backend", "vllm"),
                   vllm_python=g("vllm_python"), gpu_memory_utilization=float(g("gpu_memory_utilization", 0.85)),
                   engines=int(g("engines", 1)), device=g("device", "cuda"), workers=int(g("workers", 2)),
                   seed=int(g("seed", 0)))

    # -- the model and its vocabulary -----------------------------------------------------------------------------
    @cached_property
    def vocab(self):
        from icutwin.tokenization.vocab import Vocab

        return Vocab.load(self.paths.tokens / "vocab.json")

    @cached_property
    def tables(self):
        from icutwin.tokenization.fsm import compile_tables

        return compile_tables(self.vocab)

    @cached_property
    def context(self) -> int:
        cfg = json.loads((self.paths.model / "config.json").read_text(encoding="utf-8"))
        return int(cfg["max_position_embeddings"])

    @cached_property
    def gate(self) -> dict:
        """The privacy gate for these weights (raises ``generate.producer.GateNotPassed``) and its scope."""
        from icutwin.generate.producer import require_gate

        ok = require_gate(self.paths.audit, self.paths.model)
        rep = json.loads((self.paths.audit / "report.json").read_text(encoding="utf-8"))
        return {"gate": ok["gate"], "weights_sha256": ok["weights_sha256"], "amendment": rep.get("amendment"),
                "not_in_v1": rep.get("not_in_v1") or {}}

    def provenance(self, **extra) -> dict:
        """What a result was computed with (the gate only when this session checked it: sampling does)."""
        g = self.__dict__.get("gate")
        return {"run": self.run, "model": str(self.paths.model), "tokens": str(self.paths.tokens),
                "weights_sha256": g["weights_sha256"] if g else None, "gate": g["gate"] if g else None,
                "gate_amendment": g["amendment"] if g else None, "gate_not_in_v1": g["not_in_v1"] if g else None,
                "backend": self.backend, "git": git_sha(),
                "created": datetime.now(UTC).isoformat(timespec="seconds"), **extra}

    # -- inputs ---------------------------------------------------------------------------------------------------
    def patient(self, ref: str, unknown_site: bool | None = None):
        """A patient by reference: ``split:<dataset>/<split>/<stay id>`` (a tokenized stay), ``bank:<twin id>``,
        ``meds:<MEDS dir>::<split>::<stay id>`` (``toolkit.inputs``; prompted with its site unknown by default)."""
        from icutwin.toolkit import inputs

        return inputs.load(self, ref, unknown_site=unknown_site)

    def caps(self, hours: float, mult: float = 1.0, split: str = "held_out", q: float = 0.95) -> dict:
        """Token caps per dataset (module docstring), times ``mult`` and at most the context."""
        f = self.paths.out / "caps" / f"{self.paths.tokens.name}_{split}_{hours:g}h_q{q:g}.json"
        if f.exists():
            caps = json.loads(f.read_text(encoding="utf-8"))
        else:
            import numpy as np
            import pyarrow.parquet as pq

            from icutwin.factory import oracle

            rng = np.random.default_rng(0)
            seqs = {}
            for ds in DATASETS:
                p = self.paths.tokens / ds / f"{split}.parquet"
                if p.exists():
                    col = pq.read_table(p, columns=["ids"]).column(0)
                    pick = np.sort(rng.permutation(len(col))[:2000])
                    seqs[ds] = col.take(pick).to_pylist()
            caps = oracle.token_caps(seqs, self.vocab, hours, q, 0)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(caps, indent=1), encoding="utf-8", newline="\n")
        return {d: (min(int(c * mult), self.context - 1) if c else c) for d, c in caps.items()}

    def cap_for(self, caps: dict, dataset: str | None) -> int:
        """A dataset's cap; the largest one for a prompt of no known dataset (fewer censored rollouts)."""
        known = [c for c in caps.values() if c]
        return int(caps.get(dataset) or max(known))

    # -- passes and jobs ------------------------------------------------------------------------------------------
    @cached_property
    def teacher(self):
        from icutwin.toolkit.teacher import Teacher

        return Teacher(self.paths.model, self.vocab, self.paths.tokens, device=self.device)

    def job(self, tool: str, spec: dict, job_dir=None):
        from icutwin.toolkit.jobs import Job, job_id

        root = Path(job_dir) if job_dir else self.paths.out / tool / job_id(tool, spec)
        j = Job(root)
        if not j.spec_path.exists():
            j.write_spec({"tool": tool, **spec, "provenance": self.provenance()})
        return j

    def sample(self, job, batch_rollouts: int = 1024) -> dict:
        """A job's sampling stage (its ``requests.parquet``: rollouts, or ``samples.parquet``: new twins) through the
        backend; resumable. Refuses without the gate."""
        if job.done("sample"):
            return job.stage_info("sample")
        gate = self.gate
        t0 = time.time()
        if "teacher" in self.__dict__ and self.backend == "vllm":   # vLLM sizes its cache from the free memory
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        if self.backend == "hf":
            from icutwin.toolkit import stage
            from icutwin.toolkit.backends import HfSampler

            sampler = HfSampler(self.paths.model, self.paths.tokens / "vocab.json", device=self.device)
            info = stage.run(job, sampler, batch_rollouts=batch_rollouts)
        else:
            info = self._sample_vllm(job, batch_rollouts)
        info.update(gate=gate["gate"], seconds=round(time.time() - t0, 1), backend=self.backend)
        job.mark("sample", info)
        return info

    def _sample_vllm(self, job, batch_rollouts: int) -> dict:
        import icutwin

        src = str(Path(icutwin.__file__).resolve().parents[1])      # this checkout's code in vLLM's environment
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
               "PYTHONPATH": os.pathsep.join(x for x in (src, os.environ.get("PYTHONPATH")) if x),
               # the environment's tools first (FlashInfer's first engine builds its sampling kernels with ninja)
               "PATH": os.pathsep.join(x for x in (str(Path(self.vllm_python).parent), os.environ.get("PATH")) if x)}
        from icutwin.toolkit import stage

        lanes = max(1, min(self.engines, len(stage.batches(job.requests(), batch_rollouts))))   # no idle lane
        frac = self.gpu_memory_utilization / lanes
        go = job.root / "stage_go"
        go.unlink(missing_ok=True)
        with contextlib.ExitStack() as stack:
            procs = []
            for i in range(lanes):                    # one after another: each sizes its cache from the free memory
                log = job.root / f"stage_sample_{i}.log"
                cmd = [self.vllm_python, "-m", "icutwin.toolkit.stage", str(job.root), "--model",
                       str(self.paths.model), "--vocab", str(self.paths.tokens / "vocab.json"),
                       "--gpu-memory-utilization", f"{frac:.3f}", "--batch-rollouts", str(batch_rollouts),
                       "--shard", str(i), "--shards", str(lanes), "--go", str(go)]
                start = log.stat().st_size if log.exists() else 0      # a rerun appends to the lane's log
                fh = stack.enter_context(open(log, "a", encoding="utf-8"))
                p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
                procs.append(p)
                while p.poll() is None and b"engine_ready" not in log.read_bytes()[start:]:
                    time.sleep(2)
            go.touch()                                # every engine is up: the lanes draw
            codes = [p.wait() for p in procs]
        if any(codes):
            raise RuntimeError(f"the sampling stage failed (exit {codes}); see {job.root}/stage_sample_*.log")
        return {"engines": self.engines, "gpu_memory_utilization": frac}

    # -- the tools ------------------------------------------------------------------------------------------------
    def forecast(self, *a, **k):
        from icutwin.toolkit import forecast

        return forecast.run(self, *a, **k)

    def whatif(self, *a, **k):
        from icutwin.toolkit import whatif

        return whatif.run(self, *a, **k)

    def explain(self, *a, **k):
        from icutwin.toolkit import explain

        return explain.run(self, *a, **k)

    def similar(self, *a, **k):
        from icutwin.toolkit import similar

        return similar.run(self, *a, **k)

    def cohort(self, *a, **k):
        from icutwin.toolkit import cohort

        return cohort.run(self, *a, **k)

    def trial(self, *a, **k):
        from icutwin.toolkit import trial

        return trial.run(self, *a, **k)

    def teach(self, *a, **k):
        from icutwin.toolkit import teach

        return teach.run(self, *a, **k)
