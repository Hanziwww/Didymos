"""Bounded smoke test of the training path a run will use (M5 preflight; on the server before the first run).

Run it with the run's own launcher and ``train:`` config, e.g.
``accelerate launch --config_file src/icutwin/configs/accelerate/ddp.yaml -m icutwin.model.cli task=smoke mode=full``.
It trains ``steps`` optimizer steps (default 50) with a checkpoint at ``resume_at`` (25), resumes a second run
from that checkpoint to the end, and checks:

* token-normalized gradient (``gradient_check``): on every process, micro-batches of different sizes accumulate
  to the gradient of the mean over all processes' target tokens, through the backend's own all-reduce;
* token-normalized loss: the first step's logged loss equals the mean token loss recomputed from the logits;
* learning rate: every step's rate is the warmup-cosine schedule's;
* data partitioning: every process draws from its own shard with its own seed; no window is drawn by two;
* resume: the resumed run draws exactly the uninterrupted run's windows, and its losses and final NLL agree
  within ``resume_tolerance`` (GPU kernels are not bitwise deterministic; on CPU they agree exactly);
* checkpoint: the safetensors checkpoint loads as a plain HF model and gives the trained model's NLL;
* memory: peak reserved memory on every GPU against its capacity (``memory_limit``);
* throughput: tokens per second after the first steps, and the projected time of the configured run;
* round 3's options, when configured: the EMA resumes (the resumed run's average equals the uninterrupted one's);
  the input noise replaces tokens once its rate is above 0. Dropout is off for the loss check's one step.
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from pathlib import Path

from icutwin.model.data import IGNORE, layout, load_samples
from icutwin.model.presets import qwen3_config
from icutwin.model.train import TrainConfig, accumulate_gradients, evaluate, train
from icutwin.tokenization.vocab import Vocab


def _batch(rows):
    import torch

    m = max(len(r) for r in rows)
    ids = torch.zeros((len(rows), m), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, r in enumerate(rows):
        ids[i, :len(r)], mask[i, :len(r)] = r, 1
    labels = torch.where(mask == 1, ids, torch.full_like(ids, IGNORE))
    labels[:, 0] = IGNORE
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def gradient_check(acc, tolerance: float | None = None) -> dict:
    """``acc``: an Accelerator with two accumulation steps and no mixed precision. Every process gets two
    micro-batches of its own sizes; the gradient ``accumulate_gradients`` leaves (after the synchronizing backward
    pass) must equal the gradient of the mean over all processes' target tokens, computed on one unwrapped copy
    of the model (float32, no TF32)."""
    import torch
    from transformers import Qwen3ForCausalLM

    assert acc.gradient_accumulation_steps == 2
    rank, world = acc.process_index, acc.num_processes
    g = torch.Generator().manual_seed(0)
    rows = {k: [[torch.randint(3, 300, (int(torch.randint(5, 60, (1,), generator=g)),), generator=g)
                 for _ in range(1 + (k + j) % 3)] for j in range(2)] for k in range(world)}
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(qwen3_config("tiny", 300, 0, 1, 2))
        ref = copy.deepcopy(model).to(acc.device)
        model = acc.prepare(model)
        loss = float(accumulate_gradients(acc, model, [_batch(mb) for mb in rows[rank]]))
        got = [p.grad.detach().float().cpu() for p in acc.unwrap_model(model, keep_torch_compile=False).parameters()]
        union = {k: t.to(acc.device) for k, t in _batch([r for k in range(world) for mb in rows[k] for r in mb]).items()}
        want_loss = ref(**union).loss
        want_loss.backward()
        want = [p.grad.detach().float().cpu() for p in ref.parameters()]
    finally:
        torch.set_float32_matmul_precision(precision)
    tol = tolerance or (1e-4 if acc.device.type == "cpu" else 1e-3)
    err = max(float((a - b).abs().max() / (b.abs().max() + 1e-12)) for a, b in zip(got, want))
    want_loss = float(want_loss.detach())
    return {"processes": world, "loss": loss, "full_batch_loss": want_loss, "max_rel_grad_error": err, "tolerance": tol,
            "passed": err < tol and abs(loss - want_loss) <= tol * abs(want_loss)}


def expected_lr(cfg: TrainConfig, step: int) -> float:
    """The learning rate after ``step`` optimizer steps: linear warmup, then cosine to ``min_lr_ratio`` of the
    peak (transformers' ``get_cosine_with_min_lr_schedule_with_warmup``)."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def smoke(cfg: TrainConfig, steps: int = 50, resume_at: int = 25, resume_tolerance: float | None = None,
          memory_limit: float = 0.9, skip_first: int = 5) -> dict:
    """The checks of the module docstring on ``cfg`` (the configured run), under ``cfg.run_dir/smoke``."""
    from accelerate import Accelerator, PartialState
    from accelerate.state import AcceleratorState, GradientState

    def fresh():   # every Accelerator below sets its own precision and accumulation
        AcceleratorState._reset_state(True)
        GradientState._reset_state()

    acc = Accelerator(mixed_precision="no", gradient_accumulation_steps=2)
    processes, rank, cpu = acc.num_processes, acc.process_index, acc.device.type == "cpu"
    checks: dict = {"token_normalized_gradient": gradient_check(acc)}
    acc.wait_for_everyone()
    del acc
    fresh()
    root = Path(cfg.run_dir) / "smoke"
    base = replace(cfg, run_dir=str(root / "uninterrupted"), max_steps=steps,
                   warmup_steps=max(1, min(cfg.warmup_steps, steps // 5)), eval_every=0, log_every=1,
                   checkpoint_every=resume_at, record_windows=steps, check_loss=True, resume_from=None,
                   eval_max_windows=cfg.eval_max_windows or 256)
    full = train(base)
    fresh()
    resumed = train(replace(base, run_dir=str(root / "resumed"), check_loss=False,
                            resume_from=str(root / "uninterrupted" / f"step_{resume_at}")))
    fresh()

    first = full["first_step"]
    checks["token_normalized_loss"] = {**first, "rel_error": abs(first["loss"] - first["recomputed"]) / first["recomputed"],
                                       "passed": abs(first["loss"] - first["recomputed"]) <= 1e-3 * first["recomputed"]}
    lr_err = max(abs(h["lr"] - expected_lr(base, h["step"])) / expected_lr(base, h["step"])
                 for run in (full, resumed) for h in run["history"] if "lr" in h)
    checks["learning_rate"] = {"warmup_steps": base.warmup_steps, "max_rel_error": lr_err, "passed": lr_err < 1e-6}

    pp = full["per_process"]
    drawn = [{tuple(w) for st in p["drawn"] for w in st} for p in pp]
    overlap = sum(len(drawn[a] & drawn[b]) for a in range(len(drawn)) for b in range(a + 1, len(drawn)))
    checks["data_partition"] = {
        "processes": processes, "seeds": [p["seed"] for p in pp], "shards": [p["shard"] for p in pp],
        "shard_windows": [p["shard_windows"] for p in pp], "windows_drawn": [len(d) for d in drawn],
        "drawn_by_two_processes": overlap,
        "passed": overlap == 0 and len({p["seed"] for p in pp}) == processes
        and sorted(tuple(p["shard"]) for p in pp) == [(k, processes) for k in range(processes)]}

    tol = resume_tolerance or (1e-4 if cpu else 5e-3)   # GPU: 4e-5 measured locally over 25 resumed steps
    same_windows = all(a["drawn"][resume_at:] == b["drawn"] for a, b in zip(full["per_process"], resumed["per_process"]))
    loss_full = {h["step"]: h["loss"] for h in full["history"] if "loss" in h}
    loss_resumed = {h["step"]: h["loss"] for h in resumed["history"] if "loss" in h}
    step_err = max(abs(loss_resumed[s] - loss_full[s]) / loss_full[s] for s in loss_resumed)
    nll_err = abs(resumed["final_eval"]["nll"] - full["final_eval"]["nll"]) / full["final_eval"]["nll"]
    checks["resume"] = {"same_windows": same_windows, "max_rel_step_loss_diff": step_err, "rel_final_nll_diff": nll_err,
                        "tolerance": tol, "passed": same_windows and step_err <= tol and nll_err <= tol / 2}

    files = [root / "uninterrupted" / f"step_{resume_at}" / "hf" / "model.safetensors",
             root / "uninterrupted" / "final" / "hf" / "model.safetensors",
             root / "uninterrupted" / "final" / "trainer_state.json"]
    ck = {"files": all(f.exists() for f in files)}
    state = PartialState(cpu=cpu)
    if rank == 0:                 # the inference-facing checkpoint, loaded as a plain model
        from transformers import Qwen3ForCausalLM

        v = Vocab.load(Path(cfg.tokens_root) / "vocab.json")
        model = Qwen3ForCausalLM.from_pretrained(str(root / "uninterrupted" / "final" / "hf")).to(state.device)
        ev = evaluate(model, load_samples(Path(cfg.tokens_root), list(cfg.datasets), cfg.eval_split, v), layout(v),
                      cfg.tokens_per_batch, state.device, base.eval_max_windows)
        ck.update(loaded_nll=ev["nll"], trained_nll=full["final_eval"]["nll"],
                  rel_diff=abs(ev["nll"] - full["final_eval"]["nll"]) / full["final_eval"]["nll"])
        ck["passed"] = ck["files"] and ck["rel_diff"] <= (1e-5 if cpu else 1e-3)
        del model
    state.wait_for_everyone()
    checks["checkpoint"] = ck
    if cfg.ema_decay:            # the average is part of the run's state: a resume must continue it exactly
        from safetensors.torch import load_file

        def ema_weights(run_root: Path) -> dict:
            final = run_root / "final"
            return load_file(str(final / ("hf" if (final / "hf_raw").exists() else "hf_ema") / "model.safetensors"))

        a, b = ema_weights(root / "uninterrupted"), ema_weights(root / "resumed")
        err = max(float((a[k].float() - b[k].float()).abs().max() / (a[k].float().abs().max() + 1e-12)) for k in a)
        checks["ema_resume"] = {"max_rel_diff": err, "tolerance": tol, "passed": set(a) == set(b) and err <= tol}
    if cfg.input_noise > 0:
        shares = [(h["noise_q"], h["noise_share"]) for h in full["history"] if "noise_q" in h]
        checks["input_noise"] = {"last_rate": shares[-1][0] if shares else None,
                                 "last_share_replaced": shares[-1][1] if shares else None,
                                 "passed": any(q > 0 and s > 0 for q, s in shares)}

    if not cpu:
        frac = [p["peak_gpu_reserved_gb"] / p["gpu_total_gb"] for p in pp]
        checks["memory"] = {"peak_allocated_gb": [p["peak_gpu_gb"] for p in pp],
                            "peak_reserved_gb": [p["peak_gpu_reserved_gb"] for p in pp],
                            "gpu_total_gb": [p["gpu_total_gb"] for p in pp], "max_reserved_fraction": max(frac),
                            "limit": memory_limit, "passed": max(frac) <= memory_limit}
    hist = [h for h in full["history"] if "loss" in h]
    a, b = hist[min(skip_first, len(hist) - 2)], hist[-1]
    tps = (b["tokens"] - a["tokens"]) / (b["seconds"] - a["seconds"])
    planned = cfg.max_steps * cfg.tokens_per_batch * cfg.grad_accum * processes   # padded budget: an upper bound
    checks["throughput"] = {"tokens_per_s": tps, "tokens_per_s_per_process": tps / processes,
                            "tokens_per_step": (b["tokens"] - a["tokens"]) / (b["step"] - a["step"]),
                            "configured_run_tokens_at_most": planned, "configured_run_hours_at_most": planned / tps / 3600,
                            "passed": tps > 0}
    passed = all(c.get("passed", True) for c in checks.values())
    return {"passed": passed, "processes": processes, "process_index": rank, "device": "cpu" if cpu else "cuda",
            "steps": steps,
            "resume_at": resume_at, "checks": checks,
            "losses": {"uninterrupted": [loss_full[s] for s in sorted(loss_full)],
                       "resumed": [loss_resumed[s] for s in sorted(loss_resumed)]}}
