"""Bounded M5 pilot on the local GPU (the user's request; full-scale training stays on the server, M9).

1. Resources: GPU, RAM and disk before anything runs; the pilot caps its GPU memory (``gpu_memory_fraction``).
2. Data loading: every stored window of every split loads, fits the context, and every training transform of it
   (goal block, header order, dataset and header-field dropout) is accepted by the grammar automaton.
3. Loss masking: the labels of every checked sample equal the mask derived independently from the parser's
   structure (first token, goal block, a continuation window's pinned context).
4. Window-state continuity and input/target separation: the S5 validation report (every stay, both checks) and,
   in the training data, goal tokens only inside a masked goal block and never in evaluation inputs.
5. A tiny-model overfit test on one batch (the roadmap's sanity check).
   (``attention_parity``, ``task=parity``: padding-free packing keeps windows apart; FlashAttention-2 on the
   server, its SDPA mask locally.)
6. Memory and throughput per preset at full context (a few optimizer steps each, out-of-memory caught).
7. Short training runs of the pilot presets with validation NLL per dataset and token family (tuning split), at
   a few learning rates: in the standard parameterization the best rate falls with width, so sizes compare at
   their own best rate.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from icutwin.model.data import (IGNORE, TransformConfig, eval_items, flat_batches, labels_for, layout, load_samples,
                                pad_batches, training_stream, transform)
from icutwin.model.presets import CONTEXT, PRESETS, n_params
from icutwin.model.train import TrainConfig, build_model, train
from icutwin.tokenization.fsm import accepts, compile_tables
from icutwin.tokenization.parser import parse
from icutwin.tokenization.vocab import Vocab


def resources(data_root: Path, *paths: Path) -> dict:
    """GPUs, memory, disk and the training stack's versions and kernels (``task=resources`` on a server)."""
    import os
    from importlib.metadata import PackageNotFoundError, version

    import psutil
    import torch
    from transformers.utils import is_flash_attn_2_available

    def ver(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    out = {"ram_total_gb": psutil.virtual_memory().total / 2**30, "ram_available_gb": psutil.virtual_memory().available / 2**30,
           "disk_free_gb": shutil.disk_usage(data_root).free / 2**30, "cuda": torch.cuda.is_available(),
           "cpus": os.cpu_count(), "torch": torch.__version__, "cuda_version": torch.version.cuda,
           "transformers": ver("transformers"), "accelerate": ver("accelerate"), "flash_attn": ver("flash-attn"),
           "flash_attention_2_available": bool(is_flash_attn_2_available()),
           "nccl_available": bool(torch.distributed.is_available() and torch.distributed.is_nccl_available()),
           "disk_free_gb_by_path": {str(p): shutil.disk_usage(p).free / 2**30 for p in (data_root, *paths) if p.exists()}}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        out.update(gpu=torch.cuda.get_device_name(0), gpu_total_gb=total / 2**30, gpu_free_gb=free / 2**30,
                   gpus=[{"name": torch.cuda.get_device_name(k),
                          "total_gb": torch.cuda.get_device_properties(k).total_memory / 2**30,
                          "capability": ".".join(map(str, torch.cuda.get_device_capability(k)))}
                         for k in range(torch.cuda.device_count())])
    return out


def _expected_labels(ids: np.ndarray, v: Vocab) -> np.ndarray:
    """The loss mask from the parser's structure (independent of ``data.labels_for``)."""
    p = parse(ids.tolist(), v, whole=ids[-1] == v.id("[EOS]"))
    head = 1 + len(p.prefix) + len(p.header)
    goal = len(p.goal) + 2 if p.goal is not None else 0
    labels = ids.astype(np.int64).copy()
    labels[0] = IGNORE
    if p.cont:
        context = head + goal + 1 + (len(p.state) + 2 if p.state else 0) + 1   # ... [CONT] [STATE] ... [/STATE] [TL]
        labels[:context] = IGNORE
    elif goal:
        labels[head:head + goal] = IGNORE
    if p.clock:                       # clock blocks: their tokens by class in the vocabulary
        labels[np.array([v.meta[int(t)].get("cls") == "clock" for t in ids])] = IGNORE
    return labels


def check_data(tokens_root: Path, datasets: list[str], qa_root: Path, n_train: int, seed: int) -> dict:
    v = Vocab.load(tokens_root / "vocab.json")
    lay = layout(v)
    tables = compile_tables(v)
    max_tokens = CONTEXT
    out: dict = {"splits": {}, "transform_rejected": 0, "label_mismatch": 0, "goal_outside_block": 0,
                 "eval_inputs_with_goal": 0, "checked_samples": 0}
    samples = {split: load_samples(tokens_root, datasets, split, v) for split in ("train", "tuning", "held_out")}
    for split, ss in samples.items():
        lens = np.array([len(s.ids) for s in ss])
        out["splits"][split] = {"windows": len(ss), "continuation_windows": int(sum(s.continuation for s in ss)),
                                "tokens": int(lens.sum()), "max_window": int(lens.max()),
                                "over_context": int((lens > max_tokens).sum()),
                                "ids_out_of_range": int(sum(int(s.ids.max()) >= len(v.tokens) or int(s.ids.min()) < 0 for s in ss))}
    rng = np.random.default_rng(seed)
    cfg = TransformConfig()
    pick = rng.choice(len(samples["train"]), size=min(n_train, len(samples["train"])), replace=False)
    goal_tokens = {i for i, m in enumerate(v.meta) if m.get("cls") == "goal"}
    target_classes: dict = {}
    for s in [samples["train"][i] for i in pick] + samples["tuning"]:
        for k in range(3):   # a few random transforms of each window
            ids = transform(s, lay, rng, cfg)
            whole = ids[-1] == v.id("[EOS]")
            ok, _why = accepts(tables, ids.tolist(), whole=whole)
            out["transform_rejected"] += int(not ok)
            labels = labels_for(ids, lay)
            out["label_mismatch"] += int(not np.array_equal(labels, _expected_labels(ids, v)))
            inside = np.zeros(len(ids), bool)
            if lay.goal in ids:
                a, b = int(np.flatnonzero(ids == lay.goal)[0]), int(np.flatnonzero(ids == lay.goal_end)[0])
                inside[a:b + 1] = True
            out["goal_outside_block"] += int(any(t in goal_tokens and not inside[i] for i, t in enumerate(ids.tolist())))
            for t in ids[labels != IGNORE].tolist():
                c = v.meta[t].get("cls") or "special"
                target_classes[c] = target_classes.get(c, 0) + 1
            out["checked_samples"] += 1
    for ids, _labels, _ds in eval_items(samples["tuning"], lay):
        out["eval_inputs_with_goal"] += int(lay.goal in ids or any(t in goal_tokens for t in ids.tolist()))
    total = sum(target_classes.values())
    out["target_share_by_class"] = {c: round(n / total, 4) for c, n in sorted(target_classes.items())}
    rep = json.loads((qa_root / "tokenizer_report.json").read_text(encoding="utf-8"))["datasets"]
    out["tokenizer_validation"] = {ds: {k: r[k] for k in ("stays", "continuity_failed", "separation_failed",
                                                          "round_trip_failed")} for ds, r in rep.items()}
    out["passed"] = (out["transform_rejected"] == 0 and out["label_mismatch"] == 0 and out["goal_outside_block"] == 0
                     and out["eval_inputs_with_goal"] == 0
                     and all(r["continuity_failed"] == 0 and r["separation_failed"] == 0 for r in rep.values())
                     and all(x["over_context"] == 0 and x["ids_out_of_range"] == 0 for x in out["splits"].values()))
    return out


def overfit(tokens_root: Path, datasets: list[str], steps: int, seed: int) -> dict:
    """One fixed batch, a tiny model: the loss must fall to near zero (data, labels and model wiring)."""
    import torch

    v = Vocab.load(tokens_root / "vocab.json")
    lay = layout(v)
    torch.manual_seed(seed)
    stream = training_stream(load_samples(tokens_root, datasets, "tuning", v), lay, TransformConfig(), 0.5, seed)
    batch = next(pad_batches(stream, lay, 4096, bucket=8))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("tiny", v, "sdpa").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    inputs = {k: t.to(dev) for k, t in batch.items() if k != "meta"}
    losses = []
    for _ in range(steps):
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss = model(**inputs).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    return {"steps": steps, "batch_tokens": int(inputs["attention_mask"].sum()), "first_loss": losses[0],
            "last_loss": losses[-1], "passed": losses[-1] < 0.05 * losses[0]}


def probe(tokens_root: Path, datasets: list[str], preset: str, tokens_per_batch: int, steps: int,
          gpu_memory_fraction: float, seed: int) -> dict:
    """Peak memory and throughput of a few optimizer steps at full context (windows up to 8,192 tokens)."""
    import torch

    v = Vocab.load(tokens_root / "vocab.json")
    lay = layout(v)
    info = {"preset": preset, "params": n_params(preset, len(v.tokens)), "tokens_per_batch": tokens_per_batch}
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    samples = [s for s in load_samples(tokens_root, datasets, "train", v) if len(s.ids) > 7000]   # longest windows
    stream = training_stream(samples, lay, TransformConfig(), 0.5, seed)
    batches = pad_batches(stream, lay, tokens_per_batch, bucket=4)
    dev = torch.device("cuda")
    model = opt = None
    try:
        torch.manual_seed(seed)
        model = build_model(preset, v, "sdpa").to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        times, toks = [], 0
        for k in range(steps + 1):
            b = next(batches)
            inputs = {key: t.to(dev) for key, t in b.items() if key != "meta"}
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**inputs).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if k:   # the first step warms up kernels
                times.append(time.time() - t0)
                toks += int(inputs["attention_mask"].sum())
        info.update(ok=True, peak_gpu_gb=torch.cuda.max_memory_allocated() / 2**30,
                    tokens_per_s=toks / sum(times), seconds_per_step=float(np.mean(times)))
    except torch.OutOfMemoryError:
        info.update(ok=False, error="out of memory under the pilot's cap")
    finally:
        del model, opt
        torch.cuda.empty_cache()
    return info


def attention_parity(tokens_root: Path, datasets: list[str], attn: str, n_windows: int = 12, seed: int = 0,
                     device: str | None = None, split: str = "tuning", preset: str = "pilot") -> dict:
    """Padding-free packing check (run on the server before training, with the run's preset): the last hidden
    states of a few tuning windows, each run alone, must equal those of the same windows packed into one row with
    per-window position resets under ``attn``, i.e. no window attends to another. A negative control packs them
    without the resets and must fail the same check (a leak is detectable)."""
    import torch

    v = Vocab.load(tokens_root / "vocab.json")
    lay = layout(v)
    samples = load_samples(tokens_root, datasets, split, v)
    if len(samples) < 2:
        raise ValueError(f"the parity check packs at least two windows; {split} has {len(samples)}")
    pick = np.random.default_rng(seed).choice(len(samples), size=min(n_windows, len(samples)), replace=False)
    items = [(samples[i].ids[:2048].astype(np.int32), labels_for(samples[i].ids[:2048], lay), samples[i].dataset)
             for i in pick]
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    half = attn in ("flash_attention_2", "varlen")   # their kernels take bf16 only
    dtype = torch.bfloat16 if half else torch.float32
    torch.manual_seed(seed)
    ref = build_model(preset, v, "sdpa").to(dev, dtype).eval()
    model = build_model(preset, v, attn).to(dev, dtype).eval()
    model.load_state_dict(ref.state_dict())
    flat = next(flat_batches(iter(items), sum(len(x) for x, _, _ in items)))
    assert len(flat["meta"]) == len(items)
    ids = flat["input_ids"].to(dev)

    def last_hidden(m, **kw):
        return m(input_ids=ids, output_hidden_states=True, **kw).hidden_states[-1][0].float()

    with torch.no_grad():
        alone = torch.cat([ref(input_ids=torch.from_numpy(x.astype(np.int64))[None].to(dev),
                               output_hidden_states=True).hidden_states[-1][0].float() for x, _, _ in items])
        packed = last_hidden(model, position_ids=flat["position_ids"].to(dev),
                             **{k: (flat[k].to(dev) if hasattr(flat[k], "to") else flat[k])
                                for k in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")})
        control = last_hidden(model, position_ids=torch.arange(ids.shape[1], device=dev)[None])

    def rel(h):   # per-token relative error against the windows run alone
        return ((h - alone).norm(dim=-1) / alone.norm(dim=-1)).cpu().numpy()

    tol = 0.05 if half else 1e-3
    first = len(items[0][0])                      # the first window has nothing before it to leak from
    out = {"attn": attn, "preset": preset, "windows": len(items), "tokens": int(ids.shape[1]), "tolerance": tol,
           "packed_max_rel_error": float(rel(packed).max()),
           "control_max_rel_error_after_first_window": float(rel(control)[first:].max())}
    out["passed"] = out["packed_max_rel_error"] < tol and out["control_max_rel_error_after_first_window"] > 10 * tol
    return out


def run(tokens_root: Path, qa_root: Path, run_root: Path, datasets: list[str], runs: list[dict], steps: int,
        tokens_per_batch: int, probes: list[str], gpu_memory_fraction: float, seed: int, n_check: int,
        grad_accum: int = 1) -> dict:
    """The pilot stages in order; the report is written after each, so a failure keeps what ran before it."""
    import gc

    import torch

    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / "pilot_report.json"

    def save():
        path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")

    report: dict = {"resources": resources(tokens_root)}
    if report["resources"]["cuda"] and report["resources"]["gpu_free_gb"] < 4:
        raise RuntimeError(f"not enough free GPU memory for the pilot: {report['resources']}")
    report["data_checks"] = check_data(tokens_root, datasets, qa_root, n_check, seed)
    save()
    if not report["data_checks"]["passed"]:
        raise RuntimeError(f"pilot data checks failed: {json.dumps(report['data_checks'])[:2000]}")
    report["overfit"] = overfit(tokens_root, datasets, 300, seed)
    save()
    report["probes"] = []
    for p in probes:
        report["probes"].append(probe(tokens_root, datasets, p, CONTEXT, 5, gpu_memory_fraction, seed))
        save()
    report["runs"] = {}
    for r in runs:
        gc.collect()
        torch.cuda.empty_cache()
        name = f"{r['preset']}_lr{r['lr']:g}"
        cfg = TrainConfig(tokens_root=str(tokens_root), run_dir=str(run_root / name), datasets=datasets,
                          preset=r["preset"], lr=float(r["lr"]), max_steps=steps, tokens_per_batch=tokens_per_batch,
                          grad_accum=grad_accum, warmup_steps=max(steps // 20, 10), eval_every=max(steps // 4, 1),
                          log_every=10, seed=seed, gpu_memory_fraction=gpu_memory_fraction)
        out = train(cfg)
        out.update(preset=r["preset"], lr=float(r["lr"]), params_count=n_params(r["preset"], out_vocab(tokens_root)))
        report["runs"][name] = out
        save()
    return report


def out_vocab(tokens_root: Path) -> int:
    return len(Vocab.load(tokens_root / "vocab.json").tokens)


assert all(p in PRESETS for p in ("tiny", "pilot", "pilot_wide", "S", "M"))
