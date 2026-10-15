"""S6 training (plan): Qwen3 from scratch on the S5 token shards.

CoMET's recipe: AdamW (beta 0.9/0.95, weight decay 0.1 on matrices), linear warmup, cosine decay to 10% of the
peak learning rate, gradient clipping at 1.0, bf16 autocast. HF Accelerate runs it on one device or across
devices (DDP, FSDP2: ``configs/accelerate/``); FlashAttention-2 with padding-free packing where installed,
padded batches with SDPA otherwise. Checkpoints are safetensors (``save_pretrained``, loadable by vLLM) plus the
optimizer, scheduler and RNG states (``Accelerator.save_state``); the data stream is seeded, so a resumed run
replays it to the same position. Validation: teacher-forced NLL per dataset and per token family on the
untransformed windows of the evaluation split.

Across processes and micro-batches:
* every process draws its own seeded stream from its shard of the windows (a single process reads them all);
* the loss of an optimizer step is the mean over all its target tokens, on every process and micro-batch
  (``num_items_in_batch``, as the HF Trainer since its gradient-accumulation fix), not a mean of per-batch means,
  which would weight a token by the size of the batch it fell into;
* the scheduler steps once per optimizer step (Accelerate steps a prepared scheduler once per process);
* only the main process writes files; checkpoints gather the full weights first (FSDP2).

Round 3's options (``docs/m7_refinement.md``), all off by default (the earlier runs unchanged):
* ``dropout``: residual dropout on every layer's attention and MLP outputs and on the input embeddings, in training
  only (forward hooks; the packed attention kernel has no attention dropout);
* ``ema_decay``: an exponential moving average of the weights, checkpointed with the run; at the end the weights
  with the lower tuning NLL become ``final/hf`` (the others ``final/hf_raw`` or ``final/hf_ema``);
* ``input_noise``: input-noise augmentation (parallel scheduled sampling): at a share of the timeline's event and
  interval inputs, ramping from 0 to ``input_noise`` over ``input_noise_ramp`` of the run, the input token is
  replaced by one drawn from the model's own prediction for that position (no gradient through the draw), kept only
  if of the same class; the labels stay the real tokens, so the model learns to continue the real stay from a
  context with its own errors in it.

Amendment 2's option: ``optimizer: muon``, Muon for the hidden weight matrices and AdamW for the embeddings and norms
(``model.optim``), on the recipe's learning rate, schedule and weight decay (data parallel only).
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from icutwin.model.data import (IGNORE, Layout, TransformConfig, eval_items, flat_batches, layout, load_samples,
                                pad_batches, training_stream)
from icutwin.model import attention  # noqa: F401  (registers the "varlen" attention implementation)
from icutwin.model.presets import qwen3_config
from icutwin.tokenization.vocab import Vocab

PACKED = ("flash_attention_2", attention.NAME)   # attention implementations that take padding-free packed batches

log = logging.getLogger(__name__)

DATA_SEED_STRIDE = 7919   # process k draws its training stream with seed + k * stride


@dataclass
class TrainConfig:
    tokens_root: str
    run_dir: str
    datasets: list
    preset: str = "pilot"
    eval_split: str = "tuning"
    alpha: float = 0.5
    passes: dict | None = None             # target passes per dataset ({name: passes, "default": passes}) over alpha
    transforms: TransformConfig = field(default_factory=TransformConfig)
    tokens_per_batch: int = 32768          # per micro-batch, padding included
    grad_accum: int = 1
    max_steps: int = 1000
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    precision: str = "bf16"
    attn_implementation: str = "sdpa"      # varlen / flash_attention_2: padding-free packed batches (PACKED)
    eval_every: int = 200
    eval_max_windows: int | None = None    # cap for quick evaluations (None: every window of the split)
    log_every: int = 10
    checkpoint_every: int = 0              # 0: only at the end
    seed: int = 1234
    gpu_memory_fraction: float | None = None   # torch.cuda.set_per_process_memory_fraction
    resume_from: str | None = None
    init_from: str | None = None           # amendment 17: start from a finished model's weights (final/hf), fresh
    #                                        optimizer and schedule (resume_from continues the same run instead)
    record_windows: int = 0                # smoke test: the windows every process draws in its first N steps
    check_loss: bool = False               # smoke test: the first step's loss recomputed from the logits
    compile: bool = False                  # torch.compile (inductor, dynamic shapes: packed batches vary in length)
    dropout: float = 0.0                   # round 3: residual and embedding dropout (module docstring)
    ema_decay: float | None = None         # round 3: EMA of the weights
    input_noise: float = 0.0               # round 3: peak share of timeline inputs replaced by the model's own draws
    input_noise_ramp: float = 0.5          # share of the run over which that rate ramps up from 0
    optimizer: str = "adamw"               # amendment 2: "muon" (Muon on hidden matrices, AdamW on the rest)
    muon_momentum: float = 0.95


def memory_efficient_sdpa() -> bool:
    """Make SDPA attention memory-efficient where PyTorch has no FlashAttention kernel (this machine's Windows build).
    transformers passes ``enable_gqa=True`` to SDPA when a batch has no padding mask, and without FlashAttention
    PyTorch then has only its math kernel, whose memory grows with the square of the context (12 GB for one layer
    at 8,192 tokens). Expanding the KV heads first, transformers' own path for masked batches, keeps the
    memory-efficient kernel. Linux server builds have FlashAttention (and the server uses FlashAttention-2), so
    nothing changes there. Returns whether the workaround is active."""
    import torch
    from transformers.integrations import sdpa_attention

    if torch.cuda.is_available() and not torch.backends.cuda.is_flash_attention_available():
        sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False
        return True
    return False


def build_model(preset: str, v: Vocab, attn: str):
    import torch
    from transformers import Qwen3ForCausalLM

    if attn == "sdpa":
        memory_efficient_sdpa()

    cfg = qwen3_config(preset, len(v.tokens), v.id("[PAD]"), v.id("[BOS]"), v.id("[EOS]"), attn_implementation=attn)
    model = Qwen3ForCausalLM(cfg)
    return model.to(torch.float32)


def init_weights(model, hf_dir: Path):
    """Load a finished model's weights (``<run>/final/hf/model.safetensors``) into ``model``, built with the same preset
    and vocabulary: every parameter must be present with the same shape (a tied output head may be absent)."""
    from safetensors.torch import load_file

    sd = load_file(str(Path(hf_dir) / "model.safetensors"))
    shapes = {k: tuple(p.shape) for k, p in model.state_dict().items()}
    bad = {k: (tuple(t.shape), shapes.get(k)) for k, t in sd.items() if shapes.get(k) != tuple(t.shape)}
    if bad:
        raise ValueError(f"init_from: shapes differ from the preset's: {dict(list(bad.items())[:5])}")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    tied = {"lm_head.weight"} if getattr(model.config, "tie_word_embeddings", False) else set()
    if unexpected or set(missing) - tied:
        raise ValueError(f"init_from: missing {sorted(set(missing) - tied)[:5]}, unexpected {sorted(unexpected)[:5]}")
    if set(missing) & tied:
        model.tie_weights()
    return model


def add_dropout(model, p: float) -> dict:
    """Residual dropout (module docstring) by forward hooks; returns the control whose ``"p"`` the hooks read (the
    smoke test's loss check sets it to 0 for one step)."""
    import torch.nn.functional as F

    ctl = {"p": float(p)}

    def hook(module, _args, out):
        if not module.training or ctl["p"] <= 0:
            return out
        if isinstance(out, tuple):                   # attention: (output, weights)
            return (F.dropout(out[0], ctl["p"], True), *out[1:])
        return F.dropout(out, ctl["p"], True)

    model.model.embed_tokens.register_forward_hook(hook)
    for layer in model.model.layers:
        layer.self_attn.register_forward_hook(hook)
        layer.mlp.register_forward_hook(hook)
    return ctl


class EMA:
    """Exponential moving average of a model's parameters (float32), with the usual warmup of the decay
    (min(decay, (1 + n) / (10 + n)) after n updates); ``state_dict``/``load_state_dict`` for Accelerate's
    checkpoints."""

    def __init__(self, model, decay: float):
        self.decay, self.steps = float(decay), 0
        self.shadow = {n: p.detach().float().clone() for n, p in model.named_parameters()}

    def update(self, model) -> None:
        import torch

        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        with torch.no_grad():
            for n, p in model.named_parameters():
                self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1 - d)

    def swap(self, model) -> None:
        """Exchange the model's parameters and the average (twice restores both)."""
        import torch

        with torch.no_grad():
            for n, p in model.named_parameters():
                tmp = p.detach().clone()
                p.copy_(self.shadow[n].to(p.dtype))
                self.shadow[n].copy_(tmp.float())

    def state_dict(self) -> dict:
        return {"decay": self.decay, "steps": self.steps, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay, self.steps = state["decay"], state["steps"]
        for n, t in state["shadow"].items():
            self.shadow[n].copy_(t)


def noise_classes(v: Vocab) -> np.ndarray:
    """Per token: 1 an event, 2 an interval, 0 anything else (never replaced, never a replacement); the ICU admission
    and discharge markers are 0 (the stay's frame stays real)."""
    out = np.zeros(len(v.tokens), dtype=np.int64)
    for i, m in enumerate(v.meta):
        code = m.get("code") or ""
        if m.get("cls") == "interval":
            out[i] = 2
        elif m.get("cls") == "event" and not code.startswith(("ICU//admission", "ICU//discharge")):
            out[i] = 1
    return out


def noise_rate(cfg: TrainConfig, step: int) -> float:
    if cfg.input_noise <= 0:
        return 0.0
    ramp = max(1.0, cfg.input_noise_ramp * cfg.max_steps)
    return float(cfg.input_noise * min(1.0, step / ramp))


def corrupt(model, inputs: dict, ok, q: float, classes) -> tuple[int, int]:
    """Input-noise augmentation of one packed micro-batch (module docstring), in place on ``inputs["input_ids"]``:
    each eligible position (``ok``) is picked with probability ``q`` and its token replaced by a draw from the
    model's prediction at the position before (temperature 1, no gradient), when the draw has the same class.
    Returns (replaced, eligible)."""
    import torch

    ids = inputs["input_ids"]
    pick = ok & (torch.rand(ok.shape, device=ok.device) < q)
    pick[..., 0] = False
    pos = pick[0].nonzero(as_tuple=True)[0]
    if len(pos):
        with torch.no_grad():
            logits = model(**{k: t for k, t in inputs.items() if k != "labels"}, logits_to_keep=pos - 1).logits[0]
            new = torch.multinomial(torch.softmax(logits.float(), -1), 1)[:, 0]
        same = classes[new] == classes[ids[0, pos]]
        ids[0, pos[same]] = new[same]
        return int(same.sum()), int(ok.sum())
    return 0, int(ok.sum())


def _optimizer(model, cfg: TrainConfig):
    import torch

    if cfg.optimizer == "muon":
        from icutwin.model.optim import MuonAdamW, split_parameters

        hidden, other, vectors = split_parameters(model)
        return MuonAdamW(hidden, [{"params": other + vectors, "weight_decay": 0.0}], lr=cfg.lr,
                         weight_decay=cfg.weight_decay, betas=tuple(cfg.betas), momentum=cfg.muon_momentum,
                         fused=torch.cuda.is_available())
    if cfg.optimizer != "adamw":
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim >= 2 and "embed" not in name else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas), eps=1e-8, fused=fused)


def eval_subset(samples: list, max_windows: int | None) -> list:
    """The windows an evaluation reads: all, or a fixed random subset of ``max_windows`` (the same at every
    evaluation, from every dataset: the samples are stored dataset by dataset)."""
    if not max_windows or max_windows >= len(samples):
        return samples
    keep = np.sort(np.random.default_rng(0).choice(len(samples), size=max_windows, replace=False))
    return [samples[i] for i in keep]


def evaluate(model, samples, lay: Layout, tokens_per_batch: int, device, max_windows: int | None = None,
             shard: tuple[int, int] = (0, 1), reduce=None) -> dict:
    """Teacher-forced NLL (nats per target token) overall, per dataset and per token family, accumulated on the
    device. With ``shard`` (process, processes) every process evaluates its share of the windows and ``reduce``
    (summing a tensor over the processes) combines the sums; under FSDP2 every process must instead run the same
    windows (identical forward passes, as its collectives need): the default."""
    import torch

    model.eval()
    windows = eval_subset(samples, max_windows)
    names = sorted({s.dataset for s in windows})
    items = sorted(eval_items(windows, lay), key=lambda x: len(x[0]))[shard[0]::shard[1]]
    fam = torch.from_numpy(lay.family.astype(np.int64)).to(device)
    sums = {k: torch.zeros(n, dtype=torch.float64, device=device)
            for k, n in (("ds", len(names)), ("fam", len(lay.families)))}
    counts = {k: torch.zeros(len(v), dtype=torch.float64, device=device) for k, v in sums.items()}
    with torch.no_grad():
        i = 0
        while i < len(items):
            batch = [items[i]]
            i += 1
            while i < len(items) and len(items[i][0]) * (len(batch) + 1) <= tokens_per_batch:
                batch.append(items[i])
                i += 1
            n, m = len(batch), max(len(b[0]) for b in batch)
            ids = torch.full((n, m), lay.pad, dtype=torch.long)
            labels = torch.full((n, m), IGNORE, dtype=torch.long)
            mask = torch.zeros((n, m), dtype=torch.long)
            for r, (x, y, _ds) in enumerate(batch):
                ids[r, :len(x)] = torch.from_numpy(x.astype(np.int64))
                labels[r, :len(x)] = torch.from_numpy(y)
                mask[r, :len(x)] = 1
            row_ds = torch.tensor([names.index(b[2]) for b in batch], device=device)
            ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(input_ids=ids, attention_mask=mask).logits
            tgt = labels[:, 1:]
            nll = torch.nn.functional.cross_entropy(logits[:, :-1].float().transpose(1, 2), tgt, ignore_index=IGNORE,
                                                    reduction="none")
            keep = tgt != IGNORE
            nll, ones = nll[keep].double(), torch.ones(int(keep.sum()), dtype=torch.float64, device=device)
            for key, index in (("ds", row_ds[:, None].expand_as(tgt)[keep]), ("fam", fam[tgt[keep]])):
                sums[key].index_add_(0, index, nll)
                counts[key].index_add_(0, index, ones)
    if reduce is not None:
        sums, counts = {k: reduce(v) for k, v in sums.items()}, {k: reduce(v) for k, v in counts.items()}
    model.train()
    s_ds, c_ds = sums["ds"].tolist(), counts["ds"].tolist()
    s_fam, c_fam = sums["fam"].tolist(), counts["fam"].tolist()
    count = int(sum(c_ds))
    return {"nll": sum(s_ds) / max(count, 1), "target_tokens": count,
            "by_dataset": {d: s_ds[j] / c_ds[j] for j, d in enumerate(names) if c_ds[j]},
            "by_family": {f: {"nll": s_fam[j] / c_fam[j], "tokens": int(c_fam[j])}
                          for j, f in enumerate(lay.families) if c_fam[j]}}


AUX = ("meta", "noise_ok")      # batch entries that are not model inputs


def accumulate_gradients(acc, model, micro: list[dict], noise: dict | None = None):
    """Forward and backward passes of one optimizer step's micro-batches (``acc.gradient_accumulation_steps`` of
    them) on every process. The gradient is that of the mean loss over the step's target tokens on all processes;
    returns that mean. The optimizer steps after the last micro-batch, whose backward pass synchronizes. With
    ``noise`` ({"q", "classes"}; packed batches with ``noise_ok``), each micro-batch's inputs are corrupted first
    (``corrupt``), and ``noise["replaced"]``/``noise["eligible"]`` count it."""
    import torch

    assert len(micro) == acc.gradient_accumulation_steps
    n_items = acc.reduce(torch.tensor(sum(int((b["labels"][..., 1:] != IGNORE).sum()) for b in micro),
                                      device=acc.device), "sum")
    total = torch.zeros((), device=acc.device)
    for b in micro:
        inputs = {k: t.to(acc.device) if hasattr(t, "to") else t for k, t in b.items() if k not in AUX}
        if noise is not None and noise["q"] > 0 and "noise_ok" in b:
            r, e = corrupt(model, inputs, b["noise_ok"].to(acc.device), noise["q"], noise["classes"])
            noise["replaced"] += r
            noise["eligible"] += e
            b["input_ids"] = inputs["input_ids"].cpu()       # the step's batch as trained (the smoke test's check)
        with acc.accumulate(model):
            out = model(**inputs, num_items_in_batch=n_items)   # this batch's summed token losses / n_items
            # Accelerate divides by the accumulation steps and data parallelism averages over processes: undo both
            acc.backward(out.loss * (acc.gradient_accumulation_steps * acc.num_processes))
        total += out.loss.detach()
    assert acc.sync_gradients
    return acc.reduce(total, "sum")


def _token_loss_sums(model, micro: list[dict], device) -> tuple[float, int]:
    """The summed token losses and the target tokens of a step's micro-batches, from the logits (a path separate
    from the model's loss function, for the smoke test)."""
    import torch

    total, n = 0.0, 0
    with torch.no_grad():
        for b in micro:
            logits = model(**{k: t.to(device) if hasattr(t, "to") else t for k, t in b.items()
                              if k not in (*AUX, "labels")}).logits
            labels = b["labels"][..., 1:].to(device)
            keep = labels != IGNORE
            lp = torch.log_softmax(logits[..., :-1, :].float(), dim=-1)
            total += float(-lp[keep].gather(-1, labels[keep][:, None]).sum())
            n += int(keep.sum())
    return total, n


def train(cfg: TrainConfig) -> dict:
    import torch
    from accelerate import Accelerator
    from transformers import get_cosine_with_min_lr_schedule_with_warmup

    from accelerate.utils import TorchDynamoPlugin

    acc = Accelerator(mixed_precision=cfg.precision if torch.cuda.is_available() else "no",
                      gradient_accumulation_steps=cfg.grad_accum,
                      dynamo_plugin=TorchDynamoPlugin(backend="inductor", dynamic=True) if cfg.compile else None)
    main = acc.is_main_process
    run = Path(cfg.run_dir)
    if main:
        run.mkdir(parents=True, exist_ok=True)
        (run / "config.json").write_text(json.dumps(asdict(cfg), indent=1, default=str), encoding="utf-8")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)                      # the same initial weights on every process
    if cfg.gpu_memory_fraction and acc.device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(cfg.gpu_memory_fraction, acc.device.index or 0)
    v = Vocab.load(Path(cfg.tokens_root) / "vocab.json")
    lay = layout(v)
    train_samples = load_samples(Path(cfg.tokens_root), list(cfg.datasets), "train", v)
    eval_samples = load_samples(Path(cfg.tokens_root), list(cfg.datasets), cfg.eval_split, v)
    seed, shard = cfg.seed + DATA_SEED_STRIDE * acc.process_index, (acc.process_index, acc.num_processes)
    stream = training_stream(train_samples, lay, cfg.transforms, cfg.alpha, seed, shard, cfg.passes)
    flat = cfg.attn_implementation in PACKED
    noise_spec = None
    if cfg.input_noise > 0:
        if not flat:
            raise ValueError("input noise needs packed batches (attn_implementation varlen or flash_attention_2)")
        noise_spec = (noise_classes(v), v.id("[TL]"), v.id("[END]"))
    batches = (flat_batches(stream, cfg.tokens_per_batch, noise_spec) if flat
               else pad_batches(stream, lay, cfg.tokens_per_batch))
    model = build_model(cfg.preset, v, cfg.attn_implementation)
    if cfg.init_from:
        if cfg.resume_from:
            raise ValueError("init_from starts a new run; resume_from continues one: give one of them")
        init_weights(model, Path(cfg.init_from))
    ctl = add_dropout(model, cfg.dropout) if cfg.dropout > 0 else None
    opt = _optimizer(model, cfg)
    sched = get_cosine_with_min_lr_schedule_with_warmup(opt, cfg.warmup_steps, cfg.max_steps,
                                                       min_lr_rate=cfg.min_lr_ratio)
    from accelerate.utils import DistributedType

    if cfg.optimizer == "muon" and acc.distributed_type == DistributedType.FSDP:
        raise ValueError("Muon orthogonalizes whole matrices: data parallel only, not FSDP shards")
    model, opt = acc.prepare(model, opt)
    acc.register_for_checkpointing(sched)            # not prepared: stepped here once per optimizer step
    ema = EMA(acc.unwrap_model(model, keep_torch_compile=False), cfg.ema_decay) if cfg.ema_decay else None
    if ema is not None:
        acc.register_for_checkpointing(ema)
    classes_t = torch.from_numpy(noise_spec[0]).to(acc.device) if noise_spec else None
    start = 0
    if cfg.resume_from:
        acc.load_state(cfg.resume_from)
        start = json.loads((Path(cfg.resume_from) / "trainer_state.json").read_text())["step"]
        for _ in range(start * cfg.grad_accum):      # replay the seeded data stream to the saved position
            next(batches)
    metrics = (run / "metrics.jsonl").open("a", encoding="utf-8") if main else None
    # evaluation: every process its share of the windows (DDP, one process); FSDP2 runs all of them everywhere
    eval_shards = {} if acc.distributed_type == DistributedType.FSDP else {
        "shard": (acc.process_index, acc.num_processes), "reduce": lambda t: acc.reduce(t, "sum")}

    def write(rec: dict) -> None:
        if metrics:
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()

    n_params = sum(p.numel() for p in model.parameters())
    counts = torch.zeros(2, dtype=torch.float64, device=acc.device)   # tokens, target tokens (this process)
    t0 = time.time()
    busy = 0.0                                       # seconds in training steps (not evaluations or checkpoints)
    first_step, drawn = None, []
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    history = []
    for step in range(start, cfg.max_steps):
        t_step = time.time()
        micro = [next(batches) for _ in range(cfg.grad_accum)]
        noise = ({"q": noise_rate(cfg, step), "classes": classes_t, "replaced": 0, "eligible": 0}
                 if noise_spec else None)
        if ctl is not None and cfg.check_loss and step == start:
            ctl["p"] = 0.0                        # the loss check compares two forward passes: none dropped here
        loss = float(accumulate_gradients(acc, model, micro, noise))
        if cfg.check_loss and step == start:      # same weights as the loss above: the update comes next
            sums = acc.reduce(torch.tensor(_token_loss_sums(model, micro, acc.device), dtype=torch.float64,
                                           device=acc.device), "sum")
            first_step = {"step": step + 1, "loss": loss, "recomputed": float(sums[0] / sums[1]),
                          "target_tokens": int(sums[1])}
        if ctl is not None:
            ctl["p"] = cfg.dropout
        if step - start < cfg.record_windows:
            drawn.append([list(m) for b in micro for m in b["meta"]])
        acc.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if ema is not None:
            ema.update(acc.unwrap_model(model, keep_torch_compile=False))
        for b in micro:
            counts[0] += int(b["attention_mask"].sum()) if "attention_mask" in b else int(b["input_ids"].numel())
            counts[1] += int((b["labels"][..., 1:] != IGNORE).sum())
        busy += time.time() - t_step
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        if (step + 1) % cfg.log_every == 0 or step == start:
            tok_all, tok_tgt = (int(x) for x in acc.reduce(counts.clone(), "sum").tolist())
            rec = {"step": step + 1, "loss": loss, "lr": sched.get_last_lr()[0], "tokens": tok_all,
                   "target_tokens": tok_tgt, "tokens_per_s": tok_all / busy, "seconds": busy,
                   "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
            if noise is not None:                 # this process's step: the rate and the share replaced
                rec.update(noise_q=noise["q"], noise_share=noise["replaced"] / max(noise["eligible"], 1))
            write(rec)
            history.append(rec)
        if cfg.eval_every and (step + 1) % cfg.eval_every == 0 and step + 1 < cfg.max_steps:
            ev = evaluate(acc.unwrap_model(model, keep_torch_compile=False), eval_samples, lay, cfg.tokens_per_batch,
                          acc.device, cfg.eval_max_windows, **eval_shards)
            write({"step": step + 1, "eval": ev})
            history.append({"step": step + 1, "eval_nll": ev["nll"]})
            if main:
                log.info("step %d eval nll %.4f", step + 1, ev["nll"])
        if cfg.checkpoint_every and (step + 1) % cfg.checkpoint_every == 0:
            _checkpoint(acc, model, run / f"step_{step + 1}", step + 1, ema)
    unwrapped = acc.unwrap_model(model, keep_torch_compile=False)
    raw = evaluate(unwrapped, eval_samples, lay, cfg.tokens_per_batch, acc.device, cfg.eval_max_windows, **eval_shards)
    final, extra = raw, {}
    if ema is not None:                           # the average's tuning NLL; the lower one becomes final/hf
        ema.swap(unwrapped)
        averaged = evaluate(unwrapped, eval_samples, lay, cfg.tokens_per_batch, acc.device, cfg.eval_max_windows,
                            **eval_shards)
        ema.swap(unwrapped)
        use_ema = averaged["nll"] < raw["nll"]
        final = averaged if use_ema else raw
        extra = {"eval_raw": raw, "eval_ema": averaged, "weights": "ema" if use_ema else "raw"}
    write({"step": cfg.max_steps, "eval": final, **extra})
    if metrics:
        metrics.close()
    _checkpoint(acc, model, run / "final", cfg.max_steps, ema)
    if extra.get("weights") == "ema" and main:
        (run / "final" / "hf").rename(run / "final" / "hf_raw")
        (run / "final" / "hf_ema").rename(run / "final" / "hf")
    acc.wait_for_everyone()
    tok_all, tok_tgt = (int(x) for x in acc.reduce(counts.clone(), "sum").tolist())
    from accelerate.utils import gather_object

    gpu = acc.device.type == "cuda"
    shard_windows = sum(len(range(shard[0], n, shard[1])) or n for n in Counter(s.dataset for s in train_samples).values())
    per_process = gather_object([{
        "process": acc.process_index, "seed": seed, "shard": list(shard), "shard_windows": shard_windows,
        "peak_gpu_gb": torch.cuda.max_memory_allocated(acc.device) / 2**30 if gpu else None,
        "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved(acc.device) / 2**30 if gpu else None,
        "gpu_total_gb": torch.cuda.get_device_properties(acc.device).total_memory / 2**30 if gpu else None,
        "drawn": drawn}])
    return {"params": n_params, "steps": cfg.max_steps, "tokens": tok_all, "target_tokens": tok_tgt,
            "first_step": first_step, "per_process": per_process,
            "seconds": busy, "wall_seconds": time.time() - t0, "tokens_per_s": tok_all / busy,
            "processes": acc.num_processes, "process_index": acc.process_index,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
            "final_eval": final, **{f"final_{k}": x for k, x in extra.items()}, "history": history}


def _checkpoint(acc, model, path: Path, step: int, ema: EMA | None = None) -> None:
    """Optimizer, scheduler and RNG states of every process (the EMA with them), and the full weights as safetensors
    (gathered from the shards under FSDP2; written by the main process); the EMA's weights as ``hf_ema``."""
    acc.save_state(str(path))
    state = {k.replace("_orig_mod.", ""): v for k, v in acc.get_state_dict(model).items()}   # compiled: prefixed keys
    unwrapped = acc.unwrap_model(model, keep_torch_compile=False)
    unwrapped.save_pretrained(str(path / "hf"), is_main_process=acc.is_main_process, state_dict=state)
    if ema is not None:
        unwrapped.save_pretrained(str(path / "hf_ema"), is_main_process=acc.is_main_process, state_dict=dict(ema.shadow))
    if acc.is_main_process:
        (path / "trainer_state.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    acc.wait_for_everyone()
