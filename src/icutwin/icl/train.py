"""Training TrajPFN on factory episodes (plan Phase G, "Training"; this round: the learning confirmation).

* data: ``factory.stream.EpisodeStream`` (episode i a function of the store, pool, seed and i), read by DataLoader
  workers that also build the batches (``batch.Builder``), split into micro-batches under a token budget;
* optimizer: Muon on the hidden matrices (``model.optim.MuonAdamW``: TabICLv2 trains with Muon), AdamW on embeddings,
  heads and small projections; warmup, then cosine decay to ``min_lr_ratio``; gradient clipping; bf16 autocast;
  an EMA of the weights (``model.train.EMA``);
* evaluation: the model's and the base-rate baseline's per-episode losses on a fixed set of held-out episodes
  (twin validation split), by label type;
* checkpoints: model, optimizer, EMA, step and the last episode index (a resumed run starts after it).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from icutwin.icl import batch as B
from icutwin.icl import heads as H
from icutwin.icl import model as M


@dataclass
class TrainConfig:
    store: str
    pool: str | None
    vocab: str
    run_dir: str
    preset: str = "S"
    stage: int = 1
    seed: int = 0
    episodes: int = 200_000              # training episodes to consume
    start_index: int = 0
    batch_episodes: int = 8
    max_tokens: int = 400_000            # tokens per micro-batch (split above it)
    max_rows: int = 64                   # train rows per context patient
    lr: float = 8e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.05
    optimizer: str = "muon"
    grad_clip: float = 1.0
    ema_decay: float | None = 0.999
    workers: int = 8
    eval_every: int = 1000
    eval_set: str | None = None          # a dump of evaluation episodes (``make_set``)
    checkpoint_every: int = 2000
    log_every: int = 25
    gpu_memory_fraction: float | None = 0.8
    precision: str = "bf16"
    overfit: int = 0                     # > 0: train on that many fixed episodes, over and over (check L0)
    model_overrides: dict = field(default_factory=dict)


def channel_table(store_dir: str, vocab: str) -> tuple[list, int]:
    from icutwin.factory import store as fst
    from icutwin.factory.trajectories import static_vocab
    from icutwin.tokenization.vocab import Vocab

    st = fst.Store(Path(store_dir))
    return list(st.channels.names), len(static_vocab(Vocab.load(Path(vocab))))


class Collate:
    """Episodes → micro-batches (lists of tensor dicts), each under ``max_tokens`` estimated tokens."""

    def __init__(self, builder: B.Builder, max_tokens: int):
        self.builder, self.max_tokens = builder, max_tokens

    @staticmethod
    def tokens(ep: dict) -> int:
        n = 0
        for part in ("context", "query"):
            d = ep[part]
            n += len(d["ev_t"]) + 2 * len(d["iv_s"]) + len(d["st_k"]) + 2 * len(d["lab_t"])
        return n

    def __call__(self, eps: list) -> list:
        out, cur, size = [], [], 0
        for ep in sorted(eps, key=self.tokens):
            n = self.tokens(ep)
            if cur and size + n > self.max_tokens:
                out.append(B.to_torch(self.builder(cur)))
                cur, size = [], 0
            cur.append(ep)
            size += n
        if cur:
            out.append(B.to_torch(self.builder(cur)))
        return out


class Episodes(torch.utils.data.IterableDataset):
    """An ``EpisodeStream`` as an IterableDataset (module level, so spawned workers can unpickle it)."""

    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def __iter__(self):
        return iter(self.stream)


def optimizer(model: torch.nn.Module, cfg: TrainConfig):
    from icutwin.model.optim import MuonAdamW

    muon, decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        hidden = p.ndim == 2 and "embed" not in name and not name.startswith("heads.") and min(p.shape) >= 16
        if cfg.optimizer == "muon" and hidden:
            muon.append(p)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if cfg.optimizer == "muon" and muon:
        return MuonAdamW(muon, groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(groups + ([{"params": muon, "weight_decay": cfg.weight_decay}] if muon else []),
                             lr=cfg.lr, betas=(0.9, 0.95))


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    frac = min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * frac)))


def to_device(b: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, batches: list, device, autocast) -> dict:
    """Per label type: the model's and the base-rate baseline's mean per-episode loss, and the share of episodes
    where the model is better."""
    model.eval()
    rows: dict = {}
    for b in batches:
        b = to_device(b, device)
        with autocast():
            res = model(b)
        n_ep = int(b["n_episodes"])
        mine, has = H.episode_mean(res["row_loss"].float(), b["te_ep"], n_ep)
        base, _ = H.episode_mean(H.baseline_losses(b, model.cfg.bins), b["te_ep"], n_ep)
        for e in range(n_ep):
            if has[e] and torch.isfinite(base[e]):
                kind = B.LABELS[int(b["ep_label"][e])]
                rows.setdefault(kind, []).append((float(mine[e]), float(base[e])))
    model.train()
    out = {}
    for kind, xs in rows.items():
        a = np.asarray(xs)
        out[kind] = {"episodes": len(a), "model": float(a[:, 0].mean()), "base_rate": float(a[:, 1].mean()),
                     "better_share": float((a[:, 0] < a[:, 1]).mean())}
    allr = np.asarray([x for xs in rows.values() for x in xs]) if rows else np.zeros((0, 2))
    out["all"] = {"episodes": len(allr), "model": float(allr[:, 0].mean()) if len(allr) else None,
                  "base_rate": float(allr[:, 1].mean()) if len(allr) else None}
    return out


def make_set(store: str, pool: str | None, vocab: str, out: str, n: int, seed: int, split: str | None,
             stage: int = 1, stats_from: str | None = None) -> dict:
    """A frozen set of evaluation episodes (NPZ shards with a manifest): indices 0..n-1 of the stream of ``store``
    (``split``; the pool's programs when given). ``stats_from``: a twin store whose channel statistics standardize a
    real store's values (one standardization for twins and real stays)."""
    from icutwin.factory import episodes, pool as fpool, store as fst, stream
    from icutwin.tokenization.vocab import Vocab

    st = fst.Store(Path(store))
    if stats_from:
        st.channels.stats = fst.Store(Path(stats_from)).channels.stats
    progs, weights, support = fpool.load(Path(pool)) if pool else (None, None, None)
    s = episodes.Sampler(st, Vocab.load(Path(vocab)), stage=stage, seed=seed, programs=progs, weights=weights,
                         targets=fpool.load_targets() if pool else None, split=split, support=support)
    eps = []
    for i in range(n):
        try:
            eps.append(s.episode(i))
        except episodes.NoEpisode:
            continue
    od = Path(out)
    od.mkdir(parents=True, exist_ok=True)
    shards = []
    for k in range(0, len(eps), 128):
        p = od / f"part_{k // 128:04d}.npz"
        shards.append({"file": p.name, "sha256": stream.write_shard(eps[k:k + 128], p), "n": len(eps[k:k + 128])})
    man = {"store": store, "pool": pool, "split": split, "seed": seed, "stage": stage, "episodes": len(eps),
           "requested": n, "shards": shards, "stats_from": stats_from}
    (od / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


def load_set(path: str) -> list:
    from icutwin.factory import stream

    man = json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8"))
    return [ep for s in man["shards"] for ep in stream.read_shard(Path(path) / s["file"])]


def train(cfg: TrainConfig) -> dict:
    from contextlib import nullcontext

    from icutwin.factory import stream
    from icutwin.model.train import EMA

    run = Path(cfg.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if cfg.gpu_memory_fraction:
            torch.cuda.set_per_process_memory_fraction(cfg.gpu_memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = True
    names, n_static = channel_table(cfg.store, cfg.vocab)
    mcfg = M.config(cfg.preset, len(names), n_static, **cfg.model_overrides)
    model = M.TrajPFN(mcfg).to(device)
    opt = optimizer(model, cfg)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    step, next_index = 0, cfg.start_index
    ck = run / "checkpoint.pt"
    if ck.exists():
        state = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        if ema and state.get("ema"):
            ema.load_state_dict(state["ema"])
        step, next_index = state["step"], state["next_index"]
    builder = B.Builder(names, n_static, train=True, max_rows=cfg.max_rows, d_id=mcfg.d_id, seed=cfg.seed)
    collate = Collate(builder, cfg.max_tokens)
    use_bf16 = cfg.precision == "bf16" and device.type == "cuda"

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    eval_batches = []
    if cfg.eval_set:
        ev_builder = B.Builder(names, n_static, train=False, max_rows=cfg.max_rows, d_id=mcfg.d_id, seed=cfg.seed)
        ev = load_set(cfg.eval_set)
        eval_batches = [b for k in range(0, len(ev), cfg.batch_episodes)
                        for b in Collate(ev_builder, cfg.max_tokens)(ev[k:k + cfg.batch_episodes])]
    total_steps = math.ceil(cfg.episodes / cfg.batch_episodes)
    if cfg.overfit:
        es = stream.EpisodeStream(Path(cfg.store), Path(cfg.vocab), Path(cfg.pool) if cfg.pool else None,
                                  cfg.stage, cfg.seed, 0, cfg.overfit * 4, "train")
        fixed = list(es)[:cfg.overfit]
        batches = (collate(fixed) for _ in iter(int, 1))
    else:
        es = stream.EpisodeStream(Path(cfg.store), Path(cfg.vocab), Path(cfg.pool) if cfg.pool else None,
                                  cfg.stage, cfg.seed, next_index, None, "train")
        loader = torch.utils.data.DataLoader(Episodes(es), batch_size=cfg.batch_episodes, collate_fn=collate,
                                             num_workers=cfg.workers, pin_memory=device.type == "cuda",
                                             prefetch_factor=4 if cfg.workers else None,
                                             persistent_workers=bool(cfg.workers))
        batches = iter(loader)
    log = (run / "log.jsonl").open("a", encoding="utf-8")
    (run / "config.json").write_text(json.dumps({"train": asdict(cfg), "model": asdict(mcfg),
                                                 "parameters": M.parameter_count(model)}, indent=1), encoding="utf-8")
    model.train()
    t0, seen_ep, seen_tok, acc_loss, acc_n = time.time(), 0, 0, 0.0, 0
    by_kind: dict = {}
    while step < total_steps:
        micro = next(batches)
        lr = lr_at(step, total_steps, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        n_all = sum(int(b["n_episodes"]) for b in micro)
        opt.zero_grad(set_to_none=True)
        for b in micro:
            b = to_device(b, device)
            with autocast():
                loss, res = model.loss(b)
            n_ep = int(b["n_episodes"])
            (loss * n_ep / n_all).backward()
            acc_loss += float(loss) * n_ep
            acc_n += n_ep
            seen_tok += len(b["tok_kind"])
            next_index = max(next_index, int(b["ep_index"].max()) + 1)
            for e in range(n_ep):
                if res["has"][e]:
                    kind = B.LABELS[int(b["ep_label"][e])]
                    by_kind.setdefault(kind, []).append(float(res["per_episode"][e]))
        seen_ep += n_all
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if ema:
            ema.update(model)
        step += 1
        if step % cfg.log_every == 0:
            dt = time.time() - t0
            rec = {"step": step, "lr": lr, "loss": acc_loss / max(acc_n, 1), "grad_norm": float(gn),
                   "episodes_per_s": seen_ep / dt, "tokens_per_s": seen_tok / dt, "next_index": next_index,
                   "by_kind": {k: float(np.mean(v)) for k, v in by_kind.items()},
                   "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t0, seen_ep, seen_tok, acc_loss, acc_n, by_kind = time.time(), 0, 0, 0.0, 0, {}
        if eval_batches and (step % cfg.eval_every == 0 or step == total_steps):
            res_ev = evaluate(model, eval_batches, device, autocast)
            if ema:
                ema.swap(model)
                res_ev["ema"] = evaluate(model, eval_batches, device, autocast)
                ema.swap(model)
            log.write(json.dumps({"step": step, "eval": res_ev}) + "\n")
            log.flush()
        if step % cfg.checkpoint_every == 0 or step == total_steps:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "ema": ema.state_dict() if ema else None, "step": step, "next_index": next_index,
                        "config": asdict(cfg)}, ck)
    log.close()
    return {"steps": step, "next_index": next_index}
