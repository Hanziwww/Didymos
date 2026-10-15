"""Training data from the S5 token shards (plan S6): windows, sampling, sample transforms, loss masks, batches.

* **Samples** are the stored windows (``<dataset>/<split>.parquet``, column ``windows``) with their stay's goal
  block (from ``goal_ids``) and dataset. A window is never cut or merged, so its pinned head and state block stay
  as the encoder wrote them.
* **Sampling:** dataset temperature sampling on tokens: a dataset's expected share of the training tokens is
  proportional to (its train tokens)^alpha (alpha 0.5), or, with target passes per dataset, to its passes times
  its train tokens (every dataset drawn about as often: the M7 privacy gate, docs/m7_privacy.md). Windows are drawn whole, so a dataset is drawn with
  probability proportional to that share over its mean window length (windows average 742 tokens in NWICU and
  2,831 in AmsterdamUMCdb on the subsets: drawing datasets per window would give the long-stay datasets the
  upsampling meant for the small ones). Within a dataset, windows come in shuffled passes (without replacement,
  as an epoch-based data loader), so every window is seen equally often; with several processes each takes a
  disjoint shard of every dataset (as a distributed sampler).
* **Transforms** (training only; every transformed window stays grammatical, which the pilot checks):
  - the goal block, fields in random order, inserted after the header in ``goal_fraction`` of samples (0.25);
    with ``goal_subsets``, a random non-empty subset of the fields, its size uniform (a request
    with some goals then conditions on a block of the kind training wrote, not an unseen partial block);
  - header fields in random order (any-order header);
  - dataset and site set to ``unknown`` in ``dataset_unknown`` of samples (0.10: the unconditional mode);
  - each header field set to ``unknown`` with ``header_unknown`` probability (prospective inputs carry header
    fields documented after their cutoff as unknown: ``tokenization.encode.prospective``).
* **Loss mask** (label -100): the first token; the goal block (conditioning, never generated); in a continuation
  window everything up to its ``[TL]`` (the pinned prefix and header, ``[CONT]`` and the state block, which the
  sampler rebuilds from its registers); the clock blocks (spec ``clock``, written by the sampler); padding.
  Everything else is a target: the first window's prefix and header, the timeline and the end block.
* **Batches:** padded to the longest sample of a length-sorted bucket (any attention implementation), or
  flattened with per-sample position resets for FlashAttention-2 (padding-free packing, plan S6).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from icutwin.tokenization.spec import load_spec
from icutwin.tokenization.vocab import Vocab

IGNORE = -100


@dataclass(frozen=True)
class Layout:
    """Token ids and positions the transforms and masks use, from the spec and the fitted vocabulary."""
    pad: int
    bos: int
    tl: int
    cont: int
    goal: int
    goal_end: int
    n_prefix: int
    n_header: int
    dataset_pos: int
    site_pos: int
    dataset_unknown: int
    site_unknown: int
    header_unknown: dict          # header field -> its unknown token
    header_field: dict            # header token -> field
    family: np.ndarray            # token id -> family index
    families: tuple               # family names
    clock: np.ndarray | None = None   # token id -> a clock token (round 3: written by the engine, never a target)


def token_family(meta: dict) -> str:
    """The loss family of a token: its class, events by domain (``event:VITAL``)."""
    cls = meta.get("cls") or "special"
    if cls in ("event", "end") and meta.get("code"):
        return f"{cls}:{meta['code'].split('//')[0]}"
    return cls


def layout(v: Vocab) -> Layout:
    spec = load_spec()
    prefix = [f["field"] for f in spec["prefix"]]
    unknown = spec["missing"]
    fams = sorted({token_family(m) for m in v.meta})
    fam_index = {f: i for i, f in enumerate(fams)}
    return Layout(
        pad=v.id("[PAD]"), bos=v.id("[BOS]"), tl=v.id("[TL]"), cont=v.id("[CONT]"), goal=v.id("[GOAL]"),
        goal_end=v.id("[/GOAL]"), n_prefix=len(prefix), n_header=len(spec["header"]),
        dataset_pos=1 + prefix.index("dataset"), site_pos=1 + prefix.index("site"),
        dataset_unknown=v.id(f"PREFIX//dataset//{unknown}"), site_unknown=v.id(f"PREFIX//site//{unknown}"),
        header_unknown={f["field"]: v.id(f"HEADER//{f['field']}//{unknown}") for f in spec["header"]},
        header_field={i: m["field"] for i, m in enumerate(v.meta) if m.get("cls") == "header"},
        family=np.array([fam_index[token_family(m)] for m in v.meta], dtype=np.int16), families=tuple(fams),
        clock=np.array([m.get("cls") == "clock" for m in v.meta], dtype=bool))


@dataclass
class Sample:
    ids: np.ndarray                  # the stored window
    goal: np.ndarray | None          # the stay's goal block, [GOAL] ... [/GOAL]
    dataset: str
    continuation: bool
    key: int = -1                    # the window's position in its dataset's split (stable across runs)


def load_samples(tokens_root: Path, datasets: list[str], split: str, v: Vocab) -> list[Sample]:
    """The stored windows of a split, read column-wise: a window is a view into its shard's token array (no
    Python list per token, which at full scale would cost tens of GB per training process). The goal block
    follows the header in ``goal_ids``, so only each sequence's head is read for it."""
    import pyarrow.compute as pc

    spec = load_spec()
    out = []
    goal, goal_end, cont = v.id("[GOAL]"), v.id("[/GOAL]"), v.id("[CONT]")
    head = 1 + len(spec["prefix"]) + len(spec["header"]) + len(spec["goal"]) + 2
    for ds in datasets:
        path = tokens_root / ds / f"{split}.parquet"
        if not path.exists():
            continue
        t = pq.read_table(path, columns=["goal_ids", "windows"])
        goals = []
        for h in pc.list_slice(t.column("goal_ids"), 0, head).to_pylist():
            if goal in h:
                a, b = h.index(goal), h.index(goal_end)
                goals.append(np.asarray(h[a:b + 1], dtype=np.int32))
            else:
                goals.append(None)
        col = t.column("windows").combine_chunks()
        per_stay = pc.list_value_length(col).to_numpy(zero_copy_only=False)
        windows = col.flatten()
        bounds = np.concatenate([[0], np.cumsum(pc.list_value_length(windows).to_numpy(zero_copy_only=False))])
        tokens = windows.flatten().to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        k = 0
        for g, n in zip(goals, per_stay):
            for _ in range(int(n)):
                w = tokens[bounds[k]:bounds[k + 1]]
                out.append(Sample(w, g, ds, bool((w == cont).any()), k))
                k += 1
    return out


def split_sizes(tokens_root: Path, datasets: list[str], split: str) -> dict[str, dict]:
    """Windows and tokens of a split per dataset, from the shards' list lengths (no token is materialized)."""
    import pyarrow.compute as pc

    out = {}
    for ds in datasets:
        path = tokens_root / ds / f"{split}.parquet"
        if path.exists():
            col = pq.read_table(path, columns=["windows"]).column("windows").combine_chunks()
            out[ds] = {"windows": int(pc.sum(pc.list_value_length(col)).as_py() or 0),
                       "tokens": int(pc.sum(pc.list_value_length(col.flatten())).as_py() or 0)}
    return out


def temperature_shares(tokens: np.ndarray, windows: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """(share, p): each dataset's expected share of the sampled tokens, proportional to tokens^alpha, and the
    probability of drawing it for one whole window (share over its mean window length, normalized)."""
    w = np.asarray(tokens, dtype=np.float64) ** alpha
    share = w / w.sum()
    p = share / (np.asarray(tokens, dtype=np.float64) / np.asarray(windows, dtype=np.float64))
    return share, p / p.sum()


def pass_shares(tokens: np.ndarray, windows: np.ndarray, passes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(share, p) for target passes: a dataset's share of the sampled tokens is its passes times its train tokens,
    so a budget of sum(passes x tokens) draws every dataset's train windows that many times."""
    w = np.asarray(passes, dtype=np.float64) * np.asarray(tokens, dtype=np.float64)
    share = w / w.sum()
    p = share / (np.asarray(tokens, dtype=np.float64) / np.asarray(windows, dtype=np.float64))
    return share, p / p.sum()


def target_passes(names: list[str], passes: dict | None) -> np.ndarray | None:
    """Per dataset (``names``) its target passes from ``passes`` ({dataset: passes, "default": passes}); None
    without targets (temperature sampling)."""
    if not passes:
        return None
    return np.array([float(passes.get(d, passes["default"])) for d in names], dtype=np.float64)


def mixing_shares(tokens: np.ndarray, windows: np.ndarray, alpha: float, passes: np.ndarray | None = None):
    """Target-pass shares when ``passes`` is given, else temperature shares."""
    return temperature_shares(tokens, windows, alpha) if passes is None else pass_shares(tokens, windows, passes)


def exposure(sizes: dict[str, dict], alpha: float, budget_tokens: float, passes: dict | None = None) -> dict[str, dict]:
    """Per dataset: train windows and tokens, the natural and the sampled token share, and the expected passes
    over its train windows when ``budget_tokens`` tokens are sampled (shuffled passes: each window is seen
    floor or ceil of that many times). ``passes``: target passes per dataset instead of temperature ``alpha``."""
    names = sorted(sizes)
    tokens = np.array([sizes[d]["tokens"] for d in names], dtype=np.float64)
    windows = np.array([sizes[d]["windows"] for d in names], dtype=np.float64)
    share, _p = mixing_shares(tokens, windows, alpha, target_passes(names, passes))
    return {d: {"windows": int(windows[i]), "tokens": int(tokens[i]), "natural_share": float(tokens[i] / tokens.sum()),
                "sampled_share": float(share[i]), "passes": float(share[i] * budget_tokens / tokens[i])}
            for i, d in enumerate(names)}


@dataclass
class TransformConfig:
    goal_fraction: float = 0.25
    shuffle_header: bool = True
    dataset_unknown: float = 0.10
    header_unknown: float = 0.05
    goal_subsets: bool = False      # off for the runs before #110 (their data order unchanged)
    goal_targets: bool = False      # the goal block's fields as targets (goal-first sampling); off for earlier runs


def transform(s: Sample, lay: Layout, rng: np.random.Generator, cfg: TransformConfig) -> np.ndarray:
    """A training sample from a stored window (module docstring); the stored window is not changed."""
    ids = s.ids.tolist()
    a = 1 + lay.n_prefix
    prefix, header, rest = ids[1:a], ids[a:a + lay.n_header], ids[a + lay.n_header:]
    if rng.random() < cfg.dataset_unknown:
        prefix[lay.dataset_pos - 1], prefix[lay.site_pos - 1] = lay.dataset_unknown, lay.site_unknown
    header = [lay.header_unknown[lay.header_field[t]] if rng.random() < cfg.header_unknown else t for t in header]
    if cfg.shuffle_header:
        rng.shuffle(header)
    block = []
    if s.goal is not None and rng.random() < cfg.goal_fraction:
        fields = s.goal[1:-1].tolist()
        rng.shuffle(fields)
        if cfg.goal_subsets and fields:  # the first k of the shuffled fields: a uniform subset of uniform size
            fields = fields[:int(rng.integers(1, len(fields) + 1))]
        block = [lay.goal, *fields, lay.goal_end]
    return np.asarray([lay.bos, *prefix, *header, *block, *rest], dtype=np.int32)


def labels_for(ids: np.ndarray, lay: Layout, goal_targets: bool = False) -> np.ndarray:
    """Targets of a sample (module docstring): -100 where the model is not trained to predict the token. With
    ``goal_targets`` (``TransformConfig.goal_targets``) the goal block's field tokens are targets, so the model learns
    the outcomes from the header (goal-first sampling draws them); its markers stay conditioning."""
    labels = ids.astype(np.int64).copy()
    labels[0] = IGNORE
    if lay.goal in ids:
        a = int(np.flatnonzero(ids == lay.goal)[0])
        b = int(np.flatnonzero(ids == lay.goal_end)[0])
        if goal_targets:
            labels[a] = labels[b] = IGNORE
        else:
            labels[a:b + 1] = IGNORE
    if lay.cont in ids:
        labels[:int(np.flatnonzero(ids == lay.tl)[0]) + 1] = IGNORE
    if lay.clock is not None:          # clock blocks: conditioning the sampler writes (spec clock)
        labels[lay.clock[ids]] = IGNORE
    return labels


class TemperatureSampler:
    """Infinite stream of training samples: a dataset, then the next window of its current shuffled pass (seeded,
    so a run's data order is reproducible). ``share`` is each dataset's expected share of the sampled tokens,
    proportional to (its train tokens)^alpha, or with ``passes`` ({dataset: passes, "default": passes}) to its
    target passes times its train tokens; ``p``, the probability of drawing the dataset, is that share over its
    mean window length (module docstring); ``natural`` is its share of the train tokens. With ``shard``
    (process, processes), the shares and probabilities are those of all windows and the passes run over the
    process's shard of each dataset."""

    def __init__(self, samples: list[Sample], alpha: float, seed: int, shard: tuple[int, int] = (0, 1),
                 passes: dict | None = None):
        groups: dict[str, list[Sample]] = defaultdict(list)
        for s in samples:
            groups[s.dataset].append(s)
        self.datasets = sorted(groups)
        tokens = np.array([sum(len(s.ids) for s in groups[d]) for d in self.datasets], dtype=np.float64)
        windows = np.array([len(groups[d]) for d in self.datasets], dtype=np.float64)
        self.share, self.p = mixing_shares(tokens, windows, alpha, target_passes(self.datasets, passes))
        self.natural = tokens / tokens.sum()
        k, n = shard
        self.groups = {d: groups[d][k::n] or groups[d] for d in self.datasets}   # tiny datasets: no empty shard
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        order: dict[str, list[int]] = {d: [] for d in self.datasets}
        while True:
            d = self.datasets[self.rng.choice(len(self.datasets), p=self.p)]
            if not order[d]:                          # a new pass over the dataset (its shard), reshuffled
                order[d] = self.rng.permutation(len(self.groups[d])).tolist()
            yield self.groups[d][order[d].pop()]


def pad_batches(stream, lay: Layout, tokens_per_batch: int, bucket: int = 64):
    """Batches of transformed samples padded to their longest one: a bucket of samples sorted by length and cut
    so that rows x longest <= tokens_per_batch (a sample longer than the budget is a batch of its own). A finite
    stream ends with its last, partial batch."""
    import itertools

    import torch

    while True:
        items = list(itertools.islice(stream, bucket))
        if not items:
            return
        items.sort(key=lambda x: len(x[0]))
        batch: list = []
        for it in items:
            longest = max([len(b[0]) for b in batch] + [len(it[0])])
            if batch and longest * (len(batch) + 1) > tokens_per_batch:
                yield _pad(batch, lay, torch)
                batch = []
            batch.append(it)
        if batch:
            yield _pad(batch, lay, torch)


def _pad(batch: list, lay: Layout, torch):
    n, m = len(batch), max(len(b[0]) for b in batch)
    ids = np.full((n, m), lay.pad, dtype=np.int64)
    labels = np.full((n, m), IGNORE, dtype=np.int64)
    mask = np.zeros((n, m), dtype=np.int64)
    meta = []
    for i, (x, y, info) in enumerate(batch):
        ids[i, :len(x)], labels[i, :len(x)], mask[i, :len(x)] = x, y, 1
        meta.append(info)
    return {"input_ids": torch.from_numpy(ids), "labels": torch.from_numpy(labels),
            "attention_mask": torch.from_numpy(mask), "meta": meta}


def noise_mask(x: np.ndarray, classes: np.ndarray, tl: int, end: int) -> np.ndarray:
    """Round 3's input noise: the positions of a sample whose input may be replaced, its timeline's event and
    interval tokens (``classes`` > 0) after the first [TL] and before [END]; never the prefix, header, goal or state
    blocks or the end block."""
    ok = classes[x] > 0
    at = np.flatnonzero(x == tl)
    ok[:at[0] + 1 if len(at) else len(x)] = False
    at = np.flatnonzero(x == end)
    if len(at):
        ok[at[0]:] = False
    return ok


def flat_batches(stream, tokens_per_batch: int, noise: tuple | None = None):
    """Padding-free batches for packed attention (FlashAttention-2, ``varlen``): samples concatenated into one row
    with per-sample position ids and the cumulative sample lengths with the longest sample (as transformers'
    DataCollatorWithFlattening with flash-attention kwargs), so attention and positions reset at every sample.
    A finite stream ends with its last, partial batch. With ``noise`` (classes per token, the [TL] and [END] ids),
    ``noise_ok`` marks the positions round 3's input noise may replace (``noise_mask``)."""
    import torch

    pending = None
    while True:
        xs, ys, pos, meta, n = [], [], [], [], 0
        while True:
            it = pending if pending is not None else next(stream, None)
            pending = None
            if it is None:
                break
            if xs and n + len(it[0]) > tokens_per_batch:
                pending = it
                break
            xs.append(it[0])
            ys.append(it[1])
            pos.append(np.arange(len(it[0])))
            meta.append(it[2])
            n += len(it[0])
        if not xs:
            return
        cu = torch.from_numpy(np.concatenate([[0], np.cumsum([len(x) for x in xs])]).astype(np.int32))
        longest = max(len(x) for x in xs)
        out = {"input_ids": torch.from_numpy(np.concatenate(xs).astype(np.int64))[None],
               "labels": torch.from_numpy(np.concatenate(ys))[None],
               "position_ids": torch.from_numpy(np.concatenate(pos).astype(np.int64))[None],
               "cu_seq_lens_q": cu, "cu_seq_lens_k": cu, "max_length_q": longest, "max_length_k": longest,
               "meta": meta}
        if noise is not None:
            out["noise_ok"] = torch.from_numpy(np.concatenate([noise_mask(x, *noise) for x in xs]))[None]
        yield out


def training_stream(samples: list[Sample], lay: Layout, cfg: TransformConfig, alpha: float, seed: int,
                    shard: tuple[int, int] = (0, 1), passes: dict | None = None):
    """(ids, labels, (dataset, window key)) of transformed samples drawn by the temperature sampler (from
    ``shard``; with target ``passes`` per dataset instead of ``alpha``)."""
    sampler = iter(TemperatureSampler(samples, alpha, seed, shard, passes))
    rng = np.random.default_rng(seed + 1)
    while True:
        s = next(sampler)
        ids = transform(s, lay, rng, cfg)
        yield ids, labels_for(ids, lay, cfg.goal_targets), (s.dataset, s.key)


def eval_items(samples: list[Sample], lay: Layout):
    """(ids, labels, dataset) of the stored windows, untransformed (no goal block, header in spec order)."""
    for s in samples:
        yield s.ids.astype(np.int32), labels_for(s.ids, lay), s.dataset
