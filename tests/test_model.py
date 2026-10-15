"""S6 training data and model (M5): transforms stay grammatical, loss masks, sampling, batches, presets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from test_tokenization import SPLITS, _write_meds

from icutwin.model.data import (IGNORE, Sample, TemperatureSampler, TransformConfig, flat_batches, labels_for, layout,
                                pad_batches, transform)
from icutwin.model.pilot import _expected_labels
from icutwin.model.presets import n_params
from icutwin.tokenization.encode import encode
from icutwin.tokenization.fsm import accepts, compile_tables
from icutwin.tokenization.stays import iter_stays
from icutwin.tokenization.vocab import fit


@pytest.fixture(autouse=True)
def _fresh_accelerate_state():
    """Accelerate keeps process-wide singletons: reset them after each test, as accelerate's own tests do."""
    yield
    from accelerate.state import AcceleratorState, GradientState
    AcceleratorState._reset_state(True)
    GradientState._reset_state()


@pytest.fixture()
def tok(tmp_path: Path):
    meds = _write_meds(tmp_path / "meds", SPLITS)
    v = fit(iter_stays(meds, "train"), min_subjects=1)
    samples = []
    for split in SPLITS:
        for s in iter_stays(meds, split):
            g = encode(s, v, goal=True).ids
            goal = np.asarray(g[g.index(v.id("[GOAL]")):g.index(v.id("[/GOAL]")) + 1], dtype=np.int32)
            for w in encode(s, v, max_tokens=28).windows:     # short windows: continuation windows with state
                samples.append(Sample(np.asarray(w, dtype=np.int32), goal, "mimic" if s.subject_id % 2 else "eicu",
                                      v.id("[CONT]") in w))
    return v, samples


def test_transforms_stay_grammatical_and_masks_match_the_parser(tok) -> None:
    v, samples = tok
    lay, tables = layout(v), compile_tables(v)
    assert any(s.continuation for s in samples)
    rng = np.random.default_rng(0)
    cfg = TransformConfig(goal_fraction=0.5, dataset_unknown=0.3, header_unknown=0.3)
    seen_goal = seen_unknown = 0
    for s in samples:
        for _ in range(20):
            ids = transform(s, lay, rng, cfg)
            ok, why = accepts(tables, ids.tolist(), whole=ids[-1] == v.id("[EOS]"))
            assert ok, why
            labels = labels_for(ids, lay)
            assert np.array_equal(labels, _expected_labels(ids, v))
            seen_goal += lay.goal in ids
            seen_unknown += lay.dataset_unknown in ids
            if lay.goal in ids:   # the goal block is conditioning: never a target
                a, b = int(np.flatnonzero(ids == lay.goal)[0]), int(np.flatnonzero(ids == lay.goal_end)[0])
                assert (labels[a:b + 1] == IGNORE).all()
            if s.continuation:    # pinned context of a continuation window: no target before the timeline
                assert (labels[:int(np.flatnonzero(ids == v.id("[TL]"))[0]) + 1] == IGNORE).all()
            else:                 # the first window's prefix and header are targets
                assert (labels[1:1 + lay.n_prefix] != IGNORE).all()
    assert seen_goal and seen_unknown
    assert s.ids.tolist() == samples[-1].ids.tolist()   # stored windows are never changed in place


def test_goal_subsets_are_grammatical_and_uniform_in_size(tok) -> None:
    """#110: with ``goal_subsets`` the goal block holds a random non-empty subset of the stay's goal fields, every
    size about equally often, and stays grammatical; without it the block holds every field (earlier runs)."""
    v, samples = tok
    lay, tables = layout(v), compile_tables(v)
    rng = np.random.default_rng(0)
    full = len(samples[0].goal) - 2
    sizes = Counter()
    for s in samples:
        for _ in range(40):
            ids = transform(s, lay, rng, TransformConfig(goal_fraction=1.0, goal_subsets=True))
            ok, why = accepts(tables, ids.tolist(), whole=ids[-1] == v.id("[EOS]"))
            assert ok, why
            a, b = int(np.flatnonzero(ids == lay.goal)[0]), int(np.flatnonzero(ids == lay.goal_end)[0])
            block = ids[a + 1:b].tolist()
            assert block and set(block) <= set(s.goal[1:-1].tolist()) and len(set(block)) == len(block)
            sizes[len(block)] += 1
    n = sum(sizes.values())
    assert set(sizes) == set(range(1, full + 1)), sizes
    assert all(abs(c / n - 1 / full) < 0.03 for c in sizes.values()), sizes
    ids = transform(samples[0], lay, rng, TransformConfig(goal_fraction=1.0))
    assert int(np.flatnonzero(ids == lay.goal_end)[0]) - int(np.flatnonzero(ids == lay.goal)[0]) - 1 == full


def test_goal_targets_make_the_goal_fields_targets(tok) -> None:
    """``goal_targets``: the goal block's field tokens are targets (the model learns the outcomes from the header,
    which goal-first sampling draws), its markers are not; without it the whole block is conditioning; the stored
    windows (evaluation, P1) have no goal block, so their targets do not change."""
    v, samples = tok
    lay = layout(v)
    rng = np.random.default_rng(0)
    ids = transform(samples[0], lay, rng, TransformConfig(goal_fraction=1.0))
    a, b = int(np.flatnonzero(ids == lay.goal)[0]), int(np.flatnonzero(ids == lay.goal_end)[0])
    on, off = labels_for(ids, lay, goal_targets=True), labels_for(ids, lay)
    assert (off[a:b + 1] == IGNORE).all()
    assert on[a] == on[b] == IGNORE and (on[a + 1:b] == ids[a + 1:b]).all() and b - a > 1
    assert np.array_equal(np.delete(on, np.arange(a, b + 1)), np.delete(off, np.arange(a, b + 1)))
    s = samples[0]
    assert np.array_equal(labels_for(s.ids, lay, goal_targets=True), labels_for(s.ids, lay))


def test_round3_dropout_ema_and_input_noise(tok) -> None:
    """Round 3's training options: dropout acts in training only (and not while its control is 0); the EMA is the
    exact moving average and swaps losslessly; the input noise replaces only eligible timeline positions, only by
    tokens of the same class, never the targets."""
    import copy

    import torch
    from transformers import Qwen3ForCausalLM

    from icutwin.model.data import flat_batches, noise_mask
    from icutwin.model.presets import qwen3_config
    from icutwin.model.train import EMA, add_dropout, corrupt, noise_classes

    v, samples = tok
    torch.manual_seed(0)
    base = Qwen3ForCausalLM(qwen3_config("tiny", len(v.tokens), v.id("[PAD]"), v.id("[BOS]"), v.id("[EOS]")))
    model = copy.deepcopy(base)
    ctl = add_dropout(model, 0.5)
    x = torch.tensor(samples[0].ids.astype(np.int64))[None]
    with torch.no_grad():
        base.eval(), model.eval()
        assert torch.equal(base(input_ids=x).logits, model(input_ids=x).logits)
        base.train(), model.train()
        assert not torch.allclose(base(input_ids=x).logits, model(input_ids=x).logits)
        ctl["p"] = 0.0
        assert torch.equal(base(input_ids=x).logits, model(input_ids=x).logits)
    # EMA: decay warmup min(0.9, (1 + n) / (10 + n)) after n updates
    ema = EMA(base, 0.9)
    manual = {n: p.detach().clone() for n, p in base.named_parameters()}
    for k in range(3):
        with torch.no_grad():
            for p in base.parameters():
                p.add_(0.01 * (k + 1))
        ema.update(base)
        d = min(0.9, (2 + k) / (11 + k))
        manual = {n: d * manual[n] + (1 - d) * p.detach() for n, p in base.named_parameters()}
    assert all(torch.allclose(ema.shadow[n], manual[n], atol=1e-6) for n in manual)
    current = {n: p.detach().clone() for n, p in base.named_parameters()}
    ema.swap(base)
    assert all(torch.allclose(p, manual[n], atol=1e-6) for n, p in base.named_parameters())
    ema.swap(base)
    assert all(torch.equal(p, current[n]) for n, p in base.named_parameters())
    # input noise on a packed batch
    lay, classes = layout(v), noise_classes(v)
    tl, end = v.id("[TL]"), v.id("[END]")
    rng = np.random.default_rng(0)
    items = []
    for i, s in enumerate(samples[:12]):
        ids = transform(s, lay, rng, TransformConfig(goal_fraction=0.5))
        items.append((ids, labels_for(ids, lay), ("mimic", i)))
    b = next(flat_batches(iter(items), 100_000, (classes, tl, end)))
    ok = b["noise_ok"][0].numpy()
    for (ids, _y, _m), a in zip(items, b["cu_seq_lens_q"].numpy()[:-1]):
        o = ok[a:a + len(ids)]
        assert np.array_equal(o, noise_mask(ids, classes, tl, end)) and (classes[ids[o]] > 0).all()
        assert not o[:int(np.flatnonzero(ids == tl)[0]) + 1].any()
        stop = np.flatnonzero(ids == end)
        assert not len(stop) or not o[stop[0]:].any()
    assert ok.any() and not ok[classes[b["input_ids"][0].numpy()] == 0].any()
    inputs = {k: (t.clone() if hasattr(t, "clone") else t) for k, t in b.items() if k not in ("meta", "noise_ok")}
    torch.manual_seed(1)
    replaced, eligible = corrupt(base.eval(), inputs, b["noise_ok"], 1.0, torch.from_numpy(classes))
    before, after = b["input_ids"][0].numpy(), inputs["input_ids"][0].numpy()
    changed = before != after
    assert eligible == ok.sum() and 0 < changed.sum() <= replaced
    assert not changed[~ok].any() and np.array_equal(classes[after[changed]], classes[before[changed]])
    assert torch.equal(inputs["labels"], b["labels"])                     # the targets stay the real tokens
    same = {k: (t.clone() if hasattr(t, "clone") else t) for k, t in b.items() if k not in ("meta", "noise_ok")}
    assert corrupt(base, same, b["noise_ok"], 0.0, torch.from_numpy(classes))[0] == 0
    assert torch.equal(same["input_ids"], b["input_ids"])


def test_muon_adamw_is_the_two_optimizers_in_one(tok) -> None:
    """Amendment 2's optimizer: its steps equal torch's Muon on the hidden matrices plus AdamW on the embeddings and
    norms; a scheduler's learning rate reaches both; its state survives a save and load (the next step is the same)."""
    import copy

    import torch
    from transformers import Qwen3ForCausalLM

    from icutwin.model.optim import split_parameters
    from icutwin.model.presets import qwen3_config
    from icutwin.model.train import TrainConfig, _optimizer

    v, samples = tok
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(qwen3_config("tiny", len(v.tokens), v.id("[PAD]"), v.id("[BOS]"), v.id("[EOS]")))
    ref = copy.deepcopy(model)
    hidden, other, vectors = split_parameters(model)
    names = {id(p): n for n, p in model.named_parameters()}
    assert hidden and all(p.ndim == 2 and "proj" in names[id(p)] for p in hidden)
    assert [names[id(p)] for p in other] == ["model.embed_tokens.weight"] and all(p.ndim == 1 for p in vectors)
    cfg = TrainConfig(tokens_root="", run_dir="", datasets=[], lr=1e-3, optimizer="muon")
    opt = _optimizer(model, cfg)
    h, o, w = split_parameters(ref)
    muon = torch.optim.Muon(h, lr=1e-3, weight_decay=0.1, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw")
    adamw = torch.optim.AdamW([{"params": o + w, "weight_decay": 0.0}], lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    x = torch.tensor(samples[0].ids.astype(np.int64))[None]
    init = [p.detach().clone() for p in model.parameters()]

    def step(m, opts) -> None:
        m(input_ids=x, labels=x).loss.backward()
        for one in opts:
            one.step()
            one.zero_grad(set_to_none=True)

    step(model, [opt])
    step(ref, [muon, adamw])
    # one step: equal up to rounding (Linux's AdamW kernels round the embedding's update differently, 7e-9; later
    # steps amplify that through the next gradients' Newton-Schulz orthogonalization, so they are not compared)
    assert all(torch.allclose(a, b, rtol=0, atol=1e-7) for a, b in zip(model.parameters(), ref.parameters()))
    for _ in range(2):
        step(model, [opt])
    assert all(not torch.equal(a, b) for a, b in zip(model.parameters(), init) if a.ndim == 2)   # every matrix moved
    torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5)
    assert {g["lr"] for g in opt.muon.param_groups + opt.adamw.param_groups} == {5e-4}
    twin_model = copy.deepcopy(model)
    twin = _optimizer(twin_model, cfg)
    twin.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert twin.param_groups[0] is twin.muon.param_groups[0] and twin.param_groups[-1]["lr"] == 5e-4
    step(model, [opt])
    step(twin_model, [twin])
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), twin_model.parameters()))
    with pytest.raises(ValueError):
        _optimizer(model, TrainConfig(tokens_root="", run_dir="", datasets=[], optimizer="sgd"))


def test_dataset_temperature_sampling(tok) -> None:
    """Temperature sampling on tokens: a dataset's share of the sampled tokens is proportional to (its train
    tokens)^alpha, whatever its window lengths (the long-stay datasets would otherwise take the small ones' share)."""
    _v, samples = tok
    samples = samples + [Sample(np.concatenate([s.ids, s.ids[1:]]), s.goal, "hirid", s.continuation)
                         for s in samples[:6]]        # a third dataset with few, long windows
    sampler = TemperatureSampler(samples, alpha=0.5, seed=3)
    assert np.allclose(sampler.share, sampler.natural ** 0.5 / (sampler.natural ** 0.5).sum())
    it = iter(sampler)
    drawn = [next(it) for _ in range(30000)]
    counts = Counter(s.dataset for s in drawn)
    tokens = Counter()
    for s in drawn:
        tokens[s.dataset] += len(s.ids)
    for d, p, share in zip(sampler.datasets, sampler.p, sampler.share):
        assert abs(counts[d] / len(drawn) - p) < 0.02
        assert abs(tokens[d] / sum(tokens.values()) - share) < 0.02
    lengths = {d: np.mean([len(s.ids) for s in sampler.groups[d]]) for d in sampler.datasets}
    assert lengths["hirid"] > 1.5 * lengths["mimic"] and not np.allclose(sampler.p, sampler.share)
    # shuffled passes: within a dataset every window is drawn equally often (once per pass)
    for d in sampler.datasets:
        seen = Counter(id(s) for s in drawn if s.dataset == d)
        assert len(seen) == len(sampler.groups[d]) and max(seen.values()) - min(seen.values()) <= 1
    # processes: the same probabilities, disjoint shards that cover every window
    shards = [TemperatureSampler(samples, alpha=0.5, seed=3 + k, shard=(k, 2)) for k in range(2)]
    assert all(np.allclose(s.p, sampler.p) for s in shards)
    for d in sampler.datasets:
        a, b = ({id(s) for s in sh.groups[d]} for sh in shards)
        assert not a & b and len(a | b) == len(sampler.groups[d])


def test_target_passes_draw_each_dataset_its_passes(tok) -> None:
    """With target passes, a dataset's share of the sampled tokens is its passes times its train tokens: a budget of
    sum(passes x tokens) draws every dataset's windows that many times, whatever the datasets' sizes."""
    from icutwin.model.data import exposure

    _v, samples = tok
    samples = samples + [Sample(np.concatenate([s.ids, s.ids[1:]]), s.goal, "hirid", s.continuation)
                         for s in samples[:6]]
    target = {"default": 3.0, "hirid": 1.5}
    sampler = TemperatureSampler(samples, alpha=0.5, seed=3, passes=target)
    tokens = np.array([sum(len(s.ids) for s in sampler.groups[d]) for d in sampler.datasets], dtype=np.float64)
    want = np.array([target.get(d, 3.0) for d in sampler.datasets]) * tokens
    assert np.allclose(sampler.share, want / want.sum())
    it = iter(sampler)
    drawn = Counter()
    for _ in range(30000):
        s = next(it)
        drawn[s.dataset] += len(s.ids)
    for d, share in zip(sampler.datasets, sampler.share):
        assert abs(drawn[d] / sum(drawn.values()) - share) < 0.02
    passes = {d: drawn[d] / tokens[i] for i, d in enumerate(sampler.datasets)}   # in the ratio of the targets
    assert abs(passes["hirid"] / passes["mimic"] - 0.5) < 0.05 and abs(passes["eicu"] / passes["mimic"] - 1) < 0.05
    sizes = {d: {"tokens": int(tokens[i]), "windows": len(sampler.groups[d])} for i, d in enumerate(sampler.datasets)}
    ex = exposure(sizes, 0.5, float(want.sum()), target)
    assert {d: round(x["passes"], 6) for d, x in ex.items()} == {d: target.get(d, 3.0) for d in ex}


def test_padded_and_flattened_batches(tok) -> None:
    v, samples = tok
    lay = layout(v)
    items = [(s.ids, labels_for(s.ids, lay), s.dataset) for s in samples]
    b = next(pad_batches(iter(items * 10), lay, 200, bucket=16))
    n, m = b["input_ids"].shape
    assert n * m <= 200 or n == 1
    assert ((b["labels"] == IGNORE) | (b["attention_mask"] == 1)).all()            # padding is never a target
    f = next(flat_batches(iter(items * 10), 200))
    pos = f["position_ids"][0].numpy()
    starts = np.flatnonzero(pos == 0)
    assert len(starts) == len(f["meta"]) and starts[0] == 0
    assert (f["labels"][0].numpy()[starts] == IGNORE).all()                        # no prediction across samples
    lengths = np.diff(np.append(starts, len(pos)))                                 # packed-attention kwargs
    assert f["cu_seq_lens_q"].tolist() == [0, *np.cumsum(lengths).tolist()] and f["max_length_q"] == lengths.max()


@pytest.mark.skipif(not __import__("torch").cuda.is_available()
                    or not __import__("torch").backends.cuda.is_flash_attention_available(),
                    reason="PyTorch's varlen FlashAttention kernels (CUDA, not on Windows)")
def test_varlen_attention_matches_sdpa() -> None:
    """The ``varlen`` implementation: a packed row gives each window's SDPA result alone, and right-padded rows
    give SDPA's result on their real tokens (bf16 tolerance); packing without the resets leaks (control)."""
    import torch
    from transformers import Qwen3ForCausalLM

    from icutwin.model import attention  # noqa: F401
    from icutwin.model.presets import qwen3_config

    torch.manual_seed(0)
    ref = Qwen3ForCausalLM(qwen3_config("tiny", 300, 0, 1, 2, attn_implementation="sdpa")).cuda().bfloat16().eval()
    var = Qwen3ForCausalLM(qwen3_config("tiny", 300, 0, 1, 2, attn_implementation="varlen")).cuda().bfloat16().eval()
    var.load_state_dict(ref.state_dict())
    rows = [torch.randint(3, 300, (n,)) for n in (17, 64, 33)]
    hidden = lambda m, ids, **kw: m(input_ids=ids.cuda(), output_hidden_states=True, **kw).hidden_states[-1][0].float()  # noqa: E731
    rel = lambda a, b: float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())  # noqa: E731
    with torch.no_grad():
        alone = torch.cat([hidden(ref, r[None]) for r in rows])
        ids = torch.cat(rows)[None]
        cu = torch.tensor([0, 17, 81, 114], dtype=torch.int32).cuda()
        pos = torch.cat([torch.arange(len(r)) for r in rows])[None].cuda()
        packed = hidden(var, ids, position_ids=pos, cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=64, max_length_k=64)
        leak = hidden(var, ids, position_ids=torch.arange(114)[None].cuda())
        assert rel(packed, alone) < 0.05 and rel(leak[17:], alone[17:]) > 0.5
        padded = torch.zeros((3, 64), dtype=torch.long)
        mask = torch.zeros_like(padded)
        for i, r in enumerate(rows):
            padded[i, :len(r)], mask[i, :len(r)] = r, 1
        out = var(input_ids=padded.cuda(), attention_mask=mask.cuda(), output_hidden_states=True).hidden_states[-1].float()
        want = ref(input_ids=padded.cuda(), attention_mask=mask.cuda(), output_hidden_states=True).hidden_states[-1].float()
        for i, r in enumerate(rows):
            assert rel(out[i, :len(r)], want[i, :len(r)]) < 0.05
    # training's setting: float32 master weights under bf16 autocast (the q/k norms then return float32)
    ref32 = Qwen3ForCausalLM(qwen3_config("tiny", 300, 0, 1, 2, attn_implementation="sdpa")).cuda().eval()
    var32 = Qwen3ForCausalLM(qwen3_config("tiny", 300, 0, 1, 2, attn_implementation="varlen")).cuda().eval()
    var32.load_state_dict(ref32.state_dict())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        alone32 = torch.cat([hidden(ref32, r[None]) for r in rows])
        packed32 = hidden(var32, ids, position_ids=pos, cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=64,
                          max_length_k=64)
    assert rel(packed32, alone32) < 0.05


def test_gradient_accumulation_matches_the_full_batch_mean() -> None:
    """Micro-batches of different sizes accumulate to the gradient of the mean over all their target tokens, as
    one batch would give (not a mean of per-batch means), and the step loss is that mean."""
    import torch
    from accelerate import Accelerator
    from transformers import Qwen3ForCausalLM

    from icutwin.model.presets import qwen3_config
    from icutwin.model.train import accumulate_gradients

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(qwen3_config("tiny", 300, 0, 1, 2))
    rows = [torch.randint(3, 300, (n,)) for n in (12, 40, 40, 33)]

    def batch(rs):
        m = max(len(r) for r in rs)
        ids = torch.zeros((len(rs), m), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, r in enumerate(rs):
            ids[i, :len(r)], mask[i, :len(r)] = r, 1
        labels = torch.where(mask == 1, ids, torch.full_like(ids, IGNORE))
        labels[:, 0] = IGNORE
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    full = batch(rows)
    model(**full).loss.backward()
    want = [p.grad.clone() for p in model.parameters()]
    want_loss = float(model(**full).loss.detach())
    model.zero_grad()
    acc = Accelerator(cpu=True, gradient_accumulation_steps=2)
    loss = accumulate_gradients(acc, model, [batch(rows[:1]), batch(rows[1:])])   # 11 and 109 target tokens
    assert float(loss) == pytest.approx(want_loss, rel=1e-5)
    for g, w in zip((p.grad for p in model.parameters()), want):
        assert torch.allclose(g, w, rtol=1e-4, atol=1e-6)


def _tokens(tmp_path: Path) -> Path:
    """Token shards of the fixture MEDS (one dataset, every split) from the S5 CLI."""
    from icutwin.tokenization.cli import encode_all, fit_vocab

    meds_root = tmp_path / "meds"
    _write_meds(meds_root / "mimic", SPLITS)
    tokens = tmp_path / "tokens"
    v = fit_vocab(meds_root, tmp_path / "no_derived", ["mimic"], tokens)
    encode_all(v, meds_root, tmp_path / "no_derived", ["mimic"], ["train", "tuning", "held_out"], tokens)
    return tokens


def test_data_parallel_smoke(tmp_path: Path) -> None:
    """The smoke test under two CPU processes (gloo, ``ddp_worker.py``): the accumulated gradient is that of the
    mean over both processes' target tokens through the all-reduce; the scheduler steps once per optimizer step;
    each process draws its own shard with its own seed and no window twice across them; the resumed run draws the
    same windows and keeps the replicas equal; the metrics are written once."""
    import json
    import os
    import socket
    import subprocess
    import sys

    tokens = _tokens(tmp_path)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "WORLD_SIZE": "2",
           "LOCAL_WORLD_SIZE": "2", "USE_LIBUV": "0", "CUDA_VISIBLE_DEVICES": "-1", "ACCELERATE_USE_CPU": "true",
           "OMP_NUM_THREADS": "2"}
    out = tmp_path / "ddp"
    procs = [subprocess.Popen([sys.executable, str(Path(__file__).with_name("ddp_worker.py")), str(tokens), str(out)],
                              env={**env, "RANK": str(k), "LOCAL_RANK": str(k)}, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) for k in range(2)]
    logs = [p.communicate(timeout=900)[0].decode(errors="replace") for p in procs]
    assert all(p.returncode == 0 for p in procs), "\n".join(log[-3000:] for log in logs)
    r0, r1 = (json.loads((out / f"rank{k}.json").read_text()) for k in range(2))
    assert r0["passed"] and r0["processes"] == 2, r0["checks"]
    c = r0["checks"]
    assert c["token_normalized_gradient"]["processes"] == 2 and c["token_normalized_gradient"]["passed"]
    assert c["data_partition"]["shards"] == [[0, 2], [1, 2]] and len(set(c["data_partition"]["seeds"])) == 2
    assert c["data_partition"]["drawn_by_two_processes"] == 0 and min(c["data_partition"]["windows_drawn"]) > 0
    assert c["resume"]["same_windows"] and c["learning_rate"]["passed"] and c["checkpoint"]["files"]
    assert r1["losses"] == r0["losses"]                                          # the processes agree
    metrics = (out / "run" / "smoke" / "uninterrupted" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert sum('"loss"' in line for line in metrics) == 6                         # one writer


def test_smoke_passes_and_catches_a_broken_resume(tmp_path: Path, monkeypatch) -> None:
    """The smoke test on one process, and its negative control: a resume that does not replay the data stream
    (another seed) fails the resume check."""
    from dataclasses import replace

    import icutwin.model.smoke as sm
    from icutwin.model.train import TrainConfig

    tokens = _tokens(tmp_path)
    cfg = TrainConfig(tokens_root=str(tokens), run_dir=str(tmp_path / "run"), datasets=["mimic"], preset="tiny",
                      max_steps=100, tokens_per_batch=512, grad_accum=2, warmup_steps=1, lr=1e-3)
    ok = sm.smoke(cfg, steps=6, resume_at=3)
    assert ok["passed"], ok["checks"]
    assert ok["checks"]["token_normalized_loss"]["rel_error"] < 1e-3 and ok["checks"]["checkpoint"]["passed"]
    # round 3's dropout and EMA on the same path: the loss check (dropout off for its step), an exact resume of
    # the average, the final weights the lower tuning NLL of the two
    reg = sm.smoke(replace(cfg, run_dir=str(tmp_path / "reg"), dropout=0.1, ema_decay=0.9), steps=6, resume_at=3)
    assert reg["passed"], reg["checks"]
    assert reg["checks"]["ema_resume"]["passed"] and reg["checks"]["token_normalized_loss"]["rel_error"] < 1e-3
    final = tmp_path / "reg" / "smoke" / "uninterrupted" / "final"
    assert (final / "hf").exists() and ((final / "hf_ema").exists() != (final / "hf_raw").exists())
    # amendment 2's Muon: the same checks, its nested optimizer state through Accelerate's checkpoint and resume
    mu = sm.smoke(replace(cfg, run_dir=str(tmp_path / "muon"), optimizer="muon"), steps=6, resume_at=3)
    assert mu["passed"], mu["checks"]
    assert mu["checks"]["resume"]["max_rel_step_loss_diff"] < 1e-6
    real = sm.train
    monkeypatch.setattr(sm, "train", lambda c: real(replace(c, seed=c.seed + 1) if c.resume_from else c))
    broken = sm.smoke(replace(cfg, run_dir=str(tmp_path / "broken")), steps=6, resume_at=3)
    assert not broken["passed"] and not broken["checks"]["resume"]["same_windows"]


def test_eval_subset_is_fixed_and_spans_datasets(tok) -> None:
    from icutwin.model.train import eval_subset

    _v, samples = tok
    a, b = eval_subset(samples, 10), eval_subset(samples, 10)
    assert len(a) == 10 and [id(s) for s in a] == [id(s) for s in b]
    assert {s.dataset for s in a} == {s.dataset for s in samples}
    assert eval_subset(samples, None) is samples


def test_parameter_counts_and_a_tiny_model_learns() -> None:
    import torch
    from transformers import Qwen3ForCausalLM

    from icutwin.model.presets import qwen3_config
    for preset in ("tiny", "pilot"):
        model = Qwen3ForCausalLM(qwen3_config(preset, 300, 0, 1, 2))
        assert sum(p.numel() for p in model.parameters()) == n_params(preset, 300)["total"]
    s, m = n_params("S", 11284)["total"], n_params("M", 11284)["total"]
    assert 55e6 < s < 70e6 and 105e6 < m < 125e6
    with torch.device("meta"):         # the server presets, counted exactly without allocating them
        for preset in ("M", "ML"):
            big = Qwen3ForCausalLM(qwen3_config(preset, 25566, 0, 1, 2))
            assert sum(p.numel() for p in big.parameters()) == n_params(preset, 25566)["total"]
    assert 265e6 < n_params("ML", 25566)["total"] < 275e6
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(qwen3_config("tiny", 300, 0, 1, 2))
    x = torch.randint(3, 300, (2, 48))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = None
    for _ in range(80):
        loss = model(input_ids=x, labels=x).loss
        first = first if first is not None else float(loss.detach())
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert float(loss.detach()) < 0.2 * first


def test_training_loop_checkpoints_and_resumes(tmp_path: Path) -> None:
    """The S6 loop end to end on CPU: shards from the S5 CLI, a few steps, evaluation per dataset and token
    family, a safetensors checkpoint, and a resumed run that replays the seeded data stream."""
    import json

    import pyarrow.parquet as pq

    from icutwin.model.train import TrainConfig, train

    tokens = _tokens(tmp_path)

    from icutwin.model.data import load_samples
    from icutwin.tokenization.vocab import Vocab
    naive = [(w, ds) for ds in ["mimic"] for split in ["train"]
             for ws in pq.read_table(tokens / ds / f"{split}.parquet").column("windows").to_pylist() for w in ws]
    loaded = load_samples(tokens, ["mimic"], "train", Vocab.load(tokens / "vocab.json"))
    assert [(s.ids.tolist(), s.dataset) for s in loaded] == naive and all(s.goal is not None for s in loaded)
    cfg = dict(tokens_root=str(tokens), datasets=["mimic"], preset="tiny", max_steps=4, tokens_per_batch=512,
               grad_accum=2, warmup_steps=1, eval_every=2, log_every=1, checkpoint_every=2, lr=1e-3,
               gpu_memory_fraction=0.5)
    out = train(TrainConfig(run_dir=str(tmp_path / "run"), **cfg))
    assert out["steps"] == 4 and set(out["final_eval"]["by_dataset"]) == {"mimic"} and out["processes"] == 1
    assert out["final_eval"]["target_tokens"] > 0 and "interval" in out["final_eval"]["by_family"]
    # the scheduler steps once per optimizer step: warmup 1, then cosine from 1e-3 to 1e-4 over the 3 others
    lrs = [h["lr"] for h in out["history"] if "lr" in h]
    assert lrs == pytest.approx([1e-3, 7.75e-4, 3.25e-4, 1e-4])
    ckpt = tmp_path / "run" / "step_2"
    assert (ckpt / "hf" / "model.safetensors").exists() and json.loads((ckpt / "trainer_state.json").read_text())["step"] == 2
    resumed = train(TrainConfig(run_dir=str(tmp_path / "resumed"), resume_from=str(ckpt), **cfg))
    assert resumed["final_eval"]["nll"] == pytest.approx(out["final_eval"]["nll"], rel=1e-4)
    # padding-free packing keeps windows apart (SDPA's mask from the position resets here; FlashAttention-2 on the
    # server), and the check's negative control (no resets) detects a leak
    from icutwin.model.pilot import attention_parity
    parity = attention_parity(tokens, ["mimic"], "sdpa", device="cpu", split="train")
    assert parity["windows"] >= 2 and parity["passed"], parity


@pytest.mark.skipif(not __import__("torch").cuda.is_available()
                    or not __import__("torch").backends.cuda.is_flash_attention_available(),
                    reason="the server's path: CUDA, varlen kernels, inductor (not on Windows)")
def test_compiled_packed_training_on_gpu(tmp_path: Path) -> None:
    """The configured run's path: packed batches, varlen attention, torch.compile; its checkpoint holds plain
    parameter names (no compile prefix), loads as an HF model, and a resumed run continues it."""
    from safetensors.torch import load_file

    from icutwin.model.train import TrainConfig, train

    tokens = _tokens(tmp_path)
    cfg = dict(tokens_root=str(tokens), datasets=["mimic"], preset="tiny", max_steps=4, tokens_per_batch=512,
               grad_accum=2, warmup_steps=1, eval_every=0, log_every=1, checkpoint_every=2, lr=1e-3,
               attn_implementation="varlen", compile=True)
    out = train(TrainConfig(run_dir=str(tmp_path / "run"), **cfg))
    weights = load_file(str(tmp_path / "run" / "final" / "hf" / "model.safetensors"))
    assert weights and not any("_orig_mod" in k for k in weights)
    resumed = train(TrainConfig(run_dir=str(tmp_path / "resumed"), resume_from=str(tmp_path / "run" / "step_2"), **cfg))
    assert resumed["final_eval"]["nll"] == pytest.approx(out["final_eval"]["nll"], rel=1e-2)
    # round 3's options on the server's path: compiled dropout hooks, the EMA, the input noise's second forward pass
    r3 = train(TrainConfig(run_dir=str(tmp_path / "r3"), dropout=0.1, ema_decay=0.9, input_noise=0.15,
                           input_noise_ramp=0.25, **{**cfg, "max_steps": 6}))
    assert any(h.get("noise_share", 0) > 0 for h in r3["history"]) and r3["final_weights"] in ("ema", "raw")
    final = tmp_path / "r3" / "final"
    assert (final / "hf" / "model.safetensors").exists() and ((final / "hf_ema").exists() != (final / "hf_raw").exists())


def test_init_from_loads_a_finished_models_weights_exactly(tok, tmp_path: Path) -> None:
    """Amendment 17: a run started from a finished model's weights gives that model's logits before any step."""
    import torch
    from transformers import Qwen3ForCausalLM

    from icutwin.model.presets import qwen3_config
    from icutwin.model.train import build_model, init_weights

    v, samples = tok
    src = Qwen3ForCausalLM(qwen3_config("tiny", len(v.tokens), v.id("[PAD]"), v.id("[BOS]"), v.id("[EOS]"),
                                        attn_implementation="sdpa")).eval()
    src.save_pretrained(tmp_path / "hf")
    dst = init_weights(build_model("tiny", v, "sdpa"), tmp_path / "hf").eval()
    ids = torch.as_tensor(samples[0].ids[None, :].astype(np.int64))
    with torch.no_grad():
        assert torch.equal(src(ids).logits, dst(ids).logits)
    other = Qwen3ForCausalLM(qwen3_config("tiny", len(v.tokens) + 8, 0, 1, 2, attn_implementation="sdpa"))
    other.save_pretrained(tmp_path / "other")
    with pytest.raises(ValueError):
        init_weights(build_model("tiny", v, "sdpa"), tmp_path / "other")
