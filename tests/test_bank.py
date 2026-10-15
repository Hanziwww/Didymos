"""The bank's plan and filters (S7): quotas and weights, exact and near duplicates, the novelty filter (copies of
training stays are dropped, fresh stays kept), on the 2% subset's real stays where needed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from icutwin.generate import bank

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"


def test_plan_quotas_and_importance_weights() -> None:
    natural = {"eicu": 600, "mimic": 300, "inspire": 100}
    p = bank.GenerationPlan.from_mix(natural, 1000)
    assert p.quotas == {"eicu": 600, "inspire": 100, "mimic": 300} and all(abs(w - 1) < 1e-9 for w in p.weights.values())
    f = bank.GenerationPlan.from_mix(natural, 1000, "flattened", power=0.5)
    assert sum(f.quotas.values()) == 1000 and f.quotas["inspire"] > 100 and f.quotas["eicu"] < 600
    assert f.weights["inspire"] < 1 < f.weights["eicu"]            # p_natural / p_plan
    u = bank.GenerationPlan.from_mix(natural, 999, "uniform")
    assert sorted(u.quotas.values()) == [333, 333, 333]


def test_duplicates_and_novelty() -> None:
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's tokens")
    import torch

    from icutwin.eval import privacy as pv
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    event = np.array([m.get("cls") == "event" for m in v.meta])
    train = pv.first_windows(T, "eicu", "train")
    held = pv.first_windows(T, "eicu", "held_out")
    rng = np.random.default_rng(0)
    fresh = held[:200]
    copies = [train[i] for i in rng.choice(len(train), 50, replace=False)]
    seqs = fresh + [fresh[3], fresh[7]]                                   # two exact repeats
    assert bank.exact_duplicates(seqs).tolist() == [False] * 200 + [True, True]
    near = list(fresh[:50])
    s = np.array(fresh[0])
    s[-3] = s[-2]                                                         # one token changed
    near.append(s)
    nd = bank.near_duplicates(near, event)
    assert nd[-1] and not nd[:50].any()
    special = {i for i, m in enumerate(v.meta) if m.get("cls") == "special"}
    keep = bank.novelty(fresh + copies, ["eicu"] * 250, train, held[200:], ["eicu"] * len(held[200:]), len(v.tokens),
                        special, torch.device("cpu"))
    assert not keep[200:].any()                   # copies of training stays: dropped
    assert keep[:200].mean() > 0.8                # held-out stays: mostly kept (5% by construction of the quantile)
    # the prepared reference (shared by a bank's shards) decides as the one-call function, shard by shard
    nov = bank.Novelty(train, held[200:], ["eicu"] * len(held[200:]), len(v.tokens), special, torch.device("cpu"))
    parts = [nov(x, ["eicu"] * len(x)) for x in (fresh[:120], fresh[120:] + copies)]
    assert np.array_equal(np.concatenate(parts), keep)
    # the sparse reference (one 32 GB GPU beside the engines): the same distances, saved once and loaded per process
    sp = bank.Novelty(train, held[200:], ["eicu"] * len(held[200:]), len(v.tokens), special, torch.device("cpu"),
                      sparse=True)
    assert np.allclose(sp.distances(fresh + copies), nov.distances(fresh + copies), atol=1e-5)
    assert abs(sp.thr["eicu"] - nov.thr["eicu"]) < 1e-5 and np.array_equal(sp(fresh + copies, ["eicu"] * 250), keep)
    path = Path(__file__).resolve().parents[1] / ".tmp" / "novelty_test.npz"
    path.parent.mkdir(exist_ok=True)
    sp.save(path)
    back = bank.Novelty.load(path, torch.device("cpu"))
    assert back.thr == sp.thr and np.array_equal(back(fresh + copies, ["eicu"] * 250), keep)
    path.unlink()
