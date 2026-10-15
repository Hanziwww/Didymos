"""Task realism (S8 *Tasks*) on the 2% subset's real stays (skipped without them): two real caches (train and
held-out stays) agree task by task (positive control); a cache whose outcomes are scrambled across stays does not
(negative control: its labels lose their relation to the features)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from icutwin.eval import realism as rl
from icutwin.factory import stream as fs

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"


@pytest.fixture(scope="module")
def caches():
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's tokens")
    import pyarrow.parquet as pq

    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    held, train = [], []
    for ds in ("mimic", "eicu"):
        held += pq.read_table(T / ds / "held_out.parquet", columns=["ids"]).column(0).to_pylist()
        train += pq.read_table(T / ds / "train.parquet", columns=["ids"]).column(0).to_pylist()[:2500]
    real, _ = fs.cache_from_sequences(held, v, hours=48)
    twin, _ = fs.cache_from_sequences(train, v, hours=48, concepts=real.concepts)
    return real, twin


def test_positive_and_negative_controls(caches) -> None:
    import copy

    real, twin = caches
    pos = rl.compare(real, twin, n_tasks=20, rows=300, seed=0)
    assert pos["tasks"] >= 10
    scr = copy.deepcopy(twin)
    rng = np.random.default_rng(0)
    p = rng.permutation(len(scr.los_hours))
    scr.los_hours, scr.death = scr.los_hours[p], scr.death[p]
    scr.onset = {k: x[p] for k, x in scr.onset.items()}
    scr.grid = scr.grid[rng.permutation(len(scr.los_hours))]
    neg = rl.compare(real, scr, n_tasks=20, rows=300, seed=0)
    ps, ns = pos["summary"], neg["summary"]
    assert ps["mi_rank_corr"] > ns["mi_rank_corr"]
    assert ps["tstr_over_trtr"] is not None and ps["tstr_over_trtr"] > 0.6
    assert ns["tstr_over_trtr"] is None or ns["tstr_over_trtr"] < ps["tstr_over_trtr"] - 0.3
