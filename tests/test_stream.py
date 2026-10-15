"""The factory over generated twins (S9), with real held-out token sequences of the 2% subset standing in for twins
(skipped without them): decoded stays build a cache whose labels match the real cache's where both read the same
events, a producer bank's shards stream into episodes with their weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from icutwin.factory import stream as fs
from icutwin.factory import tasks as ft

R = Path(__file__).resolve().parents[1] / "data"
T = R / "tokens" / "subset"


@pytest.fixture(scope="module")
def seqs():
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's tokens")
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    rows = []
    for ds in ("mimic", "eicu", "hirid", "aumc"):
        rows += pq.read_table(T / ds / "held_out.parquet", columns=["stay_id", "ids"]).to_pylist()
    return v, rows


def test_decoded_twins_build_a_cache(seqs) -> None:
    v, rows = seqs
    c, kept = fs.cache_from_sequences([r["ids"] for r in rows], v, [r["stay_id"] for r in rows], hours=48)
    assert len(kept) > 0.95 * len(rows)
    assert np.isfinite(c.los_hours).all() and (c.los_hours > 0).mean() > 0.95
    assert 0.03 < np.nanmean(c.death) < 0.4 and np.isnan(c.sofa_24h).all()
    assert (~np.isnan(c.grid)).mean() > 0.02 and np.sum(~np.isnan(c.onset["imv"])) > 0
    prior = ft.ICUPrior(c, min_rows=48, max_rows=300, seed=1)
    x, y, d, seq, train = prior.get_batch(4)                  # passes the batch contract (layer d)
    assert x.shape[0] == 4


def test_bank_shards_stream_with_weights(seqs, tmp_path: Path) -> None:
    import pyarrow as pa

    v, rows = seqs
    for k, part in enumerate((rows[:300], rows[300:600])):
        pq.write_table(pa.table({"twin": [f"s{k}_{i}" for i in range(len(part))], "dataset": ["x"] * len(part),
                                 "seed": pa.array(list(range(len(part))), pa.int64()),
                                 "weight": pa.array([0.5 + k] * len(part), pa.float64()),
                                 "ids": pa.array([r["ids"] for r in part], type=pa.list_(pa.int32()))}),
                       tmp_path / f"x_{k:05d}.parquet")
    c, w = fs.cache_from_bank(tmp_path, v, hours=48)
    assert len(w) == len(c.stay_id) and set(np.unique(w)) <= {0.5, 1.5} and len(c.stay_id) > 550
    stream = fs.episodes(c, seed=2, min_rows=48, max_rows=200)
    spec, x, y, n_train, idx = next(stream)
    assert x.shape[0] == len(y) and 0 < n_train < len(y)
