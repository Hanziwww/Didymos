"""Production I/O (S7, ``factory.stream``): the trajectory-episode stream gives the same episode for an index whatever
the number of workers, the reference collate's offsets give every episode back, a shard round trip keeps every array
and the contract, and a dump is resumable with checksums (a store from real held-out token sequences; skipped
without them)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from icutwin.factory import contracts, store as fst, stream as fs

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "full_r4"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not (T / "vocab.json").exists() or not (T / "mimic" / "held_out.parquet").exists():
        pytest.skip("the full r4 tokens")
    rows = []
    for ds in ("mimic", "eicu"):
        t = pq.read_table(T / ds / "held_out.parquet", columns=["stay_id", "ids"]).slice(0, 60).to_pylist()
        rows += [{"stay_id": r["stay_id"], "ids": r["ids"], "split": "train"} for r in t]
    root = tmp_path_factory.mktemp("io")
    fst.build(rows, T / "vocab.json", root / "s", "seg_0000", {"kind": "test"}, workers=1, chunk=60)
    st = fst.Store(root / "s")
    st.channels.stats = fst.channel_stats(st)
    st.channels.save(root / "s" / "channels.json")
    return root / "s"


def _eq(a: dict, b: dict) -> bool:
    if a["recipe"] != b["recipe"]:
        return False
    for part in ("context", "query"):
        for k, x in a[part].items():
            y = b[part][k]
            if isinstance(x, np.ndarray):
                if x.shape != np.asarray(y).shape or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
                    return False
            elif x != y:
                return False
    return True


def test_stream_is_worker_independent(built) -> None:
    one = fs.EpisodeStream(built, T / "vocab.json", stop=12)
    by_one = {i: one.sampler().episode(i) for i in one.iter_indices()}
    three = [fs.EpisodeStream(built, T / "vocab.json", stop=12) for _ in range(3)]
    for w, s in enumerate(three):
        for i in s.iter_indices(w, 3):
            assert _eq(by_one[i], s.sampler().episode(i)), i
    eps = list(fs.EpisodeStream(built, T / "vocab.json", start=5, stop=8))     # resumable from any index
    assert [e["recipe"]["index"] for e in eps] == [5, 6, 7] and _eq(eps[0], by_one[5])


def test_collate_gives_every_episode_back(built) -> None:
    s = fs.EpisodeStream(built, T / "vocab.json")
    eps = [s.sampler().episode(i) for i in range(5)]
    batch = fs.collate(eps)
    for part in ("context", "query"):
        pat = batch[f"{part}_patients"]
        assert pat[-1] == sum(len(e[part]["ev_off"]) - 1 for e in eps)
        off = batch[f"{part}.ev_off"]
        assert len(off) == pat[-1] + 1 and (np.diff(off) >= 0).all()
        for j, e in enumerate(eps):
            a, b = off[pat[j]], off[pat[j + 1]]
            np.testing.assert_array_equal(batch[f"{part}.ev_t"][a:b], e[part]["ev_t"])
            lo = batch[f"{part}.lab_off"]
            np.testing.assert_array_equal(batch[f"{part}.lab_t"][lo[pat[j]]:lo[pat[j + 1]]], e[part]["lab_t"])


def test_shards_round_trip_and_dumps(built, tmp_path: Path) -> None:
    s = fs.EpisodeStream(built, T / "vocab.json")
    eps = [s.sampler().episode(i) for i in range(6)]
    sha = fs.write_shard(eps, tmp_path / "x.npz")
    back = fs.read_shard(tmp_path / "x.npz")
    assert len(sha) == 64 and len(back) == 6
    for a, b in zip(eps, back):
        assert _eq(a, b) and not contracts.episode_failures(b)
    man = fs.dump(s, list(range(10)), tmp_path / "d", shard_size=4, name="val")
    assert len(man["shards"]) == 3 and sum(x["episodes"] for x in man["shards"].values()) == 10
    again = fs.dump(s, list(range(10)), tmp_path / "d", shard_size=4, name="val")
    assert again["shards"] == man["shards"]
