"""The stay store (S1, ``factory.store`` and ``factory.channels``) on real held-out token sequences of the full r4
tokens standing in for twins (skipped without them): the channel table, the exact event round trip (values, times,
bins against the tokens), state intervals, parity with the 72-hour cache (``factory.cache.build``) including the
derived concepts, and a build independent of its worker count and byte-reproducible."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from icutwin.factory import store as fst
from icutwin.factory import twins
from icutwin.factory.channels import ACTIONS, STATE_CHANNELS, Channels

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "full_r4"


@pytest.fixture(scope="module")
def data():
    if not (T / "vocab.json").exists() or not (T / "mimic" / "held_out.parquet").exists():
        pytest.skip("the full r4 tokens")
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    rows = []
    for ds in ("mimic", "eicu", "hirid", "aumc"):
        t = pq.read_table(T / ds / "held_out.parquet", columns=["stay_id", "ids"]).slice(0, 15).to_pylist()
        rows += [{"stay_id": f"{ds}:{r['stay_id']}", "ids": r["ids"], "split": fst.twin_split(f"{ds}:{r['stay_id']}"),
                  "weight": 1.0, "cell": "base"} for r in t]
    return v, rows


@pytest.fixture(scope="module")
def built(data, tmp_path_factory):
    v, rows = data
    root = tmp_path_factory.mktemp("store")
    fst.build(rows, T / "vocab.json", root / "s", "seg_0000", {"kind": "test"}, derived=True, workers=1, chunk=30)
    return fst.Store(root / "s")


def test_channels(data) -> None:
    v, _rows = data
    ch = Channels.from_vocab(v)
    assert ch.names == sorted(ch.names) and len(set(ch.names)) == len(ch.names)
    assert all(s in ch.index for s in STATE_CHANNELS)
    cid, act = ch.of("MED_CON//norepinephrine//mcg/kg/min//start//iv")
    assert ch.names[cid] == "MED_CON//norepinephrine//mcg/kg/min//iv" and act == 1 + ACTIONS.index("start")
    assert ch.kind[cid] == "numeric" and ch.edges[cid] and ch.group[cid] == "infusions"
    assert ch.of("VITAL//heart_rate")[1] == 0 and ch.physiological[ch.index["VITAL//heart_rate"]]
    assert not ch.is_input(ch.index["HOSP//discharge_category//home"])
    assert 150 < sum(ch.physiological) < 260


def test_round_trip_and_bins(data, built) -> None:
    v, rows = data
    ch = built.channels
    for i, r in enumerate(rows):
        s = twins.decode_features(r["ids"], v, r["stay_id"])
        ev = built.events(i)
        kept = [e for e in s.timeline if ch.of(e[2])[0] >= 0]
        assert len(ev["t"]) == len(kept)
        for k, (t, _rk, code, num, _x) in enumerate(kept):
            assert ev["t"][k] == math.floor((t - s.icu_in).total_seconds() / 60)
            assert ch.of(code) == (int(ev["c"][k]), int(ev["f"][k]) >> 2)
            if num is None:
                assert np.isnan(ev["v"][k]) and ev["bin"][k] == fst.NO_BIN
            else:
                assert ev["v"][k] == np.float32(num)
        # bins: the tokens' own bins, in order, for the timeline's valued events
        body = r["ids"][r["ids"].index(v.id("[TL]")) + 1:r["ids"].index(v.id("[END]"))]
        tok_bins = [v.meta[x]["bin"] for x in body
                    if v.meta[x].get("cls") == "event" and v.meta[x].get("kind") == "bin"]
        store_bins = [int(b) for b in ev["bin"] if b != fst.NO_BIN]
        assert store_bins == tok_bins
        row = built.stays.iloc[i]
        assert row["stay_id"] == r["stay_id"]
        assert row["los_min"] == math.floor((s.icu_out - s.icu_in).total_seconds() / 60)


def test_intervals(built) -> None:
    ch = built.channels
    seen_inf = 0
    for i in range(len(built)):
        iv, ev = built.intervals(i), built.events(i)
        assert np.all(iv["e"] > iv["s"])
        for c in np.unique(iv["c"]):
            m = iv["c"] == c
            s, e = iv["s"][m], iv["e"][m]
            assert np.all(s[1:] >= e[:-1]), "intervals of one channel overlap"
            if ch.group[c] == "infusions":
                seen_inf += 1
                starts = set(ev["t"][(ev["c"] == c) & np.isin(fst.acts(ev["f"]), [1, 2, 3])].tolist())
                assert set(s.tolist()) <= starts
    assert seen_inf > 0


def test_parity_with_the_cache(data, built) -> None:
    from icutwin.factory import cache

    v, rows = data
    stays = [twins.decode_features(r["ids"], v, r["stay_id"]) for r in rows]
    labs = [twins.decode_labels(r["ids"], v, r["stay_id"]) for r in rows]
    from icutwin.generate import clif_out

    der = clif_out.derived(labs, Path(built.root) / "_parity")
    c = cache.build(stays, der, hours=72)
    ch = built.channels
    cols = [(j, ch.index[code]) for j, code in enumerate(c.concepts) if code in ch.index and ch.of(code)[1] == 0]
    assert len(cols) > 40
    for i in range(len(rows)):
        ev = built.events(i)
        grid = np.full((72, len(c.concepts)), np.nan, dtype=np.float32)
        for t, cid, x in zip(ev["t"], ev["c"], ev["v"]):
            h = int(t) // 60
            if 0 <= h < 72 and not np.isnan(x):
                for j, cc in cols:
                    if cc == cid:
                        grid[h, j] = x
        js = [j for j, _ in cols]
        np.testing.assert_array_equal(np.isnan(grid[:, js]), np.isnan(c.grid[i][:, js]))
        np.testing.assert_allclose(np.nan_to_num(grid[:, js]), np.nan_to_num(c.grid[i][:, js]))
        dv = built.derived(i)
        for ep in ("imv", "vasopressor", "rrt"):
            starts = dv["s"][(dv["k"] == fst.DK[f"ep:{ep}"]) & (dv["s"] >= 0)]
            on = c.onset[ep][i]
            if np.isnan(on):
                assert len(starts) == 0 or starts.min() / 60 > c.los_hours[i] + 48     # outside the cache's window
            else:
                assert abs(starts.min() / 60 - on) < 1 / 60 + 1e-9


def test_build_is_reproducible_and_worker_independent(data, tmp_path: Path) -> None:
    v, rows = data
    a = fst.build(rows, T / "vocab.json", tmp_path / "a", "seg_0000", {"kind": "test"}, derived=False, workers=1,
                  chunk=20)
    b = fst.build(rows, T / "vocab.json", tmp_path / "b", "seg_0000", {"kind": "test"}, derived=False, workers=3,
                  chunk=20)
    assert a["files"] == b["files"] and a["stays"] == len(rows)
    st = fst.Store(tmp_path / "a")
    stats = fst.channel_stats(st)
    hr = stats["VITAL//heart_rate"]
    assert 40 < hr["p05"] < hr["median"] < hr["p95"] < 160


def test_key_major_index_matches_the_fallback(data, tmp_path: Path) -> None:
    """A channel's (or a derived concept's) rows read from the key-major copies equal those of the per-segment sort a
    store without them uses; ``index_segments`` adds them to such a store, and its manifest records them."""
    import json
    import shutil

    v, rows = data
    for k, part in enumerate((rows[:30], rows[30:])):                       # two segments
        fst.build(part, T / "vocab.json", tmp_path / "s", f"seg_{k:04d}", {"kind": "test"}, derived=True,
                  workers=1, chunk=30)
    st = fst.Store(tmp_path / "s")
    hr = st.channels.index["VITAL//heart_rate"]
    want = {"hr": st.channel_events(hr), "kdigo": st.derived_events("kdigo"), "imv": st.derived_events("ep:imv")}
    assert len(want["hr"]["t"]) > 0
    old = tmp_path / "old"
    shutil.copytree(tmp_path / "s", old)
    for seg in ("seg_0000", "seg_0001"):
        for p in (old / seg).glob("*x_*.npy"):
            p.unlink()
    st_old = fst.Store(old)
    assert "evx_ptr" not in st_old.segments[0].arr
    got = {"hr": st_old.channel_events(hr), "kdigo": st_old.derived_events("kdigo"),
           "imv": st_old.derived_events("ep:imv")}
    assert fst.index_segments(old) == ["seg_0000", "seg_0001"]
    again = fst.Store(old)
    got2 = {"hr": again.channel_events(hr), "kdigo": again.derived_events("kdigo"),
            "imv": again.derived_events("ep:imv")}
    for name in want:
        for f in want[name]:
            np.testing.assert_array_equal(np.asarray(want[name][f]), np.asarray(got[name][f]))
            np.testing.assert_array_equal(np.asarray(want[name][f]), np.asarray(got2[name][f]))
    man = json.loads((old / "manifest.json").read_text())
    assert all("evx_ptr.npy" in s["files"] and "dvx_stay.npy" in s["files"] for s in man["segments"])
