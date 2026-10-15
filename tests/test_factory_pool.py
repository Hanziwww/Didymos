"""The label sampler's pool (``factory.pool``) on a store built from real held-out token sequences (skipped without
them): candidates' attributes, near-duplicate clusters, raking toward P* within the caps, per-cell effective sizes,
and episodes drawing pooled and fresh programs that pass the contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from icutwin.factory import contracts, episodes, pool, programs as fpr, store as fst

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "full_r4"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not (T / "vocab.json").exists() or not (T / "mimic" / "held_out.parquet").exists():
        pytest.skip("the full r4 tokens")
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    rows = []
    for ds in ("mimic", "eicu", "aumc", "hirid"):
        t = pq.read_table(T / ds / "held_out.parquet", columns=["stay_id", "ids"]).slice(0, 50).to_pylist()
        rows += [{"stay_id": r["stay_id"], "ids": r["ids"], "split": "train"} for r in t]
    root = tmp_path_factory.mktemp("pool")
    fst.build(rows, T / "vocab.json", root / "s", "seg_0000", {"kind": "test"}, workers=1, chunk=100)
    st = fst.Store(root / "s")
    st.channels.stats = fst.channel_stats(st)
    tg = pool.load_targets()
    progs = fpr.catalog(st.channels) + pool.f6_candidates(st.channels, horizons=(240, 1440, 10080))[:3000] \
        + pool.fh_candidates(st.channels, 150, 0, tg)
    df, panel = pool.build(st, targets=tg, programs=progs, n_stays=200)
    return v, st, tg, df, panel


def test_attributes_and_clusters(built) -> None:
    _v, _st, _tg, df, panel = built
    assert set(df["group"]) == {"clinical", "threshold", "values", "functional"}
    assert df["domain"].isin(list(pool.load_targets()["axes"]["domain"])).all()
    b = df[df["label"] == "binary"]
    prev = b["prevalence"].dropna()
    assert ((prev >= 0) & (prev <= 1)).all() and len(prev) > 0.9 * len(b)
    # F6 at the two longest horizons often label the same rows: they must share clusters more than chance
    f6 = df[df["family"] == "F6"]
    assert f6["cluster"].nunique() < len(f6)
    # one cluster per identical positive set
    sig = df["sig"].map(lambda s: s if s is not None else b"")
    for s, g in df.groupby(sig):
        if s and len(g) > 1:
            assert g["cluster"].nunique() == 1


def test_raking_moves_toward_p_star(built) -> None:
    _v, _st, tg, df, _panel = built
    sub = df[df["group"] == "threshold"]
    w = sub["weight"].to_numpy()
    assert np.isclose(w.sum(), len(sub)) and (w > 0).mean() > 0.8      # a category without a target share: 0
    cl = sub["cluster"].to_numpy()
    _u, inv, cnt = np.unique(cl, return_inverse=True, return_counts=True)
    base = 1.0 / cnt[inv]
    on = w > 0
    ratio = (w[on] / w[on].sum()) / (base[on] / base[on].sum())
    lo, hi = tg["caps"]
    assert ratio.min() >= lo - 1e-6 and ratio.max() <= hi + 1e-6              # bounded raking
    hz = pool.axis_features(sub)["horizon_bin"]                               # where the horizon applies
    keep = hz != "na"
    s2, w2 = sub[keep].assign(h=hz[keep]), w[keep]
    target = pool.p_star(tg, "horizon", sorted(set(s2["h"])))
    raw = s2.groupby("h").size() / len(s2)
    raked = s2.assign(w=w2).groupby("h")["w"].sum() / w2.sum()
    gap_raw = sum(abs(raw.get(k, 0) - x) for k, x in target.items())
    gap_raked = sum(abs(raked.get(k, 0) - x) for k, x in target.items())
    assert gap_raked < gap_raw
    cells = pool.cells(df, tg)
    assert all(c["ess"] <= c["clusters"] + 1e-9 for c in cells) and sum(c["share"] for c in cells) > 3.9


def test_episodes_from_the_pool(built) -> None:
    v, st, tg, df, _panel = built
    progs, w = pool.load_programs(df)
    assert [p.hash() for p in progs] == df["hash"].tolist()
    s = episodes.Sampler(st, v, seed=3, programs=progs, weights=w, targets=tg)
    fams = []
    for i in range(60):
        ep = s.episode(i)
        assert not contracts.episode_failures(ep), (i, contracts.episode_failures(ep))
        fams.append(ep["recipe"]["program"]["family"])
    assert "FH" in fams and "F6" in fams


def test_build_is_the_same_with_workers(built, tmp_path) -> None:
    """Candidates evaluated in worker processes give the table one process gives."""
    import shutil

    _v, st, tg, _df, _panel = built
    root = tmp_path / "s"
    shutil.copytree(st.root, root)
    st2 = fst.Store(root)
    st2.channels.stats = st.channels.stats
    st2.channels.save(root / "channels.json")                 # workers read the statistics from the store
    progs = fpr.catalog(st2.channels) + pool.f6_candidates(st2.channels, horizons=(240,))[:300] \
        + pool.fh_candidates(st2.channels, 40, 1, tg)
    one, _p = pool.build(st2, targets=tg, programs=progs, n_stays=150)
    two, _p = pool.build(st2, targets=tg, programs=progs, n_stays=150, workers=2, chunk=60)
    cols = ["hash", "prevalence", "per_stay", "valid_share", "cluster", "weight", "sites"]
    import pandas as pd

    pd.testing.assert_frame_equal(one[cols], two[cols])


def test_sites_follow_the_pools_support(built) -> None:
    """An episode's site carries its program on the pool's panel (a positive stay for an event), and its labels come
    from a cohort of that site's stays."""
    v, st, tg, df, _panel = built
    progs, w = pool.load_programs(df)
    sup = pool.load_support(df)
    assert sup and all(isinstance(x, tuple) and len(x) == 2 for s in sup.values() for x in s.values())
    s = episodes.Sampler(st, v, seed=5, programs=progs, weights=w, targets=tg, support=sup)
    for i in range(40):
        try:
            ep = s.episode(i)
        except episodes.NoEpisode:
            continue
        r = ep["recipe"]
        h = r["program"]["hash"]
        if h in sup and r["site_variant"]["kind"] != "cross_site":
            k = 1 if ep["task"]["label"] in ("binary", "hosp_death", "tte", "multihorizon") else 0
            if any(x[k] >= 1 for x in sup[h].values()):                # else every site is drawn
                assert sup[h].get(r["site"], (0, 0))[k] >= 1


def test_acceptance_weights() -> None:
    """Weights divided by their cell's acceptance (floored), each group keeping its total."""
    import pandas as pd

    df = pd.DataFrame({"group": ["threshold"] * 3 + ["values"] * 2, "label": ["binary"] * 3 + ["regression"] * 2,
                       "horizon_bin": ["2-6h", "2-6h", "6-12h", "2-6h", "6-12h"],
                       "prevalence_bin": ["1-2%", "1-2%", "1-2%", "na", "na"], "weight": [1.0, 1.0, 2.0, 1.0, 1.0]})
    acc = {"threshold|binary|2-6h|1-2%": 0.5, "threshold|binary|6-12h|1-2%": 1.0, "values|regression|2-6h|na": 0.01}
    w = pool.acceptance_weights(df, acc)
    assert np.isclose(w[:3].sum(), 4.0) and np.isclose(w[3:].sum(), 2.0)
    assert np.isclose(w[0] / w[2], (1 / 0.5) / (2 / 1.0))                  # 1/0.5 against 2/1
    assert np.isclose(w[3] / w[4], 1 / pool.ACCEPT_FLOOR)                   # floored; an unmeasured cell counts 1
