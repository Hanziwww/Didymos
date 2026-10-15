"""Generic trajectories (S6, ``factory.generic``): latent series are standardized GP draws, the observation process
charts vital-like channels hourly and lab-like channels sparsely and more often at extremes, and a generic store opens
as a store whose episodes (threshold, functional and forecast programs on anonymous channels) pass the contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from icutwin.factory import contracts, episodes, generic, pool, store as fst


def test_latent_series_and_observation() -> None:
    rng = np.random.default_rng(0)
    z = generic.kernel_synth(rng, 200)
    assert z.shape == (200,) and abs(z.mean()) < 1e-6 and abs(z.std() - 1) < 1e-6
    m = generic.ts_mixup(rng, 120)
    assert m.shape == (120,) and np.isfinite(m).all()
    t, v = generic.observe(rng, z, "vital", 80.0, 10.0)
    assert (np.diff(t) > 0).all() and 0.6 * 200 < len(t) <= 200 and 40 < np.median(v) < 120
    zz = np.r_[np.zeros(100), np.full(100, 3.0)]                     # extremes are charted more often
    tl, _ = generic.observe(np.random.default_rng(1), zz, "lab", 0.0, 1.0)
    assert (tl >= 100 * 60).sum() > (tl < 100 * 60).sum()


def test_generic_store_and_episodes(tmp_path: Path) -> None:
    generic.build_store(tmp_path / "g", n_stays=120, n_channels=8, seed=1)
    st = fst.Store(tmp_path / "g")
    assert len(st) == 120 and st.channels.names[0] == "GEN//c000" and all(st.channels.physiological)
    tg = pool.load_targets()
    progs = pool.f6_candidates(st.channels, horizons=(240, 1440))[:400] + pool.fh_candidates(st.channels, 60, 0, tg) \
        + pool.fc_candidates(st.channels, 20, 0)
    s = episodes.Sampler(st, None, seed=0, programs=progs, targets=tg)
    labels = set()
    for i in range(30):
        ep = s.episode(i)
        assert not contracts.episode_failures(ep), contracts.episode_failures(ep)
        labels.add(ep["task"]["label"])
    assert "binary" in labels
