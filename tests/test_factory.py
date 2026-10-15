"""The ICL factory (S9) on the 2% subset's real held-out stays (skipped without them): the cache, the task prior and
the episode contracts (TabICL's get_batch layout, the time-series layout), and the leakage guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from icutwin.factory import cache as fc
from icutwin.factory import tasks as ft

R = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def cache():
    meds = R / "meds" / "subset"
    if not (meds / "mimic" / "metadata" / "stays.parquet").exists():
        pytest.skip("the subset's MEDS")
    from icutwin.tokenization.stays import iter_stays, load_derived

    stays, derived = [], {"episodes": {}, "kdigo": {}, "sofa_24h": {}}
    for ds in ("mimic", "eicu", "hirid", "aumc", "sicdb", "nwicu", "inspire"):
        d = load_derived(R / "derived" / "subset" / ds)
        for k in derived:
            derived[k].update(d[k])
        stays += list(iter_stays(meds / ds, "held_out", R / "derived" / "subset" / ds))
    return fc.build(stays, derived, hours=72, n_concepts=48)


def test_cache_holds_features_and_labels(cache, tmp_path: Path) -> None:
    c = cache
    assert c.grid.shape == (len(c.stay_id), 72, 48) and len(c.concepts) == 48
    assert 0.05 < np.mean(~np.isnan(c.grid[:, :24])) < 0.9         # sparse, not empty
    assert np.isfinite(c.los_hours).all() and (c.los_hours > 0).mean() > 0.99
    assert {0.0, 1.0} <= set(np.unique(c.death[~np.isnan(c.death)]))
    assert np.sum(~np.isnan(c.onset["imv"])) > 0 and np.sum(~np.isnan(c.sofa_24h)) > 0
    c.save(tmp_path / "c.npz")
    back = fc.Cache.load(tmp_path / "c.npz")
    assert back.stay_id == c.stay_id and np.array_equal(back.grid, c.grid, equal_nan=True)


def test_batches_meet_the_tabicl_contract(cache) -> None:
    prior = ft.ICUPrior(cache, min_rows=48, max_rows=400, seed=0)
    x, y, d, seq, train = prior.get_batch(12)
    b, t, h = x.shape
    assert y.shape == (b, t) and d.shape == seq.shape == train.shape == (b,)
    assert (d <= h).all() and (d >= 1).all() and (d <= 100).all() and (seq <= t).all()
    for i, spec in enumerate(prior.last_specs):
        n, k = int(seq[i]), int(d[i])
        assert 0.1 * n - 1 <= int(train[i]) <= 0.9 * n + 1
        xi, yi = x[i, :n, :k].numpy(), y[i, :n].numpy()
        assert not np.isnan(xi).any() and (np.ptp(xi[:int(train[i])], axis=0) > 0).all()   # no constant feature
        assert (x[i, n:] == 0).all() and (x[i, :, k:] == 0).all()                           # zero padding
        if spec.family != "regression":
            cls = np.unique(yi)
            assert len(cls) <= 10 and set(cls) == set(np.arange(len(cls)))
            assert set(np.unique(yi[:int(train[i])])) == set(cls) == set(np.unique(yi[int(train[i]):]))
    again = ft.ICUPrior(cache, min_rows=48, max_rows=400, seed=0).get_batch(12)
    assert all(np.array_equal(a.numpy(), b.numpy()) for a, b in zip((x, y, d, seq, train), again))


def test_time_series_batches(cache) -> None:
    prior = ft.ICUPrior(cache, min_rows=48, max_rows=300, seed=1)
    xs, ms, st, y, train = prior.get_ts_batch(6, channels=8)
    b, t, length, ch = xs.shape
    assert ms.shape == xs.shape and st.shape[:2] == (b, t) and y.shape == (b, t) and ch == 8
    assert (xs[~ms] == 0).all() and ms.any()


def test_features_never_read_after_the_trigger(cache) -> None:
    """Changing every value at or after the trigger leaves the design unchanged (leakage guard)."""
    c = cache
    spec = ft.TaskSpec("binary", "vasopressor_within", trigger=12, horizon=24,
                       features=[(j, a) for j in range(10) for a in ft.AGGREGATORS], statics=("age", "unit_type"))
    idx, _y = ft.labels(c, spec)
    ctx = np.arange(len(idx) // 2)
    x0 = ft.design(c, spec, idx, ctx)
    g = c.grid.copy()
    c.grid[:, 12:] = np.where(np.isnan(c.grid[:, 12:]), 7.0, c.grid[:, 12:] * 3 + 1)
    try:
        x1 = ft.design(c, spec, idx, ctx)
    finally:
        c.grid[:] = g
    assert np.array_equal(x0, x1)
    # header fields documented after the trigger (spec.yaml available_within_hours) are unknown at it, whatever
    # their value (design-review correction 3)
    late = {"eicu", "aumc", "sicdb", "hirid"}                       # within 24 hours (INSPIRE: 1 hour)
    spec = ft.TaskSpec("binary", "vasopressor_within", trigger=12, horizon=24, features=[(0, "last")],
                       statics=("adm_dx_group",))
    idx, _y = ft.labels(c, spec)
    idx = idx[np.isin(c.dataset[idx], list(late))]                  # stays whose diagnosis group comes later
    assert len(idx) > 10
    s0 = c.static["adm_dx_group"].copy()
    rng = np.random.default_rng(0)
    try:
        c.static["adm_dx_group"][:] = rng.integers(0, 5, len(s0))
        xa = ft.design(c, spec, idx, np.arange(len(idx)))
        c.static["adm_dx_group"][:] = rng.integers(0, 5, len(s0))
        xb = ft.design(c, spec, idx, np.arange(len(idx)))
    finally:
        c.static["adm_dx_group"][:] = s0
    np.testing.assert_array_equal(xa, xb)


def test_episodes_run_through_a_tabicl_forward_pass(cache) -> None:
    """Our classification episodes feed TabICL (a small random-initialized model; one episode per call, as its
    y_train shares one context size across a batch): finite logits for every query row."""
    import torch

    tabicl = pytest.importorskip("tabicl._model.tabicl")
    torch.manual_seed(0)
    model = tabicl.TabICL(max_classes=10, embed_dim=32, col_num_blocks=1, col_nhead=4, col_num_inds=16,
                          row_num_blocks=1, row_nhead=4, row_num_cls=2, icl_num_blocks=2, icl_nhead=4).train()
    prior = ft.ICUPrior(cache, min_rows=48, max_rows=200, seed=3, families=("binary", "multiclass"))
    x, y, d, seq, train = prior.get_batch(4)
    for i in range(len(d)):
        n, k, m = int(seq[i]), int(d[i]), int(train[i])
        out = model(x[i:i + 1, :n, :k], y[i:i + 1, :m], d=d[i:i + 1])
        assert out.shape[:2] == (1, n - m) and torch.isfinite(out).all()


def test_extratrees_filter_separates_signal_from_noise() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 6))
    signal = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(float)
    noise = rng.integers(0, 2, size=400).astype(float)
    assert ft.informative(x, signal, 200, "binary")[0]
    assert not ft.informative(x, noise, 200, "binary")[0]
    three = np.digitize(x[:, 0], [-0.5, 0.5]).astype(float)
    assert ft.informative(x, three, 200, "multiclass")[0]
    assert ft.informative(x, 2 * x[:, 2] + rng.normal(scale=0.3, size=400), 200, "regression")[0]
    assert not ft.informative(x, rng.normal(size=400), 200, "regression")[0]


def test_prior_filters_dedups_and_writes_cards(cache) -> None:
    prior = ft.ICUPrior(cache, min_rows=48, max_rows=300, seed=5)
    prior.get_batch(16)
    st = prior.stats
    assert st["accepted"] == 16 and st["drawn"] == sum(v for k, v in st.items() if ":" not in k and k != "drawn")
    assert len(prior.seen) == 16                                  # one task, one episode
    assert len({c["hash"] for c in prior.last_cards}) == 16
    for card, spec in zip(prior.last_cards, prior.last_specs):
        assert card["hash"] == ft.spec_hash(spec) and card["family"] == spec.family
        score = card["filter_score"]
        assert score > (0.05 if spec.family == "regression" else 0.55)          # at least the margin
        assert card["description"] and card["windows"]["input"] == f"[0, {spec.trigger}) h"
        assert len(card["features"]) == len(spec.features) + len(spec.statics)
    s = prior.last_specs[0]
    same = ft.TaskSpec(**{**s.__dict__, "rows": s.rows + 1, "train_ratio": 0.3})
    assert ft.spec_hash(same) == ft.spec_hash(s)                  # the episode's size is not the task
    other = ft.TaskSpec(**{**s.__dict__, "trigger": s.trigger + 1})
    assert ft.spec_hash(other) != ft.spec_hash(s)


def test_curriculum_stages() -> None:
    assert ft.CURRICULUM[0] == (1024, 1024) and ft.CURRICULUM[2] == (400, 60000)
    c = fc.Cache(["a"], np.array(["mimic"]), np.array(["mimic"]), {}, {}, ["x"], np.zeros((1, 72, 1), np.float32),
                 np.ones(1), np.zeros(1))
    p = ft.ICUPrior(c, stage=1)
    assert (p.min_rows, p.max_rows) == (400, 10240)
    rows = [p.sample_spec().rows for _ in range(400)]
    assert 400 <= min(rows) and max(rows) <= 10240 and np.median(rows) < 3000     # log-uniform


def test_family_floors_hold_after_filtering(cache) -> None:
    """A family the filter rejects more often is drawn until it is back above its floor."""
    prior = ft.ICUPrior(cache, min_rows=48, max_rows=300, seed=7, floors={"binary": 0.3, "multiclass": 0.1,
                                                                          "regression": 0.1})
    for _ in range(40):
        prior.episode()
    n = prior.stats["accepted"]
    assert n == 40 and prior.stats["accepted:binary"] / n >= 0.3 - 1 / n
    assert all(prior.stats[f"accepted:{f}"] > 0 for f in prior.families)


def test_mined_targets_and_realism(cache) -> None:
    c = cache
    k = 0
    prior = ft.ICUPrior(c, min_rows=48, max_rows=300, seed=11)
    thr = prior.threshold(k, 0.9)
    spec = ft.TaskSpec("binary", "concept_above", trigger=12, horizon=24, concept=k, threshold=thr,
                       features=[(j, a) for j in range(6) for a in ("last", "max")], statics=("age",))
    idx, y = ft.labels(c, spec)
    assert len(idx) > 50 and set(np.unique(y)) <= {0.0, 1.0} and 0 < y.mean() < 1
    w = c.grid[idx, 12:36, k]
    assert (~np.isnan(w).all(1)).all()                                  # every eligible stay has a value there
    chg = ft.TaskSpec("regression", "concept_change", trigger=12, horizon=6, concept=k, features=spec.features)
    i2, y2 = ft.labels(c, chg)
    assert len(i2) > 50 and np.isfinite(y2).all()
    # realism: reproducible, more missing values, never reads past the trigger
    real = ft.TaskSpec(**{**spec.__dict__, "realism": {"mcar": 0.2, "mnar": 0.2, "noise": 0.05, "units": 1}})
    ctx = np.arange(len(idx) // 2)
    x0, x1 = ft.design(c, spec, idx, ctx), ft.design(c, real, idx, ctx)
    assert np.array_equal(x1, ft.design(c, real, idx, ctx)) and x1.shape[1] > x0.shape[1]   # more indicators
    g = c.grid.copy()
    c.grid[:, 12:] = np.where(np.isnan(c.grid[:, 12:]), 7.0, c.grid[:, 12:] * 3 + 1)
    try:
        assert np.array_equal(x1, ft.design(c, real, idx, ctx))
    finally:
        c.grid[:] = g


def test_shift_stress_and_mined_cards(cache) -> None:
    prior = ft.ICUPrior(cache, min_rows=48, max_rows=400, seed=13, mined_share=1.0, realism_share=1.0)
    shifted = 0
    for _ in range(25):
        spec, x, y, n_train, idx = prior.episode()
        card = ft.task_card(cache, spec, y, n_train)
        assert card["description"]
        if spec.target in ft.MINED_BINARY:
            assert spec.threshold is not None and "within" in card["description"]
        if spec.realism.get("shift"):
            shifted += 1
            ds = cache.dataset[idx]
            home = ds[0]
            assert np.mean(ds[:n_train] == home) >= np.mean(ds[n_train:] == home)
    assert prior.stats["accepted"] == 25


def test_batch_contract_layer(cache) -> None:
    """Layer (d): real batches pass; every kind of corruption is caught."""
    from icutwin.factory import contracts as fct

    prior = ft.ICUPrior(cache, min_rows=48, max_rows=300, seed=17, families=("binary", "multiclass"))
    x, y, d, seq, train = (a.numpy() for a in prior.get_batch(6))
    fam = [s.family for s in prior.last_specs]
    assert fct.batch_failures(x, y, d, seq, train, fam) == []
    bad = x.copy()
    bad[0, int(seq[0]):] = 1.0
    assert any("padding" in f for f in fct.batch_failures(bad, y, d, seq, train, fam))
    bad = x.copy()
    bad[1, 0, 0] = np.nan
    assert any("non-finite" in f for f in fct.batch_failures(bad, y, d, seq, train, fam))
    yy = y.copy()
    yy[2, :int(seq[2])] = yy[2, :int(seq[2])] + 1                  # classes 1..k
    assert any("classes" in f for f in fct.batch_failures(x, yy, d, seq, train, fam))
    tt = train.copy()
    tt[3] = 0
    assert any("context rows" in f for f in fct.batch_failures(x, y, d, seq, tt, fam))
    with pytest.raises(fct.ContractError):
        fct.check_batch(bad, y, d, seq, train, families=fam)


def test_time_to_event_and_its_binary_reduction(cache) -> None:
    """Time-to-event labels reduce, censoring-aware, to the 'event within the horizon' labels of the same stays."""
    c = cache
    for event in ("imv", "vasopressor"):
        idx, y_time, y_event = ft.labels_tte(c, event, trigger=12)
        assert len(idx) > 50 and (y_time > 0).all() and set(np.unique(y_event)) <= {0.0, 1.0}
        assert (y_time[y_event == 0] <= c.los_hours[idx][y_event == 0] - 12 + 1e-9).all()      # censored at discharge
        keep, yb = ft.binary_at_horizon(y_time, y_event, 24)
        bidx, by = ft.labels(c, ft.TaskSpec("binary", f"{event}_within", trigger=12, horizon=24))
        mine = dict(zip(idx[keep].tolist(), yb.tolist()))
        theirs = dict(zip(bidx.tolist(), by.tolist()))
        assert all(theirs[k] == v for k, v in mine.items())                 # same label where both have the stay
        assert set(theirs) - set(mine) <= {k for k in theirs if theirs[k] == 0}   # left out: censored without event
    prior = ft.ICUPrior(c, min_rows=48, max_rows=300, seed=19)
    spec, x, y_time, y_event, n_train, idx = prior.tte_episode()
    assert spec.family == "time_to_event" and len(y_time) == len(y_event) == x.shape[0] == len(idx)
    assert 0 < y_event[:n_train].sum() < n_train


def test_multilabel_and_its_binary_adapter(cache) -> None:
    c = cache
    idx, y = ft.labels_multilabel(c, ("imv", "vasopressor", "rrt"), trigger=12, horizon=24)
    assert y.shape == (len(idx), 3) and np.isnan(y).any() and np.nanmax(y) == 1.0
    for k, e in enumerate(("imv", "vasopressor", "rrt")):
        keep, yk = ft.multilabel_to_binary(y, k)
        bidx, by = ft.labels(c, ft.TaskSpec("binary", f"{e}_within", trigger=12, horizon=24))
        assert np.array_equal(idx[keep], bidx) and np.array_equal(yk, by)     # the adapter is the binary task


def test_forecast_batches_never_read_the_future(cache) -> None:
    prior = ft.ICUPrior(cache, min_rows=48, max_rows=200, seed=23)
    xs, ms, st, yf, mf, train = prior.get_forecast_batch(4, channels=8, targets=3, horizon=12)
    b, t, length, ch = xs.shape
    assert yf.shape == (b, t, 12, 3) and mf.shape == yf.shape and mf.any()
    assert (yf[~mf] == 0).all() and (xs[~ms] == 0).all()
    for i, (trig, _tg) in enumerate(prior.last_forecast):
        assert not ms[i, :, trig:].any()                     # inputs end at the trigger


def test_counterfactual_labels() -> None:
    do = [[{"death": True}, {"death": False}], [{"death": None}], [{"death": False}] * 4]
    dont = [[{"death": True}] * 2, [{"death": True}], [{"death": True}, {"death": False}]]
    y0, y1, eff = ft.counterfactual_labels(do, dont, "death")
    assert y1[0] == 0.5 and y0[0] == 1.0 and eff[0] == -0.5
    assert np.isnan(y1[1]) and np.isnan(eff[1])
    assert eff[2] == -0.5
