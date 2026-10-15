"""The quality suite (``factory.report``): its measures behave on controls (Gini of equal and of concentrated use,
kappa of duplicated label functions, coverage of a set by itself) and run on episodes of a store built from real
held-out token sequences (skipped without them)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from icutwin.factory import episodes, pool, programs as fpr, report, store as fst

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "full_r4"


def test_controls() -> None:
    assert report.gini([3, 3, 3, 3]) == 0.0
    assert report.gini([0, 0, 0, 12]) == pytest.approx(0.75)
    assert report.tv({"a": 0.5, "b": 0.5}, {"a": 1.0}) == pytest.approx(0.5)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not (T / "vocab.json").exists() or not (T / "mimic" / "held_out.parquet").exists():
        pytest.skip("the full r4 tokens")
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    rows = []
    for ds in ("mimic", "eicu", "aumc"):
        t = pq.read_table(T / ds / "held_out.parquet", columns=["stay_id", "ids"]).slice(0, 50).to_pylist()
        rows += [{"stay_id": r["stay_id"], "ids": r["ids"], "split": "train"} for r in t]
    root = tmp_path_factory.mktemp("rep")
    fst.build(rows, T / "vocab.json", root / "s", "seg_0000", {"kind": "test"}, workers=1, chunk=75)
    st = fst.Store(root / "s")
    st.channels.stats = fst.channel_stats(st)
    return v, st


def test_reports_on_episodes(built) -> None:
    v, st = built
    s = episodes.Sampler(st, v, seed=4, targets=pool.load_targets())
    eps = [s.episode(i) for i in range(40)]
    rows = [report.summarize(e, st) for e in eps]
    mix = report.mix_report(rows, s.cfg)
    assert abs(sum(mix["groups"]["realised"].values()) - 1) < 1e-9 and 0 <= mix["anonymized"]["realised"] <= 1
    dist = report.distribution_report(rows, pool.load_targets())
    assert all(0 <= ax["tv"] <= 1 for g in dist.values() for ax in g.values() if isinstance(ax, dict))
    progs = report.program_report(rows, {p.hash() for p in fpr.catalog(st.channels)})
    assert progs["distinct_programs"] > 5 and 0 < progs["catalog_reached"] <= 1
    reuse = report.reuse_report(rows, st)
    assert 0 <= reuse["gini"] <= 1 and reuse["max_over_mean"] >= 1
    scores = [report.difficulty(s, e) for e in eps]
    assert any(x is not None for x in scores)
    assert report.context_vendi(st, list(range(10))) >= 1.0


def test_label_functions_and_coverage(built) -> None:
    _v, st = built
    cat = [p for p in fpr.catalog(st.channels) if p.label == "binary"]
    lf = report.label_functions(st, cat + cat[:5], n_rows=800)
    assert lf["near_duplicates"] > 0 and 0 <= lf["median_abs_kappa"] <= 1
    assert 1 <= lf["effective_label_functions"] <= lf["programs"]
    half = np.arange(len(st))
    cov = report.coverage(st, half[::2], st, half[1::2])
    assert cov["coverage"] > 0.5 and cov["density"] > 0.3


def test_shift_probe_on_cross_site_episodes(built) -> None:
    """The cross-site shift probe scores cross-site binary episodes only, as a local-minus-cross AUROC difference."""
    v, st = built
    cfg = episodes.load_config()
    cfg = {**cfg, "sites": {"plain": 0.0, "virtual": 0.0, "label_shift": 0.0, "cross_site": 1.0}}
    s = episodes.Sampler(st, v, seed=1, config=cfg)
    got = []
    for i in range(60):
        try:
            ep = s.episode(i)
        except episodes.NoEpisode:
            continue
        assert ep["recipe"]["site_variant"]["kind"] == "cross_site"
        r = report.shift_loss(s, ep)
        if r is not None:
            assert -1 <= r["loss"] <= 1 and np.isclose(r["loss"], r["local"] - r["cross"])
            got.append(r)
    assert got
    plain = episodes.Sampler(st, v, seed=1).episode(0)
    if plain["recipe"]["site_variant"]["kind"] != "cross_site":
        assert report.shift_loss(s, plain) is None


def test_base_rate_coverage() -> None:
    """A program's real rate inside its episodes' context-prevalence p10-p90, and their spread against the twin-to-real
    gap."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from factory_report import base_rate_coverage

    p = fpr.Program("F1", ("death",), "binary", 1440, ("in_icu",), name="icu_death")
    q = fpr.Program("F2", ("event", "imv"), "binary", 720, ("in_icu", "not_in_event"), name="imv_onset")
    rng = np.random.default_rng(0)
    rows = [{"hash": p.hash(), "cadence": "rolling", "prevalence": float(x)} for x in rng.uniform(0.02, 0.12, 50)]
    rows += [{"hash": q.hash(), "cadence": "rolling", "prevalence": float(x)} for x in rng.uniform(0.010, 0.011, 50)]
    rows += [{"hash": q.hash(), "cadence": "landmark", "prevalence": 0.5}]          # other cadences are not read
    align = {"programs": {"icu_death@1440": {"dev": 0.061, "twins": 0.042},
                          "imv_onset@720": {"dev": 0.03, "twins": 0.0105}}}
    out = base_rate_coverage(rows, align, [p, q])
    assert out["scored"] == 2
    assert out["programs"]["icu_death@1440"]["real_within"] and not out["programs"]["imv_onset@720"]["real_within"]
    assert out["programs"]["icu_death@1440"]["spread"] >= out["programs"]["icu_death@1440"]["gap"]
    assert out["share_real_within"] == 0.5 and out["share_spread_beyond_gap"] == 0.5
    assert out["scored_as_drawn"] == 0 and out["programs"]["imv_onset@720"]["no_case"] == 0
    rows += [{"hash": q.hash(), "cadence": "rolling", "prevalence": 0.0, "as_drawn": True}] * 20   # contexts taken
    rows += [{"hash": q.hash(), "cadence": "rolling", "prevalence": 0.06, "as_drawn": True}] * 10  # as drawn
    out = base_rate_coverage(rows, align, [p, q])["programs"]["imv_onset@720"]
    assert out["no_case"] == 0.25 and out["as_drawn_p10_p50_p90"][0] == 0.0 and out["real_within_as_drawn"]
