"""Twin forecasting's outcome reader and scores (S8) on the 2% subset's real stays (skipped without them)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from icutwin.eval import forecast as fo

R = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def stays():
    tokens = R / "tokens" / "subset"
    if not (tokens / "vocab.json").exists():
        pytest.skip("the subset's tokens")
    from icutwin.tokenization.stays import load_stays
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(tokens / "vocab.json")
    ids, los = [], []
    for ds in ("mimic", "eicu", "hirid", "aumc"):
        t = pq.read_table(tokens / ds / "held_out.parquet", columns=["stay_id", "ids"]).to_pylist()
        meta = {r["stay_id"]: r for r in load_stays(R / "meds" / "subset" / ds).iter_rows(named=True)}
        for r in t:
            m = meta[r["stay_id"]]
            ids.append(r["ids"])
            los.append((m["icu_out"] - m["icu_in"]).total_seconds() / 86400)
    return v, ids, np.array(los)


def test_outcomes_of_real_stays(stays) -> None:
    v, ids, los = stays
    rd = fo.Reader(v)
    flags = [rd.outcomes(s) for s in ids]
    assert sum(f["death"] for f in flags if f["death"] is not None) > 0
    assert sum(f["imv"] for f in flags) > 0 and sum(f["vasopressor"] for f in flags) > 0
    known = [(f["icu_los_gt_7d"], d > 7) for f, d in zip(flags, los) if f["icu_los_gt_7d"] is not None]
    assert len(known) > 0.8 * len(ids)
    assert np.mean([a == b for a, b in known]) > 0.9            # interval medians reproduce the real ICU stay


def test_flags_within_horizons(stays) -> None:
    """Round 3's horizons: over an unlimited horizon the flags are the whole stay's; they only grow with the
    horizon; from the ICU admission marker, an ICU discharge within h hours is an ICU stay shorter than h."""
    v, ids, _los = stays
    rd = fo.Reader(v)
    checked = 0
    for s in ids:
        whole, inf = rd.outcomes(s), rd.within(s, 0, 1e9)
        assert all(inf[k] == whole[k] for k in ("imv", "vasopressor", "rrt"))
        flags = [rd.within(s, 0, h) for h in fo.HORIZONS]
        assert all(not a[k] or b[k] for a, b in zip(flags, flags[1:]) for k in fo.HORIZON_OUTCOMES)
        adm = next((i for i, t in enumerate(s) if t in rd.adm), None)
        if adm is not None and whole["icu_los_hours"] is not None and abs(whole["icu_los_hours"] - 24) > 0.5:
            assert rd.within(s, adm, 24)["icu_discharge"] == (whole["icu_los_hours"] < 24)
            checked += 1
    assert checked > 0.5 * len(ids)
    truth = [rd.within(s, 0, 24) for s in ids]
    perfect = fo.evaluate(truth, [[t] * 3 for t in truth], fo.HORIZON_OUTCOMES)
    assert perfect["icu_discharge"]["auroc"] == 1.0 and "icu_los_hours" not in perfect


def test_scores_of_a_perfect_and_a_random_forecaster(stays) -> None:
    v, ids, _los = stays
    rd = fo.Reader(v)
    truth = [rd.outcomes(s) for s in ids]
    perfect = fo.evaluate(truth, [[t] * 4 for t in truth])
    assert perfect["death"]["auroc"] == 1.0 and perfect["death"]["ece"] < 1e-9
    rng = np.random.default_rng(0)
    shuffled = [truth[i] for i in rng.permutation(len(truth))]
    rand = fo.evaluate(truth, [[t] for t in shuffled])
    assert abs(rand["death"]["auroc"] - 0.5) < 0.1


def test_dispersion_flags_mis_dispersed_forecasters() -> None:
    """PIT, CRPS and spread-skill on synthetic forecasts: a calibrated ensemble has a uniform PIT and spread-skill
    near 1; an under-dispersed one (too narrow) and an over-dispersed one are told apart; CRPS ranks them."""
    rng = np.random.default_rng(0)
    mu = rng.uniform(24, 240, size=600)
    truth = (mu + rng.normal(0, 20, size=600)).tolist()
    calibrated = [(m + rng.normal(0, 20, size=32)).tolist() for m in mu]
    narrow = [(m + rng.normal(0, 4, size=32)).tolist() for m in mu]
    wide = [(m + rng.normal(0, 80, size=32)).tolist() for m in mu]
    c, n, w = (fo.dispersion(truth, e) for e in (calibrated, narrow, wide))
    assert c["pit_ks"] < 0.08 and 0.85 < c["spread_skill"] < 1.15
    assert n["pit_ks"] > 0.2 and n["spread_skill"] < 0.5
    assert w["pit_ks"] > 0.15 and w["spread_skill"] > 2
    assert c["crps"] < n["crps"] and c["crps"] < w["crps"]
    assert abs(fo.crps_ensemble(np.array([1.0, 1.0]), 1.0)) < 1e-12 and fo.crps_ensemble(np.array([0.0]), 2.0) == 2.0
