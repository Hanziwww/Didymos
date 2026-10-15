"""Twin-level calibration (round 3): raking matches every margin and leaves a pool already on target uniform; the
draw includes each twin with probability proportional to its weight, without replacement; the features read the
utility tasks' labels off real stays (the subset's tokens; skipped without them)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from icutwin.generate import calibrate as cal


def _pool(n: int = 3000, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        long = rng.random() < 0.4
        out.append({"los": "96-168" if long else "0-24",
                    "rrt": "na" if rng.random() < 0.3 else ("yes" if rng.random() < (0.15 if long else 0.05) else "no"),
                    "imv": "yes" if rng.random() < 0.2 else "no"})
    return out


def _shares(feats, w, m):
    tot = Counter()
    for f, x in zip(feats, w):
        tot[f[m]] += x
    s = sum(tot.values())
    return {k: c / s for k, c in tot.items()}


def test_raking_matches_margins_and_is_uniform_on_target() -> None:
    feats = _pool()
    own = {m: _shares(feats, np.ones(len(feats)), m) for m in ("los", "rrt", "imv")}
    w, rep = cal.rake(feats, own)
    assert np.allclose(w, 1.0, atol=1e-9) and rep["max_margin_error"] < 1e-6
    target = {"los": {"0-24": 0.75, "96-168": 0.25}, "rrt": {"na": 0.3, "yes": 0.04, "no": 0.66},
              "imv": {"yes": 0.1, "no": 0.9}}
    w, rep = cal.rake(feats, target)
    assert rep["max_margin_error"] < 1e-6 and 0 < rep["effective_size"] < len(feats)
    for m, t in target.items():
        got = _shares(feats, w, m)
        assert all(abs(got[k] - s) < 1e-5 for k, s in t.items()), (m, got)
    # a category the target gives no share gets no weight; one the pool lacks is reported, its share renormalized
    w, rep = cal.rake(feats, {"los": {"0-24": 1.0}, "imv": {"yes": 0.2, "no": 0.7, "absent": 0.1}})
    assert all(x == 0 for f, x in zip(feats, w) if f["los"] != "0-24")
    assert rep["unmatched"] == {"imv": {"absent": 0.1}} and abs(_shares(feats, w, "imv")["yes"] - 0.2 / 0.9) < 1e-5


def test_draw_includes_proportionally_without_replacement() -> None:
    rng = np.random.default_rng(1)
    w = rng.gamma(2.0, size=600)
    w[:5] = 200.0                                          # weights so large the twin is always taken
    n, reps = 150, 3000
    seen = np.zeros(len(w))
    for _ in range(reps):
        k = cal.draw(w, n, rng)
        assert len(k) == n and len(set(k.tolist())) == n
        seen[k] += 1
    freq = seen / reps
    pi, capped = np.zeros(len(w)), np.zeros(len(w), dtype=bool)     # expected: proportional, capped at 1
    while True:
        pi[~capped] = (n - capped.sum()) * w[~capped] / w[~capped].sum()
        pi[capped] = 1.0
        if not (pi[~capped] > 1).any():
            break
        capped |= pi > 1
    assert (freq[:5] == 1).all() and capped[:5].all()
    assert np.abs(freq - pi).max() < 0.04 and abs(freq.sum() - n) < 1e-6


def test_resampled_margins_follow_the_target() -> None:
    feats = _pool(4000, seed=2)
    target = {"los": {"0-24": 0.7, "96-168": 0.3}, "rrt": {"na": 0.3, "yes": 0.03, "no": 0.67}}
    w, _rep = cal.rake(feats, target)
    k = cal.draw(w, 1000, np.random.default_rng(3))
    kept = [feats[i] for i in k]
    for m, t in target.items():
        got = _shares(kept, np.ones(len(kept)), m)
        assert all(abs(got.get(c, 0) - s) < 0.03 for c, s in t.items()), (m, got)


def test_events_margin_counts_as_the_harness_and_rakes_the_tails() -> None:
    """Amendment 6: the events-per-stay band counts the harness's events (``eval.fidelity``) and cuts at the
    training quantiles; raked to the training shares, the kept twins' 90th and 99th percentiles follow the real ones
    even when the pool's tail is twice as long."""
    T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"
    assert [cal.event_band(c, [10.0, 20.0]) for c in (5, 10, 19, 20, 25)] == ["e0", "e1", "e1", "e2", "e2"]
    rng = np.random.default_rng(4)
    real = rng.lognormal(5, 0.8, 20_000)
    edges = list(np.quantile(real, cal.EVENT_QUANTILES))
    target = {"events": {k: c / len(real) for k, c in Counter(cal.event_band(x, edges) for x in real).items()}}
    pool = np.r_[rng.lognormal(5, 0.8, 2400), rng.lognormal(6, 0.8, 600)]      # a twin pool with a heavy tail
    feats = [{"events": cal.event_band(x, edges)} for x in pool]
    w, _rep = cal.rake(feats, target)
    kept = pool[cal.draw(w, 1000, np.random.default_rng(5))]
    for q in (90, 99):
        assert abs(np.percentile(kept, q) / np.percentile(real, q) - 1) < 0.12, q
    assert np.percentile(pool, 99) / np.percentile(real, 99) > 1.4
    if not (T / "vocab.json").exists():
        return
    from icutwin.eval import fidelity as fd
    from icutwin.eval import privacy as pv
    from icutwin.eval.forecast import Reader
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    seqs = pv.first_windows(T, "mimic", "held_out")[:50]
    ev = cal.event_mask(v)
    voc = fd.vocabulary(v)
    assert [int(ev[np.asarray(s, dtype=np.int64)].sum()) for s in seqs] == [fd.features(s, voc).events for s in seqs]
    f = cal.features(seqs, Reader(v), edges=[50.0, 100.0])
    assert {x["events"] for x in f} <= {"e0", "e1", "e2"} and all("los" in x for x in f)


def test_features_are_the_utility_labels_of_real_stays() -> None:
    T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's tokens")
    from icutwin.eval import privacy as pv
    from icutwin.eval import utility as ut
    from icutwin.eval.forecast import Reader
    from icutwin.tokenization.vocab import Vocab

    rd = Reader(Vocab.load(T / "vocab.json"))
    real = pv.first_windows(T, "mimic", "held_out")
    feats = cal.features(real, rd)
    with_landmark = [f for s, f in zip(real, feats) if ut.landmark([int(x) for x in s], rd) is not None]
    _w, labels = ut.examples(real, rd)
    for k in cal.TREATMENTS:
        mine = np.array([np.nan if f[k] == "na" else float(f[k] == "yes") for f in with_landmark])
        assert np.array_equal(np.isnan(mine), np.isnan(labels[k])) and np.nansum(mine) == np.nansum(labels[k])
    assert {f["los"] for f in feats} <= {*cal.los.LABELS, "none"}
