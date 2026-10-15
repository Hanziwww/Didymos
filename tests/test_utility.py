"""S8 utility (TSTR / TRTR) on the 2% subset's real stays (skipped without them): the landmark and labels read off
tokens agree with the stays, real training stays in a candidate's place reach the TRTR value (positive control),
and swapped outcomes fall to chance (negative control)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from icutwin.eval import privacy as pv
from icutwin.eval import utility as ut
from icutwin.eval.forecast import Reader

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"
DATASETS = ("mimic", "eicu", "hirid", "aumc", "sicdb", "nwicu", "inspire")


@pytest.fixture(scope="module")
def data():
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's token shards")
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    train, held = [], []
    for ds in DATASETS:
        train += pv.first_windows(T, ds, "train")
        held += pv.first_windows(T, ds, "held_out")
    return v, train, held


def test_landmark_and_labels(data) -> None:
    v, train, _held = data
    r = Reader(v)
    w, lab = ut.examples(train[:2000], r)
    assert 0.3 < len(w) / 2000 < 0.95                         # stays still in the ICU at 24 h
    for x in w[:200]:
        t, start = 0.0, None
        for tok in x:
            if start is None and int(tok) in r.adm:
                start = t
            t += r.minutes.get(int(tok), 0.0)
        assert start is not None and t - start < 24 * 60     # the window ends before the landmark
    for k in ut.NEW_ONSET:                                   # an event before the landmark: not a new onset
        for x, y in zip(w, lab[k]):
            if r.sets[k] & set(x.tolist()):
                assert np.isnan(y)
    assert 0.03 < np.nanmean(lab["death"]) < 0.4


def test_tstr_controls(data) -> None:
    v, train, held = data
    rng = np.random.default_rng(0)
    k = rng.permutation(len(train))
    half = len(k) // 2
    a, b = [train[i] for i in k[:half]], [train[i] for i in k[half:2 * half]]
    rep = ut.tstr(a, held, {"real_twin": b}, v)
    print(json.dumps({"trtr": {t: x["auroc"] for t, x in rep["trtr"].items()},
                      **{s: {t: (x["auroc"], x["utility_ratio"]) for t, x in res.items()} for s, res in
                         rep["sets"].items()}, "stays": rep["stays"]}, indent=1))
    for t in ("death", "icu_los_gt_7d"):
        assert rep["trtr"][t]["auroc"] > 0.7, t
        assert 0.8 < rep["sets"]["real_twin"][t]["utility_ratio"] < 1.2, t
        assert abs(rep["sets"]["swapped_outcomes"][t]["auroc"] - 0.5) < 0.08, t
