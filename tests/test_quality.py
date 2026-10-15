"""The quality pilot's seeds: one per sample, distinct across datasets whatever subset of them a process is given;
the per-dataset prevalence block (round 3's S1) on the subset's real stays (skipped without them)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from icutwin.eval import quality as q

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"


def test_pilot_seeds_are_distinct_across_datasets_and_processes() -> None:
    seeds = [s for ds in q.DATASETS for s in q.sample_seeds(0, ds, 1000)]
    assert len(set(seeds)) == len(seeds) == 7000
    split = q.sample_seeds(0, "eicu", 1000) + q.sample_seeds(0, "nwicu", 1000)       # one process's share
    assert set(split) <= set(seeds)
    assert not set(q.sample_seeds(1, "mimic", 1000)) & set(seeds)


def test_prevalence_of_real_stays() -> None:
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's tokens")
    from icutwin.eval import privacy as pv
    from icutwin.eval import utility as ut
    from icutwin.eval.forecast import Reader
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    rd = Reader(v)
    real = pv.first_windows(T, "mimic", "held_out")
    cut = [np.asarray(s[:len(s) // 2]) for s in real]            # stays cut short: no ICU discharge, no end block
    p = q.prevalence(cut, real, rd)
    assert p["real"]["imv"]["n"] > 0 and 0 <= p["real"]["imv"]["rate"] <= 1
    _w, labels = ut.examples(real, rd)
    assert p["real"]["rrt"]["positives"] == int(np.nansum(labels["rrt"]))
    assert p["real"]["no_icu_discharge"] < 0.1 < p["twins"]["no_icu_discharge"]
    same = q.prevalence(real, real, rd)
    assert same["twins"] == same["real"]
