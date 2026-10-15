"""S8 fidelity harness on the 2% subset's real stays (skipped without them): the positive control (train against
held-out) stays near ideal, and every negative control is flagged by the metric meant for it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from icutwin.eval import fidelity as fd
from icutwin.eval import privacy as pv

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"


@pytest.fixture(scope="module")
def stays():
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's token shards")
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    rng = np.random.default_rng(0)
    held, train = [], []
    for ds in ("mimic", "eicu", "hirid", "aumc", "sicdb", "nwicu", "inspire"):
        h = pv.first_windows(T, ds, "held_out")
        t = pv.first_windows(T, ds, "train")
        held += h
        train += [t[i] for i in rng.choice(len(t), size=len(h), replace=False)]
    return fd.vocabulary(v), held, train


def test_controls_are_detected_by_their_metrics(stays) -> None:
    voc, held, train = stays
    rep = fd.harness(held, train, voc, {}, seed=0)
    pos = rep["positive_control"]
    assert pos["discriminator_auc"] < 0.6 and pos["code_jsd"] < 0.05 and pos["coverage"] > 0.8
    neg = {k: set(x["flags"]) for k, x in rep["negative_controls"].items()}
    assert {"discriminator_auc"} & neg["shuffled_times"], neg
    assert {"discriminator_auc"} & neg["shuffled_values"], neg
    assert {"code_jsd_by_dataset_max", "discriminator_auc"} & neg["swapped_datasets"], neg
    assert {"recall", "coverage"} & neg["collapsed"], neg
    assert rep["controls_detected"]
