"""S8 diversity metrics on the 2% subset's real stays (skipped without them): the positive control sits near the
ideal values, and the collapsed, copying and dataset-swapping generators are flagged by the metrics meant for them
(plan *Diversity assurance* 6: a metric that flags no control is not trusted)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from icutwin.eval import fidelity as fd
from icutwin.eval import privacy as pv

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"


@pytest.fixture(scope="module")
def report():
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's token shards")
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    rng = np.random.default_rng(0)
    held, ref, pool = [], [], []
    for ds in ("mimic", "eicu", "hirid", "aumc", "sicdb", "nwicu", "inspire"):
        h = pv.first_windows(T, ds, "held_out")
        t = pv.first_windows(T, ds, "train")
        k = rng.permutation(len(t))
        held += h
        ref += [t[i] for i in k[:len(h)]]
        pool += [t[i] for i in k[len(h):]]
    rep = fd.harness(held, ref, fd.vocabulary(v), {}, seed=0, train=pool)
    print(json.dumps({"positive": rep["positive_control"]["diversity"],
                      **{k: {"flags": x["flags"], "diversity": x["metrics"]["diversity"]}
                         for k, x in rep["negative_controls"].items()}}, indent=1, default=float))
    return rep


def test_positive_control_near_ideal(report) -> None:
    d = report["positive_control"]["diversity"]
    assert 0.85 < d["vendi_ratio"] < 1.15 and 0.85 < d["distinct_ratio"] < 1.15 and 0.8 < d["nn_ratio"] < 1.25
    assert d["duplicate_rate"] < 0.01 and d["copy_rate"] < 0.2
    assert d["rare_code_recall"] > 0.5 and d["mmd_corr"] > 0.9 and d["site_confusion_diff"] < 0.1


def test_generators_flagged_by_their_metrics(report) -> None:
    neg = {k: set(x["flags"]) for k, x in report["negative_controls"].items()}
    assert "copy_rate" in neg["copying"], neg
    assert {"vendi_ratio", "duplicate_rate", "nn_ratio", "distinct_ratio"} <= neg["collapsed"], neg
    assert {"site_confusion_diff", "mmd_corr"} & neg["swapped_datasets"], neg
    assert report["controls_detected"]
