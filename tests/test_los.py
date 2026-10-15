"""The twins' length marginal (round 2): length bands of real stays, and twin-level resampling to target band shares
(on the 2% subset's real stays, skipped without them)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from icutwin.generate import los

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"


def test_band_labels() -> None:
    assert los.band(3) == "0-24" and los.band(24) == "24-48" and los.band(400) == "336+" and los.band(None) == "none"


def test_resample_matches_the_target_shares() -> None:
    rng = np.random.default_rng(0)
    hours = list(rng.exponential(90, size=3000))                 # too long a tail on purpose
    target = {"0-24": 0.35, "24-48": 0.25, "48-96": 0.2, "96-168": 0.1, "168-336": 0.07, "336+": 0.03}
    keep, rep = los.resample(list(range(3000)), hours, target, 1000, rng)
    assert len(keep) == len(set(keep)) == 1000 and rep["kept"] == 1000
    got = Counter(los.band(hours[i]) for i in keep)
    for k, p in target.items():
        assert abs(got[k] / 1000 - p) < 0.01 or k in rep["shortfall"]
    keep2, rep2 = los.resample(list(range(50)), hours[:50], target, 40, np.random.default_rng(1))
    assert len(keep2) == 40 and sum(rep2["shortfall"].values()) >= 0


def test_bands_of_real_stays() -> None:
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's tokens")
    b = los.bands(T, ["eicu", "mimic"], per_dataset=500)
    for ds, shares in b.items():
        assert abs(sum(shares.values()) - 1) < 1e-9 and shares["0-24"] > 0.1 and shares["none"] < 0.05
