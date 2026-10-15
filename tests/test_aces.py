"""Label parity between the factory's extraction and the ACES reference implementation (plan *Verification*: label
parity for the task types ACES supports), on the 2% subset's real held-out stays (skipped without them): the same
cohort and the same label for every stay, per task."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from icutwin.factory import aces as fa
from icutwin.factory import cache as fc
from icutwin.factory import tasks as ft

R = Path(__file__).resolve().parents[1] / "data"
DATASETS = ("mimic", "eicu", "hirid", "aumc", "sicdb", "nwicu", "inspire")


@pytest.fixture(scope="module")
def built():
    meds = R / "meds" / "subset"
    if not (meds / "mimic" / "metadata" / "stays.parquet").exists():
        pytest.skip("the subset's MEDS")
    from icutwin.tokenization.stays import iter_stays, load_derived

    stays, derived = [], {"episodes": {}, "kdigo": {}, "sofa_24h": {}}
    for ds in DATASETS:
        d = load_derived(R / "derived" / "subset" / ds)
        for k in derived:
            derived[k].update(d[k])
        stays += list(iter_stays(meds / ds, "held_out", R / "derived" / "subset" / ds))
    c = fc.build(stays, derived, hours=72, n_concepts=8)
    c.onset.update(fa.code_onsets(stays))
    files = sorted(f for ds in DATASETS for f in (meds / ds / "data" / "held_out").glob("*.parquet"))
    return c, np.array([s.subject_id for s in stays]), files


TASKS = [("death", "death", 24, 0), ("in_icu_at", "icu_discharge_within", 24, 24),
         ("imv_within", "imv_within", 24, 24), ("vasopressor_within", "vasopressor_within", 12, 24),
         ("rrt_within", "rrt_within", 24, 48)]


@pytest.mark.parametrize("ours,theirs,t,h", TASKS)
def test_labels_match_aces(built, ours: str, theirs: str, t: int, h: int) -> None:
    c, sids, files = built
    spec = ft.TaskSpec("binary", ours, trigger=t, horizon=h)
    idx, y = ft.labels(c, spec)
    mine = dict(zip(sids[idx].tolist(), y.astype(bool).tolist()))
    ref = fa.extract(fa.config(theirs, t, h), files)
    if theirs == "icu_discharge_within":
        ref = {k: not v for k, v in ref.items()}          # still in the ICU at t + h
    if ours == "death":                                   # ours: stays with a known hospital outcome only
        ref = {k: v for k, v in ref.items() if k in mine}
    missing, extra = set(mine) - set(ref), set(ref) - set(mine)
    wrong = [k for k in set(mine) & set(ref) if mine[k] != ref[k]]
    assert len(mine) > 50 and sum(mine.values()) > 0
    assert not missing and not extra and not wrong, (len(missing), len(extra), len(wrong), sorted(wrong)[:5])
