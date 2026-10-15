"""Paper P amendment 10: the clinical scores against hand-worked cases of their reference definitions (ricu's NEWS,
MEWS, qSOFA, SIRS and AVPU; mimic-code's SAPS II, OASIS and KDIGO creatinine stage)."""

import numpy as np
import pytest

from icutwin.eval import clinical as cl

nan = np.nan


def test_map_vals_upper_inclusive_breaks():
    rr = [8, 9, 11, 12, 20, 21, 24, 25, nan]
    assert cl.map_vals(rr, (8, 11, 20, 24), (3, 1, 0, 2, 3)).tolist() == [3, 1, 1, 0, 0, 2, 2, 3, 0]


def test_avpu_from_gcs():
    out = cl.avpu([2, 3, 4, 9, 10, 13, 14, 15, 16, nan])
    assert np.array_equal(out, [nan, 3, 2, 2, 1, 1, 0, 0, nan, nan], equal_nan=True)


def test_news():
    # RR 22 (2), SpO2 93 (2), oxygen (2), 38.5 C (1), SBP 105 (1), HR 115 (2), GCS 15 (0)
    assert cl.news([22], [93], [1], [38.5], [105], [115], [15]).tolist() == [10]
    # every parameter normal, GCS 13 (V: not alert, 3); nothing observed scores 0
    assert cl.news([16, nan], [97, nan], [0, nan], [37, nan], [120, nan], [70, nan], [13, nan]).tolist() == [3, 0]
    # the extremes: RR 8, SpO2 91, 35 C, SBP 220, HR 131
    assert cl.news([8], [91], [0], [35], [220], [131], [15]).tolist() == [15]


def test_mews():
    # SBP 75 (2), HR 105 (1), RR 25 (2), 38.6 C (2), GCS 12 (V, 1)
    assert cl.mews([75], [105], [25], [38.6], [12]).tolist() == [8]
    assert cl.mews([120], [80], [12], [37], [15]).tolist() == [0]


def test_qsofa_and_sirs():
    assert cl.qsofa([22, 21, nan], [100, 101, nan], [13, 14, nan]).tolist() == [3, 0, 0]
    # 38.5 C, HR 95, RR 18 but PaCO2 30, WBC 13
    assert cl.sirs([38.5], [95], [18], [30], [13]).tolist() == [4]
    assert cl.sirs([37], [80], [16], [nan], [8]).tolist() == [0]


def test_sapsii_worked_case():
    s = cl.sapsii(age=[65], hr_min=[60], hr_max=[130], sbp_min=[85], sbp_max=[140], temp_max=[39.2],
                  pf_vent_min=[150], urine_ml=[800], bun_max=[40], wbc_min=[8], wbc_max=[25], k_min=[4], k_max=[5.5],
                  na_min=[130], na_max=[140], hco3_min=[18], bili_max=[2], gcs_min=[10],
                  surgical_status=["emergency_surgery"])
    assert s.tolist() == [12 + 4 + 5 + 3 + 9 + 4 + 6 + 3 + 3 + 0 + 3 + 0 + 7 + 8]
    assert cl.sapsii_prob(s)[0] == pytest.approx(1 / (1 + np.exp(-(-7.7631 + 0.0737 * 67 + 0.9971 * np.log(68)))))
    empty = cl.sapsii(*([[nan]] * 18), surgical_status=["unknown"])
    assert empty.tolist() == [0]


def test_oasis_worked_case():
    s = cl.oasis(pre_icu_minutes=[120], age=[70], gcs_min=[12], hr_min=[80], hr_max=[110], map_min=[55], map_max=[100],
                 rr_min=[15], rr_max=[25], temp_min=[36.5], temp_max=[38.0], urine_ml=[1200], ventilated=[1],
                 surgical_status=["nonsurgical"])
    assert s.tolist() == [3 + 6 + 4 + 3 + 2 + 1 + 2 + 5 + 9 + 6]
    assert cl.oasis_prob(s)[0] == pytest.approx(1 / (1 + np.exp(-(-6.1746 + 0.1275 * 41))))
    # elective surgery scores 0, an unknown status 0 (mimic-code's NULL)
    base = dict(pre_icu_minutes=[600], age=[30], gcs_min=[15], hr_min=[70], hr_max=[80], map_min=[70], map_max=[90],
                rr_min=[14], rr_max=[18], temp_min=[36.5], temp_max=[36.8], urine_ml=[3000], ventilated=[0])
    assert cl.oasis(**base, surgical_status=["elective_surgery"]).tolist() == [3]
    assert cl.oasis(**base, surgical_status=["unknown"]).tolist() == [3]


def test_saps3_prob():
    assert cl.saps3_prob([50])[0] == pytest.approx(1 / (1 + np.exp(32.6659 - np.log(70.5958) * 7.3068)))


def test_kdigo_creatinine():
    assert cl.kdigo_creatinine([2.0, 1.4, 4.5, 1.0, 3.0], [1.0, 1.0, 4.1, nan, 2.0],
                               [0.9, 1.2, 3.0, nan, 1.0]).tolist() == [2, 1, 3, 0, 3]


def test_indicators():
    assert cl.shock_index([100, 80], [100, 0]).tolist()[0] == 1.0
    assert np.isnan(cl.shock_index([100, 80], [100, 0])[1])
    assert cl.rox([96], [0.4], [20]).tolist() == [12.0]


def test_auroc_fast_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    from icutwin.eval import bench

    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(20, 400))
        y = (rng.random(n) < 0.3).astype(float)
        s = np.round(rng.normal(size=n) + y, 1)          # ties
        w = rng.uniform(0.1, 3.0, n)
        if 0 < y.sum() < n:
            assert bench.auroc_fast(y, s, w) == pytest.approx(roc_auc_score(y, s, sample_weight=w))
            assert bench.auroc_fast(y, s) == pytest.approx(roc_auc_score(y, s))


def test_resamples_keep_groups_within_strata():
    from icutwin.eval import bench

    groups = np.array(["a", "a", "b", "c", "c", "c", "d"])
    strata = np.array([0, 0, 0, 1, 1, 1, 1])
    for idx in bench.resamples(groups, n=50, seed=1, strata=strata):
        assert (strata[idx] == 0).sum() in (2, 3, 4)    # two draws of a (2 rows) or b (1 row)
        drawn = groups[idx]
        for g in ("a", "c"):                            # a group's rows come together
            assert (drawn == g).sum() % {"a": 2, "c": 3}[g] == 0
