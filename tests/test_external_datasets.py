"""Amendment 16a's plumbing for inference-only databases: dataset names from stay ids, unit-less medication tables,
several splits at once (no real rows)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from icutwin.clif.external import _all_null
from icutwin.eval import bench
from icutwin.stage.spec import stay_dataset


def test_stay_dataset_names_only_external_databases() -> None:
    assert stay_dataset("unknown", "ricd:10_1@24#3") == "ricd"
    assert stay_dataset("", "zigong:11_1#real") == "zigong"
    assert stay_dataset("unknown", "mimic:300@24#1") == "unknown"     # a training database prompted as unknown
    assert stay_dataset("eicu", "ricd:10_1") == "eicu"                # a decoded name is kept
    assert stay_dataset("unknown", "twin:7") == "unknown"


def test_all_null_detects_unit_less_tables(tmp_path: Path) -> None:
    pq.write_table(pa.table({"med_dose_unit": pa.array([None, None], pa.string())}), tmp_path / "a.parquet")
    pq.write_table(pa.table({"med_dose_unit": pa.array([None, "mg"], pa.string())}), tmp_path / "b.parquet")
    pq.write_table(pa.table({"med_dose_unit": pa.array([], pa.string())}), tmp_path / "c.parquet")
    assert _all_null(tmp_path / "a.parquet", "med_dose_unit")
    assert not _all_null(tmp_path / "b.parquet", "med_dose_unit")
    assert not _all_null(tmp_path / "c.parquet", "med_dose_unit")    # no rows: CLIF_MEDS runs as before
    assert not _all_null(tmp_path / "a.parquet", "other")


def test_folds_and_cross_fitted_recalibration() -> None:
    import sys

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from paper_external import folds_of, recalibrate

    pats = [f"p{i}" for i in range(50)]
    fo = folds_of(pats)
    assert sorted(set(fo.values())) == [0, 1, 2, 3, 4] and fo == folds_of(list(reversed(pats)))
    rng = np.random.default_rng(0)
    z = rng.normal(size=2000)
    y = (rng.random(2000) < 1 / (1 + np.exp(-(0.5 + z)))).astype(float)
    p = 1 / (1 + np.exp(-(z - 1.5)))                       # miscalibrated: too low by 2 on the logit
    fold = np.arange(2000) % 5
    q = recalibrate(y, p, np.ones(2000), fold)
    assert np.isfinite(q).all()
    assert abs(y.mean() / q.mean() - 1) < 0.05 < abs(y.mean() / p.mean() - 1)


def test_ext17_split_is_deterministic_and_proportioned() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from paper_ext17 import slice_of

    a = [slice_of("ricd", str(i)) for i in range(20000)]
    assert a == [slice_of("ricd", str(i)) for i in range(20000)]           # a pure function of database and patient
    share = {s: a.count(s) / len(a) for s in ("test", "ft", "label")}
    assert abs(share["test"] - 0.50) < 0.02 and abs(share["ft"] - 0.35) < 0.02 and abs(share["label"] - 0.15) < 0.02
    assert [slice_of("zigong", str(i)) for i in range(50)] != a[:50]       # the database enters the hash


def test_ext17_trainroot_helpers() -> None:
    import sys

    import pytest

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from paper_ext17 import site_unknown, val_of

    held = [val_of("ricd", str(i)) for i in range(20000)]
    assert abs(sum(held) / len(held) - 0.10) < 0.01 and held == [val_of("ricd", str(i)) for i in range(20000)]
    r = {"stay_id": "ricd:1_1", "ids": [1, 17, 227, 9, 8], "goal_ids": [1, 17, 227, 5], "windows": [[1, 17, 227, 9], [1, 17, 227, 8]]}
    out = site_unknown(r, 2, 227, 228)
    assert out["ids"] == [1, 17, 228, 9, 8] and out["goal_ids"] == [1, 17, 228, 5]
    assert out["windows"] == [[1, 17, 228, 9], [1, 17, 228, 8]] and r["ids"][2] == 227      # the input is not changed
    with pytest.raises(ValueError):
        site_unknown({**r, "windows": [[1, 17, 9, 227]]}, 2, 227, 228)


def test_ext17_recalibration_full_and_intercept_only() -> None:
    import sys

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from paper_ext17 import _logit, recal_apply, recal_fit

    rng = np.random.default_rng(1)
    z = rng.normal(-2, 1, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-(1.0 + 0.5 * z)))).astype(float)
    p = 1 / (1 + np.exp(-z))
    m = recal_fit(y, p, np.ones(4000))
    assert not m["intercept_only"] and abs(m["a"] - 1.0) < 0.2 and abs(m["b"] - 0.5) < 0.1
    few = np.zeros(200)
    few[:5] = 1                                                   # 5 positives: the intercept alone
    q = np.full(200, 0.1)
    m2 = recal_fit(few, q, np.ones(200))
    assert m2["intercept_only"] and m2["b"] == 1.0
    assert abs(recal_apply(m2, q).mean() - 0.025) < 1e-6           # the observed rate, the ranks unchanged
    assert np.allclose(_logit(recal_apply(m2, np.array([0.05, 0.2]))) - _logit([0.05, 0.2]), m2["a"])
    assert recal_fit(np.zeros(50), q[:50], np.ones(50)) is None   # no positive: no map
    assert np.isnan(recal_apply(None, q)).all()


def test_real_rows_reads_several_splits(tmp_path: Path) -> None:
    for sp, sid in (("train", "ricd:1_1"), ("held_out", "ricd:2_1")):
        (tmp_path / "ricd").mkdir(exist_ok=True)
        pq.write_table(pa.table({"stay_id": [sid], "ids": pa.array([[1, 2]], pa.list_(pa.int32()))}),
                       tmp_path / "ricd" / f"{sp}.parquet")
    rows = bench.real_rows(tmp_path, "train,held_out", {"ricd:1_1", "ricd:2_1"}, ("ricd",))
    assert [r["stay_id"] for r in rows] == ["ricd:1_1", "ricd:2_1"]
    assert [r["stay_id"] for r in bench.real_rows(tmp_path, "held_out", {"ricd:1_1", "ricd:2_1"}, ("ricd",))] \
        == ["ricd:2_1"]


def test_backoff_rewrites_only_codes_the_vocabulary_lacks(monkeypatch) -> None:
    from types import SimpleNamespace

    from icutwin.tokenization import backoff as b

    kept = ("MED_CON//propofol//ml/hr//start//iv", "MED_CON//propofol//ml/hr//stop//iv",
            "MED_INT//ext_enoxaparin//mcg//given//im", "MED_INT//sodium_chloride//UNK//given//iv",
            "MED_INT//sodium_chloride//ml//given//iv", "OUTPUT//gi_output_other", "LAB//sodium//mmol/l//bmp")
    v = SimpleNamespace(codes={**{c: {"kind": "numeric"} for c in kept}, "MED_CON//OOV": {"kind": "oov"}})
    r = b.load("default")
    assert b.backoff_code("LAB//sodium//mmol/l//bmp", 140.0, v, r) == ("LAB//sodium//mmol/l//bmp", 140.0, "kept")
    assert b.backoff_code("MED_CON//propofol//UNK//start//iv", None, v, r)[:2] == \
        ("MED_CON//propofol//ml/hr//start//iv", None)
    assert b.backoff_code("MED_CON//propofol//UNK//stop//iv", 0.0, v, r)[:2] == ("MED_CON//propofol//ml/hr//stop//iv", 0.0)
    c, n, _ = b.backoff_code("MED_INT//ext_enoxaparin//u//given//im", 4000.0, v, r)
    assert c == "MED_INT//ext_enoxaparin//mcg//given//im" and abs(n - 40000.0) < 1e-6      # 100 IU = 1 mg
    assert b.backoff_code("MED_INT//sodium_chloride//mg//given//iv", 900.0, v, r)[:2] == \
        ("MED_INT//sodium_chloride//UNK//given//iv", None)                                # mass to volume: no dose
    assert b.backoff_code("OUTPUT//stool", 200.0, v, r)[:2] == ("OUTPUT//gi_output_other", 200.0)
    assert b.backoff_code("MED_INT//polymyxin_b//mcg//given//iv", 5.0, v, r)[2] == "none"  # no kept variant
    monkeypatch.delenv(b.ENV, raising=False)
    s = SimpleNamespace(timeline=[(0, 0, "OUTPUT//stool", 1.0, None)])
    assert b.rewrite(s, v) is s                                                           # off by default
