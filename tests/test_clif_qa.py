"""S3 cross-dataset QA flags on hand-made per-dataset statistics."""

from __future__ import annotations

from icutwin.clif.qa import _flags


def _ds(median: float, coverage: float, n: int = 100) -> dict:
    return {"tables": {"labs": {"free_t4": {"n": n, "coverage": coverage,
                                            "q": [median * 0.5, median * 0.8, median, median * 1.2, median * 2]}}},
            "coverage": {"labs": {"free_t4": coverage}}}


def test_unit_error_is_flagged_and_case_mix_is_not() -> None:
    stats = {"a": _ds(1.2, 0.4), "b": _ds(1.1, 0.5), "c": _ds(1.3, 0.45), "d": _ds(0.1, 0.4)}   # d: wrong factor
    flags = _flags(stats, min_n=20, min_datasets=3, ratio=3.0, gap=0.2, min_coverage=0.3)
    assert {(f["dataset"], f["flag"]) for f in flags} == {("d", "median_ratio"), ("d", "iqr_disjoint")}


def test_coverage_gap_needs_a_well_covered_reference() -> None:
    stats = {"a": _ds(1.0, 0.6), "b": _ds(1.0, 0.7), "c": _ds(1.0, 0.65), "d": _ds(1.0, 0.05)}
    flags = _flags(stats, min_n=20, min_datasets=3, ratio=3.0, gap=0.2, min_coverage=0.3)
    assert [(f["dataset"], f["flag"]) for f in flags] == [("d", "coverage_gap")]
    rare = {k: _ds(1.0, v) for k, v in {"a": 0.1, "b": 0.12, "c": 0.08, "d": 0.0}.items()}
    assert _flags(rare, min_n=20, min_datasets=3, ratio=3.0, gap=0.2, min_coverage=0.3) == []
