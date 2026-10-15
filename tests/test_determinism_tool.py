"""The S2 determinism check compares CLIF tables of two runs row for row."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from icutwin.clif.tools.determinism import differing_rows


def test_differing_rows_counts_both_directions(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pd.DataFrame({"h": ["1", "1", "2"], "v": [1.0, 1.0, 2.0]}).to_parquet(a / "clif_vitals.parquet")
    pd.DataFrame({"h": ["1", "2", "2"], "v": [1.0, 2.0, 3.0]}).to_parquet(b / "clif_vitals.parquet")
    pd.DataFrame({"h": ["1"]}).to_parquet(a / "clif_labs.parquet")
    pd.DataFrame({"h": ["1"]}).to_parquet(b / "clif_labs.parquet")
    assert differing_rows(a, b) == {"clif_vitals.parquet": (1, 1)}   # a duplicate row counts (EXCEPT ALL)
