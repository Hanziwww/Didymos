"""Paper P amendment 10: a validity check of ``tools/paper_clinical.py``'s SOFA. At the 24-hour landmark, the window
(the 24 hours before the prediction time) nearly coincides with the first ICU day, so the windowed SOFA should match
``derived/<dataset>/sofa_24h.parquet`` (the pipeline's first-day SOFA from the same CLIF tables). Writes
``<out>/clinical_check.json``: per dataset, the anchors, the correlation, the share identical, the mean absolute
difference and both medians.

    python tools/paper_clinical_check.py --clinical runs/paper/f10/clinical --derived data/derived/full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--derived", required=True)
    a = ap.parse_args(argv)
    import pandas as pd

    v = pd.read_parquet(Path(a.clinical) / "variables.parquet")
    v = v[v["design"] == "L24"]
    rep = {}
    for ds, g in v.groupby("dataset"):
        s = pd.read_parquet(Path(a.derived) / ds / "sofa_24h.parquet").set_index("stay_id")
        m = g.set_index("stay_id").join(s[["sofa_total"]], rsuffix="_day1", how="inner").dropna(
            subset=["sofa_total", "sofa_total_day1"])
        x, y = m["sofa_total"].to_numpy(float), m["sofa_total_day1"].to_numpy(float)
        rep[ds] = {"anchors": int(len(m)),
                   "correlation": float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None,
                   "identical": float(np.mean(x == y)), "mean_abs_diff": float(np.mean(np.abs(x - y))),
                   "median_window": float(np.median(x)), "median_day1": float(np.median(y))}
    (Path(a.clinical) / "clinical_check.json").write_text(json.dumps(rep, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
