"""Shared inputs of Paper P's figures (``tools/paper_figures.py``): the aggregate files, amendment 13's statistics,
and the headline rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import PRIMARY  # noqa: E402

ROWS = [("R128", p) for p in PRIMARY["R"]] + [("L24x", p) for p in (*PRIMARY["L24"], "decompensation@1440")]
E2_PROGRAMS = {"R128": ("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880"),
               "L24x": ("icu_death@-1", "decompensation@1440")}
GROUP = {"R128": "Rolling anchors", "L24x": "24-hour landmark"}
ANCHORS = {"R128": 1050, "L24x": 700}
CLINICAL = ("sofa", "news", "mews", "qsofa", "sirs", "sapsii_prob", "oasis_prob")
LEARNED = ("lgbm_pooled", "lgbm_per_dataset", "logistic_pooled", "tabpfn_v3_5", "tabpfn_v2", "logistic_l1",
           "logistic_enet", "cart", "random_forest", "extra_trees", "naive_bayes", "knn", "mlp")


class Inputs:
    """The aggregate trees (``runs/paper``, ``runs/toolkit``) and the statistics, read once."""

    def __init__(self, root: Path, toolkit: Path):
        self.root, self.toolkit = Path(root), Path(toolkit)
        self._cache: dict = {}
        p = self.root / "stats/stats.json"
        self.stats = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        self.stats_path = p

    def path(self, rel: str, toolkit: bool = False) -> Path:
        return (self.toolkit if toolkit else self.root) / rel

    def json(self, rel: str, toolkit: bool = False, sd=None) -> dict:
        p = self.path(rel, toolkit)
        if sd is not None:
            sd.source(p)
        if p not in self._cache:
            self._cache[p] = json.loads(p.read_text(encoding="utf-8"))
        return self._cache[p]

    def family(self, k) -> list:
        return self.stats.get("families", {}).get(str(k), [])


def find(rows: list, **kw) -> dict | None:
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def at_floor(row: dict | None, key: str = "p") -> bool:
    """True when a bootstrap P is at its floor, 2 / (finite draws + 1): no resample fell on the other side of the
    null, so the P value (and its Holm adjustment) is an upper bound."""
    if not row or row.get(key) is None or not row.get("draws"):
        return False
    return abs(row[key] - 2 / (row["draws"] + 1)) < 1e-12


def p_holm(row: dict | None, key: str = "p_holm", raw: str = "p") -> tuple[float | None, bool]:
    """A row's Holm-adjusted P and whether it is a bound (as amendment 13's script marks it; else from the raw P)."""
    if not row:
        return None, False
    return row.get(key), bool(row.get(f"{key}_bound", at_floor(row, raw)))
