"""Paper P amendment 16's analysis code on synthetic inputs only (no real rows): the L48x case-cohort subset, F2's
per-anchor estimates and F4's analysis with knockout added and its per-group outputs saved."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import paper_c2b  # noqa: E402
import paper_f1_subsets  # noqa: E402


def _f4_fixture(d: Path, n_base: int = 3, groups: int = 6, n_roll: int = 48, seed: int = 0) -> None:
    """An F4-style design: per base anchor, the original prompt and one edit per group, with hazards whose death
    increments make each group's generator effect equal to its index (so a score that ranks by index agrees)."""
    rng = np.random.default_rng(seed)
    rows, idx, roll, inc = [], [], [], []
    for b in range(n_base):
        base = f"s{b}@60"
        for g in ["", *[f"g{k}" for k in range(groups)]]:
            i = len(rows)
            rows.append({"anchor": f"{base}|{g}" if g else base, "twin": f"s{b}", "dataset": "mimic", "t_min": 60,
                         "hours": 1.0, "weight": 1.0, "program": "decompensation@1440", "base_anchor": base,
                         "group": g, "prompt_len": 10})
            level = 0.5 if not g else 0.5 - 0.05 * (int(g[1:]) + 1)       # the edit lowers the risk by its index
            for k in range(n_roll):
                idx.append(i)
                roll.append(k)
                row = np.zeros(30)
                row[:24] = level / 24 + rng.normal(0, 1e-4, 24)
                inc.append(row)
    (d / "hazards").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(d / "anchors.parquet", index=False)
    np.savez(d / "hazards/part_0000.npz", anchor_idx=np.array(idx), rollout=np.array(roll),
             inc_death=np.array(inc), inc_imv=np.zeros((len(idx), 30)), edges=np.arange(31, dtype=float))


def test_c2b_knockout_and_dump(tmp_path):
    d = tmp_path / "f4r"
    _f4_fixture(d)
    groups = [f"g{k}" for k in range(6)]
    att = tmp_path / "attrib.jsonl"
    ko = tmp_path / "knockout.jsonl"
    with open(att, "w") as fa, open(ko, "w") as fk:
        for b in range(3):
            base = f"s{b}@60"
            # editing ranks the groups as the generator does; knockout reverses them and skips one group
            fa.write(json.dumps({"program": "decompensation@1440", "anchor": base,
                                 "edit": {g: float(k) for k, g in enumerate(groups)}}) + "\n")
            fk.write(json.dumps({"program": "decompensation@1440", "anchor": base,
                                 "knockout": {g: float(-k) for k, g in enumerate(groups[:-1])}}) + "\n")
    paper_c2b.main(["--f4", str(d), "--attrib", str(att), "--knockout", str(ko), "--work", str(tmp_path / "w"),
                    "--out", str(tmp_path / "out"), "--dump", str(tmp_path / "dump")])
    rep = json.loads((tmp_path / "out/c2b.json").read_text())["programs"]["decompensation@1440"]
    assert rep["agreement"]["edit"]["spearman_median"] == pytest.approx(1.0)
    assert rep["agreement"]["knockout"]["spearman_median"] == pytest.approx(-1.0)
    kv = rep["knockout_vs_edit"]
    assert kv["anchors"] == 3 and kv["edit_median"] == pytest.approx(1.0) and kv["knockout_median"] == pytest.approx(-1.0)
    eff = pd.read_parquet(tmp_path / "dump/effects.parquet")
    assert len(eff) == 3 * 6 and {"effect", "half_a", "half_b", "se"} <= set(eff.columns)
    np.testing.assert_allclose(eff.sort_values(["base_anchor", "group"])["effect"].to_numpy()[:6],
                               0.05 * np.arange(1, 7), atol=1e-3)
    pairs = pd.read_parquet(tmp_path / "dump/knockout_vs_edit.parquet")
    assert (pairs["groups"] == 5).all()                  # the groups both methods score
    per = pd.read_parquet(tmp_path / "dump/agreement_per_anchor.parquet")
    assert set(per["method"]) == {"edit", "knockout"}


def test_c2b_without_amendment16_options_is_unchanged(tmp_path):
    d = tmp_path / "f4"
    _f4_fixture(d)
    att = tmp_path / "attrib.jsonl"
    att.write_text("\n".join(json.dumps({"program": "decompensation@1440", "anchor": f"s{b}@60",
                                         "edit": {f"g{k}": float(k) for k in range(6)}}) for b in range(3)) + "\n")
    paper_c2b.main(["--f4", str(d), "--attrib", str(att), "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")])
    rep = json.loads((tmp_path / "o/c2b.json").read_text())["programs"]["decompensation@1440"]
    assert "knockout" not in rep["agreement"] and "knockout_vs_edit" not in rep
    assert not (tmp_path / "dump").exists()


def test_case_cohort_subset(tmp_path):
    an = pd.DataFrame({"anchor": [f"a{i}" for i in range(40)], "twin": [f"s{i}" for i in range(40)],
                       "t_min": 2880, "hours": 48.0, "weight": 1.0, "dataset": "mimic"})
    y = np.r_[np.ones(10), np.zeros(30)]
    d = tmp_path / "test_L48"
    d.mkdir()
    an.to_parquet(d / "anchors.parquet", index=False)
    pd.DataFrame({"anchor": an["anchor"], "program": "hospital_death@0", "label": y, "p": 0.1}).to_parquet(
        d / "predictions.parquet", index=False)
    s, rep = paper_f1_subsets.case_cohort(d, 12, np.random.default_rng(16), tmp_path)
    assert rep == {"excluded_continued_window_or_label": 0, "anchors": 22, "cases": 10, "controls": 12,
                   "control_weight": 2.5}
    assert (s.loc[s["stratum"] == "case", "weight"] == 1.0).all()
    assert (s.loc[s["stratum"] == "control", "weight"] == 2.5).all()
    # the same seed draws the same controls
    s2, _ = paper_f1_subsets.case_cohort(d, 12, np.random.default_rng(16), tmp_path)
    assert s.equals(s2)
