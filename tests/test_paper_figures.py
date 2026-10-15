"""Paper P's manuscript figures and amendment 13's statistics (``tools/paper_figstyle.py``, ``tools/paper_fig*.py``,
``tools/paper_stats.py``): the design system's rules, the statistics' building blocks, and, where the local copy of
the aggregates exists, every figure built and its numbers checked against the materials' tables."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import paper_figstyle as st  # noqa: E402
import paper_stats as ps  # noqa: E402

RUNS = ROOT / "runs" / "paper"
HAVE_DATA = (RUNS / "f1" / "R128" / "f2.json").exists() and (RUNS / "stats" / "stats.json").exists()
HAVE_TOOLKIT = (ROOT / "runs" / "toolkit" / "v1" / "v1_adjusted.json").exists()
TABLES = ROOT / "docs" / "paper" / "tables"


# ---- statistics ----------------------------------------------------------------------------------------------------

def test_bootstrap_draws_are_the_bootstrap_behind_the_interval() -> None:
    from icutwin.eval import bench

    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(60), 2)
    s = np.repeat(rng.integers(0, 3, 60), 2)
    x = rng.normal(size=120)
    for seed in (0, 5):
        ci = bench.bootstrap(np.mean, g, x, n=300, seed=seed, strata=s)
        d = bench.bootstrap_draws(np.mean, g, x, n=300, seed=seed, strata=s)
        assert np.allclose(np.percentile(d[np.isfinite(d)], [2.5, 97.5]), ci, rtol=0, atol=1e-15)


def test_holm_matches_statsmodels_and_the_clinician_analysis() -> None:
    from clinician_study_analysis import holm as holm_c

    ps_ = [0.01, 0.04, 0.03, 0.2, None, 0.0002]
    adj, flag = ps.holm_bounds(ps_, [False, False, False, False, False, True])
    ref = ps.holm(ps_)
    assert all((a is None and b is None) or abs(a - b) < 1e-12 for a, b in zip(adj, ref, strict=True))
    vals = [p for p in ps_ if p is not None]
    ref_c = holm_c(vals)
    assert np.allclose([a for a in adj if a is not None], ref_c)
    assert flag[5] and not flag[3]            # the floor value sets only its own adjusted value here


def test_benjamini_hochberg_matches_statsmodels_and_marks_bounds() -> None:
    from statsmodels.stats.multitest import multipletests

    ps_ = [0.01, 0.04, 0.03, 0.2, None, 0.0002, 0.0002, 0.9]
    q, flag = ps.bh_bounds(ps_, [False, False, False, False, False, True, False, False])
    ref = multipletests([p for p in ps_ if p is not None], method="fdr_bh")[1]
    assert np.allclose([x for x in q if x is not None], ref, rtol=0, atol=1e-12)
    assert q[4] is None and flag[5]               # a floor value's own q is a bound
    assert not flag[6] and not flag[3] and not flag[7]   # a tied exact value above it is not
    q2, f2 = ps.bh_bounds([0.0002, 0.0002, 0.5], [False, True, False])   # tie order does not matter
    assert f2 == [False, True, False] and np.allclose(q2, [0.0003, 0.0003, 0.5], rtol=0, atol=1e-12)


def test_rank_summary_ranks_and_tests() -> None:
    mat = np.array([[0.9, 0.8, 0.7], [0.85, 0.86, 0.6], [0.95, 0.7, 0.7], [0.8, 0.75, 0.75], [0.88, 0.8, 0.79],
                    [0.9, 0.85, 0.6]])
    r = ps.rank_summary(mat, ["a", "b", "c"], "a", n_boot=2000)
    mean = {p["predictor"]: p["mean_rank"] for p in r["predictors"]}
    assert r["predictors"][0]["predictor"] == "a" and abs(mean["a"] - (1 + 2 + 1 + 1 + 1 + 1) / 6) < 1e-12
    assert abs(mean["c"] - (3 + 3 + 2.5 + 2.5 + 3 + 3) / 6) < 1e-12        # ties take mid-ranks
    assert r["predictors"][0]["firsts"] == 5
    vs_b = next(c for c in r["comparisons"] if c["comparator"] == "b")
    assert vs_b["wins"] == 5 and vs_b["losses"] == 1 and 0 < vs_b["p"] <= 1 and vs_b["p_holm"] >= vs_b["p"]


def test_label_efficiency_interpolates_on_log_n(tmp_path: Path) -> None:
    import json

    for design, progs in (("R", ps.PRIMARY["R"][:1]), ("L24", ps.PRIMARY["L24"][:1])):
        d = tmp_path / "a1" / f"test_{design}"
        d.mkdir(parents=True)
        curve = {"mimic": {"100": 0.70, "300": 0.80, "1000": 0.90, "4000": 0.95},     # crosses 0.85 at 300-1,000
                 "eicu": {"100": 0.95, "300": 0.96},                                    # at most 100
                 "hirid": {"100": 0.60, "300": 0.65, "2500": 0.70}}                     # never
        (d / "baselines.json").write_text(json.dumps({"programs": {progs[0]: {"label_efficiency": curve}}}))
        (d / "a1.json").write_text(json.dumps({"programs": {progs[0]: {"datasets": {k: {"auroc": 0.85}
                                                                                    for k in curve}}}}))
    rows, summary = ps.label_efficiency(tmp_path)
    by = {(r["design"], r["dataset"]): r for r in rows}
    m = by[("R", "mimic")]
    assert m["status"] == "interpolated" and abs(m["n_match"] - 10 ** (np.log10(300) + 0.5 * np.log10(1000 / 300))) < 1e-6
    assert by[("R", "eicu")]["status"] == "at most" and by[("R", "eicu")]["n_match"] == 100
    assert by[("R", "hirid")]["status"] == "not reached" and by[("R", "hirid")]["over_1000"]
    assert summary["cells"] == 6 and summary["not_reached"] == 2


@pytest.mark.skipif(not (ROOT / "runs/paper/figures/internal/clinician_answers/aggregates.json").exists(),
                    reason="no local copy of the clinician analysis's aggregates")
def test_figure8_reads_only_pooled_aggregates() -> None:
    import json

    import paper_fig8

    g = json.loads(paper_fig8.AGGREGATES.read_text(encoding="utf-8"))
    text = json.dumps(g)
    assert "_internal" not in text and "a_per_rater" not in text and "a_item_detection" not in text

    def no_records(x):
        if isinstance(x, dict):
            return all(no_records(v) for v in x.values())
        if isinstance(x, list):
            return len(x) <= 2 and all(isinstance(v, (int, float)) for v in x)
        return True

    assert no_records(g)


def test_bootstrap_p_recovers_a_known_effect_and_a_null() -> None:
    rng = np.random.default_rng(1)
    p_effect, _ = ps.p_boot(rng.normal(0.5, 0.1, 10_000), 0.5)
    p_null, _ = ps.p_boot(rng.normal(0.0, 0.1, 10_000), 0.01)
    assert p_effect == 2 / 10_001 and ps.at_floor(p_effect, 10_000)
    assert p_null > 0.5


def test_p_from_an_interval() -> None:
    assert abs(ps.p_ci(1.96, 0.0, 3.92) - 0.05) < 1e-4
    assert ps.p_ci(0.0, -1.0, 1.0) == 1.0
    assert ps.p_ci(-1.0, -2.0, 0.5) < ps.p_ci(-1.0, -2.0, 2.0) + 1e-12


def test_exact_spearman_permutation() -> None:
    from scipy.stats import spearmanr

    x, y = [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5]
    r, p = ps.spearman_exact(x, y)
    assert abs(r - spearmanr(x, y).statistic) < 1e-12 and 0 < p < 0.2
    assert ps.spearman_exact(x, x)[1] == pytest.approx(2 / 720)


# ---- the design system ----------------------------------------------------------------------------------------------

def test_p_formatting() -> None:
    assert st.p_value(0.0123) == "0.012" and st.p_value(0.41) == "0.41" and st.p_value(0.0004) == "<0.001"
    assert st.p_value(0.0014, bound=True) == "≤0.0014" and st.p_value(0.00261, bound=True) == "≤0.0027"
    assert st.p_text(0.05).endswith("= 0.05") and r"\mathit{P}" in st.p_text(0.05)
    assert st.num(-0.006, 3) == "−0.006" and st.num(0.081, 3, sign=True) == "+0.081"


def test_check_catches_violations() -> None:
    import matplotlib.pyplot as plt

    cv = st.Canvas(st.FULL, 60)
    ax = cv.axes(20, 10, 60, 40)
    ax.plot([0, 1], [0, 1], lw=0.5)
    ax.text(0.5, 0.5, "small", fontsize=5)
    ax.text(0.5, 0.2, "E2 beats R128")
    cv.letter(0, 0, "b")
    issues = st.check(cv)
    plt.close(cv.fig)
    assert any("line 0.5" in i for i in issues)
    assert any("font size 5" in i for i in issues)
    assert any("internal code" in i for i in issues)
    assert any("panel letters" in i for i in issues)
    cv = st.Canvas(st.FULL, 40)
    ax = cv.axes(20, 8, 60, 25)
    ax.plot([0, 1], [0, 1])
    cv.letter(0, 0, "a")
    assert st.check(cv) == []
    plt.close(cv.fig)


def test_the_palette_survives_colour_blindness() -> None:
    names = ("didymos", "lgbm", "tabpfn", "clinical", "learner", "probe", "flag")
    d = st.distinguishable({k: st.PALETTE[k] for k in names})
    for kind in ("normal", "protanopia", "deuteranopia", "tritanopia"):
        assert d[kind]["min"] >= 10.0, (kind, d[kind]["below"])
    outcomes = st.distinguishable(st.OUTCOME)                # the enhanced scheme's outcome hues, all pairs
    for kind in ("normal", "protanopia", "deuteranopia", "tritanopia"):
        assert outcomes[kind]["min"] >= 10.0, (kind, outcomes[kind]["below"])


def test_the_blue_ramp_is_one_hue_with_monotone_lightness() -> None:
    from matplotlib.colors import to_rgb

    steps = [st.BLUE[k] for k in sorted(st.BLUE)]
    lab = st._lab(np.array([to_rgb(c) for c in steps]))
    assert all(a > b for a, b in zip(lab[:, 0], lab[1:, 0], strict=False))        # lighter to darker
    hue = np.degrees(np.arctan2(lab[2:, 2], lab[2:, 1])) % 360                     # the hued steps (300-900)
    assert np.ptp(hue) < 12.0


@pytest.mark.parametrize("name", sorted(st.USE_ARTWORK))
def test_artwork_is_cropped_to_its_content(name: str) -> None:
    from PIL import Image

    x0, y0, x1, y1 = st.artwork_box(name)
    with Image.open(st.ARTWORK / f"{name}.png") as im:
        a = np.asarray(im.convert("RGB")).astype(int)
    h, w, _ = a.shape
    bg = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), axis=0)
    ink = np.abs(a - bg).max(axis=2) > st.ART_TOL
    outside = np.ones(ink.shape, bool)
    outside[y0:y1, x0:x1] = False
    assert ink[outside].sum() <= 3                                       # nothing cut but isolated specks
    rows, cols = np.flatnonzero(ink.sum(axis=1) >= st.ART_MIN_PX), np.flatnonzero(ink.sum(axis=0) >= st.ART_MIN_PX)
    assert (x0, y0) == (max(cols[0] - st.ART_PAD, 0), max(rows[0] - st.ART_PAD, 0))    # tight on every side
    assert (x1, y1) == (min(cols[-1] + 1 + st.ART_PAD, w), min(rows[-1] + 1 + st.ART_PAD, h))
    img = st.artwork_pixels(name, 50.0)
    assert img.width == round(50.0 / 25.4 * st.ART_DPI)
    assert abs(img.height / img.width - (y1 - y0) / (x1 - x0)) < 2 / img.width   # the box's aspect, as placed
    assert abs(st.artwork_height(name, 50.0) - 50.0 * (y1 - y0) / (x1 - x0)) < 1e-9


# ---- the figures, on the local aggregates ---------------------------------------------------------------------------

def _rows(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.reader(f))


@pytest.mark.skipif(not HAVE_DATA, reason="no local copy of runs/paper and its statistics")
@pytest.mark.parametrize("name", ["fig2", "fig3", "fig4"])
def test_figure_builds_and_complies(name: str, tmp_path: Path) -> None:
    import paper_figures

    from paper_figdata import Inputs

    inp = Inputs(RUNS, ROOT / "runs" / "toolkit")
    m = paper_figures.build(name, inp, tmp_path, tmp_path / "preview", cvd=False)
    assert m["checks"] == "pass", m["checks"]
    assert (tmp_path / f"{name}.pdf").exists() and (tmp_path / "source_data" / f"{name}.csv").exists()
    a = (tmp_path / f"{name}.pdf").read_bytes()
    m2 = paper_figures.build(name, inp, tmp_path / "again", tmp_path / "preview2", cvd=False)
    assert (tmp_path / "again" / f"{name}.pdf").read_bytes() == a, "the PDF is not deterministic"
    assert m2["checks"] == "pass"


@pytest.mark.skipif(not HAVE_DATA, reason="no local copy of runs/paper and its statistics")
def test_figure2_numbers_equal_the_materials_tables() -> None:
    from paper_results import NAMES

    import paper_fig2
    from paper_figdata import Inputs

    names = {**NAMES, "remaining_los_gt_7d": "ICU stay > 7 days"}
    rows = paper_fig2.data(Inputs(RUNS, ROOT / "runs" / "toolkit"), st.SourceData("t"))
    f1 = {("R128", r[0]): r for r in _rows(TABLES / "followup" / "f1_01_r128.csv")[1:]}
    f1.update({("L24x", r[0]): r for r in _rows(TABLES / "followup" / "f1_03_l24x.csv")[1:]})
    f2 = {("R128", r[0]): r for r in _rows(TABLES / "followup" / "f2_01_r128.csv")[1:]}
    f2.update({("L24x", r[0]): r for r in _rows(TABLES / "followup" / "f2_04_l24x.csv")[1:]})
    checked = 0
    for r in rows:
        key = (r["subset"], names[r["program"]])
        if r["hazard"]:
            t = f2[key]
            assert f"{r['auroc']:.3f}" == t[5]
            assert t[10].startswith(f"{r['d_lgbm']['diff']:.3f}".replace("-0.000", "0.000"))
        else:
            t = f1[key]
            assert f"{r['auroc']:.3f}" == t[3] and f"{r['prespecified']:.3f}" == t[4]
            assert f"{r['lgbm']:.3f}" == t[5]
            assert t[9].startswith(f"{r['d_lgbm']['diff']:.3f}")
        checked += 1
    assert checked == 13


@pytest.mark.parametrize("name, at_least", [("figure_legends.md", 7), ("supplementary_legends.md", 11)])
def test_legends_are_short_and_free_of_internal_codes(name: str, at_least: int) -> None:
    import re

    path = ROOT / "docs" / "paper" / "manuscript" / name
    if not path.exists():
        pytest.skip("the manuscript drafts are not part of this release")
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"^## ", text, flags=re.M)[1:]
    assert len(sections) >= at_least
    for sec in sections:
        head, _, body = sec.partition("\n")
        title = re.search(r"^\*\*([^*]+)\*\*\s*$", body, flags=re.M)
        if not head.startswith("Figure 8"):
            assert title, head
        rest = body.replace(title.group(0), "", 1) if title else body
        assert len(re.findall(r"\S+", rest)) <= 350, head
        assert not st.INTERNAL.search(rest.replace("FiO2", "").replace("SpO2", "")), head
        assert "owner" not in body, head


@pytest.mark.skipif(not (HAVE_DATA and HAVE_TOOLKIT), reason="no local copy of runs/paper and runs/toolkit")
@pytest.mark.parametrize("name", [f"supp{i}" for i in range(1, 17)])
def test_supplementary_figure_builds_and_complies(name: str, tmp_path: Path) -> None:
    import paper_figures

    from paper_figdata import Inputs

    m = paper_figures.build(name, Inputs(RUNS, ROOT / "runs" / "toolkit"), tmp_path, tmp_path / "preview", cvd=False)
    assert m["checks"] == "pass", m["checks"]
    assert (tmp_path / f"{name}.pdf").exists() and (tmp_path / "source_data" / f"{name}.csv").exists()


def test_internal_figures_stay_out_of_the_public_folders() -> None:
    import json

    import paper_figures

    assert not set(paper_figures.INTERNAL) & {*paper_figures.PUBLIC, *paper_figures.SUPPLEMENTARY}
    public = ROOT / "docs" / "paper" / "figures"
    for f in public.rglob("*"):
        assert f.stem.split(".")[0] not in paper_figures.INTERNAL, f
    for m in public.rglob("manifest.json"):
        assert not set(json.loads(m.read_text(encoding="utf-8"))) & set(paper_figures.INTERNAL), m


@pytest.mark.skipif(not (HAVE_DATA and HAVE_TOOLKIT), reason="no local copy of runs/paper and runs/toolkit")
def test_figure7_type1_note_counts_one_relative_risk() -> None:
    import paper_fig7

    from paper_figdata import Inputs

    inp = Inputs(RUNS, ROOT / "runs" / "toolkit")
    _cv, sd = paper_fig7.figure(inp)
    row = next(r for r in sd.rows if r["series"].startswith("type I error check"))
    rr08 = [r["type1_adjusted"]["estimate"] for r in inp.stats["trials"] if r["cell"].split("|")[2] == "0.8"]
    assert row["cells"] == len(rr08) and row["minimum"] == min(rr08) and row["maximum"] == max(rr08)
    assert row["within_0_04_0_06"] == sum(0.04 <= x <= 0.06 for x in rr08)


@pytest.mark.skipif(not HAVE_DATA, reason="no local copy of runs/paper and its statistics")
def test_printed_p_values_equal_the_statistics() -> None:
    from paper_figdata import Inputs, find, p_holm

    inp = Inputs(RUNS, ROOT / "runs" / "toolkit")
    table = {(r[0], r[1]): r for r in _rows(TABLES / "manuscript" / "stats_family1.csv")[1:]}
    for r in inp.family(1):
        p, bound = p_holm(find(inp.family(1), subset=r["subset"], program=r["program"]))
        shown = st.p_value(p, bound)
        csv_p = table[(r["subset"], r["program"])][8]
        assert shown.replace("≤", "<=") == csv_p or (p is None and csv_p == "-")
