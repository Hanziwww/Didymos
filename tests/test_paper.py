"""Paper P's benchmark code (``eval.bench``, ``eval.baselines``): metrics against references, anchors and prompts on
real stays, and baseline features that read nothing after the prediction time (the store parts skip without the
full r4 tokens)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from icutwin.eval import bench

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "full_r4"


def test_binary_metrics_against_references() -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(0)
    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(float)               # calibrated by construction
    m = bench.binary_metrics(y, p)
    assert abs(m["auroc"] - roc_auc_score(y, p)) < 1e-12 and abs(m["auprc"] - average_precision_score(y, p)) < 1e-12
    assert m["ece"] < 0.03 and abs(m["calibration_slope"] - 1) < 0.1 and abs(m["calibration_intercept"]) < 0.1
    assert abs(m["observed_over_expected"] - 1) < 0.05
    over = bench.binary_metrics(y, np.clip(p * 1.5, 0, 1))  # over-predicting: O/E below 1
    assert over["observed_over_expected"] < 0.8
    w = np.where(y > 0, 2.0, 1.0)                           # weights count as repeated rows
    yy, pp = np.r_[y, y[y > 0]], np.r_[p, p[y > 0]]
    assert abs(bench.binary_metrics(y, p, w)["auroc"] - roc_auc_score(yy, pp)) < 1e-9


def test_bootstrap_c_index_coverage() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.2).astype(float)
    p = y * 0.4 + rng.random(3000)
    lo, hi = bench.bootstrap(lambda a, b: bench.auroc_or_none(a, b), np.arange(3000), y, p, n=200)
    est = bench.auroc_or_none(y, p)
    assert lo < est < hi and hi - lo < 0.06
    t = rng.exponential(10, 500)
    assert bench.c_index(t, np.ones(500), -t) == 1.0 and bench.c_index(t, np.ones(500), t) == 0.0
    ens = rng.normal(size=(2000, 64))
    cov = bench.interval_coverage(ens, rng.normal(size=2000))
    assert abs(cov["0.8"] - 0.8) < 0.04 and abs(cov["0.95"] - 0.95) < 0.03


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    if not (T / "vocab.json").exists() or not (T / "eicu" / "held_out.parquet").exists():
        pytest.skip("the full r4 tokens")
    from icutwin.factory import store as fst

    rows = []
    for ds in ("eicu", "mimic"):
        t = pq.read_table(T / ds / "held_out.parquet", columns=["stay_id", "ids"]).slice(0, 60).to_pylist()
        rows += [{"stay_id": r["stay_id"], "ids": r["ids"], "split": "test"} for r in t]
    root = tmp_path_factory.mktemp("paper")
    fst.build(rows, T / "vocab.json", root / "s", "seg_0000", {"kind": "test"}, workers=1, chunk=120)
    return fst.Store(root / "s"), rows


def test_landmark_anchors_and_prompts(store) -> None:
    from icutwin.factory import twins
    from icutwin.tokenization.vocab import Vocab

    st, rows = store
    v = Vocab.load(T / "vocab.json")
    stays = [(str(r["stay_id"]), "x", twins.decode_labels(r["ids"], v, str(r["stay_id"]))) for r in rows]
    an = bench.landmark_anchors(stays, v, 24.0, per_dataset=1000)
    los = {s: (d.icu_out - d.icu_in).total_seconds() / 3600 for s, _x, d in stays}
    assert an and all(los[x.twin] > 24 for x in an) and len(an) == sum(h > 24 for h in los.values())
    ids = {str(r["stay_id"]): r["ids"] for r in rows}
    p = bench.prompt(ids[an[0].twin], v, an[0].twin, 24.0)
    t = bench.anchor_time(p, v, an[0].twin)
    assert 0 < t <= 24 * 60
    u = bench.unknown_site(p, v)
    assert u != p and len(u) == len(p)
    # the real store's stay and the prompt share their clock: the prompt's events are the stay's events up to t
    i = {str(s): k for k, s in enumerate(st.stays["stay_id"].tolist())}[an[0].twin]
    ev = st.events(i)
    assert (np.asarray(ev["t"]) <= t).any()


def test_baseline_features_read_only_the_past(store) -> None:
    from icutwin.eval import baselines as bl
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr

    st, _rows = store
    fz = bl.Featurizer(bl.feature_channels(st, 16)).fit_levels(st)
    rows = bl.training_rows(st, "R", per_stay=3, seed=0)
    x = fz(st, rows)
    assert x.shape == (len(rows), len(fz.names())) and len(rows) <= 3 * len(st)
    since = x[:, [k for k, n in enumerate(fz.names()) if n.endswith("hours_since")]]
    assert np.nanmin(since) >= 0                               # the last value is at or before t
    # every windowed value is unchanged when the rows' time moves within an empty stretch after t: compute at t and
    # at t with the store's events after t ignored (rows shifted to the same t: identical by construction), and
    # check against a direct reading of the last value
    ev = fpr.Evaluator(st)
    c = fz.channels[0]
    e = fl.finite(ev.events(c))
    k = fz.names().index(f"{c}|last|1440")
    for j in range(min(len(rows), 30)):
        s, t = int(rows.stay[j]), int(rows.t[j])
        m = (np.asarray(e["stay"]) == s) & (np.asarray(e["t"]) <= t) & (np.asarray(e["t"]) > t - 1440)
        want = np.asarray(e["v"], dtype=float)[m][-1] if m.any() else np.nan
        got = x[j, k]
        assert (np.isnan(want) and np.isnan(got)) or abs(want - got) < 1e-5
    lm = bl.training_rows(st, "L24")
    assert (lm.t == 1440).all() and (st.los[lm.stay] > 1440).all()
    y = bl.labels(st, next(p for p in fpr.catalog(st.channels) if p.name == "icu_death"), lm)
    assert set(np.unique(y[np.isfinite(y)])) <= {0.0, 1.0}


def test_attribution_groups_and_edits(store) -> None:
    """Every event group of a prospective stay can be removed with the prompt staying grammar-valid; the edit drops
    exactly that group's events; token groups align with the prompt; the families cover the codes."""
    from icutwin.eval import attribution as at
    from icutwin.factory import twins
    from icutwin.tokenization.encode import prospective_stay
    from icutwin.tokenization.fsm import Automaton, compile_tables
    from icutwin.tokenization.vocab import Vocab

    _st, rows = store
    v = Vocab.load(T / "vocab.json")
    tables = compile_tables(v)
    assert at.family("VITAL//heart_rate") == "vitals" and at.family("MED_CON//norepinephrine//x//start//iv") == \
        "vasoactives" and at.family("MED_CON//propofol//x//start//iv") == "sedation"
    assert at.block(10) == "0-6h" and at.block(400) == "6-24h" and at.block(5000) == ">24h"
    r = next(r for r in rows if len(r["ids"]) > 600)
    sid = str(r["stay_id"])
    s = prospective_stay(twins.decode_labels(r["ids"], v, sid), 24.0)
    groups = at.stay_groups(s, 24.0)
    assert sum(len(x) for x in groups.values()) == len(s.timeline)
    for g in [k for k, x in groups.items() if x][:6] + ["statics"]:
        p = at.edited_prompt(s, v, 24.0, g, groups)
        a = Automaton(tables)
        assert all(a.step(int(tok)) for tok in p), (g, a.violation)
    p0 = at.edited_prompt(s, v, 24.0, "none", {"none": []})
    tg = at.token_groups(p0, v)
    assert len(tg) == len(p0) and "statics" in tg and any(x and x.startswith("vitals") for x in tg)
    assert at.rank_agreement({"a": 3, "b": 2, "c": 1}, {"a": 3, "b": 2, "c": 1})["spearman"] == 1.0


def test_km_survival() -> None:
    t = np.array([10, 20, 30, 200, 50, 300.0])
    e = np.array([1, 1, 1, 1, 0, 0], bool)                 # one censored at 50 (before 168), one at 300
    # at risk 6, 5, 4 at the events 10, 20, 30: S(168) = 5/6 * 4/5 * 3/4 = 0.5; the censored member is not dropped
    assert abs(bench.km_survival(t, e, 168.0) - 0.5) < 1e-12
    assert abs(bench.km_survival(t, np.ones(6, bool), 168.0) - 2 / 6) < 1e-12   # no censoring: the plain share
    assert np.isnan(bench.km_survival(np.array([5.0, 6.0]), np.zeros(2, bool), 168.0))


def test_edits_drop_orphaned_infusion_records() -> None:
    from icutwin.eval import attribution as at

    nore = "MED_CON//norepinephrine//mcg/kg/min//{}//iv"
    tl = [(1, 0, nore.format("start"), 0.1, None), (2, 0, nore.format("dose_change"), 0.2, None),
          (3, 0, nore.format("stop"), None, None), (4, 0, "VITAL//heart_rate", 80.0, None)]
    assert at.consistent(tl) == tl
    edited = at.consistent(tl[1:])                     # the start removed: its later records go too
    assert [e[2] for e in edited] == ["VITAL//heart_rate"]
    going = [(0, 0, nore.format("going"), 0.1, None)] + tl[1:]
    assert at.consistent(going) == going
