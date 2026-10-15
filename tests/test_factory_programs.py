"""Label programs (S2, ``factory.programs``): the catalog resolves (identities exclude names and unused horizons),
every catalog program evaluates on a store built from real held-out token sequences (skipped without them), parity
with M8's ``tasks.labels`` at landmarks, and two properties on random hand-built stays: a label within a horizon
does not change when anything after the horizon changes, and the horizon reduction equals a direct count of onsets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from icutwin.factory import labels as fl
from icutwin.factory import processes as fp
from icutwin.factory import programs as fpr
from icutwin.factory import store as fst
from icutwin.factory import twins
from icutwin.factory.channels import Channels

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "full_r4"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not (T / "vocab.json").exists() or not (T / "mimic" / "held_out.parquet").exists():
        pytest.skip("the full r4 tokens")
    rows = []
    for ds in ("mimic", "eicu", "aumc", "hirid"):
        t = pq.read_table(T / ds / "held_out.parquet", columns=["stay_id", "ids"]).slice(0, 40).to_pylist()
        rows += [{"stay_id": r["stay_id"], "ids": r["ids"], "split": "test"} for r in t]
    root = tmp_path_factory.mktemp("prog")
    fst.build(rows, T / "vocab.json", root / "s", "seg_0000", {"kind": "test"}, workers=1, chunk=80)
    return rows, fst.Store(root / "s")


def test_catalog_identities(built) -> None:
    _rows, st = built
    progs = fpr.catalog(st.channels)
    assert len(progs) > 60 and {p.family for p in progs} == {"F1", "F2", "F3", "F4", "F5", "F6", "F7"}
    hashes = [p.hash() for p in progs]
    assert len(set(hashes)) == len(hashes)
    los = [p for p in progs if p.name == "remaining_los"][0]
    assert "horizon" not in los.spec()
    assert fpr.Program("F7", ("los",), "regression", 999, ("in_icu",)).hash() == los.hash()
    hypo = [p for p in progs if p.name == "hypotension"][0]
    a = hypo.target[1]
    assert a[0] == "cross" and st.channels.edges[st.channels.index[a[1]]][a[3]] == pytest.approx(67.0)


def test_every_program_evaluates(built) -> None:
    _rows, st = built
    ev = fpr.Evaluator(st)
    rows = fl.rolling(st, 60)
    for p in fpr.catalog(st.channels):
        b = ev.evaluate(p, rows)
        assert len(b.y) == len(rows) and b.valid.dtype == bool
        assert np.isfinite(b.y[b.valid]).all(), p.name
        assert np.isnan(b.y[~b.valid]).all() or p.label in ("binary", "tte")
        if p.label == "binary" and b.valid.any():
            assert set(np.unique(b.y[b.valid])) <= {0.0, 1.0}
    dec = [p for p in fpr.catalog(st.channels) if p.name == "decompensation"][0]
    b = ev.evaluate(dec, rows)
    assert 0 < b.y[b.valid].mean() < 0.2 and b.valid.mean() > 0.8


def test_parity_with_m8_at_landmarks(built) -> None:
    from icutwin.factory import cache, tasks
    from icutwin.generate import clif_out
    from icutwin.tokenization.vocab import Vocab

    rows, st = built
    v = Vocab.load(T / "vocab.json")
    feats = [twins.decode_features(r["ids"], v, r["stay_id"]) for r in rows]
    labs = [twins.decode_labels(r["ids"], v, r["stay_id"]) for r in rows]
    c = cache.build(feats, clif_out.derived(labs, Path(st.root) / "_m8"), hours=72)
    ev = fpr.Evaluator(st)
    agree, total = 0, 0
    for trig in (6, 12, 24, 48):
        for h in (6, 12, 24):
            for event in ("imv", "vasopressor", "rrt"):
                idx, y = tasks.labels(c, tasks.TaskSpec("binary", f"{event}_within", trig, h))
                rws = fl.landmark(st, trig * 60)
                b = ev.evaluate(fpr.Program("F4", ("event", event), "binary", h * 60, ("in_icu", "never_before")), rws)
                ours = {int(s): (bool(ok), y_) for s, ok, y_ in zip(rws.stay, b.valid, b.y)}
                for i, yy in zip(idx, y):
                    if int(i) in ours and ours[int(i)][0]:
                        total += 1
                        agree += ours[int(i)][1] == yy
    assert total > 200 and agree / total > 0.99


def _random_store(rng, n=40):
    ch = Channels(["VITAL//map"], ["VITAL"], ["vitals"], ["numeric"], [[55.0, 60.0, 65.0, 70.0, 80.0]], [True])
    stays = []
    for _ in range(n):
        los = int(rng.integers(600, 3000))
        times = np.sort(rng.choice(np.arange(-120, los), size=int(rng.integers(5, 40)), replace=False))
        stays.append({"los": los, "end": str(rng.choice(["alive", "dead", "censored"])),
                      "events": [(int(t), "VITAL//map", float(rng.uniform(50, 90))) for t in times]})
    return ch, stays


def test_future_invariance_and_horizon_identity() -> None:
    rng = np.random.default_rng(0)
    ch, stays = _random_store(rng)
    prog = fpr.Program("F6", ("cross", "VITAL//map", "below", 2, 0, 120), "binary", 240)
    m = fp.MemStore(ch, stays)
    rows = fl.rolling(m, 60)
    base = fpr.Evaluator(m).evaluate(prog, rows)
    for k in range(len(rows)):                                  # change everything after t + h (and before the end)
        i, t = int(rows.stay[k]), int(rows.t[k])
        if not base.valid[k]:
            continue
        mod = [dict(s) for s in stays]
        mod[i] = dict(stays[i], events=[e for e in stays[i]["events"] if e[0] <= t + 240]
                      + [(t + 241 + j, "VITAL//map", 50.0) for j in range(3) if t + 241 + j < stays[i]["los"]])
        other = fpr.Evaluator(fp.MemStore(ch, mod)).evaluate(prog, rows.take([k]))
        assert other.y[0] == base.y[k] and other.valid[0] == base.valid[k]
        if k > 400:
            break
    # the horizon reduction against a direct count of onsets in (t, t + h]
    proc = fpr.Evaluator(m).process(prog.target)
    for k in np.flatnonzero(base.valid)[:300]:
        i, t = int(rows.stay[k]), int(rows.t[k])
        on = proc.on_t[(proc.on_stay == i) & (proc.on_t > t) & (proc.on_t <= min(t + 240, stays[i]["los"]))]
        assert base.y[k] == float(len(on) > 0)
