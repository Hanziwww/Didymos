"""The calibration benchmark's rollouts (``factory.oracle``) with real held-out token sequences of the full r4 tokens
standing in for twins (skipped without them): the anchors' design (bands, enrichment, weights, admissible hours,
reproducible), the prompts (grammatical prefixes that end on a timeline event), the token caps, and the batch runner
with a stand-in sampler (rows, manifest, resume, deadline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from icutwin.factory import oracle, twins

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "full_r4"


@pytest.fixture(scope="module")
def bank():
    if not (T / "vocab.json").exists() or not (T / "mimic" / "held_out.parquet").exists():
        pytest.skip("the full r4 tokens")
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    rows = []
    for ds in ("mimic", "eicu", "aumc"):
        t = pq.read_table(T / ds / "held_out.parquet", columns=["stay_id", "ids"]).slice(0, 120).to_pylist()
        rows += [(f"{ds}:{r['stay_id']}", ds, r["ids"]) for r in t]
    stays = [(tw, ds, twins.decode_labels(ids, v, tw)) for tw, ds, ids in rows]
    return v, rows, stays


def test_anchors_design(bank) -> None:
    v, _rows, stays = bank
    a = oracle.choose(stays, v, per_dataset=30, seed=0)
    assert a == oracle.choose(stays, v, per_dataset=30, seed=0)               # reproducible
    assert len({x.anchor for x in a}) == len(a)                               # each (twin, hour) once
    by_ds = {ds: [x for x in a if x.dataset == ds] for ds in ("mimic", "eicu", "aumc")}
    assert all(len(x) == 30 for x in by_ds.values())
    for x in a:
        assert 4 <= x.hours <= x.los_hours - 1 and x.weight > 0 and x.anchor == f"{x.twin}@{x.hours:g}"
        assert not x.enriched or x.flag                                      # the enriched half is flagged
    assert sum(x.enriched for x in a) >= 0.3 * len(a)
    for xs in by_ds.values():              # bands spread evenly (a band short of distinct anchors passes them on)
        counts = np.bincount([x.band for x in xs])
        full = counts[counts >= 30 // len(counts[counts > 0])]
        assert full.max() - full.min() <= 1 or len(xs) < 2 * len(oracle.LOS_BANDS)
    flagged = [x for x in a if x.flag and not x.enriched]
    plain = [x for x in a if not x.flag]
    if flagged and plain:                                                     # enrichment lowers flagged weights
        assert max(x.weight for x in flagged) < min(x.weight for x in plain)


def test_prompts_are_prefixes_of_the_stay(bank) -> None:
    from icutwin.tokenization.fsm import Automaton, compile_tables

    v, rows, stays = bank
    t = compile_tables(v)
    ids = {tw: x for tw, _ds, x in rows}
    a = oracle.choose(stays, v, per_dataset=6, seed=1)
    for x in a:
        p = oracle.prompt(ids[x.twin], v, x.twin, x.hours)
        aut = Automaton(t)
        assert all(aut.step(tok) for tok in p), aut.violation
        assert aut.phase == "tl" and aut.last == "event" and v.id("[END]") not in p
        s = twins.decode_labels(p + [v.id("[END]"), v.id("[EOS]")], v, x.twin)
        last = max((e[0] for e in s.timeline), default=s.icu_in)
        assert (last - s.icu_in).total_seconds() / 3600 <= x.hours + 1e-6


def test_token_caps(bank) -> None:
    v, rows, _stays = bank
    seqs = {}
    for _tw, ds, x in rows:
        seqs.setdefault(ds, []).append(x)
    c24 = oracle.token_caps(seqs, v, hours=24)
    c72 = oracle.token_caps(seqs, v, hours=72)
    assert set(c72) == {"mimic", "eicu", "aumc"} and all(c24[d] <= c72[d] for d in c72) and min(c72.values()) > 50


class Stub:
    """A stand-in sampler: every rollout ends at once (as a grammar-capped rollout would)."""

    def __init__(self, v):
        self.end = [v.id("[END]"), v.id("[EOS]")]
        self.calls = []

    def simulate(self, prefixes, n, seed=0, interventions=None, max_new=None):
        self.calls.append((len(prefixes), n, seed, max_new))
        return [[list(self.end) for _ in range(n)] for _ in prefixes]


def test_runner_with_a_stand_in_sampler(bank, tmp_path: Path) -> None:
    v, rows, stays = bank
    ids = {tw: x for tw, _ds, x in rows}
    a = oracle.choose(stays, v, per_dataset=10, seed=2)
    prompts = [oracle.prompt(ids[x.twin], v, x.twin, x.hours) for x in a]
    caps = {"mimic": 900, "eicu": 700, "aumc": 1200}
    eng = Stub(v)
    man = oracle.run(eng, a, prompts, 4, tmp_path / "r", caps, seed=3, size=4)
    assert man["complete"] and len(eng.calls) == len(oracle.batches(a, 4))
    assert {c[3] for c in eng.calls} == set(caps.values()) and len({c[2] for c in eng.calls}) == len(eng.calls)
    t = [r for f in sorted((tmp_path / "r").glob("*.parquet")) for r in pq.read_table(f).to_pylist()]
    assert len(t) == 4 * len(a) and {r["rollout"] for r in t} == {0, 1, 2, 3}
    assert all(r["continuation"] == eng.end for r in t)
    first = {r["anchor"]: r["prompt"] for r in t}
    assert all(first[x.anchor] == list(map(int, p)) for x, p in zip(a, prompts))
    again = Stub(v)
    assert oracle.run(again, a, prompts, 4, tmp_path / "r", caps, seed=3, size=4)["complete"] and not again.calls
    late = oracle.run(Stub(v), a, prompts, 4, tmp_path / "late", caps, size=4, deadline=0.0)
    assert late["stopped"] == "deadline" and not late["complete"] and not list((tmp_path / "late").glob("*.parquet"))
