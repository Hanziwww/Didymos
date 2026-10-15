"""The bank producer (S7): refuses without a passed M7 gate for the same weights; writes reproducible, filtered
shards with a manifest; resumes where it stopped; caps the ring. The sampler is a stand-in (real held-out token
sequences of the 2% subset, picked by seed), so no model and no synthetic data are involved."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from icutwin.generate import bank
from icutwin.generate import producer as pr

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"


def _gate_dir(tmp: Path, status: str, sha: str) -> Path:
    d = tmp / "audit"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps({"gate": status}))
    (d / "p1.json").write_text(json.dumps({"model": {"weights_sha256": sha}}))
    return d


def test_owner_exemption_covers_only_inconclusive_for_its_weights_and_scope(tmp_path: Path) -> None:
    model = tmp_path / "hf"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    sha = pr.weights_sha256(model)
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps({"gate": "INCONCLUSIVE", "weights_sha256": sha, "scopes": ["paper_rollouts"]}))
    inc = _gate_dir(tmp_path / "a", "INCONCLUSIVE", sha)
    with pytest.raises(pr.GateNotPassed):
        pr.require_gate_or_exemption(inc, model, None, "paper_rollouts")                # no exemption
    got = pr.require_gate_or_exemption(inc, model, ex, "paper_rollouts")
    assert got["gate"] == "INCONCLUSIVE (study lead's exemption)"
    with pytest.raises(pr.GateNotPassed):
        pr.require_gate_or_exemption(inc, model, ex, "bank")                            # another use
    with pytest.raises(pr.GateNotPassed):
        pr.require_gate_or_exemption(_gate_dir(tmp_path / "b", "FAIL", sha), model, ex, "paper_rollouts")
    other = tmp_path / "ex2.json"
    other.write_text(json.dumps({"gate": "INCONCLUSIVE", "weights_sha256": "0" * 64, "scopes": ["paper_rollouts"]}))
    with pytest.raises(pr.GateNotPassed):
        pr.require_gate_or_exemption(inc, model, other, "paper_rollouts")               # other weights
    with pytest.raises(pr.GateNotPassed):
        pr.require_gate(inc, model)                                                     # the interlock unchanged
    assert pr.require_gate_or_exemption(_gate_dir(tmp_path / "c", "PASS", sha), model, None, "x")["gate"] == "PASS"


def test_gate_interlock(tmp_path: Path) -> None:
    model = tmp_path / "hf"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    sha = pr.weights_sha256(model)
    with pytest.raises(pr.GateNotPassed):
        pr.require_gate(tmp_path / "none", model)
    with pytest.raises(pr.GateNotPassed):
        pr.require_gate(_gate_dir(tmp_path / "a", "INCONCLUSIVE", sha), model)
    with pytest.raises(pr.GateNotPassed):
        pr.require_gate(_gate_dir(tmp_path / "b", "PASS", "0" * 64), model)      # another model's gate
    assert pr.require_gate(_gate_dir(tmp_path / "c", "PASS", sha), model)["gate"] == "PASS"
    with pytest.raises(pr.GateNotPassed):
        pr.produce(lambda r, s: [], bank.GenerationPlan({"x": 1}, {"x": 1.0}), tmp_path / "bank", seed=0,
                   shard_size=1, event=np.zeros(1, bool), gate={"gate": "FAIL"}, model={})


@pytest.fixture(scope="module")
def pool():
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's token shards")
    from icutwin.eval import privacy as pv
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    event = np.array([m.get("cls") == "event" for m in v.meta])
    return {ds: pv.first_windows(T, ds, "held_out") for ds in ("eicu", "mimic")}, event


def _sampler(pool, calls):
    def sample(requests, seeds):
        calls.append(len(seeds))
        out = []
        for r, s in zip(requests, seeds):
            p = pool[r]
            out.append(p[(s * 7919) % len(p)])       # a stand-in: a held-out stay picked by the seed
        return out
    return sample


def test_shards_manifest_resume_and_ring(pool, tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    stays, event = pool
    plan = bank.GenerationPlan.from_mix({"eicu": 600, "mimic": 300}, 90, "flattened")
    gate = {"gate": "PASS", "weights_sha256": "x"}
    calls = []
    kw = dict(seed=3, shard_size=20, event=event, gate=gate, model={"name": "stand-in"}, request=lambda ds: ds)
    man = pr.produce(_sampler(stays, calls), plan, tmp_path / "bank", **kw)
    assert man["complete"] and sum(s["drawn"] for s in man["shards"]) == 90
    names = [s["name"] for s in man["shards"]]
    assert len(names) == sum(-(-q // 20) for q in plan.quotas.values())
    rows = [r for n in names for r in pq.read_table(tmp_path / "bank" / f"{n}.parquet").to_pylist()]
    assert len(rows) == sum(s["kept"] for s in man["shards"])
    assert len({r["seed"] for r in rows}) == len(rows) and min(r["seed"] for r in rows) >= 3 * pr.SEED_STRIDE
    assert len({bank.seq_hash(r["ids"]) for r in rows}) == len(rows)                 # no duplicate across the bank
    assert all(abs(r["weight"] - plan.weights[r["dataset"]]) < 1e-12 for r in rows)
    # resume: nothing is drawn again; a lost shard is redrawn identically
    first = (tmp_path / "bank" / f"{names[0]}.parquet").read_bytes()
    man2 = pr.produce(_sampler(stays, calls2 := []), plan, tmp_path / "bank", **kw)
    assert calls2 == [] and man2["shards"] == man["shards"]
    m = json.loads((tmp_path / "bank" / "manifest.json").read_text())
    m["shards"] = m["shards"][1:]
    (tmp_path / "bank" / "manifest.json").write_text(json.dumps(m))
    (tmp_path / "bank" / "hashes.txt").write_text("")
    pr.produce(_sampler(stays, calls3 := []), plan, tmp_path / "bank", **kw)
    assert calls3 == [20] and (tmp_path / "bank" / f"{names[0]}.parquet").read_bytes() == first
    # ring: at most 2 shards kept, the oldest evicted
    ring = pr.produce(_sampler(stays, []), plan, tmp_path / "ring", **{**kw, "seed": 4}, ring_shards=2)
    assert len(ring["shards"]) == 2 and len(ring["evicted"]) == len(names) - 2 and ring["complete"]
    assert sorted(p.stem for p in (tmp_path / "ring").glob("*.parquet")) == sorted(s["name"] for s in ring["shards"])
    with pytest.raises(ValueError):
        pr.produce(_sampler(stays, []), plan, tmp_path / "bank", **{**kw, "seed": 5})


def test_oversampled_shards_keep_the_selected_twins(pool, tmp_path: Path) -> None:
    """Round 3's twin-level calibration in the bank: a shard draws ``oversample`` times its twins, the selection keeps
    n (here the shortest), the filters follow, and every kept twin records its own seed; an existing bank refuses
    another oversampling."""
    import pyarrow.parquet as pq

    stays, event = pool
    plan = bank.GenerationPlan({"eicu": 30, "mimic": 20}, {"eicu": 1.0, "mimic": 1.0})
    calls = []
    picked = {}

    def shortest(seqs, ds, n, rng):
        order = np.argsort([len(s) for s in seqs], kind="stable")[:n]
        picked.setdefault(ds, []).append(sorted(len(seqs[i]) for i in order))
        return order

    kw = dict(seed=6, shard_size=10, event=event, gate={"gate": "PASS"}, model={}, request=lambda ds: ds)
    man = pr.produce(_sampler(stays, calls), plan, tmp_path / "bank", oversample=3, select=shortest, **kw)
    assert calls == [30] * 5 and man["oversample"] == 3 and man["complete"]
    for s in man["shards"]:
        assert s["drawn"] == 30 and s["dropped"]["not_selected"] == 20 and s["kept"] <= 10
        rows = pq.read_table(tmp_path / "bank" / f"{s['name']}.parquet").to_pylist()
        lo = 6 * pr.SEED_STRIDE + s["first_index"] * 3
        assert all(lo <= r["seed"] < lo + 30 for r in rows) and len({r["seed"] for r in rows}) == len(rows)
        assert max(len(r["ids"]) for r in rows) <= max(max(p) for p in picked[s["dataset"]])
    with pytest.raises(ValueError):
        pr.produce(_sampler(stays, []), plan, tmp_path / "bank", oversample=2, select=shortest, **kw)
