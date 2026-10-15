"""The twin bank's producer (plan S7 ``bank.py``: sharded, seeded, resumable generation; the continuous producer's
size-capped ring directory with a manifest).

``produce`` draws the plan's quotas (``bank.GenerationPlan``) in shards of ``shard_size`` twins per dataset, each
twin with its own seed (``SEED_STRIDE * seed`` plus its index in the plan, so a shard is reproducible on its own),
through an injected sampler (``TwinEngine.sample`` on the server; a stand-in in tests). Before a shard is written its
twins pass the filters: exact duplicates (a hash set over the whole bank, kept in the manifest's directory), near
duplicates within the shard (MinHash), and the novelty filter against training stays (``bank.novelty``, when a
reference is given). A dropped twin is not redrawn (its seed is spent; the shard is smaller and says so). Shards
are written atomically (a temporary file, then a rename) and the manifest after them, so an interrupted run resumes
at the first shard the manifest lacks. ``ring_shards`` caps the bank: the oldest shards are deleted first and the
manifest records the evictions.

The interlock (plan: no bank before the privacy gate passes): ``require_gate`` reads the candidate's M7 report and
refuses unless its gate is PASS for the very weights being sampled (the sha256 the gate checked).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from icutwin.generate import bank

SEED_STRIDE = 100_000_000
MANIFEST = "manifest.json"


class GateNotPassed(RuntimeError):
    pass


def weights_sha256(model_dir: Path) -> str:
    h = hashlib.sha256()
    with (Path(model_dir) / "model.safetensors").open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def require_gate(audit_dir: Path, model_dir: Path) -> dict:
    """The M7 report of the candidate, if its gate is PASS for these weights; raises ``GateNotPassed`` otherwise."""
    audit_dir = Path(audit_dir)
    rep = audit_dir / "report.json"
    if not rep.exists():
        raise GateNotPassed(f"no M7 report in {audit_dir}")
    report = json.loads(rep.read_text(encoding="utf-8"))
    if report.get("gate") != "PASS":
        raise GateNotPassed(f"M7 gate is {report.get('gate')}, not PASS ({rep})")
    p1 = json.loads((audit_dir / "p1.json").read_text(encoding="utf-8"))
    gated = (p1.get("model") or {}).get("weights_sha256")
    have = weights_sha256(model_dir)
    if gated != have:
        raise GateNotPassed(f"the gate checked weights {gated}, these are {have}")
    return {"report": str(rep), "weights_sha256": have, "gate": "PASS"}


EXEMPT = ("INCONCLUSIVE",)          # gate statuses the study lead's exemption can cover (never FAIL or PENDING)


def require_gate_or_exemption(audit_dir: Path, model_dir: Path, exemption: Path | None, scope: str) -> dict:
    """``require_gate``, or, where the gate is INCONCLUSIVE, the study lead's recorded exemption (Paper P amendment
    17d): ``exemption`` names the weights' sha256, the gate status it covers and its scopes, and must match these
    weights, the report and ``scope``. The producers (bank, supply, toolkit) never take this path: they call
    ``require_gate``."""
    try:
        return require_gate(audit_dir, model_dir)
    except GateNotPassed:
        if exemption is None:
            raise
    report = json.loads((Path(audit_dir) / "report.json").read_text(encoding="utf-8"))
    ex = json.loads(Path(exemption).read_text(encoding="utf-8"))
    have = weights_sha256(model_dir)
    p1 = json.loads((Path(audit_dir) / "p1.json").read_text(encoding="utf-8"))
    if report.get("gate") not in EXEMPT or ex.get("gate") != report.get("gate"):
        raise GateNotPassed(f"M7 gate is {report.get('gate')}; the exemption covers {ex.get('gate')}")
    if ex.get("weights_sha256") != have or (p1.get("model") or {}).get("weights_sha256") != have:
        raise GateNotPassed("the exemption or the gate is for other weights")
    if scope not in ex.get("scopes", []):
        raise GateNotPassed(f"the exemption does not cover {scope!r}")
    return {"report": str(Path(audit_dir) / "report.json"), "weights_sha256": have,
            "gate": f"{report['gate']} (study lead's exemption)", "exemption": str(exemption)}


@dataclass
class Shard:
    name: str
    dataset: str
    first_index: int            # the plan index of its first twin (seeds follow from it)
    drawn: int
    kept: int
    dropped: dict = field(default_factory=dict)
    sha256: str = ""


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _load_manifest(out: Path) -> dict:
    p = out / MANIFEST
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def produce(sample, plan: bank.GenerationPlan, out_dir: Path, *, seed: int, shard_size: int, event: np.ndarray,
            gate: dict, model: dict, novelty_ref: dict | None = None, ring_shards: int | None = None,
            request=None, oversample: int = 1, select=None, novelty=None) -> dict:
    """Generate the plan's twins into ``out_dir`` (module docstring). ``sample(requests, seeds)`` returns token
    sequences; ``request(dataset)`` builds a request (default: ``engine.Request(dataset=...)``); ``gate``: the
    result of ``require_gate``; ``model``: what the manifest records of the model (name, weights, vocabulary);
    ``novelty_ref``: keyword arguments of ``bank.novelty`` without the samples (train, held, held_ds, n_vocab,
    exclude, device). With ``select(seqs, dataset, n, rng) -> indices`` (round 3's twin-level calibration,
    ``generate.calibrate.resample``), a shard draws ``oversample`` times its twins and keeps the ``n`` selected
    before the filters; each kept twin records its own seed. ``novelty``: a prepared ``bank.Novelty`` (instead of
    ``novelty_ref``, which rebuilds the reference for every shard)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if gate.get("gate") != "PASS":
        raise GateNotPassed("produce needs a passed M7 gate (require_gate)")
    if request is None:
        from icutwin.generate.engine import Request

        def request(ds):
            return Request(dataset=ds)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man = _load_manifest(out)
    if man and (man.get("seed") != seed or man.get("quotas") != plan.quotas or man.get("shard_size") != shard_size
                or man.get("oversample", 1) != oversample):
        raise ValueError("an existing bank with another plan, seed, shard size or oversampling")
    man = man or {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed, "shard_size": shard_size,
                  "oversample": oversample, "quotas": plan.quotas, "weights": plan.weights, "model": model,
                  "gate": gate, "shards": [], "evicted": []}
    hashes_path = out / "hashes.txt"
    seen = set(hashes_path.read_text().split()) if hashes_path.exists() else set()
    done = {s["name"] for s in man["shards"]} | {s["name"] for s in man["evicted"]}
    index = 0
    for ds in sorted(plan.quotas):
        for start in range(0, plan.quotas[ds], shard_size):
            n = min(shard_size, plan.quotas[ds] - start)
            name = f"{ds}_{start // shard_size:05d}"
            first = index
            index += n
            if name in done:
                continue
            seeds = [SEED_STRIDE * seed + first * oversample + i for i in range(n * oversample)]
            seqs = [np.asarray(x, dtype=np.int32) for x in sample([request(ds) for _ in seeds], seeds)]
            drawn = len(seqs)
            if select is not None:            # the twin-level calibration keeps n of the oversampled draws
                pick = sorted(int(i) for i in select(seqs, ds, n, np.random.default_rng(seeds[0])))
                seqs, seeds = [seqs[i] for i in pick], [seeds[i] for i in pick]
            keep = np.ones(len(seqs), dtype=bool)
            dropped = {"not_selected": drawn - len(seqs)} if select is not None else {}
            hs = [bank.seq_hash(s) for s in seqs]
            dup = np.array([h in seen for h in hs]) | bank.exact_duplicates(seqs)
            dropped["exact_duplicate"] = int(dup.sum())
            keep &= ~dup
            near = bank.near_duplicates(seqs, event) & keep
            dropped["near_duplicate"] = int(near.sum())
            keep &= ~near
            if (novelty is not None or novelty_ref is not None) and keep.any():
                idx = np.flatnonzero(keep)
                subset = [seqs[i] for i in idx]
                novel = (novelty(subset, [ds] * len(idx)) if novelty is not None
                         else bank.novelty(subset, [ds] * len(idx), **novelty_ref))
                dropped["not_novel"] = int((~novel).sum())
                keep[idx[~novel]] = False
            kept = np.flatnonzero(keep)
            table = pa.table({"twin": [f"{name}_{i:05d}" for i in kept], "dataset": [ds] * len(kept),
                              "seed": pa.array([seeds[i] for i in kept], pa.int64()),
                              "weight": pa.array([plan.weights[ds]] * len(kept), pa.float64()),
                              "ids": pa.array([seqs[i].tolist() for i in kept], type=pa.list_(pa.int32()))})
            sink = pa.BufferOutputStream()
            pq.write_table(table, sink)
            data = sink.getvalue().to_pybytes()
            _atomic_write(out / f"{name}.parquet", data)
            seen.update(hs[i] for i in kept)
            _atomic_write(hashes_path, "\n".join(sorted(seen)).encode())
            man["shards"].append(Shard(name, ds, first, drawn, int(len(kept)), dropped,
                                       hashlib.sha256(data).hexdigest()).__dict__)
            while ring_shards is not None and len(man["shards"]) > ring_shards:
                old = man["shards"].pop(0)
                (out / f"{old['name']}.parquet").unlink(missing_ok=True)
                man["evicted"].append({"name": old["name"], "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            _atomic_write(out / MANIFEST, json.dumps(man, indent=1).encode())
    man["complete"] = all(any(s["name"] == f"{ds}_{k:05d}" for s in man["shards"] + man["evicted"])
                          for ds in plan.quotas for k in range(-(-plan.quotas[ds] // shard_size)))
    _atomic_write(out / MANIFEST, json.dumps(man, indent=1).encode())
    return man
