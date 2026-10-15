"""A toolkit job (``docs/toolkit.md``): one directory under ``runs/toolkit/<tool>/`` that the stages hand files through,
so the sampling stage can run in vLLM's environment and the rest in the main one.

* ``job.json``: the tool, its arguments and the session's provenance;
* ``requests.parquet``: rollouts to draw (``SimRequest``: an id, the prompt, n, a seed, the token cap, the
  intervention), or ``sample_requests.parquet``: new twins (``SampleRequest``: the conditions of
  ``generate.engine.Request``);
* ``rollouts/part_<k>.parquet`` (id, rollout, continuation: the intervention's tokens, then the sampled ones) or
  ``samples/part_<k>.parquet`` (id, ids), written atomically by ``toolkit.stage``;
* ``stages/<stage>.json``: a finished stage and what it recorded (a rerun skips it).

Seeds: rollout j of a request has the seed ``request seed x 1,000,000 + j`` (``generate.engine.rollout_requests``),
so a request's rollouts do not depend on how requests are batched.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class SimRequest:
    id: str
    prompt: tuple
    n: int
    seed: int
    max_new: int | None = None
    give: tuple = ()
    withhold: tuple = ()
    gap: str = "INT//1-5m"


@dataclass(frozen=True)
class SampleRequest:
    id: str
    seed: int
    dataset: str | None = None
    site: str | None = None
    header: dict = field(default_factory=dict)
    prefix: dict = field(default_factory=dict)
    goals: dict = field(default_factory=dict)
    max_new: int | None = None


def job_id(tool: str, spec: dict) -> str:
    h = hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{h}"


def _atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_parquet(path: Path, cols: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    sink = pa.BufferOutputStream()
    pq.write_table(pa.table(cols), sink)
    _atomic(Path(path), sink.getvalue().to_pybytes())


def _ints(rows) -> object:
    import pyarrow as pa

    return pa.array([[int(x) for x in r] for r in rows], type=pa.list_(pa.int32()))


class Job:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def spec_path(self) -> Path:
        return self.root / "job.json"

    def write_spec(self, spec: dict) -> None:
        _atomic(self.spec_path, json.dumps(spec, indent=1, default=str).encode())

    def spec(self) -> dict:
        return json.loads(self.spec_path.read_text(encoding="utf-8"))

    # -- stages ---------------------------------------------------------------------------------------------------
    def done(self, stage: str) -> bool:
        return (self.root / "stages" / f"{stage}.json").exists()

    def mark(self, stage: str, info: dict) -> None:
        (self.root / "stages").mkdir(exist_ok=True)
        _atomic(self.root / "stages" / f"{stage}.json", json.dumps(info, indent=1, default=str).encode())

    def stage_info(self, stage: str) -> dict:
        return json.loads((self.root / "stages" / f"{stage}.json").read_text(encoding="utf-8"))

    # -- requests -------------------------------------------------------------------------------------------------
    def write_requests(self, reqs: list) -> None:
        if not reqs:
            raise ValueError("no requests")
        if isinstance(reqs[0], SimRequest):
            write_parquet(self.root / "requests.parquet", {
                "id": [r.id for r in reqs], "prompt": _ints(r.prompt for r in reqs), "n": [int(r.n) for r in reqs],
                "seed": [int(r.seed) for r in reqs], "max_new": [-1 if r.max_new is None else int(r.max_new) for r in reqs],
                "give": _ints(r.give for r in reqs), "withhold": _ints(r.withhold for r in reqs),
                "gap": [r.gap for r in reqs]})
        else:
            write_parquet(self.root / "sample_requests.parquet", {
                "id": [r.id for r in reqs], "seed": [int(r.seed) for r in reqs],
                "request": [json.dumps({"dataset": r.dataset, "site": r.site, "header": r.header, "prefix": r.prefix,
                                        "goals": r.goals}, sort_keys=True) for r in reqs],
                "max_new": [-1 if r.max_new is None else int(r.max_new) for r in reqs]})

    def requests(self) -> list:
        import pyarrow.parquet as pq

        if (self.root / "requests.parquet").exists():
            t = pq.read_table(self.root / "requests.parquet").to_pylist()
            return [SimRequest(r["id"], tuple(r["prompt"]), int(r["n"]), int(r["seed"]),
                               None if r["max_new"] < 0 else int(r["max_new"]), tuple(r["give"] or ()),
                               tuple(r["withhold"] or ()), r["gap"]) for r in t]
        t = pq.read_table(self.root / "sample_requests.parquet").to_pylist()
        out = []
        for r in t:
            c = json.loads(r["request"])
            out.append(SampleRequest(r["id"], int(r["seed"]), c["dataset"], c["site"], c["header"], c["prefix"],
                                     c["goals"], None if r["max_new"] < 0 else int(r["max_new"])))
        return out

    # -- outputs --------------------------------------------------------------------------------------------------
    def rollouts(self) -> dict:
        """id -> its continuations in rollout order (numpy int32 arrays)."""
        import numpy as np
        import pyarrow.parquet as pq

        by: dict = {}
        for f in sorted((self.root / "rollouts").glob("part_*.parquet")):
            t = pq.read_table(f)
            for i, k, c in zip(t.column("id").to_pylist(), t.column("rollout").to_pylist(),
                               t.column("continuation").to_pylist(), strict=True):
                by.setdefault(i, {})[int(k)] = np.asarray(c, dtype=np.int32)
        return {i: [x[k] for k in sorted(x)] for i, x in by.items()}

    def samples(self) -> dict:
        import pyarrow.parquet as pq

        out = {}
        for f in sorted((self.root / "samples").glob("part_*.parquet")):
            t = pq.read_table(f)
            out.update(zip(t.column("id").to_pylist(), t.column("ids").to_pylist(), strict=True))
        return out

    def write_result(self, result, name: str = "result.json") -> Path:
        from icutwin.toolkit.schema import clean

        p = self.root / name
        _atomic(p, json.dumps(clean(result), indent=1).encode())
        return p
