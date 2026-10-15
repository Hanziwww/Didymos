"""The sampling stage of a toolkit job (``toolkit.jobs``): the job's requests in batches through a sampler, each batch
written atomically (``rollouts/part_<k>.parquet`` or ``samples/part_<k>.parquet``; a written batch is skipped on a
rerun). The batches follow the requests' order (at most ``batch_rollouts`` rollouts each, a request never split), so a
rerun with the same size resumes; ``--shard i --shards n`` draws batches i, i + n, ... (lanes of one GPU).

Runs in vLLM's environment as a subprocess of ``Session.sample`` (imports pyarrow and numpy only, besides the
engine):
    python -m icutwin.toolkit.stage <job dir> --model <hf dir> --vocab <vocab.json> [--gpu-memory-utilization 0.85]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def batches(reqs: list, batch_rollouts: int) -> list[list]:
    out, cur, size = [], [], 0
    for r in reqs:
        k = getattr(r, "n", 1)
        if cur and size + k > batch_rollouts:
            out.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += k
    if cur:
        out.append(cur)
    return out


def run(job, sampler, batch_rollouts: int = 1024, shard: int = 0, shards: int = 1) -> dict:
    from icutwin.toolkit.jobs import SimRequest, _ints, write_parquet

    reqs = job.requests()
    sim = isinstance(reqs[0], SimRequest)
    out = job.root / ("rollouts" if sim else "samples")
    out.mkdir(exist_ok=True)
    bs = batches(reqs, batch_rollouts)
    done = drawn = 0
    for b, group in enumerate(bs):
        if b % shards != shard:
            continue
        f = out / f"part_{b:05d}.parquet"
        if f.exists():
            continue
        t0 = time.time()
        if sim:
            res = sampler.simulate(group)
            rows = [(r.id, k, c) for r in group for k, c in enumerate(res[r.id])]
            write_parquet(f, {"id": [x[0] for x in rows], "rollout": [x[1] for x in rows],
                              "continuation": _ints(x[2] for x in rows)})
        else:
            res = sampler.sample(group)
            rows = [(r.id, res[r.id]) for r in group]
            write_parquet(f, {"id": [x[0] for x in rows], "ids": _ints(x[1] for x in rows)})
        done += 1
        drawn += len(rows)
        print(json.dumps({"batch": f.name, "sequences": len(rows), "seconds": round(time.time() - t0, 1)}), flush=True)
    return {"batches": len(bs), "written": done, "sequences": drawn, "batch_rollouts": batch_rollouts}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--batch-rollouts", type=int, default=1024)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--go", default=None,
                    help="wait for this file after the engine is ready: no lane releases GPU memory while another "
                         "lane's engine measures the free memory at its start")
    a = ap.parse_args(argv)
    from icutwin.toolkit.backends import VllmSampler
    from icutwin.toolkit.jobs import Job

    t0 = time.time()
    sampler = VllmSampler(Path(a.model), Path(a.vocab), gpu_memory_utilization=a.gpu_memory_utilization)
    print(json.dumps({"engine_ready": round(time.time() - t0, 1)}), flush=True)
    while a.go and not Path(a.go).exists():
        time.sleep(1)
    info = run(Job(a.job), sampler, a.batch_rollouts, a.shard, a.shards)
    print(json.dumps({"done": info, "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
