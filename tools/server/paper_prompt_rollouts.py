"""icutwin on the AutoDL host: rollouts of given prompts (Paper P, C2's rollout-level check). Reads a parquet of
prompts (``id``, ``dataset``, ``prompt``; e.g. ``paper_attrib.py``'s ``rollout_prompts.parquet``) and draws ``n``
rollouts of each with the gated model, capped at the split's ``--cap-hours`` token quantile; writes
``<out>/rollouts/batch_<k>.parquet`` (id, rollout, continuation; atomic, resumable). Runs in vLLM's environment.

Amendment 7 (F4): ``--per-dataset-caps`` batches each dataset apart under its own cap (else a batch takes its
datasets' largest); ``--shard i --shards n`` draws one lane's share into ``rollouts/shard<i>`` with seeds apart from
the other lanes'; ``--design`` also writes a design directory for ``paper_hazards.py`` and the analyses
(``anchors.parquet``: one row per prompt, its id as the anchor, its stay, dataset and prediction time; ``design.json``
with the caps; the rollouts with their prompts).

Amendment 16 (16c): ``--caps-from <design.json>`` takes the token caps recorded by an earlier run (F4's) instead of
computing them from the store's sequences, so a host without the test store draws under exactly the same caps.
    python paper_prompt_rollouts.py --run MUC32_main --tokens <root> --split held_out --store <real_test> \\
        --prompts <file> --out <dir> [--n 16 --cap-hours 48]"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

R = Path("/root/autodl-tmp/icutwin")


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("run", "tokens", "split", "prompts", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--store", default=None, help="the real store whose sequences set the caps (or --caps-from)")
    ap.add_argument("--caps-from", default=None, help="a design.json whose recorded caps are used as they are")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--cap-hours", type=float, default=48.0)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--per-dataset-caps", action="store_true")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--design", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    import pyarrow as pa
    import pyarrow.parquet as pq

    from icutwin.eval import bench
    from icutwin.factory import oracle
    from icutwin.generate import producer
    from icutwin.generate.engine import TwinEngine
    from icutwin.tokenization.vocab import Vocab

    tokens, out = Path(a.tokens), Path(a.out)
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    hf = R / "runs/train/full" / a.run / "final/hf"
    gate = producer.require_gate(R / "runs/audit" / f"m7_{a.run}", hf)
    v = Vocab.load(tokens / "vocab.json")
    cols = ["id", "dataset", "prompt"] + (["twin", "t_min", "program", "anchor", "group"] if a.design else [])
    items = pq.read_table(a.prompts, columns=cols).to_pylist()
    if a.caps_from:                                   # an earlier run's recorded caps, as they are
        caps = json.loads(Path(a.caps_from).read_text())["caps"]
    else:
        if not a.store:
            raise SystemExit("--store or --caps-from is required")
        seqs: dict = {}
        for r in bench.real_rows(tokens, a.split, bench.store_stay_ids(a.store)):
            seqs.setdefault(r["dataset"], []).append(r["ids"])
        caps = oracle.token_caps(seqs, v, a.cap_hours, 0.95, a.seed)
    if a.shard is None or a.shard == 0:
        (out / "design.json").write_text(json.dumps({**vars(a), "caps": caps, "gate": gate["gate"],
                                                     "prompts": len(items)}, indent=1))
        if a.design:                                 # one row per prompt (its id as the anchor)
            pq.write_table(pa.table({"anchor": [x["id"] for x in items], "twin": [str(x["twin"]) for x in items],
                                     "dataset": [x["dataset"] for x in items],
                                     "t_min": [int(x["t_min"]) for x in items],
                                     "hours": [int(x["t_min"]) / 60.0 for x in items], "weight": [1.0] * len(items),
                                     "program": [x["program"] for x in items],
                                     "base_anchor": [x["anchor"] for x in items],
                                     "group": [x["group"] for x in items],
                                     "prompt_len": [len(x["prompt"]) for x in items]}), out / "anchors.parquet")
    roll_dir, seed = out / "rollouts", a.seed
    if a.per_dataset_caps:                           # each dataset's prompts together, under its own cap
        items = sorted(items, key=lambda x: x["dataset"])
    if a.shard is not None:
        items = items[a.shard::a.shards]
        roll_dir, seed = roll_dir / f"shard{a.shard}", a.seed * 10 + a.shard
    roll_dir.mkdir(parents=True, exist_ok=True)
    batches, cur = [], []
    for x in items:
        if cur and (len(cur) == a.size or (a.per_dataset_caps and x["dataset"] != cur[-1]["dataset"])):
            batches.append(cur)
            cur = []
        cur.append(x)
    if cur:
        batches.append(cur)
    eng = TwinEngine(hf, tokens / "vocab.json", gpu_memory_utilization=a.gpu_memory_utilization)
    print(json.dumps({"engine_ready_s": round(time.time() - t0, 1), "batches": len(batches)}), flush=True)
    for k, batch in enumerate(batches):
        path = roll_dir / f"batch_{k:05d}.parquet"
        if path.exists():
            continue
        t1 = time.time()
        conts = eng.simulate([x["prompt"] for x in batch], a.n, seed=seed * 10_000 + k,
                             max_new=max(caps.get(x["dataset"]) or 4096 for x in batch))
        recs = [(x, j, c) for x, cs in zip(batch, conts, strict=True) for j, c in enumerate(cs)]
        cols_out = {"id": [x["id"] for x, _j, _c in recs], "rollout": [j for _x, j, _c in recs],
                    "continuation": pa.array([list(map(int, c)) for _x, _j, c in recs], type=pa.list_(pa.int32()))}
        if a.design:
            cols_out["anchor"] = cols_out["id"]
            cols_out["prompt"] = pa.array([list(map(int, x["prompt"])) for x, _j, _c in recs], type=pa.list_(pa.int32()))
        sink = pa.BufferOutputStream()
        pq.write_table(pa.table(cols_out), sink)
        data = sink.getvalue().to_pybytes()
        tmp = path.with_name(path.name + ".partial")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        print(json.dumps({"batch": path.name, "rollouts": len(recs), "seconds": round(time.time() - t1, 1),
                          "sha256": hashlib.sha256(data).hexdigest()[:16]}), flush=True)
    print(json.dumps({"done": True, "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
