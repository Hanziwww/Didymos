"""icutwin on the AutoDL host: the factory's calibration benchmark (plan Phase F, S0; ``factory.oracle``). Anchors from
the twin bank (each twin decoded for labels on its own clock), the twin as known at each anchor, and ``n`` rollouts per
anchor with the accepted model (MUC32_main; the M7 gate must be PASS for these weights), capped at the dataset's
72-hour token quantile. Outputs ``<out>/anchors.parquet`` (the design), ``<out>/rollouts/<dataset>_<k>.parquet`` and
``<out>/rollouts/manifest.json``. Runs in vLLM's environment.
    python rollouts_bench.py --run MUC32_main --tokens <root> --bank <dir> --out <dir> [--per-dataset 200 --n 20]"""
import argparse
import json
import time
from pathlib import Path

R = Path("/root/autodl-tmp/icutwin")


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("run", "tokens", "bank", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--per-dataset", type=int, default=200)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=float, default=48.0)
    ap.add_argument("--cap-hours", type=float, default=72.0)
    ap.add_argument("--cap-q", type=float, default=0.95)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--deadline-hours", type=float, default=8.0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--datasets", nargs="*", default=None)
    a = ap.parse_args()
    t_start = time.time()

    import pyarrow as pa
    import pyarrow.parquet as pq

    from icutwin.factory import oracle, twins
    from icutwin.generate import producer
    from icutwin.generate.engine import TwinEngine
    from icutwin.tokenization.vocab import Vocab

    tokens, out = Path(a.tokens), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hf = R / "runs/train/full" / a.run / "final/hf"
    gate = producer.require_gate(R / "runs/audit" / f"m7_{a.run}", hf)
    v = Vocab.load(tokens / "vocab.json")
    rows = []
    for f in sorted(Path(a.bank).rglob("*.parquet")):
        rows += pq.read_table(f, columns=["twin", "dataset", "ids"]).to_pylist()
    if a.datasets:
        rows = [r for r in rows if r["dataset"] in a.datasets]
    stays = [(r["twin"], r["dataset"], twins.decode_labels(r["ids"], v, r["twin"])) for r in rows]
    print(json.dumps({"twins": len(rows), "decoded_s": round(time.time() - t_start, 1), "gate": gate["gate"]}),
          flush=True)
    anchors = oracle.choose(stays, v, a.per_dataset, a.seed, horizon=a.horizon)
    ids = {r["twin"]: r["ids"] for r in rows}
    prompts = [oracle.prompt(ids[x.twin], v, x.twin, x.hours) for x in anchors]
    seqs: dict = {}
    for r in rows:
        seqs.setdefault(r["dataset"], []).append(r["ids"])
    caps = oracle.token_caps(seqs, v, a.cap_hours, a.cap_q, a.seed)
    design = {f: [getattr(x, f) for x in anchors] for f in oracle.Anchor.__dataclass_fields__}
    design["prompt_len"] = [len(p) for p in prompts]
    pq.write_table(pa.table(design), out / "anchors.parquet")
    print(json.dumps({"anchors": len(anchors), "enriched": sum(x.enriched for x in anchors),
                      "flagged": sum(x.flag for x in anchors), "caps": caps,
                      "prompt_len_p50_p95": [int(sorted(design["prompt_len"])[len(prompts) // 2]),
                                             int(sorted(design["prompt_len"])[int(0.95 * (len(prompts) - 1))])],
                      "setup_s": round(time.time() - t_start, 1)}), flush=True)
    t0 = time.time()
    eng = TwinEngine(hf, tokens / "vocab.json", gpu_memory_utilization=a.gpu_memory_utilization)
    print(json.dumps({"engine_ready_s": round(time.time() - t0, 1)}), flush=True)
    man = oracle.run(eng, anchors, prompts, a.n, out / "rollouts", caps, a.seed, a.size,
                     deadline=t_start + a.deadline_hours * 3600)
    print(json.dumps({"complete": man.get("complete"), "batches": len(man["batches"]), "stopped": man.get("stopped"),
                      "seconds": round(time.time() - t_start, 1)}), flush=True)


if __name__ == "__main__":
    main()
