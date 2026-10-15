"""icutwin on the AutoDL host: Paper P amendment 7, F5's what-if rollouts (``docs/p1_paper_protocol.md``): the arms
of ``tools/paper_c1_design.py``'s ``design.parquet`` (one row per anchor and arm: the A/A withhold arms on C1's
prefixes, and the factual design's starters and matched non-starters with give and withhold). Each anchor's prompt is
its held-out stay as known at its hour; *give* inserts the arm's start token at the prompt's end and *withhold* masks
the treatment's tokens throughout (``generate.engine.Intervention``, C1's definitions in ``paper_whatif.py``); ``n``
rollouts per arm, capped at the split's 24-hour token quantile (C1's). An intervention the grammar rejects is not run
(counted). Writes ``<out>/anchors.parquet`` (one row per anchor: its treatment, group, pair, stay, hour and prediction
time) and ``<out>/rollouts/[shard<i>/]<arm>_<k>.parquet`` (anchor, arm, rollout, prompt, continuation; atomic,
resumable). Runs in vLLM's environment; ``--shard i --shards n`` for lanes.
    python paper_whatif2.py --run MUC32_main --tokens <root> --split held_out --store <real_test> \\
        --design <design.parquet> --out <dir>"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

R = Path("/root/autodl-tmp/icutwin")
sys.path.insert(0, str(R / "tools/server"))


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("run", "tokens", "split", "store", "design", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from paper_whatif import TREATMENTS

    from icutwin.eval import bench
    from icutwin.factory import oracle
    from icutwin.generate import producer
    from icutwin.generate.engine import Intervention, TwinEngine, intervene, treatment_tokens
    from icutwin.tokenization.fsm import compile_tables
    from icutwin.tokenization.vocab import Vocab

    tokens, out = Path(a.tokens), Path(a.out)
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    hf = R / "runs/train/full" / a.run / "final/hf"
    gate = producer.require_gate(R / "runs/audit" / f"m7_{a.run}", hf)
    v = Vocab.load(tokens / "vocab.json")
    code_of = [m.get("code") or "" for m in v.meta]
    d = pq.read_table(a.design).to_pylist()
    rows = {r["stay_id"]: r for r in bench.real_rows(tokens, a.split, {str(x["twin"]) for x in d})}
    fam = {}
    for name, (_s, f, _dose) in TREATMENTS.items():
        fam[name] = sorted(treatment_tokens(v, f) if f in ("rrt", "imv") else
                           frozenset(i for i, c in enumerate(code_of) if re.match(f, c)))
    prompts, meta = {}, {}
    for x in d:
        if x["anchor"] not in prompts:
            p = bench.prompt(rows[str(x["twin"])]["ids"], v, str(x["twin"]), float(x["hours"]))
            prompts[x["anchor"]] = p
            meta[x["anchor"]] = {k: x[k] for k in ("anchor", "treatment", "group", "twin", "dataset", "hours", "pair")}
            meta[x["anchor"]]["t_min"] = bench.anchor_time(p, v, str(x["twin"]))
            meta[x["anchor"]]["prompt_len"] = len(p)
    seqs: dict = {}
    for r in bench.real_rows(tokens, a.split, bench.store_stay_ids(a.store)):
        seqs.setdefault(r["dataset"], []).append(r["ids"])
    caps = oracle.token_caps(seqs, v, 24.0, 0.95, 0)
    if a.shard is None or a.shard == 0:
        m = list(meta.values())
        pq.write_table(pa.table({k: [x[k] for x in m] for k in m[0]}), out / "anchors.parquet")
        (out / "rollouts_design.json").write_text(json.dumps({**vars(a), "caps": caps, "gate": gate["gate"],
                                                              "anchors": len(m), "rows": len(d)}, indent=1, default=str))
    rejected: dict = {}
    todo = []
    tables = compile_tables(v)
    for x in d:                                      # the grammar must accept the intervention at the prompt's end
        if x["arm"] == "give":
            try:
                intervene(prompts[x["anchor"]], Intervention(give=(int(x["give_token"]),)), v, tables)
            except ValueError:
                rejected[x["treatment"]] = rejected.get(x["treatment"], 0) + 1
                continue
        todo.append(x)
    print(json.dumps({"anchors": len(meta), "arms": len(todo), "rejected": rejected,
                      "setup_s": round(time.time() - t_start, 1)}), flush=True)
    if a.dry:
        return
    groups: dict = {}
    for x in todo:
        groups.setdefault((x["arm"], int(x["seed"]), x["dataset"]), []).append(x)
    batches = []
    for (arm, seed, ds), xs in sorted(groups.items()):
        for k in range(0, len(xs), a.size):
            batches.append((arm, seed, ds, k // a.size, xs[k:k + a.size]))
    roll_dir = out / "rollouts"
    if a.shard is not None:
        batches = batches[a.shard::a.shards]
        roll_dir = roll_dir / f"shard{a.shard}"
    roll_dir.mkdir(parents=True, exist_ok=True)
    eng = TwinEngine(hf, tokens / "vocab.json", gpu_memory_utilization=a.gpu_memory_utilization)
    print(json.dumps({"engine_ready_s": round(time.time() - t_start, 1), "batches": len(batches)}), flush=True)
    for arm, seed, ds, k, xs in batches:
        path = roll_dir / f"{arm}_s{seed}_{ds}_{k:04d}.parquet"
        if path.exists():
            continue
        ivs = [Intervention(withhold=frozenset(fam[x["treatment"]])) if arm.startswith("withhold")
               else Intervention(give=(int(x["give_token"]),)) for x in xs]
        t0 = time.time()
        bseed = int(hashlib.sha256(f"{arm}|{ds}|{k}".encode()).hexdigest()[:6], 16) % 9973 + seed * 10_000
        conts = eng.simulate([prompts[x["anchor"]] for x in xs], a.n, seed=bseed, interventions=ivs,
                             max_new=caps.get(ds) or 4096)
        recs = [(x, j, c) for x, cs in zip(xs, conts, strict=True) for j, c in enumerate(cs)]
        tab = pa.table({"anchor": [x["anchor"] for x, _j, _c in recs], "arm": [arm] * len(recs),
                        "rollout": [j for _x, j, _c in recs],
                        "prompt": pa.array([list(map(int, prompts[x["anchor"]])) for x, _j, _c in recs],
                                           type=pa.list_(pa.int32())),
                        "continuation": pa.array([list(map(int, c)) for _x, _j, c in recs], type=pa.list_(pa.int32()))})
        sink = pa.BufferOutputStream()
        pq.write_table(tab, sink)
        data = sink.getvalue().to_pybytes()
        tmp = path.with_name(path.name + ".partial")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        print(json.dumps({"batch": path.name, "rollouts": len(recs), "seconds": round(time.time() - t0, 1),
                          "sha256": hashlib.sha256(data).hexdigest()[:16]}), flush=True)
    print(json.dumps({"done": True, "seconds": round(time.time() - t_start, 1)}), flush=True)


if __name__ == "__main__":
    main()
