"""Audit samples of the M7 privacy gate with vLLM (plan S7: vLLM V1 is the bank's engine): ancestral sampling at
temperature 1 without top-k or top-p truncation, from ``[BOS]``, until ``[EOS]`` or the model's context, with a seed
per sample (``sample_seeds``: ``SEED_STRIDE * seed`` plus the sample's index among all ``n``, so no two samples of
any shard share one), no grammar mask (docs/m7_privacy.md). The first version seeded ``1000 * seed + 100 * shard``
plus the index in the shard: shards of more than 100 samples reused each other's seeds.
Continuous batching keeps the GPU full while samples end at very different lengths, which a batch of Hugging Face
``generate`` cannot (it runs until its longest sample ends).

Runs in vLLM's own environment (it pins its own dependencies) and imports nothing of icutwin's:
    vllm-env/bin/python -m icutwin.generate.vllm_audit --model <hf dir> --vocab <vocab.json> --out <dir> \\
        --n 5000 --shards 4 --shard 0 --seed 0 [--max-new 32767]
Writes ``<out>/shard_<k>.parquet`` (column ``ids``: [BOS] and the sampled tokens, as list<int32>), the format of
``icutwin.eval.cli step=sample``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

README = ("AUDIT-ONLY samples of the frozen candidate for the M7 privacy gate (docs/m7_privacy.md). Not synthetic "
          "data: never decode, analyse otherwise, share or train on them; deleted when the gate is decided.\n")


SEED_STRIDE = 100_000_000


def shard_size(n: int, shards: int, shard: int) -> int:
    return n // shards + (1 if shard < n % shards else 0)


def sample_seeds(seed: int, n: int, shards: int, shard: int) -> list[int]:
    """The shard's per-sample seeds: ``SEED_STRIDE * seed`` plus each sample's index among all ``n`` samples
    (the shards' index ranges are consecutive and disjoint)."""
    if n > SEED_STRIDE:
        raise ValueError(f"at most {SEED_STRIDE} samples per seed")
    start = sum(shard_size(n, shards, k) for k in range(shard))
    return [SEED_STRIDE * seed + start + i for i in range(shard_size(n, shards, shard))]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=0, help="new tokens at most (0: the model's context - 1)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    a = ap.parse_args(argv)

    import pyarrow as pa
    import pyarrow.parquet as pq
    from vllm import LLM, SamplingParams

    tokens = json.loads(Path(a.vocab).read_text(encoding="utf-8"))["tokens"]
    bos, eos = tokens.index("[BOS]"), tokens.index("[EOS]")
    cfg = json.loads((Path(a.model) / "config.json").read_text(encoding="utf-8"))
    context = int(cfg["max_position_embeddings"])
    max_new = a.max_new or context - 1
    seeds = sample_seeds(a.seed, a.n, a.shards, a.shard)
    n = len(seeds)
    llm = LLM(model=a.model, skip_tokenizer_init=True, dtype=a.dtype, max_model_len=min(context, max_new + 1),
              gpu_memory_utilization=a.gpu_memory_utilization, seed=a.seed, enable_prefix_caching=False)
    params = [SamplingParams(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, max_tokens=max_new, seed=s,
                             stop_token_ids=[eos], detokenize=False) for s in seeds]
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": [bos]} for _ in range(n)], params, use_tqdm=False)
    seqs = [[bos, *o.outputs[0].token_ids] for o in outs]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"ids": pa.array(seqs, type=pa.list_(pa.int32()))}), out / f"shard_{a.shard}.parquet")
    (out / "README.txt").write_text(README, encoding="utf-8")
    ended = sum(1 for s in seqs if s[-1] == eos)
    print(json.dumps({"shard": a.shard, "samples": n, "seeds": [seeds[0], seeds[-1]] if seeds else None,
                      "distinct_samples": len({tuple(s) for s in seqs}), "ended_with_eos": ended,
                      "tokens": sum(map(len, seqs)), "max_new": max_new, "seconds": round(time.time() - t0, 1)}),
          flush=True)


if __name__ == "__main__":
    main()
