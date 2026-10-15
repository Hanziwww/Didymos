"""icutwin on the RTX 5090 host: Paper P amendment 17's next-token loss of a finished model (its ``final/hf``) on one
split of a token root, per dataset and token family (``model.train.evaluate``, the training run's own final
evaluation). For MUC32_main on ``adapt_r4``'s tuning split: the loss before adaptation, on the held-out FT patients and
on the seven databases' tuning split (forgetting, 17b).
    python ext17_eval.py --hf <run>/final/hf --tokens <root> --split tuning --out <json> [--datasets ...]
"""
import argparse
import json
import time
from pathlib import Path

DATASETS = ("mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire", "ricd", "zigong")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--split", default="tuning")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="M")
    ap.add_argument("--attn", default="varlen")
    ap.add_argument("--tokens-per-batch", type=int, default=65536)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    a = ap.parse_args()
    import torch

    from icutwin.model.data import layout, load_samples
    from icutwin.model.train import build_model, evaluate, init_weights
    from icutwin.tokenization.vocab import Vocab

    t0 = time.time()
    tokens = Path(a.tokens)
    v = Vocab.load(tokens / "vocab.json")
    model = init_weights(build_model(a.preset, v, a.attn), Path(a.hf)).to(torch.device("cuda"))
    samples = load_samples(tokens, list(a.datasets), a.split, v)
    ev = evaluate(model, samples, layout(v), a.tokens_per_batch, torch.device("cuda"))
    rep = {"hf": a.hf, "tokens": a.tokens, "split": a.split, "windows": len(samples), "eval": ev,
           "seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("windows", "seconds")} | {"nll": ev["nll"], "by_dataset": ev["by_dataset"]}))


if __name__ == "__main__":
    main()
