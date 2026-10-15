"""icutwin on the AutoDL host: one part of the twin bank (plan S7, ``generate.producer``) with the model and sampler
the study lead accepted (docs/m7_refinement.md: MUC32_main, round 3's sampler: length and treatment raking). The plan is
the whole bank's (``bank.GenerationPlan.from_mix`` over the training stays of every dataset); a part draws its
datasets' quotas into its own directory (one manifest per part, one GPU each). Interlock: the model's M7 gate must be
PASS for these very weights (``producer.require_gate``). Filters: exact and near duplicates, novelty against every
training first window (``bank.Novelty``, prepared once). Runs in vLLM's environment.
    python bank_produce.py --run MUC32_main --tokens <root> --out <dir> --datasets aumc --total 14000"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

R = Path("/root/autodl-tmp/icutwin")
ALL = ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]


def build_novelty(tokens: Path, v, dev, sparse: bool):
    """The gate's reference: every training first window; thresholds from every held-out first window."""
    from icutwin.eval import privacy as pv
    from icutwin.generate import bank

    special = {i for i, m in enumerate(v.meta) if m.get("cls") == "special"}
    train, held, held_ds = [], [], []
    for ds in ALL:
        train += pv.first_windows(tokens, ds, "train")
        hw = pv.first_windows(tokens, ds, "held_out")
        held += hw
        held_ds += [ds] * len(hw)
    return bank.Novelty(train, held, held_ds, len(v.tokens), special, dev, sparse=sparse)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--mix", default="flattened")
    ap.add_argument("--power", type=float, default=0.5)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--oversample", type=int, default=3)
    ap.add_argument("--calibration", default="calibration.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ap.add_argument("--novelty-ref", default=None,
                    help="a saved sparse novelty reference (bank.Novelty.save); built here when it does not exist")
    a = ap.parse_args()

    import torch

    from icutwin.eval.forecast import Reader
    from icutwin.generate import bank, calibrate, producer
    from icutwin.generate.engine import TwinEngine
    from icutwin.tokenization.vocab import Vocab

    tokens = Path(a.tokens)
    hf = R / "runs/train/full" / a.run / "final/hf"
    gate = producer.require_gate(R / "runs/audit" / f"m7_{a.run}", hf)
    v = Vocab.load(tokens / "vocab.json")
    natural = {ds: pq.ParquetFile(tokens / ds / "train.parquet").metadata.num_rows for ds in ALL}
    whole = bank.GenerationPlan.from_mix(natural, a.total, a.mix, a.power)
    plan = bank.GenerationPlan({d: whole.quotas[d] for d in a.datasets}, {d: whole.weights[d] for d in a.datasets})
    print(json.dumps({"whole_plan": whole.quotas, "this_part": plan.quotas, "gate": gate}), flush=True)
    t0 = time.time()
    dev = torch.device("cuda")
    if a.novelty_ref and Path(a.novelty_ref).exists():      # built once, shared by every engine process
        novelty = bank.Novelty.load(Path(a.novelty_ref), dev)
    else:
        novelty = build_novelty(tokens, v, dev, sparse=bool(a.novelty_ref))
        if a.novelty_ref:
            novelty.save(Path(a.novelty_ref))
    print(json.dumps({"novelty_reference_s": round(time.time() - t0, 1), "thresholds": novelty.thr}), flush=True)
    torch.cuda.empty_cache()                 # the build's temporaries (GBs): the engines share this GPU
    calib = json.loads((tokens / a.calibration).read_text(encoding="utf-8"))
    rd = Reader(v)

    def select(seqs, ds, n, rng):                    # round 3's twin-level calibration
        return calibrate.resample(seqs, rd, calib[ds], n, rng)[0]

    eng = TwinEngine(hf, tokens / "vocab.json", gpu_memory_utilization=a.gpu_memory_utilization, seed=a.seed)
    free, total = torch.cuda.mem_get_info()                # the device's, every engine included
    print(json.dumps({"engine_ready_s": round(time.time() - t0, 1), "gpu_used_gib": round((total - free) / 2**30, 2)}),
          flush=True)                                       # bank_single.sh starts the next engine on this line
    event =np.array([m.get("cls") == "event" for m in v.meta])
    model = {"run": a.run, "weights_sha256": gate["weights_sha256"], "vocab": str(tokens / "vocab.json"),
             "sampler": f"round 3: length and treatment raking ({a.calibration}), oversample {a.oversample}",
             "mix": a.mix, "power": a.power}
    man = producer.produce(eng.sample, plan, Path(a.out), seed=a.seed, shard_size=a.shard_size, event=event,
                           gate=gate, model=model, novelty=novelty, oversample=a.oversample, select=select)
    kept = {}
    for s in man["shards"]:
        kept[s["dataset"]] = kept.get(s["dataset"], 0) + s["kept"]
    drops = {}
    for s in man["shards"]:
        for k, n in s["dropped"].items():
            drops[k] = drops.get(k, 0) + n
    print(json.dumps({"complete": man["complete"], "kept": kept, "dropped": drops,
                      "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
