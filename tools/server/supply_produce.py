"""icutwin on a GPU host: one part of a twin-supply run (plan Phase F, "Twin supply"; S1b; ``generate.supply``) with the
model and sampler the study lead accepted (MUC32_main, round 3's length and treatment raking for the base cells). A part
draws the shards ``k`` of the plan with ``k % parts == part`` into ``<out>/part<part>`` (its own manifest); parts run
side by side, one engine each (several per GPU: ``supply_produce.sh``). Interlock: the model's M7 gate must be PASS for
these very weights. Filters: exact and near duplicates, novelty against every training first window (the bank's
saved sparse reference). Runs in vLLM's environment. ``--dry-run`` reads the plan only: the part's shards, twins and
draws per mechanism and the hours they take at ``--rate`` kept twins per engine-hour; no model, no GPU.

    python supply_produce.py --run MUC32_main --tokens <root> --plan <plan.json> --out <dir> --part 0 --parts 4
"""
import argparse
import json
import time
from pathlib import Path

R = Path("/root/autodl-tmp/icutwin")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="MUC32_main")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--oversample", type=int, default=3)
    ap.add_argument("--calibration", default="calibration.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.18)
    ap.add_argument("--novelty-ref", default=str(R / "data/bank/MUC32_main_bank/novelty_ref.npz"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rate", type=float, default=1150.0, help="kept twins per engine-hour, for the dry run "
                    "(the bank: 13,416 in 2.9 h on four engines)")
    a = ap.parse_args()
    from icutwin.generate import supply

    plan = supply.SupplyPlan.load(Path(a.plan))
    mine = [s for k, s in enumerate(supply.shards(plan, a.shard_size)) if k % a.parts == a.part]
    twins, draws = {}, {}
    for s in mine:
        for c, _r in s["requests"]:
            m = plan.cells[c].mechanism
            twins[m] = twins.get(m, 0) + 1
            draws[m] = draws.get(m, 0) + (a.oversample if s["select"] else 1)
    print(json.dumps({"plan": plan.name, "part": [a.part, a.parts], "shards": len(mine), "twins": twins,
                      "draws": draws, "hours_at_rate": round(sum(twins.values()) / a.rate, 2)}), flush=True)
    if a.dry_run:
        return

    import numpy as np
    import torch

    from icutwin.eval.forecast import Reader
    from icutwin.generate import bank, calibrate, producer
    from icutwin.generate.engine import TwinEngine
    from icutwin.tokenization.vocab import Vocab

    tokens = Path(a.tokens)
    hf = R / "runs/train/full" / a.run / "final/hf"
    gate = producer.require_gate(R / "runs/audit" / f"m7_{a.run}", hf)
    v = Vocab.load(tokens / "vocab.json")
    t0 = time.time()
    dev = torch.device("cuda")
    novelty = bank.Novelty.load(Path(a.novelty_ref), dev)
    torch.cuda.empty_cache()
    calib = json.loads((tokens / a.calibration).read_text(encoding="utf-8"))
    rd = Reader(v)

    def select(seqs, ds, n, rng):                    # round 3's twin-level calibration, base cells only
        return calibrate.resample(seqs, rd, calib[ds], n, rng)[0]

    eng = TwinEngine(hf, tokens / "vocab.json", gpu_memory_utilization=a.gpu_memory_utilization, seed=a.seed)
    free, total = torch.cuda.mem_get_info()
    print(json.dumps({"engine_ready_s": round(time.time() - t0, 1), "gpu_used_gib": round((total - free) / 2**30, 2)}),
          flush=True)                                 # supply_produce.sh starts the next engine on this line
    event = np.array([m.get("cls") == "event" for m in v.meta])
    model = {"run": a.run, "weights_sha256": gate["weights_sha256"], "vocab": str(tokens / "vocab.json"),
             "sampler": f"round 3: length and treatment raking ({a.calibration}) for base cells, oversample "
                        f"{a.oversample}; conditioned cells unraked"}
    man = supply.produce(eng.sample, plan, Path(a.out) / f"part{a.part}", seed=a.seed, shard_size=a.shard_size,
                         event=event, gate=gate, model=model, v=v, part=a.part, parts=a.parts, novelty=novelty,
                         oversample=a.oversample, select=select)
    kept, drops = {}, {}
    for s in man["shards"]:
        for c, n in s["cells"].items():
            m = c.split("/")[0]
            kept[m] = kept.get(m, 0) + n
        for k, n in s["dropped"].items():
            drops[k] = drops.get(k, 0) + n
    print(json.dumps({"complete": man["complete"], "kept": kept, "dropped": drops,
                      "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
