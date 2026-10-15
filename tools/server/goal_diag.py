"""icutwin on the GPU host: why goal-conditioned twins miss their event goals (the S1b smoke: ICU death kept by 54% on
the tokens). ``--n`` twins per goal (MIMIC, the accepted model and sampler, no raking) and per twin: its length, whether
it reached the budget (the grammar then ends it censored, the budget winning over the obligation), whether its tokens
keep the goal. Aggregates only; nothing is written but the summary.
    python goal_diag.py --tokens <root> --n 40
"""
import argparse
import json
from pathlib import Path

R = Path("/root/autodl-tmp/icutwin")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="MUC32_main")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--dataset", default="mimic")
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    a = ap.parse_args()
    import numpy as np

    from icutwin.generate import producer, supply
    from icutwin.generate.engine import Request, TwinEngine, strip_goal
    from icutwin.tokenization.vocab import Vocab

    tokens = Path(a.tokens)
    hf = R / "runs/train/full" / a.run / "final/hf"
    producer.require_gate(R / "runs/audit" / f"m7_{a.run}", hf)
    v = Vocab.load(tokens / "vocab.json")
    eng = TwinEngine(hf, tokens / "vocab.json", gpu_memory_utilization=a.gpu_memory_utilization, seed=0)
    budget = a.max_new or eng.context - 1
    out = {"dataset": a.dataset, "budget": budget, "goals": {}}
    for goals in ({"icu_outcome": "dead"}, {"rrt": "yes"}, {}):
        reqs = [Request(dataset=a.dataset, goals=goals) for _ in range(a.n)]
        seqs = eng.sample(reqs, [10_000 + k for k in range(a.n)], max_new=a.max_new)
        lens, at_budget, met, dead = [], [], [], []
        for s in seqs:
            ids, _g = strip_goal(list(s), v)
            lens.append(len(s))
            at_budget.append(len(s) >= budget - 64)
            met.append(supply.event_goal_met(ids, v, goals))
            dead.append(v.id("ICU//discharge//dead") in ids)
        key = json.dumps(goals) if goals else "none"
        out["goals"][key] = {"twins": a.n, "length_p50_p90_max": np.quantile(lens, [0.5, 0.9, 1.0]).astype(int).tolist(),
                             "at_budget": int(sum(at_budget)), "goal_met": sum(m is True for m in met),
                             "goal_met_not_at_budget": sum(m is True for m, b in zip(met, at_budget) if not b),
                             "not_at_budget": int(len(lens) - sum(at_budget)), "icu_death": int(sum(dead))}
        print(json.dumps({key: out["goals"][key]}), flush=True)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
