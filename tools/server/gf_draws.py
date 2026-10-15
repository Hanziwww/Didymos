"""icutwin on the AutoDL host: the goal blocks goal-first sampling draws (prefix, header and goal block only; the
transformers reference sampler with the generation mask), per dataset, against the goal labels of the real held-out
stays (aggregates only): the share of each label of the length-of-stay, ICU outcome and treatment fields.
    python gf_draws.py <run> <tokens root> <n per dataset> [datasets...]"""
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import GenerationConfig, LogitsProcessorList

from icutwin.eval import privacy as pv
from icutwin.generate.constrain import GrammarProcessor, HeaderAugmentation
from icutwin.generate.engine import Request, conditions, goal_plan, header_unknown_rate
from icutwin.generate.support import Support
from icutwin.tokenization.fsm import compile_tables, mask_tables
from icutwin.tokenization.vocab import Vocab

run, tokens, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
datasets = sys.argv[4:] or ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
FIELDS = ("icu_los", "icu_outcome", "imv", "vasopressor", "rrt")
v = Vocab.load(tokens / "vocab.json")
t = compile_tables(v)
m = mask_tables(t, len(v.tokens))
hf = Path(f"/root/autodl-tmp/icutwin/runs/train/full/{run}/final/hf")
dev = torch.device("cuda")
model = pv.load_model(hf, dev).to(torch.bfloat16)
sup = Support.load(v, tokens / "support.json")
undo = HeaderAugmentation(v, header_unknown_rate(hf))
g0, g1 = v.id("[GOAL]"), v.id("[/GOAL]")


def shares(blocks):
    out = {f: Counter() for f in FIELDS}
    for b in blocks:
        for tok in b:
            _g, f, lab = v.tokens[tok].split("//", 2)
            if f in out:
                out[f][lab] += 1
    return {f: {k: round(c / max(sum(out[f].values()), 1), 3) for k, c in sorted(out[f].items())} for f in FIELDS}


for ds in datasets:
    cond = replace(conditions(v, Request(dataset=ds)), draw_goals=True)
    plan = goal_plan(v, {})                       # no binding: the draws only
    blocks = []
    for k in range(0, n, 250):
        torch.manual_seed(k)
        proc = GrammarProcessor(t, m, v.id("[PAD]"), conditions=cond, support=sup, goal_plan=plan,
                                header_augmentation=undo)
        cfg = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=40,
                               eos_token_id=g1, pad_token_id=v.id("[PAD]"))
        ids = torch.full((min(250, n - k), 1), v.id("[BOS]"), dtype=torch.long, device=dev)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=cfg,
                                 logits_processor=LogitsProcessorList([proc]))
        for row in out.cpu().tolist():
            if g0 in row and g1 in row:
                blocks.append(row[row.index(g0) + 1:row.index(g1)])
    real = []
    for goal in pq.read_table(tokens / ds / "held_out.parquet", columns=["goal_ids"]).column(0).to_pylist():
        if goal and g0 in goal:
            real.append(goal[goal.index(g0) + 1:goal.index(g1)])
    d, r = shares(blocks), shares(real)
    print(f"== {ds}: drawn {len(blocks)}, real {len(real)}", flush=True)
    for f in FIELDS:
        print(f"  {f:12s} drawn {d[f]}\n  {'':12s} real  {r[f]}", flush=True)
