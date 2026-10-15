"""icutwin on the AutoDL host: a probe of goal-first sampling with the transformers reference sampler (aggregates per
twin, nothing written but this summary): per twin the drawn treatment and outcome goals, its length, whether it
reached the ICU discharge and which obligations were open when the budget ended it.
    python gf_probe.py <run> <tokens root> <dataset> [n] [max_new]"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from transformers import GenerationConfig, LogitsProcessorList

from icutwin.eval import privacy as pv
from icutwin.eval.forecast import Reader
from icutwin.generate.constrain import GrammarProcessor
from icutwin.generate.engine import Request, conditions, goal_plan, strip_goal
from icutwin.generate.support import Support
from icutwin.tokenization.fsm import compile_tables, mask_tables
from icutwin.tokenization.vocab import Vocab

run, tokens, ds = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 16
max_new = int(sys.argv[5]) if len(sys.argv) > 5 else 6000
v = Vocab.load(tokens / "vocab.json")
t = compile_tables(v)
m = mask_tables(t, len(v.tokens))
plan, rd = goal_plan(v), Reader(v)
dev = torch.device("cuda")
model = pv.load_model(Path(f"/root/autodl-tmp/icutwin/runs/train/full/{run}/final/hf"), dev)
model = model.to(torch.bfloat16)

cond = replace(conditions(v, Request(dataset=ds)), draw_goals=True)
sup = Support.load(v, tokens / "support.json")
torch.manual_seed(0)
proc = GrammarProcessor(t, m, v.id("[PAD]"), max_length=1 + max_new, conditions=cond, support=sup, goal_plan=plan)
cfg = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=max_new,
                       eos_token_id=v.id("[EOS]"), pad_token_id=v.id("[PAD]"))
ids = torch.full((n, 1), v.id("[BOS]"), dtype=torch.long, device=dev)
with torch.no_grad():
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=cfg,
                         logits_processor=LogitsProcessorList([proc]))
for row, o in zip(out.cpu().numpy(), proc.obligations):
    stop = np.flatnonzero(row == v.id("[EOS]"))
    s = row[:stop[0] + 1].tolist() if len(stop) else row.tolist()
    bare, drawn = strip_goal(s, v)
    flags = rd.outcomes(bare)
    open_icu = [sorted({v.tokens[x].split("//")[0] + "//" + v.tokens[x].split("//")[1] for x in r})[:2]
                for r, d in zip(o.require_icu, o.done_icu) if not d]
    print(f"len {len(s):5d} eos {int(bool(len(stop)))} discharged {flags['icu_los_hours'] is not None} "
          f"forced {int(o.forced)} goals " + " ".join(f"{k}={drawn.get(k)}" for k in
          ("imv", "vasopressor", "rrt", "mcs", "icu_outcome", "icu_los")) + f" open {open_icu}", flush=True)
