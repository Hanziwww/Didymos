"""icutwin on the AutoDL host: round 3, step 0d: tuning NLL (``model.train.evaluate``, the whole tuning split, one
GPU) of the weight average of a run's last k checkpoints, k = 1 being the final weights alone:
    python ckpt_average.py <run> <k> [<k> ...]      e.g.  M32_main 2 3 5
Reads the run's checkpoints (never writes into the run); writes runs/eval/<run>/ckpt_average.json (aggregates)."""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

import icutwin.model.attention  # noqa: F401  registers the varlen attention
from icutwin.model.data import layout, load_samples
from icutwin.model.train import build_model, evaluate
from icutwin.tokenization.vocab import Vocab

R = Path("/root/autodl-tmp/icutwin")
run, ks = sys.argv[1], sorted({1, *(int(x) for x in sys.argv[2:])})
RUN = R / "runs/train/full" / run
cfg = json.loads((RUN / "config.json").read_text())
tokens = Path(cfg["tokens_root"])
v = Vocab.load(tokens / "vocab.json")
lay = layout(v)
samples = load_samples(tokens, list(cfg["datasets"]), "tuning", v)
ckpts = sorted(RUN.glob("step_*"), key=lambda p: int(p.name.split("_")[1])) + [RUN / "final"]
dev = torch.device("cuda")
model = build_model(cfg["preset"], v, cfg["attn_implementation"]).to(dev)
out = {}
for k in ks:
    states = [load_file(str(p / "hf" / "model.safetensors")) for p in ckpts[-k:]]
    avg = {n: sum(s[n].float() for s in states) / k for n in states[0]}
    missing, unexpected = model.load_state_dict(avg, strict=False)
    assert not unexpected and all("lm_head" in m for m in missing), (missing, unexpected)   # tied output layer
    ev = evaluate(model, samples, lay, int(cfg["tokens_per_batch"]), dev)
    out[str(k)] = {"checkpoints": [p.name for p in ckpts[-k:]], "nll": ev["nll"], "by_dataset": ev["by_dataset"]}
    print(k, round(ev["nll"], 4), {d: round(x, 4) for d, x in ev["by_dataset"].items()}, flush=True)
dest = R / "runs/eval" / run
dest.mkdir(parents=True, exist_ok=True)
(dest / "ckpt_average.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
