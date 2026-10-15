#!/usr/bin/env bash
# icutwin on the AutoDL host: the second training run, M32_main (32k context, target passes per dataset), on the r3
# tokens (data/tokens/full_r3): packing parity at 32k (one GPU), the 50-step DDP smoke test with checkpoint and
# resume, the step budget from the r3 exposure (sum of target passes x train tokens / tokens per step), the run.
# Starts when r3 has written its tokens (logs/tok_encode_r3.exit = 0 and logs/exposure_r3.exit = 0).
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=$R/data/tokens/full_r3
Q3=$R/data/qa/full_r3
cd "$BASE" || exit 1
code() { [ -e "$L/$1.exit" ] && cut -d' ' -f1 "$L/$1.exit"; }
ok() { [ "$(code "$1")" = 0 ]; }
log() { echo "$(date '+%F %T') $*"; }
until [ -e "$L/exposure_r3.exit" ] && [ -e "$L/tok_encode_r3.exit" ]; do sleep 30; done
ok tok_encode_r3 && ok exposure_r3 || { log "r3 tokens not ready: encode $(code tok_encode_r3), exposure $(code exposure_r3)"; exit 1; }
log "r3 tokens ready"
bash tools/run.sh m32_parity env ICUTWIN_TMP="$R/.tmp/m32_parity" CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m icutwin.model.cli \
  task=parity mode=full "tokens_root=$T3" "qa_root=$Q3" "hydra.run.dir=$R/runs/hydra/model/m32_parity"
log "parity exit $(cat "$L/m32_parity.exit")"
ok m32_parity || exit 1
bash tools/run.sh m32_smoke env ICUTWIN_TMP="$R/.tmp/m32_smoke" .venv/bin/accelerate launch \
  --config_file src/icutwin/configs/accelerate/ddp.yaml -m icutwin.model.cli task=smoke mode=full \
  "tokens_root=$T3" "qa_root=$Q3" "hydra.run.dir=$R/runs/hydra/model/m32_smoke"
log "smoke exit $(cat "$L/m32_smoke.exit")"
ok m32_smoke || exit 1
steps=$(cd "$R" && .venv/bin/python - <<EOF
import json, math
from omegaconf import OmegaConf
e = json.load(open("$Q3/exposure.json"))
t = OmegaConf.load("src/icutwin/configs/train.yaml").train
passes = dict(t.passes)
budget = sum(passes.get(d, passes["default"]) * s["tokens"] for d, s in e["splits"]["train"].items())
per_step = int(t.tokens_per_batch) * int(t.grad_accum) * 4
print(math.ceil(budget / per_step))
EOF
)
log "budget: $steps steps"
bash tools/run.sh train_M32_main bash "$BASE/tools/train.sh" M32_main "$steps" "tokens_root=$T3" "qa_root=$Q3"
log "train exit $(cat "$L/train_M32_main.exit")"
