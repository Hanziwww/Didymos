#!/usr/bin/env bash
# icutwin on the AutoDL host: round 2's third run (study lead's decision 2026-10-06; docs/m7_authenticity.md, "Third
# run", pre-registered). First the reference audit of M32_main with the step-4 sampler and #111 (AUTH_DIR auth_c);
# once its GPU work is done, ML32_main: packing parity at 32k (one GPU), the 50-step DDP smoke test with checkpoint
# and resume, the step budget from the r3 exposure (M32_main's passes), the run, the M7 gate (m7_gate.sh) and, if it
# passes, the same authenticity audit (runs/eval/ML32_main/auth). Logs: logs/ml32_*, train_ML32_main, g_ML32_main_*,
# authc_*, authml_*.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=$R/data/tokens/full_r3
Q3=$R/data/qa/full_r3
RUN=ML32_main
cd "$BASE" || exit 1
code() { [ -e "$L/$1.exit" ] && cut -d' ' -f1 "$L/$1.exit"; }
ok() { [ "$(code "$1")" = 0 ]; }
log() { echo "$(date '+%F %T') $*"; }
start=$(date +%s)
AUTH_DIR=auth_c LOG_PREFIX=authc LOS_BANDS=$T3/los_bands.json bash "$BASE/tools/authenticity_m32.sh" \
  > "$L/authc_run.log" 2>&1 &
ref=$!
until { [ -e "$L/authc_model.exit" ] && [ "$(stat -c %Y "$L/authc_model.exit")" -ge "$start" ]; } \
  || ! kill -0 $ref 2>/dev/null; do sleep 30; done
log "reference audit: GPU work done ($(tail -n 1 "$L/authc_run.log"))"
bash tools/run.sh ml32_parity env ICUTWIN_TMP="$R/.tmp/ml32_parity" CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
  icutwin.model.cli task=parity mode=full "tokens_root=$T3" "qa_root=$Q3" "hydra.run.dir=$R/runs/hydra/model/ml32_parity"
log "parity exit $(cat "$L/ml32_parity.exit")"
ok ml32_parity || exit 1
bash tools/run.sh ml32_smoke env ICUTWIN_TMP="$R/.tmp/ml32_smoke" .venv/bin/accelerate launch \
  --config_file src/icutwin/configs/accelerate/ddp.yaml -m icutwin.model.cli task=smoke mode=full \
  "tokens_root=$T3" "qa_root=$Q3" "hydra.run.dir=$R/runs/hydra/model/ml32_smoke"
log "smoke exit $(cat "$L/ml32_smoke.exit")"
ok ml32_smoke || exit 1
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
bash tools/run.sh "train_$RUN" bash "$BASE/tools/train.sh" "$RUN" "$steps" "tokens_root=$T3" "qa_root=$Q3"
log "train exit $(cat "$L/train_$RUN.exit")"
ok "train_$RUN" || exit 1
bash "$BASE/tools/m7_gate.sh" "$RUN" > "$L/g_${RUN}_run.log" 2>&1
log "gate: $(tail -n 1 "$L/g_${RUN}_run.log")"
gate=$(cd "$R" && .venv/bin/python -c "import json; print(json.load(open('runs/audit/m7_$RUN/report.json'))['gate'])")
[ "$gate" = PASS ] || { log "gate $gate: no audit; M32_main stays the candidate"; exit 0; }
wait $ref
log "reference audit: $(tail -n 1 "$L/authc_run.log")"
RUN=$RUN AUTH_DIR=auth LOG_PREFIX=authml LOS_BANDS=$T3/los_bands.json bash "$BASE/tools/authenticity_m32.sh" \
  > "$L/authml_run.log" 2>&1
log "audit: $(tail -n 1 "$L/authml_run.log")"
