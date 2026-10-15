#!/usr/bin/env bash
# icutwin on the AutoDL host: round 3, amendment 1 (docs/m7_refinement.md): the two candidates on the r4 tokens
# (r3 with the clock blocks), side by side on two GPUs each (four micro-batches per step: M32_main's 0.52M tokens
# per step and recipe): MC32_main (clock blocks) and MCN32_main (clock blocks and input noise 0.15). Each: the
# 50-step DDP smoke test, the run, the freeze, P1, the masked teacher-forced calibration and its QA; then the
# 300-twin screens of both (all four GPUs). Waits for the r4 tokens and for the step-0c audit's GPU work.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T4=$R/data/tokens/full_r4
Q4=$R/data/qa/full_r4
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
code() { [ -e "$L/$1.exit" ] && cut -d' ' -f1 "$L/$1.exit"; }
ok() { [ "$(code "$1")" = 0 ]; }
until [ -e "$L/r4_clock.exit" ]; do sleep 30; done
ok r4_clock || { log "r4 transform failed: $(cat "$L/r4_clock.exit")"; exit 1; }
log "r4 tokens ready"
bash tools/run.sh r4_exposure .venv/bin/python -m icutwin.model.cli task=exposure mode=full "tokens_root=$T4" \
  "qa_root=$Q4" "hydra.run.dir=$R/runs/hydra/model/r4_exposure"
ok r4_exposure || { log "exposure failed"; exit 1; }
steps=$(cd "$R" && .venv/bin/python - <<EOF
import json, math
from omegaconf import OmegaConf
e = json.load(open("$Q4/exposure.json"))
passes = dict(OmegaConf.load("src/icutwin/configs/train.yaml").train.passes)
budget = sum(passes.get(d, passes["default"]) * s["tokens"] for d, s in e["splits"]["train"].items())
print(math.ceil(budget / (65536 * 4 * 2)))
EOF
)
log "budget: $steps steps (65,536 tokens x 4 micro-batches x 2 processes)"
launch() {   # <log name> <gpus> <port> <args...>
  local name=$1 gpus=$2 port=$3
  shift 3
  bash tools/run.sh "$name" env CUDA_VISIBLE_DEVICES=$gpus ICUTWIN_TMP="$R/.tmp/$name" .venv/bin/accelerate launch \
    --config_file src/icutwin/configs/accelerate/ddp.yaml --num_processes 2 --main_process_port "$port" \
    -m icutwin.model.cli mode=full "tokens_root=$T4" "qa_root=$Q4" train.grad_accum=4 train.dropout=0.0 "$@"
}
one() {   # <run> <gpu a> <gpu b> <port> [overrides...]
  local run=$1 ga=$2 gb=$3 port=$4
  shift 4
  launch "${run}_smoke" "$ga,$gb" "$port" task=smoke "train.name=$run" "$@" "hydra.run.dir=$R/runs/hydra/model/${run}_smoke"
  log "$run smoke exit $(cat "$L/${run}_smoke.exit")"
  ok "${run}_smoke" || return 1
  launch "train_$run" "$ga,$gb" "$port" task=train "train.name=$run" "train.max_steps=$steps" "$@" \
    "hydra.run.dir=$R/runs/hydra/train/$run"
  log "$run train exit $(cat "$L/train_$run.exit")"
  ok "train_$run" || return 1
  bash tools/run.sh "${run}_freeze" .venv/bin/python "$BASE/tools/freeze_run.py" "$run" "$T4" \
    "round-3 candidate (amendment 1), frozen for its screen and the M7 gate: no further training; no deliverable data"
  ok "${run}_freeze" || return 1
  bash tools/run.sh "g_${run}_p1" env CUDA_VISIBLE_DEVICES=$ga .venv/bin/python -m icutwin.eval.cli mode=full \
    "run=$run" "tokens_root=$T4" "out_dir=$R/runs/audit/m7_$run" step=p1 gpu=0 "hydra.run.dir=$R/runs/hydra/eval/g_${run}_p1" &
  bash tools/run.sh "${run}_qa" env CUDA_VISIBLE_DEVICES=$gb .venv/bin/python -m icutwin.eval.calibration --qa --n 1000 \
    --tokens "$T4" --model "$R/runs/train/full/$run/final/hf" --out "$R/runs/eval/$run/calibration_qa.json" &
  wait
  log "$run P1 exit $(cat "$L/g_${run}_p1.exit"); QA exit $(cat "$L/${run}_qa.exit")"
  RUN=$run TOKENS=$T4 GPUS="$ga $gb" bash "$BASE/tools/calib_masked.sh" > "$L/${run}_calm.log" 2>&1
  log "$run masked calibration: $(tail -n 1 "$L/${run}_calm.log")"
}
mkdir -p "$R/runs/eval/MC32_main" "$R/runs/eval/MCN32_main"
one MC32_main 0 1 29511 &
one MCN32_main 2 3 29512 train.input_noise=0.15 &
wait
for run in MC32_main MCN32_main; do
  ok "train_$run" || continue
  RUN=$run TOKENS=$T4 REQUIRE=p1 PER_DATASET=300 AUTH_DIR=screen LOG_PREFIX="${run}_screen" \
    LOS_BANDS=$T4/los_bands.json bash "$BASE/tools/authenticity_m32.sh" > "$L/${run}_screen_run.log" 2>&1
  log "$run screen: $(tail -n 1 "$L/${run}_screen_run.log")"
done
log "done"
