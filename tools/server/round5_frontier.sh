#!/usr/bin/env bash
# icutwin on the AutoDL host: round 3, amendment 4 (docs/m7_refinement.md): Muon with per-dataset passes projected to
# a membership AUC, side by side on two GPUs each (four micro-batches per step, MU32_main's recipe): MUF32_main
# (point 0.565) and MUC32_main (0.555). Each: the run, the freeze, P1, the masked teacher-forced calibration (S4) and
# its QA; then the 300-twin screens of those whose P1 passes, with the sampler amendment 3's rule chose
# (logs/round3_goalfirst.txt: GOALFIRST 1 = goal-first draws, AUTH_DIR screen_gf; MC32_main screened again with it).
# Waits for amendment 3's audits (logs/gf0.exit, gf1.exit).
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T4=$R/data/tokens/full_r4
Q4=$R/data/qa/full_r4
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
code() { [ -e "$L/$1.exit" ] && cut -d' ' -f1 "$L/$1.exit"; }
ok() { [ "$(code "$1")" = 0 ]; }
until [ -e "$L/gf0.exit" ] && [ -e "$L/gf1.exit" ]; do sleep 30; done
log "amendment 3's audits done"
PF="{default:3.0,mimic:1.87,aumc:3.6,eicu:3.15,hirid:2.86,sicdb:2.37,nwicu:2.32,inspire:2.13}"
PC="{default:3.0,mimic:1.59,aumc:3.05,eicu:2.66,hirid:2.42,sicdb:2.01,nwicu:1.96,inspire:1.81}"
steps() {   # <passes dict as above>: the step budget from the r4 exposure
  (cd "$R" && .venv/bin/python - "$1" <<'EOF'
import json, math, sys
e = json.load(open("data/qa/full_r4/exposure.json"))
passes = dict(kv.split(":") for kv in sys.argv[1].strip("{}").split(","))
passes = {k: float(v) for k, v in passes.items()}
budget = sum(passes.get(d, passes["default"]) * s["tokens"] for d, s in e["splits"]["train"].items())
print(math.ceil(budget / (65536 * 4 * 2)))
EOF
  )
}
launch() {   # <log name> <gpus> <port> <args...>
  local name=$1 gpus=$2 port=$3
  shift 3
  bash tools/run.sh "$name" env CUDA_VISIBLE_DEVICES=$gpus ICUTWIN_TMP="$R/.tmp/$name" .venv/bin/accelerate launch \
    --config_file src/icutwin/configs/accelerate/ddp.yaml --num_processes 2 --main_process_port "$port" \
    -m icutwin.model.cli mode=full "tokens_root=$T4" "qa_root=$Q4" train.grad_accum=4 train.dropout=0.0 \
    train.optimizer=muon "$@"
}
one() {   # <run> <gpu a> <gpu b> <port> <passes>
  local run=$1 ga=$2 gb=$3 port=$4 passes=$5 n
  n=$(steps "$passes")
  log "$run: passes $passes, $n steps"
  launch "train_$run" "$ga,$gb" "$port" task=train "train.name=$run" "train.max_steps=$n" "++train.passes=$passes" \
    "hydra.run.dir=$R/runs/hydra/train/$run"
  log "$run train exit $(cat "$L/train_$run.exit")"
  ok "train_$run" || return 1
  bash tools/run.sh "${run}_freeze" .venv/bin/python "$BASE/tools/freeze_run.py" "$run" "$T4" \
    "round-3 candidate (amendment 4), frozen for its screen and the M7 gate: no further training; no deliverable data"
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
mkdir -p "$R/runs/eval/MUF32_main" "$R/runs/eval/MUC32_main"
one MUF32_main 0 1 29531 "$PF" &
sleep 20
one MUC32_main 2 3 29532 "$PC" &
wait
GF=$(awk '/^GOALFIRST/ {print $2}' "$L/round3_goalfirst.txt" 2>/dev/null)
log "sampler for the screens: goal-first draws ${GF:-0}"
screens="MUF32_main MUC32_main"
[ "$GF" = 1 ] && screens="$screens MC32_main"
for run in $screens; do
  [ "$run" = MC32_main ] || ok "train_$run" || continue
  if [ "$GF" = 1 ]; then
    DRAW_GOALS=1 RUN=$run TOKENS=$T4 REQUIRE=p1 PER_DATASET=300 AUTH_DIR=screen_gf LOG_PREFIX="${run}_screengf" \
      LOS_BANDS=$T4/los_bands.json bash "$BASE/tools/authenticity_m32.sh" > "$L/${run}_screengf_run.log" 2>&1
    log "$run screen (goal-first): $(tail -n 1 "$L/${run}_screengf_run.log")"
  else
    RUN=$run TOKENS=$T4 REQUIRE=p1 PER_DATASET=300 AUTH_DIR=screen LOG_PREFIX="${run}_screen" \
      LOS_BANDS=$T4/los_bands.json bash "$BASE/tools/authenticity_m32.sh" > "$L/${run}_screen_run.log" 2>&1
    log "$run screen: $(tail -n 1 "$L/${run}_screen_run.log")"
  fi
done
log "done"
