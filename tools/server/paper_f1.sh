#!/usr/bin/env bash
# icutwin on the one-GPU host: Paper P amendment 7, F1-F3 (docs/p1_paper_protocol.md). D1 first (the real futures of
# every test_R and test_L24 anchor through the hazard scorer), then per subset (R128, then L24x): its design files
# (--dry), 128 rollouts per anchor in three lanes (one vLLM engine each at MEM, started in turn), the hazards of the
# rollouts and real futures (GPU, after the lanes), and the subset's A1-style analysis (CPU, in the background).
# Logs: logs/f1_*.{log,exit}, logs/paper_f1.log. SMOKE=1: 3 R anchors, 4 rollouts, one lane, into runs/paper/f1/smoke.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
T4=$R/data/tokens/full_r4
F=$R/runs/paper/f1
A1=$R/runs/paper/a1
MEM=${MEM:-0.27}
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*" >> "$L/paper_f1.log"; }
ok() { [ "$(cut -d' ' -f1 "$L/$1.exit" 2>/dev/null)" = 0 ]; }
hazards() {   # <name> <design dir> [extra...]
  local name=$1 dir=$2
  shift 2
  bash tools/run.sh "f1_hz_$name" env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "$R/tools/server/paper_hazards.py" --run MUC32_main --tokens "$T4" --split held_out --design-dir "$dir" "$@"
  log "hazards $name exit $(cat "$L/f1_hz_$name.exit")"
}
ready() { until grep -q engine_ready_s "$L/$1.log" 2>/dev/null || [ -f "$L/$1.exit" ]; do sleep 20; done; }
subset() {   # <name> <design> <n> <lanes> [extra...]
  local name=$1 design=$2 n=$3 lanes=$4
  shift 4
  local args=(--split held_out --store "$S/real_test" --design "$design" --out "$F/$name"
              --subset "$F/subsets/$name.parquet" --n "$n" --rollout-seed 7 "$@")
  NAME=f1_${name}_dry bash "$R/tools/server/paper_rollouts.sh" "${args[@]}" --dry
  ok "f1_${name}_dry" || { log "design $name failed"; return 1; }
  log "design $name: $(grep -o '"anchors": [0-9]*' "$L/f1_${name}_dry.log" | head -1)"
  local k
  for ((k = 0; k < lanes; k++)); do
    rm -f "$L/f1_${name}_s$k.exit"
    ( NAME=f1_${name}_s$k bash "$R/tools/server/paper_rollouts.sh" "${args[@]}" --shard "$k" --shards "$lanes" \
        --gpu-memory-utilization "$MEM"; log "lane $name/$k exit $(cat "$L/f1_${name}_s$k.exit")" ) &
    ready "f1_${name}_s$k"
  done
  wait
  for ((k = 0; k < lanes; k++)); do ok "f1_${name}_s$k" || { log "lanes $name failed"; return 1; }; done
}
analyse() {   # <name> <min valid>
  setsid -f bash tools/run.sh "f1_a1_$1" env OMP_NUM_THREADS=1 "$PY" "$R/tools/paper_a1.py" --design-dir "$F/$1" \
    --store "$S/real_test" --work "$R/data/paper/stores/f1_$1" --out "$F/$1" --workers 4 --chunk 100 \
    --min-valid "$2" --curve-draws 20 > /dev/null 2>&1 < /dev/null
  log "analysis $1 started"
}
if [ "${SMOKE:-0}" = 1 ]; then
  "$PY" -c "import pandas as pd; r=pd.read_parquet('$F/subsets/R128.parquet'); r.iloc[[0,200,700]].to_parquet('$F/subsets/smoke.parquet', index=False)"
  rm -rf "$F/smoke"
  subset smoke R 4 1 && hazards smoke "$F/smoke" --real --check 50
  exit
fi
log "start"
hazards d1_R "$A1/test_R" --real --no-rollouts
hazards d1_L24 "$A1/test_L24" --real --no-rollouts --long
subset R128 R 128 3 && hazards R128 "$F/R128" --real && analyse R128 64
subset L24x L24 128 3 --cap-mult 4 && hazards L24x "$F/L24x" --real --long --a1-caps "$A1/test_L24/design.json" \
  && analyse L24x 64
log "done"
