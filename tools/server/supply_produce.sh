#!/usr/bin/env bash
# icutwin on a GPU host: a twin-supply run (plan Phase F, "Twin supply"; S1b; generate.supply) over every GPU, ENGINES
# vLLM engines per GPU (generation is CPU-bound: one core per engine's grammar mask). Part p = gpu * ENGINES + engine
# draws the plan's shards k with k % parts == p into OUT/part<p>. On one GPU the engines start one after another (vLLM
# sizes its KV cache from the device's free memory at startup); GPUs start side by side. The RTX PRO 6000 server
# (4 x 96 GB) runs this in the training phase (study lead, Q&A 7); this phase runs a dry run there and a small smoke on
# the RTX 4080 host (supply_smoke.sh).
#   PLAN=... OUT=... [GPUS="0 1 2 3"] [ENGINES=4] [GPU_MEM=0.18] [SHARD=200] [DRY=1] bash supply_produce.sh
# Internal, DUA-derived twins: they stay on the approved hosts. Logs: logs/supply_<p>.{log,exit}.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T4=$R/data/tokens/full_r4
PLAN=${PLAN:?PLAN: the plan JSON (tools/supply_plan.py)}
OUT=${OUT:?OUT: the run directory}
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')}
GPUS=${GPUS:-0}
ENGINES=${ENGINES:-4}
GM=${GPU_MEM:-0.18}
NAME=${NAME:-supply}
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
read -r -a gpus <<< "$GPUS"
parts=$(( ${#gpus[@]} * ENGINES ))
cp "$R/tools/server/supply_produce.py" "$BASE/tools/${NAME}_produce.py"      # a running copy is never overwritten
mkdir -p "$OUT" "$L/prev"
args() {   # <part>
  echo --run MUC32_main --tokens "$T4" --plan "$PLAN" --out "$OUT" --part "$1" --parts "$parts" \
    --shard-size "${SHARD:-200}" --gpu-memory-utilization "$GM"
}
if [ -n "$DRY" ]; then
  for p in $(seq 0 $(( parts - 1 ))); do
    # shellcheck disable=SC2046
    "$BASE/vllm-env/bin/python" "$BASE/tools/${NAME}_produce.py" $(args "$p") --dry-run
  done
  exit 0
fi
cat > "$OUT/README.txt" <<EOF
INTERNAL synthetic ICU stays (a twin-supply run) from MUC32_main, a model trained on DUA-covered data (MIMIC-IV,
eICU, HiRID, AmsterdamUMCdb, SICdb, NWICU, INSPIRE). DUA-derived: keep on the approved hosts; never share unless cleared.
EOF
one() {    # <part> <gpu>
  local p=$1 g=$2
  for f in "$L/${NAME}_$p.log" "$L/${NAME}_$p.exit"; do [ -e "$f" ] && mv -f "$f" "$L/prev/$(basename "$f").$(date +%s)"; done
  # shellcheck disable=SC2046
  bash tools/run.sh "${NAME}_$p" env PATH="$BASE/vllm-env/bin:$PATH" ICUTWIN_VOCAB="$T4/vocab.json" \
    CUDA_VISIBLE_DEVICES="$g" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$BASE/vllm-env/bin/python" "$BASE/tools/${NAME}_produce.py" $(args "$p")
}
ready() {  # <part>: until its engine is up or the part has ended (at most 30 minutes)
  local i
  for i in $(seq 1 120); do
    grep -q '"engine_ready_s"' "$L/${NAME}_$1.log" 2>/dev/null && return 0
    [ -e "$L/${NAME}_$1.exit" ] && { log "${NAME}_$1 ended before its engine was up: $(tail -n 3 "$L/${NAME}_$1.log")"; return 0; }
    sleep 15
  done
  log "${NAME}_$1: no engine after 30 minutes; starting the next one anyway"
}
gpu_group() {  # <gpu position>: its engines, one after another
  local k=$1 e p
  for e in $(seq 0 $(( ENGINES - 1 ))); do
    p=$(( k * ENGINES + e ))
    one "$p" "${gpus[$k]}" &
    ready "$p"
  done
  wait
}
for k in "${!gpus[@]}"; do gpu_group "$k" & done
wait
for p in $(seq 0 $(( parts - 1 ))); do
  log "${NAME}_$p exit $(cat "$L/${NAME}_$p.exit" 2>/dev/null): $(grep -E '^\{"complete' "$L/${NAME}_$p.log" | tail -n 1)"
done
log "done"
