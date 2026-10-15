#!/usr/bin/env bash
# icutwin on the one-GPU host: Paper P's remaining rollout designs (docs/p1_paper_protocol.md A1, B2) in three lanes,
# one vLLM engine each (a single engine is CPU-bound by the grammar masks: 34% GPU), each engine at MEM of the GPU and
# started only when the previous lane's engine is ready (vLLM sizes its cache from the free memory at start). Waits for
# WAIT_FOR's exit file first. Each design's analysis (tools/paper_a1.py, CPU) starts in the background after its
# rollouts. Logs: logs/p1_<name>{,_a1}.{log,exit}; logs/paper_lanes.log.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
MEM=${MEM:-0.27}
WAIT_FOR=${WAIT_FOR:-p1_dev_L24}
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*" >> "$L/paper_lanes.log"; }
analyse() {   # <name> <store>
  setsid -f bash tools/run.sh "p1_$1_a1" env OMP_NUM_THREADS=1 "$PY" "$R/tools/paper_a1.py" \
    --design-dir "$R/runs/paper/a1/$1" --store "$2" --work "$R/data/paper/stores/a1_$1" --out "$R/runs/paper/a1/$1" \
    --workers 6 > /dev/null 2>&1 < /dev/null
  log "analysis $1 started"
}
roll() {   # <name> <split> <store> <design> <per-dataset> [extra...]
  local name=$1 split=$2 store=$3 design=$4 per=$5
  shift 5
  NAME=p1_$name bash "$R/tools/server/paper_rollouts.sh" --split "$split" --store "$store" --design "$design" \
    --out "$R/runs/paper/a1/$name" --per-dataset "$per" --n 32 --gpu-memory-utilization "$MEM" "$@"
  log "rollouts $name exit $(cat "$L/p1_$name.exit")"
  [ "$(cut -d' ' -f1 "$L/p1_$name.exit")" = 0 ] && analyse "$name" "$store"
}
ready() {   # <name>: wait until its engine is ready (or its run ended)
  until grep -q engine_ready_s "$L/p1_$1.log" 2>/dev/null || [ -f "$L/p1_$1.exit" ]; do sleep 20; done
}
until [ -f "$L/$WAIT_FOR.exit" ]; do sleep 30; done
log "$WAIT_FOR done: $(cat "$L/$WAIT_FOR.exit")"
analyse dev_L24 "$S/real_dev"
rm -f "$L"/p1_dev_L48.* "$L"/p1_dev_R.* "$L"/p1_test_L48.*
( roll dev_L48 tuning "$S/real_dev" L48 100; roll test_L24 held_out "$S/real_test" L24 500 ) &
ready dev_L48
( roll dev_R tuning "$S/real_dev" R 100; roll test_R held_out "$S/real_test" R 500;
  roll b2_L24_unknown held_out "$S/real_test" L24 300 --unknown-site ) &
ready dev_R
( roll test_L48 held_out "$S/real_test" L48 500 ) &
wait
log "lanes done"
