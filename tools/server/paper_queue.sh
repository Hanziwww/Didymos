#!/usr/bin/env bash
# icutwin on the one-GPU host: Paper P's GPU work after the A1/B2 rollout lanes (docs/p1_paper_protocol.md), one job at
# a time on GPU 0, each CPU analysis in the background as soon as its inputs exist:
#   1. model states (MUC32_main) at train and held-out anchors of L24, L48, R -> the Twin probes (A1), C4;
#   2. C1 what-if rollouts -> C1 analysis;
#   3. C2 attribution (needs the probes) -> its rollout-level prompts -> C2 analysis;
#   4. B3: L24 states of the other r3/r4 checkpoints -> their probes.
# Waits for logs/paper_lanes.log to say "lanes done" (WAIT=0 skips). Logs: logs/pq_<step>.{log,exit}; logs/paper_queue.log.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
VPY=$BASE/vllm-env/bin/python
S=$R/data/factory/stores
T4=$R/data/tokens/full_r4
T3=$R/data/tokens/full_r3
P=$R/runs/paper
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*" >> "$L/paper_queue.log"; }
gpu() {   # <name> <command...>: a GPU step in the foreground
  local n=$1
  shift
  bash tools/run.sh "pq_$n" env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  log "$n exit $(cat "$L/pq_$n.exit")"
}
cpu() {   # <name> <command...>: a CPU step in the background
  local n=$1
  shift
  setsid -f bash tools/run.sh "pq_$n" env OMP_NUM_THREADS=2 "$@" > /dev/null 2>&1 < /dev/null
  log "$n started"
}
waitfor() { until [ -f "$L/pq_$1.exit" ]; do sleep 30; done; }
if [ "${WAIT:-1}" = 1 ]; then until grep -q "lanes done" "$L/paper_lanes.log" 2>/dev/null; do sleep 60; done; fi
log "queue start"
# 0. the supervised baselines and A4 (CPU, beside the GPU steps)
for d in L24 L48 R; do
  cpu "baselines_$d" "$PY" "$R/tools/paper_baselines.py" --design-dir "$P/a1/test_$d" --train "$S/real_train" \
    --val "$S/real_dev" --test "$S/real_test" --twins "$S/bank_dev" --out "$P/a1/test_$d"
done
# 1. states and probes
for d in L24 L48 R; do
  gpu "states_train_$d" "$PY" "$R/tools/server/paper_states.py" --run MUC32_main --tokens "$T4" --split train \
    --store "$S/real_train" --design "$d" --per-dataset 1500 --out "$P/states/train_$d"
  gpu "states_test_$d" "$PY" "$R/tools/server/paper_states.py" --run MUC32_main --tokens "$T4" --split held_out \
    --store "$S/real_test" --anchors-from "$P/a1/test_$d" --out "$P/states/test_$d"
  cpu "probe_$d" "$PY" "$R/tools/paper_probe.py" --train-states "$P/states/train_$d" --train-store "$S/real_train" \
    --test-states "$P/states/test_$d" --test-store "$S/real_test" --design "$d" --out "$P/a1/test_$d"
done
cpu c4 "$PY" "$R/tools/paper_c4.py" --states "$P/states/test_L24" --store "$S/real_test" --tokens "$T4" --out "$P/c4"
# 2. what-if
gpu whatif env PATH="$BASE/vllm-env/bin:$PATH" ICUTWIN_VOCAB="$T4/vocab.json" "$VPY" "$R/tools/server/paper_whatif.py" \
  --run MUC32_main --tokens "$T4" --split held_out --store "$S/real_test" --out "$P/c1"
cpu c1 "$PY" "$R/tools/paper_c1.py" --whatif "$P/c1" --work "$R/data/paper/stores/c1" --out "$P/c1" --workers 6
# 3. attribution (the probes first)
waitfor probe_L24
waitfor probe_R
gpu attrib_L24 "$PY" "$R/tools/server/paper_attrib.py" --run MUC32_main --tokens "$T4" --split held_out \
  --store "$S/real_test" --anchors-from "$P/states/test_L24" --probe "$P/a1/test_L24" --programs "icu_death@-1" \
  --out "$P/c2/test_L24"
gpu attrib_R "$PY" "$R/tools/server/paper_attrib.py" --run MUC32_main --tokens "$T4" --split held_out \
  --store "$S/real_test" --anchors-from "$P/states/test_R" --probe "$P/a1/test_R" \
  --programs "imv_onset@1440" "aki_2@2880" "hypotension@240" --out "$P/c2/test_R"
for d in L24 R; do
  gpu "c2roll_$d" env PATH="$BASE/vllm-env/bin:$PATH" ICUTWIN_VOCAB="$T4/vocab.json" "$VPY" \
    "$R/tools/server/paper_prompt_rollouts.py" --run MUC32_main --tokens "$T4" --split held_out --store "$S/real_test" \
    --prompts "$P/c2/test_$d/rollout_prompts.parquet" --out "$P/c2/test_$d" --n 16 --cap-hours 72
  cpu "c2_$d" "$PY" "$R/tools/paper_c2.py" --attrib "$P/c2/test_$d" --work "$R/data/paper/stores/c2_$d" \
    --out "$P/c2/test_$d" --workers 6
done
# 4. B3: the other checkpoints' L24 states and probes
for run in MC32_main MCN32_main MU32_main MUH32_main MUF32_main M32_main ML32_main; do
  case $run in M32_main|ML32_main) T=$T3 ;; *) T=$T4 ;; esac
  gpu "b3_train_$run" "$PY" "$R/tools/server/paper_states.py" --run "$run" --tokens "$T" --split train \
    --store "$S/real_train" --anchors-from "$P/states/train_L24" --out "$P/b3/$run/train_L24"
  gpu "b3_test_$run" "$PY" "$R/tools/server/paper_states.py" --run "$run" --tokens "$T" --split held_out \
    --store "$S/real_test" --anchors-from "$P/a1/test_L24" --out "$P/b3/$run/test_L24"
  cpu "b3_probe_$run" "$PY" "$R/tools/paper_probe.py" --train-states "$P/b3/$run/train_L24" \
    --train-store "$S/real_train" --test-states "$P/b3/$run/test_L24" --test-store "$S/real_test" --design L24 \
    --out "$P/b3/$run"
done
log "queue done"
