#!/usr/bin/env bash
# icutwin on the one-GPU host: Paper P's work after the A1/B2 rollouts (docs/p1_paper_protocol.md), each step started
# as soon as what it reads exists, to use the hours while the B2 lane still generates:
#   a. each held-out design's CPU baselines (LightGBM, logistic, A4) once its rollouts are done;
#   b. once test_L24 and test_L48 are done (the B2 engine alone holds about 9 GB of the GPU): TabPFN v2, the model
#      states (train and test anchors of L24, L48, R) with their probes (CPU) and C4, then B3's checkpoint states and
#      probes;
#   c. after the lanes (B2 done): TabPFN-3.5 (training context 50,000 rows, retried at 20,000 if it fails), the C1
#      what-if rollouts and analysis, C2 attribution, its rollout-level check and analysis;
#   d. the comparisons per design (tools/paper_compare.py) once every input exists.
# Logs: logs/pq2_<step>.{log,exit}; logs/paper_queue2.log (a line per start and end).
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
VPY=$BASE/vllm-env/bin/python
S=$R/data/factory/stores
T4=$R/data/tokens/full_r4
T3=$R/data/tokens/full_r3
P=$R/runs/paper
CK=$BASE/tabpfn-3.5/tabpfn-v3.5-20260909.safetensors
TPF=(env PYTHONPATH="$BASE/tabpfn-pkgs" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4)
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*" >> "$L/paper_queue2.log"; }
gpu() {   # <name> <command...>: a GPU step in the foreground
  local n=$1
  shift
  log "start $n"
  bash tools/run.sh "pq2_$n" env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  log "end $n: $(cat "$L/pq2_$n.exit")"
}
cpu() {   # <name> <command...>: a CPU step in the background
  local n=$1
  shift
  setsid -f bash tools/run.sh "pq2_$n" env OMP_NUM_THREADS=2 "$@" > /dev/null 2>&1 < /dev/null
  log "started $n (background)"
}
waitlog() { until [ -f "$L/$1.exit" ]; do sleep 30; done; }
ok() { [ "$(cut -d' ' -f1 "$L/$1.exit" 2>/dev/null)" = 0 ]; }
baselines() {
  cpu "baselines_$1" "$PY" "$R/tools/paper_baselines.py" --design-dir "$P/a1/test_$1" --train "$S/real_train" \
    --val "$S/real_dev" --test "$S/real_test" --twins "$S/bank_dev" --out "$P/a1/test_$1"
}
log "queue2 start"
# a. baselines as each design's rollouts end
waitlog p1_test_R; baselines R
waitlog p1_test_L24; baselines L24
waitlog p1_test_L48; baselines L48
# b. beside the B2 engine
for d in L24 L48 R; do
  gpu "tabpfn2_$d" "${TPF[@]}" TABPFN_MODEL_VERSION=v2 HF_ENDPOINT=https://hf-mirror.com "$PY" "$R/tools/paper_tabpfn.py" \
    --design-dir "$P/a1/test_$d" --train "$S/real_train" --test "$S/real_test" --out "$P/a1/test_$d"
done
for d in L24 R L48; do
  gpu "states_train_$d" "$PY" "$R/tools/server/paper_states.py" --run MUC32_main --tokens "$T4" --split train \
    --store "$S/real_train" --design "$d" --per-dataset 1500 --out "$P/states/train_$d" --tokens-per-batch 32768
  gpu "states_test_$d" "$PY" "$R/tools/server/paper_states.py" --run MUC32_main --tokens "$T4" --split held_out \
    --store "$S/real_test" --anchors-from "$P/a1/test_$d" --out "$P/states/test_$d" --tokens-per-batch 32768
  cpu "probe_$d" "$PY" "$R/tools/paper_probe.py" --train-states "$P/states/train_$d" --train-store "$S/real_train" \
    --test-states "$P/states/test_$d" --test-store "$S/real_test" --design "$d" --out "$P/a1/test_$d"
done
cpu c4 "$PY" "$R/tools/paper_c4.py" --states "$P/states/test_L24" --store "$S/real_test" --tokens "$T4" --out "$P/c4"
for run in MC32_main MCN32_main MU32_main MUH32_main MUF32_main M32_main ML32_main; do
  case $run in M32_main|ML32_main) T=$T3 ;; *) T=$T4 ;; esac
  gpu "b3_train_$run" "$PY" "$R/tools/server/paper_states.py" --run "$run" --tokens "$T" --split train \
    --store "$S/real_train" --anchors-from "$P/states/train_L24" --out "$P/b3/$run/train_L24" --tokens-per-batch 32768
  gpu "b3_test_$run" "$PY" "$R/tools/server/paper_states.py" --run "$run" --tokens "$T" --split held_out \
    --store "$S/real_test" --anchors-from "$P/a1/test_L24" --out "$P/b3/$run/test_L24" --tokens-per-batch 32768
  cpu "b3_probe_$run" "$PY" "$R/tools/paper_probe.py" --train-states "$P/b3/$run/train_L24" \
    --train-store "$S/real_train" --test-states "$P/b3/$run/test_L24" --test-store "$S/real_test" --design L24 \
    --out "$P/b3/$run"
done
# c. after the lanes
until grep -q "lanes done" "$L/paper_lanes.log" 2>/dev/null; do sleep 60; done
log "lanes done seen"
for d in L24 L48 R; do
  gpu "tabpfn35_$d" "${TPF[@]}" "$PY" "$R/tools/paper_tabpfn.py" --design-dir "$P/a1/test_$d" --train "$S/real_train" \
    --test "$S/real_test" --out "$P/a1/test_$d" --model-path "$CK" --name tabpfn_v3_5 --all-features --context 50000
  if ! ok "pq2_tabpfn35_$d"; then
    gpu "tabpfn35_${d}_20k" "${TPF[@]}" "$PY" "$R/tools/paper_tabpfn.py" --design-dir "$P/a1/test_$d" \
      --train "$S/real_train" --test "$S/real_test" --out "$P/a1/test_$d" --model-path "$CK" --name tabpfn_v3_5 \
      --all-features --context 20000
  fi
done
gpu whatif env PATH="$BASE/vllm-env/bin:$PATH" ICUTWIN_VOCAB="$T4/vocab.json" "$VPY" "$R/tools/server/paper_whatif.py" \
  --run MUC32_main --tokens "$T4" --split held_out --store "$S/real_test" --out "$P/c1"
cpu c1 "$PY" "$R/tools/paper_c1.py" --whatif "$P/c1" --work "$R/data/paper/stores/c1" --out "$P/c1" --workers 6
waitlog pq2_probe_L24
waitlog pq2_probe_R
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
# d. the comparisons
for d in L24 L48 R; do
  waitlog "p1_test_${d}_a1"; waitlog "pq2_baselines_$d"; waitlog "pq2_probe_$d"
done
waitlog p1_b2_L24_unknown_a1
for d in L24 L48 R; do
  extra=()
  [ "$d" = L24 ] && extra=(--unknown "$P/a1/b2_L24_unknown")
  cpu "compare_$d" "$PY" "$R/tools/paper_compare.py" --design-dir "$P/a1/test_$d" --dev "$P/a1/dev_$d" \
    --store "$S/real_test" --out "$P/a1/test_$d" "${extra[@]}"
done
log "queue2 done"
