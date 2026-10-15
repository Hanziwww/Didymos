#!/usr/bin/env bash
# icutwin on the one-GPU host: Paper P's C2 chain again (attribution -> rollout-level check -> analysis), as in
# paper_queue2.sh, after a fix. Logs: logs/pq2_<step>.{log,exit} (the names the comparisons read).
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
VPY=$BASE/vllm-env/bin/python
S=$R/data/factory/stores
T4=$R/data/tokens/full_r4
P=$R/runs/paper
cd "$BASE" || exit 1
gpu() { local n=$1; shift; rm -f "$L/pq2_$n.exit"
  bash tools/run.sh "pq2_$n" env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  echo "$(date '+%F %T') rerun $n: $(cat "$L/pq2_$n.exit")" >> "$L/paper_queue2.log"; }
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
  rm -f "$L/pq2_c2_$d.exit"
  bash tools/run.sh "pq2_c2_$d" env OMP_NUM_THREADS=2 "$PY" "$R/tools/paper_c2.py" --attrib "$P/c2/test_$d" \
    --work "$R/data/paper/stores/c2_$d" --out "$P/c2/test_$d" --workers 3
  echo "$(date '+%F %T') rerun c2_$d: $(cat "$L/pq2_c2_$d.exit")" >> "$L/paper_queue2.log"
done
