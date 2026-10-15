#!/usr/bin/env bash
# icutwin on the one-GPU host: Paper P's C2 attribution again with integrated gradients from the padding embedding
# (amendment 6), after the rolling design's rollout-level check (logs/pq2_c2roll_R.exit) and the C2 rerun chain
# (paper_c2_rerun_run2.sh) are done. The edits, and so the rollout prompts and their rollouts, are unchanged (seeded),
# so only attrib.jsonl is rewritten and the analyses rerun. Logs: logs/pq2_{attrib,c2}_<d>.*.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
T4=$R/data/tokens/full_r4
P=$R/runs/paper
cd "$BASE" || exit 1
until [ -f "$L/pq2_c2roll_R.exit" ] && ! pgrep -f "paper_c2_rerun_run2.sh" > /dev/null; do sleep 60; done
run() { local n=$1; shift; rm -f "$L/pq2_$n.exit"; bash tools/run.sh "pq2_$n" "$@"
  echo "$(date '+%F %T') ig rerun $n: $(cat "$L/pq2_$n.exit")" >> "$L/paper_queue2.log"; }
for d in L24 R; do
  cp "$P/c2/test_$d/rollout_prompts.parquet" "$P/c2/test_$d/rollout_prompts.first.parquet"
done
run attrib_L24 env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" \
  "$R/tools/server/paper_attrib.py" --run MUC32_main --tokens "$T4" --split held_out --store "$S/real_test" \
  --anchors-from "$P/states/test_L24" --probe "$P/a1/test_L24" --programs "icu_death@-1" --out "$P/c2/test_L24"
run attrib_R env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" \
  "$R/tools/server/paper_attrib.py" --run MUC32_main --tokens "$T4" --split held_out --store "$S/real_test" \
  --anchors-from "$P/states/test_R" --probe "$P/a1/test_R" --programs "imv_onset@1440" "aki_2@2880" "hypotension@240" \
  --out "$P/c2/test_R"
for d in L24 R; do
  "$PY" -c "import pandas as pd,sys; a=pd.read_parquet(sys.argv[1]); b=pd.read_parquet(sys.argv[2]);
sys.exit(0 if a['id'].tolist()==b['id'].tolist() and all((x==y).all() for x,y in zip(a['prompt'],b['prompt'])) else 1)" \
    "$P/c2/test_$d/rollout_prompts.parquet" "$P/c2/test_$d/rollout_prompts.first.parquet" \
    && echo "$(date '+%F %T') rollout prompts $d unchanged" >> "$L/paper_queue2.log" \
    || echo "$(date '+%F %T') rollout prompts $d CHANGED: the rollout check needs new rollouts" >> "$L/paper_queue2.log"
  run "c2_$d" env OMP_NUM_THREADS=2 "$PY" "$R/tools/paper_c2.py" --attrib "$P/c2/test_$d" \
    --work "$R/data/paper/stores/c2_$d" --out "$P/c2/test_$d" --workers 3
done
