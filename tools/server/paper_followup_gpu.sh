#!/usr/bin/env bash
# icutwin on the one-GPU host: Paper P amendment 7's GPU work after F1 (tools/server/paper_f1.sh): F7's random-weight
# states, F4 (the attributions: trained, random backbone, permuted-label probe; the edited prompts' rollouts in three
# lanes; their hazards), F5 (the what-if arms in three lanes, once the CPU driver has written the design) and F9a (R128
# with the site unknown and swapped, three lanes each). The analyses go to the CPU driver's queue
# (logs/cpu_queue.txt). Logs: logs/fg_*.{log,exit}, logs/paper_followup.log.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
P=$R/runs/paper
T4=$R/data/tokens/full_r4
MEM=${MEM:-0.27}
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') gpu $*" >> "$L/paper_followup.log"; }
ok() { [ "$(cut -d' ' -f1 "$L/$1.exit" 2>/dev/null)" = 0 ]; }
queue() { echo "$1|$2" >> "$L/cpu_queue.txt"; log "queued $1"; }
gpu() {   # <name> <command...>
  local name=$1
  shift
  bash tools/run.sh "fg_$name" env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  log "$name exit $(cat "$L/fg_$name.exit")"
}
ready() { until grep -q engine_ready_s "$L/$1.log" 2>/dev/null || [ -f "$L/$1.exit" ]; do sleep 20; done; }
lanes() {   # <name> <script> <lanes> [args...]: one engine per lane, started in turn
  local name=$1 script=$2 n=$3 k
  shift 3
  for ((k = 0; k < n; k++)); do
    rm -f "$L/fg_${name}_s$k.exit"
    ( NAME=fg_${name}_s$k SCRIPT=$script bash "$R/tools/server/paper_vllm.sh" "$@" --shard "$k" --shards "$n" \
        --gpu-memory-utilization "$MEM"; log "lane $name/$k exit $(cat "$L/fg_${name}_s$k.exit")" ) &
    ready "fg_${name}_s$k"
  done
  wait
  for ((k = 0; k < n; k++)); do ok "fg_${name}_s$k" || return 1; done
}
until grep -q " done$" "$L/paper_f1.log" 2>/dev/null; do sleep 120; done
log start
# F7: the random-weight states (a negative control of C4)
for d in L24 R; do
  gpu "states_random_$d" "$PY" "$R/tools/server/paper_states.py" --run MUC32_main --tokens "$T4" --split held_out \
    --store "$S/real_test" --anchors-from "$P/a1/test_$d" --random-init 7 --out "$P/states/test_${d}_random"
done
queue f7_c4b_L24 "$PY $R/tools/paper_c4b.py --design L24 --states $P/states/test_L24 --random-states $P/states/test_L24_random --a1 $P/a1/test_L24 --out $P/c4"
queue f7_c4b_R "$PY $R/tools/paper_c4b.py --design R --states $P/states/test_R --random-states $P/states/test_R_random --a1 $P/a1/test_R --out $P/c4"
# F4: attributions (the permuted-label probe first, on CPU here: minutes)
F4P="decompensation@1440 imv_onset@1440 aki_2@2880 hypotension@240"
gpu f4_probe_perm "$PY" "$R/tools/paper_probe.py" --train-states "$P/states/train_R" --train-store "$S/real_train" \
  --test-states "$P/states/test_R" --test-store "$S/real_test" --design R --out "$P/f4/probe_perm" --permute-seed 7 \
  --programs $F4P
ATT=(--run MUC32_main --tokens "$T4" --split held_out --store "$S/real_test" --anchors-from "$P/states/test_R"
     --probe "$P/a1/test_R" --programs $F4P)
gpu f4_attrib "$PY" "$R/tools/server/paper_attrib.py" "${ATT[@]}" --select tophalf --n-select 40 --seed 7 \
  --labels "$P/a1/test_R/predictions.parquet" --rollout-all --out "$P/f4/attrib"
"$PY" -c "import json, pandas as pd; r=[json.loads(l) for l in open('$P/f4/attrib/attrib.jsonl')]; pd.DataFrame({'program':[x['program'] for x in r],'anchor':[x['anchor'] for x in r]}).to_parquet('$P/f4/attrib/anchors_f4.parquet', index=False)"
gpu f4_attrib_random "$PY" "$R/tools/server/paper_attrib.py" "${ATT[@]}" --anchors-file "$P/f4/attrib/anchors_f4.parquet" \
  --random-backbone 7 --no-curves --out "$P/f4/attrib_random"
gpu f4_attrib_perm "$PY" "$R/tools/server/paper_attrib.py" "${ATT[@]}" --anchors-file "$P/f4/attrib/anchors_f4.parquet" \
  --probe-file "$P/f4/probe_perm/probe_models.npz" --no-curves --out "$P/f4/attrib_perm"
lanes f4 paper_prompt_rollouts.py 3 --split held_out --store "$S/real_test" --prompts "$P/f4/attrib/rollout_prompts.parquet" \
  --out "$P/f4/R" --n 48 --cap-hours 72 --per-dataset-caps --design --seed 7 && \
  gpu f4_hazards "$PY" "$R/tools/server/paper_hazards.py" --run MUC32_main --tokens "$T4" --split held_out \
    --design-dir "$P/f4/R" --check 0 && \
  queue f4_c2b "$PY $R/tools/paper_c2b.py --f4 $P/f4/R --attrib $P/f4/attrib/attrib.jsonl --attrib-random $P/f4/attrib_random/attrib.jsonl --attrib-perm $P/f4/attrib_perm/attrib.jsonl --work $R/data/paper/stores/f4 --out $P/f4"
# F5: the what-if arms (the design comes from the CPU driver)
until [ -f "$L/fc_f5_design.exit" ]; do sleep 60; done
ok fc_f5_design && lanes f5 paper_whatif2.py 3 --split held_out --store "$S/real_test" --design "$P/f5/design.parquet" \
  --out "$P/f5" --n 16 && \
  queue f5_c1b "$PY $R/tools/paper_c1b.py --whatif $P/f5 --work $R/data/paper/stores/f5 --out $P/f5 --workers 4"
# F9a: R128 with the site unknown, and swapped to another dataset (32 rollouts each)
for t in unknown swap; do
  NAME=fg_f9_${t}_dry bash "$R/tools/server/paper_rollouts.sh" --split held_out --store "$S/real_test" --design R \
    --out "$P/f9/R_$t" --subset "$P/f1/subsets/R128.parquet" --n 32 --rollout-seed 7 --prompt-transform "$t" --dry
  for ((k = 0; k < 3; k++)); do
    rm -f "$L/fg_f9_${t}_s$k.exit"
    ( NAME=fg_f9_${t}_s$k bash "$R/tools/server/paper_rollouts.sh" --split held_out --store "$S/real_test" --design R \
        --out "$P/f9/R_$t" --subset "$P/f1/subsets/R128.parquet" --n 32 --rollout-seed 7 --prompt-transform "$t" \
        --shard "$k" --shards 3 --gpu-memory-utilization "$MEM"; log "lane f9_$t/$k exit $(cat "$L/fg_f9_${t}_s$k.exit")" ) &
    ready "fg_f9_${t}_s$k"
  done
  wait
  queue "f9a_a1_$t" "$PY $R/tools/paper_a1.py --design-dir $P/f9/R_$t --store $S/real_test --work $R/data/paper/stores/f9_$t --out $P/f9/R_$t --workers 4 --chunk 100"
done
queue f9a_first32 "$PY $R/tools/paper_a1.py --design-dir $P/f1/R128 --store $S/real_test --work $R/data/paper/stores/f1_R128 --out $P/f9/R128_first32 --workers 4 --chunk 100 --max-rollouts 32"
queue f9_transfer_R "$PY $R/tools/paper_transfer.py --design R --a1 $P/a1/test_R --states $P/states --known $P/f9/R128_first32 --unknown $P/f9/R_unknown --swap $P/f9/R_swap --out $P/f9 --train $S/real_train --val $S/real_dev --test $S/real_test"
echo END >> "$L/cpu_queue.txt"
log done
