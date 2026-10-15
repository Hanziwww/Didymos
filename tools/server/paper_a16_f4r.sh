#!/usr/bin/env bash
# icutwin on the RTX 5090 host (westd): Paper P amendment 16c (docs/p1_paper_protocol.md), attention knockout against
# the generator, a replicate of F4's generator side (tools/server/paper_followup_gpu.sh, F4). In order: knockout on
# F4's 160 anchors (GPU, minutes; before the lanes, which must not share the GPU), F4's 2,698 saved prompts drawn again
# (48 rollouts each, seed 16, F4's recorded caps, three vLLM lanes started in turn), the rollouts' hazards, and F4's
# analysis with knockout added and every per-group effect saved (tools/paper_c2b.py --knockout --dump, CPU). The
# rollouts and their label store are deleted after the analysis. Logs: logs/a16c_*.{log,exit}, logs/paper_a16_f4r.log.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
P=$R/runs/paper
T4=$R/data/tokens/full_r4
MEM=${MEM:-0.27}
LANES=${LANES:-3}
cd "$BASE" || exit 1
mkdir -p "$L" "$P/f4r"
log() { echo "$(date '+%F %T') $*" >> "$L/paper_a16_f4r.log"; }
ok() { [ "$(cut -d' ' -f1 "$L/$1.exit" 2>/dev/null)" = 0 ]; }
ready() { until grep -q engine_ready_s "$L/$1.log" 2>/dev/null || [ -f "$L/$1.exit" ]; do sleep 20; done; }
log "start"
if ! ok a16c_knockout; then
  bash tools/run.sh a16c_knockout env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "$R/tools/server/paper_knockout_f4.py" --run MUC32_main --tokens "$T4" --split held_out \
    --anchors-from "$P/states/test_R" --probe "$P/a1/test_R" --anchors-file "$P/f4/attrib/anchors_f4.parquet" \
    --attrib "$P/f4/attrib/attrib.jsonl" --check-against "$P/f4/attn_diag/per_anchor.jsonl" --out "$P/f4r"
  log "knockout exit $(cat "$L/a16c_knockout.exit")"
  ok a16c_knockout || exit 1
fi
args=(--split held_out --caps-from "$P/f4/R/design.json" --prompts "$P/f4/attrib/rollout_prompts.parquet"
      --out "$P/f4r/R" --n 48 --cap-hours 72 --per-dataset-caps --design --seed 16)
for ((k = 0; k < LANES; k++)); do
  ok "a16c_s$k" && continue                       # a finished lane is not redrawn (batches are resumable)
  rm -f "$L/a16c_s$k.exit"
  ( NAME=a16c_s$k SCRIPT=paper_prompt_rollouts.py bash "$R/tools/server/paper_vllm.sh" "${args[@]}" --shard "$k" \
      --shards "$LANES" --gpu-memory-utilization "$MEM"; log "lane $k exit $(cat "$L/a16c_s$k.exit")" ) &
  ready "a16c_s$k"
done
wait
for ((k = 0; k < LANES; k++)); do ok "a16c_s$k" || { log "lanes failed"; exit 1; }; done
if ! ok a16c_hz; then
  bash tools/run.sh a16c_hz env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "$R/tools/server/paper_hazards.py" --run MUC32_main --tokens "$T4" --split held_out --design-dir "$P/f4r/R" \
    --check 0
  log "hazards exit $(cat "$L/a16c_hz.exit")"
  ok a16c_hz || exit 1
fi
bash tools/run.sh a16c_c2b env OMP_NUM_THREADS=4 "$PY" "$R/tools/paper_c2b.py" --f4 "$P/f4r/R" \
  --attrib "$P/f4/attrib/attrib.jsonl" --attrib-random "$P/f4/attrib_random/attrib.jsonl" \
  --attrib-perm "$P/f4/attrib_perm/attrib.jsonl" --knockout "$P/f4r/knockout.jsonl" --tokens "$T4" \
  --work "$R/data/paper/stores/f4r" --out "$P/f4r" --dump "$P/f4r"
log "c2b exit $(cat "$L/a16c_c2b.exit")"
if ok a16c_c2b; then
  rm -rf "$P/f4r/R/rollouts" "$R/data/paper/stores/f4r"     # the protocol: rollouts deleted after their analyses
  log "rollouts deleted"
fi
log "done"
