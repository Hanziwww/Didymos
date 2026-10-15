#!/usr/bin/env bash
# icutwin on the RTX 4080 host (westc, which holds the test store): Paper P amendment 16b, L48x (docs/p1_paper_protocol.md).
# The 48-hour landmark with the follow-up's settings, as F1-F3 did at L24x (tools/server/paper_f1.sh): the subset's
# design files (--dry), 128 rollouts per anchor in three lanes (one vLLM engine each at MEM, started in turn), rollout
# seed 16, four times A1's caps; the hazards of the rollouts and of the real futures (GPU, after the lanes); then the
# comparators for death within 24 h at the landmark (TabPFN-3.5 on the GPU, the eight learners of amendment 10 on the
# CPU) and the subset's analysis (tools/paper_a1.py, then tools/paper_f2.py, CPU). The rollouts are deleted after the
# analysis. Needs runs/paper/f1/subsets/L48x.parquet (tools/paper_f1_subsets.py --design L48 --name L48x --seed 16).
# Logs: logs/a16b_*.{log,exit}, logs/paper_a16_L48x.log.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
T4=$R/data/tokens/full_r4
F=$R/runs/paper/f1
A1=$R/runs/paper/a1
O=$R/runs/paper/a16
MEM=${MEM:-0.27}
LANES=${LANES:-3}
CK=${CK:-$BASE/tabpfn-3.5/tabpfn-v3.5-20260909.safetensors}
cd "$BASE" || exit 1
mkdir -p "$O"
log() { echo "$(date '+%F %T') $*" >> "$L/paper_a16_L48x.log"; }
ok() { [ "$(cut -d' ' -f1 "$L/$1.exit" 2>/dev/null)" = 0 ]; }
ready() { until grep -q engine_ready_s "$L/$1.log" 2>/dev/null || [ -f "$L/$1.exit" ]; do sleep 20; done; }
log "start"
args=(--split held_out --store "$S/real_test" --design L48 --out "$F/L48x" --subset "$F/subsets/L48x.parquet"
      --n 128 --rollout-seed 16 --cap-mult 4)
if ! ok a16b_dry; then
  NAME=a16b_dry bash "$R/tools/server/paper_rollouts.sh" "${args[@]}" --dry
  ok a16b_dry || { log "design failed"; exit 1; }
fi
log "design: $(grep -o '"anchors": [0-9]*' "$L/a16b_dry.log" | head -1)"
for ((k = 0; k < LANES; k++)); do
  ok "a16b_s$k" && continue                       # a finished lane is not redrawn (oracle.run resumes batches)
  rm -f "$L/a16b_s$k.exit"
  ( NAME=a16b_s$k bash "$R/tools/server/paper_rollouts.sh" "${args[@]}" --shard "$k" --shards "$LANES" \
      --gpu-memory-utilization "$MEM"; log "lane $k exit $(cat "$L/a16b_s$k.exit")" ) &
  ready "a16b_s$k"
done
wait
for ((k = 0; k < LANES; k++)); do ok "a16b_s$k" || { log "lanes failed"; exit 1; }; done
if ! ok a16b_hz; then
  bash tools/run.sh a16b_hz env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "$R/tools/server/paper_hazards.py" --run MUC32_main --tokens "$T4" --split held_out --design-dir "$F/L48x" \
    --real --long --a1-caps "$A1/test_L48/design.json"
  log "hazards exit $(cat "$L/a16b_hz.exit")"
  ok a16b_hz || exit 1
fi
if ! ok a16b_tabpfn; then                          # the GPU is free again: TabPFN-3.5 for the added endpoint
  mkdir -p "$O/L48x_tabpfn"
  bash tools/run.sh a16b_tabpfn env PYTHONPATH="$BASE/tabpfn-pkgs" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 "$PY" "$R/tools/paper_tabpfn.py" \
    --design-dir "$A1/test_L48" --train "$S/real_train" --test "$S/real_test" --out "$O/L48x_tabpfn" \
    --model-path "$CK" --name tabpfn_v3_5 --all-features --context 50000 --programs decompensation@1440
  log "tabpfn exit $(cat "$L/a16b_tabpfn.exit")"
fi
if ! ok a16b_ml; then
  bash tools/run.sh a16b_ml env OMP_NUM_THREADS=1 "$PY" "$R/tools/paper_baselines_more.py" \
    --design-dir "$A1/test_L48" --train "$S/real_train" --val "$S/real_dev" --test "$S/real_test" \
    --out "$O/L48x_ml" --programs decompensation@1440 --n-jobs 8
  log "learners exit $(cat "$L/a16b_ml.exit")"
fi
if ! ok a16b_a1; then
  bash tools/run.sh a16b_a1 env OMP_NUM_THREADS=1 "$PY" "$R/tools/paper_a1.py" --design-dir "$F/L48x" \
    --store "$S/real_test" --work "$R/data/paper/stores/a16_L48x" --out "$F/L48x" --workers 4 --chunk 100 \
    --min-valid 64 --curve-draws 20
  log "analysis exit $(cat "$L/a16b_a1.exit")"
  ok a16b_a1 || exit 1
fi
bash tools/run.sh a16b_f2 env OMP_NUM_THREADS=4 "$PY" "$R/tools/paper_f2.py" --subset "$F/L48x" --a1 "$A1/test_L48" \
  --out "$F/L48x" --long
log "f2 exit $(cat "$L/a16b_f2.exit")"
if ok a16b_f2; then
  rm -rf "$F/L48x/rollouts" "$R/data/paper/stores/a16_L48x"     # the protocol: rollouts deleted after their analyses
  log "rollouts deleted"
fi
log "done"
