#!/usr/bin/env bash
# icutwin on the RTX 5090 host (westd): Paper P amendment 16a (docs/p1_paper_protocol.md), transfer to RICD and
# zigongICU, two databases never seen in training. Needs data/tokens/ext_r4/{ricd,zigong} (encoded with MUC32_main's
# frozen vocabulary; its vocabulary files are copied from full_r4 here and checked) and, from the local design step
# (tools/paper_external.py design), runs/paper/ext/subsets/{R,L24e}.parquet, runs/paper/ext/local/<design>_anchors.parquet
# and runs/paper/ext/local/L24e_design.json (the landmark's caps at one times the 7-day quantile, A1's cap).
# In order:
#   1. the external store (tools/factory_store.py tokens --all-splits: every stay is a test stay);
#   2. per design, its files (--dry), checked against the local design (the same anchors, prompt lengths and times);
#   3. 128 rollouts per anchor in three lanes (one vLLM engine each at MEM, started in turn), prompts with the
#      database and site tokens unknown, rollout seed 16; R at A1's caps (the 72-hour quantile), L24e at four times
#      the 7-day quantile;
#   4. the hazards of the rollouts and of the real futures (GPU, after the lanes);
#   5. the analysis (tools/paper_a1.py, then tools/paper_f2.py, CPU) and the model states at the anchors for the probe
#      (tools/server/paper_states.py, GPU).
# The rollouts and work stores are deleted after the analysis. Logs: logs/a16a_*.{log,exit}, logs/paper_a16_ext.log.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
T4=$R/data/tokens/full_r4
TE=$R/data/tokens/ext_r4
P=$R/runs/paper/ext
SPLITS=train,tuning,held_out
MEM=${MEM:-0.27}
LANES=${LANES:-3}
DESIGNS=${DESIGNS:-"R L24e"}
cd "$BASE" || exit 1
mkdir -p "$P"
log() { echo "$(date '+%F %T') $*" >> "$L/paper_a16_ext.log"; }
ok() { [ "$(cut -d' ' -f1 "$L/$1.exit" 2>/dev/null)" = 0 ]; }
ready() { until grep -q engine_ready_s "$L/$1.log" 2>/dev/null || [ -f "$L/$1.exit" ]; do sleep 20; done; }
log "start"
for f in vocab.json vocabulary.parquet support.json rank_persistence.json goal_bindings.json los_bands.json \
         calibration.json calibration_events.json; do
  [ -f "$TE/$f" ] || cp -p "$T4/$f" "$TE/$f"
  cmp -s "$T4/$f" "$TE/$f" || { log "$f differs from full_r4"; exit 1; }
done
if ! ok a16a_store; then
  rm -rf "$S/ext_test"
  bash tools/run.sh a16a_store env OMP_NUM_THREADS=1 "$PY" "$R/tools/factory_store.py" tokens --tokens "$TE" \
    --all-splits --label test --out "$S/ext_test" --datasets zigong ricd --workers 8
  log "store exit $(cat "$L/a16a_store.exit")"
  ok a16a_store || exit 1
fi
for D in $DESIGNS; do
  case $D in
    R) dargs=(--design R --per-dataset 500); hz=(); f2=() ;;
    L24e) dargs=(--design L24 --per-dataset 1000000000 --cap-mult 4); f2=(--long)
          hz=(--long --a1-caps "$P/local/L24e_design.json") ;;      # A1's cap: the 7-day quantile itself
    *) log "unknown design $D"; exit 1 ;;
  esac
  args=(--tokens "$TE" --split "$SPLITS" --store "$S/ext_test" "${dargs[@]}" --seed 16 --out "$P/$D"
        --subset "$P/subsets/$D.parquet" --n 128 --rollout-seed 16 --prompt-transform unknown --datasets ricd zigong)
  if ! ok "a16a_${D}_dry"; then
    NAME=a16a_${D}_dry bash "$R/tools/server/paper_rollouts.sh" "${args[@]}" --dry
    ok "a16a_${D}_dry" || { log "$D design failed"; exit 1; }
    "$PY" - "$P/$D/anchors.parquet" "$P/local/${D}_anchors.parquet" <<'EOF' || { log "$D differs from the local design"; exit 1; }
import sys

import pandas as pd

host, local = (pd.read_parquet(p).set_index("anchor") for p in sys.argv[1:3])
cols = ["twin", "dataset", "hours", "prompt_len", "t_min"]
same = host[cols].equals(local.loc[host.index, cols])
print({"host": len(host), "local": len(local), "same": bool(same)})
sys.exit(0 if same else 1)
EOF
  fi
  log "$D design: $(grep -o '"anchors": [0-9]*' "$L/a16a_${D}_dry.log" | head -1)"
  for ((k = 0; k < LANES; k++)); do
    ok "a16a_${D}_s$k" && continue                 # a finished lane is not redrawn (oracle.run resumes batches)
    rm -f "$L/a16a_${D}_s$k.exit"
    ( NAME=a16a_${D}_s$k bash "$R/tools/server/paper_rollouts.sh" "${args[@]}" --shard "$k" --shards "$LANES" \
        --gpu-memory-utilization "$MEM"; log "$D lane $k exit $(cat "$L/a16a_${D}_s$k.exit")" ) &
    ready "a16a_${D}_s$k"
  done
  wait
  for ((k = 0; k < LANES; k++)); do ok "a16a_${D}_s$k" || { log "$D lanes failed"; exit 1; }; done
  if ! ok "a16a_${D}_hz"; then
    bash tools/run.sh "a16a_${D}_hz" env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" "$R/tools/server/paper_hazards.py" --run MUC32_main --tokens "$TE" --split "$SPLITS" --design-dir "$P/$D" \
      --datasets ricd zigong --real "${hz[@]}"
    log "$D hazards exit $(cat "$L/a16a_${D}_hz.exit")"
    ok "a16a_${D}_hz" || exit 1
  fi
  if ! ok "a16a_${D}_states"; then                  # the GPU is free: the model states for the probe
    bash tools/run.sh "a16a_${D}_states" env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" "$R/tools/server/paper_states.py" --run MUC32_main --tokens "$TE" --split "$SPLITS" --store "$S/ext_test" \
      --anchors-from "$P/$D" --datasets ricd zigong --out "$P/states_$D"
    log "$D states exit $(cat "$L/a16a_${D}_states.exit")"
  fi
  if ! ok "a16a_${D}_a1"; then
    bash tools/run.sh "a16a_${D}_a1" env OMP_NUM_THREADS=1 "$PY" "$R/tools/paper_a1.py" --design-dir "$P/$D" \
      --store "$S/ext_test" --tokens "$TE" --work "$R/data/paper/stores/a16_ext_$D" --out "$P/$D" --workers 8 \
      --chunk 100 --min-valid 64 --curve-draws 20
    log "$D analysis exit $(cat "$L/a16a_${D}_a1.exit")"
    ok "a16a_${D}_a1" || exit 1
  fi
  bash tools/run.sh "a16a_${D}_f2" env OMP_NUM_THREADS=4 "$PY" "$R/tools/paper_f2.py" --subset "$P/$D" --a1 "$P/$D" \
    --out "$P/$D" "${f2[@]}"
  log "$D f2 exit $(cat "$L/a16a_${D}_f2.exit")"
  if ok "a16a_${D}_f2"; then
    rm -rf "$P/$D/rollouts" "$R/data/paper/stores/a16_ext_$D"   # the protocol: rollouts deleted after their analyses
    log "$D rollouts deleted"
  fi
done
log "done"
