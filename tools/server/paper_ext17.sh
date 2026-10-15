#!/usr/bin/env bash
# icutwin on the RTX 5090 host (westd): Paper P amendment 17's futures (docs/p1_paper_protocol.md), arms Z1 (MUC32_main)
# and F1 (the adapted run chosen in runs/paper/ext17/grid.json, after its M7 gate) on the improved encoding
# (data/tokens/ext_r4b, uploaded; vocabulary files checked against full_r4). For each arm and anchor set
# (runs/paper/ext17/{test,label}_{R,L24e}.parquet, frozen by tools/paper_ext17.py split):
#   1. the design files (--dry, the anchors read from the file, amendment 16a's caps from its design.json), checked
#      against the anchor file (the same anchors, stays, hours and weights);
#   2. 128 futures per anchor in three lanes (one vLLM engine each at MEM, started in turn), prompts with the database
#      and site tokens unknown, rollout seed 17;
#   3. the hazards of the futures and of the real stays (L24e: A1's cap, runs/paper/ext/local/L24e_design.json);
#   4. tools/paper_a1.py with amendment 16a's labels (the ext_test store, its encoding) and tools/paper_f2.py;
#   5. the futures and work stores deleted.
# ARMS (default "Z1 F1"), SETS (default "test_R test_L24e label_R label_L24e"). Outputs runs/paper/ext17/<arm>/<set>;
# logs logs/e17r_*.{log,exit}, logs/paper_ext17.log.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
T4=$R/data/tokens/full_r4
TB=$R/data/tokens/ext_r4b
P=$R/runs/paper/ext17
PE=$R/runs/paper/ext
SPLITS=train,tuning,held_out
MEM=${MEM:-0.27}
LANES=${LANES:-3}
ARMS=${ARMS:-"Z1 F1"}
SETS=${SETS:-"test_R test_L24e label_R label_L24e"}
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*" >> "$L/paper_ext17.log"; }
ok() { [ "$(cut -d' ' -f1 "$L/$1.exit" 2>/dev/null)" = 0 ]; }
ready() { until grep -q engine_ready_s "$L/$1.log" 2>/dev/null || [ -f "$L/$1.exit" ]; do sleep 20; done; }
log "start ($ARMS; $SETS)"
for f in vocab.json vocabulary.parquet support.json rank_persistence.json goal_bindings.json los_bands.json \
         calibration.json calibration_events.json; do
  cmp -s "$T4/$f" "$TB/$f" || { log "$f differs from full_r4"; exit 1; }
done
for ARM in $ARMS; do
  case $ARM in
    Z1) RUN=MUC32_main; gx=() ;;
    F1) RUN=$("$PY" -c "import json; print(json.load(open('$P/grid.json'))['chosen'])") || exit 1
        EX=$R/runs/audit/m7_$RUN/owner_exemption.json     # amendment 17d: checked by paper_rollouts.py
        if [ "$("$PY" -c "import json; print(json.load(open('$R/runs/audit/m7_$RUN/report.json'))['gate'])")" = PASS ]; then
          gx=()
        elif [ -f "$EX" ]; then
          gx=(--gate-exemption "$EX"); log "F1: the gate of $RUN is not PASS; the study lead's exemption ($EX)"
        else
          log "F1: the M7 gate of $RUN is not PASS"; exit 1
        fi ;;
    *) log "unknown arm $ARM"; exit 1 ;;
  esac
  for SET in $SETS; do
    D=${SET#*_}
    case $D in
      R) dargs=(--design R); hz=(); f2=() ;;
      L24e) dargs=(--design L24); hz=(--long --a1-caps "$PE/local/L24e_design.json"); f2=(--long) ;;
      *) log "unknown set $SET"; exit 1 ;;
    esac
    O=$P/$ARM/$SET
    N=e17r_${ARM}_$SET
    mkdir -p "$O"
    args=(--run "$RUN" --tokens "$TB" --split "$SPLITS" --store "$S/ext_test" "${dargs[@]}" --seed 17 --out "$O"
          --anchors-file "$P/$SET.parquet" --caps-from "$PE/$D/design.json" --n 128 --rollout-seed 17
          --prompt-transform unknown --datasets ricd zigong "${gx[@]}")
    if ! ok "${N}_dry"; then
      NAME=${N}_dry bash "$R/tools/server/paper_rollouts.sh" "${args[@]}" --dry
      ok "${N}_dry" || { log "$ARM $SET design failed"; exit 1; }
      "$PY" - "$O/anchors.parquet" "$P/$SET.parquet" <<'EOF' || { log "$ARM $SET differs from the anchor file"; exit 1; }
import sys

import pandas as pd

host, frozen = (pd.read_parquet(p).set_index("anchor") for p in sys.argv[1:3])
cols = ["twin", "dataset", "hours", "weight"]
same = set(host.index) == set(frozen.index) and host[cols].equals(frozen.loc[host.index, cols])
print({"host": len(host), "frozen": len(frozen), "same": bool(same)})
sys.exit(0 if same else 1)
EOF
    fi
    log "$ARM $SET design: $(grep -o '"anchors": [0-9]*' "$L/${N}_dry.log" | head -1)"
    for ((k = 0; k < LANES; k++)); do
      ok "${N}_s$k" && continue                     # a finished lane is not redrawn (oracle.run resumes batches)
      rm -f "$L/${N}_s$k.exit"
      ( NAME=${N}_s$k bash "$R/tools/server/paper_rollouts.sh" "${args[@]}" --shard "$k" --shards "$LANES" \
          --gpu-memory-utilization "$MEM"; log "$ARM $SET lane $k exit $(cat "$L/${N}_s$k.exit")" ) &
      ready "${N}_s$k"
    done
    wait
    for ((k = 0; k < LANES; k++)); do ok "${N}_s$k" || { log "$ARM $SET lanes failed"; exit 1; }; done
    if ! ok "${N}_hz"; then
      bash tools/run.sh "${N}_hz" env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PY" "$R/tools/server/paper_hazards.py" --run "$RUN" --tokens "$TB" --split "$SPLITS" --design-dir "$O" \
        --datasets ricd zigong --real "${hz[@]}"
      log "$ARM $SET hazards exit $(cat "$L/${N}_hz.exit")"
      ok "${N}_hz" || exit 1
    fi
    if ! ok "${N}_a1"; then
      bash tools/run.sh "${N}_a1" env OMP_NUM_THREADS=1 "$PY" "$R/tools/paper_a1.py" --design-dir "$O" \
        --store "$S/ext_test" --tokens "$TB" --work "$R/data/paper/stores/e17_${ARM}_$SET" --out "$O" --workers 8 \
        --chunk 100 --min-valid 64 --curve-draws 20
      log "$ARM $SET analysis exit $(cat "$L/${N}_a1.exit")"
      ok "${N}_a1" || exit 1
    fi
    bash tools/run.sh "${N}_f2" env OMP_NUM_THREADS=4 "$PY" "$R/tools/paper_f2.py" --subset "$O" --a1 "$O" --out "$O" \
      "${f2[@]}"
    log "$ARM $SET f2 exit $(cat "$L/${N}_f2.exit")"
    if ok "${N}_f2"; then
      rm -rf "$O/rollouts" "$R/data/paper/stores/e17_${ARM}_$SET"   # the protocol: futures deleted after their analyses
      log "$ARM $SET rollouts deleted"
    fi
  done
done
log "done"
