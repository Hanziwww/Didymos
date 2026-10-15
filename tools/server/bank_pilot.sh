#!/usr/bin/env bash
# icutwin on the AutoDL host: the pilot twin bank (study lead's decision 2026-10-06: MUC32_main with round 3's sampler,
# internal use): TOTAL twins (14,000) over the seven datasets in the flattened mix (alpha 0.5), drawn 3x and raked to
# the training margins, filtered (duplicates, novelty), one part per GPU (datasets grouped by expected tokens) into
# data/bank/$NAME/part<k>. Internal, DUA-derived data: it stays on this host. Logs: logs/bank_<k>.{log,exit}.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T4=$R/data/tokens/full_r4
NAME=${NAME:-MUC32_main_pilot}
TOTAL=${TOTAL:-14000}
OUT=$R/data/bank/$NAME
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
mkdir -p "$OUT"
cat > "$OUT/README.txt" <<EOF
INTERNAL synthetic ICU stays (twin bank) from MUC32_main, a model trained on DUA-covered data (MIMIC-IV, eICU,
HiRID, AmsterdamUMCdb, SICdb, NWICU, INSPIRE). DUA-derived: keep on the approved hosts; never share unless cleared.
Accepted for internal use with documented limits (docs/m7_refinement.md, "The study lead's decision").
EOF
part() {   # <gpu> <datasets...>
  local g=$1
  shift
  bash tools/run.sh "bank_$g" env PATH="$BASE/vllm-env/bin:$PATH" ICUTWIN_VOCAB="$T4/vocab.json" CUDA_VISIBLE_DEVICES=$g \
    "$BASE/vllm-env/bin/python" "$BASE/tools/bank_produce.py" --run MUC32_main --tokens "$T4" --out "$OUT/part$g" \
    --total "$TOTAL" --shard-size "${SHARD:-500}" --datasets "$@"
}
part 0 aumc &
sleep 30
part 1 hirid &
sleep 30
part 2 mimic nwicu &
sleep 30
part 3 eicu sicdb inspire &
wait
for g in 0 1 2 3; do log "bank_$g exit $(cat "$L/bank_$g.exit"): $(tail -n 1 "$L/bank_$g.log")"; done
log "done"
