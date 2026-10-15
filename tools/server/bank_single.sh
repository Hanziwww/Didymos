#!/usr/bin/env bash
# icutwin on a one-GPU host (2026-10-07: RTX 4080 SUPER 32 GB, 16 CPUs, 62 GiB): the twin bank (study lead's decision
# 2026-10-06: MUC32_main with round 3's sampler, internal use) with four vLLM engines sharing GPU 0 (generation is
# CPU-bound: one core per engine's grammar mask). First the sparse novelty reference, built once (every training
# first window, 0.65 GB) into the bank's directory; then one part per engine, each loading it, the datasets grouped
# by expected work (the mini bank's rates: twins per second, AmsterdamUMCdb 1.2, HiRID 2, MIMIC and NWICU 2.4, eICU,
# SICdb and INSPIRE 3.6). An engine starts only when the previous one is ready (bank_produce.py's "engine_ready_s"
# line, or that part's end): vLLM sizes its KV cache from the device's memory at startup, which engines starting side
# by side would misread. TOTAL twins (14,000) in the flattened mix, drawn 3x and raked, filtered (duplicates,
# novelty). Internal, DUA-derived: stays on the approved hosts. Logs: logs/bank_<k>.{log,exit} (earlier ones are
# moved to logs/prev/ first).
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T4=$R/data/tokens/full_r4
NAME=${NAME:-MUC32_main_bank}
TOTAL=${TOTAL:-14000}
GM=${GPU_MEM:-0.18}
OUT=$R/data/bank/$NAME
REF=$OUT/novelty_ref.npz
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
mkdir -p "$OUT" "$L/prev"
for k in 0 1 2 3; do
  for f in "$L/bank_$k.log" "$L/bank_$k.exit"; do [ -e "$f" ] && mv -f "$f" "$L/prev/$(basename "$f").$(date +%s)"; done
done
cat > "$OUT/README.txt" <<EOF
INTERNAL synthetic ICU stays (twin bank) from MUC32_main, a model trained on DUA-covered data (MIMIC-IV, eICU,
HiRID, AmsterdamUMCdb, SICdb, NWICU, INSPIRE). DUA-derived: keep on the approved hosts; never share unless cleared.
Accepted for internal use with documented limits (docs/m7_refinement.md, "The study lead's decision").
EOF
part() {   # <k> <datasets...>
  local k=$1
  shift
  bash tools/run.sh "bank_$k" env PATH="$BASE/vllm-env/bin:$PATH" ICUTWIN_VOCAB="$T4/vocab.json" CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$BASE/vllm-env/bin/python" "$BASE/tools/bank_produce.py" \
    --run MUC32_main --tokens "$T4" --out "$OUT/part$k" --total "$TOTAL" --shard-size "${SHARD:-500}" \
    --gpu-memory-utilization "$GM" --novelty-ref "$REF" --datasets "$@"
}
ready() {  # <k>: until part k's engine is up or the part has ended (at most 30 minutes)
  local i
  for i in $(seq 1 120); do
    grep -q '"engine_ready_s"' "$L/bank_$1.log" 2>/dev/null && { log "bank_$1 $(grep '"engine_ready_s"' "$L/bank_$1.log")"; return 0; }
    [ -e "$L/bank_$1.exit" ] && { log "bank_$1 ended before its engine was up: $(tail -n 3 "$L/bank_$1.log")"; return 0; }
    sleep 15
  done
  log "bank_$1: no engine after 30 minutes; starting the next part anyway"
}
part 0 eicu inspire &              # builds and saves the reference when it does not exist; the others load it
until [ -s "$REF" ] || [ -e "$L/bank_0.exit" ]; do sleep 15; done
[ -s "$REF" ] || { log "novelty reference failed: $(tail -n 3 "$L/bank_0.log")"; exit 1; }
log "novelty reference ready"
ready 0
part 1 mimic &
ready 1
part 2 aumc sicdb &
ready 2
part 3 hirid nwicu &
ready 3
wait
for k in 0 1 2 3; do log "bank_$k exit $(cat "$L/bank_$k.exit"): $(grep -E '^\{"complete' "$L/bank_$k.log" | tail -n 1)"; done
log "done"
