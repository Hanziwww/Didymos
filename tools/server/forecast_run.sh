#!/usr/bin/env bash
# icutwin on the AutoDL host: twin forecasting of a gated candidate (the CoMET protocol, eval.forecast; round 3 adds
# the 6/24/72-hour horizons): STAYS held-out stays per dataset still in the ICU at 24 h (300), ROLLOUTS each (16),
# one GPU per dataset group; then the scores (aggregates, runs/eval/$RUN/$FC_OUT) and the rollouts deleted.
# Environment: RUN (M32_main), FC_OUT (forecast_h.json), LOG_PREFIX (fch), STAYS, ROLLOUTS.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=${TOKENS:-$R/data/tokens/full_r3}   # round 3: TOKENS for the r4 tokens (clock blocks)
RUN=${RUN:-M32_main}
P=${LOG_PREFIX:-fch}
HF=$R/runs/train/full/$RUN/final/hf
AUDIT=$R/runs/audit/m7_$RUN
E=$R/runs/eval/$RUN
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
code() { cut -d' ' -f1 "$L/$1.exit" 2>/dev/null; }
gate=$(cd "$R" && .venv/bin/python -c "import json; print(json.load(open('$AUDIT/report.json'))['gate'])")
[ "$gate" = PASS ] || { log "M7 gate is $gate, not PASS: no forecasting"; exit 1; }
want=$(cd "$R" && .venv/bin/python -c "import json; print(json.load(open('runs/train/full/$RUN/FROZEN.json'))['checkpoint_files_sha256']['final/hf/model.safetensors'])")
have=$(sha256sum "$HF/model.safetensors" | cut -d' ' -f1)
[ "$want" = "$have" ] || { log "weights differ from FROZEN.json"; exit 1; }
log "$RUN: gate PASS; weights match FROZEN.json"
D=$E/rollouts_$P
mkdir -p "$E"
rm -rf "$D"
V="env PATH=$BASE/vllm-env/bin:$PATH ICUTWIN_VOCAB=$T3/vocab.json"
roll() {  # <gpu> <datasets...>
  local g=$1
  shift
  bash tools/run.sh "${P}_rollouts$g" $V CUDA_VISIBLE_DEVICES=$g "$BASE/vllm-env/bin/python" -m icutwin.eval.forecast \
    rollouts --model "$HF" --vocab "$T3/vocab.json" --meds "$R/data/meds/full" --derived "$R/data/derived/full" \
    --out "$D" --stays "${STAYS:-300}" --n "${ROLLOUTS:-16}" --cutoff 24 --gpu-memory-utilization "${GPU_MEM:-0.85}" \
    --datasets "$@"
}
roll 0 aumc & roll 1 mimic hirid & roll 2 sicdb nwicu & roll 3 eicu inspire &
wait
fail=0
for g in 0 1 2 3; do log "${P}_rollouts$g exit $(cat "$L/${P}_rollouts$g.exit")"; [ "$(code "${P}_rollouts$g")" = 0 ] || fail=1; done
bash tools/run.sh "${P}_score" .venv/bin/python -m icutwin.eval.forecast score --tokens "$T3" --rollouts "$D" \
  --out "$E/${FC_OUT:-forecast_h.json}"
log "${P}_score exit $(cat "$L/${P}_score.exit")"
[ "$(code "${P}_score")" = 0 ] || fail=1
rm -rf "$D"
log "rollouts deleted"
echo "$fail" > "$L/$P.exit"
log "done: steps failed=$fail"
