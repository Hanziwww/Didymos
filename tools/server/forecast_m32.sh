#!/usr/bin/env bash
# icutwin on the AutoDL host: twin forecasting of M32_main (the CoMET protocol, eval.forecast) with the corrected
# engine (support mask; the header comes from the real prefix): 300 held-out stays per dataset still in the ICU at
# 24 h, 32 rollouts each, one GPU per dataset group; then the scores (aggregates) and the rollouts deleted. Same
# gate check as the pilot. Logs: logs/fc_<step>.{log,exit}; output runs/eval/M32_main/forecast.json.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=$R/data/tokens/full_r3
RUN=M32_main
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
log "gate PASS; weights match FROZEN.json"
mkdir -p "$E"
rm -rf "$E/rollouts"
V="env PATH=$BASE/vllm-env/bin:$PATH ICUTWIN_VOCAB=$T3/vocab.json"
roll() {  # <gpu> <datasets...>
  local g=$1
  shift
  bash tools/run.sh "fc_rollouts$g" $V CUDA_VISIBLE_DEVICES=$g "$BASE/vllm-env/bin/python" -m icutwin.eval.forecast \
    rollouts --model "$HF" --vocab "$T3/vocab.json" --meds "$R/data/meds/full" --derived "$R/data/derived/full" \
    --out "$E/rollouts" --stays "${STAYS:-300}" --n "${ROLLOUTS:-32}" --cutoff 24 --datasets "$@"
}
roll 0 aumc & roll 1 mimic hirid & roll 2 sicdb nwicu & roll 3 eicu inspire &
wait
fail=0
for g in 0 1 2 3; do log "fc_rollouts$g exit $(cat "$L/fc_rollouts$g.exit")"; [ "$(code "fc_rollouts$g")" = 0 ] || fail=1; done
bash tools/run.sh fc_score .venv/bin/python -m icutwin.eval.forecast score --tokens "$T3" --rollouts "$E/rollouts" \
  --out "$E/forecast.json"
log "fc_score exit $(cat "$L/fc_score.exit")"
[ "$(code fc_score)" = 0 ] || fail=1
rm -rf "$E/rollouts"
log "rollouts deleted"
echo "$fail" > "$L/fc.exit"
log "done: steps failed=$fail"
