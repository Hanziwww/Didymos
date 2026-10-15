#!/usr/bin/env bash
# icutwin on the AutoDL host: the small-scale quality pilot of M32_main and its twin forecasting (plan S8), run only
# when the M7 gate's report is PASS for the frozen weights. Evaluation artefacts, not deliverables: the pilot's
# samples and the forecasting rollouts stay on this host, are never shared, and are deleted once scored; the outputs
# are aggregates (runs/eval/M32_main/{quality,forecast}.json and report.html). Logs: logs/pilot_<step>.{log,exit}.
#   pilot samples: 1,000 dataset-conditioned, grammar-constrained stays per dataset (vLLM, the engine)
#   forecasting: 300 held-out stays per dataset still in the ICU at 24 h, 32 rollouts each
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
[ "$gate" = PASS ] || { log "M7 gate is $gate, not PASS: no pilot"; exit 1; }
want=$(cd "$R" && .venv/bin/python -c "import json; print(json.load(open('runs/train/full/$RUN/FROZEN.json'))['checkpoint_files_sha256']['final/hf/model.safetensors'])")
have=$(sha256sum "$HF/model.safetensors" | cut -d' ' -f1)
[ "$want" = "$have" ] || { log "weights differ from FROZEN.json"; exit 1; }
log "gate PASS; weights match FROZEN.json"
mkdir -p "$E"
rm -rf "$E/pilot_samples" "$E/rollouts"
V="env PATH=$BASE/vllm-env/bin:$PATH ICUTWIN_VOCAB=$T3/vocab.json"
gen() {   # <gpu> <datasets...>
  local g=$1
  shift
  bash tools/run.sh "pilot_generate$g" $V CUDA_VISIBLE_DEVICES=$g "$BASE/vllm-env/bin/python" -m icutwin.eval.quality \
    generate --model "$HF" --vocab "$T3/vocab.json" --out "$E/pilot_samples" --per-dataset 1000 --datasets "$@"
}
gen 0 mimic aumc & gen 1 eicu hirid & gen 2 sicdb nwicu & gen 3 inspire &
wait
fail=0
for g in 0 1 2 3; do log "pilot_generate$g exit $(cat "$L/pilot_generate$g.exit")"; [ "$(code "pilot_generate$g")" = 0 ] || fail=1; done
roll() {  # <gpu> <datasets...>
  local g=$1
  shift
  bash tools/run.sh "pilot_rollouts$g" $V CUDA_VISIBLE_DEVICES=$g "$BASE/vllm-env/bin/python" -m icutwin.eval.forecast \
    rollouts --model "$HF" --vocab "$T3/vocab.json" --meds "$R/data/meds/full" --derived "$R/data/derived/full" \
    --out "$E/rollouts" --stays 300 --n 32 --cutoff 24 --datasets "$@"
}
roll 0 mimic aumc & roll 1 eicu hirid & roll 2 sicdb nwicu & roll 3 inspire &
bash tools/run.sh pilot_evaluate .venv/bin/python -m icutwin.eval.quality evaluate --tokens "$T3" \
  --samples "$E/pilot_samples" --out "$E/quality.json" &
wait
for s in pilot_rollouts0 pilot_rollouts1 pilot_rollouts2 pilot_rollouts3 pilot_evaluate; do
  log "$s exit $(cat "$L/$s.exit")"; [ "$(code "$s")" = 0 ] || fail=1
done
bash tools/run.sh pilot_score .venv/bin/python -m icutwin.eval.forecast score --tokens "$T3" --rollouts "$E/rollouts" \
  --out "$E/forecast.json"
log "pilot_score exit $(cat "$L/pilot_score.exit")"
bash tools/run.sh pilot_report .venv/bin/python -m icutwin.eval.report --title "icutwin M32_main evaluation" \
  --privacy "$AUDIT" --quality "$E/quality.json" --forecast "$E/forecast.json" --out "$E/report.html"
log "pilot_report exit $(cat "$L/pilot_report.exit")"
rm -rf "$E/pilot_samples" "$E/rollouts"
log "pilot samples and rollouts deleted"
echo "$fail" > "$L/pilot.exit"
log "done: steps failed=$fail"
