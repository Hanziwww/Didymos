#!/usr/bin/env bash
# icutwin on the AutoDL host: the quality pilot of M32_main again with the sampling corrections it motivated (the
# training-time header augmentation undone; the dataset support mask, support.json beside the vocabulary), into
# runs/eval/M32_main/pilot_b, for comparison with the first pilot. Same gate check, same sizes and seeds, no
# forecasting. Evaluation artefacts: the samples are deleted once evaluated and diagnosed (aggregates kept).
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=$R/data/tokens/full_r3
RUN=M32_main
HF=$R/runs/train/full/$RUN/final/hf
AUDIT=$R/runs/audit/m7_$RUN
E=$R/runs/eval/$RUN/pilot_b
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
code() { cut -d' ' -f1 "$L/$1.exit" 2>/dev/null; }
gate=$(cd "$R" && .venv/bin/python -c "import json; print(json.load(open('$AUDIT/report.json'))['gate'])")
[ "$gate" = PASS ] || { log "M7 gate is $gate, not PASS: no pilot"; exit 1; }
want=$(cd "$R" && .venv/bin/python -c "import json; print(json.load(open('runs/train/full/$RUN/FROZEN.json'))['checkpoint_files_sha256']['final/hf/model.safetensors'])")
have=$(sha256sum "$HF/model.safetensors" | cut -d' ' -f1)
[ "$want" = "$have" ] || { log "weights differ from FROZEN.json"; exit 1; }
[ -s "$T3/support.json" ] || { log "no support.json beside the vocabulary"; exit 1; }
log "gate PASS; weights match FROZEN.json; support.json present"
mkdir -p "$E"
rm -rf "$E/pilot_samples"
V="env PATH=$BASE/vllm-env/bin:$PATH ICUTWIN_VOCAB=$T3/vocab.json"
gen() {   # <gpu> <datasets...>
  local g=$1
  shift
  bash tools/run.sh "pilotb_generate$g" $V CUDA_VISIBLE_DEVICES=$g "$BASE/vllm-env/bin/python" -m icutwin.eval.quality \
    generate --model "$HF" --vocab "$T3/vocab.json" --out "$E/pilot_samples" --per-dataset 1000 --datasets "$@"
}
gen 0 aumc & gen 1 mimic hirid & gen 2 sicdb nwicu & gen 3 eicu inspire &
wait
fail=0
for g in 0 1 2 3; do log "pilotb_generate$g exit $(cat "$L/pilotb_generate$g.exit")"; [ "$(code "pilotb_generate$g")" = 0 ] || fail=1; done
bash tools/run.sh pilotb_evaluate .venv/bin/python -m icutwin.eval.quality evaluate --tokens "$T3" \
  --samples "$E/pilot_samples" --out "$E/quality.json" &
bash tools/run.sh pilotb_diagnose .venv/bin/python "$BASE/tools/pilot_diagnose.py" "$T3" "$E/pilot_samples" &
wait
for s in pilotb_evaluate pilotb_diagnose; do log "$s exit $(cat "$L/$s.exit")"; [ "$(code "$s")" = 0 ] || fail=1; done
bash tools/run.sh pilotb_report .venv/bin/python -m icutwin.eval.report --title "icutwin M32_main quality pilot (b)" \
  --privacy "$AUDIT" --quality "$E/quality.json" --out "$E/report.html"
rm -rf "$E/pilot_samples"
log "pilot samples deleted"
echo "$fail" > "$L/pilotb.exit"
log "done: steps failed=$fail"
