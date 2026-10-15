#!/usr/bin/env bash
# icutwin on the AutoDL host: round 2's authenticity audit of a gated candidate (RUN, default M32_main;
# docs/m7_authenticity.md, pre-registered). Twins: 1,000 per dataset with pilot (b)'s sampler and seeds (vLLM, one
# GPU per dataset group; LOS_BANDS: twice as many drawn and resampled to the training stays' length bands); then, in
# parallel, the model-free audit, its explain measure and the pooled harness with utility (CPU), and the model
# checks (GPU 0); the targets A1-A7; the twins deleted (evaluation artefacts, aggregates kept in
# runs/eval/$RUN/$AUTH_DIR). Logs: logs/${LOG_PREFIX:-auth}_<step>; logs/<prefix>_model.exit marks the end of the
# GPU work. SEED: the twins' seed (replicates; the real sets stay the registered ones); CALIBRATION: round 3's
# calibration.json (twins resampled to the training stays' length and treatment margins, three times as many drawn);
# PER_DATASET (1000) and REQUIRE=p1 for round 3's screen of a candidate (its P1 instead of the whole gate);
# DRAW_GOALS=1: goal-first sampling (amendment 3: the model draws the goal block, which binds the stay; stripped).
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=${TOKENS:-$R/data/tokens/full_r3}   # round 3: TOKENS for the r4 tokens (clock blocks)
RUN=${RUN:-M32_main}
P=${LOG_PREFIX:-auth}
HF=$R/runs/train/full/$RUN/final/hf
AUDIT=$R/runs/audit/m7_$RUN
E=$R/runs/eval/$RUN/${AUTH_DIR:-auth}
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
code() { cut -d' ' -f1 "$L/$1.exit" 2>/dev/null; }
if [ "${REQUIRE:-gate}" = p1 ]; then    # round 3's screen of a candidate: its P1 (membership) result must pass
  gate=$(cd "$R" && .venv/bin/python -c "import json; print(json.load(open('$AUDIT/p1.json'))['status'])")
else
  gate=$(cd "$R" && .venv/bin/python -c "import json; print(json.load(open('$AUDIT/report.json'))['gate'])")
fi
[ "$gate" = PASS ] || { log "M7 ${REQUIRE:-gate} is $gate, not PASS"; exit 1; }
want=$(cd "$R" && .venv/bin/python -c "import json; print(json.load(open('runs/train/full/$RUN/FROZEN.json'))['checkpoint_files_sha256']['final/hf/model.safetensors'])")
have=$(sha256sum "$HF/model.safetensors" | cut -d' ' -f1)
[ "$want" = "$have" ] || { log "weights differ from FROZEN.json"; exit 1; }
[ -s "$T3/support.json" ] || { log "no support.json beside the vocabulary"; exit 1; }
log "$RUN: gate PASS; weights match FROZEN.json; support.json present"
mkdir -p "$E"
rm -rf "$E/twins"
V="env PATH=$BASE/vllm-env/bin:$PATH ICUTWIN_VOCAB=$T3/vocab.json"
gen() {   # <gpu> <datasets...>
  local g=$1
  shift
  bash tools/run.sh "${P}_generate$g" $V CUDA_VISIBLE_DEVICES=$g "$BASE/vllm-env/bin/python" -m icutwin.eval.quality \
    generate --model "$HF" --vocab "$T3/vocab.json" --out "$E/twins" --per-dataset "${PER_DATASET:-1000}" --seed "${SEED:-0}" \
    --gpu-memory-utilization "${GPU_MEM:-0.85}" ${LOS_BANDS:+--los-bands "$LOS_BANDS"} \
    ${CALIBRATION:+--calibration "$CALIBRATION" --oversample 3} ${DRAW_GOALS:+--draw-goals} --datasets "$@"
}
gen 0 aumc & gen 1 mimic hirid & gen 2 sicdb nwicu & gen 3 eicu inspire &
wait
fail=0
for g in 0 1 2 3; do log "${P}_generate$g exit $(cat "$L/${P}_generate$g.exit")"; [ "$(code "${P}_generate$g")" = 0 ] || fail=1; done
[ $fail = 0 ] || { rm -rf "$E/twins"; log "generation failed; twins deleted"; echo 1 > "$L/$P.exit"; exit 1; }
bash tools/run.sh "${P}_audit" .venv/bin/python -m icutwin.eval.authenticity audit --tokens "$T3" --twins "$E/twins" \
  --out "$E/audit.json" &
bash tools/run.sh "${P}_quality" .venv/bin/python -m icutwin.eval.quality evaluate --tokens "$T3" --samples "$E/twins" \
  --out "$E/quality.json" &
bash tools/run.sh "${P}_model" env CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m icutwin.eval.authenticity model \
  --tokens "$T3" --twins "$E/twins" --model "$HF" --out "$E/model.json" &
bash tools/run.sh "${P}_diagnose" .venv/bin/python "$BASE/tools/pilot_diagnose.py" "$T3" "$E/twins" &
bash tools/run.sh "${P}_explain" .venv/bin/python -m icutwin.eval.authenticity explain --tokens "$T3" --twins "$E/twins" \
  --out "$E/explain.json" &
wait
for s in audit quality model diagnose explain; do
  log "${P}_$s exit $(cat "$L/${P}_$s.exit")"; [ "$(code "${P}_$s")" = 0 ] || fail=1
done
bash tools/run.sh "${P}_targets" .venv/bin/python -m icutwin.eval.authenticity targets --audit "$E/audit.json" \
  --quality "$E/quality.json" --out "$E/targets.json"
log "${P}_targets exit $(cat "$L/${P}_targets.exit")"
rm -rf "$E/twins"
log "twins deleted"
echo "$fail" > "$L/$P.exit"
log "done: steps failed=$fail"
