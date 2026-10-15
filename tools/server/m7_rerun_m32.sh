#!/usr/bin/env bash
# icutwin on the AutoDL host: M32_main's P3-P5 re-run under protocol v1.1 after #105 (the first run's audit samples
# reused seeds across shards: 1,550 distinct seeds for 5,000 samples, so its P3-P5 are void). P1 and P2 drew no
# samples and stand. Checks the frozen weights' sha256, keeps the void results for the record, draws the registered
# 5,000 audit samples with distinct seeds (vLLM, one shard per GPU), runs P3-P5 with both controls and the report,
# breaks the samples' grammar validity down (aggregates), and deletes the samples. Logs: logs/m7c_<step>.{log,exit}.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=$R/data/tokens/full_r3
RUN=M32_main
OUT=$R/runs/audit/m7_$RUN
HF=$R/runs/train/full/$RUN/final/hf
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
code() { cut -d' ' -f1 "$L/$1.exit" 2>/dev/null; }
want=$(cd "$R" && .venv/bin/python -c "import json; print(json.load(open('runs/train/full/$RUN/FROZEN.json'))['checkpoint_files_sha256']['final/hf/model.safetensors'])")
have=$(sha256sum "$HF/model.safetensors" | cut -d' ' -f1)
[ "$want" = "$have" ] || { log "weights differ from FROZEN.json: $have != $want"; exit 1; }
log "weights match FROZEN.json ($have)"
[ -e "$OUT/synthetic.json" ] && mv "$OUT/synthetic.json" "$OUT/synthetic_run1_void_105.json"
[ -e "$OUT/report.json" ] && mv "$OUT/report.json" "$OUT/report_run1_void_105.json"
rm -rf "$OUT/audit_samples"
export ICUTWIN_VOCAB=$T3/vocab.json
for k in 0 1 2 3; do
  bash tools/run.sh "m7c_sample$k" env PATH="$BASE/vllm-env/bin:$PATH" CUDA_VISIBLE_DEVICES=$k \
    "$BASE/vllm-env/bin/python" -m icutwin.generate.vllm_audit --model "$HF" --vocab "$T3/vocab.json" \
    --out "$OUT/audit_samples" --n 5000 --shards 4 --shard $k --seed 0 &
done
wait
fail=0
for k in 0 1 2 3; do log "m7c_sample$k exit $(cat "$L/m7c_sample$k.exit")"; [ "$(code "m7c_sample$k")" = 0 ] || fail=1; done
ev() {   # <name> <cli args...>
  local name=$1
  shift
  bash tools/run.sh "m7c_$name" .venv/bin/python -m icutwin.eval.cli mode=full "run=$RUN" "tokens_root=$T3" \
    "out_dir=$OUT" "$@" "hydra.run.dir=$R/runs/hydra/eval/m7c_$name"
  log "m7c_$name exit $(cat "$L/m7c_$name.exit")"
}
if [ $fail = 0 ]; then
  ev synthetic step=synthetic gpu=0
  [ "$(code m7c_synthetic)" = 0 ] || fail=1
fi
ev report step=report
bash tools/run.sh m7c_validity bash -c "cd $R && .venv/bin/python $BASE/tools/audit_validity.py $RUN $T3"
log "validity exit $(cat "$L/m7c_validity.exit")"
ev purge step=purge
echo "$fail" > "$L/m7c_gate.exit"
log "done: steps failed=$fail"
