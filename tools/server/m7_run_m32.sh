#!/usr/bin/env bash
# icutwin on the AutoDL host: the M7 privacy gate of M32_main (protocol v1 with amendment v1.1: sampling cap = the
# context - 1, audit samples with vLLM; docs/m7_privacy.md). Waits for the training run, freezes it, then P1 and P2
# on GPUs 0 and 1, the audit samples (vLLM, one shard per GPU), P3-P5 with both controls, and the report. Writes
# logs/m7b_<step>.{log,exit} and logs/m7b_gate.exit.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=$R/data/tokens/full_r3
RUN=M32_main
OUT=$R/runs/audit/m7_$RUN
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
code() { cut -d' ' -f1 "$L/$1.exit" 2>/dev/null; }
until [ -e "$L/train_$RUN.exit" ]; do sleep 60; done
[ "$(code "train_$RUN")" = 0 ] || { log "training failed: $(cat "$L/train_$RUN.exit")"; exit 1; }
bash tools/run.sh m7b_freeze .venv/bin/python "$BASE/tools/freeze_run.py" "$RUN" "$T3" \
  "candidate model, frozen for the M7 privacy gate: no further training; no deliverable synthetic data before the gate"
log "freeze exit $(cat "$L/m7b_freeze.exit")"
[ "$(code m7b_freeze)" = 0 ] || exit 1
ev() {   # <name> <cli args...>
  local name=$1
  shift
  bash tools/run.sh "m7b_$name" .venv/bin/python -m icutwin.eval.cli mode=full "run=$RUN" "tokens_root=$T3" \
    "out_dir=$OUT" "$@" "hydra.run.dir=$R/runs/hydra/eval/m7b_$name"
  log "m7b_$name exit $(cat "$L/m7b_$name.exit")"
}
log "gate start ($RUN, protocol v1.1)"
ev p1 step=p1 gpu=0 & p1=$!
ev p2 step=p2 gpu=1 & p2=$!
wait $p1 $p2
export ICUTWIN_VOCAB=$T3/vocab.json
for k in 0 1 2 3; do
  bash tools/run.sh "m7b_sample$k" env PATH="$BASE/vllm-env/bin:$PATH" CUDA_VISIBLE_DEVICES=$k \
    "$BASE/vllm-env/bin/python" -m icutwin.generate.vllm_audit --model "$R/runs/train/full/$RUN/final/hf" \
    --vocab "$T3/vocab.json" --out "$OUT/audit_samples" --n 5000 --shards 4 --shard $k --seed 0 &
done
wait
for k in 0 1 2 3; do log "m7b_sample$k exit $(cat "$L/m7b_sample$k.exit")"; done
fail=0
for s in p1 p2 sample0 sample1 sample2 sample3; do [ "$(code "m7b_$s")" = 0 ] || fail=1; done
[ $fail = 0 ] && { ev synthetic step=synthetic gpu=0; [ "$(code m7b_synthetic)" = 0 ] || fail=1; }
ev report step=report
echo "$fail" > "$L/m7b_gate.exit"
log "done: steps failed=$fail"
