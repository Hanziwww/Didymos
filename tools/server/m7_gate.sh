#!/usr/bin/env bash
# icutwin on the AutoDL host: the M7 privacy gate of a trained candidate, the protocol in force (v1 with amendment
# v1.1, and amendment v2's P5, adopted 2026-10-05; docs/m7_privacy.md):
#   m7_gate.sh <run name>
# Freezes the run (unless frozen), then P1 and P2 on GPUs 0 and 1, the v1.1 audit samples (vLLM, one shard per GPU),
# P3-P5 with both controls, the v2 audit samples, P5 v2, the report (gate from P1-P4 and v2's P5, v1.1's P5 kept
# beside it) and the purge of every audit sample. Logs: logs/g_<run>_<step>.{log,exit}; logs/g_<run>_gate.exit holds
# the number of failed steps (the verdict is runs/audit/m7_<run>/report.json).
# GPUS (default "0 1 2 3"): the GPUs to use; with one (amendment 17, the RTX 5090 host) P1 and P2 run in turn and the
# four audit-sample shards one after another on it (the same shards and seeds). EVAL_ARGS: extra eval.cli overrides
# for every step (the 5090: tokens_per_batch=32768, as eval.yaml's 131,072 was sized for 96 GB cards).
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=${TOKENS:-$R/data/tokens/full_r3}   # round 3: TOKENS for the r4 tokens (clock blocks)
RUN=${1:?run name}
OUT=$R/runs/audit/m7_$RUN
HF=$R/runs/train/full/$RUN/final/hf
P=g_$RUN
read -r -a GPU <<< "${GPUS:-0 1 2 3}"
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
code() { cut -d' ' -f1 "$L/$1.exit" 2>/dev/null; }
if [ ! -e "$R/runs/train/full/$RUN/FROZEN.json" ]; then
  bash tools/run.sh "${P}_freeze" .venv/bin/python "$BASE/tools/freeze_run.py" "$RUN" "$T3" \
    "candidate model, frozen for the M7 privacy gate: no further training; no deliverable synthetic data before the gate"
  log "freeze exit $(cat "$L/${P}_freeze.exit")"
  [ "$(code "${P}_freeze")" = 0 ] || { echo 1 > "$L/${P}_gate.exit"; exit 1; }
fi
want=$(cd "$R" && .venv/bin/python -c "import json; print(json.load(open('runs/train/full/$RUN/FROZEN.json'))['checkpoint_files_sha256']['final/hf/model.safetensors'])")
have=$(sha256sum "$HF/model.safetensors" | cut -d' ' -f1)
[ "$want" = "$have" ] || { log "weights differ from FROZEN.json: $have != $want"; echo 1 > "$L/${P}_gate.exit"; exit 1; }
read -r adopted n2 seed2 < <(cd "$R" && .venv/bin/python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('src/icutwin/configs/eval.yaml').p5v2
print(str(bool(c.adopted)).lower(), int(c.samples), int(c.sample_seed))")
[ "$adopted" = true ] || { log "amendment v2 is not adopted: the protocol in force differs, nothing run"; exit 1; }
ev() {   # <name> <cli args...>
  local name=$1
  shift
  bash tools/run.sh "${P}_$name" .venv/bin/python -m icutwin.eval.cli mode=full "run=$RUN" "tokens_root=$T3" \
    "out_dir=$OUT" "$@" ${EVAL_ARGS:-} "hydra.run.dir=$R/runs/hydra/eval/${P}_$name"
  log "${P}_$name exit $(cat "$L/${P}_$name.exit")"
}
samples() {   # <step name> <out dir> <n> <seed>
  local k
  for k in 0 1 2 3; do
    bash tools/run.sh "${P}_$1$k" env PATH="$BASE/vllm-env/bin:$PATH" ICUTWIN_VOCAB="$T3/vocab.json" \
      CUDA_VISIBLE_DEVICES="${GPU[k % ${#GPU[@]}]}" "$BASE/vllm-env/bin/python" -m icutwin.generate.vllm_audit \
      --model "$HF" --vocab "$T3/vocab.json" --out "$2" --n "$3" --shards 4 --shard $k --seed "$4" \
      --gpu-memory-utilization "${GPU_MEM:-0.85}" &
    [ ${#GPU[@]} -ge 4 ] || wait                   # fewer GPUs: one shard at a time
  done
  wait
  for k in 0 1 2 3; do log "${P}_$1$k exit $(cat "$L/${P}_$1$k.exit")"; done
}
log "gate start ($RUN; protocol v1.1 with amendment v2)"
rm -rf "$OUT/audit_samples" "$OUT/audit_samples_p5v2"
if [ -s "$OUT/p1.json" ] && [ "${REUSE_P1:-0}" = 1 ]; then   # round 3: the screen's P1 (same frozen weights, same test)
  log "P1 reused: $(cd "$R" && .venv/bin/python -c "import json; print(json.load(open('$OUT/p1.json'))['status'])")"
  echo "0 0" > "$L/${P}_p1.exit"
  ev p2 step=p2 gpu=$(( ${#GPU[@]} >= 2 ? 1 : 0 ))
else
  if [ ${#GPU[@]} -ge 2 ]; then
    ev p1 step=p1 gpu=0 & p1=$!
    ev p2 step=p2 gpu=1 & p2=$!
    wait $p1 $p2
  else
    ev p1 step=p1 gpu=0
    ev p2 step=p2 gpu=0
  fi
fi
samples sample "$OUT/audit_samples" 5000 0
fail=0
for s in p1 p2 sample0 sample1 sample2 sample3; do [ "$(code "${P}_$s")" = 0 ] || fail=1; done
[ $fail = 0 ] && { ev synthetic step=synthetic gpu=0; [ "$(code "${P}_synthetic")" = 0 ] || fail=1; }
if [ $fail = 0 ]; then
  samples v2sample "$OUT/audit_samples_p5v2" "$n2" "$seed2"
  for k in 0 1 2 3; do [ "$(code "${P}_v2sample$k")" = 0 ] || fail=1; done
  [ $fail = 0 ] && { ev p5v2 step=p5v2 gpu=0; [ "$(code "${P}_p5v2")" = 0 ] || fail=1; }
fi
ev report step=report
ev purge step=purge
echo "$fail" > "$L/${P}_gate.exit"
log "done: steps failed=$fail; gate $(cd "$R" && .venv/bin/python -c "import json; print(json.load(open('$OUT/report.json'))['gate'])")"
