#!/usr/bin/env bash
# icutwin on the AutoDL host: P5 of the proposed amendment v2 for M32_main (docs/m7_privacy.md). Runs only once the
# amendment is adopted: `p5v2.adopted: true` committed in configs/eval.yaml (no command-line override here, so the
# adoption is a recorded change made before the run). Checks the frozen weights, draws the v2 audit samples (vLLM,
# seed p5v2.sample_seed, one shard per GPU) into audit_samples_p5v2, runs step=p5v2 and the report (P5 from v2, v1.1's
# kept beside it), and deletes the samples. Logs: logs/m7d_<step>.{log,exit}.
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
read -r adopted n seed < <(cd "$R" && .venv/bin/python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('src/icutwin/configs/eval.yaml').p5v2
print(str(bool(c.adopted)).lower(), int(c.samples), int(c.sample_seed))")
[ "$adopted" = true ] || { log "amendment v2 is not adopted (configs/eval.yaml p5v2.adopted): nothing run"; exit 1; }
want=$(cd "$R" && .venv/bin/python -c "import json; print(json.load(open('runs/train/full/$RUN/FROZEN.json'))['checkpoint_files_sha256']['final/hf/model.safetensors'])")
have=$(sha256sum "$HF/model.safetensors" | cut -d' ' -f1)
[ "$want" = "$have" ] || { log "weights differ from FROZEN.json: $have != $want"; exit 1; }
log "amendment v2 adopted; weights match FROZEN.json; $n samples, seed $seed"
rm -rf "$OUT/audit_samples_p5v2"
export ICUTWIN_VOCAB=$T3/vocab.json
for k in 0 1 2 3; do
  bash tools/run.sh "m7d_sample$k" env PATH="$BASE/vllm-env/bin:$PATH" CUDA_VISIBLE_DEVICES=$k \
    "$BASE/vllm-env/bin/python" -m icutwin.generate.vllm_audit --model "$HF" --vocab "$T3/vocab.json" \
    --out "$OUT/audit_samples_p5v2" --n "$n" --shards 4 --shard $k --seed "$seed" &
done
wait
fail=0
for k in 0 1 2 3; do log "m7d_sample$k exit $(cat "$L/m7d_sample$k.exit")"; [ "$(code "m7d_sample$k")" = 0 ] || fail=1; done
ev() {   # <name> <cli args...>
  local name=$1
  shift
  bash tools/run.sh "m7d_$name" .venv/bin/python -m icutwin.eval.cli mode=full "run=$RUN" "tokens_root=$T3" \
    "out_dir=$OUT" "$@" "hydra.run.dir=$R/runs/hydra/eval/m7d_$name"
  log "m7d_$name exit $(cat "$L/m7d_$name.exit")"
}
if [ $fail = 0 ]; then
  ev p5v2 step=p5v2 gpu=0
  [ "$(code m7d_p5v2)" = 0 ] || fail=1
fi
ev report step=report
ev purge step=purge
echo "$fail" > "$L/m7d_gate.exit"
log "done: steps failed=$fail"
