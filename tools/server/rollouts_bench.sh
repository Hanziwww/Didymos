#!/usr/bin/env bash
# icutwin on the one-GPU host: the factory's calibration benchmark (plan Phase F, S0; factory.oracle): anchors from the
# twin bank, rollouts with the accepted model (MUC32_main) in vLLM's environment, one engine on GPU 0. Extra arguments
# go to rollouts_bench.py (e.g. a smoke: OUT=.../smoke rollouts_bench.sh --per-dataset 1 --n 4). Log:
# logs/rollouts_bench.{log,exit}; the script is copied to a new name so a running copy is never overwritten.
BASE=/root/autodl-tmp
R=$BASE/icutwin
T4=$R/data/tokens/full_r4
OUT=${OUT:-$R/runs/oracle/MUC32_main}
NAME=${NAME:-rollouts_bench}
cd "$BASE" || exit 1
cp "$R/tools/server/rollouts_bench.py" "$BASE/tools/$NAME.py"
bash tools/run.sh "$NAME" env PATH="$BASE/vllm-env/bin:$PATH" ICUTWIN_VOCAB="$T4/vocab.json" CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$BASE/vllm-env/bin/python" "$BASE/tools/$NAME.py" \
  --run MUC32_main --tokens "$T4" --bank "$R/data/bank/MUC32_main_bank" --out "$OUT" "$@"
