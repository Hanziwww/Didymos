#!/usr/bin/env bash
# icutwin on the one-GPU host: Paper P's rollouts on real anchors (docs/p1_paper_protocol.md A1, B2), one design per
# call, in vLLM's environment with one engine on GPU 0. Arguments go to paper_rollouts.py; NAME names the log
# (logs/$NAME.{log,exit}) and the copy of the script that runs (a running copy is never overwritten).
#   NAME=p1_dev_L24 bash paper_rollouts.sh --split tuning --store $R/data/factory/stores/real_dev --design L24 \
#     --out $R/runs/paper/a1/tuning/L24 --per-dataset 50 --n 32
BASE=/root/autodl-tmp
R=$BASE/icutwin
T4=$R/data/tokens/full_r4
NAME=${NAME:-paper_rollouts}
cd "$BASE" || exit 1
cp "$R/tools/server/paper_rollouts.py" "$BASE/tools/$NAME.py"
bash tools/run.sh "$NAME" env PATH="$BASE/vllm-env/bin:$PATH" ICUTWIN_VOCAB="$T4/vocab.json" CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$BASE/vllm-env/bin/python" "$BASE/tools/$NAME.py" \
  --run MUC32_main --tokens "$T4" "$@"
