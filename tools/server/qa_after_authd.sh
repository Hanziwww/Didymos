#!/usr/bin/env bash
# icutwin on the AutoDL host: round 3's treatment QA of M32_main (eval.calibration --qa), queued behind the step-0c
# audit's GPU work (logs/authd_model.exit), on GPU 3.
B=/root/autodl-tmp
R=$B/icutwin
cd "$B" || exit 1
until [ -e logs/authd_model.exit ]; do sleep 30; done
bash tools/run.sh r3_qa env CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m icutwin.eval.calibration --qa --n 1000 \
  --tokens "$R/data/tokens/full_r3" --model "$R/runs/train/full/M32_main/final/hf" \
  --out "$R/runs/eval/M32_main/calibration_qa.json"
