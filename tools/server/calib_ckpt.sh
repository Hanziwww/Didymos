#!/usr/bin/env bash
# icutwin on the AutoDL host: round 3's quick check of the onset over-prediction along training: the masked
# teacher-forced calibration (held-out first windows, 1,000 per dataset, a quarter of the positions at risk) of one
# checkpoint of a run, all seven datasets on one GPU:
#   calib_ckpt.sh <run> <checkpoint dir: final | step_1000 ...> <gpu> <tag> [tokens root]
# Results: runs/eval/<run>/calibration_masked_<tag>_<dataset>.json; logs calck_<run>_<tag>_<dataset>.
BASE=/root/autodl-tmp
R=$BASE/icutwin
RUN=${1:?run}
CK=${2:?checkpoint}
GPU=${3:?gpu}
TAG=${4:?tag}
T=${5:-$R/data/tokens/full_r3}
HF=$R/runs/train/full/$RUN/$CK/hf
E=$R/runs/eval/$RUN
cd "$BASE" || exit 1
mkdir -p "$E"
for ds in aumc mimic hirid sicdb nwicu eicu inspire; do
  bash tools/run.sh "calck_${RUN}_${TAG}_$ds" env CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -m icutwin.eval.calibration --tokens "$T" --model "$HF" --out "$E/calibration_masked_${TAG}_$ds.json" \
    --datasets "$ds" --n 1000 --masked --share 0.25 &
done
wait
echo "done"
