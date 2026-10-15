#!/usr/bin/env bash
# icutwin on the AutoDL host: round 3's quick check of the onset over-prediction: the masked teacher-forced
# calibration of RUN (default M32_main) on the training stays (members, GPU 0) and the tuning stays (GPU 1), 1,000
# first windows per dataset, a quarter of the positions at risk (as calib_masked.sh on the held-out stays). Results:
# runs/eval/$RUN/calibration_masked_<split>_<dataset>.json; logs calsplit_<run>_<split>_<dataset>.
BASE=/root/autodl-tmp
R=$BASE/icutwin
T3=${TOKENS:-$R/data/tokens/full_r3}
RUN=${RUN:-M32_main}
HF=$R/runs/train/full/$RUN/final/hf
E=$R/runs/eval/$RUN
cd "$BASE" || exit 1
one() {   # <gpu> <split> <dataset>
  bash tools/run.sh "calsplit_${RUN}_$2_$3" env CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="$BASE/tools" .venv/bin/python "$BASE/tools/calib_split.py" "$2" --tokens "$T3" --model "$HF" \
    --out "$E/calibration_masked_$2_$3.json" --datasets "$3" --n 1000 --masked --share 0.25
}
for ds in aumc mimic hirid sicdb nwicu eicu inspire; do
  one 0 train "$ds" &
  one 1 tuning "$ds" &
done
wait
echo "done"
