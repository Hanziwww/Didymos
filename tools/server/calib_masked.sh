#!/usr/bin/env bash
# icutwin on the AutoDL host: round 3, step 0a under the generation mask (grammar and support, as the sampler
# draws): one process per dataset (CPU-bound masks; two per GPU beside the running vLLM engines, about 4 GB each),
# 1,000 held-out stays each, a quarter of the positions at risk; merged into runs/eval/$RUN/calibration_masked.json (RUN, default M32_main; TOKENS).
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=${TOKENS:-$R/data/tokens/full_r3}   # round 3: TOKENS for the r4 tokens (clock blocks)
RUN=${RUN:-M32_main}
HF=$R/runs/train/full/$RUN/final/hf
E=$R/runs/eval/$RUN
cd "$BASE" || exit 1
mkdir -p "$E"
one() {   # <gpu> <dataset>
  bash tools/run.sh "calm_${RUN}_$2" env CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -m icutwin.eval.calibration --tokens "$T3" --model "$HF" --out "$E/calibration_masked_$2.json" \
    --datasets "$2" --n 1000 --masked --share 0.25
}
read -r -a G <<< "${GPUS:-1 2 3}"      # the GPUs to spread the datasets over (round 3: a candidate's own two)
k=0
for ds in aumc mimic hirid sicdb nwicu eicu inspire; do
  one "${G[$((k % ${#G[@]}))]}" "$ds" &
  k=$((k + 1))
done
wait
cd "$R" && .venv/bin/python - <<EOF
import json
from pathlib import Path
E = Path("$E")
out = {}
for p in sorted(E.glob("calibration_masked_*.json")):
    out.update(json.loads(p.read_text()))
(E / "calibration_masked.json").write_text(json.dumps(out, indent=1))
print("merged", sorted(out))
EOF
echo "done"
