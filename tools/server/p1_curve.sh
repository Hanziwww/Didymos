#!/usr/bin/env bash
# icutwin on the AutoDL host: membership against fit along training (round 3, after amendment 2): MIMIC's membership
# AUC (p1_families.py, 800 members and 800 non-members, the same draw for every checkpoint) at checkpoints of the
# AdamW reference and of the Muon run, to compare the two at equal tuning NLL. One GPU (GPU, default 3).
BASE=/root/autodl-tmp
R=$BASE/icutwin
cd "$BASE" || exit 1
one() {   # <run> <checkpoint> <tokens>
  bash tools/run.sh "p1c_$1_$2" env CUDA_VISIBLE_DEVICES=${GPU:-3} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python "$BASE/tools/p1_families.py" "$1" "$3" "$R/runs/eval/$1/p1_curve_$2.json" --datasets mimic \
    --n 800 --ckpt "$2"
}
T3=$R/data/tokens/full_r3
T4=$R/data/tokens/full_r4
for ck in step_1500 step_2000 final; do one M32_main "$ck" "$T3"; done
for ck in step_1000 step_1500 step_2000; do one MU32_main "$ck" "$T4"; done
echo done
