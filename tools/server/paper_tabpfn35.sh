#!/usr/bin/env bash
# icutwin on the one-GPU host: Paper P's TabPFN-3.5 baseline (amendment 5; tools/paper_tabpfn.py) on the three
# held-out designs, from the study lead's local checkpoint (/root/autodl-tmp/tabpfn-3.5, checksums verified), offline,
# after the TabPFN v2 runs (logs/pt_R.exit; WAIT=0 skips). Logs: logs/pt35_<design>.{log,exit}.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
S=$R/data/factory/stores
CK=${CK:-$BASE/tabpfn-3.5/tabpfn-v3.5-20260909.safetensors}
CONTEXT=${CONTEXT:-50000}
cd "$BASE" || exit 1
if [ "${WAIT:-1}" = 1 ]; then until [ -f "$L/pt_R.exit" ]; do sleep 60; done; fi
for d in L24 L48 R; do
  bash tools/run.sh "pt35_$d" env PYTHONPATH="$BASE/tabpfn-pkgs" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 "$R/.venv/bin/python" \
    "$R/tools/paper_tabpfn.py" --design-dir "$R/runs/paper/a1/test_$d" --train "$S/real_train" --test "$S/real_test" \
    --out "$R/runs/paper/a1/test_$d" --model-path "$CK" --name tabpfn_v3_5 --all-features --context "$CONTEXT"
done
