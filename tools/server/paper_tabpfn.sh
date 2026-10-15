#!/usr/bin/env bash
# icutwin on the one-GPU host: Paper P's TabPFN v2 baseline (amendment 4; tools/paper_tabpfn.py) on the three held-out
# designs, after the rollout lanes (WAIT=0 skips the wait). TabPFN runs locally from tabpfn-pkgs (no data leaves the
# host); the licence token sits in /root/.cache/tabpfn/auth_token. Logs: logs/pt_<design>.{log,exit}.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
S=$R/data/factory/stores
cd "$BASE" || exit 1
if [ "${WAIT:-1}" = 1 ]; then until grep -q "lanes done" "$L/paper_lanes.log" 2>/dev/null; do sleep 60; done; fi
for d in L24 L48 R; do
  bash tools/run.sh "pt_$d" env PYTHONPATH="$BASE/tabpfn-pkgs" TABPFN_MODEL_VERSION=v2 \
    HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 "$R/.venv/bin/python" \
    "$R/tools/paper_tabpfn.py" --design-dir "$R/runs/paper/a1/test_$d" --train "$S/real_train" --test "$S/real_test" \
    --out "$R/runs/paper/a1/test_$d"
done
