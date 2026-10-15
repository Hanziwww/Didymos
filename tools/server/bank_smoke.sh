#!/usr/bin/env bash
# Smoke test of the bank path on the one-GPU host (2026-10-07): one tiny eICU part (gate interlock, raking, filters,
# manifest) with the vLLM engine at the bank's memory share; it builds the shared sparse novelty reference into the
# bank's directory (reused by bank_single.sh). GPU memory traced every 5 s. The smoke twins are deleted after.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T4=$R/data/tokens/full_r4
OUT=$R/data/bank/MUC32_main_bank
SMOKE=$R/data/bank/_smoke
mkdir -p "$OUT" "$L"
cd "$BASE" || exit 1
( while true; do echo "$(date '+%T') $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; sleep 5; done ) \
  > "$L/bank_smoke_gpu.log" 2>&1 &
TRACE=$!
rm -rf "$SMOKE"
bash tools/run.sh bank_smoke env PATH="$BASE/vllm-env/bin:$PATH" ICUTWIN_VOCAB="$T4/vocab.json" CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$BASE/vllm-env/bin/python" "$BASE/tools/bank_produce.py" \
  --run MUC32_main --tokens "$T4" --out "$SMOKE" --total 140 --shard-size 500 --gpu-memory-utilization 0.18 \
  --novelty-ref "$OUT/novelty_ref.npz" --datasets eicu
kill "$TRACE"
ls -la "$SMOKE" > "$L/bank_smoke_files.txt" 2>&1
cp "$SMOKE/manifest.json" "$L/bank_smoke_manifest.json" 2>/dev/null
rm -rf "$SMOKE"
echo "$(date '+%F %T') done" >> "$L/bank_smoke_files.txt"
