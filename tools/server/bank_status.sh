#!/usr/bin/env bash
# icutwin on the one-GPU host: one status line of a bank_single.sh run (shards written, error lines in the part logs
# apart from vLLM's chat-template warmup warning, GPU memory used, the driver's last line).
cd /root/autodl-tmp || exit 1
B=icutwin/data/bank/${NAME:-MUC32_main_bank}
shards=$(ls "$B"/part*/*.parquet 2>/dev/null | wc -l)
errs=$(cat logs/bank_[0-3].log 2>/dev/null | grep -E "Traceback|Error|out of memory|Killed|CUDA error" | grep -cvE "base\.py:3[0-9][0-9]")
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' ')
echo "shards=$shards errors=$errs gpu=$gpu | $(tail -n 1 logs/bank_single.log 2>/dev/null | cut -c1-200)"
