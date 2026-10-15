#!/usr/bin/env bash
# icutwin on the one-GPU host: run one of tools/server's vLLM scripts (Paper P amendment 7: paper_prompt_rollouts.py,
# paper_whatif2.py) in vLLM's environment on GPU 0. SCRIPT names the script, NAME the log (logs/$NAME.{log,exit}) and
# the copy that runs (a running copy is never overwritten); the arguments go to the script after --run and --tokens.
#   NAME=f4_s0 SCRIPT=paper_prompt_rollouts.py bash paper_vllm.sh --split held_out --store ... --prompts ... --out ...
BASE=/root/autodl-tmp
R=$BASE/icutwin
T4=$R/data/tokens/full_r4
NAME=${NAME:?}
SCRIPT=${SCRIPT:?}
cd "$BASE" || exit 1
cp "$R/tools/server/$SCRIPT" "$BASE/tools/$NAME.py"
bash tools/run.sh "$NAME" env PATH="$BASE/vllm-env/bin:$PATH" ICUTWIN_VOCAB="$T4/vocab.json" CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$R/tools/server" "$BASE/vllm-env/bin/python" \
  "$BASE/tools/$NAME.py" --run MUC32_main --tokens "$T4" "$@"
