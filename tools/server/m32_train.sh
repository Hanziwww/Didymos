#!/usr/bin/env bash
# icutwin on the AutoDL host: (re)launch the M32_main training run after m32_run.sh's parity and smoke steps passed
# (the run there failed to start: tools/train.sh was resolved inside the repository by run.sh's cd).
BASE=/root/autodl-tmp
R=$BASE/icutwin
T3=$R/data/tokens/full_r3
Q3=$R/data/qa/full_r3
cd "$BASE" || exit 1
bash tools/run.sh train_M32_main bash "$BASE/tools/train.sh" M32_main "${1:?steps}" "tokens_root=$T3" "qa_root=$Q3"
echo "$(date '+%F %T') train exit $(cat "$BASE/logs/train_M32_main.exit")"
