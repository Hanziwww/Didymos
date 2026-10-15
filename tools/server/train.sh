#!/usr/bin/env bash
# icutwin on the AutoDL host: the formal training run (run as a step through tools/run.sh), e.g.
#   train.sh <run name> <max steps> [extra hydra overrides...]
# DDP over the 4 GPUs (configs/accelerate/ddp.yaml); the run directory is runs/train/full/<run name>.
set -e
R=/root/autodl-tmp/icutwin
name=$1
steps=$2
shift 2
export ICUTWIN_TMP=$R/.tmp/train_$name
.venv/bin/accelerate launch --config_file src/icutwin/configs/accelerate/ddp.yaml -m icutwin.model.cli task=train \
  mode=full "train.name=$name" "train.max_steps=$steps" "hydra.run.dir=$R/runs/hydra/train/$name" "$@"
