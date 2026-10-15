#!/usr/bin/env bash
# icutwin on the AutoDL host: one dataset's CLIF chain (run as a step through tools/run.sh), e.g.
#   clif_chain.sh <dataset> <duckdb threads> <duckdb memory> <task>...
# The first task builds clif_root (supplement or adapter); logs/clif_<dataset>.built marks it done, so steps that
# read clif_root only (the reconciliation) can start while harmonize, validate and CLIF_MEDS run.
set -e
R=/root/autodl-tmp/icutwin
L=/root/autodl-tmp/logs
d=$1
threads=$2
mem=$3
shift 3
export ICUTWIN_TMP=$R/.tmp/clif_$d ICUTWIN_DUCKDB_THREADS=$threads ICUTWIN_DUCKDB_MEMORY_LIMIT=$mem POLARS_MAX_THREADS=$threads
rm -f "$L/clif_$d.built"
first=1
for task in "$@"; do
  echo "== $task $(date '+%T')"
  .venv/bin/python -m icutwin.clif.cli "task=$task" mode=full "datasets=[$d]" "hydra.run.dir=$R/runs/hydra/clif/${d}_$task"
  if [ $first = 1 ]; then touch "$L/clif_$d.built"; first=0; fi
done
