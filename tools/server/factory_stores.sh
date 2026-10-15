#!/usr/bin/env bash
# icutwin on the one-GPU host: the factory's stay stores (plan Phase F, S1; tools/factory_store.py), CPU only, beside a
# GPU job: the twin bank (one segment per part, twins split 90/10 by a hash of their ids), the real dev (tuning split,
# at most 3,000 stays per dataset), test (held-out, 2,000 per dataset) and control (train, 2,000 per dataset) stores
# through the twins' decode, the real-raw dev store from MEDS (500 per dataset; the quantization report), and the
# training twins' channel statistics. Logs: logs/fstore_<name>.{log,exit}.
BASE=/root/autodl-tmp
R=$BASE/icutwin
T4=$R/data/tokens/full_r4
S=$R/data/factory/stores
W=${WORKERS:-8}
log() { echo "$(date '+%F %T') $*"; }
cd "$BASE" || exit 1
export ICUTWIN_DUCKDB_THREADS=1 ICUTWIN_DUCKDB_MEMORY_LIMIT=4GB
step() {   # <name> <args...>
  local name=$1
  shift
  bash tools/run.sh "fstore_$name" "$R/.venv/bin/python" "$R/tools/factory_store.py" "$@"
  log "fstore_$name exit $(cat "$BASE/logs/fstore_$name.exit")"
}
step bank bank --bank "$R/data/bank/MUC32_main_bank" --tokens "$T4" --out "$S/bank_dev" --val-pct 10 --workers "$W" --chunk 250
step stats stats --store "$S/bank_dev"
step dev tokens --tokens "$T4" --split tuning --label dev --per-dataset 3000 --out "$S/real_dev" --workers "$W" --chunk 250
step test tokens --tokens "$T4" --split held_out --label test --per-dataset 2000 --out "$S/real_test" --workers "$W" --chunk 250
step control tokens --tokens "$T4" --split train --label control --per-dataset 2000 --out "$S/real_control" --workers "$W" --chunk 250
step raw meds --meds "$R/data/meds/full" --derived "$R/data/derived/full" --tokens "$T4" --split tuning --label dev_raw --per-dataset 500 --out "$S/real_raw_dev"
log "done"
