#!/usr/bin/env bash
# icutwin on the host: the factory's measurement chain on the development bank and the rebuilt pool (plan Phase F,
# S3 gate -> S4 measures -> the alignment-diversity frontier -> the general-task screen), one step after another, each
# logged (logs/fchain_<step>.{log,exit}); a failed step stops the chain. Reports under runs/factory/ (aggregates).
#   STEPS="bench report frontier screen" bash factory_chain.sh
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
POOL=${POOL:-$R/data/factory/pools/bank_dev}
V=$R/data/tokens/full_r4/vocab.json
W=${WORKERS:-12}
STEPS=${STEPS:-"bench report frontier screen"}
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
step() {   # <name> <command...>
  local n=$1
  shift
  bash tools/run.sh "fchain_$n" "$@"
  local code
  code=$(cut -d' ' -f1 "$L/fchain_$n.exit")
  log "$n exit $code ($(cut -d' ' -f2 "$L/fchain_$n.exit") s)"
  [ "$code" = 0 ] || exit 1
}
for s in $STEPS; do
  case $s in
    bench) step bench "$PY" "$R/tools/factory_bench.py" --store "$S/bank_dev" --pool "$POOL" --vocab "$V" --n 10000 \
             --workers "$W" --out "$R/runs/factory/bench_s3.json" ;;
    report) step report "$PY" "$R/tools/factory_report.py" --store "$S/bank_dev" --pool "$POOL" --vocab "$V" \
              --real "$S/real_dev" --control "$S/real_control" --n "${REPORT_N:-20000}" --workers "$W" \
              --out "$R/runs/factory/report_s4.json" ;;
    frontier) step frontier "$PY" "$R/tools/factory_frontier.py" --pool "$POOL" --out "$R/runs/factory/frontier.json" ;;
    screen) step screen "$PY" "$R/tools/factory_screen.py" --real "$S/real_dev" --twins "$S/bank_dev" \
              --generic "$S/generic_dev" --vocab "$V" --n "${SCREEN_N:-600}" --workers "$W" \
              --out "$R/runs/factory/screen_s6.json" ;;
    final) step final "$PY" "$R/tools/factory_report.py" --store "$S/bank_dev" --pool "$POOL" --vocab "$V" \
             --real "$S/real_test" --control "$S/real_control" --n "${FINAL_N:-100000}" --workers "$W" \
             --out "$R/runs/factory/report_final.json" ;;
    *) log "unknown step $s"; exit 1 ;;
  esac
done
log "done"
