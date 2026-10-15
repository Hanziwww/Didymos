#!/usr/bin/env bash
# icutwin on the one-GPU host: Paper P's rollout runs (docs/p1_paper_protocol.md A1, B2) one after another on GPU 0;
# each design's analysis (tools/paper_a1.py, CPU) starts in the background as soon as its rollouts are written, so the
# GPU never waits. A failed rollout step stops the chain. Logs: logs/p1_<split>_<design>{,_a1}.{log,exit}.
#   STEPS="dev test b2" bash paper_chain.sh
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
STEPS=${STEPS:-"dev test b2"}
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
roll() {   # <name> <split> <store> <design> <per-dataset> [extra args...]
  local name=$1 split=$2 store=$3 design=$4 per=$5
  shift 5
  local out=$R/runs/paper/a1/$name
  NAME=p1_$name bash "$R/tools/server/paper_rollouts.sh" --split "$split" --store "$store" --design "$design" \
    --out "$out" --per-dataset "$per" --n 32 "$@"
  local code
  code=$(cut -d' ' -f1 "$L/p1_$name.exit")
  log "rollouts $name exit $code ($(cut -d' ' -f2 "$L/p1_$name.exit") s)"
  [ "$code" = 0 ] || exit 1
  setsid -f bash tools/run.sh "p1_${name}_a1" env OMP_NUM_THREADS=1 "$PY" "$R/tools/paper_a1.py" \
    --design-dir "$out" --store "$store" --work "$R/data/paper/stores/a1_$name" --out "$out" --workers 8 \
    > /dev/null 2>&1 < /dev/null
  log "analysis $name started"
}
for s in $STEPS; do
  case $s in
    dev) roll dev_L24 tuning "$S/real_dev" L24 100
         roll dev_L48 tuning "$S/real_dev" L48 100
         roll dev_R tuning "$S/real_dev" R 100 ;;
    test) roll test_R held_out "$S/real_test" R 500
          roll test_L24 held_out "$S/real_test" L24 500
          roll test_L48 held_out "$S/real_test" L48 500 ;;
    b2) roll b2_L24_unknown held_out "$S/real_test" L24 300 --unknown-site ;;   # its anchors are test_L24's first
                                                       # 300 per dataset (same seed): the known-site arm is read there
    *) log "unknown step $s"; exit 1 ;;
  esac
done
log "done"
