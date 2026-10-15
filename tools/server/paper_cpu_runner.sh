#!/usr/bin/env bash
# icutwin on the one-GPU host: Paper P's CPU analyses one at a time, after the landmark rollouts. The container's
# memory limit is 62 GiB: the lanes' and the queue's parallel analyses (six 12-GB decode workers each, beside three
# vLLM engines) pushed it to the limit, the engines' pages were evicted and re-read at 1.1 GB/s and the GPU idled.
# This runner:
#   - kills, every 20 s, any paper_a1 / paper_baselines job that another driver starts (paper_lanes' analyse,
#     paper_queue2's baselines), identified by a process group other than this runner's;
#   - once test_L24 and test_L48 have generated, runs the A1 analyses (test_R, test_L24, test_L48) with WORKERS decode
#     workers, then the baselines (R, L24, L48), then the B2 analysis once B2 has generated, under the log names the
#     queue waits for (logs/p1_<name>_a1.*, logs/pq2_baselines_<d>.*).
# Log: logs/paper_cpu_runner.log.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
P=$R/runs/paper
WORKERS=${WORKERS:-4}
cd "$BASE" || exit 1
ME=$(ps -o pgid= -p $$ | tr -d ' ')
log() { echo "$(date '+%F %T') $*" >> "$L/paper_cpu_runner.log"; }
reap() {   # kill the analysis jobs of other process groups
  local p g
  for p in $(pgrep -f 'tools/paper_(a1|baselines)\.py'); do
    g=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
    if [ -n "$g" ] && [ "$g" != "$ME" ]; then
      kill -KILL -"$g" 2>/dev/null && log "killed a foreign analysis group $g ($(ps -o args= -p "$p" 2>/dev/null | cut -c1-60))"
    fi
  done
}
( while kill -0 $$ 2>/dev/null; do reap; sleep 20; done ) &
waitlog() { until [ -f "$L/$1.exit" ]; do sleep 30; done; }
a1() {   # <name> <store>
  rm -f "$L/p1_$1_a1.exit"
  rm -rf "$R/data/paper/stores/a1_$1/_work"
  bash tools/run.sh "p1_$1_a1" env OMP_NUM_THREADS=1 "$PY" "$R/tools/paper_a1.py" --design-dir "$P/a1/$1" --store "$2" \
    --work "$R/data/paper/stores/a1_$1" --out "$P/a1/$1" --workers "$WORKERS" --chunk 100
  log "a1 $1: $(cat "$L/p1_$1_a1.exit")"
}
bl() {   # <design>
  rm -f "$L/pq2_baselines_$1.exit"
  bash tools/run.sh "pq2_baselines_$1" env OMP_NUM_THREADS=2 "$PY" "$R/tools/paper_baselines.py" \
    --design-dir "$P/a1/test_$1" --train "$S/real_train" --val "$S/real_dev" --test "$S/real_test" \
    --twins "$S/bank_dev" --out "$P/a1/test_$1"
  log "baselines $1: $(cat "$L/pq2_baselines_$1.exit")"
}
log "runner on (pgid $ME)"
waitlog p1_test_L24
waitlog p1_test_L48
log "landmark rollouts done"
a1 test_R "$S/real_test"
a1 test_L24 "$S/real_test"
a1 test_L48 "$S/real_test"
bl R
bl L24
bl L48
waitlog p1_b2_L24_unknown
a1 b2_L24_unknown "$S/real_test"
log "runner done"
