#!/usr/bin/env bash
# icutwin on the one-GPU host: Paper P amendment 7's CPU work, one job at a time (the container's 62 GiB: parallel
# analyses thrash). Each job waits for its inputs (exit files of logs/), and never runs beside a store-building
# analysis of another driver (tools/paper_a1.py). Logs: logs/fc_<job>.{log,exit}, logs/paper_followup.log.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
P=$R/runs/paper
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') cpu $*" >> "$L/paper_followup.log"; }
ok() { [ "$(cut -d' ' -f1 "$L/$1.exit" 2>/dev/null)" = 0 ]; }
wait_exit() { until [ -f "$L/$1.exit" ]; do sleep 60; done; }
quiet() { while pgrep -f "tools/paper_a1.py" > /dev/null; do sleep 60; done; }
job() {   # <name> <command...>
  local name=$1
  shift
  quiet
  bash tools/run.sh "fc_$name" env OMP_NUM_THREADS=2 "$@"
  log "$name exit $(cat "$L/fc_$name.exit")"
}
log start
job f5_design "$PY" "$R/tools/paper_c1_design.py" --tokens "$R/data/tokens/full_r4" --c1 "$P/c1" --out "$P/f5" \
  --test "$S/real_test" --train "$S/real_train" --val "$S/real_dev"
job f6_c3b "$PY" "$R/tools/paper_c3b.py" --twins "$S/bank_dev" --real "$S/real_test" --control "$S/real_control" \
  --out "$P/c3"
for d in L24 R; do
  job "f8_aug_$d" "$PY" "$R/tools/paper_aug.py" --design-dir "$P/a1/test_$d" --train "$S/real_train" \
    --val "$S/real_dev" --test "$S/real_test" --twins "$S/bank_dev" --out "$P/f8" --processes 4
done
for d in L24 R; do
  job "f9bc_$d" "$PY" "$R/tools/paper_transfer.py" --design "$d" --a1 "$P/a1/test_$d" --states "$P/states" \
    --out "$P/f9" --train "$S/real_train" --val "$S/real_dev" --test "$S/real_test"
done
# F1-F3: after each subset's hazards and its A1-style analysis (tools/server/paper_f1.sh)
wait_exit f1_hz_R128
wait_exit f1_a1_R128
job f2_R128 "$PY" "$R/tools/paper_f2.py" --subset "$P/f1/R128" --a1 "$P/a1/test_R" --out "$P/f1/R128"
wait_exit f1_hz_L24x
wait_exit f1_a1_L24x
job f2_L24x "$PY" "$R/tools/paper_f2.py" --subset "$P/f1/L24x" --a1 "$P/a1/test_L24" --out "$P/f1/L24x" --long
# then the jobs the GPU driver queues (logs/cpu_queue.txt: "<name>|<command>" per line; "END" ends)
Q=$L/cpu_queue.txt
touch "$Q"
n=0
while true; do
  line=$(sed -n "$((n + 1))p" "$Q")
  if [ -z "$line" ]; then sleep 60; continue; fi
  n=$((n + 1))
  [ "$line" = END ] && break
  job "${line%%|*}" bash -c "${line#*|}"
done
log done
