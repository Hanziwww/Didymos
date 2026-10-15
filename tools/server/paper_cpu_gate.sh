#!/usr/bin/env bash
# icutwin on the one-GPU host: keep Paper P's CPU analyses (paper_a1, paper_baselines) paused (SIGSTOP) while the
# listed rollout designs still generate: the container's 16-CPU quota throttles the vLLM engines with them (GPU at 0%,
# a 6-minute batch taking 70). When every design's exit file exists, resume all of them (SIGCONT). Nothing is lost:
# a stopped process continues where it stopped. Log: logs/paper_cpu_gate.log.
#   DESIGNS="p1_test_L24 p1_test_L48" bash paper_cpu_gate.sh
BASE=/root/autodl-tmp
L=$BASE/logs
DESIGNS=${DESIGNS:-"p1_test_L24 p1_test_L48"}
log() { echo "$(date '+%F %T') $*" >> "$L/paper_cpu_gate.log"; }
procs() {   # the analyses' processes and their workers
  local p
  for p in $(pgrep -f 'tools/paper_(a1|baselines)\.py'); do echo "$p"; pgrep -P "$p"; done
  pgrep -f 'run\.sh (p1_[a-z0-9_]+_a1|pq2_baselines_)'
}
done_all() { local d; for d in $DESIGNS; do [ -f "$L/$d.exit" ] || return 1; done; }
log "gate on until: $DESIGNS"
while ! done_all; do
  p=$(procs | sort -u | tr '\n' ' ')
  [ -n "$p" ] && kill -STOP $p 2>/dev/null
  sleep 30
done
p=$(procs | sort -u | tr '\n' ' ')
[ -n "$p" ] && kill -CONT $p 2>/dev/null
[ -f "$L/paused_pids.txt" ] && kill -CONT $(cat "$L/paused_pids.txt") 2>/dev/null
log "gate off: resumed $p"
