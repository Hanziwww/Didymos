#!/usr/bin/env bash
# icutwin on the AutoDL host: round 3, amendment 3 as revised (docs/m7_refinement.md): goal-first sampling on M32_main,
# round 2's sampler (length resampling) with goal-first draws (bindings adopted per dataset, goal_bindings.json
# beside the vocabulary), twin seeds 0 and 1 on the registered real sets (runs/eval/M32_main/auth_g0, auth_g1; two
# audits sharing the GPUs), then amendment 3's rule (round3_select.py goalfirst -> logs/round3_goalfirst.txt).
# Waits for round5_frontier.sh (its screens).
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=$R/data/tokens/full_r3
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
until grep -qE "^[0-9-]+ [0-9:]+ done$" "$L/round5_frontier.log" 2>/dev/null; do sleep 30; done
log "amendment 4's screens done"
[ -s "$T3/goal_bindings.json" ] || { log "no goal_bindings.json beside the vocabulary"; exit 1; }
gf() {   # <seed>
  RUN=M32_main TOKENS=$T3 AUTH_DIR=auth_g$1 SEED=$1 LOG_PREFIX=gfb$1 GPU_MEM=0.4 DRAW_GOALS=1 \
    LOS_BANDS=$T3/los_bands.json bash "$BASE/tools/authenticity_m32.sh" > "$L/gfb$1_run.log" 2>&1
}
gf 0 &
sleep 60
gf 1 &
wait
log "goal-first audits: $(tail -n 1 "$L/gfb0_run.log") | $(tail -n 1 "$L/gfb1_run.log")"
"$R/.venv/bin/python" "$BASE/tools/round3_select.py" goalfirst > "$L/round3_goalfirst.txt" 2>&1
cat "$L/round3_goalfirst.txt"
log "done"
