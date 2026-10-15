#!/usr/bin/env bash
# icutwin on the AutoDL host: round 3, amendment 6 (docs/m7_refinement.md): MUC32_main re-audited twice (twin seeds 0
# and 1 on the registered real sets) with the twin-level raking plus the events-per-stay margin
# (calibration_events.json beside the r4 vocabulary), the two audits sharing the GPUs; then the bank criteria on the
# replicates' means (tools/server/target_means.py). Waits for calibration_events.json.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T4=$R/data/tokens/full_r4
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
until [ -e "$L/calib_events_r4.exit" ]; do sleep 20; done
[ "$(cut -d' ' -f1 "$L/calib_events_r4.exit")" = 0 ] && [ -s "$T4/calibration_events.json" ] \
  || { log "calibration_events.json failed: $(cat "$L/calib_events_r4.exit")"; exit 1; }
log "calibration_events.json ready"
audit() {   # <seed>
  RUN=MUC32_main TOKENS=$T4 AUTH_DIR=auth_e$1 SEED=$1 LOG_PREFIX=ev$1 GPU_MEM=0.4 \
    CALIBRATION=$T4/calibration_events.json bash "$BASE/tools/authenticity_m32.sh" > "$L/ev$1_run.log" 2>&1
}
audit 0 &
sleep 60
audit 1 &
wait
log "audits: $(tail -n 1 "$L/ev0_run.log") | $(tail -n 1 "$L/ev1_run.log")"
"$R/.venv/bin/python" "$BASE/tools/target_means.py" "$R/runs/eval/MUC32_main/auth_e0" "$R/runs/eval/MUC32_main/auth_e1" \
  > "$L/reaudit_events_means.txt" 2>&1
cat "$L/reaudit_events_means.txt"
log "done"
