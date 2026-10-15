#!/usr/bin/env bash
# icutwin on the AutoDL host: round 3 after amendment 2's candidates (docs/m7_refinement.md), as pre-registered:
#  1. waits for round4_muon.sh (the Muon candidates' screens);
#  2. amendment 3: goal-first sampling on M32_main, round 2's sampler with goal-first draws, twin seeds 0 and 1
#     (runs/eval/M32_main/auth_g0, auth_g1; two audits sharing the GPUs), and its rule (round3_select.py goalfirst);
#  3. the finalist rule over MC32_main, MU32_main and MUH32_main (round3_select.py finalist);
#  4. the decision run of the finalist (round3_decide.sh), its audits with goal-first draws if adopted.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=$R/data/tokens/full_r3
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
until grep -qE "^[0-9-]+ [0-9:]+ done$" "$L/round4_muon.log" 2>/dev/null; do sleep 30; done
log "amendment 2's screens done"
gf() {   # <seed>
  RUN=M32_main TOKENS=$T3 AUTH_DIR=auth_g$1 SEED=$1 LOG_PREFIX=gf$1 GPU_MEM=0.4 DRAW_GOALS=1 \
    LOS_BANDS=$T3/los_bands.json bash "$BASE/tools/authenticity_m32.sh" > "$L/gf$1_run.log" 2>&1
}
gf 0 &
sleep 60
gf 1 &
wait
log "goal-first audits: $(tail -n 1 "$L/gf0_run.log") | $(tail -n 1 "$L/gf1_run.log")"
"$R/.venv/bin/python" "$BASE/tools/round3_select.py" goalfirst > "$L/round3_goalfirst.txt" 2>&1
cat "$L/round3_goalfirst.txt"
GF=$(awk '/^GOALFIRST/ {print $2}' "$L/round3_goalfirst.txt")
"$R/.venv/bin/python" "$BASE/tools/round3_select.py" finalist > "$L/round3_finalist.txt" 2>&1
cat "$L/round3_finalist.txt"
FIN=$(awk '/^FINALIST/ {print $2}' "$L/round3_finalist.txt")
if [ -z "$FIN" ] || [ "$FIN" = none ]; then
  log "no finalist: the round is reported, no decision run"
  log "done"
  exit 0
fi
log "decision run: $FIN, goal-first draws: ${GF:-0}"
if [ "$GF" = 1 ]; then
  DRAW_GOALS=1 bash "$BASE/tools/round3_decide.sh" "$FIN" > "$L/round3_decide.log" 2>&1
else
  bash "$BASE/tools/round3_decide.sh" "$FIN" > "$L/round3_decide.log" 2>&1
fi
log "decision run finished: $(tail -n 1 "$L/round3_decide.log")"
log "done"
