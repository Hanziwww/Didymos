#!/usr/bin/env bash
# icutwin on the AutoDL host: round 3's decision run (docs/m7_refinement.md, "Targets and decision", amendment 1):
#   round3_decide.sh <finalist>
# Phases of concurrent jobs sharing the GPUs (vLLM's generation is CPU-bound: GPU_MEM shares per engine, jobs
# started a minute apart so the engines do not profile memory at the same time):
#  1. the finalist's M7 gate (P1 reused from the screen) and its S1 audit (1,000 twins, round-2 sampler);
#  2. the finalist's two decision audits (round-3 sampler: twins raked to length and treatment margins; seeds 0, 1);
#  3. the reference's two decision audits (M32_main, the same sampler, seeds 0, 1) and the finalist's rollouts (S2).
# Each phase waits for the previous one's GPU work (the audits' model checks, the gate's samples).
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=$R/data/tokens/full_r3
T4=$R/data/tokens/full_r4
RUN=${1:?finalist run name}
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
waitfor() { until [ -e "$L/$1.exit" ]; do sleep 30; done; }
audit() {   # <prefix> <run> <tokens> <auth dir> <seed> [CALIBRATION file]
  RUN=$2 TOKENS=$3 AUTH_DIR=$4 SEED=$5 LOG_PREFIX=$1 REQUIRE=p1 GPU_MEM=$GM CALIBRATION=$6 \
    LOS_BANDS=$( [ -z "$6" ] && echo "$3/los_bands.json" ) bash "$BASE/tools/authenticity_m32.sh" > "$L/$1_run.log" 2>&1
}
log "decision run: $RUN"
GM=0.4
TOKENS=$T4 REUSE_P1=1 GPU_MEM=$GM bash "$BASE/tools/m7_gate.sh" "$RUN" > "$L/g_${RUN}_run.log" 2>&1 &
sleep 60
audit "s1_$RUN" "$RUN" "$T4" auth_s1 0 "" &
until [ -e "$L/g_${RUN}_v2sample3.exit" ] || [ -e "$L/g_${RUN}_gate.exit" ]; do sleep 30; done   # samples done or a failed gate
waitfor "s1_${RUN}_model"
log "phase 1 GPU work done"
audit "r0_$RUN" "$RUN" "$T4" auth_r0 0 "$T4/calibration.json" &
sleep 60
audit "r1_$RUN" "$RUN" "$T4" auth_r1 1 "$T4/calibration.json" &
waitfor "r0_${RUN}_model"
waitfor "r1_${RUN}_model"
log "phase 2 GPU work done"
GM=0.3
audit refr0 M32_main "$T3" auth_ref0 0 "$T3/calibration.json" &
sleep 60
audit refr1 M32_main "$T3" auth_ref1 1 "$T3/calibration.json" &
sleep 60
RUN=$RUN TOKENS=$T4 FC_OUT=forecast_h.json LOG_PREFIX="fc_$RUN" GPU_MEM=$GM bash "$BASE/tools/forecast_run.sh" \
  > "$L/fc_${RUN}_run.log" 2>&1 &
wait
log "done: gate $(tail -n 1 "$L/g_${RUN}_run.log")"
