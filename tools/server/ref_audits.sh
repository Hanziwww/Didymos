#!/usr/bin/env bash
# icutwin on the AutoDL host: round 3's reference for the decision (docs/m7_refinement.md, "Targets and decision"):
# M32_main audited twice (twin seeds 0 and 1) with the round-3 sampler (twins raked to the training stays' length
# and treatment margins, calibration.json), queued behind the step-0c audit's GPU work; each audit's generation
# waits for the previous one's GPU work. Output runs/eval/M32_main/auth_ref{0,1}; logs authref{0,1}_*.
B=/root/autodl-tmp
T3=$B/icutwin/data/tokens/full_r3
cd "$B" || exit 1
until [ -e logs/authd_model.exit ]; do sleep 30; done
SEED=0 AUTH_DIR=auth_ref0 LOG_PREFIX=authref0 CALIBRATION=$T3/calibration.json bash tools/authenticity_m32.sh \
  > logs/authref0_run.log 2>&1 &
until [ -e logs/authref0_model.exit ]; do sleep 30; done
SEED=1 AUTH_DIR=auth_ref1 LOG_PREFIX=authref1 CALIBRATION=$T3/calibration.json bash tools/authenticity_m32.sh \
  > logs/authref1_run.log 2>&1
wait
echo "$(date '+%F %T') reference audits done"
