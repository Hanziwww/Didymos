#!/usr/bin/env bash
# icutwin on the AutoDL host: one round-3 candidate (docs/m7_refinement.md, "Candidates"):
#   round3_candidate.sh <run name> [hydra overrides of the train section...]
# e.g.  round3_candidate.sh R2_main train.dropout=0.1 train.input_noise=0.15
# The 50-step DDP smoke test with the candidate's settings, the step budget from the r3 exposure (the configured
# passes), the run, the freeze, P1 (membership, GPU 0), and the screen: 300 twins per dataset with the round-2
# sampler (length-resampled), the audit's measures and the per-dataset treatment rates (runs/eval/<run>/screen).
# Logs: logs/<run>_*.{log,exit}; logs/<run>_screen_run.log.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=$R/data/tokens/full_r3
Q3=$R/data/qa/full_r3
RUN=${1:?run name}
shift
cd "$BASE" || exit 1
code() { [ -e "$L/$1.exit" ] && cut -d' ' -f1 "$L/$1.exit"; }
ok() { [ "$(code "$1")" = 0 ]; }
log() { echo "$(date '+%F %T') $*"; }
bash tools/run.sh "${RUN}_smoke" env ICUTWIN_TMP="$R/.tmp/${RUN}_smoke" .venv/bin/accelerate launch \
  --config_file src/icutwin/configs/accelerate/ddp.yaml -m icutwin.model.cli task=smoke mode=full \
  "tokens_root=$T3" "qa_root=$Q3" "train.name=$RUN" "$@" "hydra.run.dir=$R/runs/hydra/model/${RUN}_smoke"
log "smoke exit $(cat "$L/${RUN}_smoke.exit")"
ok "${RUN}_smoke" || exit 1
steps=$(cd "$R" && .venv/bin/python - <<EOF
import json, math
from omegaconf import OmegaConf
e = json.load(open("$Q3/exposure.json"))
t = OmegaConf.load("src/icutwin/configs/train.yaml").train
passes = dict(t.passes)
budget = sum(passes.get(d, passes["default"]) * s["tokens"] for d, s in e["splits"]["train"].items())
print(math.ceil(budget / (int(t.tokens_per_batch) * int(t.grad_accum) * 4)))
EOF
)
log "budget: $steps steps"
bash tools/run.sh "train_$RUN" bash "$BASE/tools/train.sh" "$RUN" "$steps" "tokens_root=$T3" "qa_root=$Q3" "$@"
log "train exit $(cat "$L/train_$RUN.exit")"
ok "train_$RUN" || exit 1
bash tools/run.sh "${RUN}_freeze" .venv/bin/python "$BASE/tools/freeze_run.py" "$RUN" "$T3" \
  "round-3 candidate, frozen for its screen and the M7 gate: no further training; no deliverable synthetic data"
log "freeze exit $(cat "$L/${RUN}_freeze.exit")"
ok "${RUN}_freeze" || exit 1
bash tools/run.sh "g_${RUN}_p1" .venv/bin/python -m icutwin.eval.cli mode=full "run=$RUN" "tokens_root=$T3" \
  "out_dir=$R/runs/audit/m7_$RUN" step=p1 gpu=0 "hydra.run.dir=$R/runs/hydra/eval/g_${RUN}_p1"
log "P1 exit $(cat "$L/g_${RUN}_p1.exit")"
ok "g_${RUN}_p1" || exit 1
RUN=$RUN REQUIRE=p1 PER_DATASET=300 AUTH_DIR=screen LOG_PREFIX="${RUN}_screen" LOS_BANDS=$T3/los_bands.json \
  bash "$BASE/tools/authenticity_m32.sh" > "$L/${RUN}_screen_run.log" 2>&1
log "screen: $(tail -n 1 "$L/${RUN}_screen_run.log")"
