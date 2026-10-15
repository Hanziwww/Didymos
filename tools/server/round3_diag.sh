#!/usr/bin/env bash
# icutwin on the AutoDL host: round 3, step 0 (docs/m7_refinement.md), the diagnosis of M32_main, nothing trained:
# 0a the teacher-forced hazard calibration (GPU 0) and 0d the checkpoint averages (GPU 1) in parallel; 0b the forecast
# rollouts with horizons (four GPUs); 0c a replicate audit with twin seed 1 under the round-2 sampler (per-dataset
# treatment rates and runaway shares). Aggregates only; twins and rollouts deleted by their scripts.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=$R/data/tokens/full_r3
RUN=M32_main
HF=$R/runs/train/full/$RUN/final/hf
E=$R/runs/eval/$RUN
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
want=$(cd "$R" && .venv/bin/python -c "import json; print(json.load(open('runs/train/full/$RUN/FROZEN.json'))['checkpoint_files_sha256']['final/hf/model.safetensors'])")
have=$(sha256sum "$HF/model.safetensors" | cut -d' ' -f1)
[ "$want" = "$have" ] || { log "weights differ from FROZEN.json"; exit 1; }
mkdir -p "$E"
bash tools/run.sh r3_calibration env CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m icutwin.eval.calibration \
  --tokens "$T3" --model "$HF" --out "$E/calibration.json" &
bash tools/run.sh r3_ckpt_average env CUDA_VISIBLE_DEVICES=1 .venv/bin/python "$BASE/tools/ckpt_average.py" "$RUN" 2 3 5 &
wait
log "0a calibration exit $(cat "$L/r3_calibration.exit"); 0d checkpoint averages exit $(cat "$L/r3_ckpt_average.exit")"
RUN=$RUN FC_OUT=forecast_h.json LOG_PREFIX=fch bash "$BASE/tools/forecast_run.sh" > "$L/fch_run.log" 2>&1
log "0b forecast: $(tail -n 1 "$L/fch_run.log")"
SEED=1 AUTH_DIR=auth_d LOG_PREFIX=authd LOS_BANDS=$T3/los_bands.json bash "$BASE/tools/authenticity_m32.sh" \
  > "$L/authd_run.log" 2>&1
log "0c audit (seed 1): $(tail -n 1 "$L/authd_run.log")"
log "done"
