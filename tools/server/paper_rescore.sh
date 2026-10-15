#!/usr/bin/env bash
# icutwin on the one-GPU host: Paper P amendment 7, the hazards rescored with the sampler's renormalization
# (eval.hazards.normalized: the grammar's timeline rules, the infusion rule and the dataset's support; checked against
# the exact masked probabilities by paper_hazard_check.py), after the first R128 scoring with the model's raw
# probabilities (kept as hazards_raw/). Also reruns the R128 A1-style analysis (its store build failed on a decode
# chunk without vitals: meds.derived fixed). Then the F2/F3 analyses of D1 and R128. Logs: logs/rs_*, logs/paper_f1.log.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
T4=$R/data/tokens/full_r4
P=$R/runs/paper
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') rescore $*" >> "$L/paper_f1.log"; }
setsid -f bash tools/run.sh f1_a1_R128 env OMP_NUM_THREADS=1 "$PY" "$R/tools/paper_a1.py" --design-dir "$P/f1/R128" \
  --store "$S/real_test" --work "$R/data/paper/stores/f1_R128" --out "$P/f1/R128" --workers 4 --chunk 100 \
  --min-valid 64 --curve-draws 20 > /dev/null 2>&1 < /dev/null
log "analysis R128 restarted"
hz() {   # <name> <design dir> [extra...]
  local name=$1 dir=$2
  shift 2
  [ -d "$dir/hazards" ] && [ ! -d "$dir/hazards_raw" ] && mv "$dir/hazards" "$dir/hazards_raw"
  bash tools/run.sh "rs_hz_$name" env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "$R/tools/server/paper_hazards.py" --run MUC32_main --tokens "$T4" --split held_out --design-dir "$dir" \
    --tokens-per-batch 32768 --chunk 2048 "$@"
  log "hazards $name exit $(cat "$L/rs_hz_$name.exit")"
}
hz d1_R "$P/a1/test_R" --real --no-rollouts
hz d1_L24 "$P/a1/test_L24" --real --no-rollouts --long
hz R128 "$P/f1/R128" --real
for d in R L24; do
  X=""
  [ "$d" = L24 ] && X="--long"
  bash tools/run.sh "rs_f2_d1_$d" env OMP_NUM_THREADS=2 "$PY" "$R/tools/paper_f2.py" --subset "$P/a1/test_$d" \
    --a1 "$P/a1/test_$d" --out "$P/f3/D1_$d" $X
  log "D1 $d analysis exit $(cat "$L/rs_f2_d1_$d.exit")"
done
until [ -f "$L/f1_a1_R128.exit" ]; do sleep 60; done
bash tools/run.sh rs_f2_R128 env OMP_NUM_THREADS=2 "$PY" "$R/tools/paper_f2.py" --subset "$P/f1/R128" \
  --a1 "$P/a1/test_R" --out "$P/f1/R128"
log "R128 analysis exit $(cat "$L/rs_f2_R128.exit")"
