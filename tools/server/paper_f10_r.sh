#!/usr/bin/env bash
# icutwin on the host: Paper P amendment 10's learned baselines for the R design in one process with forked workers
# sharing the features. liblinear copies the rows (about 6.5 GB a fit): 12 linear fits at once broke the pool, so
# the run used 10 workers with 3 linear ones, then resumed (once the other models were done) with 8 workers, 7 of
# them linear. Each finished fit is saved, and a broken run resumes (up to 3 attempts). Then the comparisons again
# once tools/server/paper_f10.sh is done.
# Logs: logs/f10_ml_R.{log,exit}, logs/f10_comparators2.{log,exit}, logs/paper_f10.log.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
P=$R/runs/paper
cd "$BASE" || exit 1
for attempt in 1 2 3; do
  bash tools/run.sh f10_ml_R env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" \
    "$R/tools/paper_baselines_more.py" --train "$S/real_train" --val "$S/real_dev" --test "$S/real_test" --n-jobs 1 \
    --workers 8 --linear-workers 7 --design-dir "$P/a1/test_R" --out "$P/f10/ml_R"
  echo "$(date '+%F %T') f10_ml_R attempt $attempt exit $(cat "$L/f10_ml_R.exit")" >> "$L/paper_f10.log"
  [ "$(cut -d' ' -f1 "$L/f10_ml_R.exit")" = 0 ] && break
done
until grep -qE "^[0-9-]+ [0-9:]+ done$" "$L/paper_f10.log"; do sleep 30; done
bash tools/run.sh f10_comparators2 "$PY" "$R/tools/paper_comparators.py" --a1 "$P/a1" --subsets "$P/f1/R128" \
  "$P/f1/L24x" --ml "$P/f10" --clinical "$P/f10/clinical/scores.parquet" --out "$P/f10"
echo "$(date '+%F %T') f10_comparators2 exit $(cat "$L/f10_comparators2.exit")" >> "$L/paper_f10.log"
echo "$(date '+%F %T') rc done" >> "$L/paper_f10.log"
