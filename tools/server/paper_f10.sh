#!/usr/bin/env bash
# icutwin on the host: Paper P amendment 10 (docs/p1_paper_protocol.md), CPU only. The clinical scores of A1's
# held-out anchors (tools/paper_clinical.py), the learned baselines in four lanes (tools/paper_baselines_more.py: R's
# nine endpoints in three, L24 then L48 in the fourth; 4 threads each), then the comparisons
# (tools/paper_comparators.py). Logs: logs/f10_*.{log,exit}, logs/paper_f10.log.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
P=$R/runs/paper
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*" >> "$L/paper_f10.log"; }
run() { bash tools/run.sh "$@"; log "$1 exit $(cat "$L/$1.exit")"; }
T=(env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4)
ML=("$PY" "$R/tools/paper_baselines_more.py" --train "$S/real_train" --val "$S/real_dev" --test "$S/real_test"
    --n-jobs 4)
log start
run f10_clinical env ICUTWIN_DUCKDB_THREADS=12 ICUTWIN_DUCKDB_MEMORY_LIMIT=40GB "$PY" "$R/tools/paper_clinical.py" \
  --designs "$P/a1/test_L24" "$P/a1/test_L48" "$P/a1/test_R" --clif "$R/data/clif_canonical/full" \
  --derived "$R/data/derived/full" --raw "$R/data/raw/full" --out "$P/f10/clinical"
(run f10_ml_R_a "${T[@]}" "${ML[@]}" --design-dir "$P/a1/test_R" --out "$P/f10/ml_R_a" \
   --programs decompensation@1440 imv_onset@1440 vasopressor_onset@1440) &
(run f10_ml_R_b "${T[@]}" "${ML[@]}" --design-dir "$P/a1/test_R" --out "$P/f10/ml_R_b" \
   --programs rrt_onset@2880 aki_2@2880 sepsis_ase@1440) &
(run f10_ml_R_c "${T[@]}" "${ML[@]}" --design-dir "$P/a1/test_R" --out "$P/f10/ml_R_c" \
   --programs hypotension@240 hyperkalemia_top@1440 thrombocytopenia@2880) &
(run f10_ml_L24 "${T[@]}" "${ML[@]}" --design-dir "$P/a1/test_L24" --out "$P/f10/ml_L24" &&
   run f10_ml_L48 "${T[@]}" "${ML[@]}" --design-dir "$P/a1/test_L48" --out "$P/f10/ml_L48") &
wait
run f10_comparators "$PY" "$R/tools/paper_comparators.py" --a1 "$P/a1" --subsets "$P/f1/R128" "$P/f1/L24x" \
  --ml "$P/f10" --clinical "$P/f10/clinical/scores.parquet" --out "$P/f10"
log done
