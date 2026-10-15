#!/usr/bin/env bash
# icutwin on the one-GPU host: rerun A1 analyses that failed, after the CPU runner is done (one at a time, as the
# runner does; tools/paper_a1.py resumes from the store segments already built). Names as arguments:
#   bash paper_rerun_a1.sh test_L24
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
cd "$BASE" || exit 1
until grep -q "runner done" "$L/paper_cpu_runner.log" 2>/dev/null; do sleep 60; done
for n in "$@"; do
  rm -f "$L/p1_${n}_a1.exit"
  rm -rf "$R/data/paper/stores/a1_$n/_work"
  bash tools/run.sh "p1_${n}_a1" env OMP_NUM_THREADS=1 "$R/.venv/bin/python" "$R/tools/paper_a1.py" \
    --design-dir "$R/runs/paper/a1/$n" --store "$R/data/factory/stores/real_test" --work "$R/data/paper/stores/a1_$n" \
    --out "$R/runs/paper/a1/$n" --workers 4 --chunk 100
  echo "$(date '+%F %T') rerun a1 $n: $(cat "$L/p1_${n}_a1.exit")" >> "$L/paper_cpu_runner.log"
done
