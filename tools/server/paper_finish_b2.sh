#!/usr/bin/env bash
# icutwin on the one-GPU host: Paper P's B2 analysis (tools/paper_a1.py on the unknown-site rollouts), then the L24
# comparison with it (tools/paper_compare.py --unknown). Logs: logs/p1_b2_L24_unknown_a1.*, logs/pq2_compare_L24.*.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
P=$R/runs/paper
cd "$BASE" || exit 1
rm -f "$L/p1_b2_L24_unknown_a1.exit"
rm -rf "$R/data/paper/stores/a1_b2_L24_unknown"
bash tools/run.sh p1_b2_L24_unknown_a1 env OMP_NUM_THREADS=1 "$PY" "$R/tools/paper_a1.py" \
  --design-dir "$P/a1/b2_L24_unknown" --store "$S/real_test" --work "$R/data/paper/stores/a1_b2_L24_unknown" \
  --out "$P/a1/b2_L24_unknown" --workers 4 --chunk 100
rm -f "$L/pq2_compare_L24.exit"
bash tools/run.sh pq2_compare_L24 env OMP_NUM_THREADS=2 "$PY" "$R/tools/paper_compare.py" --design-dir "$P/a1/test_L24" \
  --dev "$P/a1/dev_L24" --store "$S/real_test" --out "$P/a1/test_L24" --unknown "$P/a1/b2_L24_unknown"
echo "$(date '+%F %T') finish_b2: a1 $(cat "$L/p1_b2_L24_unknown_a1.exit"), compare $(cat "$L/pq2_compare_L24.exit")" \
  >> "$L/paper_queue2.log"
