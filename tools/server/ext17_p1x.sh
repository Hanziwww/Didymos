#!/usr/bin/env bash
# icutwin on the RTX 5090 host (westd): Paper P amendment 17c (docs/p1_paper_protocol.md). The adapted run's M7 gate
# was INCONCLUSIVE on P1 (eICU) only: P1 again on the same frozen weights with up to 10,000 members and 10,000
# non-members per database (p1.cap 10000; the same attack, criteria and seed), the cap-3,000 P1 and report kept as
# runs/audit/m7_<run>/{p1,report}_cap3000.json; then the report (P2-P5 as run), and F1's futures
# (tools/server/paper_ext17.sh, ARMS=F1) only if the gate is PASS. Logs: logs/e17x_*.{log,exit}, logs/ext17_train.log.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
T4=$R/data/tokens/full_r4
RUN=$("$PY" -c "import json; print(json.load(open('$R/runs/paper/ext17/grid.json'))['chosen'])") || exit 1
OUT=$R/runs/audit/m7_$RUN
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') 17c: $*" >> "$L/ext17_train.log"; }
ok() { [ "$(cut -d' ' -f1 "$L/$1.exit" 2>/dev/null)" = 0 ]; }
[ -f "$OUT/p1_cap3000.json" ] || cp -p "$OUT/p1.json" "$OUT/p1_cap3000.json"
[ -f "$OUT/report_cap3000.json" ] || cp -p "$OUT/report.json" "$OUT/report_cap3000.json"
for step in p1 report; do
  ok "e17x_$step" && continue
  extra=()
  [ $step = p1 ] && extra=(p1.cap=10000 gpu=0 tokens_per_batch=32768)
  bash tools/run.sh "e17x_$step" "$PY" -m icutwin.eval.cli mode=full "run=$RUN" "tokens_root=$T4" "out_dir=$OUT" \
    "step=$step" "${extra[@]}" "hydra.run.dir=$R/runs/hydra/eval/e17x_$step"
  log "$step exit $(cat "$L/e17x_$step.exit")"
  ok "e17x_$step" || exit 1
done
gate=$("$PY" -c "import json; print(json.load(open('$OUT/report.json'))['gate'])")
log "gate of $RUN with P1 at 10,000: $gate ($("$PY" -c "
import json
g = json.load(open('$OUT/p1.json'))['groups']
print('; '.join(f\"{k} {x['members']} TPR {x['tpr_at_1pct_fpr'][0]:.4f} [{x['tpr_at_1pct_fpr'][1]:.4f}, {x['tpr_at_1pct_fpr'][2]:.4f}] {x['status']}\" for k, x in g.items()))"))"
[ "$gate" = PASS ] || exit 1
cp "$R/tools/server/paper_ext17.sh" "$BASE/tools/e17_rollouts_f1.sh"
ARMS=F1 bash "$BASE/tools/e17_rollouts_f1.sh"
log "F1 futures ended"
