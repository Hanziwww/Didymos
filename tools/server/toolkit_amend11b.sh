#!/usr/bin/env bash
# icutwin on the GPU host: Paper P amendment 11's post-hoc steps after V2 and V3 (the study lead's decisions, recorded
# in the protocol before they ran): V2b (logistic recalibration of the treatment starts), V3x (V3's loss test at P1's
# sizes), V3's final verdict, and the gate record when it is PASS. Each step through tools/run.sh.
#   setsid -f bash /root/autodl-tmp/tools/toolkit_amend11b.sh > /root/autodl-tmp/logs/a11b_driver.log 2>&1 < /dev/null
BASE=/root/autodl-tmp
R=$BASE/icutwin
PY=$R/.venv/bin/python
step() {
  local name=$1
  shift
  if [ -f "$BASE/logs/$name.exit" ] && [ "$(cut -d' ' -f1 "$BASE/logs/$name.exit")" = 0 ]; then
    echo "$(date '+%F %T') $name: done before"
    return 0
  fi
  echo "$(date '+%F %T') $name: start"
  bash "$BASE/tools/run.sh" "$name" "$@"
  local code
  code=$(cut -d' ' -f1 "$BASE/logs/$name.exit")
  echo "$(date '+%F %T') $name: exit $code"
  [ "$code" = 0 ] || exit 1
}
step a11b_v2b "$PY" "$R/tools/server/toolkit_v2b.py"
step a11b_v3x_design "$PY" "$R/tools/server/toolkit_v3.py" --part design --out "$R/runs/toolkit/v3x" --seed 1 \
  --per-side 3000 --per-side-b 0 --per-side-c 0 --balance
step a11b_v3x_loss "$PY" "$R/tools/server/toolkit_v3.py" --part loss --out "$R/runs/toolkit/v3x"
step a11b_v3x_analyse "$PY" "$R/tools/server/toolkit_v3.py" --part analyse_loss --out "$R/runs/toolkit/v3x" --seed 1
step a11b_v3_combine "$PY" "$R/tools/server/toolkit_v3.py" --part combine
step a11b_gate "$PY" "$R/tools/server/gate_record_v3.py"
echo "$(date '+%F %T') done"
