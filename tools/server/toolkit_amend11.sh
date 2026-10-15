#!/usr/bin/env bash
# icutwin on the GPU host: Paper P amendment 11's validation in order, one GPU stage at a time (each step through
# tools/run.sh: logs/<name>.log and .exit). V1 first; V2, V3, V5 (and V4 on the CPU, the similarity indexes) only when
# V1 passes, so no later result exists before V1 is decided.
#   setsid -f bash /root/autodl-tmp/tools/toolkit_amend11.sh > /root/autodl-tmp/logs/a11_driver.log 2>&1 < /dev/null
BASE=/root/autodl-tmp
R=$BASE/icutwin
PY=$R/.venv/bin/python
ENGINES=${ENGINES:-3}
WORKERS=${WORKERS:-5}
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
step a11_v1_forecast "$PY" "$R/tools/server/toolkit_v1.py" --part forecast --engines "$ENGINES" --workers "$WORKERS"
step a11_v1_whatif "$PY" "$R/tools/server/toolkit_v1.py" --part whatif --engines "$ENGINES" --workers "$WORKERS"
step a11_v1_analyse "$PY" "$R/tools/server/toolkit_v1.py" --part analyse
GATE_FILE=v1.json
if [ "${GATE:-v1}" = noise ] || [ "${GATE:-v1}" = adjusted ]; then   # amendment 11's post-hoc check (the study lead's)
  step a11_v1_noise "$PY" "$R/tools/server/toolkit_v1_noise.py"
  GATE_FILE=v1_noise.json
fi
if [ "${GATE:-v1}" = adjusted ]; then     # the study lead's adjusted criterion (post hoc, after both results)
  step a11_v1_adjusted "$PY" "$R/tools/server/toolkit_v1_adjusted.py"
  GATE_FILE=v1_adjusted.json
fi
if ! "$PY" -c "import json, sys; sys.exit(0 if json.load(open('$R/runs/toolkit/v1/$GATE_FILE'))['pass'] else 3)"; then
  echo "$(date '+%F %T') V1 did not pass ($GATE_FILE): stop (V2-V5 not run)"
  exit 3
fi
step a11_v2_fit "$PY" "$R/tools/server/toolkit_v2.py" --part fit --engines "$ENGINES" --workers "$WORKERS"
step a11_v2_evaluate "$PY" "$R/tools/server/toolkit_v2.py" --part evaluate
step a11_v3_design "$PY" "$R/tools/server/toolkit_v3.py" --part design
step a11_v3_loss "$PY" "$R/tools/server/toolkit_v3.py" --part loss
step a11_v3_sample "$PY" "$R/tools/server/toolkit_v3.py" --part sample --engines "$ENGINES"
step a11_v3_analyse "$PY" "$R/tools/server/toolkit_v3.py" --part analyse
step a11_v5 "$PY" "$R/tools/server/toolkit_v5.py" --engines 2 --workers "$WORKERS"
step a11_v4 "$PY" "$R/tools/toolkit_v4.py" --workers 12
step a11_index_R "$PY" -m icutwin.toolkit.cli tool=index name=test_R anchors_dir="$R/runs/paper/a1/test_R" \
  labels="$R/runs/paper/a1/test_R/predictions.parquet" card=false
step a11_index_L24 "$PY" -m icutwin.toolkit.cli tool=index name=test_L24 anchors_dir="$R/runs/paper/a1/test_L24" \
  labels="$R/runs/paper/a1/test_L24/predictions.parquet" card=false
echo "$(date '+%F %T') done"
