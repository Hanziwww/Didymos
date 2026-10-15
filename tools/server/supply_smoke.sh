#!/usr/bin/env bash
# icutwin on the RTX 4080 host: the twin-supply conditioning smoke (plan Phase F, S1b; study lead: no pool is produced
# in this phase, the smoke tests the tooling only). TOTAL (2,000) twins across the four mechanisms with max-strata
# strata per dataset, four engines on GPU 0; then the segment appended twice (byte-reproducibility), the pool's weights,
# the report (aggregates), and the twins deleted. Logs: logs/s1b_*.{log,exit}; report: runs/supply/s1b_smoke.json.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T4=$R/data/tokens/full_r4
PY=$R/.venv/bin/python
NAME=${NAME:-s1b}
TOTAL=${TOTAL:-2000}
RUN=$R/data/supply/$NAME
STORE=$R/data/factory/stores/supply_$NAME
MARG=$T4/supply_marginals.json
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
step() {   # <name> <command...>: one step, stop on failure
  local n=$1
  shift
  bash tools/run.sh "${NAME}_$n" "$@"
  local code
  code=$(cut -d' ' -f1 "$L/${NAME}_$n.exit")
  log "$n exit $code: $(grep -E '^\{' "$L/${NAME}_$n.log" | tail -n 1 | cut -c1-300)"
  [ "$code" = 0 ] || exit 1
}
[ -s "$MARG" ] || step marginals "$PY" "$R/tools/factory_store.py" marginals --tokens "$T4" --out "$MARG"
step plan "$PY" "$R/tools/supply_plan.py" --marginals "$MARG" --total "$TOTAL" --max-strata "${MAX_STRATA:-6}" \
  --name "$NAME" --out "$RUN/plan.json"
cp "$R/tools/server/supply_produce.sh" "$BASE/tools/${NAME}_produce.sh"     # a running copy is never overwritten
PLAN=$RUN/plan.json OUT=$RUN GPUS=0 ENGINES=${ENGINES:-4} NAME=$NAME bash "$BASE/tools/${NAME}_produce.sh"
for p in $(seq 0 $(( ${ENGINES:-4} - 1 ))); do
  [ "$(cut -d' ' -f1 "$L/${NAME}_$p.exit" 2>/dev/null)" = 0 ] || { log "part $p failed"; exit 1; }
done
rm -rf "$STORE" "${STORE}_again"
step store "$PY" "$R/tools/factory_store.py" supply --tokens "$T4" --shards "$RUN" --out "$STORE" --segment seg_0000 \
  --workers "${WORKERS:-12}"
step store_again "$PY" "$R/tools/factory_store.py" supply --tokens "$T4" --shards "$RUN" --out "${STORE}_again" \
  --segment seg_0000 --workers 4
step weights "$PY" "$R/tools/factory_store.py" weights --out "$STORE" --plan "$RUN/plan.json" --marginals "$MARG"
step report "$PY" "$R/tools/supply_report.py" --store "$STORE" --store2 "${STORE}_again" --plan "$RUN/plan.json" \
  --marginals "$MARG" --shards "$RUN" --build-log "$L/${NAME}_store.log" --out "$R/runs/supply/${NAME}_smoke.json"
cp "$RUN/plan.json" "$R/runs/supply/${NAME}_plan.json"
rm -rf "$RUN" "$STORE" "${STORE}_again"                      # the twins are deleted; aggregates are kept
log "done: runs/supply/${NAME}_smoke.json"
