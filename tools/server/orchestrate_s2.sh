#!/usr/bin/env bash
# icutwin on the AutoDL host: start S1b (parquet mirror) and S2 (CLIF pipeline) per dataset as their inputs complete.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
cd "$BASE" || exit 1
code() { [ -e "$L/$1.exit" ] && cut -d' ' -f1 "$L/$1.exit"; }
ok() { [ "$(code "$1")" = 0 ]; }
start_once() {   # <step name> <command...>: starts a step once (a marker file records the start)
  local name=$1
  shift
  [ -e "$L/$name.started" ] && return
  touch "$L/$name.started"
  rm -f "$L/$name.exit"
  setsid -f bash tools/run.sh "$name" "$@" > /dev/null 2>&1 < /dev/null
  echo "$(date '+%F %T') started $name"
}
while :; do
  if ok stage_full; then
    start_once mirror_mimic env ICUTWIN_TMP="$R/.tmp/mirror_mimic" .venv/bin/python -m icutwin.stage.cli task=mirror \
      mode=full "datasets=[mimic]" duckdb.threads=48 duckdb.memory_limit=200GB hydra.run.dir="$R/runs/hydra/stage/mirror_mimic"
  fi
  for d in hirid aumc eicu mimic; do
    if ok "mirror_$d" && { [ "$d" != mimic ] || [ -e "$L/mirror_mimic.started" ]; }; then
      start_once "clif_$d" env ICUTWIN_TMP="$R/.tmp/clif_$d" ICUTWIN_DUCKDB_THREADS=32 ICUTWIN_DUCKDB_MEMORY_LIMIT=150GB \
        POLARS_MAX_THREADS=32 .venv/bin/python -m icutwin.clif.cli task=pipeline mode=full "datasets=[$d]" \
        hydra.run.dir="$R/runs/hydra/clif/pipeline_$d"
    fi
  done
  n=0
  for d in hirid aumc eicu mimic; do [ -e "$L/clif_$d.started" ] && n=$((n + 1)); done
  [ "$n" = 4 ] && { echo "$(date '+%F %T') all S2 steps started"; exit 0; }
  sleep 30
done
