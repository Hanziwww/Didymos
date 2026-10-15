#!/usr/bin/env bash
# icutwin on the AutoDL host: run one pipeline step in the project environment, detached from the SSH session.
#   bash tools/run.sh <name> <command...>      e.g.  bash tools/run.sh stage .venv/bin/python -m icutwin.stage.cli ...
# Writes logs/<name>.log, and logs/<name>.exit ("<exit code> <seconds>") when the step ends.
BASE=/root/autodl-tmp
ROOT=$BASE/icutwin
# shellcheck disable=SC1091
. "$ROOT/.cache/server.env"
cd "$ROOT" || exit 1
name=$1
shift
rm -f "$BASE/logs/$name.exit"
start=$(date +%s)
"$@" > "$BASE/logs/$name.log" 2>&1
code=$?
echo "$code $(( $(date +%s) - start ))" > "$BASE/logs/$name.exit"
