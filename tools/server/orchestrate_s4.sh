#!/usr/bin/env bash
# icutwin on the AutoDL host: S3 QA and reconciliation, S4 MEDS and S5 tokens, each step as soon as its inputs
# are complete (dependencies below); a failed step stops what depends on it (the others go on).
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
cd "$BASE" || exit 1
DATASETS="mimic aumc eicu hirid sicdb nwicu inspire"
LIMITS="ICUTWIN_DUCKDB_THREADS=24 ICUTWIN_DUCKDB_MEMORY_LIMIT=120GB POLARS_MAX_THREADS=24"
code() { [ -e "$L/$1.exit" ] && cut -d' ' -f1 "$L/$1.exit"; }
ok() { [ "$(code "$1")" = 0 ]; }
all_ok() { local p=$1; shift; for d in "$@"; do ok "${p}_$d" || return 1; done; }
start_once() {   # <step> <command...>
  local name=$1
  shift
  [ -e "$L/$name.started" ] && return
  touch "$L/$name.started"
  rm -f "$L/$name.exit"
  setsid -f bash tools/run.sh "$name" "$@" > /dev/null 2>&1 < /dev/null
  echo "$(date '+%F %T') started $name"
}
py() { echo env $LIMITS ICUTWIN_TMP="$R/.tmp/$1" .venv/bin/python -m; }
while :; do
  for d in $DATASETS; do
    if ok "clif_$d"; then
      for t in parity build derived; do
        # shellcheck disable=SC2046
        start_once "meds_${t}_$d" $(py "meds_${t}_$d") icutwin.meds.cli "task=$t" mode=full "datasets=[$d]" \
          "hydra.run.dir=$R/runs/hydra/meds/${t}_$d"
      done
    fi
    if ok "meds_build_$d"; then
      # shellcheck disable=SC2046
      start_once "meds_weights_$d" $(py "meds_weights_$d") icutwin.meds.cli task=weights mode=full "datasets=[$d]" \
        "hydra.run.dir=$R/runs/hydra/meds/weights_$d"
    fi
  done
  if ok clif_mimic && ok mirror_mimic; then
    # shellcheck disable=SC2046
    start_once meds_mimic_code $(py meds_mimic_code) icutwin.meds.cli task=mimic_code mode=full "datasets=[mimic]" \
      "hydra.run.dir=$R/runs/hydra/meds/mimic_code"
  fi
  # shellcheck disable=SC2086
  if all_ok clif $DATASETS; then
    # shellcheck disable=SC2046
    start_once clif_qa $(py clif_qa) icutwin.clif.cli task=qa mode=full "hydra.run.dir=$R/runs/hydra/clif/qa"
    # shellcheck disable=SC2086
    if all_ok community $DATASETS; then
      # shellcheck disable=SC2046
      start_once clif_reconcile $(py clif_reconcile) icutwin.clif.cli task=reconcile mode=full \
        "hydra.run.dir=$R/runs/hydra/clif/reconcile"
    fi
  fi
  # shellcheck disable=SC2086
  if all_ok meds_build $DATASETS && all_ok meds_derived $DATASETS; then
    # shellcheck disable=SC2046
    start_once tok_fit $(py tok_fit) icutwin.tokenization.cli task=fit mode=full workers=100 "hydra.run.dir=$R/runs/hydra/tokenize/fit"
    if ok tok_fit; then
      # shellcheck disable=SC2046
      start_once tok_encode $(py tok_encode) icutwin.tokenization.cli task=encode mode=full workers=100 \
        "hydra.run.dir=$R/runs/hydra/tokenize/encode"
    fi
    if ok tok_encode; then
      # shellcheck disable=SC2046
      start_once tok_validate $(py tok_validate) icutwin.tokenization.cli task=validate mode=full workers=100 \
        "hydra.run.dir=$R/runs/hydra/tokenize/validate"
    fi
  fi
  [ -e "$L/tok_validate.started" ] && [ -e "$L/clif_reconcile.started" ] && [ -e "$L/meds_mimic_code.started" ] && {
    echo "$(date '+%F %T') every step started"; exit 0; }
  sleep 30
done
