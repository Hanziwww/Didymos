#!/usr/bin/env bash
# icutwin on the AutoDL host, rebuild r3 for the second training run (M32_main): INSPIRE (#95) and HiRID (#98) through
# their CLIF chain (adapter, harmonize, validate, CLIF_MEDS), the MEDS steps and the reconciliation; S3 QA; then the
# tokens of all seven datasets at 32k windows into data/tokens/full_r3 (M_main's frozen snapshot, data/tokens/full,
# is not touched). HiRID starts once the MIMIC reconciliation has ended (memory: the container's 480 GiB).
# A step's result is logs/<step>.exit ("<code> <seconds>"); logs/r3.exit is the whole rebuild's.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=$R/data/tokens/full_r3
Q3=$R/data/qa/full_r3
cd "$BASE" || exit 1
code() { [ -e "$L/$1.exit" ] && cut -d' ' -f1 "$L/$1.exit"; }
ok() { [ "$(code "$1")" = 0 ]; }
log() { echo "$(date '+%F %T') $*"; }
step() {   # <step> <threads> <memory> <python -m args...>
  local name=$1 th=$2 mem=$3
  shift 3
  rm -f "$L/$name.exit"
  rm -rf "$R/.tmp/$name"
  log "started $name"
  bash tools/run.sh "$name" env ICUTWIN_DUCKDB_THREADS="$th" ICUTWIN_DUCKDB_MEMORY_LIMIT="$mem" POLARS_MAX_THREADS="$th" \
    ICUTWIN_TMP="$R/.tmp/$name" .venv/bin/python -m "$@" "hydra.run.dir=$R/runs/hydra/r3/$name"
  log "$name exit $(cat "$L/$name.exit")"
  ok "$name"
}
chain() {   # <dataset> <threads> <memory>: CLIF from raw (our adapter), canonical CLIF, CLIFpy gate, CLIF_MEDS
  local d=$1
  rm -f "$L/clif_$d.exit"
  log "started clif_$d"
  bash tools/run.sh "clif_$d" bash "$BASE/tools/clif_chain.sh" "$d" "$2" "$3" adapter harmonize validate meds_tool
  log "clif_$d exit $(cat "$L/clif_$d.exit")"
  ok "clif_$d"
}
meds_dataset() {   # <dataset>: parity, build and derived side by side, then weights
  local d=$1 fail=0
  step "meds_parity_$d" 12 32GB icutwin.meds.cli task=parity mode=full "datasets=[$d]" &
  local p1=$!
  step "meds_derived_$d" 12 32GB icutwin.meds.cli task=derived mode=full "datasets=[$d]" &
  local p2=$!
  step "meds_build_$d" 16 48GB icutwin.meds.cli task=build mode=full "datasets=[$d]" || fail=1
  [ $fail = 0 ] && { step "meds_weights_$d" 8 20GB icutwin.meds.cli task=weights mode=full "datasets=[$d]" || fail=1; }
  wait $p1 || fail=1
  wait $p2 || fail=1
  return $fail
}
recon() {   # <dataset> <threads> <memory>
  step "recon_$1" "$2" "$3" icutwin.clif.cli task=reconcile mode=full "datasets=[$1]" \
    "qa_root=$R/data/qa/full/reconcile_parts/$1"
}
start=$(date +%s)
log "r3 start"
inspire_lane() { chain inspire 16 60GB && meds_dataset inspire && recon inspire 12 40GB; }
hirid_lane() {
  until [ -e "$L/recon_mimic.exit" ]; do sleep 30; done
  chain hirid 32 150GB && meds_dataset hirid && recon hirid 48 200GB
}
inspire_lane & a=$!
hirid_lane & b=$!
fail=0
wait $a || fail=1
wait $b || fail=1
log "data lanes exit $fail"
# the merged reconciliation report of every dataset's latest part
(cd "$R" && .venv/bin/python - <<'EOF'
import json
from pathlib import Path
from icutwin.clif import reconcile
qa = Path("data/qa/full")
reports = []
for d in ("mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"):
    p = qa / "reconcile_parts" / d / f"{reconcile.REPORT}.json"
    if p.exists():
        reports += [r for r in json.loads(p.read_text(encoding="utf-8")) if r["dataset"] == d]
(qa / f"{reconcile.REPORT}.json").unlink(missing_ok=True)
reconcile.write(reports, qa)
print("merged", [r["dataset"] for r in reports])
EOF
)
step clif_qa_r3 16 60GB icutwin.clif.cli task=qa mode=full "qa_root=$Q3" || fail=1
step tok_fit_r3 24 200GB icutwin.tokenization.cli task=fit mode=full workers=64 "out_root=$T3" "qa_root=$Q3" &&
  step tok_encode_r3 24 200GB icutwin.tokenization.cli task=encode mode=full workers=64 "out_root=$T3" "qa_root=$Q3" && {
    step exposure_r3 8 20GB icutwin.model.cli task=exposure mode=full "tokens_root=$T3" "qa_root=$Q3" &
    e=$!
    step tok_validate_r3 24 200GB icutwin.tokenization.cli task=validate mode=full workers=64 "out_root=$T3" "qa_root=$Q3" || fail=1
    wait $e || fail=1
  } || fail=1
echo "$fail $(( $(date +%s) - start ))" > "$L/r3.exit"
log "r3 done: exit $fail"
