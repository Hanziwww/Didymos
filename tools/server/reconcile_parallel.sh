#!/usr/bin/env bash
# icutwin on the AutoDL host: the S3 reconciliation, one process per dataset (each its own report under
# qa/full/reconcile_parts/<dataset>), then merged into qa/full/reconciliation.{json,md}. Writes the step result
# logs/clif_reconcile.exit ("<code> <seconds>": 0 when every dataset has every row paired or explained).
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
cd "$BASE" || exit 1
start=$(date +%s)
rm -f "$L/clif_reconcile.exit"
# dataset threads memory
SPECS="mimic 32 160GB
aumc 24 120GB
eicu 24 120GB
hirid 16 60GB
sicdb 12 40GB
nwicu 12 30GB
inspire 12 30GB"
while read -r d th mem; do
  rm -f "$L/recon_$d.exit"
  setsid -f bash tools/run.sh "recon_$d" env ICUTWIN_DUCKDB_THREADS="$th" ICUTWIN_DUCKDB_MEMORY_LIMIT="$mem" \
    POLARS_MAX_THREADS="$th" ICUTWIN_TMP="$R/.tmp/recon_$d" .venv/bin/python -m icutwin.clif.cli task=reconcile \
    mode=full "datasets=[$d]" "qa_root=$R/data/qa/full/reconcile_parts/$d" "hydra.run.dir=$R/runs/hydra/clif/recon_$d" \
    > /dev/null 2>&1 < /dev/null
  echo "$(date '+%F %T') started recon_$d"
done <<< "$SPECS"
fail=0
for d in mimic aumc eicu hirid sicdb nwicu inspire; do
  until [ -e "$L/recon_$d.exit" ]; do sleep 20; done
  c=$(cut -d' ' -f1 "$L/recon_$d.exit")
  echo "$(date '+%F %T') recon_$d exit $(cat "$L/recon_$d.exit")"
  [ "$c" = 0 ] || fail=1
done
cd "$R" && .venv/bin/python - <<'EOF'
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
merged=$?
[ $merged = 0 ] || fail=1
echo "$fail $(( $(date +%s) - start ))" > "$L/clif_reconcile.exit"
echo "$(date '+%F %T') reconciliation done: exit $fail"
