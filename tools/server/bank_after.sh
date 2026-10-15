#!/usr/bin/env bash
# icutwin on the AutoDL host, after bank_single.sh: bring the repository up to date (a bundle, BUNDLE), validate the
# bank (tools/bank_validate.py: manifests, grammar, MEDS layer b, the factory's smoke with derived concepts), and
# package it for the study lead's download without the novelty reference (regenerable: bank_produce.py builds it). The
# package is DUA-derived: internal only. Logs: logs/bank_validate.{log,exit}, logs/bank_after.log.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
NAME=${NAME:-MUC32_main_bank}
BANK=$R/data/bank/$NAME
log() { echo "$(date '+%F %T') $*"; }
cd "$R" || exit 1
if pgrep -f "tools/bank_produce.py" > /dev/null; then log "a bank part is still running"; exit 1; fi
if [ -n "${BUNDLE:-}" ]; then git pull -q --ff-only "$BUNDLE" main || { log "pull failed"; exit 1; }; fi
log "repository at $(git log --oneline | head -1 | cut -c1-60)"
cp tools/server/bank_validate.py "$BASE/tools/bank_validate.py"
bash "$BASE/tools/run.sh" bank_validate "$R/.venv/bin/python" "$BASE/tools/bank_validate.py" "$BANK" \
  "$R/data/tokens/full_r4" --meds "$R/data/meds/full" --sample 300
log "validate exit $(cat "$L/bank_validate.exit")"
mkdir -p "$BASE/export"
PKG=$BASE/export/${NAME}_$(date +%Y%m%d).tar
tar -C "$R" -cf "$PKG" --exclude=novelty_ref.npz "data/bank/$NAME"
sha256sum "$PKG" > "$PKG.sha256"
log "package $(du -h "$PKG" | cut -f1) $(cut -c1-16 "$PKG.sha256")"
log "done"
