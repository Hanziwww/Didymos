#!/usr/bin/env bash
# icutwin on the AutoDL host: round 3, step 0c finished by hand. Its run (authenticity_m32.sh, AUTH_DIR auth_d) died
# after generating the twins: the script file was replaced while bash was still reading it. The CPU steps on the
# twins it wrote (audit, explain, the pooled harness with utility and per-dataset prevalence, targets; the model
# checks are skipped: the GPUs train, and round 2 measured them for M32_main), then the twins deleted.
B=/root/autodl-tmp
R=$B/icutwin
T3=$R/data/tokens/full_r3
E=$R/runs/eval/M32_main/auth_d
cd "$B" || exit 1
bash tools/run.sh authd_audit .venv/bin/python -m icutwin.eval.authenticity audit --tokens "$T3" --twins "$E/twins" \
  --out "$E/audit.json" &
bash tools/run.sh authd_quality .venv/bin/python -m icutwin.eval.quality evaluate --tokens "$T3" --samples "$E/twins" \
  --out "$E/quality.json" &
bash tools/run.sh authd_explain .venv/bin/python -m icutwin.eval.authenticity explain --tokens "$T3" --twins "$E/twins" \
  --out "$E/explain.json" &
wait
bash tools/run.sh authd_targets .venv/bin/python -m icutwin.eval.authenticity targets --audit "$E/audit.json" \
  --quality "$E/quality.json" --out "$E/targets.json"
rm -rf "$E/twins"
echo "$(date '+%F %T') done: audit $(cat logs/authd_audit.exit), quality $(cat logs/authd_quality.exit), explain $(cat logs/authd_explain.exit), targets $(cat logs/authd_targets.exit); twins deleted"
