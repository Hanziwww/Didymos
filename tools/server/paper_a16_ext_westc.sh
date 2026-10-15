#!/usr/bin/env bash
# icutwin on the RTX 4080 host (westc, which holds the train store): Paper P amendment 16a's transferred comparators
# (docs/p1_paper_protocol.md), trained on the seven databases and scored at the external anchors. The amendment's
# code runs from a clone (C, on PYTHONPATH) so the checkout another job runs from is not touched. In order:
#   1. the external store (tools/factory_store.py tokens --all-splits) and the two designs' directories (the kept
#      anchors with their weights, from the local design: runs/paper/ext/{local,subsets});
#   2. the refit check: A1's LightGBM and logistic regression refit with A1's recipe (tools/paper_baselines.py,
#      --skip-a4) on A1's test_R and test_L24 must give A1's stored predictions;
#   3. the same models' predictions at the external anchors (a new database's one-hot columns are all zero);
#   4. once the GPU is free (L48x done: never beside vLLM engines), TabPFN-3.5 on the scored primary endpoints.
# Logs: logs/a16x_*.{log,exit}, logs/paper_a16_ext_westc.log.
BASE=/root/autodl-tmp
C=${C:-$BASE/icutwin_a16}
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
S=$R/data/factory/stores
TE=$R/data/tokens/ext_r4
A1=$R/runs/paper/a1
P=$R/runs/paper/ext
O=$R/runs/paper/a16
CK=${CK:-$BASE/tabpfn-3.5/tabpfn-v3.5-20260909.safetensors}
CENV=(PYTHONPATH="$C/src" ICUTWIN_REFS="$R/refs")   # the clone's code; the reference tools of the checkout
cd "$BASE" || exit 1
mkdir -p "$O"
log() { echo "$(date '+%F %T') $*" >> "$L/paper_a16_ext_westc.log"; }
ok() { [ "$(cut -d' ' -f1 "$L/$1.exit" 2>/dev/null)" = 0 ]; }
log "start $(git -C "$C" log --oneline -1)"
if ! ok a16x_store; then
  rm -rf "$S/ext_test"
  bash tools/run.sh a16x_store env "${CENV[@]}" OMP_NUM_THREADS=1 "$PY" "$C/tools/factory_store.py" tokens \
    --tokens "$TE" --all-splits --label test --out "$S/ext_test" --datasets zigong ricd --workers 6
  log "store exit $(cat "$L/a16x_store.exit")"
  ok a16x_store || exit 1
fi
env "${CENV[@]}" "$PY" - "$P" "$O" <<'EOF' || { log "design directories failed"; exit 1; }
import json, sys
from pathlib import Path

import pandas as pd

p, o = Path(sys.argv[1]), Path(sys.argv[2])
for name, design in (("R", "R"), ("L24e", "L24")):
    an = pd.read_parquet(p / "local" / f"{name}_anchors.parquet")
    sub = pd.read_parquet(p / "subsets" / f"{name}.parquet")
    an = an.drop(columns="weight").merge(sub[["anchor", "weight"]], on="anchor")
    d = o / f"ext_{name}_design"
    d.mkdir(parents=True, exist_ok=True)
    an.to_parquet(d / "anchors.parquet", index=False)
    (d / "design.json").write_text(json.dumps({"design": design, "source": "tools/paper_external.py design"}))
    print(name, len(an))
EOF
for D in R L24; do                                  # 2. the refit check on A1's own anchors
  if ! ok "a16x_refit_$D"; then
    bash tools/run.sh "a16x_refit_$D" env "${CENV[@]}" OMP_NUM_THREADS=4 "$PY" "$C/tools/paper_baselines.py" \
      --design-dir "$A1/test_$D" --train "$S/real_train" --val "$S/real_dev" --test "$S/real_test" \
      --out "$O/refit_$D" --skip-a4
    log "refit $D exit $(cat "$L/a16x_refit_$D.exit")"
  fi
  env "${CENV[@]}" "$PY" - "$A1/test_$D/baseline_predictions.parquet" "$O/refit_$D/baseline_predictions.parquet" \
    "$O/refit_$D/refit_check.json" <<'EOF'
import json, sys

import numpy as np
import pandas as pd

old, new = (pd.read_parquet(f) for f in sys.argv[1:3])
k = ["anchor", "program", "model"]
old = old[old["model"].isin(["lgbm_pooled", "logistic_pooled"])]
m = old.merge(new, on=k, how="outer", suffixes=("_stored", "_refit"), indicator=True)
both = m["_merge"] == "both"
d = np.abs(m.loc[both, "p_stored"] - m.loc[both, "p_refit"])
rep = {"pairs": int(both.sum()), "only_stored": int((m["_merge"] == "left_only").sum()),
       "only_refit_these_models": int(((m["_merge"] == "right_only") & m["model"].isin(["lgbm_pooled", "logistic_pooled"])).sum()),
       "max_abs_diff": float(d.max()) if len(d) else None, "identical_share": float((d == 0).mean()) if len(d) else None,
       "by_model": {g: float(x.max()) for g, x in d.groupby(m.loc[both, "model"])}}
open(sys.argv[3], "w").write(json.dumps(rep, indent=1))
print(json.dumps(rep))
EOF
  log "refit $D check $(cat "$O/refit_$D/refit_check.json" | tr -d '\n ' | cut -c1-300)"
done
for D in R L24e; do                                 # 3. the transferred models at the external anchors
  if ! ok "a16x_bl_$D"; then
    bash tools/run.sh "a16x_bl_$D" env "${CENV[@]}" OMP_NUM_THREADS=4 "$PY" "$C/tools/paper_baselines.py" \
      --design-dir "$O/ext_${D}_design" --train "$S/real_train" --val "$S/real_dev" --test "$S/ext_test" \
      --out "$O/ext_$D" --skip-a4
    log "baselines $D exit $(cat "$L/a16x_bl_$D.exit")"
  fi
done
until grep -q " done$" "$L/paper_a16_L48x.log" 2>/dev/null; do sleep 120; done   # 4. the GPU is free
for D in R L24e; do
  case $D in
    R) progs=(decompensation@1440 vasopressor_onset@1440 aki_2@2880 hyperkalemia_top@1440 thrombocytopenia@2880) ;;
    L24e) progs=(icu_death@-1 hospital_death@0 remaining_los_gt_7d decompensation@1440) ;;
  esac
  if ! ok "a16x_tabpfn_$D"; then
    bash tools/run.sh "a16x_tabpfn_$D" env "${CENV[@]}" PYTHONPATH="$C/src:$BASE/tabpfn-pkgs" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 "$PY" "$C/tools/paper_tabpfn.py" \
      --design-dir "$O/ext_${D}_design" --train "$S/real_train" --test "$S/ext_test" --out "$O/ext_$D" \
      --model-path "$CK" --name tabpfn_v3_5 --all-features --context 50000 --programs "${progs[@]}"
    log "tabpfn $D exit $(cat "$L/a16x_tabpfn_$D.exit")"
  fi
done
log "done"
