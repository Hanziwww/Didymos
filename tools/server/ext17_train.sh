#!/usr/bin/env bash
# icutwin on the RTX 5090 host (westd): Paper P amendment 17b's adapted model (docs/p1_paper_protocol.md), in order:
#   1. the training root data/tokens/adapt_r4: the FT shards (tools/paper_ext17.py trainroot, uploaded) with the seven
#      databases' shards linked from full_r4 beside them (vocabulary files checked against full_r4, shards against
#      runs/paper/ext17/adapt.json);
#   2. MUC32_main's next-token loss on adapt_r4's tuning split (the held-out FT patients and the seven databases'
#      tuning split): the loss before adaptation (tools/server/ext17_eval.py);
#   3. the grid, four runs from MUC32_main's weights (train.init_from, fresh optimizer): learning rate {3e-5, 1e-4} x
#      {1, 3} passes over the FT pool, FT mixed 1:1 by windows with the seven databases' train windows (adapt.json's
#      target passes and steps), MUC32_main's recipe otherwise (Muon, its transforms, no dropout), warm-up 2%, cosine
#      to 10%, 131,072 tokens per step on the one GPU (32,768 x 4: MUC32_main's 65,536-token micro-batch ran
#      out of the 5090's 32 GB), seed 17; each run's held-out FT loss (ext17_eval.py, ricd and
#      zigong pooled) and its final evaluation on the whole tuning split (forgetting);
#   4. the choice: the cell with the lowest held-out FT loss (runs/paper/ext17/grid.json);
#   5. the chosen run frozen (freeze_run.py), the membership check of FT against TEST stays (eval.cli step=p1 on
#      data/tokens/adapt_r4_member, for the chosen run and for MUC32_main as the reference), and the M7 gate on the
#      seven databases (m7_gate.sh on the one GPU, full_r4 tokens).
# Logs: logs/e17_*.{log,exit}, logs/ext17_train.log; the gate's logs/g_<run>_*.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
PY=$R/.venv/bin/python
T4=$R/data/tokens/full_r4
A=$R/data/tokens/adapt_r4
M=$R/data/tokens/adapt_r4_member
P=$R/runs/paper/ext17
HF0=$R/runs/train/full/MUC32_main/final/hf
SEVEN="mimic aumc eicu hirid sicdb nwicu inspire"
DS="[mimic,aumc,eicu,hirid,sicdb,nwicu,inspire,ricd,zigong]"
cd "$BASE" || exit 1
mkdir -p "$P/eval" "$L"
log() { echo "$(date '+%F %T') $*" >> "$L/ext17_train.log"; }
ok() { [ "$(cut -d' ' -f1 "$L/$1.exit" 2>/dev/null)" = 0 ]; }
log "start"
for d in $SEVEN; do ln -sfn "$T4/$d" "$A/$d"; done
for f in vocab.json vocabulary.parquet support.json rank_persistence.json goal_bindings.json los_bands.json \
         calibration.json calibration_events.json; do
  for root in "$A" "$M"; do cmp -s "$T4/$f" "$root/$f" || { log "$root/$f differs from full_r4"; exit 1; }; done
done
"$PY" - "$A" "$P/adapt.json" <<'EOF' || { log "adapt_r4 shards differ from adapt.json"; exit 1; }
import hashlib
import json
import sys
from pathlib import Path

root, rep = Path(sys.argv[1]), json.load(open(sys.argv[2]))
bad = [k for k, x in rep["sizes"].items() if "sha256" in x
       and hashlib.sha256((root / f"{k}.parquet").read_bytes()).hexdigest() != x["sha256"]]
print({"checked": sum("sha256" in x for x in rep["sizes"].values()), "differ": bad})
sys.exit(1 if bad else 0)
EOF
ev() {   # <step> <hf dir> <out json> [datasets...]
  local step=$1 hf=$2 out=$3 extra=()
  shift 3
  [ $# -gt 0 ] && extra=(--datasets "$@")
  ok "$step" && return 0
  bash tools/run.sh "$step" env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" \
    "$R/tools/server/ext17_eval.py" --hf "$hf" --tokens "$A" --split tuning --out "$out" "${extra[@]}"
  log "$step exit $(cat "$L/$step.exit")"
  ok "$step"
}
ev e17_base "$HF0" "$P/eval/base.json" || exit 1
ev e17_base_ft "$HF0" "$P/eval/base_ft.json" ricd zigong || exit 1
mapfile -t CELLS < <("$PY" -c "
import json
for k, c in json.load(open('$P/adapt.json'))['cells'].items():
    print(k, c['lr'], c['ft_passes'], repr(c['passes']['default']), c['max_steps'], c['warmup_steps'])")
[ ${#CELLS[@]} = 4 ] || { log "expected four grid cells"; exit 1; }
for line in "${CELLS[@]}"; do
  read -r cell lr e q steps wu <<< "$line"
  RUN=E17_$cell
  if ! ok "e17_$cell"; then
    rm -rf "$R/runs/train/full/$RUN"
    bash tools/run.sh "e17_$cell" env CUDA_VISIBLE_DEVICES=0 ICUTWIN_TMP="$R/.tmp/train_$RUN" "$PY" -m icutwin.model.cli \
      task=train mode=full "tokens_root=$A" "datasets=$DS" seed=17 "train.name=$RUN" "train.init_from=$HF0" \
      "train.lr=$lr" "train.max_steps=$steps" "train.warmup_steps=$wu" train.min_lr_ratio=0.1 train.preset=M \
      "~train.passes" "+train.passes={default:$q,ricd:$e,zigong:$e}" "~train.transforms" \
      "+train.transforms={goal_fraction:0.25,shuffle_header:true,dataset_unknown:0.1,header_unknown:0.05,goal_subsets:true}" \
      train.optimizer=muon train.dropout=0.0 train.ema_decay=null train.input_noise=0.0 train.tokens_per_batch=32768 \
      train.grad_accum=4 train.eval_every=0 train.checkpoint_every=0 train.compile=true train.attn_implementation=varlen \
      train.eval_split=tuning "hydra.run.dir=$R/runs/hydra/train/$RUN"
    log "$RUN exit $(cat "$L/e17_$cell.exit")"
    ok "e17_$cell" || exit 1
  fi
  ev "e17_${cell}_ft" "$R/runs/train/full/$RUN/final/hf" "$P/eval/${cell}_ft.json" ricd zigong || exit 1
done
"$PY" - "$P" "$R/runs/train/full" <<'EOF' || { log "choice failed"; exit 1; }
import json
import sys
from pathlib import Path

p, runs = Path(sys.argv[1]), Path(sys.argv[2])
cells = json.load(open(p / "adapt.json"))["cells"]
base, base_ft = (json.load(open(p / "eval" / f)) for f in ("base.json", "base_ft.json"))
out = {"criterion": "next-token loss on the held-out 10% of FT patients (ricd and zigong pooled)",
       "before": {"ft_nll": base_ft["eval"]["nll"], "by_dataset": base["eval"]["by_dataset"]}, "cells": {}}
for k, c in cells.items():
    ft = json.load(open(p / "eval" / f"{k}_ft.json"))["eval"]
    rep = json.load(open(runs / f"E17_{k}" / "train_report.json"))
    by = rep["final_eval"]["by_dataset"]
    out["cells"][k] = {"run": f"E17_{k}", "lr": c["lr"], "ft_passes": c["ft_passes"], "steps": rep["steps"],
                       "tokens": rep["tokens"], "seconds": rep["seconds"], "ft_nll": ft["nll"], "by_dataset": by,
                       "forgetting": {d: by[d] - out["before"]["by_dataset"][d] for d in by}}
best = min(out["cells"], key=lambda k: out["cells"][k]["ft_nll"])
out["chosen"] = out["cells"][best]["run"]
(p / "grid.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: round(x["ft_nll"], 4) for k, x in out["cells"].items()} | {"before": out["before"]["ft_nll"],
                                                                                 "chosen": out["chosen"]}))
EOF
RUN=$("$PY" -c "import json; print(json.load(open('$P/grid.json'))['chosen'])")
log "chosen $RUN"
if [ ! -e "$R/runs/train/full/$RUN/FROZEN.json" ]; then
  bash tools/run.sh e17_freeze "$PY" "$R/tools/server/freeze_run.py" "$RUN" "$A" \
    "amendment 17 adapted model (Didymos-ext), frozen for the M7 gate: no further training; no deliverable data"
  log "freeze exit $(cat "$L/e17_freeze.exit")"
  ok e17_freeze || exit 1
fi
for who in "$RUN" MUC32_main; do                    # membership: FT train stays (members) against TEST stays
  if ! ok "e17_member_$who"; then
    bash tools/run.sh "e17_member_$who" "$PY" -m icutwin.eval.cli mode=full "run=$who" "tokens_root=$M" \
      "datasets=[ricd,zigong]" "out_dir=$P/membership/$who" step=p1 gpu=0 tokens_per_batch=32768 \
      "hydra.run.dir=$R/runs/hydra/eval/e17_member_$who"
    log "membership $who exit $(cat "$L/e17_member_$who.exit")"
  fi
done
if [ "$("$PY" -c "import json; print(json.load(open('$R/runs/audit/m7_$RUN/report.json')).get('gate'))" 2>/dev/null)" != PASS ]; then
  GPUS=0 TOKENS=$T4 EVAL_ARGS=tokens_per_batch=32768 bash "$R/tools/server/m7_gate.sh" "$RUN" >> "$L/ext17_train.log" 2>&1
fi
log "gate exit $(cat "$L/g_${RUN}_gate.exit" 2>/dev/null): $(cat "$R/runs/audit/m7_$RUN/report.json" 2>/dev/null | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("gate"))')"
log "done"
