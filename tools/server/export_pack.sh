#!/usr/bin/env bash
# icutwin on the AutoDL host: packages for the study lead to download before the server is shut down (2026-10-06), into
# /root/autodl-tmp/export, with SHA-256 sums and contents lists. Paths inside are relative to the repository root,
# so `tar -xf <package> -C <repository>` restores them in place.
#  1. icutwin_essentials_20261006.tar (about 1 GB): the accepted model MUC32_main and the earlier candidate M32_main
#     (final weights and records), every run's records (no other weights), the evaluation and audit aggregates, the
#     r3 and r4 vocabularies with their aggregates (support, calibration, goal bindings, length bands), the mini twin
#     bank and the server logs.
#  2. icutwin_data_20261006.tar (about 4.3 GB): the r4 token shards, the full MEDS and the derived concepts.
# DUA-derived content (credentialed data, models, synthetic twins): keep on D: or approved hosts only.
set -e
B=/root/autodl-tmp
R=$B/icutwin
X=$B/export
mkdir -p "$X"
cd "$R"
records=""
for r in runs/train/full/*/; do
  for f in FROZEN.json config.json metrics.jsonl train_report.json; do [ -e "$r$f" ] && records="$records $r$f"; done
done
meta=""
for t in full_r4 full_r3; do
  for f in vocab.json vocabulary.parquet support.json goal_bindings.json calibration.json calibration_events.json \
           los_bands.json index.json; do [ -e "data/tokens/$t/$f" ] && meta="$meta data/tokens/$t/$f"; done
done
rm -rf "$X/server_logs" && mkdir -p "$X/server_logs" && cp -r "$B/logs" "$X/server_logs/logs_2026-10-06"
tar -cf "$X/icutwin_essentials_20261006.tar" $records runs/train/full/MUC32_main/final/hf \
  runs/train/full/M32_main/final/hf runs/eval runs/audit data/qa $meta data/bank/MUC32_main_mini \
  -C "$X" server_logs
rm -rf "$X/server_logs"
tar -cf "$X/icutwin_data_20261006.tar" data/tokens/full_r4 data/meds/full data/derived/full
cd "$X"
sha256sum icutwin_essentials_20261006.tar icutwin_data_20261006.tar > SHA256SUMS
for p in icutwin_essentials_20261006 icutwin_data_20261006; do tar -tvf "$p.tar" > "$p.CONTENTS.txt"; done
ls -la "$X"
cat SHA256SUMS
