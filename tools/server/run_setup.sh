#!/usr/bin/env bash
# icutwin on the AutoDL host: setup from the uploaded package; everything on /root/autodl-tmp (the root disk is 30 GB).
set -euo pipefail
BASE=/root/autodl-tmp
ROOT=$BASE/icutwin
export PATH="$BASE/tools/bin:$PATH:/root/miniconda3/bin"
# the raw archives sit directly under $BASE: a dataset/ view of them (no symlink cycle through the repository)
mkdir -p "$BASE/dataset"
for d in HIRDI INSPIRE NWICU SICDB amsterdamUMCdb eICU mimic; do
  [ -e "$BASE/dataset/$d" ] || ln -s "$BASE/$d" "$BASE/dataset/$d"
done
export TMPDIR=$BASE/tmp
mkdir -p "$TMPDIR"
bash "$BASE/upload/server_setup.sh" "$BASE/upload" "$ROOT" "$BASE/dataset"
# caches, temp files and tools off the root disk, for every later shell
cat >> "$ROOT/.cache/server.env" <<EOF
export PATH="$BASE/tools/bin:\$PATH:/root/miniconda3/bin"
export XDG_CACHE_HOME="$ROOT/.cache/xdg" TRITON_CACHE_DIR="$ROOT/.cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/inductor"
export TMPDIR="$ROOT/.tmp/tmp" ICUTWIN_TMP="$ROOT/.tmp"
EOF
mkdir -p "$ROOT/.tmp/tmp"
echo "SETUP_WRAPPER_DONE"
