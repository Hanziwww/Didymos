#!/usr/bin/env bash
# icutwin on the RTX 5090 host (2026-10-10): the repository's two environments, as on the 4080 host.
#   .venv     uv sync --frozen (uv.lock; Linux torch 2.13.0 is a CUDA 13 build)
#   vllm-env  vLLM 0.31.0 with the 4080 host's versions (torch 2.13.0, transformers 5.17.0, numpy 2.3.5, pyarrow
#             25.0.1) and the packages the vocabulary and engine import (polars, duckdb, scipy, omegaconf), icutwin
#             editable without its dependencies
# Public packages only (PyPI); no data goes through this script. FlashInfer builds its sampling kernels on the first
# engine (ninja from vllm-env/bin, the CUDA toolkit at /usr/local/cuda; the toolkit's stage has vllm-env/bin on PATH).
set -euo pipefail
BASE=/root/autodl-tmp
R=$BASE/icutwin
PY=/root/miniconda3/bin/python3.12
# shellcheck disable=SC1091
. "$R/.cache/server.env"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$BASE/tools/bin" "$BASE/logs"
if [ ! -x "$BASE/tools/bin/uv" ]; then
  "$PY" -m pip install --quiet --target "$BASE/tools/uvpkg" uv
  ln -sf "$BASE/tools/uvpkg/bin/uv" "$BASE/tools/bin/uv"
fi
cd "$R"
uv sync --frozen --python "$PY"
if [ ! -x "$BASE/vllm-env/bin/python" ]; then
  uv venv "$BASE/vllm-env" --python "$PY"
fi
uv pip install --python "$BASE/vllm-env/bin/python" "vllm==0.31.0" "torch==2.13.0" "transformers==5.17.0" \
  "numpy==2.3.5" "pyarrow==25.0.1" "polars==1.44.2" "duckdb==1.5.6" "scipy==1.18.1" "omegaconf==2.3.1"
uv pip install --python "$BASE/vllm-env/bin/python" --no-deps -e "$R"
"$BASE/vllm-env/bin/python" -c "import vllm, torch, icutwin.generate.engine; print('vllm-env', vllm.__version__, torch.__version__)"
