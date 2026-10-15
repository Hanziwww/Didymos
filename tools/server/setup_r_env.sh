#!/usr/bin/env bash
# The clinician study's pinned R environment (protocol section 8: lme4 in a pinned environment), on the approved
# host, where the responses are. tools/clinician_study_models.R runs with it:
#   /root/autodl-tmp/r-env/bin/Rscript tools/clinician_study_models.R runs/toolkit/clinician/analysis
# Built 2026-10-10 on the 4080 host (westc): R 4.3.3, lme4 1.1-35.5, lmerTest 3.1-3, Matrix 1.6-5, jsonlite 2.0.0.
set -euo pipefail
ENV=${1:-/root/autodl-tmp/r-env}
/root/miniconda3/bin/conda create -y -p "$ENV" --override-channels -c conda-forge \
  "r-base=4.3.3" "r-lme4=1.1_35.5" "r-lmertest=3.1_3" "r-matrix=1.6_5" "r-jsonlite=2.0.0" "r-boot=1.3_32"
"$ENV/bin/Rscript" -e 'for (p in c("lme4", "lmerTest", "Matrix", "jsonlite")) cat(p, as.character(packageVersion(p)), "\n")'
