# Didymos

Code for the article *A generative digital twin forecasts and simulates intensive care patients without task labels*.

Didymos is a generative model of intensive care stays. It reads a patient's ICU record as a stream of event tokens
and samples possible continuations of that stream. Forecasts, what-if comparisons and synthetic cohorts are all read
from the sampled futures by one label engine, without task-specific training. Didymos 1.0 is a decoder with 123
million parameters (Qwen3 architecture, 32,768-token context). It was trained on seven ICU databases harmonized to
CLIF 3.0 and MEDS v0.4.

This repository holds the study's code and records:
- data harmonization, tokenization and training;
- grammar-constrained sampling and the privacy audit;
- evaluation and statistics;
- the figures and their source data;
- the Didymos toolkit.

It holds no patient data and no model weights.

## Contents

| Path | What it holds |
|---|---|
| `src/icutwin/stage`, `clif`, `meds` | Staging of the raw archives, source adapters, harmonization to CLIF 3.0 and QA. MEDS v0.4 and the derived concepts (SOFA, KDIGO, norepinephrine-equivalent dose, ventilation) |
| `src/icutwin/tokenization` | The vocabulary, the value bins, the token grammar (a finite-state automaton) and the encoder |
| `src/icutwin/model` | The model configuration, the data pipeline and training (Transformers and Accelerate) |
| `src/icutwin/generate` | Grammar-constrained sampling (Transformers and vLLM), the twin engine, and the privacy-gate interlock |
| `src/icutwin/eval` | Forecasting, calibration, step hazards, fidelity and authenticity, and the membership-inference privacy audit |
| `src/icutwin/toolkit` | The Didymos toolkit: Forecast, What-if, Explain, Similar patients, Cohort synthesizer, Trial designer and Teaching cases |
| `src/icutwin/factory`, `src/icutwin/icl` | An in-context-learning episode factory and its model. This is development code beyond the article |
| `src/icutwin/configs` | Hydra configurations, including the token grammar (`grammar/spec.yaml`) |
| `tools/` | The article's analyses (`paper_*.py`), statistics, tables and figures. Also the clinician study's tools and the factory's tools |
| `tools/server/` | The job scripts as they were run on the study's GPU hosts (see `tools/server/README.md`) |
| `tests/` | The test suite (synthetic fixtures, no patient data) |
| `docs/` | The method and pipeline records, the study protocol with its dated amendments, the results, the figures and the tables |
| `patches/CLIF-MIMIC/` | Patches to the pinned CLIF-MIMIC ETL |
| `refs/` | The reference implementations the pipeline is checked against, pinned by commit (`REFS.lock`). They are not included |

The code and the documents use some internal names:
- *Paper P* is the article.
- `MUC32_main` is Didymos 1.0.
- The *study lead* took the decisions recorded in the protocol.
- M4 to M11 and the plan's phases are the project's milestones.

## Installation

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). `uv.lock` pins every dependency, including PyTorch 2.13.0
with CUDA 13.

```bash
uv sync --group dev
```

Optional extras:
- `serve`: vLLM, for high-throughput sampling (Linux).
- `icl`: TabICL, an in-context comparator.
- `oracle`: xgrammar, used by the grammar's differential tests.

The study sampled with vLLM 0.31.0 in a second environment. `tools/server/setup_env_5090.sh` is that environment's
recipe; the toolkit finds it through `ICUTWIN_VLLM_PYTHON`.

Run commands from the repository root. Paths derive from `ICUTWIN_ROOT`, which defaults to the current directory:

| Directory | Holds | Override |
|---|---|---|
| `dataset/` | Raw archives | `ICUTWIN_RAW_ARCHIVES` |
| `data/` | The pipeline's tables and tokens | `ICUTWIN_DATA` |
| `runs/` | Runs and their aggregates | `ICUTWIN_RUNS` |
| `refs/` | Reference checkouts | `ICUTWIN_REFS` |

The defaults are set in `src/icutwin/configs/paths.yaml`.

## Data

The seven development databases are MIMIC-IV, eICU-CRD, HiRID, AmsterdamUMCdb, SICdb, NWICU and INSPIRE. Two
external databases were used in the transfer and adaptation analyses: zigongICU and RICD.

Credentialed researchers can obtain the databases under their data-use agreements:
- MIMIC-IV, eICU-CRD, HiRID, SICdb, NWICU, INSPIRE and zigongICU through PhysioNet;
- AmsterdamUMCdb through Amsterdam Medical Data Science;
- RICD from its provider.

None of these data are in this repository. `docs/paper/data.md` gives the versions, and `docs/meds.md` and
`docs/paper/preprocessing.md` describe the processing.

## The pipeline

Each step runs per dataset. `mode=subset` runs a 2% patient subset; `mode=full` runs everything.

```bash
uv run python -m icutwin.stage.cli task=stage
```

```bash
uv run python -m icutwin.clif.cli task=pipeline
```

```bash
uv run python -m icutwin.meds.cli task=build
```

```bash
uv run python -m icutwin.meds.cli task=derived
```

```bash
uv run python -m icutwin.tokenization.cli task=fit
```

```bash
uv run python -m icutwin.tokenization.cli task=encode
```

These steps are:
- `icutwin.stage.cli task=stage`: S1, from the archives to native layouts.
- `icutwin.clif.cli task=pipeline`: S2, CLIF 2.1 tables, harmonization and the S3 gate.
- `icutwin.meds.cli task=build`: S4, MEDS v0.4, with one ICU stay as the subject.
- `icutwin.meds.cli task=derived`: SOFA, KDIGO, norepinephrine-equivalent dose, ventilation episodes and fluid balance.
- `icutwin.tokenization.cli task=fit`: S5, the vocabulary and the value bins (train split).
- `icutwin.tokenization.cli task=encode`: token shards per dataset and split.

The checks along the way are:
- `icutwin.clif.cli task=qa`: the cross-dataset QA report.
- `icutwin.clif.cli task=community` and `task=reconcile`: independent community MEDS ETLs, and the reconciliation
  with them (`docs/community_etls.md`).
- `icutwin.meds.cli task=parity`: the ELF engine against CLIF_MEDS.
- `icutwin.meds.cli task=mimic_code`: the derived concepts against mimic-code.
- `icutwin.tokenization.cli task=validate`: the token acceptance and round trip.

Training runs on GPUs with Accelerate:

```bash
uv run accelerate launch --config_file src/icutwin/configs/accelerate/ddp.yaml -m icutwin.model.cli task=train mode=full
```

The records of Didymos 1.0's training are:
- its recipe: `docs/paper/training.md`;
- the candidates and the rules that chose it: `docs/m5_training.md`, `docs/m7_authenticity.md` and
  `docs/m7_refinement.md`;
- the server run: `tools/server/round5_frontier.sh`.

## The privacy gate

Every sampler refuses a model whose privacy audit has not passed. The audit (M7, `docs/m7_privacy.md`) has five
tests, P1 to P5: loss-based membership inference, prefix-prompted extraction, exact copies and distances to the
closest record, DOMIAS and attribute inference. Each is reported per dataset and pooled.
`generate.producer.require_gate` checks two conditions before any sampling:
- the audit report says PASS;
- the audited weights' sha256 equals the loaded model's.

## The toolkit

The toolkit runs the validated capabilities on a patient or a cohort. Each output records the evidence it rests on
and its caveats (`docs/toolkit.md`).

```python
from icutwin.toolkit import Session

s = Session()
p = s.patient("split:mimic/held_out/mimic:<stay_id>")
f = s.forecast(p, 24, mode="fast")
w = s.whatif(p, 24, "norepinephrine")
```

```bash
uv run python -m icutwin.toolkit.cli tool=forecast patient=split:mimic/held_out/mimic:<stay_id> hours=24
```

The toolkit expects the model and its audit where the training run leaves them. Model packages provide them in this
layout:

| Item | Path |
|---|---|
| The model | `runs/train/full/MUC32_main/final/hf` |
| The audit | `runs/audit/m7_MUC32_main` |
| The vocabulary | `data/tokens/full_r4` |

## The article's results

- **Protocol.** `docs/p1_paper_protocol.md` holds the protocol and its amendments 1 to 17. Each amendment is dated and
  was written before its analysis ran.
- **Results.** Start with `docs/paper/results_current.md`. `docs/paper/results.md` and
  `docs/paper/results_followup.md` are generated by `tools/paper_results.py` and `tools/paper_results_followup.py`.
- **Tables.** Every table is also a CSV file under `docs/paper/tables/`.
- **Statistics.** `tools/paper_stats.py` computes the reporting statistics into `docs/paper/tables/manuscript/`.
- **Figures.** The figures are in `docs/paper/figures/` as PDF and SVG files, with their source data (`source_data/`) and
  a `manifest.json` of inputs and checks. They are redrawn with:

  ```bash
  uv run python tools/paper_figures.py --root runs/paper --toolkit runs/toolkit --out docs/paper/figures
  ```
- **GPU jobs.** The aggregate files under `runs/paper/` and `runs/toolkit/` were computed on the GPU hosts by the jobs
  in `tools/server/`. The article's Supplementary Data holds these aggregates.

## Model weights

Didymos 1.0 was trained on data under data-use agreements. Its weights and fitted
tokenizer are hosted at [HanZiwww/Didymos-1.0](https://huggingface.co/HanZiwww/Didymos-1.0)
with manual review of access requests. The model card describes the package and
the applicable access conditions. The weights are not covered by this code repository's Apache-2.0 license.

## Tests

The suite runs on CPU with synthetic fixtures. Tests that need local data copies, a GPU or an optional extra are
skipped.

```bash
uv run pytest
```

## Intended use

Didymos is research software. It is not a medical device and must not be used for clinical decisions. Outputs
computed from real stays, and synthetic stays from a model trained on them, remain covered by the agreements of the
source data.

## Licence

Apache License 2.0 (`LICENSE`). Third-party notices are in `NOTICE`.

## Citation

Use `CITATION.cff` to cite this software release. The article citation will be added at publication.

## Release provenance

This public-release packaging was prepared on 2026-10-15 from the supplied
`release/paper-v1` source snapshot, commit `344652d0dc3c1c0cb6dbfe329e06530fd78d4252`
(`COMMIT.txt`). Packaging adds citation metadata and the model-access link, corrects the README's canonical
CLIF version, declares the statistics tool's `statsmodels` dependency and uses
pytest's default temporary directory so a clean checkout can run the tests.
The model, training, sampling and evaluation implementation was preserved.
