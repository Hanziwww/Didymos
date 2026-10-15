# The ICU digital twin's application toolkit

`src/icutwin/toolkit` turns the capabilities Paper P validated into tools a researcher, a trial designer or a teacher
can run on a patient or a cohort. Every output carries the paper evidence it rests on and its caveats
(`toolkit/evidence.yaml`); the toolkit's own validation is protocol Amendment 11 (`docs/p1_paper_protocol.md`).

| Tool | Question | Evidence | GPU |
|---|---|---|---|
| Forecast | What happens next to this patient: risks of the label engine's programs, E2 incidence curves, value forecasts, the remaining stay | A1, F1, F2, amendment 10; V1, V2 | sampling + teacher pass |
| What-if | The twin's futures if a treatment is given at the prompt's end or withheld, against an A/A band | C1, F5; V1 | sampling |
| Explain | Which parts of the history move the twin's estimate (probe editing, attention knockout) | C2, F4, amendment 9 | teacher pass |
| Similar patients | Outcomes among the nearest held-out stays by the model's state | C4, F7 | teacher pass |
| Cohort synthesizer | Synthetic cohorts on request, as CLIF 3.0 and MEDS, with conformance, fidelity and novelty reports | A3, A4, C3, F6, F8 | none (select) or sampling |
| Trial designer | Event rates, prognostic enrichment and prognostic covariate adjustment from the kept predictions | V4 | none |
| Teaching cases | Synthetic cases at a decision point, with learner and teacher pages and a prognosis drill | C3, F5; V5 | sampling + teacher pass |

## Governance

- **Data stays on the approved hosts.** Anything computed from a real stay (a held-out stay's forecast, its card,
  the similarity index) carries individual data under the data-use agreements. The materials carry aggregates only.
  Paper figures and teaching cases use bank twins (synthetic, internal until cleared).
- **The privacy gate.** Every sampling call checks the M7 gate for these weights (`generate.producer.require_gate`)
  and records its scope in the result. Gate v1 had not assessed prompted or intervened sampling. Amendment 11's V3
  assessed it, and the gate report records it as PASS (the study lead's decision; INSPIRE's test inconclusive,
  sample-limited).
- **What-if is observational simulation** within the twin world, not causal effect estimation. Every what-if output
  says so, with F5's flag for each response (reproduced, not reproduced, unresolved, exploratory).

## Quick start

```python
from icutwin.toolkit import Session

s = Session()                                    # paths from ICUTWIN_ROOT / ICUTWIN_DATA / ICUTWIN_RUNS
p = s.patient("bank:eicu_00000_00001")          # or split:<dataset>/<split>/<stay id>, meds:<dir>::<split>::<id>
f = s.forecast(p, 24, mode="fast")               # 16 rollouts, E2, the label engine
w = s.whatif(p, 24, "norepinephrine")            # give (three doses) / withhold / A/A
e = s.explain(p, 24, "icu_death@-1")             # editing + knockout on the Twin probe
from icutwin.toolkit import card
card.write(card.render(f, s), "forecast.html")   # a self-contained card
```

The command line takes the same arguments (`configs/toolkit.yaml`; console script `icutwin_toolkit`):

```bash
python -m icutwin.toolkit.cli tool=forecast patient=bank:eicu_00000_00001 hours=24 mode=standard
```

```bash
python -m icutwin.toolkit.cli tool=forecast patient=split:mimic/held_out/mimic:<stay_id> hours=[6,12,24,48]
```

```bash
python -m icutwin.toolkit.cli tool=whatif patient=bank:eicu_00000_00001 hours=10 treatment=furosemide
```

```bash
python -m icutwin.toolkit.cli tool=cohort spec=configs/my_cohort.yaml
```

```bash
python -m icutwin.toolkit.cli tool=card result=runs/toolkit/forecast/<job>/result.json
```

Each call writes its job directory under `runs/toolkit/<tool>/` (git-ignored): `job.json` (arguments, provenance), the
requests and rollouts, the stores, `result.json` and, with `card=true`, the HTML cards.

## How it runs

- **Session** (`session.py`): one model with its tokens, gate, token caps (the paper's 95% quantile rule, cached as
  JSON) and sampling backend.
- **Patients and anchors** (`inputs.py`): a stay decoded on its own clock; the anchor at an hour is exactly the
  paper's prompt (`eval.bench.prompt`), with the prediction time where labels are read. A stay of a new site can be
  prompted with its dataset and site unknown (B2/F9), the default for MEDS input.
- **Jobs** (`jobs.py`, `stage.py`): the sampling stage runs `toolkit.stage` in vLLM's environment as a subprocess
  (`ICUTWIN_VLLM_PYTHON`; `engines` lanes started one after another), writing rollouts atomically; a rerun resumes.
  Rollout j of a request has seed `request seed x 1,000,000 + j`, independent of batching. `backend="hf"` samples in
  process with transformers and the grammar's reference processor (tests, small jobs).
- **Teacher passes** (`teacher.py`): the paper's E2 step hazards (F2's masks, clock and stop positions), the model's
  states, the probe and attention knockout; kept per job (`hazards.npz`).
- **Labels** (`labels.py`): rollouts decoded into a store and read by the label engine at the prediction time, as
  A1 read them (alignment rows give a twin's or a held-out stay's own labels when asked).
- **Results** (`schema.py`): JSON with `tool`, `patient`, `anchor`, `settings`, `outputs`, `evidence`, `caveats`,
  `provenance` (weights, gate status and scope, backend, git).
- **Cards** (`card.py`, `figures.py`): inline CSS, SVG and script, no network request; teaching pages in learner,
  teacher and drill modes; a blinded timeline for the clinician study; matplotlib figures for the paper (twins only).

## The tools

### Forecast
Fast mode draws 16 rollouts, standard 128, capped at the 72-hour token quantile (`cap_mult=4` with `cap_hours=168`
for ICU death until discharge, F1). Outputs per program: the share of valid rollouts, E2 where the hazards define it
(death within 24 h, new IMV and vasopressor within 24 h, new RRT within 48 h, ICU death), the risk shown (E2, else
the share; recalibrated when V2's maps exist) and the paper's O/E; the E2 and E0 incidence curves of five causes
hourly to 48 h; value forecasts (10th, 50th, 90th percentiles); the remaining stay (Kaplan-Meier). A list of hours
gives a rolling trajectory. Caveats: raw treatment starts are over-predicted about two-fold; death within 24 h is
under-predicted (generation delays deaths); value intervals are too narrow.

### What-if
Arms: give at C1's record and dose bins (norepinephrine and insulin: low, median, high), withhold, and a second
withhold arm with another seed (the A/A null). A give the grammar rejects (e.g. an infusion already running) is dropped
with its reason. Responses are C1's (channels, windows, aggregates, the urine-output cap): per arm the mean and
spread, contrasts against withhold with Welch intervals, the A/A band, and F5's flag. The faithful use is a prompt
shortly before a plausible start (F5).

### Explain
The estimate explained is the Twin probe of a program (`runs/paper/a1/test_<design>/probe_models.npz`). Editing
removes each event group (a channel family x a time block, plus the statics) and re-encodes the prompt; knockout
blocks attention to the group's tokens in every layer. Optional generator editing re-draws rollouts for the top
groups' edits. Attention rollout is not offered (it fails the randomization check, amendment 9).

### Similar patients
`similar.build_index` stores a design's held-out anchors with the model's mean state and their labels
(`runs/toolkit/index`); a query returns the weighted outcome rates among the k nearest (50 by default), Wilson
intervals, counts under the minimum cell suppressed, never a neighbour. `build_bank_index` adds bank-twin exemplars.

### Cohort synthesizer
A spec (dict or YAML):

```yaml
n: 200
mode: select              # select: the bank (no GPU); sample: new twins under the gate
datasets: {mimic: 0.5, eicu: 0.5}
filters: {vasopressor: "yes", age: [65, null], unit_type: [medical_icu, mixed_icu]}
export: [clif, meds]
seed: 0
```

Sampling conditions the dataset, the site, single-valued header fields and the sex; other filters are applied by
rejection, with the bank's acceptance rate setting the oversampling. Reports: the share meeting each filter, C3's
profile against the real held-out stays meeting the same filters, novelty when a training reference exists. Exports
carry an INTERNAL README.

### Trial designer
```yaml
design: L24               # or L48, L24x (F1's 128-rollout subset)
endpoint: icu_death@-1
population: {datasets: [mimic, eicu], sofa: [2, null], on_vasopressor: true}
effect: {rr: 0.8}
scores: [twin, sofa, sapsii, oasis, news, lgbm_pooled]
simulations: 2000
```

Outputs: the eligible anchors and event rate; the two-proportion sample size; enrichment by each score (top
fractions by rank, the screening burden); prognostic covariate adjustment (simulated power, the size for the target
power, the adjusted-to-unadjusted ratio, the type I error). Simulated effects are constant relative risks on held-out
ICU stays: the tool compares designs, it claims nothing about a treatment.

### Teaching cases
`s.teach(objective, treatment=..., options=[...], cases=3)` picks typical bank twins meeting the objective, sets the
decision point one minute before the twin's own first start of the treatment (F5's faithful case), and writes per
case `case_<k>.json`, a learner page (risk sliders, a choice of action, a reveal against the twin's forecast and its
own course, answers downloaded as JSON), a teacher page (with attributions and discussion prompts) and the prognosis
drill. Not built in this version: the two-step chain (a second decision on a branch).

## Validation (Amendment 11)

| | What | Outcome (2026-10-10; `docs/paper/results_current.md` section 11) | Where |
|---|---|---|---|
| V1 | Conformance: R128 forecast against A1 and F2; F5's norepinephrine and furosemide rows | Conforms under the study lead's criterion (within noise of A1 or F1); the pre-registered rule 14 of 15, the post-hoc noise check 7 of 9 | `tools/server/toolkit_v1.py`, `toolkit_v1_noise.py`, `toolkit_v1_adjusted.py`; `runs/toolkit/v1/` |
| V2 | E2 recalibration from the tuning split, evaluated on V1's draws | Four isotonic maps adopted (death within 24 h, AKI, hypotension, hyperkalaemia); the treatment starts' isotonic maps cost AUROC | `tools/server/toolkit_v2.py`, `runs/toolkit/v2/` |
| V2b | Logistic recalibration of the treatment starts (the study lead's decision) | Three logistic maps adopted (IMV, vasopressor, RRT; AUROC kept; IMV overcorrects, O/E 1.43) | `tools/server/toolkit_v2b.py`, `runs/toolkit/v2/v2b.json` |
| V3 | Privacy of prompted and intervened sampling, with controls; the loss test at P1's sizes (V3x) | Recorded in the gate as PASS (the study lead's decision): pooled loss AUC 0.531, 7 of 8 datasets PASS, INSPIRE inconclusive (sample-limited), no extraction, controls valid | `tools/server/toolkit_v3.py`, `gate_record_v3.py`; `runs/audit/m7_<run>/prompted.json`, `report.json` |
| V4 | Trial simulations on the kept predictions | The Twin as a covariate: 0.63-0.86 of the unadjusted size (clinical scores 0.80-1.00) | `tools/toolkit_v4.py`, `runs/toolkit/v4/v4.json` |
| V5 | Teaching cases: objectives, novelty, self-contained pages; the case figure | Nine cases, all checks met | `tools/server/toolkit_v5.py`, `runs/toolkit/v5/` |

The recalibration in force (`runs/toolkit/recalibration/<run>.json`) holds V2's adopted isotonic maps and V2b's
logistic maps (`"kind": "logistic"`); V2's other isotonic maps are in `runs/toolkit/v2/maps_all.json`. A forecast
marks every risk `calibrated: true` or `false`. Results carry the gate's scope for prompted sampling (a caveat, and
the card's badge), read from the M7 report at the session's start.

## Host notes

- vLLM's environment needs the 4080 host's versions (vLLM 0.31.0, torch 2.13.0, transformers 5.17.0, numpy 2.3.5,
  pyarrow 25.0.1) plus polars, duckdb, scipy and omegaconf (the vocabulary and engine imports), and icutwin installed
  without its dependencies; the session puts that environment's `bin` first on the stage's `PATH` (FlashInfer builds
  its sampling kernels with `ninja` and the CUDA toolkit at `/usr/local/cuda` on the first engine of a new GPU).
- Run the stages one at a time on a shared GPU: vLLM sizes its cache from the free memory at start.
- The label engine's store build needs `refs/CLIF_MEDS` (the pinned commit in `refs/REFS.lock`) and about 12 GB of
  memory per worker.
