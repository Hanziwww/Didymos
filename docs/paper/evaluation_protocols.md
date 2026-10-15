# Materials: evaluation protocols

Every protocol was written and committed before the measurement it governs; changes are dated amendments
(`docs/m7_privacy.md`, `docs/m7_authenticity.md`, `docs/m7_refinement.md`, `docs/p1_paper_protocol.md`).

## The M7 privacy gate (protocol v1.1 and amendment v2, `docs/m7_privacy.md`)

Statuses: PASS when the criterion holds with 95% confidence, FAIL when it is violated with 95% confidence,
INCONCLUSIVE otherwise; the gate passes only when every test passes. Bank generation is interlocked on a PASS for the
exact weights (`generate.producer.require_gate`).

| Test | Attack | Sets | PASS |
|---|---|---|---|
| P1 loss-based membership inference (Yeom 2018; TPR at low FPR, Carlini 2022) | a stay's mean NLL per target token | up to 3,000 train and 3,000 held-out stays per dataset | AUC upper bound <= 0.60 and TPR at 1% FPR upper bound <= 0.03, per dataset and pooled |
| P2 prefix-prompted extraction (Carlini 2021) | greedy continuation of a first window's head and first k timeline tokens; exact match of L tokens | up to 1,000 train and 1,000 held-out first windows; (k, L) = (32, 32), (128, 64) | member minus non-member exact-match rate upper bound <= 0.01 |
| P3 exact copies, DCR, NNDR | TF-IDF of a window, cosine distance to the nearest train first window | 5,000 audit samples against every train first window, baseline 5,000 held-out | copies minus held-out <= 0.001; share below the held-out 5th percentile of DCR (and of NNDR) <= 0.07 |
| P4 DOMIAS (van Breugel 2023) | density ratio of samples against a reference, in a 32-d SVD of P3's representation | 5,000 train members, held-out non-members and reference | AUC upper bound <= 0.60 |
| P5 attribute inference (amendment v2) | in-hospital death inferred from the nearest audit sample by head and first 32 timeline tokens | single-window members and non-members | member minus non-member balanced accuracy upper bound <= 0.02 |

Controls: a private reference (real stays the model never saw) must PASS P3-P5; a copying generator must FAIL them.

## Authenticity A1-A7 (pre-registered 2026-10-05, `docs/m7_authenticity.md`)

1,000 twins per dataset against 1,000 held-out stays; the positive control is the same comparison with real
training stays; criteria read on the mean of two replicates (twin seeds 0 and 1), within 3% of a threshold counting
as met (amendment 6, the study lead's principle).

| # | Measure | Target |
|---|---|---|
| A1 | TF-IDF discriminator AUC, first 24 hours after ICU admission | <= 0.70 |
| A2 | Discriminator AUC, whole stay, length-matched | <= 0.80 |
| A3 | Discriminator AUC per family view (vitals, labs, medications, respiratory, timing only) | <= 0.80 each |
| A4 | Events per stay, 90th and 99th percentiles | within +-20% of real |
| A5 | Count of a stay's most-repeated event, 99th percentile | within +-25% of real |
| A6 | Vendi ratio (pooled); rare-code recall | >= 0.95; >= positive control - 0.03 |
| A7 | The harness's other metrics; utility ratio per task; grammar validity; the privacy gate | in band; >= 0.90; 100%; PASS |

## Calibration targets of round 3 (`docs/m7_refinement.md`)

| # | Measure | Target |
|---|---|---|
| S1 | New IMV, vasopressor and RRT after the 24-hour landmark, twins against held-out stays, per dataset | within +-25% |
| S2 | Rollout observed over expected (death, IMV, vasopressor, RRT, ICU stay over 7 days), 300 prefixes x 16 rollouts per dataset | 0.8-1.25 |
| S3 | Tuning NLL | at most the reference's + 0.01 |
| Teacher-forced hazard calibration (diagnostic) | the model's probability mass on an outcome's tokens over the positions at risk after 24 h, against the observed onsets | reported by ICU day |

## The harness (`docs/m7_harness.md`)

Fidelity (per-code marginals: code and value JSD; events per stay and ICU stay KS; clinical rules such as blood-pressure
ordering), diversity (Naeem precision/recall/density/coverage, Vendi ratio, rare-code recall, duplicates and copy
rate, strata and field total variation, dataset-classifier confusion, inter-dataset MMD correlation, eICU
hospital-level total variation), utility (TSTR against TRTR at the 24-hour landmark), forecasting (the CoMET protocol:
rollouts from 24-hour prefixes, AUROC, ECE, observed over expected, PIT and spread-skill), probes, task realism.
Self-tested: the positive control (train against held-out) scores near-ideal; every negative control (shuffled times,
shuffled values, dataset-label swap, collapsed generator, copying generator) is flagged.

## Paper P's supplementary experiments (`docs/p1_paper_protocol.md`, amendments 1-2)

| Experiment | Question | Design in one line |
|---|---|---|
| A1 | Can the twin predict real patients' futures by sampling, zero-shot? | 32 rollouts from real held-out prefixes (24 h, 48 h landmarks; rolling times), the factory's 72 catalog programs as labels, against LightGBM, logistic regression, SOFA and a probe on the model's states |
| A2 | Are those predictions calibrated, and where does exposure bias enter? | O/E, slope, ECE by horizon and ICU day; teacher-forced against rollout O/E; PIT and interval coverage |
| A3 | Fidelity and privacy, and the fit-privacy frontier | the gate's tables; P1 AUC against held-out NLL over the checkpoints |
| A4 | Are twins useful training data? | TSTR/TRTR, cross-dataset, small-site augmentation |
| A5 | Does it hold across subgroups? | sex, age, dataset, admission type, race where recorded |
| B2 | What happens without the dataset token? | unknown-site prompts against the true site |
| B3 | What did each design change buy? | tuning NLL, P1, teacher-forced O/E and probes across the checkpoints |
| C1 | Do what-if rollouts respond physiologically? | give against withhold a treatment at a plausible time; pre-specified signs |
| C2 | Which events drive an individual's predicted risk? | counterfactual editing as the reference; integrated gradients and attention, their faithfulness and agreement |
| C3 | Do conditioned cohorts reproduce each site? | profiles per dataset, inter-dataset structure, eICU hospitals, rare strata |
| C4 | Does the model's state encode clinical similarity? | outcome concordance of nearest neighbours |

Follow-ups F1-F9 (amendments 7-8), the attention diagnosis (amendment 9) and the comparators (amendment 10) are in
the protocol; their results are in `docs/paper/results_followup.md`.

## The comparators (A1 and amendment 10)

Every comparator is scored on the same held-out anchors as the Twin and at the same prediction time. Learned models
see only data up to that time; clinical scores see only data recorded or resulted by then.

**Learned models.** Trained on real training stays' rows of the design (L24, L48: the landmark of every stay still
in the ICU; R: up to 4 random ICU hours per stay).
- **Features (A1, `eval.baselines.Featurizer`):**
  - for the 64 most charted numeric channels: the last value, mean, minimum, maximum and count over the 6 and 24
    hours before the prediction time, and the hours since the last value;
  - the treatment states (IMV, NIPPV, vasoactives, RRT);
  - the statics documented at ICU admission (age, sex, unit type, admission type and source), the dataset, and the
    hours since ICU admission.
  - The values are those the Twin sees, decoded from the tokens.
- **Tuning store:** settings are chosen, and LightGBM is early-stopped, on rows of the tuning split.
- **Preprocessing:** every model except LightGBM gets median imputation, missing-value indicators and
  standardization, fitted on the training rows.

| Family | Model | Settings |
|---|---|---|
| Gradient boosting | LightGBM, pooled (the dataset a feature) and per dataset (A1) | 1,000 trees at most, learning rate 0.05, 31 leaves, early stopping |
| Tabular foundation models | TabPFN-3.5, TabPFN v2 (A1, local weights) | a stratified training sample |
| Linear | logistic regression with an L2 (A1), L1 (LASSO, liblinear) or elastic-net (l1 ratio 0.5, saga) penalty | C in {0.01, 0.1, 1} for L1 and elastic net |
| Linear, on the model's states | the Twin probe (A1) | L2 logistic regression on the frozen last and mean states |
| Other trees | decision tree (CART); random forest; extra trees | depth 3, 5 or 8, at least 50 rows per leaf; 300 trees, square-root features per split, at least 5 rows per leaf |
| Other learners | Gaussian naive Bayes; k-nearest neighbours; multilayer perceptron | -; k 25 or 100, distance-weighted; hidden layers of 128 and 64, early stopping |

**Clinical scores (amendment 10, `eval.clinical`, `tools/paper_clinical.py`).**
- **Source:** the canonical CLIF tables of the anchor's stay, which hold the harmonized raw values (not the
  tokens).
- **Window:** the 24 hours before the prediction time, counting only data recorded or resulted by then (labs at
  their result time).
- **Missing components:** a component without a value scores 0, as mimic-code and ricu do.
- **Coverage** of every input is reported per dataset. Examples: NWICU and SICdb record no GCS; NWICU records no
  urine output or blood gases.

| Score | Values used | Reference implementation |
|---|---|---|
| SOFA | worst in the window | clifpy `compute_sofa_polars`, with the respiration correction of `meds.derived` |
| SAPS II (landmarks; without its chronic-disease items, which only discharge diagnoses record) | worst in the window | mimic-code `sapsii`, with its probability equation |
| OASIS (landmarks) | worst in the window | mimic-code `oasis`, with its probability equation |
| NEWS (NEWS2, SpO2 scale 1), MEWS, qSOFA, SIRS | the latest value of each parameter in the window | ricu `news_score`, `mews_score`, `qsofa_score`, `sirs_score` (AVPU from the GCS as ricu's `avpu`) |
| APACHE IVa (eICU), SAPS 3 (SICdb), ASA class (INSPIRE) | the dataset's own admission score | eICU `apachePatientResult` (predicted ICU and hospital mortality, ICU stay); SICdb, INSPIRE as recorded |

**Bedside indicators.** For the rolling endpoints, the latest value of a single clinical variable; a missing value
is set to normal.

| Endpoint | Indicator |
|---|---|
| Death within 24 h | shock index; lactate |
| New IMV | ROX index |
| New vasopressor | shock index; MAP |
| New RRT, AKI | KDIGO stage (creatinine against its 48-hour and 7-day minima, urine-output stage, CRRT); creatinine |
| Hypotension | MAP |
| Hyperkalaemia | potassium |
| Thrombocytopenia | platelets |
| Sepsis | lactate |

## The application toolkit's validation (amendment 11, `docs/toolkit.md`)

The toolkit (`src/icutwin/toolkit`) computes the paper's quantities as functions; its validation fixes what the paper
had not shown. Rollouts are new draws (each request seeded), sampled with vLLM 0.31.0 on an RTX 5090.

| Test | Design | Statistic | Rule |
|---|---|---|---|
| V1 conformance, forecast | F1's R128 anchors (1,050 held-out rolling anchors, inclusion weights), 32 rollouts, A1's caps | AUROC and O/E of A1's nine R primary programs, against A1's 32-rollout Twin on the same anchors (bootstrap over stays within datasets) | within A1's 95% intervals |
| V1 conformance, E2 | the same draws, the toolkit's teacher pass | E2 AUROC at 32 rollouts for F2's four endpoints | within 0.02 of F2's curve at k = 32 |
| V1 conformance, what-if | F5's design rows for norepinephrine (MAP) and furosemide (urine output), 16 rollouts per arm, C1's 24-hour caps | the factual contrast (starters' give minus matched non-starters' withhold); the A/A difference | within F5's 95% interval; the A/A interval covers 0 |
| V2 recalibration | isotonic maps from amendment 1's tuning-split R anchors (32 rollouts; alignment-row labels), applied to V1's draws at 32 and 16 rollouts | O/E, calibration slope and intercept, ECE, Brier, AUROC | AUROC within 0.005 (monotone maps); the rest reported |
| V3 privacy, (a) loss | train members against held-out non-members (at most 500 each per dataset), prompts cut at a uniform whole hour (4 h to the stay minus 25 h) | membership AUC and TPR at 1% FPR of the real next 24 hours' mean NLL given the prompt | P1's rule (PASS: upper bounds at most 0.60 and 0.03; FAIL: a lower bound above 0.60 or 0.05) |
| V3 privacy, (b) extraction | at most 150 each per dataset, 16 rollouts per prompt | any rollout reproducing the real future's first 32 tokens; the best event-multiset Jaccard over 24 h | P2's limit (member minus non-member at most 0.01); AUC upper bound at most 0.60 |
| V3 privacy, (c) intervened | norepinephrine prefixes as C1's (at most 100 each), the give-median arm | as (b) | as (b) |
| V3 controls | a copying generator (members' rollouts replaced by their real futures); a private reference (two halves of the non-members) | as (a) and (b) | the copy must FAIL, the reference PASS |
| V4 trial designer | the kept predictions on L24 (32 rollouts) and L24x (128); relative risks 0.8 and 0.7; 2,000 simulated trials per score | type I error at relative risk 1; the unadjusted simulated size against the analytic | 0.04-0.06; within 5% |
| V5 teaching | three objectives on the bank (shock, respiratory failure, renal support), three typical cases each | meeting the objective; novelty when the reference exists; self-contained pages | all |

**Prognostic covariate adjustment (V4).** A simulated trial draws N anchors with replacement from the eligible
population, randomizes them 1:1 and imposes the effect as a constant relative risk (a treated patient's observed
event kept with probability rr). The unadjusted analysis is the two-proportion z test; the adjusted one a logistic
regression on treatment and the prognostic score (the logit of a probability clipped at 0.5%, a raw score
standardized), tested by the treatment's Wald statistic. The size for the target power under each analysis follows
from the simulated power at the analytic unadjusted size, with the test statistic taken as normal with a mean
proportional to the square root of N.
