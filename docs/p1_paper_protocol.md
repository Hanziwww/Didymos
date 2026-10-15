# Paper P: supplementary experiments, pre-registered protocol (2026-10-08)

The twin model paper (target: *npj Digital Medicine*; plan file, section "Paper P"). This protocol fixes the
experiments' designs, tasks, predictors, baselines, metrics and expected directions **before any score is read**.
Changes after a result is seen are amendments, dated and justified, as in `docs/m7_refinement.md`.

## Scope and rules (study lead, 2026-10-08)

| Rule | Detail |
|---|---|
| Model | MUC32_main as adopted (`docs/m7_refinement.md`), the M7 gate PASS for its weights. No new training of the twin model (inference only); the other checkpoints on the host serve the ablations. |
| Sampler | Rollouts: T = 1, no top-k/top-p, grammar and dataset-support masks (the engine as in S0). Twins for A4 and C3: the internal bank (round-3 sampler: length and treatment raking). |
| Real data | Prompts and labels from the real **held-out** split (`data/factory/stores/real_test`, 2,000 stays per dataset). A dry run on the **tuning** split (`real_dev`) checks the code; its scores are not reported as results. Baselines train on the real **train** split. |
| Outputs | Aggregates only (JSON under `runs/paper/` with the code's git sha); rollouts and twins are evaluation artefacts, deleted after their analysis. No individual real trajectory leaves the host or enters the materials. |
| Seeds | Design seed 0; rollout seed 0 (a replicate seed 1 where stated). |
| Statistics | 95% intervals by a stratified bootstrap over stays (1,000 resamples); paired bootstrap for differences between predictors on the same anchors. Inclusion weights wherever a design is enriched. |

## A1. Prediction by sampling (the main benchmark)

### Anchor sets (prediction times on real held-out stays)

| Set | Anchors | Design | Token cap of a rollout |
|---|---|---|---|
| L24 | up to 500 stays per dataset in the ICU and alive at 24 h | natural sample (seed 0) | the dataset's 7-day token quantile (q 0.95) |
| L48 | up to 500 stays per dataset in the ICU and alive at 48 h | natural sample | 7-day quantile |
| R | 500 per dataset at random ICU hours | `factory.oracle.choose`: spread over LOS bands, ICU day uniform within a stay, half of each band enriched for a new IMV, vasopressor or RRT start or an ICU death in the next 48 h, with inclusion weights | 72-hour quantile |

A prompt is the stay as known at the anchor: `tokenization.encode.prospective` of the stay's decode on its own clock
(`factory.twins.decode_labels(ids, v, stay_id)`, as the real stores are built), header fields documented later
unknown, no goal, no end block. A capped rollout ends on the censored path; a label whose window it does not cover is
not valid for that rollout.

### Tasks: the factory's clinical catalog (72 programs, `factory.programs.catalog`)

Every program is evaluated by the same label engine on the real stay's own future (the label) and on each decoded
rollout (prompt + continuation on the stay's clock, derived concepts by `clif_out.derived`), at the anchor's
prediction time (the prompt's last event). Eligibility and validity as the programs define them.

- L24 and L48: every program except the event-triggered ones (extubation failure); the landmark outcomes (ICU death,
  in-hospital death, remaining ICU stay and its band) are evaluated here.
- R: binary programs with a horizon of at most 48 hours, times to event (IMV, vasopressor, death), regressions with
  windows of at most 48 hours.
- **Primary endpoints** (the headline table; the rest are secondary):

| Endpoint | Program | Set |
|---|---|---|
| ICU death | `icu_death@-1` | L24 |
| In-hospital death | `hospital_death` | L24 |
| Remaining ICU stay over 7 days | `remaining_los_band` (top bands) | L24 |
| Decompensation, 24 h | `decompensation@1440` | R |
| New IMV, 24 h | `imv_onset@1440` | R |
| New vasopressor, 24 h | `vasopressor_onset@1440` | R |
| New RRT, 48 h | `rrt_onset@2880` | R |
| AKI stage 2 or more, 48 h | `aki_2@2880` | R |
| Sepsis (ASE), 24 h | `sepsis_ase@1440` | R |
| Hypotension, 4 h | `hypotension@240` | R |
| Hyperkalaemia, 24 h (amendment 1: the bin-edge variant) | `hyperkalemia_top@1440` | R |
| Thrombocytopenia, 48 h | `thrombocytopenia@2880` | R |

### Predictors

| Name | What | Trained on real labels? |
|---|---|---|
| **Twin (rollouts)** | p̂ = share of positive valid rollouts among 32 (at least 16 valid, else not scored); regression: the rollouts' ensemble; time to event: the rollouts' event-time distribution | no (zero-shot) |
| Twin probe | logistic regression (ridge, C by 5-fold CV) on MUC32_main's last-layer state at the prompt's end (last token and mean of the last window: `eval.probes.embed`), per task | yes (train split) |
| LightGBM | per task, pooled over datasets with the dataset as a feature, and per dataset; features below; 1,000 trees max, learning rate 0.05, 31 leaves, early stopping on the tuning split | yes |
| Logistic regression | the same features, standardized, missing-indicators, L2 | yes |
| SOFA (24 h) | the first-day SOFA as the score, for ICU and in-hospital death | no |

**Baseline features** at an anchor (from the stay store, `factory.labels.window_values`): for the 64 input channels
with the highest coverage in the train store, the last value, mean, minimum, maximum and count over the 6 and 24 hours
before t, and hours since the last value; the active states at t (vasoactive infusions, IMV, NIPPV, RRT, MCS);
statics available at t; hours since ICU admission. **Training anchors**: the same three designs drawn on a real train
store (`real_train`: up to 5,000 train stays per dataset, built like `real_test`).

### Further curves

- **Rollout count:** the Twin's AUROC and Brier score with k = 1, 2, 4, 8, 16, 32 rollouts (subsets of the 32), and
  GPU seconds per anchor.
- **Label efficiency:** LightGBM per dataset trained on n = 100, 300, 1,000, 3,000 train stays and all; the n at
  which it matches the Twin's AUROC per primary endpoint (interpolated; "not reached" stated).

### Metrics

AUROC and AUPRC (weighted for R), Brier score, ECE (10 equal-mass bins), observed over expected, calibration slope
and intercept (logistic recalibration); regression: MAE, the ensemble CRPS, central-interval coverage at 50/80/95%;
time to event: Harrell's C-index from the rollouts' median event times, and the time-dependent AUROC at 24 and 48 h.
Per dataset and pooled (pooled = anchors of all datasets together).

**Pre-specified comparisons** (descriptive; no pass/fail): Twin against LightGBM-pooled AUROC per primary endpoint,
paired bootstrap difference with its interval. Literature context: YAIB and HiRID-benchmark tasks are reported only
where our definitions match theirs (stated per task), and ETHOS-ARES's published MIMIC-IV results are cited, not
re-run (different splits).

## A2. Calibration and uncertainty

- The Twin's observed over expected, calibration slope and ECE per primary endpoint, by horizon and by ICU day of the
  anchor (1, 2, 3-7, 8+).
- Teacher-forced observed over expected after the 24-hour landmark (existing: `runs/eval/MUC32_main/
  calibration_masked*.json`) beside the rollouts' for IMV, vasopressor, RRT and ICU death: the exposure-bias gap.
- Remaining ICU stay and the regression targets: randomized PIT (KS distance from uniform), spread against skill, and
  interval coverage (`eval.forecast.dispersion`).
- Recalibration: isotonic per endpoint fitted on the dry run's tuning-split anchors, applied to the held-out
  anchors; reported beside the raw Twin, never instead of it.

## A3. Fidelity and privacy (compiled from existing results, plus the frontier)

- The P1-P5 gate tables of MUC32_main (`docs/m7_privacy.md` protocol; `runs/audit/m7_MUC32_main`).
- **The fit-privacy frontier:** P1 membership AUC (and its upper bound) against tuning NLL for every checkpoint on
  the host (S, S_main, M_main, M32, MC32, MCN32, ML32, MU32, MUH32, MUF32, MUC32), per dataset; missing P1 runs are
  run (inference). Expected: one increasing curve for AdamW and Muon checkpoints alike (round 3's finding).
- Fidelity and authenticity: the decision audits' tables (`auth_r0`, `auth_r1`) and the harness pilot, regrouped.

## A4. Utility of synthetic data

- **TSTR against TRTR** on the A1 primary endpoints at the A1 designs: LightGBM (A1 features) trained on twins (the
  bank's dataset-conditioned twins, by dataset and pooled) or on as many real train stays, tested on the held-out
  anchors. Ratio of the gains over chance, as in `eval.utility`.
- **Cross-dataset:** trained on one dataset's twins, tested on each other dataset's real anchors, against the same
  with real training stays.
- **Augmentation of the small datasets** (INSPIRE, NWICU): real train stays alone against real plus the dataset's twins
  (1:1 and 1:3), tested on that dataset's held-out anchors. Expected: no loss; any gain is reported with its interval.

## A5. Subgroups

The Twin's and LightGBM's AUROC, observed over expected and calibration slope for the primary endpoints, and the
twins' fidelity (prevalence of each primary endpoint's label among twins against real, by subgroup), by sex, age band
(18-44, 45-64, 65-79, 80+), dataset, admission type, and race or ethnicity where recorded (MIMIC, eICU). Subgroups
with fewer than 30 positive labels are reported as such, not scored. Reporting follows TRIPOD+AI.

## B2. Unknown site (a proxy, not an external validation)

L24 anchors, 300 per dataset: the Twin with the dataset token as recorded against the dataset token unknown
(`DS//unk`, the training-time augmentation) and, for eICU, the site token unknown. AUROC and observed over expected
per primary endpoint at L24. Stated as a limit: no dataset was held out of training.

## B3. Ablations from existing checkpoints (inference only)

Per checkpoint: tuning NLL (existing), P1 (A3), teacher-forced observed over expected for IMV, vasopressor, RRT and ICU
death after 24 h (`eval.calibration`, 1,000 held-out stays per dataset), and the Twin probe's AUROC at L24 for ICU
death, new IMV and new RRT (single forward passes). Contrasts: size (S, M, ML), context (M_main 8k, M32 32k), the
ICU-day block (MC32), input noise (MCN32), optimizer (MU32, MUH32), passes (MUF32, MUC32). No rollouts.

## C1. What-if rollouts: physiological response tests

- **Prefixes:** R and L24 anchors where the real stay starts the treatment within the next 6 hours and it is not
  active at t (the treatment was plausible); up to 200 per treatment, pooled over datasets.
- **Arms** (`generate.engine.Intervention`): *give* the treatment's start event at t (the dataset's most common start
  record, its median rate bin) against *withhold* the treatment's tokens throughout; 16 rollouts per arm and prefix.
- **Pre-specified responses** over (t, t + 6 h], mean of the arms' rollout means, give minus withhold:

| Treatment | Response | Expected sign |
|---|---|---|
| Norepinephrine | MAP | + |
| Insulin infusion | glucose | - |
| RRT (CRRT start) | potassium; creatinine (over 24 h) | -; - |
| Furosemide | urine output | + |
| IMV (intubation) | FiO2 charted; SpO2 | + charting; + |
| Fluid bolus (crystalloid) | MAP; urine output | +; + |

- **Read-outs:** the share of prefixes whose effect has the expected sign, the pooled effect with its interval, and
  for norepinephrine and insulin the response across given rate bins (dose-response).
- **Framing:** effects within the twin world, which inherits the data's confounding by indication; not estimates of
  real treatment effects.

## C2. Attribution on individual trajectories

- **Outcomes and anchors:** ICU death (L24), new IMV 24 h, AKI stage 2 or more 48 h, hypotension 4 h (R); 100 held-out
  anchors per outcome with the highest Twin risk and 100 at random.
- **Targets:** (a) the Twin probe's risk (differentiable; per outcome, A1's probe); (b) the Twin's rollout risk.
- **Event groups:** channel family (vitals, labs by panel, vasoactives, sedation, fluids and output, respiratory
  support, RRT, other medications, assessments, statics) x time block before t (0-6 h, 6-24 h, over 24 h).
- **Methods:**
  1. *Counterfactual editing* (the reference): the group's events removed from the decoded stay and the prompt
     re-encoded (grammar-valid by construction); the change in (a) for every group, and in (b) (16 rollouts per arm)
     for the 3 highest and 3 lowest groups by (a).
  2. *Integrated gradients* on the token embeddings for (a) (50 steps, the padding embedding as baseline), summed
     per group; completeness checked.
  3. *Attention rollout* (Abnar & Zuidema 2020) and attention x gradient, summed per group.
- **Measures:** deletion and insertion curves of (a) by each method's ranking against random (AUC); Spearman rank
  agreement and top-3 overlap of IG and attention with editing; agreement of editing on (a) with editing on (b) for the
  tested groups (sign and rank); stability across rollout seeds; per-outcome group importance across anchors.
- **Framing:** attributions are causal within the model only.

## C3. Virtual cohorts

From the internal bank (dataset-conditioned twins) against real held-out stays per dataset: the inter-dataset
structure (MMD matrix correlation, dataset-classifier confusion), per-dataset profiles (age, sex, LOS, ICU and
hospital death, IMV, vasopressor and RRT rates, events per hour, labs per day) with intervals, eICU hospital-level
total variation (`diversity.hospital_tv`), and rare header strata (at least 20 training stays and under 0.5% of a
dataset): prevalence and the primary endpoints' label rates, twins against real. Goal-conditioned enrichment is
reported as filter-based only (the model barely conditions on goal blocks; `docs/m10_factory.md` S1b).

## C4. Patient similarity (supplementary)

L24 anchors: the 10 nearest held-out neighbours by the Twin's state (cosine) against by TF-IDF of the prompt; the
share of neighbours sharing the anchor's ICU death, new IMV and new RRT labels, against chance.

## Clinician study (prepared in this phase, run later)

Blinded standardized timeline renderings of twins and real held-out stays (half each, matched on dataset and LOS
band), shown on the approved host to intensivists credentialed for the datasets shown: a visual Turing test (real or
synthetic, confidence 1-5), and plausibility ratings (1-5) of C1 responses and C2 top attributions. Sizes (to be
confirmed with the clinicians): 60 timelines per rater, 20 C1 cases, 20 C2 cases; agreement by Fleiss' kappa.

## Amendment 1 (2026-10-08, after the code's smoke run on 21 tuning-split anchors, before any score was read)

- **The alignment control** (added to A1): each anchor's real stay is decoded as an extra rollout member on its
  clock, and must reproduce the real store's labels at the anchor. On the smoke, every program agrees exactly except
  the four clinical-cut variants (`hypoglycaemia`, `fever_383`, `hyperkalemia_55`): their cut points fall inside a
  token bin, so the label depends on the decoder's within-bin value draw (the factory's documented quantization,
  `docs/m10_factory.md` correction 2). The control is reported per program, with the clinical-cut variants apart.
- **Primary endpoint replaced:** hyperkalaemia is scored by its bin-edge variant `hyperkalemia_top@1440` (potassium
  above the top bin edge within 24 h) instead of `hyperkalemia_55@1440`, whose label is partly the decoder's draw.
  The clinical cut stays a secondary endpoint, flagged.
- **A dry run sized for recalibration:** the tuning-split runs use 100 anchors per dataset and design with 32
  rollouts (A2's isotonic recalibration is fitted on them).

## Amendment 2 (2026-10-08, before any C1 rollout was drawn)

- **C1's prefixes:** few A1 anchors precede a real treatment start by at most 6 hours, so C1 draws its own: held-out
  stays (`real_test`) whose first record of the treatment in the ICU is a start at least 2.5 hours after admission,
  cut at a time drawn uniformly 0.5-6 hours before it (at least 2 hours after admission). Up to 200 per treatment,
  pooled over datasets.
- **The given record:** the dataset's most common start code among those stays, at its median rate bin (bin 4 of
  0-9); for norepinephrine and insulin also bins 1 and 8 (dose-response).
- **Windows:** potassium, MAP, glucose, urine output (the sum of urinary catheter outputs), FiO2 charting (count) and
  SpO2 over (t, t + 6 h]; creatinine over (t, t + 24 h]. Rollouts are capped at the 24-hour token quantile.
- **A4's small-site augmentation:** the small-data scenario is 300 real train stays of INSPIRE or NWICU alone, plus
  300 or 900 of the dataset's twins, against all real train stays of the dataset.

## Amendment 3 (2026-10-08, after the tuning-split dry run of L24, before any held-out rollout was analysed)

- **Remaining ICU stay over 7 days:** the dry run showed 20% of L24 rollouts end at the 7-day token cap, mostly the
  longest ones; dropping them (complete-case) biased p-hat down (observed over expected 2.45 on the tuning split). The
  estimate is now the Kaplan-Meier survival at 168 hours over the rollouts, a capped rollout censored at its cap
  (`eval.bench.km_survival`); the complete-case estimate is reported beside it.
- ICU death until discharge stays the share among the rollouts that reach a discharge, with the share of capped
  rollouts reported.

## Amendment 4 (2026-10-08, the study lead's request, before any held-out score was read)

- **TabPFN added as a baseline** (`tools/paper_tabpfn.py`): TabPFN v2, Prior Labs' open weights run locally on the
  host (the licence token only unlocks the download; no data leaves the host). The newer TabPFN 2.5-3.5 weights sit
  in a gated Hugging Face repository that needs a Hugging Face account token, which the project does not hold.
- **Its inputs:** A1's baseline features without the 6-hour windows (TabPFN v2 takes at most 500 features: 430
  remain), trained on a stratified sample of at most 10,000 of the design's real train rows (positives kept up to
  half), the odds corrected back to the train rows' prevalence; 4 estimators; the primary endpoints of each design.
  Compared with the Twin like the other baselines (`tools/paper_compare.py`).

## Amendment 5 (2026-10-08, the study lead's request, before any held-out score was read)

- **TabPFN-3.5 added** (`tools/paper_tabpfn.py --model-path`): the study lead placed the TabPFN-3.5 classifier checkpoint
  on the host (`/root/autodl-tmp/tabpfn-3.5`, `tabpfn-v3.5-20260909.safetensors`, SHA-256 ece4d67e..., verified); it
  runs locally and offline (no data leaves the host; the cloud API is not used, under the data-use agreements).
- **Its inputs:** all of A1's baseline features (about 720), a stratified sample of at most 50,000 of the design's real
  train rows (positives kept up to half, odds corrected back to the train rows' prevalence), 4 estimators, the primary
  endpoints. TabPFN v2 (amendment 4) stays as the older reference.

## Amendment 6 (2026-10-08, after C1's and C2's first analyses)

- **C1 urine output:** single urine-output records above 2,000 mL are dropped before the 6-hour sums. The real store
  holds implausible records (99th percentile of indwelling-catheter output 1.5 million mL: cumulative totals in a
  source), which made the sums meaningless (tens of litres per 6 hours); the expected-sign shares did not depend on
  them and are reported unchanged.
- **C2 integrated gradients:** the baseline is the padding embedding at every position, as the protocol states
  (the first analysis used the zero embedding: under the model's RMSNorm the output barely changes along that path
  until its very start, so 32 steps missed it and the completeness error was about 100%); midpoint steps. Editing,
  attention and the rollout-level check are unchanged.

## Amendment 7 (2026-10-09, the study lead's request for follow-up experiments, before any of them was run)

The pre-registered results (A1-C4, amendments 1-6; `docs/paper/results.md`) stand unchanged. The follow-ups F1-F9 are
new experiments, fixed here before any of their outcomes is read; their results go to `docs/paper/results_followup.md`.
Rules as above (inference only, aggregates only, rollouts deleted after their analyses, 95% intervals by a bootstrap
over stays, paired for differences, inclusion weights). New seeds: design and rollout seed 7.

One exploratory look preceded this amendment and motivates F3: on A1's stored predictions, the L24 ICU-death O/E was
0.91 on anchors whose 32 rollouts all ended within the cap and 1.17 on the others (72% of anchors), while the R
decompensation O/E was 1.83 even on anchors with all 32 rollouts valid (so the cap does not explain it).

### F1. Rollout count to 128 (accuracy, inference only)

- **Anchors.** R128: 150 per dataset drawn from test_R's anchors (their inclusion weights kept). L24x: every test_L24
  anchor whose stay died in hospital (346) plus 354 of the others at random, weighted by the inverse inclusion
  probability (a case-cohort sample: weighted metrics estimate the whole L24 sample's). Anchors whose prompt clock
  disagrees with their hour by more than 6 h (a continued context window) are left out and counted.
- **Rollouts:** 128 per anchor, T = 1, the A1 engine. Caps: R128 A1's (the 72-hour quantile). L24x four times A1's (at
  most the context minus the prompt), so that A1's cap can be applied afterwards by truncation (F3).
- **Scoring:** A1's label engine and estimator, every program of the design. A program is scored when at least half
  of the rollouts are valid. The curve: k = 1, 2, 4, ..., 128 rollouts, the mean over 20 random k-subsets.
- **Comparators on the same anchors** (A1's stored predictions): A1's 32-rollout Twin (a replicate with other seeds),
  LightGBM pooled, TabPFN-3.5 and the Twin probe. Paired differences: 128 against 32 rollouts of the same draw, and
  against LightGBM and TabPFN-3.5. Expected: AUROC rises with k and flattens, most for rare endpoints.

### F2. Estimators from the model's step hazards (accuracy, inference only)

- **Hazards.** Each rollout, and each anchor's real future appended to its prompt, goes through one teacher-forced pass
  of MUC32_main (bf16; the prompt's cache shared by its rollouts). At every timeline position the pass gives the
  probability that the next token is the ICU discharge "dead", an ICU discharge alive, or the first IMV, vasopressor or
  RRT event (`forecast.Reader` sets). The probability is set to zero where the sampler could not draw the token: the
  group order (an event token below the current group's first), the clock block, and the budget's end reserve. Times
  come from the labels' decode clock (the same gap draws).
- **Self-check.** Within the rollouts, the summed hazards (the compensator) must equal the sampled events per cause
  (O/E 1 within its exact interval). The rollouts are drawn from the same model, so a failure means the extraction is
  wrong.
- **Estimators** for death within 24 h (R; also at L24), ICU death until discharge (L24x), new IMV and vasopressor
  (24 h) and new RRT (48 h; R):
  - **E0:** A1's estimator (the share of positive valid rollouts).
  - **E1:** the Aalen-Johansen cumulative incidence over the rollouts, on the sampled events. Hourly bins from the
    anchor; causes: the event, death and alive discharge compete; a rollout reaching the cap is censored there.
  - **E2, the hybrid:** E1 with each rollout's sampled events replaced by its hazard increments in each bin while at
    risk (the compensator: a Rao-Blackwellization of E1).
- **Read-outs:** all three estimators on the same rollouts, with the F1 metrics and curves. For L24x ICU death, also at
  A1's cap. Expected: E2 at or above E0 in AUROC at every k, most at small k and for rare endpoints; E1 and E2 raise the
  ICU-death risk where E0 drops capped rollouts.

### F3. Mortality: where the death deficit comes from

- **D1, step calibration at the anchors.** Every test_R and test_L24 anchor (real futures that fit the context).
  The teacher-forced compensator of the real future is compared with the observed events, over 24 h, 48 h and until
  the ICU discharge: O/E by dataset and ICU day, for death, alive discharge and the three treatments.
- **D2, generation drift.** On R128 and L24x: the rollouts' mean compensator divided by the real futures' (same anchors
  and horizons). Also the hourly death and alive-discharge hazards since the anchor (rollouts at risk against real
  paths at risk), and the minutes per token.
- **D3, censoring.** The share of capped rollouts by the real outcome; ICU death (E0, E1, E2) at A1's cap against
  four times it; in-hospital death by the rollout's ICU end (dead, alive, capped).
- **D4, the bank.** The bank twins' ICU death by length band and end kind against real held-out stays.
- **Reading (pre-specified):**
  - O/E of D1 near 1 with a drift ratio below 1: the deficit is made in generation.
  - O/E of D1 near the rollouts' O/E: per-step miscalibration.
  - A gap between the two caps: evaluation censoring.

### F4. C2: attribution checked against the generator itself

- **Endpoints** (R, A1's probes): decompensation@1440, imv_onset@1440, aki_2@2880, hypotension@240.
- **Anchors:** 40 per endpoint at random among the half with the higher probe risk (prompts of at most 8,192 tokens).
- **Rollouts:** every event group's edit (C2's edits) and the original, 48 rollouts each, the 72-hour cap.
- **The generator's edit effect** per group: rollout risk of the original minus that of the edit. E2 for death and
  IMV; E0 for AKI and hypotension.
- **Reliability:** the split-half (24 + 24 rollouts) Spearman correlation of the generator's effects within anchors,
  Spearman-Brown corrected. This is the ceiling of any agreement.
- **Agreement** of probe editing, IG, attention rollout and attention x gradient with the generator's effects:
  - Spearman per anchor (median) and top-3 overlap, each against the ceiling;
  - sign agreement on groups whose generator effect's 95% interval excludes 0.
- **Sanity checks** (Adebayo et al. 2018) on the same anchors: the four methods with the model's weights replaced by a
  random initialization (same architecture, the trained probe kept), and with the probe refit on permuted training
  labels. Rank correlation with the trained attributions; expected low.

### F5. C1: validation of the what-if rollouts

- **(a) A null for C1's read-outs.** Two further withhold arms on C1's prefixes (16 rollouts each, seeds 7 and 8). The
  "effect" between them for every pre-specified response. Expected: share of the "expected sign" about 0.5, and an
  interval covering 0.
- **(b) Factual design with an observational benchmark.** Per treatment:
  - **Starters:** up to 200 held-out stays cut just before their first start of the treatment in the ICU (at least 2.5
    h after admission).
  - **Non-starters:** each starter's match, a held-out stay without the treatment in the next 6 h, cut at the same
    ICU hour (within 6 h). Matched within the dataset on the propensity of starting within 6 h: LightGBM on A1's
    features, trained on real train stays; nearest neighbour, caliper 0.2 SD of the logit.
  - **Arms:** starters get *give* (their own recorded start token) and *withhold*; non-starters get *give* (C1's given
    record) and *withhold*. 16 rollouts per arm; C1's windows and caps.
  - **Read-outs:**
    - the observational contrast: starters' real responses minus their matches';
    - the Twin's contrast on the same starters: give minus withhold;
    - the factual check: the factual arm's ensemble (starters' give, non-starters' withhold) against the real response,
      by 80% interval coverage, CRPS against persistence (the last value before t), and the PIT.
  - **Expected:** the Twin's contrast has the observational contrast's sign (both carry the data's confounding);
    coverage near nominal.

### F6. C3: intervals and hospitals (no new twins)

- Bootstrap intervals for every per-dataset profile statistic (twins, real, control).
- eICU hospitals with at least 30 held-out stays and 30 twins: hospital-level ICU death, IMV, vasoactive and median-LOS
  rates of twins against real (Spearman). The reference is the same correlation between real held-out and real control
  stays.
- The death gap by length band and end kind (with D4).

### F7. C4: neighbours as a predictor

- On test_L24 and test_R: leave-one-out kNN risk (the mean label of the k nearest other stays, k = 5, 10, 25, 50) for
  every primary endpoint of the design, AUROC by representation:
  - the model's state;
  - TF-IDF of the prompt;
  - A1's baseline features (standardized, missing as 0 with indicators);
  - the state of a randomly initialized model of the same architecture (seed 7; a negative control).
- Also with neighbours restricted to the anchor's own dataset (the lift is not dataset identity).
- Expected: state at or above the other representations; random initialization lowest.

### F8. Augmentation, with repetitions

- Every dataset, the designs' primary endpoints (L24: three; R: decompensation, IMV, vasopressor, RRT, AKI).
- LightGBM as A1's per-dataset baseline: A1's features and hyperparameters, early stopping on the dataset's tuning
  rows.
- **Training sets:** n = 100, 300, 1,000 real train stays drawn at random, alone and with m x n of the dataset's
  training twins (m = 1, 3); twins alone (300, 1,000, 3,000 where available). 20 repetitions (seeds 7-26).
- **Test:** the dataset's held-out A1 anchors.
- **Read-outs:**
  - the mean AUROC per cell;
  - the paired difference (real + twins minus real alone, the same real draw), with an interval over repetitions and a
    bootstrap over test anchors;
  - the real n that matches n real + m x n twins (interpolated).
- Expected: no reliable gain at n of 300 or more; any gain at n = 100.

### F9. Transfer to new institutions (characterization; a model without a dataset needs training, out of scope)

- **(a) Site prompts on R128,** 32 rollouts each, against the first 32 of F1's:
  - dataset and site tokens unknown;
  - the dataset token swapped to another dataset (each anchor a random other dataset; site unknown).
  - Read-outs: AUROC and O/E per endpoint; and how far the predicted treatment risk shifts toward the swapped
    dataset's practice (the change in mean predicted risk against the two datasets' real rate difference).
- **(b) Familiarity:** the Twin's (A1) and LightGBM's AUROC by the anchor's novelty (the cosine distance of its state to
  the nearest train-split anchor's, quintiles); in eICU, by the hospital's training volume and site token.
- **(c) Supervised transfer:** LightGBM and the Twin probe trained on six datasets and tested on the seventh's held-out
  anchors (leave one dataset out), against within-dataset training, beside the zero-shot Twin.
- **Stated as a limit:** the Twin saw every dataset's training stays, so only a model trained without a dataset would
  test transfer to a new site.

**Compute:** about 20 GPU-hours on the host (F1 about 10, F4 3, F5 3, F9a 2, scoring 2); the rest CPU.

## Amendment 8 (2026-10-09, during F1-F3, before the corrected scores were read)

- **The hazards renormalized as the sampler draws.** F2's self-check on R128 failed for death: 13,436 sampled deaths
  against 12,552 summed hazards (1.07; 1.15 in AmsterdamUMCdb), while the treatments matched within 3%. A check
  against the exact masked probabilities (`eval.calibration.masked_probs`) on 40 rollouts per dataset traced it.
  - **The cause:** the first scoring took the model's raw probability (zero only where the group order forbids the
    token). The sampler renormalizes over the tokens it may draw. In end-of-life sequences the model puts mass on rate
    changes and stops of infusions that are no longer running, which the grammar forbids, so the raw death
    probability was too low there.
  - **The fix** (`eval.hazards.masked_parts`, `normalized`): the sampler's denominator, from the grammar's timeline
    rules, the infusion rule and the dataset's support. It reproduces the exact masked probabilities within 0.975-1.006
    for every cause and dataset (`paper_hazard_check.py`).
  - **The rescoring:** D1, R128 and every later subset are scored again with it. The first, raw scoring is kept
    (`hazards_raw/`) and its R128 results are reported beside the corrected ones.
- **A decode fix:** a batch of decoded stays without any vitals left a weight table undefined in the derived concepts
  (`meds.derived`: the norepinephrine equivalents). R128's A1-style analysis failed on one such batch. The table is now
  created empty, which changes nothing for batches with vitals.

## Amendment 9 (2026-10-10, the study lead's request after F4, before the diagnosis ran; exploratory, not pre-specified)

F4's model-randomization check failed for attention rollout: its group ranking in a randomly initialized model
correlated 0.96-0.97 with the trained model's. This amendment adds a diagnosis and tests candidate remedies on F4's
anchors (R, the four endpoints, 40 anchors each, prompts of at most 4,096 tokens). It uses the trained model and F4's
random initialization (seed 7, the trained probe kept). Prompts and probes are F4's; nothing is resampled
(`tools/server/paper_attn_diag.py`).

- **Hypotheses:**
  1. *Group size.* C2 sums token scores over each group, and groups differ in tokens by orders of magnitude. Any
     near-uniform token score then ranks groups by size.
  2. *Causal geometry.* Rollout multiplies 12 head-averaged attention matrices, each mixed with the identity. Under a
     causal mask such products tend to a fixed profile over positions, whatever the attention, and earlier tokens
     gain mass. The rollout of uniform causal attention (no model) is that structural null.
  3. *Target independence.* Rollout does not use the probe, so it is the same for every endpoint.
- **Measures:** per anchor, Spearman correlations over its groups; medians over anchors.
  - Trained against random (the sanity check).
  - Against the groups' token counts, and against the structural null.
  - Against the trained model's probe-level counterfactual editing (C2's and F4's reference), signed and absolute.
  - At the token level: trained against random; the distance to the null (total variation); the last token's
    attention entropy in each layer.
  - A check: the rollout and attention x gradient computed here must reproduce F4's per-anchor scores.
- **Candidate remedies,** each with the same measures:
  - group means instead of sums;
  - the rollout less the structural null, and its ratio to it;
  - the last layer's attention alone (no rollout);
  - norm-weighted rollout (Kobayashi et al. 2020): each head's attention weight times the norm of its transformed
    value;
  - gradient-weighted rollout (Chefer et al. 2021), which depends on the target;
  - attention knockout (Geva et al. 2023): attention from every other token to a group's tokens blocked in every layer,
    and the fall of the probe's logit. This is an intervention on the model's information flow, and it depends on the
    target.
- **A remedy passes** if its sanity correlation is low (about 0.3 or less) and its agreement with editing is positive.
- **Scope:** F4's attention results stand as reported; nothing here replaces them. Agreement with the generator would
  need F4's rollouts again (about 3 GPU-hours) and is not part of this amendment.
- **Added after the first run** (same day): the sanity and editing correlations with the groups' token counts held
  fixed (partial rank correlations), to separate size from position. The run was repeated with them; every other
  number was identical.

## Amendment 10 (2026-10-10, the study lead's request, before any comparator was computed)

The study lead asked for more comparators, particularly those used in clinical practice: traditional clinical scores,
linear models, basic tree models and other standard learners. A1 compared the Twin with LightGBM, L2 logistic
regression, TabPFN and, for death at the landmarks, the first-day SOFA. This amendment adds comparators on A1's
held-out anchors (L24, L48, R; 3,500 each) and on F1's subsets (R128, L24x). Nothing pre-registered changes.

- **Endpoints:** A1's primary endpoints of each design. At L24 also death within 24 h, new IMV, new vasopressor and
  new RRT (F2's L24x endpoints).
- **Clinical scores.** No training. Each is computed from the canonical CLIF tables of the anchor's stay (the
  harmonized raw values, not the tokens). Only data recorded or resulted by the anchor's prediction time count (labs
  at their result time), from the 24 hours before it. A component without a value scores 0 (normal), as mimic-code and
  ricu do. Coverage is reported per dataset.
  - *Severity scores, on the worst values:*
    - SOFA (clifpy, with `meds.derived`'s respiration correction);
    - SAPS II and OASIS (mimic-code's pinned definitions, landmarks only), with their probability equations. SAPS II
      leaves out the chronic-disease items, which only discharge diagnoses record.
  - *Early-warning and sepsis screens, on the latest value of each parameter* (ricu's pinned definitions): NEWS (the
    NEWS2 scale-1 table), MEWS, qSOFA, SIRS.
  - *The datasets' own admission scores* (landmarks, each dataset's anchors only):
    - APACHE IVa (eICU: predicted ICU and hospital mortality, predicted ICU stay);
    - SAPS 3 (SICdb, with its general probability equation);
    - the ASA class (INSPIRE).
  - *Bedside indicators* (the latest values; a missing value set to normal):
    - death within 24 h: shock index, lactate;
    - new IMV: ROX index (SpO2/FiO2 over respiratory rate; lower is worse);
    - new vasopressor: shock index, MAP (lower is worse);
    - new RRT and AKI: the KDIGO stage (mimic-code's creatinine rules on result-time creatinine against the 48-hour and
      7-day minima, and the urine-output stage), creatinine;
    - hypotension: MAP;
    - hyperkalaemia: potassium;
    - thrombocytopenia: platelets (lower is worse);
    - sepsis: lactate.
- **Learned baselines.** Trained on A1's features and rows of real training stays, as A1's LightGBM. Where a
  hyperparameter set is listed, the tuning store picks it by AUROC.
  - *Linear:* logistic regression with an L1 (LASSO) or elastic-net (l1 ratio 0.5) penalty, C in {0.01, 0.1, 1}.
    A1's L2 logistic regression stands.
  - *Trees:* a decision tree (CART; depth 3, 5 or 8; at least 50 rows per leaf); random forest and extra trees (300
    trees, square-root features per split, at least 5 rows per leaf). A1's LightGBM stands.
  - *Others:* Gaussian naive Bayes; k-nearest neighbours (k 25 or 100, distance-weighted); a multilayer perceptron
    (hidden layers of 128 and 64 units, early stopping).
  - Every model but LightGBM gets median imputation with missing indicators, and standardization.
  - Solvers (fixed before any learned baseline was scored): liblinear for L1; saga for the elastic net, at most 100
    epochs and warm-started along the C path.
- **Implementation notes,** recorded before any comparator's held-out result was read:
  - The subsets are compared on F1/F2's anchors: those with a label in A1's predictions. E2's AUROC was computed on
    exactly these anchors.
  - Where a dataset charts no MAP, MAP is computed from systolic and diastolic pressure charted together.
  - A check of the clinical extraction: at the 24-hour landmark, the windowed SOFA should match the pipeline's
    first-day SOFA.
  - Per-dataset AUROCs are tabulated for ICU death at L24 and death within 24 h on rolling anchors.
  - Running the learned baselines: a pool of forked workers. Every finished fit is saved, and a rerun resumes.
    The first pool broke (12 logistic fits at once exceeded memory) and was rerun with fewer logistic workers. This
    changes no setting.
- **Measures:**
  - AUROC (weighted as A1) with 95% intervals by a bootstrap over stays within datasets, and AUPRC. Brier score and
    O/E for comparators that give probabilities.
  - Paired differences, the Twin minus each comparator, on the anchors both scored:
    - on A1's anchors, A1's Twin (32 rollouts);
    - on R128 and L24x, the 128-rollout Twin (the share, E0). E2's AUROC (F2) is shown beside; its per-anchor values
      were not kept.
  - The native scores on their own datasets' anchors, against the Twin and LightGBM on the same anchors.
- **Reading:** descriptive, with no multiplicity adjustment. The clinical scores are not trained for these endpoints,
  so they are expected to trail learned models.
- **Compute:** CPU only on the host, about 2-3 hours.

## Amendment 11 (2026-10-10, the study lead's request for an application toolkit, before any of its validation ran)

The study lead asked for an application toolkit as part of the paper: the validated capabilities as tools for researchers,
trial designers and teachers (`src/icutwin/toolkit`, `docs/toolkit.md`): Forecast, What-if, Explain, Similar
patients, Cohort synthesizer, Trial designer and Teaching cases, each output carrying the paper's evidence and
caveats. The tools rest on A1-C4, F1-F9 and amendments 9-10. This amendment fixes the validation of what the paper
has not shown: that the toolkit reproduces the paper's numbers (V1), recalibrated risks (V2), the privacy of prompted
and intervened sampling (V3), the trial designer (V4) and the teaching cases (V5). Nothing pre-registered changes; the
paper scripts are unchanged (the toolkit copies their definitions, and a test checks they agree).

- **V1, conformance.** The toolkit's batch forecast on F1's R128 anchors (1,050 rolling held-out anchors, inclusion
  weights) with 32 rollouts per anchor and A1's caps (`a1/test_R/design.json`), sampled with vLLM, E2 by the
  toolkit's teacher pass; and F5's design rows for norepinephrine (MAP) and furosemide (urine output), 16 rollouts per
  arm at C1's 24-hour caps, read as F5 read them. The rollouts are new draws (the toolkit seeds each request), so V1
  compares estimates within sampling error. It passes when:
  - for each of A1's nine primary R programs, the toolkit's AUROC lies within the 95% bootstrap interval (over stays
    within datasets) of A1's 32-rollout AUROC on the same anchors, and so does its O/E;
  - for F2's four endpoints, the toolkit's E2 AUROC at 32 rollouts lies within 0.02 of F2's curve at k = 32;
  - for both responses, the toolkit's factual contrast (starters' give arm minus their matched non-starters' withhold
    arm) lies within F5's 95% interval, and the A/A difference's interval covers 0.
  V2-V4 are read only after V1 passes; a failure is fixed in the toolkit and V1 rerun, both reported.
- **V2, E2 recalibration** (the open item). Isotonic regression per program (inclusion weights, clipped at the ends)
  fitted on the tuning split's R anchors of amendment 1 (`a1/dev_R`: 100 per dataset), 32 rollouts at that design's
  caps; E2 for the four hazard programs, the share for A1's other R primary programs; labels from the alignment rows
  (each anchor's own stay decoded on its clock, which A1's alignment control showed equal to the real store's labels
  for bin-edge programs). Evaluated on V1's draws at 32 rollouts and at 16 (the first 16: the fast mode): O/E,
  calibration slope and intercept, ECE and Brier before and after; AUROC must stay within 0.005 (the maps are
  monotone). The maps become the toolkit's recalibration (`runs/toolkit/recalibration`).
- **V3, privacy of prompted sampling** (gate v1's `conditional_and_counterfactual_sampling: PENDING`). Members are
  train-split stays, non-members held-out stays, per dataset at most 500 each (at most 150 each for the sampling
  parts), stays of at least 30 hours, each cut at a whole hour drawn uniformly from 4 h to its stay minus 25 h.
  - (a) Loss: the model's mean NLL per token of the stay's real next 24 hours given the prompt (teacher-forced, raw
    probabilities; P1's statistic, conditional). Membership AUC and TPR at 1% FPR with bootstrap intervals, pooled
    and per dataset; PASS by P1's limits: AUC upper bound at most 0.60 and TPR at 1% FPR upper bound at most 0.03.
  - (b) Extraction: 16 rollouts per prompt at C1's 24-hour caps. Per prompt, whether any rollout reproduces the real
    future's first 32 timeline tokens exactly (P2's statistic; PASS: member minus non-member rate at most 0.01), and
    the best rollout's Jaccard similarity of event-token multisets over the first 24 hours (membership AUC, upper
    bound at most 0.60).
  - (c) Intervened: norepinephrine prefixes built as C1's (members from the train split, non-members held out; at
    most 100 each), the give-median arm, the statistics of (b).
  - Controls: a copying generator (members' rollouts replaced by their real futures) must FAIL (b); a private
    reference (two disjoint halves of the non-members) must PASS (a) and (b).
  - The result is written to `runs/audit/m7_MUC32_main/prompted.json`. Whether the gate report records it is the
    study lead's decision.
- **V4, trial designer** (CPU, the kept predictions). Endpoints: ICU and in-hospital death on L24x (128-rollout
  shares); new IMV and new vasopressor within 24 h, new RRT and AKI stage 2 within 48 h on L24 (32-rollout shares; A1
  scores no 48-hour IMV or vasopressor program at the landmark). Relative risks 0.8 and 0.7; enrichment fractions 1,
  0.75, 0.5, 0.25 (by rank, ties broken at random); prognostic scores: the Twin, SOFA, SAPS II, OASIS, NEWS and
  LightGBM (APACHE IVa on its eICU anchors, reported apart); 2,000 simulated trials per cell for the adjusted analysis
  (logistic regression on treatment and the score). Checks: the type I error at relative risk 1 within 0.04-0.06 for
  both analyses; the unadjusted simulated size within 5% of the analytic size. Reported: sizes, screening, the
  adjusted-to-unadjusted size ratio, the Twin against each clinical score; no further pass rule.
- **V5, teaching cases.** Three pre-specified objectives on the bank, three typical cases each and a drill of the
  nine: shock (a vasopressor episode; decision before the twin's first norepinephrine start; options norepinephrine
  and a fluid bolus), respiratory failure (IMV; before its first IMV record; option IMV) and renal support (RRT;
  before its first CRRT record; options CRRT and furosemide). Checks: every case meets its objective; every twin
  shown passes novelty against the training stays when the reference exists on the host (otherwise the bank's
  production filter is cited); every page is self-contained (no external URL). The clinician study, prepared in this
  protocol, stays a later phase; it would give the cases' realism evidence.
- **Implementation notes,** recorded before any validation result was read:
  - Sampling runs on the study lead's RTX 5090 host (approved 2026-10-10; vLLM 0.31.0 and the 4080 host's versions), the
    teacher passes and the label engine in the main environment; the M7 gate is checked for the same weights there.
  - E2 in the toolkit is F2's computation (the same masks, clock and stop positions) on the long bins (hourly to
    48 hours, then 6-hourly to 30 days).
  - V3 takes P1's whole rule: FAIL when a lower bound exceeds the limit (AUC 0.60, TPR at 1% FPR 0.05), PASS as
    above, INCONCLUSIVE otherwise (at 500 per side the intervals may be too wide to decide). Its real future is the
    stay's encoding after the prompt (as F3's D1), cut at 24 hours by the interval tokens' median gaps; prompts that
    continue a context window and stays whose encodings disagree are left out and counted; (b) needs a real future of
    at least 32 tokens.
- **After V1** (post hoc: the study lead's decision, recorded before the added check was computed). V1 met 14 of its 15
  checks. Sepsis (ASE) within 24 h fell outside the rule: the toolkit's AUROC was 0.963 against A1's 0.989 (95%
  interval 0.974-0.999); its O/E was within. The rule compared an independent 32-rollout replicate with A1's bootstrap
  over stays, which leaves out both replicates' rollout noise. With 8 positives on these anchors that noise is several
  AUROC points: F1's mean over random 32-rollout subsets of its 128 rollouts on the same anchors is 0.900.
  - *Added check, fixed before it was computed:* for each of the nine programs, the rollout-bootstrap standard
    deviation of the toolkit's AUROC and of its O/E (rollouts resampled within anchors, anchors fixed, 1,000
    resamples). Two independent replicates on the same anchors should differ by at most 1.96 x sqrt(2) times it.
  - If all nine meet it, V2-V5 run, and V1 is reported as passed under the corrected rule, with the original rule's
    result beside it.
  - *Its result:* 7 of 9 programs met it. Sepsis now met it. Death within 24 h did not: AUROC 0.880 against A1's
    0.903, limit 0.017; O/E 1.69 against 1.85, limit 0.15. New RRT missed by 0.0000006.
- **The study lead's adjusted criterion** (post hoc: set by the study lead on 2026-10-10, after the original rule's and the
  added check's results were known).
  - *Why:* the paper reports two runs on these anchors with the same caps. A1 drew 32 rollouts; F1 drew 128, and its
    32-rollout expectation is F1's curve, the mean over random 32-rollout subsets. The two runs differ by more than
    rollout noise: on death within 24 h, A1's AUROC is 0.903 against F1's 0.881, 3.7 noise SDs apart. The toolkit
    matches F1 there (0.880).
  - *The criterion:* a program conforms when the toolkit's AUROC and O/E both lie within the added check's limit
    (1.96 x sqrt(2) x the rollout-bootstrap SD) of either paper run. For F1, that means its curve at 32 rollouts
    and its 128-rollout O/E.
  - *Then:* when all nine conform, V2-V5 run. V1 is reported with all three verdicts: the original rule, the added
    check, and this criterion.
  - *Its result:* all nine conformed. Death within 24 h and new RRT conformed to F1, the other seven to both runs.
    V2-V5 ran.
- **V2's result and how its rule was applied** (recorded after the result).
  - *Treatment starts:* the isotonic maps brought O/E near 1 (IMV 0.55 to 1.16, vasopressor 0.46 to 1.00, RRT 0.55
    to 0.84). Their flat steps tie many risks, though, and AUROC moved by more than 0.005 for five programs: IMV,
    vasopressor, RRT, sepsis and thrombocytopenia (thrombocytopenia -0.05).
  - *Transfer:* the tuning split's miscalibration did not carry over everywhere. Death within 24 h had raw O/E 1.98
    on tuning against 1.71 on V1's draws, so its map overcorrected to 0.75.
  - *Adoption:* a map becomes the toolkit's recalibration only where AUROC stayed within 0.005 at both 32 and 16
    rollouts: death within 24 h, AKI, hypotension and hyperkalaemia. The full set is kept in
    `runs/toolkit/v2/maps_all.json`.
- **V2b, logistic recalibration of the treatment starts** (post hoc: the study lead's decision after V2's result, recorded
  before it was computed).
  - *The map:* for new IMV and new vasopressor within 24 h and new RRT within 48 h, logit(p') = a + b x logit(p).
    Here p is E2, clipped to 1e-9-(1 - 1e-9), and a and b come from weighted maximum likelihood on V2's
    tuning-split fit (the same anchors, rollouts and alignment-row labels). The map is strictly increasing when
    b > 0, so AUROC cannot change.
  - *Evaluation:* on V1's draws at 32 and 16 rollouts, with V2's metrics.
  - *Adoption:* a map replaces the program's raw risk when b > 0 and, at both 32 and 16 rollouts, its O/E is closer to
    1 and its ECE no larger than the raw risk's.
- **V3x, V3's loss test at P1's sizes** (post hoc: the study lead's decision after V3's result, recorded before it was
  computed).
  - *Why:* V3 was INCONCLUSIVE only because three datasets' TPR intervals were too wide at 500 stays per side.
  - *The run:* (a) again, with a fresh draw of up to 3,000 members and 3,000 non-members per dataset (P1's sizes),
    each cut as V3's (seed 1), and V3's statistic and P1's rule.
  - *The verdict:* V3x's (a) replaces V3's (a). V3's (b), (c) and controls stand.
  - *The gate:* when nothing is INCONCLUSIVE or FAIL, the gate report records V3 for conditional and counterfactual
    sampling (the study lead's decision: "enable V3 access"). Gate v1's other results are unchanged.
  - *V3x's result:* the pooled test and seven of eight datasets passed, including AmsterdamUMCdb, eICU and HiRID. The
    pooled AUC was 0.531 (0.524-0.537) and TPR at 1% FPR 0.013 (0.011-0.017). Nothing failed. INSPIRE was
    INCONCLUSIVE: every eligible held-out stay gave 790 per side, and its TPR interval was 0.009-0.039 against 0.03
    (AUC 0.568, upper bound 0.597).
  - *The study lead's decision after it:* the gate records V3 as PASS. The record states INSPIRE's sample-limited result
    with its numbers, and V3's final verdict file stays INCONCLUSIVE under the rule.
- **Reading:** descriptive beyond the stated pass rules; no multiplicity adjustment.
- **Compute:** about 8 GPU-hours (V1 about 4, V2 about 2, V3 about 1.5, V5 about 0.3) and a few CPU-hours (V4).

## Amendment 12 (2026-10-10, the study lead's decisions on the clinician study, before any stimulus was built or rating collected)

The clinician study prepared above ("Clinician study, prepared in this phase, run later") is fixed in full in
`docs/paper/clinician_study_protocol.md`. That document governs; this is its summary, and it is registered on OSF
before the first rating.

- **Raters:** 10-20 critical care physicians (target 16, at least 12) from 2-4 centers. All hold PhysioNet
  credentials for MIMIC-IV and eICU-CRD, and none worked on the model. The existing ethics approval covers the study;
  centers and raters are anonymous in every output.
- **Data route:** real stays only from those two datasets' held-out splits, and bank twins conditioned on them. Every
  stimulus is rendered from tokens through the same decoder. Each rater gets one self-contained HTML packet (the
  study lead's decision, before any stimulus was built). The packet holds no answer and no outcome; module C's cards stay
  obfuscated until the unaided estimates are locked. Raters attest their credentials and confidentiality, and return
  only the exported answers. The key and only aggregates leave the approved host.
- **Module A, realism:** 80 first-24-hour case sheets (40 real, 40 twins matched on dataset and stay length), after
  four unscored practice sheets; real or synthetic with confidence. Primary: pooled accuracy equivalent to chance
  (90% interval within 0.40-0.60; mixed model, raters and items).
- **Module B1, simulated responses:** 24 of F5's starter prefixes, with the next 6 hours either one random twin
  rollout under the give arm or the real course, blinded and counterbalanced; insulin is a known-failure stratum.
  Primary: twin courses non-inferior in plausibility (margin -0.5 on 1-5, insulin excluded).
- **Module B2, explanations:** 20 rolling anchors, with top factors from editing, knockout or random groups (blinded).
  Primary: editing and knockout each rated above random (Holm).
- **Module C, decision support:** 48 landmark cases (deaths enriched to about a quarter), risks estimated unaided,
  then with the twin's forecast card. Primary: the Brier score of ICU death, aided minus unaided.
  - *The card:* 128 rollouts at four times the caps; ICU death raw; the other outcomes recalibrated where V2's maps
    pass. AKI's passed (AUROC +0.0004; O/E 0.67 to 0.82, slope 0.79 to 0.90), so AKI is shown recalibrated.
  - *Outcomes asked:* ICU death and AKI stage 2 or more within 48 h. A new vasopressor within 24 h was dropped
    after the stimuli were drawn and before the pilot: 1 event in its 44 eligible cases.
  - *Key secondary:* the twin-specific gain against the counterfactual blend toward LightGBM's and SAPS II's risks.
- **Sizes:** from `tools/clinician_study_power.py`, fixed here. With 16 raters:
  - A's equivalence (80 items): power 0.91 at chance, 0.85 at 0.52, with the larger item variation;
  - B1: 0.84 if twin courses are 0.2 points worse;
  - B2: 0.97 for a 0.6-point gain;
  - C: 0.98-1.00, or 0.77-0.81 in the pessimistic scenario.
  The twin-specific gain is estimated, not tested (power at most about 0.3).
- **Order and blinding:** A first, C last; item order randomized and versions counterbalanced from fixed seeds; no
  interim look. A pilot with two non-raters (packets P01, P02) checks the case sheet's final form, the cue codebook
  and the timing before the first rater starts; any change is recorded here first.
- **Analysis:** fixed in code before the first rating and tested on a simulated study. The scripts are
  `tools/clinician_study_analysis.py` and `tools/clinician_study_models.R`, run in a pinned R environment (lme4
  1.1-35.5) on the approved host.
  - B's mixed models carry a random effect per stimulus (an item in one version).
  - A's and C's results are cross-checked by the two-way bootstrap.
  - B's are cross-checked by per-item contrasts, the sizing's analysis, because the two-way bootstrap is
    conservative for within-item contrasts.

### Amendment 12, addendum (2026-10-11, after answers were received)

- **Received:** answers for packets R01-R20, build 3d6c08cbd79e. The full record is kept with
  the study's internal records.
  - *10 Oct:* not in the packets' export format.
  - *11 Oct:* the same answers, with timestamps dated before the items and the build existed.
- **Status:**
  - *Attestation:* on 2026-10-11 the study lead attested that the answers come from the study's raters.
  - *What was run:* the pre-specified analysis, on the answers alone.
  - *Excluded:* the timing, role, experience and qualification fields, whose dates precede the items and the build.
  - *Not run:* the integrity, consent, time-per-case, seniority and eligibility checks.
- **Pre-specified verdicts:**
  - *A:* not met. Accuracy 0.408, 90% CI 0.351-0.468, below the lower margin, so below chance.
  - *B1:* non-inferiority not established (-0.23, -0.56 to 0.11).
  - *B2:* superiority not established.
  - *C:* met (-0.016, -0.031 to -0.001; p = 0.037).
- **Post hoc, declared after the results were known:** A's non-detection. The upper one-sided 95% bound of accuracy
  is 0.47, below 0.60 and below chance. It is reported as post hoc and does not change A's pre-specified verdict.
- **Deviations:**
  - the pilot was not run first;
  - the answers did not arrive as the packets' exports.
- **Becoming the study's result:** each packet's own export through `collect` on the host. The analysis then runs
  unchanged.
- **Status for the manuscript** (study lead, 2026-10-11): the analysis of the current answers has not yielded formal
  results. The manuscript carries a placeholder, and the clinician figure is a template tested on the simulated
  study.

## Amendment 13 (2026-10-11, the study lead's request for statistical tests in the manuscript, before any P value was computed)

**Scope.** These are reporting statistics for the manuscript's figures: P values for the key comparisons, and Monte
Carlo intervals for the trial simulations.
- **No new data, rollouts or GPU work.** Everything is computed on CPU from per-anchor predictions already kept, or
  from the intervals already stored.
- **Nothing reported changes:** no estimate, no interval and no pre-specified verdict. The materials' 95% intervals
  stay the ones reported.
- **The code:** `tools/paper_stats.py`. It writes `runs/paper/stats/stats.json` (with the inputs' sha256 and the git
  sha) and `docs/paper/tables/manuscript/stats_*.csv`.

**Common rules:**
- **Tests:** two-sided at α = 0.05, unless a pre-specified rule fixes the side.
- **Adjustment:** Holm within each family below. Unadjusted and adjusted values both go into the source data; the
  figures show the adjusted ones, labelled as such.
- **Format:** P is printed exactly to two significant figures, and "P < 0.001" below that. It always stands beside
  the effect and its 95% interval.
- **Pre-specified rules:** where one exists (the privacy gate's bounds, amendment 9's pass rule, the clinician study's
  tests), its verdict is reported, not a P value.

**Paired AUROC differences, exact.** Used where both predictors' per-anchor values are kept: Didymos's sampled
estimate (the share), A1's 32-rollout share, LightGBM, TabPFN, the learned baselines and the clinical scores.
1. **Reproduction first.** The stored computation is rerun first, and every stored interval must be reproduced to
   within 1e-12; otherwise the script stops before any P value is computed. The two stored computations:
   - F1's comparisons: `tools/paper_f2.py`'s `paired`. `bench.bootstrap` resamples stays within datasets, 1,000
     resamples, seed 0, with the inclusion or case-cohort weights. Stored in `f1/{R128,L24x}/f2.json` under
     `f1.<program>.paired`.
   - Amendment 10's comparisons: `tools/paper_comparators.py`'s `evaluate`. `bench.resamples`, 1,000 resamples, seed
     0. Stored in `f10/comparators.json` as `ref_minus_ci`.
2. **The P value.** The same procedure is run with 10,000 resamples and seed 1 (`bench.bootstrap_draws`, the same
   resampling as `bench.bootstrap`, returning the draws; or `bench.resamples`). P = min(1, 2(1 + k) / 10,001), where
   k counts the resamples on the other side of 0 from the estimate.

**Paired differences from stored intervals.** Used for the hazard estimate (E2), whose per-anchor values were not
kept.
- **The approximation:** z = Δ / SE, with SE = (Δ − lower) / 1.96 when Δ ≥ 0 and (upper − Δ) / 1.96 otherwise
  (Altman and Bland's method, on the side of zero); P = 2Φ(−|z|).
- **The check:** the approximation is applied to every exact comparison in families 1-4 as well, and its agreement
  with the exact P is reported: the median absolute difference in log10 P, and the share of identical decisions at
  0.05. The check is reported; it is not a gate.

**Families:**
1. **Primary: Didymos minus LightGBM (pooled) on the 13 primary endpoints.**
   - *The endpoints:* rolling anchors R128 (9) and the 24-hour landmark L24x (4).
   - *Which estimate:* the hazard estimate for death within 24 h, new IMV, new vasopressor and new RRT (R128) and for
     ICU death and death within 24 h (L24x), with the P value from the stored interval; the sampled estimate for the
     other 7 endpoints, with the exact P value.
2. **Secondary: Didymos minus TabPFN-3.5,** the same 13, built the same way.
3. **Follow-up minus pre-specified,** the same 13: Didymos at 128 rollouts against A1's 32-rollout share.
   - `e2_minus_twin_a1_32` (from the interval) and `twin_128_minus_twin_a1_32` (exact).
4. **Didymos minus the best clinical score per endpoint,** the same 13.
   - *Which score:* the best by AUROC among the clinical scores amendment 10 assigns that endpoint, on the same
     anchors, chosen after the fact (conservative). The candidates are SOFA, NEWS, MEWS, qSOFA and SIRS, plus SAPS II
     and OASIS at the landmark; bedside values are excluded.
   - *Which estimate:* the sampled estimate, exact, by amendment 10's resampling.
5. **Calibration on V1's draws.**
   - *The data:* rolling anchors, 32 rollouts, A1's labels and weights.
   - *The programs:* the seven programs whose maps are in force (`runs/toolkit/recalibration/MUC32_main.json`).
   - *The tests:* O/E = 1 raw and recalibrated, and the change in O/E and in ECE (recalibrated minus raw).
   - *How:* a paired bootstrap over stays within datasets: 1,000 resamples (seed 0) for the intervals, 10,000 (seed
     1) for P. The point estimates must equal V2's and V2b's stored values.
6. **What-if agreement:** Didymos's factual contrast minus the observational contrast, for the 9 pre-specified
   responses (`f5/c1b.json`).
   - The pairs were not kept, so the difference's standard error combines the two stored intervals' standard errors
     ((upper − lower) / 3.92) as if they were independent. That is conservative when the two estimates are
     positively correlated.
   - P is normal. The figure states that a large P does not show equivalence.
7. **The null arm:** the share of null-arm prefixes with the expected sign against 0.5, for each treatment's
   pre-specified response. An exact two-sided binomial test, with k = round(share × prefixes).
8. **Membership signal against fit:** for each dataset, the Spearman correlation across the 9 checkpoints between membership AUC
   and the member/non-member NLL gap (`a3/a3.json`, `frontier`). An exact permutation test (all 9! orderings), Holm
   across the 7 datasets.
9. **Augmentation** (`f8/aug_{L24,R}.json`): for each design, real-stay count (100, 300, 1,000) and twin multiple (1x,
   3x), the cells' mean gains (endpoint by dataset).
   - The tests: a two-sided sign test against 0.5, and a Wilcoxon signed-rank test against 0.
   - Holm across the 12 combinations, separately for each test.
   - *Clarification (2026-10-11, before these P values were used):* the cells are those of the materials' summary,
     an endpoint and dataset with at least 20 test positives (`tools/paper_results_followup.py`, F8). That is the
     definition behind "19 of 19 cells".
10. **Attribution** (`f4/attn_diag/per_anchor.jsonl`, 152 anchors): knockout's agreement with editing minus attention
    rollout's (group sums). A Wilcoxon signed-rank test over anchors, pooled, and per endpoint (Holm across the 4).
    Amendment 9's pass rule is reported as its verdict.

**Trial simulations (V4)** are design calculations, not hypothesis tests:
- **Type I error:** a Clopper-Pearson 95% interval from k = round(rate × 2,000).
- **Sample-size ratio:** a Monte Carlo 95% interval from a parametric bootstrap of the two simulated powers.
  - Each power is drawn as binomial out of 2,000, independently; that overstates the variance, because both come
    from the same simulated trials.
  - 10,000 draws, seed 1, each passed through `toolkit.trial.n_for_power`.

**Membership inference** keeps the gate's pre-specified rule; no P value is added.

**The clinician study:** only its own pre-specified tests, and only when formal results exist.

## Amendment 14 (2026-10-11, reviewers' requests on the manuscript draft, before any of these statistics was computed)

**Scope.** Two reporting analyses for the manuscript, on CPU, from per-anchor predictions and results already kept.
- **No new data, rollouts or GPU work.**
- **Nothing pre-specified changes:** no estimate, interval or verdict. The primary family (amendment 13, family 1)
  keeps its Holm adjustment.
- **The code:** `tools/paper_stats.py`, as amendment 13, writing to the same outputs.

**11. The breadth of the comparison, every scored endpoint** (post hoc: defined after A1's per-program intervals were
known; reported as exploratory).
- *The comparisons:* every binary program A1 compared with LightGBM (pooled), on each of A1's three pre-specified
  anchor sets: R (53 programs), L24 (56) and L48 (56); 165 comparisons. These are the pre-registered primary and
  secondary endpoints at the pre-specified configuration: the Twin's share of 32 rollouts against LightGBM, on each
  program's common anchors (`tools/paper_compare.py`, stored in `a1/test_<design>/compare.json` as
  `twin_minus_lgbm` and its interval).
- *The P values:* exact, as amendment 13. The stored intervals (1,000 resamples, seed 0) are reproduced first, to within
  1e-12, or the script stops; then 10,000 resamples with seed 1.
- *Multiplicity:* Benjamini-Hochberg at 5% across the 165 comparisons, reported as q values. It is valid under
  positive dependence, which the shared anchors make plausible. The number significant under Holm is reported beside
  it as a sensitivity.
- *Grouping for display:* the catalog's own families (`src/icutwin/configs/factory/catalog.yaml`): mortality, organ failure,
  sepsis, intervention onsets, discharge, threshold crossings, and the remaining stay.
- *Summaries:* for each anchor set and for all three together, the median difference across programs and the share
  of programs with a positive difference.
  - Each summary's 95% interval comes from one joint bootstrap: the held-out stays are resampled within datasets
    once per resample and applied to every program of every anchor set (1,000 resamples, seed 2).

**12. Label efficiency** (pre-specified in A1, "the n at which it matches the Twin's AUROC per primary endpoint
(interpolated; 'not reached' stated)"; the interpolation is fixed here, before it is computed).
- *The data:* A1's stored curves (`a1/test_<design>/baselines.json`, `label_efficiency`). LightGBM per dataset was
  trained on 100, 300, 1,000 and 3,000 of the dataset's train stays and on all of them (up to 5,000). Its AUROC on
  the dataset's held-out anchors is compared with the Twin's per-dataset AUROC (`a1.json`, `datasets`).
- *The endpoints:* the 12 pre-registered primary endpoints, on their designs (R: 9; L24: 3).
- *The n to match:* the first training size at which LightGBM's AUROC reaches the Twin's. The match point is
  interpolated linearly in AUROC against log10(n), between that size and the one before.
  - At 100 already: "at most 100".
  - Never, including all train stays: "not reached (n stays)".
  - The AUROC values at each size were already reported in `results.md`; only the matching sizes are new.
- *Summary:* the share of endpoint-dataset cells needing more than 1,000 labelled stays, or never matching.
  Descriptive, with no test.

## Amendment 15 (2026-10-11, the study lead's request for benchmark-style rankings, before any of these statistics was computed)

**Scope.** Rank summaries across endpoints, as machine-learning benchmarks report them.
- *Post hoc and exploratory:* defined after the per-endpoint AUROCs were known.
- **No new data or GPU work.** Computed from stored AUROCs (`f10/comparators.json`, `f1/*/f2.json`,
  `a1/test_*/compare.json`).
- **Nothing pre-specified changes.**
- **The code:** `tools/paper_stats.py` (`--amendment15`).

**13. Ranks on the primary endpoints** (the follow-up anchors).
- *Endpoints:* the 12 of the 13 primary endpoints on which every learned comparator was fitted. Death within 24 h at
  the landmark is left out, because TabPFN was not fitted there.
- *Predictors:* every predictor scored on all 12:
  - Didymos's current estimate (the hazard estimate where it applies, else the sampled estimate at 128 futures);
  - the Didymos probe;
  - the 13 learned comparators (amendment 10);
  - SOFA, NEWS, MEWS, qSOFA and SIRS.
- *Ranks:* rank 1 is the highest AUROC on an endpoint; ties take mid-ranks.
- *Reported:* each predictor's mean rank, with a 95% interval from 10,000 bootstrap resamples of the endpoints (seed
  3), and the endpoints on which it ranks first.
- *Tests:* a Friedman test across the predictors. Then, for Didymos against each other predictor, a two-sided Wilcoxon
  signed-rank test on the 12 paired AUROC differences, Holm-adjusted across the comparisons.

**14. Ranks across every scored endpoint** (A1's 165 comparisons, pre-specified anchors).
- *Predictors:* those scored on every program: Didymos (32 futures), LightGBM pooled, LightGBM per dataset,
  logistic regression and the Didymos probe, on each program's common anchors.
- *Ranks and tests:* as item 13, with the 165 programs in place of the endpoints.
- *Stated with it:* the programs share anchors and are correlated, so these tests describe consistency across tasks.
  They are not independent replications.
- *Clarification (2026-10-11, after the first run):* LightGBM per dataset was listed here in error. It was fitted only
  on the 15 primary program-design pairs, so it is not "scored on every program".
  - The first run kept all five predictors and so fell back to those 15 programs, contrary to this item's scope. On
    them, LightGBM pooled ranked first (mean rank 1.73) and Didymos second (2.87). That run is superseded and
    reported here.
  - The item's rule leaves four predictors on all 165 programs: Didymos, LightGBM pooled, logistic regression and
    the Didymos probe.

**The leaderboard (Table 2):** item 13's AUROCs, with each endpoint's best in bold and each predictor's mean rank.

## Amendment 16 (2026-10-11, the study lead's request to address three gaps named in the manuscript's Discussion, before any of these analyses was run)

**Scope.** Three new analyses, each answering a gap that the manuscript's Discussion names. The fourth gap, the lack of
a prospective evaluation, is left to the text.
- **Nothing pre-specified changes:** no estimate, interval or verdict of A1-C4, amendments 1-15 or the follow-ups.
- **Inference only:** MUC32_main is not trained or fine-tuned. Aggregates only; rollouts are deleted after their
  analyses.
- **New seeds:** 16 for every design draw and every rollout, so the results are replicates, not re-uses.
- **Results:** `docs/paper/results_followup.md` (new sections), its tables and the statistics outputs. Figures and
  manuscript text are not part of this amendment.
- **One look preceded this amendment:** a count-only audit of the two new archives, read in memory: column coverage,
  item and parameter frequencies, stay and death counts. No model was run, no label was computed and no prediction was
  made. It informed which endpoints each database can define.

### 16a. Transfer to two databases never seen in training

- **Databases** (`docs/paper/data.md`, "External databases"): RICD 3.0, the Russian Intensive Care Dataset (one
  centre for intensive care and neurorehabilitation, Moscow, 2017-2025), and zigongICU 1.1 (Zigong Fourth People's
  Hospital, China, ICU patients with infection). Neither was seen in training, tuning or model selection.
- **Cohort:** every adult ICU stay, over the paper's window (48 h before ICU admission to ICU discharge).
  - RICD: each ICU period of the patient pathway.
  - zigongICU: a run of ICU-department transfers, merged when one starts within 1 h of the previous one's end.
  - All stays are test stays; nothing is trained on them.
- **Harmonization:**
  - new CLIF adapters (`src/icutwin/clif/adapters/{ricd,zigong}.py`) with versioned mapping tables, every row
    recording its provenance;
  - values in CLIF reference units;
  - MUC32_main's frozen vocabulary (`data/tokens/full_r4/vocab.json`, never refitted); unmapped items become their
    domain's OOV token and are counted;
  - header fields carry only values known at ICU admission;
  - the era is set to unknown: neither database's period maps to the vocabulary's era tokens without borrowing another
    database's.
  - Approximations, stated in the maps and reported:
    - zigongICU: MAP = (SBP + 2 DBP)/3 from simultaneous non-invasive pressures where MAP is not charted. A vasopressor
      episode starts at the first administration record, because the drug table has times but no rates or stops. RRT
      starts at the first hemofiltration replacement-fluid record.
    - RICD: ventilation status comes from the daily SOFA assessment's ventilation flag, the only ventilation source.
  - *Found at harmonization* (2026-10-11, before any label was computed or any future drawn; the adapters' docstrings
    and maps give the details):
    - zigongICU's nursing chart records each row's date, not its hour: a stay's chart times share one time of day,
      while its laboratory, drug and transfer times carry hours. The rows of a day are in recorded order and are spread
      over the day's charted interval, each at the end of its slot. Vital signs, ventilation, consciousness and urine
      output therefore have no recorded hourly timing there, and the endpoints defined from them (new IMV, AKI,
      hypotension) fail the availability rule's first half on zigongICU.
    - zigongICU's drug table lists prescriptions without doses. Vasoactive and infusion-only drugs become presence
      episodes: a start, renewals within 25 h, and a stop 6 h after the last prescription, without a dose.
    - RICD's prescriptions are daily orders. An infusion episode starts at the first order and stops 12 h after the
      last. The recorded rate's unit is not a dose unit, so infusions carry no dose. Sources other than monitoring
      record whole hours since admission and are placed at the end of their hour.
    - RICD releases no dates. A patient's hospitalizations are placed one after another, 30 days apart, each at its
      admission's time of day (from the monitoring clock).
    - The source behind each endpoint, and whether its timing is to the hour, is fixed in `tools/paper_external.py`
      (`SOURCES`) before the labels.
- **Prompt:** the database and site tokens are set to unknown (F9's transform; training set both to unknown in 10% of
  windows), so no support mask applies and the clock uses the pooled sketches. Token caps come from each database's
  own sequences (A1's rule).
- **Anchors** (design seed 16):
  - *Rolling:* A1's design, 500 per database: length-of-stay bands, half enriched for an organ-support start or ICU
    death in the next 48 h, inverse-probability weights.
  - *L24e:* a case-cohort sample of the 24-hour landmark (stays in the ICU and alive at 24 h). Up to 350 in-hospital
    deaths and 350 other stays per database, at random, each stratum weighted by the inverse of its sampling fraction.
  - Anchors whose prompt clock disagrees with their hour by more than 6 h are left out and counted (F1's rule).
- **Futures:** 128 per anchor, T = 1, rollout seed 16. Caps: A1's (the 72-hour quantile) on rolling anchors, four
  times A1's (the 7-day quantile) on L24e.
- **Estimators:** the hazard estimate (F2's E2) for death, new IMV, a new vasopressor and new RRT; the sampled estimate
  otherwise. A program is scored when at least half of its futures are valid.
- **Endpoints:** the 13 primary endpoints, wherever the database can define them.
  - *The availability rule,* applied to real labels only and recorded before any future is drawn: the defining source
    exists with timing to the hour, and at least 20 anchors of the design are positive.
  - *Expected from the audit:* zigongICU all except sepsis (no culture data); RICD all except sepsis, new IMV (a daily
    ventilation flag only) and new RRT (no source).
- **Comparators,** on the same anchors:
  - *Trained on the seven databases and transferred:* LightGBM pooled and logistic regression, refit with A1's recipe
    and seed (a new database's one-hot columns are all zero); TabPFN-3.5 on the primary endpoints (amendment 10's
    recipe); the Didymos probe.
  - *Clinical scores* computed from the new data with amendment 10's code: SOFA, NEWS, MEWS, qSOFA and SIRS, and SAPS
    II and OASIS at the landmark where their components exist. RICD's recorded SOFA (the latest in the 24 h before the
    prediction time) serves as the database's own score.
  - *A local reference:* LightGBM trained on the database's own labelled stays (A1's recipe), 5-fold cross-fitted by
    patient, so each anchor is scored by a model that never saw its patient.
- **Read-outs:**
  - AUROC and O/E per endpoint and database, with 95% intervals from 1,000 weighted bootstrap resamples of stays, and
    paired differences with every comparator.
  - Didymos's internal AUROC on the same endpoint (follow-up anchors) shown beside it, descriptively.
  - *Secondary:* a logistic recalibration (intercept and slope on the logit of the risk), cross-fitted within each
    database by the same 5 folds, with O/E and AUROC after it.
- **Expected:** Didymos at or above the transferred LightGBM on most endpoints and at or below the local reference; O/E
  further from 1 than on the seven databases, and nearer 1 after recalibration within the site.

### 16b. The 48-hour landmark with the follow-up's settings (L48x)

- **Anchors:** a case-cohort subset of A1's `test_L48` (3,500 anchors): every anchor whose stay died in hospital, and
  354 of the others at random (as L24x), weighted by the inverse inclusion probability. F1's clock check applies, and
  exclusions are counted.
- **Futures:** 128 per anchor, T = 1, rollout seed 16, at four times A1's caps (at most the context minus the prompt),
  so that A1's cap can be applied afterwards by truncation.
- **Hazards:** on the futures and on the real futures (`--real --long`), as F2.
- **Endpoints:**
  - A1's primary endpoints at L48: ICU death, in-hospital death and an ICU stay over 7 days;
  - death within 24 h of the landmark (added, as at L24x);
  - new IMV, vasopressor and RRT (secondary).
  - The hazard estimate is used for ICU death, death within 24 h and the three treatments; the sampled estimate for
    in-hospital death and the stay over 7 days.
- **Read-outs, as F1-F3 at L24x:**
  - AUROC and O/E;
  - paired differences against A1's 32-future value on the same anchors, LightGBM pooled, TabPFN-3.5 (where fitted)
    and the probe;
  - ICU death at A1's cap against four times it;
  - step calibration along real futures and generation drift (D1-D3).
- **Comparators:** A1's stored predictions, reindexed to the subset (no refit). For death within 24 h at the landmark,
  the eight learners of amendment 10 and TabPFN-3.5 are fitted with their recipes.
- **Expected:** as at L24x, the differences against LightGBM at L48 shrink (ICU and in-hospital death), and censoring
  falls at four times the cap.

### 16c. Attention knockout against the generator (a replicate of F4's generator side)

- **Prompts:** F4's 160 anchors (40 per endpoint) and its 2,698 saved prompts, the originals and every group's edit
  (`runs/paper/f4/attrib/rollout_prompts.parquet`).
- **Futures:** 48 per prompt, T = 1, rollout seed 16, at F4's recorded caps (`runs/paper/f4/R/design.json`).
- **The generator's effect** of each group, computed exactly as `tools/paper_c2b.py` computes it:
  - death and IMV: the mean over futures of the hazard increments summed over the first 24 hourly bins;
  - AKI and hypotension: the share of positive labels among valid futures;
  - at least 8 valid futures per prompt and 4 effects per anchor.
  - The split-half ceiling (futures 0-23 against 24-47, Spearman-Brown corrected) is recomputed, and every per-group
    effect and both halves are saved.
- **Knockout,** on all 160 anchors (prompts up to 7,607 tokens): amendment 9's mask (no token outside a group attends
  to the group's tokens, in all layers) and score (the fall in the probe's logit), on the groups editing shares.
- **Agreement** with the redrawn effects, for knockout and for editing on the same groups:
  - Spearman per anchor (median), top-3 overlap, and sign agreement on groups whose effect's 95% interval excludes 0;
  - each against the recomputed ceiling;
  - F4's original editing agreement reported beside it (a different draw).
- **Expected:** knockout's agreement near editing's, since both remove a group's information; both below the ceiling.

### Statistics (numbered after item 14; amendment 13's rules)

- **15.** 16a: Didymos minus the transferred LightGBM, every evaluable endpoint and database; Holm across all of them.
- **16.** 16a: Didymos minus the local cross-fitted LightGBM; Holm.
- **17.** 16a: Didymos's sampled estimate (as family 4) minus the best clinical score per endpoint and database, chosen
  after the fact; Holm.
- **18.** 16b: Didymos minus LightGBM on the 4 endpoints; Holm.
- **19.** 16b: the follow-up minus A1's pre-specified estimate on the same anchors; Holm.
- **20.** 16c: per anchor, ρ(knockout, generator) minus ρ(editing, generator); two-sided Wilcoxon signed-rank, pooled
  and per endpoint, Holm across the 4 endpoints.
- **P values** for families 15-19 are exact paired-bootstrap values (10,000 resamples, seed 1, strata = database),
  because per-anchor estimates are now saved for every estimator, including the hazard estimate. The stored
  1,000-resample intervals are reproduced first.
- **Families 1-4 are not changed:** the new endpoints and designs are not added to the headline set.

**Compute:** about 20 GPU-hours. 16a about 11 (westd, RTX 5090), 16b about 6 (westc, RTX 4080, which holds the test
store), 16c about 2 (westd); knockout takes minutes.

## Amendment 17 (2026-10-12, the study lead's request to make the model perform strongly on new databases, before any adapted model was trained or any new future drawn)

**Scope.** Amendment 16a measured zero-shot transfer to RICD and zigongICU. This amendment tests three ways to do
better there: an improved encoding (inference only), continued unlabelled training on a new database's own stays, and
few-label methods. Didymos 1.0 (MUC32_main) is not changed; the adapted model is a new version. Nothing pre-specified
before this amendment changes. Results only (no figures, no manuscript text). GPU work on the RTX 5090 host (westd)
only. Seeds 17. Amendment 16a's results were read before this amendment; that is why its test half is locked below.

### 17a. The locked split

- **By patient, per database:** u = the first 8 bytes of blake2b("17:<database>:<patient id>") over 2^64.
  - **TEST** (u < 0.50): locked. Never used for fitting, selection, calibration or stopping.
  - **FT** (0.50 <= u < 0.85): unlabelled continued training. No outcome label is used anywhere in training.
  - **LABEL** (u >= 0.85): labelled stays for the few-label methods. Never seen by training.
- **Test anchors:** amendment 16a's kept anchors whose patient is in TEST, at 16a's hours. The rolling design keeps its
  inclusion weights. The landmark's case-cohort weights are recomputed within TEST: per database and stratum, the TEST
  landmark anchors of the stratum over the TEST anchors sampled from it.
- **Labels:** amendment 16a's real labels (its encoding) for every arm, so the truth does not change with the
  encoding. The labels under the improved encoding are compared with them and the agreement reported.
- **LABEL anchors:** 16a's rules within the LABEL stays (seed 17): the rolling design, 250 per database; the 24-hour
  landmark's case-cohort sample, up to 125 in-hospital deaths and 125 others per database.
- The split file's sha256 is recorded in every output.

### 17b. The arms (all scored on the TEST anchors with 16a's endpoints, estimators, caps and availability rule)

- **Z0, zero-shot:** amendment 16a's predictions on the TEST anchors (no new futures).
- **Z1, zero-shot with an improved encoding** (inference only):
  - an opt-in back-off in the encoder for codes the frozen vocabulary lacks, versioned with provenance
    (`tokenization/backoff.yaml`): an infusion's unit to `UNK`, then its route; an intermittent dose's unit and route
    to the nearest kept variant of the same drug; `OUTPUT` stool to `gi_output_other`; an unmapped location to the
    nearest kept location;
  - off by default: the seven databases encode byte-identically;
  - the OOV share per domain before and after is reported.
- **F1, the adapted model (Didymos-ext):**
  - MUC32_main's weights, trained further with its next-token objective on the FT stays (Z1's encoding; dataset and
    site tokens set to unknown, as the prompts carry them), mixed 1:1 by windows with the seven databases' training
    windows (their 10% unknown augmentation unchanged);
  - a fixed grid: learning rate {3e-5, 1e-4} x length {1, 3} passes over the FT pool, warm-up 2%, cosine to 10%; the
    cell with the lowest next-token loss on a held-out 10% of FT patients is chosen (no TEST or LABEL data enter
    the choice);
  - forgetting: the next-token loss on the seven databases' tuning split before and after (reported);
  - the M7 gate is run for the adapted run before any of its futures are drawn, and a membership check of FT
    against TEST stays of the new databases is reported.
- **Few-label methods**, fitted on LABEL anchors only and applied to TEST:
  - **F1+R:** a logistic recalibration (intercept and slope on the logit of the risk) per database and endpoint; the
    intercept alone where the LABEL anchors have fewer than 10 positives. Also Z1+R.
  - **F1+S:** stacking of F1 with a local LightGBM (a logistic regression on the two logits).
  - **Local references:** LightGBM with A1's recipe trained on the LABEL stays, and on the FT and LABEL stays together
    (every adaptation label: the strongest local comparator).
  - **Label efficiency:** n labelled LABEL stays in {25, 50, 100, 200, all}, 20 random draws each, for F1+R and the
    local LightGBM; the n at which the local LightGBM matches F1+R (A1's log-n interpolation).
- **Futures:** 128 per anchor, rollout seed 17, prompts with the database and site tokens unknown, 16a's caps.

### Statistics (families 21-24; amendment 13's rules; exact paired bootstrap, 10,000 resamples, seed 1)

- **21.** Each arm minus Z0 (Z1, F1, F1+R), per endpoint and database; Holm within each arm.
- **22.** F1+R minus the local LightGBM trained on every adaptation label; Holm.
- **23.** F1+R minus TabPFN-3.5 transferred (16a's predictions on the TEST anchors); Holm.
- **24.** Mean ranks over the TEST pairs (amendment 15's method), with and without the local models.
- An endpoint with fewer than 10 TEST positives in a database is reported descriptively only.
- **Expected:** Z1 above Z0 mainly on RICD's medication-driven endpoints; F1 above Z1 broadly, most on RICD; F1+R with
  O/E near 1; F1+R closing most of the gap to the local LightGBM, which leads only beyond about 200 labelled stays.

**Compute:** about 40 GPU-hours on westd: the grid about 8, the gate about 2, Z1 and F1 on the TEST anchors about 12
each, the LABEL anchors about 6.

### 17c. The adapted model's gate: P1 at a larger size (2026-10-13, the study lead's decision, before any F1 future)

- **What happened:** the M7 gate of the chosen run (E17_lr0.0001_p3; the grid, below in the results) was
  INCONCLUSIVE on one test only: P1 (membership by loss) on eICU, TPR at 1% FPR 0.0227 [0.0127, 0.0303] against the
  0.03 limit (MUC32_main 0.0240 [0.0127, 0.0297], PASS); every other P1 group and P2-P5 passed with valid controls.
  The evaluation batch was 32,768 tokens on the 5090 (`eval.yaml`'s 131,072 needs a 96 GB card; the loss does not
  depend on it beyond numerics).
- **Rule, as amendment v2 resolved M32_main's inconclusive P5:** P1 is run again on the same frozen weights with up
  to 10,000 members and 10,000 non-members per database (`p1.cap` 10000, from 3,000; the smaller of the train and
  held-out stays where fewer), the same attack, criteria, seed and pooling. It replaces the inconclusive P1; P2-P5
  stand as run; the gate then follows the same rule. Both P1 results are reported. If the gate is then PASS, F1's
  futures are drawn as 17b registers; otherwise F1 is reported without futures.

### 17d. The study lead's exemption for F1's futures (2026-10-13, after 17c's result)

- **17c's result:** with P1 at up to 10,000 per side the gate was again INCONCLUSIVE, now on AUMC only (TPR at 1%
  FPR 0.0226 [0.0153, 0.0313]; 2,747 per side both times, AUMC's held-out size, with a different draw of members;
  eICU 0.0157 [0.0125, 0.0194] and every other group and the pooled test passed). By 17c's rule F1 had no futures.
- **The study lead's decision** ("Owner exemption for F1"): F1's futures may be drawn for this research analysis despite
  the INCONCLUSIVE gate. Scope: amendment 17's F1 futures on the TEST and LABEL anchors only, scored as 17b registers
  and deleted after their analyses, as every rollout of Paper P. No twin, bank, toolkit session or other synthetic
  data is produced from F1; the producers' interlock (`generate.producer.require_gate`) stays strict. A FAIL would
  not be covered.
- **Record:** `runs/audit/m7_<run>/owner_exemption.json` names the run, its weights' sha256, the gate status it
  covers (INCONCLUSIVE) and this scope; `tools/server/paper_rollouts.py --gate-exemption` checks it against the
  weights and the report. The results report the gate as INCONCLUSIVE with the exemption, beside both P1 results.

## Materials

`docs/paper/`: the data, preprocessing, tokenization, architecture, training, evaluation protocols and results, in
tables, every number with its source (file, git sha) and an annotation (what it measures, its control, its caveat).
