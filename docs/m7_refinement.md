# M7 round 3: refining the model (pre-registered 2026-10-06)

Status: written and committed **before** any measurement of this round (the diagnosis included). The targets, the
candidates and the decision rule below are fixed by this commit; results are added below them, unchanged.

## Why

Round 2 (`docs/m7_authenticity.md`) left M32_main (123.5M parameters, 32k context, 3 / 2.5 passes, M7 gate PASS) as
the candidate:

- **Authenticity:** M32_main re-audited with #111 meets 24 of 35 per-dataset targets. MIMIC, eICU and NWICU meet
  all of theirs; AmsterdamUMCdb, HiRID, SICdb and INSPIRE fail A1 and others. Pooled, A6 and A7 fail (Vendi ratio
  0.90; density; RRT utility 0.88).
- **Generalization:** tuning NLL plateaus at 1.56. A 2.2-times larger model (ML32_main) reached the same NLL and only
  learned the training stays better (membership AUC +0.010 to 0.016). Passes are bound by the privacy gate.
- **Treatment frequency:** new onset after the 24-hour landmark, twins against real stays:

| Treatment | Twins | Real |
|---|---|---|
| RRT | 7.0% | 3.1% |
| IMV | 10.2% | 6.0% |
| Vasopressor | 10.7% | 6.8% |

Rollouts from real 24-hour histories show the same pattern (observed over expected; MIMIC, eICU and SICdb):

| Outcome | O/E |
|---|---|
| RRT | 0.47-0.87 |
| Vasopressor | 0.71-0.97 |
| ICU stay over 7 days | 0.68-0.82 (MIMIC, eICU) |
| Death | 1.18-1.21 (MIMIC, eICU) |

Free-running trajectories last longer and are treated more than real ones, even in datasets that meet every
authenticity target. **Working hypothesis:** compounding errors in free-running generation (exposure bias), on top
of per-dataset under-fitting at privacy-bound passes. Step 0 tests it before anything is trained.

## The study lead's decisions (2026-10-06)

- **Objective:** maximum likelihood plus robustness: regularization, weight averaging and input-noise augmentation.
  No preference tuning.
- **Sampler:** twin-level calibration of treatment frequency is allowed.
- **Passes:** flexible, under the measured membership signal. The gate's limits are unchanged; changing them would be a protocol
  amendment adopted before its run.
- **Targets:** treatment frequency and forecast calibration are selection targets for this round. The bank criteria
  stay A1-A7.

**Augmentation.** All seven datasets are fully used, and held-out and tuning are reserved for the gates. So
augmentation means training-time transforms that leave the targets, and with them the learned distribution,
untouched:
- existing: header shuffle and unknowns, dataset unknown, goal-field subsets (#110);
- new: input-noise augmentation (R2).

Transforms that change targets are excluded: value or time jitter would distort the distribution, and oversampling
rare stays would raise the membership signal.

## Step 0: diagnosis (M32_main as is)

- **0a. Teacher-forced hazard calibration** (`eval.calibration`). Up to 2,000 held-out stays per dataset. Expected
  = the model's probability of each outcome's tokens, summed over the positions at risk. Observed = the stays where
  it happens. Outcomes: first IMV, vasopressor and RRT event; ICU discharge; ICU discharge 'dead'. Reported per ICU
  day.
- **0b. Rollout horizons** (`eval.forecast`, `Reader.within`). Observed over expected at 6, 24 and 72 hours after the
  cutoff and to the end. 300 held-out stays per dataset, 16 rollouts each, all seven datasets.
- **0c. Per-dataset treatment rates and runaway shares** of 1,000 twins per dataset, round 2's sampler, twin seed 1
  (`quality.prevalence`). This is also a third audit of the reference, for the noise.
- **0d. Checkpoint averaging.** Tuning NLL of the weight average of M32_main's last 2, 3 and 5 checkpoints.

**Reading (fixed now):**

| Finding | Conclusion | Action |
|---|---|---|
| Teacher-forced O/E within 0.9-1.1 while rollout O/E falls with the horizon | Compounding drift | R2 is the lead candidate |
| Teacher-forced O/E outside 0.85-1.15 for a treatment in a dataset (95% interval excluding 1) | Per-step miscalibration | QA of that dataset's treatment tokens before any training (a data-quality entry; report to the study lead) |
| Averaging lowers the tuning NLL by at least 0.002 | Averaging helps | EMA in every candidate; otherwise off |

## Candidates (preset M; M32_main's data, recipe and seeds; goal-field subsets on)

- **R1, regularization:**
  - residual dropout 0.1 on each layer's attention and MLP outputs and on the embeddings, in training only;
  - EMA of the weights (decay 0.999) if 0d supports it, used if its tuning NLL is lower.
- **R2, R1 plus input-noise augmentation** (parallel scheduled sampling; Duckworth et al. 2019, Mihaylova & Martins
  2019):
  - at a share q of timeline positions (0 → 0.15 over the first half of training), the input token is replaced by
    one drawn at temperature 1 from the model's own prediction, if of the same class (event or interval);
  - the labels stay the real tokens; prefix, header, goal, state and end blocks are never touched.
- **R3, flexible passes** (only if the membership signal leaves room): the better of R1 and R2 with per-dataset passes raised
  while the projected P1 upper bound stays at or below 0.58. The working threshold is adjustable by the study lead; the
  gate's limit stays 0.60. At most +1.5 passes per dataset, projected from measured per-pass slopes.

**Screen of every candidate:**
- tuning NLL;
- P1;
- 300 twins per dataset (length-resampled): S1, runaway share, top-event p99, first-24-hour unigram AUC
  (`authenticity.explain`).

## Sampler: twin-level calibration of treatment frequency

The twins are raked to the training stays' margins:
- the ICU length band (round 2);
- new-onset IMV, vasopressor and RRT after the 24-hour landmark.

Iterative proportional fitting gives the minimum-KL weights that match these margins. Then 1,000 twins per dataset
are drawn without replacement, proportional to the weights, from three times as many. This corrects the
population's margins, not the model's dynamics. S1 is measured on twins resampled for length only.

## Targets and decision

**Bank criteria:** A1-A7 (`docs/m7_authenticity.md`), unchanged.

**Selection targets (this round):**
- **S1 treatment frequency:** new-onset IMV, vasopressor and RRT rates after the 24-hour landmark, twins (resampled
  for length only) against the same held-out sets, within ±25% per dataset. Cells with fewer than 30 real events are
  pooled across datasets.
- **S2 forecast calibration:** rollout observed over expected within 0.8-1.25 for death, IMV, vasopressor, RRT and
  ICU stay over 7 days, per dataset (same pooling).
- **S3 generalization:** tuning NLL at most M32_main's + 0.01 (1.571). The member-non-member NLL gap (P1) is
  reported.

**Noise control:** each decision audit runs twice, with twin seeds 0 and 1 on the registered real sets, and
decisions use the mean of the two.

**Finalist:** the screened candidate with the most S1 cells met. Ties go to the lower mean first-24-hour unigram
AUC, then to the lower tuning NLL. Candidates failing P1 are out.

**Adoption.** The finalist replaces M32_main if:
- its M7 gate passes;
- S3 holds;
- no pooled A6/A7 part that the reference meets fails;
- against the reference (M32_main with the same sampler, the same two audits and the same rollouts), it is better
  on at least one axis and no worse on the other:

| Axis | Measure | Better | No worse |
|---|---|---|---|
| Authenticity | Mean A1-A5 count over 2 replicates | Reference + 2 | Reference − 1 |
| Calibration | S1 + S2 cells met | Reference + 3 | Reference − 1 |

Otherwise M32_main stays. Either way the deliverable bank stays off until A1-A7 hold.

**Out of scope:** preference tuning (SPIN/DPO), larger models, label smoothing or distillation, truncated or
penalized sampling, changes to the held-out or tuning splits.

## Results: step 0 (2026-10-06, 08:05-09:22 server time; M32_main as is)

**0a. Teacher-forced hazard calibration**, 2,000 held-out stays per dataset (raw), 1,000 under the generation mask
(grammar and support, a quarter of the positions at risk). Observed over expected after the 24-hour landmark
[95% interval]:

| Masked O/E | IMV | Vasopressor | RRT | ICU discharge | ICU death |
|---|---|---|---|---|---|
| MIMIC | 0.62 [0.40, 0.92] | 0.60 [0.41, 0.86] | 0.49 [0.31, 0.73] | 1.01 | 0.96 |
| AmsterdamUMCdb | 0.71 [0.41, 1.13] | 1.28 [0.87, 1.82] | 0.48 [0.34, 0.66] | 0.98 | 0.79 [0.63, 0.98] |
| eICU | 0.88 [0.52, 1.39] | 0.53 [0.29, 0.90] | 1.07 [0.64, 1.70] | 0.94 | 0.99 |
| HiRID | 0.62 [0.35, 1.00] | 0.67 [0.45, 0.97] | 0.28 [0.18, 0.43] | 0.84 [0.77, 0.92] | 0.67 [0.45, 0.95] |
| SICdb | 1.01 [0.69, 1.43] | 0.71 [0.46, 1.05] | 0.64 [0.43, 0.91] | 0.92 | 0.85 |
| NWICU | 0.54 [0.36, 0.80] | 0.38 [0.25, 0.56] | 0.57 [0.34, 0.89] | 1.00 | 1.10 |
| INSPIRE | 0.18 [0.02, 0.67] | 0.84 [0.38, 1.59] | 0.19 [0.10, 0.35] | 0.99 | 1.81 [0.83, 3.44] |

- **By ICU day:** treatment starts are calibrated on day 1 (masked O/E 0.86-1.20 in most cells) and increasingly
  over-predicted after it, in nearly every dataset (IMV and RRT from day 2, vasopressors from day 2 except in
  AmsterdamUMCdb).
- **ICU discharge and death:** calibrated throughout, except HiRID.
- **The raw measure** (without the masks) is close to the masked one (MIMIC RRT 0.53, NWICU vasopressor 0.38).
  The masks change little except eICU RRT (0.40 raw, 1.07 masked).

**Reading (pre-registered):** per-step miscalibration of treatment onset. The model over-predicts starts in real
histories from day 2 on, before any generation drift. So the QA of the treatment tokens came next, and the candidates
wait for the study lead.

**Treatment QA** (`calibration --qa`, 1,000 held-out stays per dataset):
- **By position of the predicted token:** observed over expected about 1 within the first 2,048 tokens (IMV
  0.88-1.05; vasopressor 0.93-1.20 except NWICU 0.66; RRT 0.64-0.93 except HiRID 0.53 and INSPIRE 0.36), falling
  beyond. Examples at 2,048-8,192 tokens:

| Dataset and treatment | Expected | Observed |
|---|---|---|
| MIMIC IMV | 9.6 | 1 |
| NWICU IMV | 12.3 | 1 |
| HiRID RRT | 47.5 | 8 |
| AmsterdamUMCdb RRT | 37.0 | 11 |

  The model does not learn that a patient without a treatment after days in the ICU is unlikely to start one: it
  must infer "never had it" across thousands of tokens.
- **Tokens carrying the excess:**
  - CRRT `crrt_mode_category//UNK`: HiRID 102 expected first events against 43 observed; INSPIRE 81 against 27.
  - AmsterdamUMCdb CRRT blood-flow deciles: 47.5 against 34.
  - NWICU norepinephrine `going` records as first events: 26 expected, 0 observed. Real NWICU always charts a start
    first. The grammar allows `going` anywhere, because other datasets chart infusions already running.

**0d. Checkpoint averaging:**

| Average of the last checkpoints | Tuning NLL |
|---|---|
| Final weights alone | 1.5611 |
| Last 2 | 1.5594 |
| Last 3 | 1.5597 |
| Last 5 | 1.5640 |

The best gain is 0.0017, under the 0.002 rule, so EMA is off.

**0b (rollout horizons) and 0c (replicate audit):** running; added when done.

**Time in the ICU, not context length** (the study lead's question; QA by ICU day x position of the predicted token, and
M_main, 8k windows with state blocks, on its own r2 tokens):
- **The excess appears within the first 2,048 tokens once those tokens span several days.** NWICU, whose first week
  fits there (observed/expected):

| NWICU, first 2,048 tokens | Day 1 | Day 2 | Days 3-7 |
|---|---|---|---|
| Vasopressor | 0.84 | 0.50 (13/26) | 0.26 (11/42) |
| IMV | 1.17 | 1.05 | 0.47 (12/26) |
| RRT | 1.31 | 0.62 | 0.34 (7/20) |

  MIMIC, eICU, HiRID and INSPIRE decline the same way inside the first 2,048 tokens.
- **Within a day, later positions are not systematically worse** (e.g., SICdb RRT on days 3-7: 0.55 at short
  positions, 0.58 at long ones).
- **M_main shows the same day profile.** RRT by day (1 / 2 / 3-7 / 8+):

| RRT by day | M_main | M32_main |
|---|---|---|
| MIMIC | 0.81 / 0.58 / 0.44 / 0.37 | 0.90 / 0.64 / 0.54 / 0.37 |
| eICU | 1.00 / 0.42 / 0.31 / 0.14 | 1.13 / 0.53 / 0.41 / 0.19 |
| HiRID | 0.66 / 0.34 / 0.30 / 0.11 | 0.70 / 0.34 / 0.28 / 0.13 |

**Reading:** the model does not lower the chance of a start as untreated ICU days accumulate. Elapsed ICU time is
only implicit in the tokens (a sum of interval tokens), and so is the absence of a treatment. Neither the 32k context
nor the 8k windows with state blocks remove this. A status block at each ICU-day boundary should carry the day as
well as each treatment's state (never, active, ended).

## Amendment 1 (2026-10-06, after step 0, before any candidate was trained)

**Basis.** Step 0 found:
- treatment onset over-predicted step by step from the second ICU day, following ICU time and not context length;
- errors compounding in rollouts: ICU death under-predicted 1.3-2.2 times from 24 hours on, while teacher-forced
  death is calibrated.

The study lead then asked for a new round of training with the best modifications to the data, the training or the
architecture. Candidates R1-R3 are replaced by:

- **The clinical course made explicit (data and representation; spec `clock`, #112).**
  - After an interval token that takes the ICU clock into a new day, a block: `[DAY]`, the day, and each
    treatment's state (IMV, vasopressor, RRT: never, active, ended).
  - The grammar automaton computes it from the tokens. The encoder writes it (the r4 tokens: r3 through
    `fsm.insert_clock`), and the sampler is forced to write it. It is conditioning (no loss).
  - Positions grow to CONTEXT + 1,024: the blocks lengthen the longest first windows by up to about 800 tokens.
- **C1 MC32_main:** preset M on the r4 tokens; M32_main's passes, recipe and seeds; goal-field subsets; no dropout,
  no EMA.
- **C2 MCN32_main:** C1 plus input-noise augmentation (rate 0 to 0.15 over the first half of the run), against the
  compounding.
- **Training:** side by side on two GPUs each, four micro-batches per step (M32_main's 0.52M tokens per step).

**Screen of each candidate:**
- tuning NLL (the same target tokens as M32_main: clock tokens are not targets);
- P1;
- **S4 (new): masked teacher-forced calibration of treatment onset after the 24-hour landmark, pooled over the
  datasets per treatment.** M32_main: IMV 0.68, vasopressor 0.64, RRT 0.45; target 0.85-1.15 each. Reported beside
  it: the per-dataset cells within 0.85-1.15 (M32_main: 3 of 21);
- the 300-twin screen: S1, runaway share, top-event p99, first-24-hour unigram AUC.

**Unchanged:** the finalist rule, the adoption rule (the gate, S3, pooled A6/A7, the two axes with two replicates
under the same sampler) and the bank criteria A1-A7. The audits and the harness read the stays without clock
tokens (`quality.draws`, `bare`), so A1-A7 measure the same tokens as for M32_main.

**Added:** a finalist must bring S4 closer to 1 for at least two of the three treatments. Otherwise the blocks did
not address the diagnosed failure, and the round is reported before any adoption.

## Results: amendment 1's candidates (2026-10-06, 10:15-12:01 server time)

Both trained as registered: 2,590 steps of 0.52M tokens on the r4 tokens, the 50-step DDP smoke tests passed, frozen
before any measurement.

| | Tuning NLL | P1 (membership AUC, MIMIC .. INSPIRE) | S4 pooled masked O/E after 24 h: IMV / vasopressor / RRT | Cells within 0.85-1.15 |
|---|---|---|---|---|
| M32_main (reference) | 1.5611 | PASS, 0.527-0.561 | 0.675 / 0.637 / 0.449 | 3 of 21 |
| MC32_main (clock blocks) | 1.5636 | PASS, 0.527-0.560 | 0.705 / 0.656 / 0.471 | 2 of 21 |
| MCN32_main (clock blocks, input noise) | 1.6067 (fails S3) | PASS, 0.516-0.547 | 0.683 / 0.571 / 0.434 | - |

- **MC32_main** brings S4 closer to 1 for all three treatments, but by 0.02-0.03; its ICU-day profile is unchanged
  (IMV 1.00 / 0.70 / 0.80 / 0.46 on days 1 / 2 / 3-7 / 8+, M32_main 1.01 / 0.71 / 0.72 / 0.44). The model does not
  use the explicit day and states to lower the hazard.
- **MCN32_main** fits worse at every evaluation (1.7224 against 1.6881 at step 1,000) and calibrates worse; out
  (S3). Its twin screen was not run (cost; no rule reads it): `logs/train_MCN32_main.exit` was set to skip it in the
  running pipeline, its original kept beside it.
- **MC32_main's twin screen** (300 length-resampled twins per dataset, `runs/eval/MC32_main/screen`): new-onset
  rates after 24 hours, twins against real, pooled: IMV 0.107 / 0.068, vasopressor 0.097 / 0.074, RRT 0.080 / 0.029
  (M32_main's 1,000-twin audit `auth_d`: 0.093 / 0.064, 0.092 / 0.071, 0.071 / 0.035); runaways at most 1%; mean
  first-24-hour unigram AUC 0.586 (300-twin sets: not comparable with the 1,000-twin audits, only with the other
  screens).

**Three checks of the cause** (masked teacher-forced O/E after 24 h, 1,000 first windows per dataset, pooled;
`tools/server/calib_split.sh`, `calib_ckpt.sh`):

| | IMV | Vasopressor | RRT |
|---|---|---|---|
| M32_main on held-out stays | 0.675 | 0.637 | 0.449 |
| M32_main on its training stays (members) | 0.694 | 0.617 | 0.441 |
| M32_main on tuning stays | 0.683 | 0.724 | 0.370 |
| M32_main at step 1,000 (tuning NLL 1.6855) | 0.408 | 0.436 | 0.239 |
| ML32_main (2.2 times the parameters, same passes) | 0.683 | 0.609 | 0.396 |

- **Not memorization or variance:** the training stays are over-predicted as much as unseen ones.
- **Under-fitting:** the excess shrinks steadily with training (step 1,000 to the end: IMV 0.41 to 0.68 while the
  tuning NLL fell 0.124 nats); more parameters at the same passes do not help.
- **Where the excess sits** (`calibration --qa`, all ICU time): treatment starts are calibrated over the whole stay
  for IMV (MIMIC 386 expected against 363 observed, HiRID 583 against 583) but misplaced in time (too few on day 1,
  too many after); for RRT, a dataset's typical first record carries it (HiRID `crrt_mode_category//UNK` 98 against
  43, AmsterdamUMCdb blood-flow deciles, NWICU norepinephrine `going` 16 against 0). The hazard of a start is spread
  over thousands of positions at about 1e-5 each, where the loss barely registers an error.

## Amendment 2 (2026-10-06, after amendment 1's S3, S4 and P1, before their twin screens were read and before any round-4 candidate was trained)

**Basis:** the checks above: per-step under-fitting of rare-event timing, bound by how much the model learns from
each pass, not by capacity, information or memorization. Passes are bound by P1 (upper bounds 0.54-0.58 against
0.60). So the lever is learning more per pass.

**Candidates** (the r4 tokens; MC32_main's data, passes, seeds and recipe; no dropout, EMA or input noise):
- **MU32_main:** Muon (`torch.optim.Muon`) for the hidden weight matrices, AdamW for the embeddings (tied with the
  output layer) and the norms. Moonshot's update scaling (`match_rms_adamw`) matches the update RMS to AdamW's, so
  the recipe's learning rate (6e-4), schedule and weight decay (0.1) carry over. Momentum 0.95 (Nesterov), 5
  Newton-Schulz steps. Muon orthogonalizes each matrix's update, so directions with small gradients (rare events)
  move as fast as frequent ones (Jordan et al. 2024; Liu et al. 2025, "Muon is scalable for LLM training", about
  2 times AdamW's compute efficiency; Wang et al. 2025, tail-end associations).
- **MUH32_main:** MU32_main at twice the learning rate (1.2e-3); Muon's best rate is not known for this setting.

Side by side on two GPUs each, four micro-batches per step, as amendment 1. **Stopping rule (cost):** a candidate
whose tuning NLL at step 1,000 exceeds MC32_main's (1.6881) by more than 0.05 is stopped and reported.

**Screen and selection:** as amendment 1 (tuning NLL, P1, S4, the 300-twin screen). The finalist rule applies to
every screened candidate that holds S3 and P1: MC32_main, MU32_main and MUH32_main. The decision run
(`tools/server/round3_decide.sh`) follows for the finalist only, after this screen, instead of right after
amendment 1's screen. A finalist must still bring S4 closer to 1 for at least two of the three treatments than
M32_main. The adoption rule and the bank criteria are unchanged.

## Amendment 3 (2026-10-06, before any goal-first sample of a trained model was drawn)

**Basis.** The excess starts are low-probability mass on treatment tokens in stays that never get the treatment,
spread over thousands of positions. Training is the slow fix (amendment 2). The goal block, a summary of the stay's
outcomes the model was trained to read in 25% of training stays (ICU outcome, ICU length-of-stay band, hospital
outcome, IMV, vasopressor, RRT and MCS yes or no, KDIGO and SOFA bands), gives a sampler-level route that needs no
training: the model draws the goal block itself, after the header, and the grammar then binds the stay to it, as
it binds a requested goal block (plan S7's obligation registers: a treatment drawn 'no' is masked throughout, one
drawn 'yes' must occur before the end, the ICU outcome and the length-of-stay band are kept). Each outcome then
comes from one prediction per stay, where a model's probabilities are calibrated (the header-level fields), rather
than from a hazard summed over thousands of positions. This is ancestral sampling from the model's own joint
distribution in another order (outcomes first), with no truncation, penalty or reweighting; the goal block is
stripped from the twins (`engine.strip_goal`), which carry a stay's own tokens as real stays do.

**Sampler (`Conditions.draw_goals`, `quality generate --draw-goals`):** every goal field drawn once; a goal its
dataset's support could not keep (a required token set without an allowed token) is never drawn; the twins' drawn
goals and how the stays kept them are reported (`quality.goal_agreement`).

**Test:** M32_main (gated), round 2's sampler (length resampling) with goal-first draws, twin seeds 0 and 1 on the
registered real sets (`runs/eval/M32_main/auth_g0`, `auth_g1`), against the same sampler without them (`auth_c`,
seed 0; `auth_d`, seed 1).

| | A1-A5 met (mean of two) | S1 cells met | Pooled A6/A7 parts met |
|---|---|---|---|
| Reference (`auth_c`, `auth_d`) | 22.5 of 35 (24, 21) | 3 of 9 (`auth_d`) | A7 grammar |

**Rule (fixed now):** goal-first sampling is adopted for the twins (every later audit, the decision run's included,
and the bank's sampler) if it is better on one axis and no worse on the other, and no pooled A6/A7 part the
reference meets fails:

| Axis | Measure | Better | No worse |
|---|---|---|---|
| Authenticity | Mean A1-A5 count over the two replicates | 24.5 or more | 21.5 or more |
| Treatment frequency | S1 cells met (mean of the two replicates) | 6 or more | 2 or more |

The M7 gate's own audit samples (`generate.vllm_audit`) are unchanged. The goal agreement is reported, not a
target.

## Results: amendment 2's candidates (2026-10-06, 12:19-13:46 server time)

Both trained as registered (2,590 steps of 0.52M tokens on the r4 tokens; 50-step DDP smoke tests with save and
resume passed; the step-1,000 stopping rule held: 1.6154 and 1.5923 against 1.7381); frozen before measurement.

| | Tuning NLL | P1 (membership AUC [95% interval], MIMIC; other datasets) | S4 pooled: IMV / vasopressor / RRT |
|---|---|---|---|
| M32_main (AdamW, reference) | 1.5611 | PASS, 0.561 [0.545, 0.575]; 0.527-0.552 | 0.675 / 0.637 / 0.449 |
| MC32_main (AdamW) | 1.5636 | PASS, 0.560 [0.544, 0.574]; 0.527-0.553 | 0.705 / 0.656 / 0.471 |
| MU32_main (Muon, 6e-4) | 1.5274 | INCONCLUSIVE, 0.604 [0.588, 0.618]; 0.554-0.582 | 0.765 / 0.669 / 0.567 |
| MUH32_main (Muon, 1.2e-3) | 1.5092 | FAIL, 0.623 [0.608, 0.637]; 0.565-0.593 | 0.787 / 0.747 / 0.575 |

- **Muon learns far more per pass:** tuning NLL 0.036-0.054 lower than AdamW at the same data, passes and steps,
  on every dataset (MUH32_main: AmsterdamUMCdb 1.211 against 1.266, eICU 1.507 / 1.553, HiRID 1.643 / 1.712,
  INSPIRE 1.461 / 1.533, MIMIC 1.616 / 1.672, NWICU 1.823 / 1.914, SICdb 1.301 / 1.353); MUH32_main at step 1,250
  matched M32_main's final NLL.
- **The onset calibration improves with the fit**, as the diagnosis predicted (MUH32_main: IMV 0.79, vasopressor
  0.75; ICU discharge 0.995), still short of 0.85-1.15.
- **But the membership signal grows with it:** MIMIC's membership AUC 0.561 -> 0.604 -> 0.623. Both are out (P1);
  their twin screens did not run (`authenticity_m32.sh` requires P1 PASS). At these passes the privacy gate,
  not the optimizer or the model's size, bounds the fit.
- The finalist rule leaves MC32_main.

**Membership against fit** (`tools/server/p1_curve.sh`: MIMIC's membership AUC of 800 training against 800
held-out stays at checkpoints; the official P1 at the end agrees: 0.560 against 0.561):

| MIMIC held-out NLL | AdamW (M32_main) | Muon (MU32_main) |
|---|---|---|
| 1.745 | 0.533 (step 1,500) | 0.534 (step 1,000) |
| 1.708 / 1.698 | 0.549 (step 2,000) | 0.551 (step 1,500) |
| 1.694 / 1.675 | 0.560 (final) | 0.582 (step 2,000) |
| 1.664 | - | 0.599 (final) |

At equal held-out NLL the two optimizers give the same membership signal: membership is a function of the fit reached, not of how it
was reached. The privacy gate bounds each dataset's fit, and M32_main spends that budget unevenly: MIMIC is at its
limit (0.561 against the working threshold's 0.565 point), while the datasets that fail authenticity have room
(AmsterdamUMCdb 0.527, HiRID 0.538, INSPIRE 0.540, NWICU 0.539).

## Amendment 4 (2026-10-06, after amendment 2's results, before amendment 3's audits were read and before any round-5 candidate was trained)

**Candidates: plan R3, flexible passes, with Muon.** Each dataset's passes are set so that its membership AUC
lands at a target, projected from MU32_main's P1 (realized passes 2.87, NWICU and INSPIRE 2.39) with the excess
over 0.5 proportional to the passes (target passes = realized / 0.957, the packing's share); Muon at 6e-4 as
MU32_main, the r4 tokens, MC32_main's recipe otherwise; side by side on two GPUs each:

| Target passes | MIMIC | AmsterdamUMCdb | eICU | HiRID | SICdb | NWICU | INSPIRE | Steps |
|---|---|---|---|---|---|---|---|---|
| MUF32_main, AUC point 0.565 (upper bound about the 0.58 working threshold) | 1.87 | 3.60 | 3.15 | 2.86 | 2.37 | 2.32 | 2.13 | 2,351 |
| MUC32_main, AUC point 0.555 | 1.59 | 3.05 | 2.66 | 2.42 | 2.01 | 1.96 | 1.81 | 1,991 |
| (M32_main, MC32_main, MU32_main) | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 | 2.5 | 2.5 | 2,566-2,590 |

The gate's limit stays 0.60. **Screen and selection** as amendments 1 and 2: the finalist rule over every
screened candidate that holds S3 and P1 (MC32_main, MUF32_main, MUC32_main), the S4 condition, then the decision
run. Twin screens use the sampler in force after amendment 3's rule; if goal-first sampling is adopted, MC32_main
is screened again with it, so every candidate's S1 cells come from the same sampler.

## Amendment 3, revised (2026-10-06, before any of its audits produced a result)

The first goal-first audits (13:46-14:44 server time) were stopped before any dataset finished: drawn 'yes'
treatments held stays open (the ICU discharge waited for an event of the treatment's token set), and on several
sources the label's derivation does not match a token reading. Real training stays written with their own goal blocks
(`tools/server/check_goalfirst.py`, 300 stays per dataset) broke the bindings in 1-25% of stays: INSPIRE 25% (post-
operative patients charted once on IMV form no episode, so 'imv no'), SICdb 18% (ventilation inferred from modes
without a device record), NWICU 6%, eICU 3%. No twin or measure of these audits was read; none was kept.

**The revised sampler** (`generate.goal_bindings`, `constrain.GoalPlan`): the model still draws the whole goal
block; a drawn goal binds only in a dataset whose real training stays keep the binding (at most 0.1% of at least 100
stays with the label break it, the plan's rule for enforced constraints), counted on the full training split
(`goal_bindings.json` beside the vocabulary); a 'yes' is conditioning only. Adopted:

| Binding | Datasets (breaking share of the labelled training stays) |
|---|---|
| ICU outcome, hospital outcome | all (0 of every label) |
| Vasopressor 'no' (no derived-episode drug in the ICU) | all but INSPIRE (35 of 9,912) |
| RRT 'no' | all but HiRID (32 of 25,741) |
| MCS 'no' | all |
| IMV 'no' | none (MIMIC 782 of 42,394 ... INSPIRE 2,284 of 4,690) |

Test, rule and thresholds unchanged (`runs/eval/M32_main/auth_g0`, `auth_g1`, `tools/server/goalfirst_m32.sh`).
Amendment 4's screens ran before this rule was applied, all with the plain sampler (a like-for-like finalist
comparison); the decision run uses the sampler the rule leaves in force.

## Results: amendment 4's candidates (2026-10-06, 14:44-16:16 server time)

| | Tuning NLL | P1 (membership AUC points, MIMIC .. INSPIRE) | S4 pooled: IMV / vasopressor / RRT |
|---|---|---|---|
| M32_main (reference) | 1.5611 | PASS, 0.527-0.561 | 0.675 / 0.637 / 0.449 |
| MUF32_main (point 0.565) | 1.5388 | INCONCLUSIVE: AUCs 0.561-0.573 (all upper bounds under 0.60), but TPR at 1% FPR upper bounds 0.031 (AmsterdamUMCdb, eICU) against 0.03 | 0.74 / 0.64 / 0.60 |
| MUC32_main (point 0.555) | 1.5525 | PASS, 0.549-0.555 | 0.74 / 0.61 / 0.51 |

The projection held: every dataset landed within 0.01 of its target AUC. MUC32_main is the first candidate with a
lower tuning NLL than M32_main inside the privacy gate (every dataset; MUF32_main per dataset: AmsterdamUMCdb 1.230
against 1.263, HiRID 1.678 / 1.711, INSPIRE 1.493 / 1.534, NWICU 1.870 / 1.907, MIMIC 1.661 / 1.669).

**Screens** (300 length-resampled twins per dataset, the plain sampler): every S1 cell is pooled at this size; MC32_main
0 of 3 (twins over real: IMV 1.57, vasopressor 1.31, RRT 2.71), MUC32_main 0 of 3 (1.47, 1.30, 2.10). First-24-hour
unigram AUC: MC32_main 0.586, MUC32_main 0.597. **The finalist rule** (most S1 cells, then the lower unigram AUC):
MC32_main. The tie is broken by a difference of about one standard error of that mean.

## Amendment 3 withdrawn (2026-10-06, 17:50 server time)

Goal-first sampling rested on a false premise. The goal block is conditioning in training: its tokens are never
targets (`model.data.labels_for` masks them; tested), so the model never learned to predict them, and the draws are
not the model's outcome distribution. Sampled goal blocks (`tools/server/gf_draws.py`, 1,000 per dataset, no
binding): ICU death 61-65% against 5.5-12% real; vasopressor 'yes' 61-76% against 11-70%; eICU stays of 168-336 hours
32% against 5.5%. The twins followed their goals (the revised audits' eICU twins: 20% of stays over 336 hours, 15%
without an ICU discharge, length p99 30,799 tokens against 6,238), so both audit attempts were stopped and their
twins deleted; no measure was read. Goal-first sampling needs goal tokens as training targets (a training change for
a later round); the sampler in force stays round 3's (length and treatment raking).

## Amendment 5 (2026-10-06, the study lead's decision after the screens, before the decision run)

The finalist rule's S1 criterion was uninformative at the screens' size (every cell pooled; both candidates 0 of 3)
and its tie-break (first-24-hour unigram AUC 0.586 against 0.597) lies within one standard error. The study lead chose
to send **MUC32_main** to the decision run instead of the rule's MC32_main: the only candidate with a lower tuning
NLL than M32_main inside the privacy gate. This departs from the pre-registered finalist rule; the decision run, the
adoption rule and the bank criteria are unchanged (`tools/server/round3_decide.sh MUC32_main`, round 3's sampler;
`tools/server/round3_adopt.py MUC32_main 0`).

## Decision run and adoption (2026-10-06, 17:54-21:11 server time; amendment 5's finalist MUC32_main)

`tools/server/round3_decide.sh MUC32_main` (round 3's sampler: length and treatment raking; twin seeds 0 and 1 on the
registered real sets), the rule applied by `tools/server/round3_adopt.py MUC32_main 0`:

| | MUC32_main | M32_main (reference, same sampler) | Rule |
|---|---|---|---|
| M7 gate | PASS (P1-P4, v2's P5) | PASS | must pass |
| S3, tuning NLL | 1.5525 | 1.5611 | at most 1.571 |
| Authenticity: A1-A5 met (two audits) | 30, 31: **30.5** | 24, 26: 25.0 | better: + 2 |
| Calibration: S1 + S2 cells | 3 + 18 = 21 | 3 + 19 = 22 | no worse: - 1 |
| Pooled A6/A7 parts | A7 grammar in both | A7 grammar in both, A7 utility in one | none lost |

**MUC32_main replaces M32_main** (authenticity better, calibration no worse). Per dataset, MUC32_main meets every
target in AmsterdamUMCdb and HiRID in both audits (M32_main: 1 and 3 of 5); the first-24-hour discriminator falls
under 0.70 in every dataset but INSPIRE (AmsterdamUMCdb 0.688 / 0.680, HiRID 0.653 / 0.673, SICdb 0.662 / 0.659,
INSPIRE 0.751 / 0.754). The forecast cells (S2) stay mixed: rollouts under-predict death in AmsterdamUMCdb (O/E 1.73),
over-predict stays over 7 days in MIMIC and HiRID (0.72, 0.77) and RRT (pooled 0.55).

**The bank stays off:** the pooled A6 (Vendi ratio 0.88-0.89 against 0.95; rare-code recall 0.91-0.92 against
0.924) and A7 (harness flags: discriminator AUC, density, Vendi; utility of IMV 0.84-0.89 against 0.90) fail, as do
INSPIRE's A1 and SICdb's A4.

## Round 3 summary

1. Treatment starts are over-predicted step by step because rare-event timing is under-fitted, not for lack of
   information (members as non-members; shrinking with training; explicit day and state blocks barely help).
2. The membership signal is a function of the fit reached, whatever the optimizer: the privacy gate bounds each
   dataset's fit. Muon reaches that bound in far fewer steps.
3. M32_main had spent the budget unevenly; per-dataset passes projected to a membership target (MIMIC down, the small
   datasets up) give MUC32_main, better on every dataset inside the gate, and adopted.
4. Goal-first sampling was withdrawn: goal tokens were never training targets (`TransformConfig.goal_targets`
   prepares a later run).

## Amendment 6 (2026-10-06, after the decision run, before any re-audit; the study lead's principle)

**The study lead's principle:** an indicator very close to its threshold counts as met. Operationalized (stated to the
study lead before this amendment): a bank criterion is read on the mean of two audit replicates and counts as met when
it lies within 3% of its threshold (relative). On MUC32_main's decision audits this passes rare-code recall (1.2%
short) and eICU's top-event repeats (2.0% over); IMV utility (3.9% short) does not; five miss by 6-9%: MIMIC's
and SICdb's events-per-stay 99th percentile (A4: 1.275 and 1.306 against 1.20), INSPIRE's first-24-hour AUC (A1:
0.753), the pooled Vendi ratio (A6: 0.885) and the pooled density (A7 harness: 0.977 against 1.042).

**Diagnosis of the diversity gap** (`tools/server/diversity_breakdown.py`): per dataset the twins' density matches
the positive control's (MIMIC 1.84-1.93 against 1.91, eICU 1.84-1.88 against 1.80); only the pooled reading is low
(0.97-0.98 against 1.09). The Vendi and rare-code gaps sit in AmsterdamUMCdb (Vendi 0.83-0.85; rare codes 0.86
against 0.97), HiRID (0.83-0.86; 0.76-0.83 against 0.93), INSPIRE (0.87-0.88; 0.76-0.78 against 0.90) and SICdb's
rare codes; M32_main shows the same (pooled Vendi 0.884, density 0.967). They are the small, densely charted
datasets' under-fitted rare structure, bound by the privacy gate; no sampler change within the plan's rules adds
diversity the model lacks (curating twins for diversity would distort fidelity), so they are reported, not fixed.

**Sampler change (twin-level calibration, a mechanism approved by the study lead):** the raking gains the events-per-stay margin
(`generate.calibrate`, bands at the training stays' 50/75/90/95/99th percentiles; `calibration_events.json` beside
the vocabulary), aimed at A4's length tails. **Re-audit:** MUC32_main, two replicates (twin seeds 0 and 1 on the
registered real sets, `runs/eval/MUC32_main/auth_e0`, `auth_e1`), criteria read as above
(`tools/server/target_means.py`). The bank turns on only if every criterion holds under that reading.

## Results: amendment 6's re-audit (2026-10-06, 21:25-22:38 server time)

MUC32_main with the events-per-stay margin added to the raking (`runs/eval/MUC32_main/auth_e0`, `auth_e1`), criteria
on the two replicates' means (`logs/reaudit_events_means.txt`), against the decision audits (round 3's sampler):

| | Round 3's sampler (r0, r1) | Plus the events margin (e0, e1) | Threshold |
|---|---|---|---|
| MIMIC events p99 ratio (A4) | 1.275 | 1.067 | 0.80-1.20 |
| eICU top-event p99 ratio (A5) | 1.275 | 1.147 | 0.75-1.25 |
| SICdb events p99 ratio (A4) | 1.306 | 1.316 (1.215 / 1.418) | 0.80-1.20 |
| HiRID top-event p99 ratio (A5) | met | 1.429 (1.624 / 1.233) | 0.75-1.25 |
| INSPIRE first-24-hour AUC (A1) | 0.753 | 0.747 | at most 0.70 |
| Vendi ratio (A6) | 0.885 | 0.879 | at least 0.95 |
| Rare-code recall (A6) | 0.913 | 0.901 (within 3%) | at least 0.924 |
| Density (A7) | 0.977 | 0.967 | at least 1.042 |
| IMV utility (A7) | 0.865 | 0.804 | at least 0.90 |
| A1-A5 met (two audits) | 30.5 | 29.5 | |

The margin fixed MIMIC's tail and eICU's repeats, not SICdb's, and the 99th-percentile tails (about ten twins per
dataset) swing between replicates (HiRID 1.624 / 1.233, SICdb 1.215 / 1.418): raking at that resolution moves noise
between datasets. IMV utility fell. No net gain: round 3's sampler stays in force. **The bank stays off** under the
study lead's principle: INSPIRE's A1, the pooled Vendi ratio and density (model-level, diagnosed above), IMV utility and
one dataset's tail remain beyond 3% of their thresholds.

## The study lead's decision (2026-10-06, after amendment 6's re-audit)

**MUC32_main with round 3's sampler (length and treatment raking) is accepted for internal use** (the ICL factory's
development), with the gaps above documented: INSPIRE's first-24-hour discriminator (0.75 against 0.70), the pooled
Vendi ratio (0.88 against 0.95) and density (0.97 against 1.04), IMV utility (0.80-0.87 against 0.90) and single
datasets' 99th-percentile tails. The bank it feeds is internal (DUA-derived synthetic data: never shared outside the
approved hosts unless cleared), its twins carry these limits in their cards, and M32_main stays frozen beside it.
