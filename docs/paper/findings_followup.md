# Materials: what the follow-up experiments show (amendments 7-9, 2026-10-10)

A reading of `docs/paper/results_followup.md`, the tables of experiments F1-F9. Those experiments were fixed in
`docs/p1_paper_protocol.md` amendment 7 before they ran; amendment 8 records the one correction made while they ran.
Amendment 9 adds an exploratory diagnosis of the attention results (section 3).
The pre-registered results (`docs/paper/results.md`) are unchanged. Wherever a follow-up revises their reading, both
readings are given. All numbers are held-out, with 95% intervals by a bootstrap over stays.

## 1. Accuracy without training: more rollouts, and the model's own step hazards (F1, F2)

Two inference-only changes move the Twin from mixed to leading or level:

- **More rollouts:** 128 instead of 32.
- **The hybrid estimator (E2):** each rollout's sampled events are replaced by the model's probability of the event at
  every step, summed while the rollout is at risk. This is a Rao-Blackwellization; it keeps the same expectation with
  less variance.

| Endpoint (subset) | A1, 32 rollouts | 128 rollouts (E0) | Hybrid E2, 128 | LightGBM | TabPFN-3.5 | E2 - LightGBM (95% CI) |
|---|---|---|---|---|---|---|
| Death within 24 h (R128) | 0.903 | 0.918 | **0.960** | 0.956 | 0.955 | +0.004 (-0.006 to 0.015) |
| New IMV, 24 h (R128) | 0.837 | 0.870 | **0.886** | 0.805 | 0.797 | +0.081 (0.029 to 0.152) |
| New vasopressor, 24 h (R128) | 0.889 | 0.894 | **0.903** | 0.846 | 0.842 | +0.057 (0.017 to 0.096) |
| New RRT, 48 h (R128) | 0.931 | 0.948 | **0.959** | 0.934 | 0.933 | +0.025 (0.005 to 0.048) |
| ICU death (L24x) | 0.853 | 0.910 | **0.924** | 0.914 | 0.932 | +0.010 (-0.013 to 0.031) |
| Death within 24 h (L24x) | 0.929 | 0.981 | **0.982** | 0.957 | - | +0.025 (0.010 to 0.044) |

(E0 here is the token-level share used for F2; F1's A1-style table in `results_followup.md` reads every program with
the label engine and gives the same picture for all primary endpoints.)

- **The pre-registered weak spot closes.** In A1 the Twin trailed LightGBM on death within 24 h by 0.062. With the
  hybrid it is level, and with 128 rollouts and longer caps (F3) ICU death at the landmark is level too.
- **One hybrid rollout is worth many sampled ones.** With one rollout the hybrid's AUROC is 0.81-0.93 against
  0.61-0.66 for A1's share. For death within 24 h, 8 hybrid rollouts (0.937) already beat A1's share at 128 (0.918).
- **Calibration does not change.** Rao-Blackwellization removes variance, not bias, so the hybrid's O/E equals A1's.
- **The check that the hazards are right:** in every subset the summed hazards reproduce the sampled events (ratio
  0.99-1.02). This held only after amendment 8. The first scoring missed the sampler's renormalization: the grammar
  forbids rate changes of infusions that are no longer running, and the model puts mass on them in end-of-life
  sequences. The corrected hazards match the exact sampler probabilities within 2.5%.

## 1b. The comparators clinicians use, and the standard learners (amendment 10)

Amendment 10 adds comparators on the same held-out anchors:
- **Traditional clinical scores,** from the harmonized raw CLIF values at the prediction time: SOFA, SAPS II, OASIS,
  NEWS, MEWS, qSOFA and SIRS.
- **The datasets' own admission scores:** APACHE IVa in eICU, SAPS 3 in SICdb, the ASA class in INSPIRE.
- **Single bedside values:** lactate, shock index, ROX, MAP, KDIGO stage, creatinine, potassium, platelets.
- **Learned baselines** on A1's features: L1 and elastic-net logistic regression, a decision tree, random forest,
  extra trees, naive Bayes, k-nearest neighbours and a multilayer perceptron. A1's LightGBM, L2 logistic regression,
  TabPFN and the Twin probe stand beside them.

**Clinical scores trail the Twin on every endpoint (by 0.04 to 0.24 AUROC), and the stronger learned models.** Only
the weakest learners (naive Bayes, the single decision tree) fall to their level.

| Endpoint (24-hour landmark) | Best clinical score | Twin (A1, 32 rollouts) | LightGBM |
|---|---|---|---|
| ICU death | NEWS 0.805, SAPS II 0.778 | 0.858 | 0.915 |
| In-hospital death | SAPS II 0.745 | 0.858 | 0.867 |
| ICU stay > 7 days | SOFA 0.678 | 0.808 | 0.855 |
| New IMV / vasopressor / RRT | NEWS 0.709 / SOFA 0.675 / creatinine 0.889 | 0.838 / 0.880 / 0.939 | 0.829 / 0.806 / 0.922 |

- **The calibrated equations.** SAPS II and OASIS also give probabilities (table h in `results_followup.md`).
- **Coverage limits the scores at some sites.** NWICU and SICdb record no GCS, and NWICU records no urine output or
  blood gases. Scores there lose components (table a).
- **The windowed SOFA is checked against the pipeline's own first-day SOFA:** identical in 96-99% of 24-hour anchors
  in six datasets. INSPIRE is the exception (80%), because its window includes the operating room before ICU
  admission.

**The datasets' own scores** (endpoints with at least 10 events) also trail the Twin on their own data:

| Score | Endpoint | Score | Twin (A1) | LightGBM | Twin - score |
|---|---|---|---|---|---|
| APACHE IVa (eICU) | hospital death, 24 h | 0.850 | 0.941 | 0.868 | +0.09 (0.03 to 0.16) |
| APACHE IVa (eICU) | hospital death, 48 h | 0.766 | 0.901 | 0.903 | +0.13 (0.07 to 0.21) |
| SAPS 3 (SICdb) | hospital death, 24 h | 0.812 | 0.869 | 0.856 | +0.06 (-0.00 to 0.12) |
| SAPS 3 (SICdb) | hospital death, 48 h | 0.815 | 0.882 | 0.876 | +0.07 (0.01 to 0.12) |

APACHE IVa's predicted ICU stay is the one exception: for stays over 7 days at 24 h it is 0.763, against the Twin's
0.704 (difference -0.06, -0.17 to 0.05).

**A single bedside value comes close on the laboratory-threshold endpoints** (rolling anchors):
- **Platelets** for thrombocytopenia: 0.919, against the Twin's 0.939 and LightGBM's 0.953.
- **Potassium** for hyperkalaemia: 0.793, against 0.804 and 0.826.
- **MAP** for hypotension: 0.863, against 0.906 and 0.925.
- **Creatinine** for new RRT: 0.874, against 0.939 and 0.932.

These endpoints are thresholds on the same variable, so its current value carries most of the signal.

**Among the learned baselines,** the ranking is stable across endpoints:
- TabPFN-3.5, LightGBM, random forest and extra trees usually lead.
- L1 and elastic-net logistic regression and the multilayer perceptron follow, mostly ahead of L2 logistic
  regression. L1 leads on a few endpoints, e.g. thrombocytopenia on rolling anchors (0.962).
- k-nearest neighbours, the decision tree and naive Bayes trail.
- The Twin probe (a linear model on the Twin's frozen states) is among the best for death.

**Against the Twin:**
- **A1's anchors** (32 rollouts, the share), 24-hour landmark: the Twin trails the strongest tabular models on death
  and long stays, e.g. ICU death 0.858 against 0.932 (TabPFN-3.5) and 0.908 (random forest). It leads every learner
  on new vasopressor (0.880 against 0.826 at most, the Twin probe) and is level or ahead on new IMV and RRT.
- **The current Twin** (128 rollouts, E2 where it applies), against the best of the 13 other learned models picked
  after the fact. That choice favours the comparator.
  - **L24x:** no comparator is clearly ahead on any endpoint.
    - ICU death: E2 0.924 (E0 0.911), against TabPFN-3.5's 0.932. The E0 difference is -0.022 (-0.047 to 0.001).
    - In-hospital death: 0.883 against 0.876.
    - Stay over 7 days: 0.864 against 0.864.
    - Death within 24 h: 0.982 against 0.957.
    - New IMV: E0 0.929, against 0.851.
    - New vasopressor: E2 0.896, against 0.844.
    - New RRT: 0.964 against 0.955.
  - **R128:** the share (E0) trails TabPFN v2 for death within 24 h (-0.040, -0.064 to -0.018), but E2 (0.960) does
    not (TabPFN v2 0.957).
    - Leads: new vasopressor +0.059 (0.025 to 0.099), AKI +0.049 (0.018 to 0.079).
    - Level: new IMV, RRT, sepsis and hyperkalaemia.
    - Behind, within the intervals: hypotension -0.032 (-0.080 to 0.010; extra trees 0.907) and thrombocytopenia
      -0.010 (elastic-net logistic 0.942).
  - **The Twin probe**, a linear model on the frozen states, is close to the Twin for death. It trails the Twin on
    treatment starts and laboratory endpoints (e.g. R128 new vasopressor 0.738).

## 2. Mortality: censoring at the landmark, delay in generation (F3)

Pre-registered, the Twin under-predicted death: O/E 1.14 for ICU death at 24 h and 1.83 for death within 24 h on
rolling anchors. The decomposition separates two causes.

| Check | Result | Reading |
|---|---|---|
| **D1, step calibration.** The model's step hazards summed along each anchor's real future, against observed deaths | O/E 0.98 (0.86-1.11) for death within 24 h (R); 0.95 (0.82-1.12) for ICU death (L24) | Calibrated step by step: the model is not miscalibrated about death given a real history |
| **ICU death at the landmark, 4 times longer rollouts** (8.5% capped instead of 21.6%) | O/E 0.99, AUROC 0.911 (A1: 0.858); cut back to A1's cap: O/E 1.10-1.38 | The landmark deficit was **evaluation censoring**: rollouts stopped before the deaths of long stays |
| **D2, generation drift.** The rollouts' summed death hazard over the real futures', same anchors | 0.61 (R) and 0.57 (L24) within 24 h; 0.91 until the ICU discharge (L24) | Generated futures carry too little death hazard **early**, yet nearly the right amount over the whole stay |
| **The hourly view** | Rollouts' death hazard is about 0.5x the real paths' in the first 6 h and 0.3-0.5x up to 36 h (R128 stays near 0.3x to 48 h); alive discharge 0.8-1.2x | Combined with 0.91 over the whole stay, the Twin **delays** deaths rather than omitting them |
| **D4, the bank** | Twins' deaths are missing mainly in short stays (MIMIC deaths within the first day 3.9% against 11.5%); the bank's stays are complete (censoring under 1%, SICdb 8%) | The same delay, seen in whole twins |

**What this means for the paper.** The mortality limitation is narrower than A1 suggested, and it has a mechanism.

- **At landmarks:** with rollouts long enough to reach the outcome, the Twin predicts ICU and in-hospital death at
  the level of the best tabular models, and is calibrated.
- **Over short horizons** it under-predicts death, because free-running generation postpones dying. It is calibrated
  given real histories.
- **Treatment starts** behave differently: over-prediction is present step by step (D1 O/E 0.56-0.80) and grows in
  generation (drift 0.9-1.4). That is per-step miscalibration, as round 3 found.

## 3. Attribution checked against the generator (F4)

| Method | Agreement with the generator's edit effects (Spearman, median per anchor) | Survives randomizing the weights? |
|---|---|---|
| Counterfactual editing (probe) | 0.12-0.20 (35-40% of the reliability ceiling 0.48-0.54; hypotension 0.07 of 0.13) | yes (correlation with itself -0.06 to 0.00) |
| Integrated gradients | -0.10 to 0.08 | yes (-0.08 to 0.07) |
| Attention rollout | -0.11 to 0.08 | **no** (0.96-0.97: input structure, not learning) |
| Attention x gradient | -0.09 to 0.14 | partly (0.53-0.65) |

- **The ceiling is low.** Even with 48 rollouts per edited prompt, the generator's own per-group effects reproduce
  only moderately (split-half ceilings 0.13-0.54). Individual-level attribution at the generator level is noisy at
  this compute.
- **Editing is the only method with positive agreement.** It also agrees in sign on groups with clear effects
  (0.61-0.66).
- **Attention fails the model-randomization test** (Adebayo et al. 2018). Attention rollout ranks groups almost the
  same in a randomly initialized model, so it describes the input's structure, not what the model learned.
- **Recommendation:** present counterfactual editing as the attribution method, causal within the model and with its
  modest generator-level agreement stated. Report attention only as a negative result, not as a proxy. (Amendment 9,
  below, refines this: attention knockout passes.)

### Why attention rollout fails, and what passes (amendment 9, exploratory)

The diagnosis used F4's 152 anchors with prompts of at most 4,096 tokens and the same trained and random models; the
code reproduces F4's stored scores exactly. A rollout that involves no model at all, of uniform causal attention (the
*structural null*), ranks the groups like the trained model's rollout (0.97) and exactly like the random model's
(1.00). The failure is a property of the method in a causal model, not an error. There are three causes, in order of
weight.

1. **Rollout reduces to the causal mask's geometry.**
   - Rollout multiplies 12 matrices, each half identity and half head-averaged attention. Under a causal mask each
     attention step can move mass only to earlier tokens, and the first token attends only to itself, so mass drains
     toward the start of the prompt whatever the attention does.
   - 35-38% of the rollout lands on the first token ([BOS], in no group) in both models. Yet the last layer gives
     [BOS] only 0.2% of its attention: this is geometry, not a learned attention sink.
   - The random model attends almost uniformly (per-head entropy 0.92-0.95 of the maximum), so its rollout is the null
     (total variation 0.02).
   - The trained model's heads are sharper (0.68-0.81). Averaging the heads (0.81-0.93) and multiplying the layers
     bring its rollout back within 0.19 of the null, with the same group ranking.
   - With the groups' token counts held fixed the correlation stays 0.96: position drives it, not size.
2. **Summing over a group ranks groups by token count.**
   - Within a prompt the largest group has a median 84 times the tokens of the smallest. Summing turns any spread-out
     token score into a ranking by size.
   - The last layer's raw attention is unrelated between the two models token by token (0.03), but correlated once
     summed per group (0.78).
   - This is also why attention x gradient only partly passed in F4 (0.58). With token counts held fixed, its sanity
     correlation is 0.08.
3. **Rollout ignores the target.** It uses neither the values, the MLPs nor the probe. It is the same for every
   endpoint (the permuted-label check gave 1.00 by construction).

**Remedies.** A remedy passes if its sanity correlation is at most 0.3 and its agreement with editing is positive
(amendment 9). Pooled over the four endpoints, medians over anchors:

| Method | Trained against random (sanity) | Agreement with probe editing | Verdict |
|---|---|---|---|
| Attention rollout (C2), group sums | 0.97 | -0.08 | fails |
| Rollout, group means | 0.92 | -0.11 | fails: the geometry remains |
| Norm-weighted rollout (Kobayashi et al. 2020) | 0.96 | -0.08 | fails |
| Rollout less the null, means | 0.08 | -0.08 | no shared structure, and no signal left |
| Last layer's attention, means | 0.01 | 0.08 | passes, weakly |
| Attention x gradient, means | -0.03 | 0.14 | passes |
| Gradient-weighted rollout (Chefer et al. 2021), means | -0.04 | 0.14 | passes |
| **Attention knockout** (Geva et al. 2023) | -0.02 | **0.71** | passes, on every endpoint (0.60-0.74) |

- **Removing the confounds is not enough.** Subtracting the null removes what the two models share. What remains does
  not track the endpoint (-0.08): attention weights that ignore the target say little about the probe's risk.
- **Target-specific gradients help a little.** Attention x gradient and gradient-weighted rollout pass with group
  means, at 0.14 agreement (0.20-0.22 with token counts held fixed).
- **Attention knockout works.** It blocks attention from every other token to a group's tokens in all 12 layers and
  reads the probe's logit, an intervention through the attention mechanism.
  - It does not track group size (0.03) or the null (0.00), passes the randomization check (-0.02), and agrees with
    editing at 0.71 (interquartile range 0.53-0.80).
  - It costs what editing costs (one forward pass per group), and it needs no re-encoded prompt.
  - **Caveat:** the agreement is partly expected. Both methods remove a group's information, editing from the input and
    knockout from the attention paths. The agreement shows that the model reads the groups through attention. It is
    not independent validation, and knockout has not been checked against the generator (that needs F4's rollouts
    again, about 3 GPU-hours).

**For the paper:**
- Keep counterfactual editing as the primary method.
- If the paper keeps an attention-based method (the study lead's "attention as proxy"), use attention knockout and report
  its sanity checks.
- Report rollout as a negative result, with the structural null as its explanation.
- Aggregate any token-level score by group means, not sums.

## 4. What-if rollouts validated (F5)

- **The null holds.** Two withhold arms with different seeds give "effects" centred on 0, with the share of the
  "expected sign" between 0.40 and 0.55. C1's reported effects lie well outside this band: insulin -14 mg/dL,
  furosemide +723 mL, IMV FiO2 charting +2.5.
- **The Twin reproduces the observational contrasts.** On 158-195 propensity-matched pairs per treatment (starters cut
  1 minute before their start), the twin's version agrees with the real contrast:

  | Response | Observational | Twin |
  |---|---|---|
  | Norepinephrine, MAP | -3.3 | -3.6 |
  | Fluid bolus, MAP | -5.8 | -4.3 |
  | Furosemide, urine output (mL) | +337 | +403 |
  | IMV, FiO2 charting | +2.8 | +3.1 |
  | IMV, SpO2 | +1.1 | +1.2 |
  | CRRT, potassium | +0.24 | +0.44 |
  | CRRT, creatinine | +0.50 | +1.04 |

  The exception is insulin: observational +38 mg/dL, Twin about 0. The Twin is a faithful model of the data, including
  its confounding by indication.
- **Factual futures are predictive.** The factual arm's 16-rollout ensemble halves the persistence error (CRPS) for
  every response. Its 80% intervals cover 66-91% (mostly about 70%: slightly too narrow).
- **Timing matters.** With prefixes cut just before the real start, giving norepinephrine raises MAP in the Twin (+0.8,
  0.1-1.5). C1's sign problem came largely from cutting 0.5-6 h earlier, where the observational downward trajectory
  dominates. Fluid boluses still show the indication pattern.

## 5. Cohorts, similarity, augmentation (F6-F8)

- **Hospitals (F6).** Across the 17 eICU hospitals with enough stays, the twins' hospital-level rates track the real
  ones about as well as real-against-real does: ICU death 0.45 (control 0.61), IMV 0.63 (0.52), vasoactive 0.66
  (0.85), median stay 0.72 (0.71). The intervals are wide.
- **Similarity (F7).** Nearest neighbours by the model's state predict outcomes at least as well as TF-IDF or the
  baseline features, better for RRT (0.85 against 0.78) and AKI. A randomly initialized model is clearly worse (IMV
  0.50 against 0.64). The learned state carries outcome-relevant similarity; the margin over simple representations
  is modest.
- **Augmentation (F8).** With 20 repetitions per cell, adding twins to a small site's real stays gives small but
  consistent gains:
  - with 100 real stays, +0.04 (landmark) and +0.02 (rolling) AUROC on average, positive in 19/19 and 25/34 cells;
  - with 300 stays, +0.01;
  - with 1,000 stays, +0.01 or less.

  This revises A4's noisy small-site result: augmentation helps where data are scarcest, and stops helping by a few
  hundred stays.

## 6. Transfer to new sites (F9)

- **Leaving a dataset out (supervised models).** Practice-driven endpoints transfer poorly: LightGBM loses 0.04-0.11
  AUROC for new IMV, vasopressor and AKI. Physiology endpoints transfer almost freely (hypotension, hyperkalaemia,
  thrombocytopenia, death, RRT).
- **The zero-shot Twin** beats the left-out-dataset LightGBM by +0.09 to +0.12 on exactly those practice-driven
  endpoints. It needs no labels from the site, but it has seen the site's unlabelled data.
- **Novelty.** The Twin and LightGBM degrade alike for the most atypical fifth of anchors.
- **Site prompts (F9a, R128, 32 rollouts per anchor).**
  - **Unknown site:** discrimination is essentially unchanged (within ±0.04 for most endpoints; hypotension +0.06;
    thrombocytopenia -0.08, CI -0.20 to 0.00; sepsis has 8 positives). At a new site the Twin can be prompted with
    the unknown token.
  - **A wrong dataset token is harmful:** RRT -0.30 (-0.39 to -0.18), vasopressor -0.12, AKI -0.05. The predicted
    risks do not move toward the other dataset's prevalence (correlations at most 0.25).
  - **The cause is the dataset support mask.** A prompt labelled as another dataset inherits that dataset's recording
    schema, so it cannot generate events the other dataset never records. Of the 60 RRT tokens, MIMIC's support
    allows 44, while HiRID, INSPIRE and NWICU allow 1.
  - **Death calibration moves with the token:** O/E 1.87 with the known site, 0.56 with a swapped one.
- **Stated limit:** transfer of the generative model to a site it never saw needs a model trained without that site,
  which is outside this phase's inference-only scope.

## 6b. The application toolkit (amendment 11)

The toolkit turns the validated capabilities into seven tools. Its validation asked what the paper had not shown.

- **It reproduces the paper.** The prompts are identical (1,050 of 1,050). The 32-rollout risks agree with the
  paper's runs within their noise. On death within 24 h, the toolkit (0.880) matches F1's 32-rollout expectation
  (0.881), while A1's single replicate (0.903) sits 3.7 rollout-noise SDs above F1.
  - The pre-registered rule did not pass. Neither did the post-hoc noise rule: both compared against A1 alone, and
    the paper's two runs disagree by more than rollout noise.
  - Under the study lead's criterion (within noise of either run) every program conforms. The finding to report is the
    run-to-run spread itself: a single 32-rollout replicate is not a precise estimate of discrimination for sparse or
    near-ceiling endpoints.
- **Recalibration needs a smooth map.** Isotonic maps from 700 tuning anchors fix the treatment starts' O/E. Their
  steps tie risks, though, and cost discrimination on the rarer endpoints. A logistic map keeps the ranks.
  - With it (V2b), vasopressor and RRT reach O/E 1.02 and 0.93 with ECE cut three- to eight-fold.
  - IMV overcorrects (0.55 to 1.43), because its tuning-split miscalibration was larger than the held-out anchors'.
  - Seven maps are in force, four isotonic and three logistic.
- **Prompted sampling showed no sign of memorization,** and the gate now records it (PASS, the study lead's decision).
  - No rollout reproduced any stay's real next 32 tokens, member or not.
  - At P1's sizes the conditional loss separates members from non-members only slightly: pooled AUC 0.531.
  - Seven of eight datasets pass. INSPIRE is inconclusive only because its held-out split holds 790 eligible stays
    per side.
  - This answers gate v1's PENDING item for the toolkit's own settings: real histories, ancestral sampling and an
    intervention.
- **For trials, the twin's prognosis is worth patients.** As a covariate in the trial's analysis, the Twin's risk
  cuts the required sample by 14-37% (relative risk 0.8). That is level with or ahead of a supervised LightGBM, and
  well ahead of SOFA, SAPS II, OASIS and NEWS (0-20%). On eICU it beats APACHE IVa on its own data. Enrichment by the
  Twin's top quarter needs a fifth to a third of the patients.
- **Teaching cases** are built and checked mechanically (objectives, self-contained pages). Their realism is the
  clinician study's question (amendment 12).

## 7. What changes for the paper

1. **Prediction.** The Twin, read with the hybrid estimator and enough rollouts:
   - matches or beats LightGBM and TabPFN-3.5 on every primary endpoint of both designs;
   - leads on treatment starts and AKI;
   - is level on death, at the landmark and within 24 h.

   The pre-registered A1 numbers stay as reported, with this follow-up showing what drove them: Monte Carlo noise and
   rollout length.
2. **Mortality.** It is no longer an unexplained weakness. The landmark deficit was censoring. The short-horizon
   deficit is a delay in generation, with step calibration intact.
3. **Attribution.** Lead with counterfactual editing. Attention rollout fails the sanity checks because of the causal
   mask's geometry and the group sums (amendment 9) and should not be a proxy. Attention knockout passes them and
   agrees with editing; it is the attention-based method to report.
4. **What-if.** Present it as observationally faithful simulation (it reproduces the data's contrasts), with the null
   and the factual accuracy as validation. It is not causal effect estimation.
5. **Augmentation** is a modest small-site benefit. **Transfer** to unseen sites remains a stated limit.
6. **The toolkit** can be a section of the paper on its own: seven tools, each output carrying its evidence and
   caveats. What it adds:
   - a conformance result, with the run-to-run spread stated;
   - the privacy of prompted sampling (no extraction; pooled loss AUC 0.531; recorded in the gate);
   - the trial designer's sample-size gains, the most direct clinical-research use.

   Recalibration covers seven of the nine programs (four isotonic, three logistic maps); IMV's overcorrects.
