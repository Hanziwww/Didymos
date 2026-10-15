# Paper P: current results (2026-10-13)

The one-page state of every experiment for MUC32_main. Where a follow-up improved or corrected a result, the current
version is shown first and the pre-registered number follows it as "pre-registered". Nothing here replaces the
records:

- `results.md`: pre-registered experiments A1-C4 (protocol amendments 1-6), unchanged;
- `results_followup.md`: follow-ups F1-F9 (amendments 7-8), the attention diagnosis (amendment 9, exploratory),
  the comparators (amendment 10: clinical scores, linear, tree and other learners), the application toolkit's
  validation (amendment 11) and amendment 16 (16a transfer to RICD and zigongICU, 16b the 48-hour landmark L48x, 16c
  knockout against the generator; pre-run gate in `external_gate.md`);
- `findings_followup.md`: the reading of the follow-ups.

Every number is on the held-out split, with 95% intervals by a bootstrap over stays unless stated. Each section names
its machine-readable tables in `tables/` (one CSV per Markdown table; `tables/*/index.csv` lists them). The protocol is
`docs/p1_paper_protocol.md`.

## 1. Prediction

The Twin is zero-shot: rollouts from the prompt, no labels. Its current estimator is **E2**, the hybrid: the model's
step hazards summed along each rollout while it is at risk (a Rao-Blackwellization of the rollout share). E2 applies to
death and treatment starts; AKI and the laboratory endpoints use the share (E0) of 128 rollouts. LightGBM and
TabPFN-3.5 are trained on real stays with A1's features.

**Rolling anchors** (R128: 1,050 held-out anchors, 150 per dataset, inclusion-weighted; 128 rollouts):

| Endpoint | Twin, current | LightGBM | TabPFN-3.5 | Twin - LightGBM (95% CI) | Twin - TabPFN-3.5 | Pre-registered Twin (32 rollouts, share; same anchors) |
|---|---|---|---|---|---|---|
| Decompensation, 24 h | 0.960 (E2) | 0.956 | 0.955 | +0.004 (-0.006 to 0.015) | +0.005 (-0.005 to 0.015) | 0.903 |
| New IMV, 24 h | 0.886 (E2) | 0.805 | 0.797 | +0.081 (0.029 to 0.152) | +0.089 (0.026 to 0.189) | 0.837 |
| New vasopressor, 24 h | 0.903 (E2) | 0.846 | 0.842 | +0.057 (0.017 to 0.096) | +0.061 (0.021 to 0.102) | 0.889 |
| New RRT, 48 h | 0.959 (E2) | 0.934 | 0.933 | +0.025 (0.005 to 0.048) | +0.026 (-0.002 to 0.056) | 0.931 |
| AKI stage >= 2, 48 h | 0.888 (E0) | 0.835 | 0.835 | +0.053 (0.022 to 0.085) | +0.053 (0.018 to 0.089) | 0.887 |
| Sepsis (ASE), 24 h (8 positives) | 0.966 (E0) | 0.951 | 0.959 | +0.015 (-0.063 to 0.081) | +0.007 (-0.011 to 0.024) | 0.989 |
| Hypotension, 4 h | 0.874 (E0) | 0.900 | 0.894 | -0.025 (-0.064 to 0.005) | -0.020 (-0.057 to 0.007) | 0.875 |
| Hyperkalaemia (top bin), 24 h | 0.831 (E0) | 0.812 | 0.807 | +0.019 (-0.031 to 0.077) | +0.025 (-0.022 to 0.071) | 0.847 |
| Thrombocytopenia, 48 h | 0.932 (E0) | 0.918 | 0.910 | +0.015 (-0.002 to 0.031) | +0.022 (0.003 to 0.044) | 0.829 |

**24-hour landmark** (L24x: 700 anchors, a case-cohort sample of 346 in-hospital deaths and 354 weighted controls;
128 rollouts with 4 times A1's token cap, 8.5% of rollouts capped instead of 21.6%):

| Endpoint | Twin, current | LightGBM | TabPFN-3.5 | Twin - LightGBM (95% CI) | Twin - TabPFN-3.5 | Pre-registered Twin (same anchors) |
|---|---|---|---|---|---|---|
| ICU death | 0.924 (E2); 0.911 (E0) | 0.914 | 0.932 | +0.010 (-0.013 to 0.031) | -0.008 (-0.028 to 0.010) | 0.853 |
| In-hospital death | 0.883 (E0) | 0.858 | 0.876 | +0.025 (0.007 to 0.043) | +0.006 (-0.010 to 0.023) | 0.856 |
| ICU stay > 7 days | 0.864 (E0) | 0.850 | 0.864 | +0.013 (-0.016 to 0.044) | -0.000 (-0.035 to 0.031) | 0.843 |
| Death within 24 h | 0.982 (E2) | 0.957 | - | +0.025 (0.010 to 0.044) | - | - |

- **The reading.** The Twin is level with or ahead of both tabular models on every endpoint here. It leads on
  treatment starts and AKI, and is level on death, both at the landmark and within 24 hours. No interval shows it
  behind.
- **What drove the gain:** Monte Carlo noise and rollout length. One E2 rollout gives an AUROC of 0.81-0.93, against
  0.61-0.66 for one rollout's share. Longer caps remove the landmark's censoring (section 3).
- **Pre-registered A1** (3,500 anchors per design, 32 rollouts, share): the Twin trailed LightGBM on decompensation
  (-0.062, -0.079 to -0.048), ICU death at L24 (-0.057) and the L24 stay > 7 days (-0.047). It led on new
  vasopressor (+0.056) and AKI (+0.037).
- **48-hour landmark, re-run with the follow-up's settings** (L48x, amendment 16b: 809 anchors of test_L48, every
  in-hospital death (455) and 354 weighted controls; 128 rollouts at 4 times A1's cap, 11.2% capped against A1's
  29.1%):

  | Endpoint | Twin, current | LightGBM | TabPFN-3.5 | Twin - LightGBM (95% CI) | Twin - TabPFN-3.5 | Pre-registered Twin (same anchors) |
  |---|---|---|---|---|---|---|
  | ICU death | 0.910 (E2); 0.912 (E0) | 0.915 | 0.925 | -0.005 (-0.027 to 0.018) | -0.015 (-0.037 to 0.006) | 0.884 |
  | In-hospital death | 0.861 (E0) | 0.861 | 0.864 | -0.001 (-0.020 to 0.016) | -0.004 (-0.021 to 0.013) | 0.840 |
  | ICU stay > 7 days | 0.860 (E0) | 0.837 | 0.841 | +0.023 (-0.007 to 0.051) | +0.019 (-0.015 to 0.051) | 0.780 |
  | Death within 24 h | 0.948 (E2); 0.946 (E0) | 0.915 | 0.943 | +0.033 (0.005 to 0.066) | +0.003 (-0.018 to 0.021), E0 | 0.893 |

  - The pre-registered deficits at L48 close: the Twin is now level with LightGBM on death and the stay (family 18,
    Holm P 0.14-1.00) and ahead of its own pre-registered 32-rollout estimate on all four (+0.021 to +0.080, Holm
    P <0.001 to 0.033; family 19). TabPFN-3.5 and the eight learners of amendment 10 were fitted for death within
    24 h; none is ahead of the Twin there (the best learner, elastic-net logistic, 0.940).
  - Censoring explains the ICU-death deficit here as at L24: at A1's cap ICU death has AUROC 0.902 and O/E 1.55 (E2);
    at four times the cap, 0.910 and 1.13.
  - Treatment starts at the landmark (secondary): new IMV 0.847 (E2; LightGBM 0.683), vasopressor 0.938 (0.837), RRT
    0.946 (0.954).
  - Pre-registered A1 at L48 (3,500 anchors, 32 rollouts) against LightGBM: ICU death 0.882 (-0.021, -0.053 to
    0.010), in-hospital death 0.830 (-0.029, -0.049 to -0.010), stay > 7 days 0.797 (-0.036).
- **Regression and time-to-event programs:** pre-registered only (A1). At L24 the remaining stay has an MAE of 41.7 h,
  against 58.6 h for predicting the mean, with 80% coverage 0.70. The C-index for death time is 0.85 (L24) and 0.83
  (R).
- **Across all binary catalog programs** (A1): median Twin AUROC 0.888 (L24), 0.885 (L48) and 0.882 (R).
- Tables:
  - follow-ups: `tables/followup/f1_01_r128.csv`, `f1_03_l24x.csv`, `f2_01_r128.csv`, `f2_04_l24x.csv`; rollout
    curves in `f1_02`, `f1_04`, `f2_02`, `f2_05`;
  - L48x (amendment 16b): `tables/followup/16b_*.csv`; statistics `tables/manuscript/stats_family18.csv`,
    `stats_family19.csv`;
  - pre-registered: `tables/results/a1_*.csv`.

## 1b. Every comparator (amendment 10)

Amendment 10 added comparators on the same anchors:
- **Clinical scores,** from the harmonized raw CLIF values at the prediction time: SOFA, SAPS II, OASIS, NEWS, MEWS,
  qSOFA and SIRS.
- **The datasets' own admission scores:** APACHE IVa (eICU), SAPS 3 (SICdb), the ASA class (INSPIRE).
- **Bedside indicators:** single values such as lactate, shock index, ROX, MAP, KDIGO stage, creatinine, potassium
  and platelets.
- **Learned baselines** on A1's features: L1 and elastic-net logistic regression, a decision tree, random forest,
  extra trees, naive Bayes, k-nearest neighbours and a multilayer perceptron. A1's LightGBM, L2 logistic regression,
  TabPFN and the Twin probe stand beside them.

**The current Twin against the best of each kind** (the best learned model excludes the Twin probe; the best is
picked after the fact, which favours the comparator). Differences are the 128-rollout share (E0) minus the
comparator, paired, with 95% intervals:

*24-hour landmark (L24x):*

| Endpoint | Twin, current | Best other learned model | Twin (E0) - it (95% CI) | Twin probe | Best clinical score | Best bedside value |
|---|---|---|---|---|---|---|
| ICU death | 0.924 (E2); 0.911 (E0) | TabPFN-3.5 0.932 | -0.022 (-0.047 to 0.001) | 0.903 | NEWS 0.803 | - |
| In-hospital death | 0.883 (E0) | TabPFN-3.5 0.876 | +0.006 (-0.010 to 0.023) | 0.867 | SAPS II 0.750 | - |
| ICU stay > 7 days | 0.864 (E0) | TabPFN-3.5 0.864 | -0.000 (-0.035 to 0.031) | 0.852 | SOFA 0.703 | - |
| Death within 24 h | 0.982 (E2); 0.981 (E0) | LightGBM 0.957 | +0.024 (0.010 to 0.041) | 0.974 | SAPS II 0.884 | lactate 0.741 |
| New IMV, 24 h | 0.929 (E0); 0.893 (E2) | extra trees 0.851 | +0.077 (-0.004 to 0.170) | 0.897 | NEWS 0.787 | ROX 0.754 |
| New vasopressor, 24 h | 0.896 (E2); 0.864 (E0) | L1 logistic 0.844 | +0.020 (-0.062 to 0.098) | 0.847 | SOFA 0.644 | MAP 0.587 |
| New RRT, 48 h | 0.964 (E2); 0.958 (E0) | random forest 0.955 | +0.003 (-0.013 to 0.019) | 0.915 | SOFA 0.806 | creatinine 0.921 |

*Rolling anchors (R128):*

| Endpoint | Twin, current | Best other learned model | Twin (E0) - it (95% CI) | Twin probe | Best clinical score | Best bedside value |
|---|---|---|---|---|---|---|
| Death within 24 h | 0.960 (E2); 0.917 (E0) | TabPFN v2 0.957 | -0.040 (-0.064 to -0.018) | 0.954 | NEWS 0.877 | shock index 0.756 |
| New IMV, 24 h | 0.886 (E2); 0.855 (E0) | multilayer perceptron 0.810 | +0.045 (-0.056 to 0.158) | 0.788 | NEWS 0.748 | ROX 0.645 |
| New vasopressor, 24 h | 0.903 (E2); 0.906 (E0) | TabPFN v2 0.847 | +0.059 (0.025 to 0.099) | 0.738 | NEWS 0.749 | shock index 0.694 |
| New RRT, 48 h | 0.959 (E2); 0.951 (E0) | LightGBM 0.932 | +0.018 (-0.005 to 0.043) | 0.928 | SOFA 0.807 | creatinine 0.883 |
| AKI stage >= 2, 48 h | 0.888 (E0) | LightGBM per dataset 0.839 | +0.049 (0.018 to 0.079) | 0.816 | SOFA 0.653 | KDIGO stage 0.620 |
| Sepsis (ASE), 24 h (8 positives) | 0.966 (E0) | multilayer perceptron 0.963 | +0.003 (-0.045 to 0.052) | 0.952 | MEWS 0.719 | lactate 0.394 |
| Hypotension, 4 h | 0.874 (E0) | extra trees 0.907 | -0.032 (-0.080 to 0.010) | 0.861 | NEWS 0.745 | MAP 0.839 |
| Hyperkalaemia (top bin), 24 h | 0.831 (E0) | LightGBM per dataset 0.816 | +0.015 (-0.038 to 0.077) | 0.746 | SOFA 0.642 | potassium 0.788 |
| Thrombocytopenia, 48 h | 0.932 (E0) | elastic-net logistic 0.942 | -0.010 (-0.044 to 0.016) | 0.865 | SOFA 0.785 | platelets 0.909 |

- **Where E2 applies** (death and treatment starts), it is above every other learned model and clinical score. The
  exception is ICU death at the landmark, where TabPFN-3.5 has 0.932 against E2's 0.924. E2's per-anchor values
  were not kept, so these comparisons are unpaired.
- **Death within 24 h:** the share (E0) trails TabPFN v2 by 0.040, but E2 (0.960) does not.
- **Elsewhere:** no comparator is ahead of the share with an interval excluding 0.

- **Clinical scores trail the Twin on every endpoint, by 0.04 to 0.24 AUROC,** and they trail the stronger learned
  models too. Only the weakest learners (naive Bayes, the single decision tree) fall to their level. For death, NEWS
  and SAPS II do best.
- **The datasets' own scores trail the Twin on their own data:**
  - APACHE IVa (eICU), hospital death at 24 h: 0.850, against the Twin's 0.941 (+0.09, 0.03 to 0.16); at 48 h,
    0.766 against 0.901.
  - SAPS 3 (SICdb), hospital death: 0.812 at 24 h against 0.869, and 0.815 at 48 h against 0.882.
  - The exception is APACHE IVa's predicted ICU stay, for stays over 7 days: 0.763 against 0.704 (-0.06, -0.17 to
    0.05).
- **A single bedside value comes close on laboratory-threshold endpoints,** which are thresholds on that value
  (rolling anchors, A1's Twin):
  - platelets for thrombocytopenia 0.919, against 0.939;
  - potassium for hyperkalaemia 0.793, against 0.804;
  - MAP for hypotension 0.863, against 0.906.
- **The learned baselines rank stably:**
  - TabPFN-3.5, LightGBM, random forest and extra trees usually lead;
  - then L1 and elastic-net logistic regression and the multilayer perceptron (L1 leads on rolling
    thrombocytopenia, 0.962);
  - then L2 logistic regression, k-nearest neighbours, the decision tree and naive Bayes.
  - The Twin probe is among the best for death.
- **On A1's anchors** (32 rollouts, the share), the Twin trails the strongest tabular models on death and long stays
  (L24 ICU death 0.858 against 0.932 for TabPFN-3.5). It leads every learner on treatment starts, e.g. L24 new
  vasopressor 0.880 against at most 0.826.
- **Score inputs are missing at some sites,** which weakens those scores there (table a of the amendment 10 section).
  SICdb and NWICU record no GCS; NWICU records no urine output or blood gases.
- **A check:** the windowed SOFA equals the pipeline's own first-day SOFA in 96-99% of 24-hour anchors in six
  datasets.
- Tables: `tables/followup/amendment_10_*.csv` (the AUROC matrices and paired differences per design and subset,
  the native scores, calibration, coverage, by dataset).

## 2. Calibration (O/E: observed over expected; above 1 means the Twin under-predicts)

| Quantity | Current | Pre-registered |
|---|---|---|
| ICU death at 24 h | 0.99 (E0) / 1.01 (E2) with 4x caps (L24x); 1.10-1.38 when cut back to A1's cap | 1.14 |
| In-hospital death at 24 h | 1.10 (L24x) | 1.40 |
| ICU stay > 7 days at 24 h | 0.97 (L24x) | 0.81 |
| Decompensation, 24 h (R) | 1.77 (E2, R128) | 1.83 |
| New IMV / vasopressor / RRT (R) | 0.55 / 0.46 / 0.56 (E2, R128) | 0.69 / 0.49 / 0.66 |
| AKI / hypotension / hyperkalaemia / thrombocytopenia (R) | 0.65 / 0.96 / 0.76 / 0.79 (E0, R128) | 0.73 / 1.04 / 0.78 / 0.81 |
| Step calibration along real futures (D1) | death within 24 h 0.98 (0.86-1.11, R); ICU death 0.95 (0.82-1.12, L24); death within 24 h at L24 0.76 (0.61-0.91); treatments 0.56-0.80 | teacher-forced after 24 h: death 0.97, IMV 0.76, vasopressor 0.59, RRT 0.44 |
| Isotonic recalibration (from the tuning split) | not re-run | O/E 0.60-1.23 (most 0.82-1.13) |

- E2 removes variance, not bias, so its O/E equals the share's.
- **Death** is close to calibrated step by step on rolling anchors and for ICU death at the landmark. Death within
  24 h of the landmark is over-predicted step by step (0.76). The rolling design's remaining short-horizon deficit is
  generation delay (section 3).
- **Treatment starts** are over-predicted step by step, and mostly more so in free generation (drift 0.94-1.37).
- Tables: `tables/followup/f2_*.csv`, `f3_01_*.csv`; `tables/results/a2_*.csv`.

## 3. Mortality: why death was under-predicted (F3)

| Check | Result | Reading |
|---|---|---|
| D1: the step hazards along real futures | O/E 0.98 (R), 0.95 (L24 ICU death); 0.76 for death within 24 h at L24 | Not under-predicted given a real history |
| Landmark rollouts, 4x longer | ICU death O/E 0.99, AUROC 0.911 (A1: 0.858) | The landmark deficit was evaluation censoring |
| D2: the rollouts' death hazard over the real futures' | 0.61 (R) and 0.57 (L24) within 24 h; 0.91 until ICU discharge | Generation delays death rather than omitting it |
| Hourly | Death 0.52-0.56x in the first 6 h, 0.28-0.49x to 48 h; alive discharge 0.84-1.08x | The delay is specific to death |
| D4: the bank | Deaths missing mainly in short stays (MIMIC first day: 3.9% twins against 11.5% real) | The same delay, in whole twins |

Tables: `tables/followup/f3_01` to `f3_05`.

## 4. Attribution (C2, F4, amendment 9)

| Method | Randomized weights (sanity; want low) | Agreement with the model's own editing | Agreement with the generator (F4; ceiling 0.48-0.54) | Status |
|---|---|---|---|---|
| Counterfactual editing (probe) | -0.06 to 0.00 | (the reference) | 0.12-0.20 (hypotension 0.07 of 0.13); sign agreement 0.61-0.66 | **primary method** |
| Attention knockout (amendment 9) | -0.02 | 0.71 (0.53-0.80; per endpoint 0.60-0.74) | 0.06-0.23 on F4's redraw (16c; editing 0.09-0.22 on the same draw, ceilings 0.34-0.58) | **the attention-based method to report** |
| Attention x gradient, group means | -0.03 | 0.14 | not measured | passes, weak |
| Gradient-weighted rollout (Chefer), group means | -0.04 | 0.14 | not measured | passes, weak |
| Integrated gradients | -0.08 to 0.07 | 0.05-0.16 (C2) | -0.10 to 0.08 | weak |
| Attention x gradient, group sums (C2) | 0.53-0.65 | 0.19 | -0.09 to 0.14 | partly fails (size confound) |
| Attention rollout (C2) | **0.96-0.97** | -0.08 | -0.11 to 0.08 | **fails** |

- **Why rollout fails** (amendment 9): in a causal model, rollout reduces to the mask's geometry. The rollout of
  uniform attention, with no model at all, ranks groups like the trained model (0.97) and the random one (1.00). On
  top of that, summing over a group ranks groups by their token counts.
- **The ceiling:** 48 rollouts per edited prompt reproduce the generator's own effects only moderately (split-half
  ceilings 0.13-0.54), so individual-level attribution is noisy at this compute.
- **Knockout's 0.71** with editing is partly expected: both methods remove a group's information.
- **Knockout against the generator** (amendment 16c: F4's 2,698 prompts drawn again, 48 futures each, seed 16; all 160
  anchors):
  - Median Spearman with the redrawn effects: knockout 0.19 (AKI), 0.23 (death within 24 h), 0.06 (hypotension),
    0.06 (new IMV). Editing on the same draw: 0.13, 0.13, 0.09, 0.22. Split-half ceilings 0.58, 0.50, 0.34, 0.54.
  - Per anchor, knockout minus editing (family 20, Wilcoxon): pooled median 0.000 (P = 0.61, 158 anchors); per
    endpoint Holm P 0.99-1.00, except new IMV, where editing is ahead (median -0.10, Holm P = 0.039).
  - Sign agreement on groups whose effect's interval excludes 0: knockout 0.54-0.68, editing 0.62-0.73.
  - F4's first draw gave editing 0.07-0.20: the redraw agrees (0.09-0.22).
- **Pre-registered C2:** editing had the largest deletion drop of the probe's logit (1.88-2.66, against 0.08-0.53 for
  a random order).
- Tables: `tables/followup/f4_01_*.csv`, `f4_02_*.csv`, `amendment_9_01` to `_03`; `tables/results/c2_*.csv`.

## 5. What-if simulation (C1, F5)

- **The null holds.** Two withhold arms with different seeds share the "expected sign" in 0.40-0.55 of prefixes,
  and every mean's interval covers 0.
- **The Twin reproduces observational contrasts.** On 158-195 propensity-matched pairs per treatment it matches 7 of
  9 responses:

  | Response | Observational | Twin |
  |---|---|---|
  | Norepinephrine, MAP | -3.3 | -3.6 |
  | Fluid bolus, MAP | -5.8 | -4.3 |
  | Furosemide, urine (mL) | +337 | +403 |
  | IMV, FiO2 charting | +2.8 | +3.1 |
  | IMV, SpO2 | +1.1 | +1.2 |
  | CRRT, potassium | +0.24 | +0.44 |
  | CRRT, creatinine | +0.50 | +1.04 |

  - **The exception is insulin:** observational +38 mg/dL, Twin about 0.
  - **Fluid-bolus urine is unresolved:** observational -8 (-139 to 137), Twin -108.
- **Factual accuracy:** the factual arm's 16-rollout ensemble roughly halves the persistence error for every response
  (CRPS 0.27-0.61 times the persistence error). 80% intervals cover 0.66-0.91, mostly about 0.70.
- **Timing:** at prefixes cut 1 minute before the real start, giving norepinephrine raises MAP (+0.79, 0.11-1.47).
- **Pre-registered C1:** prefixes cut 0.5-6 h earlier. Norepinephrine's MAP effect was -0.81, the wrong sign, from
  confounding by indication. Insulin -14 mg/dL, furosemide +723 mL, IMV FiO2 charting +2.5.
- **Framing:** observationally faithful simulation, not causal effect estimation.
- Tables: `tables/followup/f5_01` to `f5_04`; `tables/results/c1_01_*.csv`.

## 6. Privacy (A3, unchanged)

- **MUC32_main passes** P1 membership inference in every dataset: AUC 0.549-0.555, TPR at 1% FPR 0.010-0.024
  (pooled AUC 0.538).
- **Across 9 checkpoints, the membership signal follows the fit reached, not the optimizer:** AUC against the member/non-member NLL
  gap has a rank correlation of 0.98-1.00.
- Tables: `tables/results/a3_*.csv`.

## 7. Synthetic data: utility and augmentation (A4, F8)

- **Train on twins, test on real** (A4): TSTR against TRTR gives a ratio of gains of 0.93-1.03 across designs and
  endpoints. Twins are nearly as useful as the same number of real rows.
- **Augmenting a small site** (F8, 20 repetitions per cell):
  - with 100 real stays: +0.038 AUROC (L24, 19/19 cells gain) and +0.021 (R, 25/34). Adding 3 x n twins gives +0.059
    and +0.033. The median real-stay equivalent of 100 real + 100 twins is 179 (L24);
  - with 300 stays: +0.012 (both designs);
  - with 1,000 stays: +0.010 or less.
- **The pre-registered A4 small-site arms** (one draw each) were noisy; F8 replaces them.
- Tables: `tables/followup/f8_01` to `f8_04`; `tables/results/a4_*.csv`.

## 8. Virtual cohorts and similarity (C3, F6, C4, F7)

- **Profiles** (C3): per dataset, the twins match the real held-out stays in age, sex, treatments, event and lab rates
  and stay length. The exception is death, lower in the twins (section 3).
  - Inter-dataset structure: twins 0.73 (0.61-0.83), real control 0.95.
  - eICU hospitals' total variation from real: 0.146 for twins, 0.145 for the control.
  - All 25 rare strata are reached.
- **Hospitals** (F6, 17 eICU hospitals): twins against real, Spearman ICU death 0.45, IMV 0.63, vasopressor 0.66,
  median stay 0.72. The real-against-real control gives 0.61, 0.52, 0.85 and 0.71. The intervals are wide.
- **Similarity** (F7): leave-one-out kNN on the model's state predicts outcomes about as well as TF-IDF or A1's
  features, and better on most endpoints.
  - Higher than TF-IDF on 8 of 12 endpoints, with the largest margin for RRT (0.845 against 0.776). Lower for new IMV
    (0.640 against 0.668) and hyperkalaemia.
  - A randomly initialized model is clearly worse (new IMV 0.500).
  - Pre-registered C4: neighbour lift 2.47-4.42 for the state, against 2.03-3.29 for TF-IDF.
- Tables: `tables/results/c3_01_*.csv`, `c4_01_*.csv`; `tables/followup/f6_01_*.csv`, `f7_*.csv`.

## 9. New sites and site prompts (B2, F9)

- **Unknown site token:** discrimination is unchanged.
  - B2 at L24: AUROC 0.85-0.88 with the site unknown, against 0.81-0.85 known.
  - F9a on R128: within ±0.04 for most endpoints.
  - At a new site, the Twin can be prompted as "unknown".
- **A wrong dataset token is harmful:** RRT -0.30 (-0.39 to -0.18), vasopressor -0.12, AKI -0.05. The prompt
  inherits that dataset's support mask, its recording schema.
- **Leaving a dataset out** (supervised):
  - LightGBM loses 0.04-0.11 AUROC on average against training within the dataset, for new IMV, vasopressor and AKI.
  - The zero-shot Twin beats the left-out LightGBM there by +0.09 to +0.12. It saw the site's unlabelled stays.
- **Novelty:** the Twin and LightGBM degrade alike for the most atypical anchors.
- **Two databases never seen in training** (amendment 16a; inference only, prompts with the database and site tokens
  unknown; `results_followup.md` 16a, gate `external_gate.md`):
  - *Data:* RICD 3.0 (Moscow; intensive care and neurorehabilitation; 5,382 ICU stays) and zigongICU 1.1 (China; ICU
    patients with infection; 2,899 stays), harmonized with new adapters and encoded with the frozen vocabulary (no
    grammar rejections; alignment control 1.0). Rolling anchors R (998) and the 24-hour landmark's case-cohort L24e
    (1,400); 128 futures each.
  - *Scored:* 15 endpoint-database pairs met the pre-registered rule (hourly source, at least 20 positives).
    zigongICU's nursing chart records dates only, so ventilation, AKI and hypotension are not scored there; RICD has
    no RRT source and a daily ventilation flag only; neither has culture data for sepsis.
  - *Discrimination:* Didymos's AUROC is 0.54-0.93 (median 0.72) on the new databases, against 0.83-0.98 for the same
    estimators on the seven. Death within 24 h transfers best (rolling anchors: RICD 0.934, zigongICU 0.829).
  - *Against the transferred models* (trained on the seven, A1's recipe): Didymos ahead on 10 of 15 pairs against
    LightGBM (differences -0.100 to +0.155; none significant after Holm, lowest 0.096; family 15) and on 6 of 15
    against TabPFN-3.5.
  - *Against a local LightGBM* trained on each database's own stays (5-fold cross-fitted by patient): Didymos behind
    on 13 of 15, significantly on 5 after Holm (RICD's three landmark endpoints, -0.089 to -0.218; zigongICU's
    vasopressor, -0.191, and stay > 7 days, -0.101); ahead on zigongICU's death within 24 h at the landmark (+0.117,
    Holm P = 0.0044; family 16).
  - *Against clinical scores* (the best per pair, chosen after the fact; family 17): ahead on 9 of 15, by up to
    +0.153; behind RICD's own recorded SOFA on its landmark mortality endpoints (-0.054, -0.065), ahead of it on
    death within 24 h at the rolling anchors (+0.036).
  - *Calibration:* O/E is far from 1 on the new databases (0.04-3.46; under-prediction of treatment starts and
    laboratory events, over-prediction of RICD's deaths). A logistic recalibration within the database (cross-fitted)
    brings every O/E to 0.93-1.01; transferred LightGBM's O/E (0.23-4.41) needs the same.
  - *Ranks* (post hoc, the study lead's request 2026-10-12; amendment 15's method over the 15 pairs, `tools/paper_external.py
    ranks`): among the 10 predictors that never saw the database, Didymos's mean rank is 3.80 (95% interval
    2.60-5.07), second to TabPFN-3.5 (3.27, 2.07-4.67), then transferred LightGBM 4.33 and the probe 4.80; NEWS, MEWS
    and SOFA 5.5-5.9; Friedman P < 0.001. Didymos against TabPFN-3.5: median -0.007, 6 wins in 15 (Wilcoxon P = 0.45).
    With the local LightGBM added it ranks first (2.07; first on 10 of 15) and Didymos third (4.67). By database:
    zigongICU 3.12 (third of 11, behind the local model and TabPFN-3.5), RICD 6.43 (seventh; SOFA and NEWS ahead).
    On the seven training databases (amendment 15) Didymos ranked first, 1.83 of 20.
  - *Reading:* the generator transfers to unseen hospitals at the level of models trained elsewhere, with lower
    discrimination than at home and a calibration that needs local recalibration; a model trained on the hospital's
    own labelled data stays ahead on several endpoints, most in RICD (long neurorehabilitation stays, sparse
    monitoring).
- **Adapting to the two databases** (amendment 17, 2026-10-12; `results_followup.md` 17; statistics families 21-24):
  - *Design:* each database's patients split once, before any adapted model was trained. TEST 50% is locked and holds
    16a's anchors there (R 490, L24e 683; 13 endpoint-database pairs with at least 10 TEST positives). FT 35% is used
    unlabelled; LABEL 15% serves the few-label methods. Labels are 16a's for every arm.
  - *Improved encoding (Z1, inference only):* a back-off to the nearest kept code cuts the events sent to OOV, e.g.
    RICD's continuous infusions 85.1% to 0.0% and intermittent doses 31.6% to 2.9%, zigongICU's infusions 55.5% to
    0.6%. Discrimination barely moves: Z1 minus Z0 -0.043 to +0.025 on the 13 pairs (none below Holm 1.00). The
    largest change is RICD's new vasopressor (+0.18; 8 positives, descriptive).
  - *Adapted model (F1, Didymos-ext):* MUC32_main trained further on the FT stays, mixed 1:1 by windows with the seven
    databases. The grid chose learning rate 1e-4 and 3 passes (228 steps, 7.5 GPU-minutes). The loss on held-out FT
    patients fell from 2.100 to 1.131 nats per token (RICD 2.095 to 1.321, zigongICU 2.103 to 0.957); the seven
    databases' tuning loss rose by +0.012 to +0.033 (forgetting, small).
  - *F1 against Z0:* ahead on 8 of 13 pairs, behind on 4 (-0.029 to +0.125; family 21, lowest Holm P 0.21). The
    gains are largest on RICD's stay > 7 days (+0.119, 0.014 to 0.214), zigongICU's new vasopressor (+0.125),
    RICD's hyperkalaemia (+0.095) and zigongICU's stay > 7 days (+0.088). RICD's mortality endpoints gain +0.04 to
    +0.06; zigongICU's mortality endpoints lose -0.014 to -0.024.
  - *Against local models:* F1+R against a LightGBM trained on every adaptation label (FT and LABEL, 50% of the
    patients) is level: ahead on 6 of 13, -0.075 to +0.164 (family 22, lowest Holm P 0.081). Against TabPFN-3.5
    transferred it is ahead on 8 of 13; RICD's stay > 7 days gives +0.167 (Holm P 0.021, family 23). Stacking F1 with
    the LABEL-only LightGBM (F1+S) is the best single predictor.
  - *Ranks* (family 24, 13 pairs): F1+S 3.15 (95% interval 2.15-4.23); 16a's local cross-fitted LightGBM 4.00; F1
    4.54; LightGBM on FT+LABEL 4.62; TabPFN-3.5 6.00; Z0 6.31; Z1 6.85. Without the three local LightGBMs: F1+S 2.15,
    F1 3.31, TabPFN-3.5 4.15, Z0 4.54. F1 against Z0: 9 wins in 13 (median +0.043, Holm P 0.16). F1 is level with
    every local LightGBM (FT+LABEL: 6 wins, median -0.002). Every clinical score sits below.
  - *Calibration:* at the landmark F1's O/E is 0.71-1.39 before recalibration (Z0 0.32-3.60) and 0.65-1.24 after
    recalibration on the LABEL anchors. On the rolling anchors' rare 24-48 h events it stays low after recalibration
    (AKI and hyperkalaemia 0.28-0.35): about 250 LABEL anchors per database are too few to fix it.
  - *Label efficiency:* a LightGBM trained on n LABEL stays reaches F1+R's AUROC on 3 of 15 pairs (at 249-393 stays);
    on the other 12 it does not, even with all LABEL stays (466-877).
  - *Privacy:* the M7 gate of the adapted run was not passed. P1 on the seven databases (membership by loss) had one
    group's TPR at 1% FPR upper bound just above the 0.03 limit: eICU 0.0303 at the registered 3,000 per side; then
    AUMC 0.0313 at amendment 17c's larger size (2,747 per side, all its held-out stays). Every other group, the
    pooled test (0.0151, 0.0132 to 0.0169) and P2-P5 were within their limits. F1's futures were drawn under the
    study lead's exemption (amendment 17d: these futures only, deleted after scoring; no twins from this run). Against its
    own FT stays, F1 shows a membership signal on the new databases: AUC 0.564 (0.548 to 0.579), against 0.502 for
    MUC32_main.
  - *Reading:* the cleaner encoding alone changes little. Unlabelled continued training on a third of a new
    database's patients lifts Didymos on most endpoints, most where the base model was weakest (RICD's length of
    stay and mortality, laboratory and medication events). It brings Didymos level with a local LightGBM trained on
    half the patients' labels, and few labelled stays do not catch up with it. zigongICU's mortality is the
    exception (slightly worse). Adaptation costs a measurable membership trace on the adapted stays.
- **Stated limit:** the generator is not retrained without any site. Adaptation to a new database (amendment 17) is
  continued training of the frozen model on that database's own unlabelled stays.
- Tables: `tables/results/b2_01_*.csv`; `tables/followup/f9_01` to `f9_07`, `16a_*.csv`, `17_*.csv`; statistics
  `tables/manuscript/stats_family15.csv` to `stats_family17.csv`, `stats_family21-23.csv`, `stats_family24.csv`.

## 10. Subgroups and ablations (A5, B3; unchanged)

- **Subgroups** (A5): Twin discrimination is similar across sex, age band, admission type and race; calibration varies
  more (e.g. in-hospital death O/E 2.25 in AmsterdamUMCdb at L24).
  - Subgroups need at least 30 positives and 30 negatives; race is recorded in MIMIC, eICU and NWICU only.
  - Tables: `tables/results/a5_*.csv`.
- **Checkpoints** (B3): probe AUROC for ICU death is 0.897-0.913 across the 8 checkpoints. MUC32_main was adopted on
  the privacy frontier.
  - Tables: `tables/results/b3_01_*.csv`.

## 11. The application toolkit (amendment 11)

The toolkit (`src/icutwin/toolkit`, `docs/toolkit.md`) packages the validated capabilities as seven tools:
- Forecast, What-if, Explain and Similar patients;
- the Cohort synthesizer and the Trial designer;
- Teaching cases.

Every output carries its paper evidence and caveats. Its validation ran on the RTX 5090 host with new draws.

- **V1, conformance.** The toolkit reproduces the paper.
  - *Forecast:* on R128's 1,050 anchors, every primary program's AUROC and O/E lies within rollout noise of a paper
    run. Death within 24 h: toolkit 0.880; F1's 32-rollout expectation 0.881; A1's replicate 0.903.
  - *E2:* within 0.011 of F2's curve.
  - *What-if:* F5's factual contrasts reproduced. Norepinephrine MAP -3.85 (F5 -3.58); furosemide urine 405 mL
    (403); both A/A nulls cover 0.
  - *The prompts:* identical to the paper's (1,050 of 1,050).
  - *Three verdicts, all reported:*
    - the pre-registered rule passed 14 of 15 checks. Sepsis, with 8 positives, fell outside A1's narrow interval;
    - a post-hoc rollout-noise check passed 7 of 9 programs;
    - the study lead's adjusted criterion (within noise of A1 or F1) passed.
  - *Why the rules disagree:* the paper's own two runs differ by more than rollout noise. On death within 24 h they
    are 3.7 noise SDs apart.
- **V2, recalibration of E2.** Isotonic maps from the tuning split.
  - *Treatment starts:* O/E near 1 (IMV 0.55 to 1.16, vasopressor 0.46 to 1.00, RRT 0.55 to 0.84).
  - *Discrimination:* the maps' flat steps tied risks. AUROC moved by more than 0.005 for five programs (worst:
    thrombocytopenia -0.05).
  - *Transfer:* the tuning split's miscalibration did not carry over everywhere (death within 24 h: overcorrected to
    O/E 0.75).
  - *Adopted:* only where AUROC held at 32 and 16 rollouts: death within 24 h, AKI, hypotension, hyperkalaemia.
- **V2b, logistic recalibration of the treatment starts** (post hoc, the study lead's decision). logit(p') = a + b
  logit(p) on the same tuning fit; AUROC unchanged by construction. All three met the adoption rule.
  - *Vasopressor:* O/E 0.46 to 1.02 (1.04 at 16 rollouts), ECE 0.030 to 0.009.
  - *RRT:* O/E 0.55 to 0.93 (0.90), ECE 0.016 to 0.002.
  - *IMV:* O/E 0.55 to 1.43, overcorrected: the tuning split's raw O/E was 0.40. Its ECE still falls from 0.031 to
    0.013.
  - *In force:* seven maps, four isotonic and three logistic.
- **V3, privacy of prompted and intervened sampling** (gate v1's PENDING item). Recorded in the gate as **PASS**, the
  study lead's decision. The rule's own verdict is INCONCLUSIVE, solely for INSPIRE.
  - *Loss at P1's sizes* (V3x, up to 3,000 per side, replacing V3's 500): pooled AUC 0.531 (0.524-0.537), TPR at
    1% FPR 0.013 (0.011-0.017). Seven of eight datasets PASS, including AmsterdamUMCdb, eICU and HiRID, which were
    inconclusive at 500.
  - *INSPIRE:* INCONCLUSIVE, sample-limited. Every eligible held-out stay gives 790 per side; its TPR interval is
    0.009-0.039 against 0.03 (AUC 0.568, upper bound 0.597).
  - *Extraction:* no rollout reproduced any real future's first 32 tokens, for members or non-members, with or
    without an intervention. Event Jaccard AUC 0.516 (0.493-0.540); intervened 0.482.
  - *Controls valid:* the copying generator fails, the private reference passes (AUC 0.493).
  - *The gate report* (`runs/audit/m7_MUC32_main/report.json`, all three copies) states INSPIRE's numbers. The
    report as it was is kept beside it.
- **V4, trial designer** (kept predictions; relative risk 0.8).
  - *Covariate adjustment:* using the Twin as the prognostic covariate needs 0.63-0.86 times the unadjusted sample
    (ICU death 0.64, in-hospital death 0.63, AKI 0.67, RRT 0.74, vasopressor 0.84, IMV 0.86). LightGBM needs
    0.64-0.93, and SOFA, SAPS II, OASIS and NEWS 0.80-1.00.
  - *On eICU:* APACHE IVa needs 0.81 for ICU death, the Twin 0.67.
  - *Enrichment:* the Twin's top quarter needs 0.18-0.30 of the unenriched size. The clinical scores' top quarter
    needs 0.25-0.88.
  - *Checks:* type I error 0.04-0.06 in 92 of 108 score cells, and the analytic size within 5% in 92 of 108.
    - The adjusted analyses' type I error ranges from 0.0375 to 0.0715 at relative risk 0.8, and from 0.0385 to
      0.079 at 0.7 (amendment 13's Clopper-Pearson intervals; `tables/manuscript/`).
    - The highest values are the Twin's in the small eICU-only subsets: vasopressor 0.079 (0.068 to 0.092) and RRT
      0.072 (0.061 to 0.084), both at 0.7. They are a limit of the adjusted test there.
  - *Caveat:* L24x is a case-cohort sample, so its event rates and sizes are not population values.
- **V5, teaching cases:** three objectives on the bank (shock, respiratory failure, renal support), three typical
  cases each, and a drill.
  - *Checks:* every case meets its objective, and every page is self-contained.
  - *Novelty:* not recomputed on this host. The twins passed the bank's production filter.
  - *Case figure:* `runs/toolkit/v5/figure_case.*` (a twin; internal).
- **Tables:** `tables/followup/amendment_11_*.csv`.

## 12. Open items

- **Done by amendment 16 (2026-10-12):** knockout against the generator (section 4), the 48-hour landmark with E2,
  128 rollouts and longer caps (section 1), and transfer to two databases never seen in training (section 9).
- **Done by amendment 17 (2026-10-13):** adaptation to RICD and zigongICU (section 9).
  - *Open:* the adapted run's M7 gate was not passed, and its use is limited to amendment 17's scored futures
    (17d).
  - *Open:* the rolling anchors' rare events stay mis-calibrated after the LABEL recalibration.
  - *Open:* zigongICU's mortality endpoints do not gain.
- **Prospective evaluation:** none; for the limitations and future work when the article is revised (the study lead's
  plan, 2026-10-11). Amendments 16 and 17's results are not yet in the manuscript or figures.
- **IMV's recalibration overcorrects** (O/E 1.43 after the logistic map): the tuning split's raw O/E (0.40) differs
  from the held-out anchors' (0.55). A larger tuning fit would settle it.
- **INSPIRE's membership test** cannot narrow further from its held-out split. A shorter-stay design would add stays.
- **The manuscript, phase 1 (2026-10-11):**
  - *Done:*
    - the theme, outline, terminology and verified references (kept with the manuscript);
    - the main figures 1-8 (`figures/`, drawn by `tools/paper_figures.py`) and their legends (kept with the
      manuscript);
    - the supplementary figures S1-S16 (in the order of the main figures; `figures/supplementary/`) and their legends
      (kept with the manuscript);
    - amendments 13-15's reporting statistics and Table 2, the leaderboard (`tables/manuscript/`).
  - *Names:* the model is **Didymos** (Didymos 1.0 = MUC32_main), the tools are **the Didymos toolkit**.
  - *Multiplicity:* with Holm across the 13 endpoints, Didymos is ahead of LightGBM on new IMV (P = 0.027), AKI
    (P = 0.0078) and death within 24 h at the landmark (P = 0.013); vasopressor (P = 0.052) and RRT (P = 0.13) do
    not survive the adjustment; no endpoint favours LightGBM.
  - *Every scored endpoint* (amendment 14, exploratory):
    - On the 165 comparisons A1 made against LightGBM, Didymos (32 futures) is ahead on 113. The median difference
      is +0.014 AUROC (0.010 to 0.019).
    - At a 5% false-discovery rate, 49 comparisons favour Didymos and 6 LightGBM. Under Holm, 30 and 3.
    - Mortality and the remaining stay trail in that configuration.
  - *Label efficiency* (A1, pre-specified): a LightGBM trained in one database needs more than 1,000 labelled stays
    to match Didymos in 50 of 78 primary endpoint-database cells, and never matches it in 38.
  - *Ranks across endpoints* (amendment 15, post hoc; S5, Table 2):
    - On the 12 primary endpoints every comparator was fitted on, Didymos has the best mean rank of 20 predictors,
      1.83 (95% interval 1.08-2.92), and the best AUROC on 8. Next: TabPFN-3.5 3.42 and LightGBM 3.67.
    - Wilcoxon across endpoints, Holm: ahead of TabPFN-3.5 on 9 of 12 (P = 0.034), of LightGBM on 11 (P = 0.024) and
      of every clinical score on 12 (P = 0.0093). Friedman P < 0.001.
    - Across all 165 scored endpoints (the four predictors scored on every one), mean rank 1.58, first on 102;
      LightGBM 2.00.
    - Item 14's first run fell back to the 15 primary programs (LightGBM first there, 1.73 against 2.87); the
      protocol's clarification reports it.
  - *Fig. 7d:* the synthetic-patient case, cleared by the study lead on 2026-10-11 (shock, prompted from AUMCdb).
  - *Fig. 8* (the study lead's request, 2026-10-11): drawn from the clinician study's answer-only analysis, pooled
    aggregates only. Framed as explanatory at the study lead's direction: estimates against chance and zero, no
    make-or-break thresholds in the figure or legend; the pre-registered rules and their outcomes go in the Methods.
  - *Open:* the clinician data's export checks (or their status stated), phase 2's text.
- **The clinician study** (amendment 12): answers for R01-R20 were received and analysed answer-only. The study lead
  attested their authenticity on 2026-10-11; the protocol's export checks have not run. The full
  record is kept with the study's internal records.
  - *A, realism:* the twins were not detected as synthetic. Accuracy was 0.41, below chance, so the pre-specified
    equivalence verdict is "not met" and the non-detection analysis is post hoc.
  - *B1, B2:* twin responses were rated close to real ones, and the model's factors were rated relevant. Neither
    pre-specified contrast was established.
  - *C, decision support:* the card improved ICU-death and AKI prognosis.
  - *To report it:* the packets' own exports must pass `collect`.

## Reproducing

**Sources.** The tables come from the aggregate JSON files under `runs/paper/` (and
`runs/eval/<run>/calibration_masked.json`) on the AutoDL host.

- **Local copy:** the same tree under `runs/paper/` in the working copy (git-ignored, per the data-use agreements).
  The models' hidden-state caches (`states.npz`) are left out of it; `tools/server/paper_states.py` regenerates them
  on a GPU.
- **Not kept:** evaluation rollouts, deleted after the analyses.

**Regenerating.** Either command rebuilds a document and replaces the CSV files in its directory:

```bash
python tools/paper_results.py --root runs/paper --eval runs/eval --out docs/paper/results.md --tables docs/paper/tables/results
```

```bash
python tools/paper_results_followup.py --root runs/paper --out docs/paper/results_followup.md --tables docs/paper/tables/followup
```

**The CSV files** hold the cells as rendered: rounded, with combined cells such as "0.97 (0.95 to 0.98)". The JSON
aggregates have full precision.
