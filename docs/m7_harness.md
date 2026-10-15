# M7: evaluation harness (S8), status 2026-10-05

The privacy gate has its own protocol (`docs/m7_privacy.md`). This page covers fidelity and forecasting, both
built and checked against their controls; the remaining areas are listed at the end.

## Fidelity (`eval.fidelity`)

Generated stays against held-out stays, from their tokens: the code mix (Jensen-Shannon divergence; per dataset
too), each code's value bins, code rates (Spearman), events and minutes per stay (Kolmogorov-Smirnov), the in-hospital
death rate, a discriminator on token unigrams and bigrams (cross-validated AUC; bigrams see order and pairings that
marginals miss), and precision / recall / density / coverage (Naeem et al. 2020) in an SVD of the TF-IDF.

**Controls** (plan S8): the positive control (train stays against held-out) sets each metric's band; four negative
controls must be flagged (a metric that flags none is not trusted). On the 2% subset (1,234 held-out stays):

| Set | Flagged by | Key values |
|---|---|---|
| Positive control | (sets the band) | discriminator AUC 0.559, code JSD 0.005, value JSD 0.007, coverage 0.96 |
| Shuffled times | discriminator | AUC 0.852 |
| Shuffled values | discriminator, recall, coverage | AUC 0.983, recall 0.15 |
| Swapped dataset labels | per-dataset code mix, discriminator | per-dataset JSD 0.46, AUC 0.725 |
| Collapsed generator | nearly every metric | coverage 0.05, recall 0.00, AUC 1.00 |

## Forecasting (`eval.forecast`, the CoMET protocol)

Held-out stays as known 24 hours after ICU admission (prospective prefixes) are continued n times by the engine
(`TwinEngine.simulate`); an outcome's predicted probability is its share among the rollouts, scored against the
real stay: in-hospital death, IMV / vasopressor / RRT after the cutoff, ICU stay over 7 days (AUROC, AUPRC,
expected calibration error, observed / expected). The reader is checked on real stays (its ICU-stay flag agrees
with the real length of stay for over 90% of subset stays) and the scores on a perfect and a shuffled forecaster.

## Diversity (`eval.diversity`, plan *Diversity assurance* 6)

Vendi score ratio (pooled and the smallest per dataset), rare-code recall, the extreme-bin share, the 99th
percentile of events per stay, distinct event 3-grams, exact and near duplicates, the synthetic-to-synthetic over
real-to-real nearest-neighbour distance, the copy rate against training stays (the held-out stays' 5th percentile
as the threshold: 0.05 for stays never trained on), the stratum and field total variation, a dataset classifier's
confusion (real cross-validated against synthetic) and the correlation of the inter-dataset MMD matrices. Ratios
are gated on both sides of 1. The harness runs them with the copying generator as a fifth negative control. On the
2% subset (1,234 held-out stays):

| Set | Flagged by (diversity) |
|---|---|
| Positive control | vendi 1.05, rare-code recall 0.97, nn ratio 1.08, copy rate 0.17 (training stays have near neighbours among other training stays: patients with several stays), site confusion 0.02, MMD correlation 0.98 |
| Shuffled values | Vendi ratio 0.65, MMD correlation 0.87 |
| Swapped dataset labels | site confusion 0.98, MMD correlation -0.08, stratum TV 0.76 |
| Collapsed generator | Vendi ratio 0.03, duplicates 0.99, distinct 3-grams 0.03, nn ratio 0, rare codes 0.04 |
| Copying generator | copy rate 1.00 (no other metric: fidelity cannot tell copies from fresh stays) |

## Utility (`eval.utility`)

TSTR against TRTR on 24-hour landmark tasks read off tokens, the same way for real and generated stays: in-hospital
death, ICU stay over 7 days, and new IMV, vasopressor or RRT after the landmark (logistic regression on the window's
TF-IDF token counts, AUROC on held-out stays); `utility_ratio` = (TSTR - 0.5) / (TRTR - 0.5). Controls on the
subset: real training stays in the generator's place reach the TRTR value (death 0.81 against 0.81, ratio 0.99;
ICU stay over 7 days 0.95), and swapped outcomes (a window joined to another stay's continuation) fall to chance
(0.42-0.49 for death and length of stay).

## Quality pilot (`eval.quality`)

For a candidate that passed the privacy gate: dataset-conditioned, grammar-constrained samples (vLLM) scored per
dataset and pooled with the fidelity and diversity harness and its controls (the copy rate against a pool of
training stays disjoint from the positive control), plus utility; samples are evaluation artefacts, deleted after
the report.

## Quality pilot of M32_main (2026-10-05, after its gate passed)

1,000 dataset-conditioned, grammar-constrained samples per dataset (vLLM, the engine), scored against as many
held-out stays per dataset and pooled (7,000 against 7,000), with the controls; samples deleted after each run.
Pilot (a) used the engine as it was; its aggregate diagnostic (`tools/server/pilot_diagnose.py`) found two sampling
defects, corrected for pilot (b): unknown header values from the training-time augmentation
(`constrain.HeaderAugmentation`) and other datasets' tokens in a dataset's twins (`generate.support`).

| Pooled | Positive control | Pilot (a) | Pilot (b), corrected |
|---|---|---|---|
| Grammar-valid samples | | 100% | 100% |
| Discriminator AUC | 0.501 | 0.985 | 0.984 |
| Code JSD / value JSD | 0.001 / 0.002 | 0.009 / 0.009 | 0.008 / 0.007 |
| Events per stay KS / ICU stay KS | 0.012 / 0.010 | 0.054 / 0.029 | 0.046 / 0.022 |
| In-hospital death rate difference | 0.003 | 0.005 | 0.010 |
| Blood-pressure rule broken | 0.002 | 0.002 | 0.002 |
| Coverage / recall | 0.973 / 0.945 | 0.928 / 0.923 | 0.928 / 0.921 |
| Vendi ratio / rare-code recall | 1.017 / 0.954 | 0.898 / 0.954 | 0.867 / 0.918 |
| Duplicates / copy rate (fresh: 0.05) | 0 / 0.051 | 0 / 0.023 | 0 / 0.029 |
| Strata TV / largest field TV | 0.030 / 0.033 | 0.071 / 0.061 | 0.042 / 0.036 |
| Dataset classifier confusion / inter-dataset MMD correlation | 0.001 / 0.997 | 0.001 / 0.997 | 0.000 / 0.998 |

Per dataset, the discriminator AUC is 0.85 (MIMIC) to 0.97 (INSPIRE) in both runs. **Utility** (TSTR against TRTR,
24-hour landmark, pooled; AUROC trained on real, on synthetic, ratio of the gains over chance), pilot (b): death
0.845 / 0.836 (0.97), ICU stay over 7 days 0.813 / 0.797 (0.95), new IMV 0.767 / 0.763 (0.99), new vasopressor
0.799 / 0.799 (1.00), new RRT 0.861 / 0.836 (0.93); pilot (a): 0.84-0.96. Every negative control was flagged.

**What still tells the twins apart** (pilot (b)'s diagnostic): the median lengths match (MIMIC 1,077 against 907
tokens, eICU 633 / 617, HiRID 745 / 728, NWICU 360 / 382), but the upper tail is longer (99th percentile 30-60%
higher) and long synthetic stays repeat one event about twice as often as real ones (99th percentile of the most
repeated event: HiRID 980 against 460, INSPIRE 1,384 against 325); the discriminator's strongest features are now
spread over value bins. Neither is a sampling defect the grammar or the sampler could remove without truncating the
distribution (the plan rules out top-k, top-p and repetition penalties): it is the model's long-horizon drift, the
target of the next training round (more training, a larger model, M9's ablations). The corrected support mask also
lowers the Vendi ratio (borrowed tokens had inflated it).

## Twin forecasting of M32_main (2026-10-05, partial)

The corrected engine (support mask), 300 held-out stays per dataset still in the ICU at 24 h, 32 rollouts each. The
run was stopped when round 2 (authenticity) began; three datasets had finished and were scored, the rollouts
deleted. AUROC (ECE; observed over expected):

| Dataset | Death | New IMV | New vasopressor | New RRT | ICU stay over 7 days | ICU stay: PIT KS / spread-skill |
|---|---|---|---|---|---|---|
| MIMIC | 0.867 (0.045; 1.21) | 0.970 (0.039; 0.90) | 0.892 (0.071; 0.71) | 0.956 (0.056; 0.62) | 0.821 (0.051; 0.82) | 0.057 / 0.77 |
| eICU | 0.791 (0.048; 1.18) | 0.904 (0.032; 0.92) | 0.816 (0.033; 0.80) | 0.669 (0.015; 0.47) | 0.845 (0.038; 0.68) | 0.105 / 0.41 |
| SICdb | 0.839 (0.039; 0.81) | 0.876 (0.045; 0.98) | 0.810 (0.065; 0.97) | 0.888 (0.017; 0.87) | 0.822 (0.034; 1.11) | 0.053 / 0.70 |

The rollouts rank the outcomes well but over-predict new vasopressor and RRT starts in MIMIC and eICU (observed over
expected 0.47-0.80, the pilot's runaway treatments seen prospectively) and are under-dispersed in the ICU stay
(spread-skill 0.41-0.77; 1 when calibrated): both are round 2's (`docs/m7_authenticity.md`).

## Report (`eval.report`)

One static HTML page of aggregates: the gate's statuses and tests, the harness tables with flagged cells marked
(positive control, negative controls, candidates; pooled and per dataset), utility, and forecasting.

## Clinical rule, dispersion and probes (2026-10-05)

- **Clinical rule** (`fidelity.bp_violation_rate`): within a time group and site, diastolic below systolic and MAP
  between them, on the bins' medians; real held-out stays break it in 0.46% of 75,802 checked groups (binning), stays
  with shuffled values in 23.6%.
- **Forecast dispersion** (`forecast.dispersion`): of the ICU length of stay over the rollouts, the randomized PIT's
  distance from uniform, the ensemble CRPS and spread against skill, with the share of rollouts censored before an
  ICU discharge; tested on calibrated, under- and over-dispersed synthetic forecasters.
- **Linear probes** (`eval.probes`): the frozen model's last-layer state at the 24-hour landmark (last token, mean)
  against the window's TF-IDF counts, cross-validated logistic regression per outcome.

- **Task realism** (`eval.realism`): the factory prior's tasks on a real cache and on a candidate's (decoded twins,
  the same grid concepts): prevalence gap, difficulty gap (ExtraTrees), the rank correlation of the features' mutual
  information with the label, and TSTR / TRTR. On the subset, two real caches (train and held-out stays): prevalence
  gap 0.02, difficulty gap 0.07, TSTR / TRTR 1.06; outcomes scrambled across stays: TSTR / TRTR 0.39. The mutual
  information's rank correlation is weak even between real caches at 300 rows per task (0.14 against 0.06): it
  needs larger tasks to say anything.

- **Per-hospital heterogeneity** (`diversity.hospital_tv`): the eICU hospitals of the twins against real
  held-out stays (total variation of the site field; real against real on the subset 0.27, about 100 hospitals and
  600 stays, a collapsed generator 0.74).

## Not built yet

Task realism and per-hospital heterogeneity in a pilot run (both built after the pilot).
