# M7 round 2: authenticity of the twins (pre-registered 2026-10-05)

Status: targets and definitions written **before** any of the measurements below was taken (study lead's decisions of
2026-10-05: scope is the realism of the twins; targets pre-registered; diagnose first, retraining only after the
diagnosis report and the study lead's decision; no deliverable bank until every target holds). The numbers below are
fixed by this commit; results are reported against them, unchanged.

## Why

M32_main passed the M7 privacy gate and the quality pilot (`docs/m7_harness.md`) found its twins valid, private and
useful, but distinguishable from real stays: a TF-IDF discriminator separates them with AUC 0.85 (MIMIC) to 0.97
(INSPIRE) per dataset, 0.98 pooled (real against real 0.50). Medians of stay length match, the tail does not (p99
30-60% longer), long twins repeat single events about twice as often, diversity is lower (Vendi ratio 0.87), and
twin forecasting over-predicts rare treatment starts and is under-dispersed.

## Sets

Per dataset (MIMIC, AmsterdamUMCdb, eICU, HiRID, SICdb, NWICU, INSPIRE):
- **Twins:** 1,000 dataset-conditioned, grammar-constrained samples of the candidate (`eval.quality generate`:
  ancestral sampling at temperature 1, no truncation, the support mask and the header-augmentation inverse;
  seeds `eval.quality.sample_seeds(0, dataset, 1000)`, pilot (b)'s).
- **Real:** 1,000 held-out first windows and, for the positive control, 1,000 train first windows, drawn as
  `eval.quality evaluate` draws them (seed 0).
Twins are evaluation artefacts: they stay on the approved host and are deleted after each run (aggregates only).

## Views and measures

- **Discriminator AUC:** `eval.fidelity.discriminator_auc` (5-fold cross-validated logistic regression on the
  sublinear TF-IDF of token unigrams and hashed bigrams), twins against real, on the view's token sequences.
- **First-24-h view:** the prefix, the header and every timeline token before the landmark 24 h after the ICU
  admission marker (`eval.utility.landmark`, interval tokens at their median gaps); a stay that ends or leaves the
  ICU earlier keeps its whole timeline; no end block.
- **Length-matched whole-stay view:** every token of the stay; the twins resampled with replacement (seed 0) so that
  their events-per-stay distribution over the real set's deciles equals the real set's. *Amended (amendment 1,
  below): both sets subsampled without replacement to the same counts per decile.*
- **Family views:** only the tokens of one family, in order: vitals (`VITAL//`), labs (`LAB//`), medications
  (`MED_CON//`, `MED_INT//`, `MED_ORD//`), respiratory (`RESP//`), timing (interval tokens).
- **Events per stay:** the stay's event tokens (`eval.fidelity.features`); percentiles over the set.
- **Most-repeated event:** per stay, the count of its most frequent event token; percentiles over the set.
- **Pooled metrics:** the harness (`eval.fidelity`, `eval.diversity`) and utility (`eval.utility`) on the pooled
  7,000 twins against 7,000 held-out stays, with the positive control (`eval.quality evaluate`).

## Targets (every dataset unless stated)

| # | Measure | Target |
|---|---|---|
| A1 | Discriminator AUC, first-24-h view | <= 0.70 |
| A2 | Discriminator AUC, length-matched whole-stay view | <= 0.80 |
| A3 | Discriminator AUC, each family view (vitals, labs, medications, respiratory, timing) | <= 0.80 |
| A4 | Events per stay, 90th and 99th percentiles, twins over real | within 0.80-1.20 |
| A5 | Most-repeated event, 99th percentile, twins over real | within 0.75-1.25 |
| A6 | Vendi ratio (pooled); rare-code recall (pooled) | >= 0.95; >= positive control - 0.03 |
| A7 | The other pooled harness metrics (all but the discriminator, the length tail and the Vendi ratio, which A1-A6 replace); utility ratio per task (pooled); grammar validity; privacy gate of any new model | not flagged against the positive control; >= 0.90; 100%; PASS |

The deliverable bank is released only when A1-A7 hold for the candidate.

## Amendment 1 (2026-10-05, before any twin was measured)

The audit's tests (`tests/test_authenticity.py`, real training stays against real held-out stays of the 2% subset)
found the length-matched view defective as defined: resampling with replacement duplicates stays, the duplicates sit
in both cross-validation folds, and the discriminator recognizes them (real against real: AUC 0.79 instead of about
0.5). Both sets are now subsampled without replacement to the same counts per decile of the real set's events per
stay (real against real: 0.57 on the subset). A2's limit (0.80) and every other definition are unchanged.

## Diagnosis (round 2, step 2; run of 2026-10-05, 21:51-22:39 server time)

M32_main as it sampled (pilot (b)'s engine and seeds: 7,000 twins), the registered sets, the positive control at
about 0.50 in every view at full scale. Outputs (aggregates): `runs/eval/M32_main/auth/{audit,model,quality,targets}`
and `*_canon` for the same twins with their headers sorted (below). Twins deleted after.

**Baseline against the targets:** A1 and A2 fail in every dataset, A3 holds in every dataset, A4 and A5 fail in six
(NWICU passes), A6 fails (Vendi ratio 0.87; rare-code recall 0.918 against 0.954), A7: utility (0.93-0.99) and
grammar (100%) hold, the harness flags density beyond the replaced metrics.

### Causes, ranked

1. **Header order (a representation artefact, not content).** The model draws the header fields in any order (the
   training transform shuffles them so that any subset can condition the rest); the encoder writes real headers in
   the spec's order. The order alone separated twins: with the headers sorted (`engine.HeaderOrder`, lossless; the
   same twins), the pooled discriminator fell from 0.985 to 0.753 and the first-6-h view from 0.95-1.00 to 0.58-0.78
   per dataset. Real stays with only their headers shuffled: 0.58 -> 0.94 on the subset.

| AUC, canonical headers | eICU | NWICU | MIMIC | HiRID | SICdb | AmsterdamUMCdb | INSPIRE | Pooled |
|---|---|---|---|---|---|---|---|---|
| First 6 h | 0.576 | 0.597 | 0.618 | 0.627 | 0.648 | 0.674 | 0.776 | 0.642 |
| First 24 h (A1 <= 0.70) | 0.605 | 0.633 | 0.637 | 0.696 | 0.730 | 0.745 | 0.806 | 0.699 |
| First 72 h | 0.613 | 0.658 | 0.672 | 0.739 | 0.747 | 0.779 | 0.824 | 0.730 |
| Whole stay | 0.622 | 0.680 | 0.706 | 0.743 | 0.760 | 0.806 | 0.832 | 0.753 |
| Length-matched (A2 <= 0.80) | 0.624 | 0.680 | 0.697 | 0.738 | 0.747 | 0.806 | 0.827 | 0.747 |
| Family views, largest (A3 <= 0.80) | 0.591 | 0.648 | 0.629 | 0.689 | 0.702 | 0.742 | 0.769 | 0.667 |
| Timing / event-kind order | 0.533 / 0.542 | 0.561 / 0.579 | 0.565 / 0.585 | 0.546 / 0.656 | 0.555 / 0.622 | 0.563 / 0.699 | 0.641 / 0.656 | 0.530 / 0.594 |

2. **Long-horizon drift into repetition.** The AUC grows with the window in every dataset (pooled 0.64 at 6 h, 0.75
   for the whole stay). Repetition (an event equal to one of the stay's last 20): AmsterdamUMCdb twins 0.21-0.23
   against 0.15 from day 1; after 14 days HiRID 0.40 against 0.29, NWICU 0.62 against 0.26, INSPIRE 0.68 against
   0.37. Late in long twins the model's own NLL falls below real stays' (AmsterdamUMCdb, HiRID, SICdb, INSPIRE:
   loops are predictable) or rises above it (eICU 1.62 against 1.18 beyond 10,000 tokens, MIMIC: wandering). The
   most-repeated event's 99th percentile is 1.4-3.7 times real (A5 fails in six datasets).
3. **Too few early discharges, so long tails.** Discharge hazard on days 1-2, twins against real: MIMIC day 2 0.34
   against 0.44, AmsterdamUMCdb day 1 0.30 against 0.37, HiRID day 1 0.45 against 0.55, INSPIRE day 1 0.32 against
   0.43. Events per stay, 99th percentile: +11% (NWICU) to +92% (SICdb); 90th +4% to +32% (A4 fails in six); medians
   longer in MIMIC (+22%) and AmsterdamUMCdb (+16%).
4. **INSPIRE under-learned.** Even its first 6 h separate (0.78) and its family views are the highest (0.64-0.77):
   the smallest dataset, drawn 2.4 times under the privacy limit.
5. **Diversity:** Vendi ratio 0.87 (control 1.02); rare-code recall 0.918 (control 0.954); density flagged.

**Ruled out:** temperature (tuning NLL lowest at T = 1.0 in every dataset; 0.95 and 1.05 cost 0.002-0.004 nats);
grammar distortion (probability mass outside the generation mask on real held-out sequences: 0.1-0.2% inside time
groups, 0.1-0.9% after an interval, end blocks 0-7.5%, the end block's canonical diagnosis order, short); the
support mask removes 3-5% in the header phase, the training augmentation's unknown values (#108); within-stay
value coherence (lag-1 same-bin agreement within a few points of real, except HiRID blood pressures 0.38 against
0.46).

**After the header fix** (`targets_canon.json`): A1 holds in eICU, NWICU, MIMIC, HiRID (fails AmsterdamUMCdb 0.745,
SICdb 0.730, INSPIRE 0.806); A2 in all but AmsterdamUMCdb (0.806) and INSPIRE (0.827); A3 in all; A4 and A5 only in
NWICU; A6 fails; A7 fails on density.

## Remediation (round 2, step 3; study lead's decision 2026-10-06: sampler fixes first, training decided after)

- **Canonical header order** (`engine.HeaderOrder`): every generated stay's header sorted into the spec's order,
  losslessly; on by default in the engine.
- **The length marginal** (`generate.los`): twins drawn unconditionally (twice as many), then per ICU length band as
  many kept as the dataset's training stays put there (`los_bands.json`, training stays only, bands read with the
  interval tokens' median gaps); the model's stay given its length is untouched. Conditioning on the band through the
  goal block was planned and dropped: training never wrote a goal block with one field (#110).
- Not addressed by the sampler: the drift into repetition within long stays, INSPIRE's content, diversity. Training
  changes are decided after the re-evaluation.

## Re-evaluation (round 2, step 4; run of 2026-10-06, 01:57-02:46 server time)

The remediated configuration against the registered targets: M32_main unchanged (weights as frozen, gate PASS); the
engine with the canonical header order, the support mask and the header-augmentation inverse; per dataset 2,000
twins drawn (`sample_seeds(0, dataset, 2000)`: pilot (b)'s 1,000 seeds and 1,000 more) and 1,000 kept by the
training stays' ICU length bands (`generate.los`; no band fell short). Real sets as registered (the diagnosis's
held-out and training stays). Outputs (aggregates): `runs/eval/M32_main/auth_b/{audit,quality,model,targets,
explain}.json`; twins deleted.

**The twins' length before resampling** (2,000 per dataset; percent of twins / of training stays):

| Dataset | < 1 d | 1-2 d | 2-4 d | 4-7 d | 7-14 d | >= 14 d | No ICU discharge |
|---|---|---|---|---|---|---|---|
| MIMIC | 18.6 / 21.3 | 27.6 / 30.1 | 25.1 / 24.9 | 13.2 / 11.6 | 9.9 / 7.6 | 4.2 / 4.4 | 1.5 / 0.06 |
| AmsterdamUMCdb | 29.2 / 36.3 | 28.4 / 22.6 | 14.9 / 13.7 | 8.3 / 8.5 | 8.5 / 8.6 | 8.6 / 9.2 | 2.3 / 1.05 |
| eICU | 33.9 / 34.7 | 24.7 / 27.7 | 22.9 / 21.1 | 8.9 / 9.2 | 5.1 / 5.3 | 1.8 / 2.0 | 2.8 / 0.01 |
| HiRID | 44.0 / 49.5 | 27.8 / 23.4 | 14.6 / 13.5 | 6.0 / 6.7 | 4.4 / 4.7 | 2.4 / 2.2 | 1.0 / 0 |
| SICdb | 43.0 / 43.1 | 21.7 / 21.2 | 16.6 / 16.3 | 8.2 / 9.3 | 5.5 / 5.8 | 3.6 / 4.1 | 1.4 / 0.1 |
| NWICU | 28.8 / 25.7 | 30.0 / 28.7 | 21.0 / 23.4 | 10.0 / 11.3 | 6.4 / 6.7 | 3.3 / 4.2 | 0.7 / 0 |
| INSPIRE | 31.4 / 42.2 | 34.7 / 30.3 | 16.4 / 14.2 | 6.3 / 7.4 | 3.7 / 3.4 | 3.9 / 2.4 | 3.8 / 0.1 |

Too few day-1 discharges in AmsterdamUMCdb, HiRID and INSPIRE (MIMIC on days 1-2), and **runaway twins**: 0.7-3.8%
never leave the ICU before the token budget forces the end (real: at most 0.1%; AmsterdamUMCdb 1.05%, where long
stays exceed the window). SICdb's bands match, yet its events p99 was 1.92 times real: the tail was the runaways and
dense long stays, not too few discharges. Resampling gives "no ICU discharge" its training share, so it discards
nearly all runaways: a correction of the marginal, not of the model.

**Against the targets** (canonical headers, the diagnosis -> resampled; bold: fails after):

| Dataset | A1 first 24 h | A2 length-matched | A3 largest family | A4 events p90, p99 | A5 top event p99 | Holds |
|---|---|---|---|---|---|---|
| MIMIC | 0.637 -> 0.619 | 0.697 -> 0.697 | 0.629 -> 0.625 | 1.32, 1.34 -> 1.17, 1.14 | 1.47 -> 1.10 | A1-A5 (was A1-A3) |
| eICU | 0.605 -> 0.623 | 0.624 -> 0.640 | 0.591 -> 0.607 | 1.09, 1.33 -> 1.16, 1.05 | 1.35 -> 1.08 | A1-A5 (was A1-A3) |
| NWICU | 0.633 -> 0.626 | 0.680 -> 0.684 | 0.648 -> 0.667 | 1.12, 1.11 -> 1.12, **1.35** | 1.03 -> 1.00 | A1-A3, A5 (was A1-A5) |
| HiRID | 0.696 -> **0.718** | 0.738 -> 0.749 | 0.689 -> 0.680 | 1.04, 1.31 -> 0.96, 1.13 | 2.91 -> **1.88** | A2-A4 (was A1-A3) |
| SICdb | 0.730 -> **0.703** | 0.747 -> 0.751 | 0.702 -> 0.698 | 1.07, 1.92 -> 1.17, **1.62** | 1.51 -> **1.55** | A2, A3 (unchanged) |
| AmsterdamUMCdb | 0.745 -> **0.765** | 0.806 -> **0.802** | 0.742 -> 0.745 | 1.27, 1.19 -> 1.17, 1.04 | 1.44 -> **1.41** | A3, A4 (was A3) |
| INSPIRE | 0.806 -> **0.801** | 0.827 -> **0.829** | 0.769 -> 0.753 | 1.21, 1.39 -> 1.05, **0.75** | 3.71 -> **1.46** | A3 (unchanged) |
| Pooled | 0.699 -> 0.706 | 0.747 -> 0.755 | 0.667 -> 0.668 | | | |

- **A6 fails:** Vendi ratio 0.869 -> 0.891 (>= 0.95); rare-code recall 0.918 -> 0.919 (>= 0.924, the control's
  0.954 - 0.03).
- **A7 fails:** grammar 100% and the privacy gate (same model) hold; the harness flags density beyond the replaced
  metrics (precision, recall, coverage 0.946, 0.919, 0.934 against the control's 0.964, 0.945, 0.973); utility
  ratios death 0.969, ICU stay over 7 days 0.950, IMV 0.960, vasopressor 1.004, **RRT 0.868** (was 0.929: about 150
  positives in the test set, so about one standard error, but below 0.90). The length-tail flag is gone.
- The view AUCs moved within about 0.02 either way: the length marginal was not what the discriminators read.
- A4's p99 ratio rests on the ten longest of 1,000 stays per set, and moved with the resampled composition (NWICU:
  more long twins, which are denser than real long stays; INSPIRE: the runaways gone).

**What still separates the twins** (the explain measure, new in this step: `authenticity explain`; the
discriminator on unigrams alone against the view's, the tokens and kind bigrams whose share of a stay differs most
by Welch's t over stays, tokens per kind, events per time group; the positive control's largest |t| is 3-4):

- **Content, not order:** the unigram-only discriminator matches each view (first 24 h: AmsterdamUMCdb 0.771,
  INSPIRE 0.790, HiRID 0.719, SICdb 0.690; MIMIC, eICU and NWICU 0.625-0.634); the kind bigrams' differences follow
  the composition (more labs, more lab-lab pairs), and events per time group are close to real.
- **Too much blood-gas monitoring in the European datasets, from day 1:** AmsterdamUMCdb twins chart 19% more lab
  tokens in the first 24 h (232 against 195 per stay; the blood-gas panel's electrolyte and CO-oximetry deciles lead,
  t 5-7) and 21% more respiratory tokens; SICdb 7% more labs, with the lowest pO2, SO2 and SpO2 deciles
  over-represented (t 4-5; SpO2's lowest decile 2.4 times real over the whole stay); HiRID 12% more labs.
- **Value calibration:** HiRID twins under-produce the highest arterial pressure deciles (systolic and mean, deciles
  7-8: 0.94 against 1.69 per stay in the first 24 h, t -4.7 to -6.2) and over-produce the low-middle ones.
- **Treatment starts after day 1 over-generated** (pooled; prevalence at the utility tasks' 24-hour landmark, twins
  against real training stays): RRT 7.0% against 3.1%, IMV 10.2% against 6.0%, vasopressor 10.7% against 6.8%, while
  death (9.3% against 10.7%) and ICU stay over 7 days (15.8% against 16.4%) are close; as the forecasting rollouts
  over-predicted (O/E 0.47-0.80).
- **Loops on rare items in long stays:** SICdb CRRT tokens 2.3 times real per stay; INSPIRE mechanical circulatory
  support 9 times and the lowest FiO2-setting decile 5 times; HiRID fluid inputs +42%, intermittent doses +31%; and
  the runaways above.
- **INSPIRE timing and medications:** fewer 15-30 minute gaps (t -8.4), a 1-5 minute gap its stays never have (4-6%
  of twins: a sampler defect, #111, fixed for the next run), 17% fewer intermittent medications in the first 24 h.
- MIMIC, eICU and NWICU: diffuse differences near the control's noise (largest |t| 3.7-4.6 against 3.3-3.9).

**Conclusion.** The sampler fixes solved what the sampler can: the header order and the length marginal. MIMIC and
eICU, the two largest datasets, now meet every per-dataset target. What remains is the model's content in the
smaller, densely charted datasets (AmsterdamUMCdb, HiRID, SICdb and INSPIRE, 16,000-34,000 stays each): present on
day 1 (under-fitting), compounding over long stays (loops, runaways, treatment starts), and the pooled diversity,
density and RRT utility. No sampler change within the plan's rules (no truncation, no penalties) reaches these, so
the deliverable bank stays off and training is the study lead's decision (below).

## Training decision (study lead, 2026-10-06: option 1, the recommended course)

Evidence for the choice:
- **Under-fitting, not memorization.** The failing datasets are the small, densely charted ones, and their errors
  start on day 1. M32_main's tuning NLL was still falling at its end (1.616 at step 1,500, 1.561 at 2,566); its
  membership AUCs are 0.53-0.56 (upper bounds 0.54-0.58 against the 0.60 limit).
- **Passes are bound by privacy, not compute.** The membership AUC grows about 0.013-0.032 per pass (M_main against
  M32_main; INSPIRE at 5.9 passes failed); M32_main trained in 38 minutes.
- **Size helped at equal data.** S -> M (1.7 times the parameters) lowered tuning NLL on every dataset by 0.025-0.043
  nats with a smaller train-tuning gap, most on medication tokens (`docs/m5_training.md`).

Options:
1. **A larger model at the same passes** (recommended): an intermediate width, e.g. 16 layers x 1,024 (about 270M
   parameters), passes 3 and 2.5 as M32_main, goal fields dropped at random (#110). Risk: larger models memorize
   faster per pass; the M7 gate decides.
2. **More passes on M:** bounded by P1 (SICdb's upper bound 0.566 at 2.87 passes, INSPIRE's 0.557 at 2.39); small
   gains expected.
3. **Either with dropout** to slow memorization per pass: an untested trade against fit.

Each candidate costs about 3 hours on the host (training 1-1.5 h, the M7 gate under 1 h, this audit about 1 h) and
is judged against the same targets; it is adopted only if its gate passes and its audit is no worse than M32_main's
on A1-A7.

## Third run: ML32_main (pre-registered 2026-10-06, before the run and before any reference twin was drawn)

**The candidate.** M32_main's data (r3 tokens), passes (3; 2.5 for NWICU and INSPIRE), context (32,768; RoPE base
1e6), batch (0.52M tokens per step) and recipe (AdamW 0.9/0.95, weight decay 0.1, 100 warmup steps, cosine to 10%,
clipping 1.0, bf16), with two changes:
- **Width:** preset `ML`, 16 layers, width 1,024, 16 heads, 4 key-value heads, SwiGLU 4,096 (269.5M parameters, 2.2
  times M). Peak rate 4e-4: the best rate scaled as 1/width in the pilot (1e-3 at 256, 5e-4 at 512), which gives
  4.5e-4 from M's 6e-4; GPT-3 Medium (1,024 wide, 24 layers) used 3e-4.
- **Goal-field subsets (#110):** a goal block, when inserted (25% of samples), holds a random non-empty subset of
  the nine fields, its size uniform (`TransformConfig.goal_subsets`).

Path: packing parity and the 50-step DDP smoke test at 32k (memory and throughput), the run, the freeze, the M7
gate in force (v1.1 with amendment v2's P5; `tools/server/m7_gate.sh`), and, if it passes, this audit.

**Reference.** M32_main audited again with the step-4 sampler plus #111 (`runs/eval/M32_main/auth_c`); the same
sets, seeds and sizes for both.

**Adoption rule.** ML32_main replaces M32_main as the candidate if all hold:
1. its M7 gate is PASS;
2. its audit (`runs/eval/ML32_main/auth`) meets at least as many of the 35 per-dataset targets A1-A5 as the
   reference, and no pooled target the reference meets (A6; A7's harness, utility and grammar parts) fails;
3. its tuning NLL over all datasets is lower than M32_main's (1.561).

Otherwise M32_main stays the candidate. Either way the deliverable bank stays off until A1-A7 hold for the candidate.

### Result (2026-10-06): not adopted; paused by the study lead

- **Reference** (`auth_c`, 05:38-06:27 server time): M32_main with #111 meets **24 of 35** per-dataset targets (step
  4: 22). The same model and seeds give other twins where vLLM's batching changes the numerics (MIMIC and eICU were
  identical to step 4, the others not), and the p99 ratios swing most (SICdb's top event 1.55 -> 1.24, NWICU's
  events 1.35 -> 1.18; AmsterdamUMCdb A2 now held, A4 not): a repeat of one configuration moves the count by about
  two. #111 lowered INSPIRE's largest family view from 0.753 to 0.725. Pooled: A6 and A7 still fail (Vendi 0.900,
  rare-code recall 0.917; RRT utility 0.882).
- **ML32_main trained** (06:10-07:21; 2,566 steps, 1.287B tokens, 0.35M tokens/s, peak 78.2 GB per GPU) to tuning
  NLL **1.563 against M32_main's 1.561**: within 0.009 per dataset (eICU 1.550 against 1.551 and INSPIRE 1.532
  against 1.534 slightly better; AmsterdamUMCdb, HiRID, MIMIC, NWICU and SICdb 0.003-0.009 worse). Criterion 3
  fails, so **ML32_main is not adopted, whatever its gate and audit; M32_main stays the candidate.**
- **Where the capacity went** (P1, the same 3,000 + 3,000 stays per dataset as M32_main's): members' NLL 0.007-0.013
  lower than under M32_main, non-members' within 0.005 of it, so the membership AUC rose by 0.010-0.016 in every
  dataset (MIMIC 0.577 [0.562, 0.592], SICdb 0.564, INSPIRE 0.553; still PASS against 0.60). At these passes the
  larger model learned the training stays, not the population. P2: PASS.
- **Paused** by the study lead once ML32_main was certain to fail the rule (07:34): the gate stopped after P1 and P2 (the
  partial audit samples deleted), no audit run. ML32_main stays frozen on the host (`runs/train/full/ML32_main`,
  FROZEN.json); `tools/server/m7_gate.sh ML32_main` would resume the gate from P1.

