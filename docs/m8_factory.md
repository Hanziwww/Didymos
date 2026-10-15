# M8: ICL dataset factory (S9), status 2026-10-05

Version 1, built and tested on real held-out stays (the plan's real benchmark); the same code reads decoded
synthetic stays once a bank exists.

- **Cache** (`factory.cache`): per stay, statics (age, sex, the header's categorical fields, pre-ICU length of stay,
  admission hour, prior ICU stays); the hourly grid of the best covered numeric concepts over the first 72 ICU
  hours (the hour's last value); label timelines in hours since ICU admission (length of stay; in-hospital death;
  first IMV, vasopressor, RRT and AKI stage >= 2 onsets from the derived concepts; 24-hour SOFA; maximum AKI
  stage). Saved as one `.npz`.
- **Task prior** (`factory.tasks`): a cohort (datasets), a trigger (6-48 h: the observation window), a target
  (binary: an event within a horizon among stays without it before the trigger, in-hospital death, still in the
  ICU; multiclass <= 10: remaining-stay band, SOFA band; regression: a concept's value at trigger + horizon),
  features (concepts x last / mean / min / max / count over the window, and statics), rows log-uniform, context
  share 0.1-0.9. Missing values become the context mean with indicators; constant columns are dropped.
- **Contracts:** `ICUPrior.get_batch` returns TabICL's `(X[B,T,H], y[B,T], d, seq_lens, train_sizes)`, zero padded;
  `get_ts_batch` the time-series layout `(X_ts[B,T,L,C], M_ts, X_static, y, train_sizes)`.
- **Tests** (`tests/test_factory.py`, the 2% subset): cache contents and round trip; the contract (shapes, d <= H
  <= 100, <= 10 classes numbered 0..k-1, every class in context and query, context 0.1-0.9, no constant feature,
  zero padding, reproducible from the seed); the leakage guard (changing every value at or after the trigger leaves
  the features unchanged); a TabICL forward pass (a small random-initialized TabICL 2.2.0 on our episodes).

- **Task-diversity controls:** the TabICLv2 ExtraTrees filter (an ExtraTrees model fitted on the context must
  beat chance on the query: AUROC above 0.5 plus the larger of 0.05 and 2.33 chance standard deviations, so a small
  query's luck does not pass; R^2 above 0.05), the TabICLv2 row curriculum (`CURRICULUM`: 1,024; 400-10,240;
  400-60,000, log-uniform), deduplication by the task's spec hash (without the episode's size and split), and
  `stats` counting every draw's fate by family, with family floors (a family the filter pushes below its minimum
  share of accepted episodes, by default half an equal share, is drawn until it is back above it). On the subset 60 of 427 draws were accepted (too few rows 142,
  classes 79, uninformative 146), every family represented.
- **Task cards:** every episode's card holds its hash, family, a description in words, its windows (trigger,
  input, target), cohort, feature names, rows and context rows, the label summary (prevalence or class shares,
  or mean and SD) and the filter's score.

- **ACES** (`factory.aces`): one predicate set over the harmonized codes for all seven datasets (ICU admission
  and discharge, in-hospital death, an IMV device record, a vasoactive start or running infusion, a CRRT or IHD
  record), the trigger at the ICU admission, and a config per binary task (death after t; an ICU discharge in
  (t, t + h]; an event in (t, t + h] among stays without it in [0, t]; every task keeps the stays in the ICU at
  t). `extract` runs the ACES reference implementation (es-aces 0.7.3) over MEDS shards. **Parity** on the subset's
  1,234 held-out stays (`tests/test_aces.py`): the factory's labels, with the cache's onsets taken from the same
  codes (`code_onsets`), and ACES's agree for every stay on cohort and label for death at 24 h, in the ICU at 48 h,
  IMV within 24 h after 24 h, vasopressor within 24 h after 12 h and RRT within 48 h after 24 h (it found #107).

- **Mined targets** (MOTOR-style, from the grid's concepts; 30% of binary and regression draws by default): a
  concept above its population 90th percentile, or below its 10th, within the horizon, among stays with a value
  there; a concept's change from its last value before the trigger.
- **Realism** (O'Prior; half the tasks by default): missingness completely at random, at random on another column
  and not at random on the value itself, multiplicative noise, unit changes of a few columns, all on the raw
  features before the context-mean imputation and seeded by the task's hash; and shift stress, the context drawn
  mostly from one dataset and the query from the others. Tested: reproducible, more missing indicators, the
  leakage guard still holds, shifted contexts are dominated by their home dataset.

- **From the bank** (`factory.stream`): decoded twins (the decoder sets each one's ICU window from its admission
  marker and discharge) build the same cache; a producer bank's shards (every part's) keep each twin's importance
  weight; `episodes` streams without end. With a work directory (`derived_dir`, since 2026-10-07) the stays get the
  derived concepts of real stays' definitions (below); without one, event onsets come from the ACES predicates'
  codes and KDIGO and SOFA are unknown. Tested with real held-out sequences standing in for twins
  (`tests/test_stream.py`).

- **Derived concepts of twins** (2026-10-07): `generate.clif_out` writes stays as canonical CLIF
  (the plan's detok -> MEDS -> CLIF), inverting the ELF configs that made their codes, and `clif_out.derived` runs
  the real stays' `meds.derived.derive` on them (treatment episodes, KDIGO, the 24-hour SOFA, NEE, fluid balance,
  ASE). The same function serves real stays' stored token sequences, so a real benchmark is built exactly as the
  bank's cache. Checked on the seven 2% subsets (`tools/clif_roundtrip.py`; every stay; agreement of the ICU's
  episodes, its highest KDIGO stage and the 24-hour SOFA total):

  | Dataset | Stays | Canonical CLIF vs MEDS -> CLIF: episodes (IMV, vaso, RRT) | KDIGO exact | SOFA exact | MEDS -> CLIF vs tokens -> CLIF (labels' decode): KDIGO exact (mean) | SOFA exact (MAD) |
  |---|---|---|---|---|---|---|
  | MIMIC | 1,806 | 1.000, 1.000, 0.998 | 0.974 | 0.978 | 0.788 (1.33 -> 1.38) | 0.730 (0.29) |
  | AmsterdamUMCdb | 385 | 1.000, 1.000, 1.000 | 0.992 | 0.982 | 0.861 (1.03 -> 1.13) | 0.766 (0.26) |
  | eICU | 4,026 | 0.998, 1.000, 1.000 | 0.988 | 0.944 | 0.818 (0.82 -> 0.82) | 0.778 (0.24) |
  | HiRID | 699 | 1.000, 1.000, 1.000 | 0.999 | 0.250 (#118) | 0.899 (0.60 -> 0.65) | 0.774 (0.25) |
  | INSPIRE | 276 | 1.000, 0.975, 1.000 | 0.982 | 0.478 (#118, #119) | 0.978 (0.27 -> 0.25) | 0.725 (0.31) |
  | NWICU | 574 | 1.000, 1.000, 1.000 | 0.978 | 0.988 | 0.871 (0.49 -> 0.44) | 0.845 (0.17) |
  | SICdb | 583 | 1.000, 0.997, 1.000 | 0.997 | 0.988 | 0.833 (1.16 -> 1.29) | 0.810 (0.22) |

  What MEDS carries, the writer restores (episode starts agree to the minute). The residue: a lab's one time is its
  result time (KDIGO reads creatinine at collection), events outside the MEDS window are absent (the 7-day creatinine
  baseline; an infusion's stop after ICU discharge: INSPIRE's 7 vasopressor stays), and the canonical SOFA misses the
  CNS score of HiRID and INSPIRE (#118) and INSPIRE's value-less respiratory rows extend its FiO2 (#119). Through the
  tokens, with the factory's labels' decode (gaps drawn, values at their bins' medians, below), episodes agree
  everywhere (starts 0.06-0.32 h apart, RRT 0.3-1.4 h), KDIGO's highest stage keeps its mean within 0.13 of the
  truth (exact 0.79-0.98) and the SOFA total within 0.15 (exact 0.73-0.85). Drawn values instead raise KDIGO: two
  equal creatinine values drawn independently within a bin can differ by 0.3 mg/dL, a stage 1 rise.

  **Which decode labels a stay** (MEDS -> CLIF against tokens -> CLIF; KDIGO exact and mean, SOFA exact):

  | Decode | MIMIC KDIGO (truth mean 1.33) | eICU KDIGO (0.82) | SOFA MIMIC, eICU |
  |---|---|---|---|
  | gaps and values drawn within their bins | 0.678 (1.64) | 0.775 (1.01) | 0.710, 0.777 |
  | the same with within-bin rank persistence (AR(1)) | 0.695 (1.62) | 0.781 (0.97) | 0.713, 0.776 |
  | gaps drawn, values at their bins' medians | **0.788 (1.38)** | **0.818 (0.82)** | **0.730, 0.778** |
  | gaps and values at medians | 0.820 (1.29) | | 0.739 |

  KDIGO's criteria take minima over many values (the lowest creatinine of 7 days, a urine-output rate): a draw
  low in a wide bin makes a later ordinary value look like a tripling (MIMIC creatinine stage 3: 151 stays from MEDS,
  305 from drawn values; urine-output stage 3: 238 and 450). Real stays' consecutive same-bin values do keep their
  rank (`tools/rank_persistence.py` on the subsets' train stays: creatinine 0.48, bilirubin 0.57, FiO2 0.78, PEEP
  0.97, urine output 0.26-0.52, vitals 0.13-0.49), but persistence does not remove that minimum's bias. So
  `factory.stream` labels a stay from values at their medians on the clock its drawn gaps make (means within 0.05 of
  the truth), and builds its features from drawn values with the real rank persistence (`rank_persistence.json`
  beside the vocabulary, `decode.load_persistence`): continuous features, unbiased labels, one clock. Real stays and
  twins are decoded the same way, so their comparison stays like with like.

- **Time to event** (`labels_tte`, `ICUPrior.tte_episode`): hours from the trigger to an event's first onset, or
  to the ICU discharge where the stay is censored, among stays in the ICU at the trigger without the event before
  it; `binary_at_horizon` reduces it censoring-aware (censored before the horizon without the event: left out). The
  reduction agrees with the 'event within the horizon' labels for every stay both have.

- **Multi-label** (`labels_multilabel`): several events within the horizon as one label matrix, a column not
  applying (NaN) where its event began before the trigger; `multilabel_to_binary` is exactly each event's binary
  task (tested).

- **Forecasting** (`ICUPrior.get_forecast_batch`): the time-series inputs over [0, trigger) and `Y_future[B, T, H_f,
  C_t]`, target concepts' hourly values over the next H_f hours, with their mask; inputs never reach past the
  trigger (tested).
- **Counterfactual** (experimental; `counterfactual_labels`): per prefix, the outcome's share among rollouts under
  do(treatment) and do(no treatment) (`engine.simulate` with an `Intervention`): y1, y0 and the twin-world effect.

Not built yet: code-level mined tasks beyond the grid's concepts (every frequent code's occurrence and time to
event), and running the counterfactual rollouts (the assembly is built).
