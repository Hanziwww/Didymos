# M11: the bedside in-context model (Phase G, "TrajPFN"), status 2026-10-07

TrajPFN predicts a patient's outcome as a probability distribution by referencing two things: the **complete
trajectories of other patients**, each with the task's label timeline, and the **patient's own trajectory** up to
the prediction time (the factory's paradigm, `docs/m10_factory.md`). It is trained from scratch on twins only.

Scope of this round (study lead, 2026-10-07): "set up the tests and confirm that the model is capable of learning (since
the complete twin library has not yet been generated)". The model and its test suite are built, and a learning
confirmation runs on the 13,416-twin development bank, **on the laptop GPU** (the study lead's exception to the
no-local-training rule; the RTX 4080 host serves Paper P). The production pool, the main run (stages 1-3) and the
generic-share sweep follow once the twin library exists.

## Decisions (study lead, 2026-10-07)

| Topic | Decision |
|---|---|
| Scope | Tests and a learning confirmation; no main run |
| Context | Other patients' complete labelled trajectories plus the patient's own past: row-level in-context learning over states of those trajectories |
| Encoder | Trained end-to-end with the in-context transformer |
| Bedside | A GPU server per site (size M, context banks cached) |
| Pilot hardware | The laptop's RTX 4090 Laptop GPU (16 GB), at most about 24 GPU-hours, an 80% memory cap |
| Baselines | The context's base rate (gated), ExtraTrees, TabICLv2 and TabPFN-3.5 on snapshots (reported) |

## Architecture

```
episode ──► TOKENS per patient: events (channel, value, decile, flags), state starts and stops, statics, own LABEL
            tokens at their t_resolved, row marks; each with its time
        ──► PATCH POOLING: tokens grouped into patches (the hourly grid, cut at every row time), multi-head
            attention pooling into one vector per patch, plus its time and the episode's task vector
        ──► TEMPORAL ENCODER: a causal transformer over each patient's patches (continuous-time RoPE in hours);
            the state h(t) at every patch end; the same weights for context and query patients
        ──► ROWS: context train rows h_c(t) + label embedding at its label times; query test rows h_q(t_k) + mask
        ──► IN-CONTEXT TRANSFORMER: task tokens and train rows attend to each other; test rows attend to them only
            (QASSMax); the context's keys and values never depend on the queries
        ──► HEADS: Bernoulli; ordinal; bar distribution (values, forecasts); competing-risk hazard (time to event,
            multi-horizon)
```

### Inputs (`icl/batch.py`)

- **Tokens:**
  - events: channel, standardized value `ev_z` (squashed, then piecewise-linear encoded), source decile, flags;
  - state intervals: a start token (with its value) and a stop token, each one minute after the change. The
    factory's query cut leaves out a state starting at the cut and clips one running past it, so a change exactly
    at t is not yet known at t;
  - statics at their availability time;
  - the patient's own labels as tokens at their `t_resolved`. Context and query states are thus built the same way:
    a state at t holds exactly the labels known by t;
  - a mark at each row time.
- **Patches:** the hourly grid, cut at every row time, so the state at a row time t contains every token up to t and
  none after it (event-triggered times stay exact).
- **Rows:**
  - context patients contribute a train row at each label time (at most 64 per patient in training, drawn uniformly
    and never by outcome, so the base rate is kept; invalid labels stay as rows with their reason);
  - queries contribute a test row at each prediction time.
- **Channels:**
  - a channel is its global embedding plus a random identity vector drawn per (episode, channel) from a seed of its
    own;
  - anonymized episodes zero the global part, and training drops it for 10% of channels, so channels are also known
    by behaviour;
  - the identity of one channel never depends on which other channels the episode holds (a channel seen only after
    t changes nothing at t).
- **Units never name a channel.** Value labels and forecast targets are standardized per episode from the context's
  labels (median and IQR, then symlog), and a descriptor threshold is read as z only.
- **Task tokens:**
  - a header: label type and encoding, cadence, the context's declared selection (policy, log case odds, guaranteed
    cases), and with a shown descriptor also horizon, family and eligibility;
  - one token per descriptor term (channel, direction, z, sustain);
  - one per multi-horizon horizon;
  - one per forecast channel.

### The model (`icl/model.py`, `icl/heads.py`)

| Part | S | M |
|---|---|---|
| Token embedder and patch pooling | d 128, 4 pooling heads | d 128, 4 pooling heads |
| Temporal encoder (causal, RoPE on hours, SwiGLU, RMSNorm) | 4 layers, d 256 | 8 layers, d 384 |
| Task encoder | 2 layers, d 128 | 2 layers, d 192 |
| In-context transformer (QASSMax) | 6 layers, d 256 | 12 layers, d 512 |
| Parameters (the channel table included) | 9.9M | 56.3M |

Heads (every loss a negative log-likelihood):
- **binary and in-hospital death:** a Bernoulli logit;
- **ordinal (up to 10 classes):** categorical logits;
- **values and forecasts:** a bar distribution of 128 bins over the standardized, squashed value;
- **time to event:** a discrete competing-risk hazard (none, event, death, discharge) on 25 log-spaced bins to 7
  days and beyond, with the censored likelihood;
- **multi-horizon:** the event's cumulative incidence at each horizon, from the same hazard.

An episode's loss is the mean over its valid targets; a batch's is the mean over its episodes.

Deviation from the plan: the patch encoder is attention pooling, not two set-attention layers. Patches hold 1-60
tokens, cross-token interaction is left to the temporal encoder, and pooling is linear in tokens and identical on
every device.

### Kernels

The laptop runs PyTorch's SDPA (memory-efficient kernel) on length buckets; the temporal encoder is causal and
right-padded, so padding is never read. A varlen path (packed sequences, `torch.nn.attention.varlen`) for the Linux
main run, with its parity test, comes with that run.

## Factory changes made for the model

| Version | Change | Why |
|---|---|---|
| phase_f.3 | Forecast tasks' context patients carry `lab_yf` / `lab_mf` at every cadence time | Row-level in-context learning needs labelled context rows of every type |
| phase_f.4 | Queries get their own event budget (`query_events`: 16,384 / 32,768 / 65,536 by stage) | Thinning a query up to its last prediction time was not causal (QA #129) |

## Tests (`tests/test_icl.py`, CPU, float32)

- **Causality, by definition:** a query's output at t_k equals its output when the episode is cut at t_k.
- **Changes exactly at t_k:** a state starting or stopping exactly at t_k is not yet known at t_k.
- **No leakage forward:** later values and unresolved own labels never reach earlier rows.
- **Context labels:** they reach the queries only through the rows, never the queries' own states.
- **Invariances:**
  - to the order of context patients;
  - a packed batch equals its episodes run alone;
  - relabelling an anonymized episode's channels changes nothing;
  - channel identities do not depend on the other channels.
- **Heads:** the hazard likelihood against brute force (and incidence plus survival summing to one); the bar
  distribution is a density.
- **Robustness:** degenerate episodes (no context, every label invalid, a forecast without observations) run
  finitely.
- **Training pieces:** the optimizer groups, the schedule, context subsets.
- **A tiny overfit.**

Mutation checks (outside the suite): a non-causal temporal encoder fails the causality tests, and a state's start
token at the change itself fails the "exactly at t_k" test.

## Learning confirmation (pre-registered before the pilot runs)

Sets (frozen, `tools/icl_train.py make-set`, 1,000 episodes each):
- `twin_val`: the bank's validation twins;
- `twin_train`: training twins, unseen episodes;
- `real_dev`: real tuning-split stays, standardized with the training twins' statistics.

The base-rate baseline is the context's own label distribution on the same rows (`heads.baseline_losses`).

| # | Check | Pass |
|---|---|---|
| L0 | S fits 64 fixed episodes | every head's loss falls >= 90% toward its floor (0; the bar grid's log bin width, -2.08, for values) |
| L1 | twin-val, by label type, against the base rate | bootstrap 95% interval of the per-episode loss difference below 0 for every label type with >= 30 episodes |
| L2 | twin-val episodes with >= 16 context patients, rerun with 0 (shown descriptor only), 2, 4, 8, 16 of them | the loss does not rise with size (1% tolerance) and falls from 2 to 16 (interval below 0) |
| L3 | binary twin-val episodes, half the positive or half the negative context patients removed | the predicted mean rises with the context's prevalence (within-episode slope, interval above 0, >= 20 episodes) |
| L4 | hidden-descriptor and anonymized twin-val episodes | each beats the base rate (interval below 0) |
| L5 | real dev episodes with real contexts | beat the base rate (pooled interval below 0); the gap to twin-val reported |
| L6 | twin-train against twin-val | the gap reported |

Reported only: AUROC, log loss and Brier against ExtraTrees, TabICLv2 and TabPFN-3.5 (`tools/icl_baselines.py`,
`tools/icl_compare.py`), and MAE and R^2 for values. At pilot scale (about 2,000 uses per twin, about 10^5-10^6
episodes against the 10^7-10^8 of a full run) the model is not expected to beat the tabular foundation models.

## Results

L0 (S, 64 fixed episodes, 300 steps on the laptop GPU): every head reaches its floor (binary 0.453 → 0.000,
time to event 15.77 → 0.001, ordinal 1.507 → 0.000, values 2.79 → -2.079, in-hospital death 0.359 → 0.000).
**Met.**

Throughput on the laptop GPU (stage 1, bf16):
- S: about 31 episodes/s (about 700k tokens/s), 5.3-7.7 GB at batches of 8-16 episodes;
- M: 12.9 episodes/s compute only, with activation checkpointing;
- the data workers keep the GPU fed.

### The S pilot (200,000 episodes, 12,500 steps of 16; 1.4 hours on the laptop GPU)

- **Settings:** Muon + AdamW at 8e-4, 500 warmup steps, cosine decay to 10%, EMA 0.999, bf16, at most 64 train
  rows per context patient, about 42 episodes/s.
- **Held-out monitoring:** 400 twin-val episodes; the EMA's mean per-episode loss against the base rate's:

  | Step | Binary | Ordinal | Time to event | Values | Forecasts | In-hospital death | All |
  |---|---|---|---|---|---|---|---|
  | 2,000 | 0.344 | 1.263 | 3.20 | 1.29 | 1.03 | 0.41 | 0.795 |
  | 6,000 | 0.322 | 1.147 | 2.93 | 1.00 | 1.08 | 0.34 | 0.715 |
  | 12,500 | 0.306 | 1.138 | 2.81 | 0.93 | 1.00 | 0.28 | 0.683 |
  | Base rate | 0.383 | 1.276 | 4.52 | 1.67 | 1.59 | 0.42 | 0.987 |

- **The checks** (EMA weights, 1,000-episode sets; `runs/icl/S_pilot/checks_ema.json`):

| # | Result | Status |
|---|---|---|
| L1 | Binary -0.080 [-0.095, -0.066] (683 episodes, TrajPFN better in 67%; median within-episode AUROC 0.67); ordinal -0.275 [-0.362, -0.190] (113); values -0.616 [-0.784, -0.463] (77); time to event -2.15 [-2.79, -1.63] (106). Forecasts (16 episodes) and in-hospital death (5) are below the base rate too, under the 30-episode floor for scoring | met |
| L2 | Loss with 2, 4, 8, 16 context patients 0.752, 0.719, 0.700, 0.685 (300 episodes); 16 against 2: -0.066 [-0.101, -0.035]. Without context (shown descriptor, 157 episodes) 0.750 | met |
| L3 | The predicted mean against the context's prevalence, within episodes: slope 0.53 [0.47, 0.59] (411 episodes) | met |
| L4 | Hidden descriptor -0.308 [-0.397, -0.231] (497 episodes); anonymized -0.333 [-0.471, -0.231] (256) | met |
| L5 | Real dev with real contexts, pooled -0.376 [-0.484, -0.288] (1,000 episodes), below the base rate on every label type (binary -0.075, ordinal -0.265, values -0.674, time to event -2.85); TrajPFN's loss on real episodes 0.091 below its twin-val loss | met |
| L6 | Twin-train 0.657 against twin-val 0.745: a gap of 0.089 (memorization of the 12,000 training twins, each used about 330 times) | reported |

**Against tabular baselines on identical rows** (reported; `compare_*_ema.json`). The baselines read snapshots of the
same episodes: the last value in the day before t of the program's own channels and core vitals and labs, with every
valid context row as their training set. Medians over episodes; "wins" is the share of episodes where TrajPFN
scores better.

| Set, label | Measure | TrajPFN | Base rate | ExtraTrees | TabICLv2 | TrajPFN wins against ET / TabICLv2 |
|---|---|---|---|---|---|---|
| twin-val, binary (562 episodes) | log loss | **0.168** | 0.277 | 0.260 | 0.327 | 63% / 74% |
| | Brier | **0.035** | 0.062 | 0.062 | 0.095 | 62% / 74% |
| | AUROC (269 with both classes) | **0.681** | 0.5 | 0.619 | 0.622 | 54% / 54% |
| real dev, binary (615) | log loss | **0.172** | 0.287 | 0.267 | 0.348 | 65% / 72% |
| | AUROC (298) | **0.714** | 0.5 | 0.577 | 0.624 | 56% / 54% |
| twin-val, values (77) | MAE | **0.76** | — | 0.99 | 0.93 | 61% / 56% |
| real dev, values (99) | MAE | **0.90** | — | 0.98 | 1.07 | 73% / 67% |

### The M pilot (56.3M, 300,000 episodes, 18,750 steps of 16; about 6.5 hours on the laptop GPU)

- **Settings:** Muon + AdamW at 6e-4, activation checkpointing, otherwise as S; 12-16 episodes/s.
- **Interruption:** the run stopped at step 13,050 when the session that launched it ended. It resumed
  from its step-13,000 checkpoint (the next episode index, optimizer and EMA restored) as a process detached from the
  session.
- **Held-out monitoring (EMA, 400 twin-val episodes):** all label types 0.767 (step 3,000), 0.699 (9,000), 0.673
  (18,750), against the base rate's 0.987.

| # | M | S, for comparison | Status |
|---|---|---|---|
| L1 | binary -0.084 [-0.099, -0.069]; ordinal -0.276; values -0.651; time to event -2.15 (forecasts -0.44, in-hospital death -0.15, both under the scoring floor) | -0.080; -0.275; -0.616; -2.15 | met |
| L2 | 2, 4, 8, 16 context patients: 0.744, 0.705, 0.687, 0.675; 16 against 2 -0.069 [-0.108, -0.035] | 0.752 ... 0.685 | met |
| L3 | slope 0.52 [0.46, 0.58] (411 episodes) | 0.53 | met |
| L4 | hidden descriptor -0.311; anonymized -0.338 | -0.308; -0.333 | met |
| L5 | real dev pooled -0.383 [-0.490, -0.294]; every label type below the base rate (in-hospital death, 14 episodes, at -0.099 [-0.210, +0.006]) | -0.376 | met |
| L6 | twin-train 0.644 against twin-val 0.740: gap 0.097 | 0.089 | reported |

**Against the tabular baselines (M):**

| Set, label | Measure | TrajPFN | ExtraTrees | TabICLv2 |
|---|---|---|---|---|
| twin-val, binary | log loss | 0.166 | 0.260 | 0.327 |
| | AUROC | 0.681 | 0.619 | 0.622 |
| real dev, binary | log loss | 0.162 | 0.267 | 0.348 |
| | AUROC | 0.705 | 0.577 | 0.624 |
| real dev, values | MAE | 0.82 | 0.98 | 1.07 |

**Reading.** Both sizes learn, and every gated check holds for both. M's gains over S are small (L1 binary -0.084
against -0.080, values -0.651 against -0.616, real-dev log loss 0.162 against 0.172), while its memorization gap is
slightly larger (0.097 against 0.089). With 12,000 training twins each reused several hundred times, a larger model
mostly has more to memorize. This is the expected limit of the development bank: scaling the model pays only with
the production pool (10^5-10^6 twins), which is why the main run waits for it.

### Conclusion of this round

TrajPFN learns in context from other patients' labelled trajectories and the patient's own past:
- it beats the base rate on every label type, and does better with more context;
- it follows the context's base rate (the study lead's principle) and reads tasks from examples when the descriptor is
  hidden or channels are anonymized;
- it transfers to real stays with real contexts;
- on identical rows it beats ExtraTrees and TabICLv2 snapshots.

**Not done this round:**
- TabPFN-3.5 (its weights are gated; local inference only);
- the varlen kernel path and its parity test;
- the bedside cache path;
- stages 2-3;
- the generic-share sweep;
- the main run on the production pool;
- the held-out test.

TabICLv2 zero-shot is worse than the base rate in log loss here (it ranks well but is miscalibrated on small,
imbalanced episodes, as in Phase F's learnability report). TabPFN-3.5 is added when its weights are in place (the
study lead's license acceptance).
