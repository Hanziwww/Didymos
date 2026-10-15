# M10: the trajectory ICL data factory (Phase F), status 2026-10-07

The factory turns twins into in-context-learning episodes for a bedside foundation model trained from scratch (its
architecture is the next round). Every prediction references the **complete trajectories of other patients**, each
with the task's label timeline, and the **patient's own trajectory** up to the prediction time. This document is the
interface for the architecture round: what the factory emits, how labels are sampled, and what was measured.

Scope (study lead, 2026-10-07): the quality of the tooling. No large pool of twins is produced in this phase; the tooling
runs on the 13,416-twin development bank and the real benchmark, and the millions-of-twins pool is generated later on
the RTX PRO 6000 server.

## Decisions (study lead's Q&A, 2026-10-07)

| # | Topic | Decision |
|---|---|---|
| 1 | Training episodes | Generated on the fly, seeded: episode i is a function of (factory version, store, pool, seed, i) |
| 2 | Derived concepts | Labels only, never input channels |
| 3 | Cadence | Hourly 35%, 6-hourly 25%, event-triggered 15%, landmark 20%, daily 5% |
| 4 | Context selection | Random 50%, outcome-stratified 25%, retrieved by similarity 25% |
| 5 | Context patients | Complete trajectories only |
| 6 | Queries (stage 1) | 1-8 patients x up to 16 prediction times |
| 7 | Twin pool | Millions on the PRO 6000 server, later; this phase builds and tests the supply only |
| 8 | Feeding | Batch segments appended to the store |
| 9-10 | Generation controls | Base 55%, header strata 15%, eICU sites 10%, goal-conditioned enrichment 20% |
| 11 | Task descriptor | Structured fields, shown in 50% of episodes |
| 12 | Alignment knob | alpha = 0.7, eps = 0.15 (the frontier's knee; the training sweep sets it finally) |
| 13 | Sites and realism | One site variant per episode (plain 45, virtual 30, label shift 15, cross-site 10), mild realism in 50%, a difficulty cap |
| 14 | Stage-1 budgets | <= 32 context patients, <= 4,096 events per trajectory, <= 65,536 context events |
| 15 | Zero-context episodes | 5%, only with a shown descriptor or a forecast |
| 16 | Twin split | 90/10 (development bank), 98/2 (production pool), by a hash of the twin id |
| 17 | Multi-task panels | Stage 2 |

General tasks follow the evidence-based policy: generic trajectories are a switchable source, 10-30% when training from
scratch until a training sweep sets the share.

## Architecture

```
twin shards --decode (features: drawn values, AR(1) persistence; labels: bin medians on the same clock)--+
real token shards (dev / test / control) ------------------------------------------------------------- +--> STAY STORE
real MEDS (the quantization report) -------------------------------------------------------------------+    (segments)
generic trajectories (KernelSynth / TSMixup -> observation process) -----------------------------------+
STAY STORE --> LABEL ENGINE (processes, label blocks) --> PROGRAM POOL (attributes, clusters, raking to P*)
           --> EPISODE SAMPLER (recipe per index) --> MATERIALIZER --> stream / dumps / context banks
           --> QUALITY SUITE (factory report)
```

Modules (`src/icutwin/factory/`):

| Module | Role |
|---|---|
| `twins` | a twin's clock and values from generators keyed by its id |
| `channels`, `configs/factory/channel_groups.yaml` | one channel per event code (infusion actions folded into their key), groups, CRRT/MCS state channels |
| `store` | the stay store: events, state intervals, derived label timelines, segments, memory-mapped reader |
| `processes`, `labels` | event processes and label blocks (validity, reasons, time to event, censoring, resolution) |
| `programs`, `configs/factory/catalog.yaml` | canonical label programs, the clinical catalog, evaluation, descriptors |
| `functional` | functional heads: random functions of future-window features |
| `pool`, `configs/factory/targets.yaml` | candidates, attributes, clusters, bounded raking to P* |
| `trajectories` | complete or cut trajectories, availability-aware statics, budgets, anonymization |
| `sites` | site variants, realism transforms, the difficulty control |
| `episodes`, `configs/factory/stages.yaml` | recipes and the materializer |
| `contracts` | the episode contract per task type |
| `stream` | the episode stream, the reference collate, shards, frozen dumps |
| `generic` | generic trajectories |
| `report` | the quality suite |
| `oracle` | the calibration benchmark's rollouts |

## The stay store

Per segment: `stays.parquet` (stay id, dataset, site, split, weight, cell, ICU length of stay, end kind, hospital
death, prefix and header fields, end codes) and arrays with offsets: events (`ev_t` minutes since ICU admission, the
availability time; `ev_c` channel; `ev_v` value; `ev_bin` decile bin; `ev_f` flags: value, text, infusion action),
state intervals (infusions with rates, device and mode, CRRT and MCS active), and derived timelines from the labels'
decode (IMV, vasopressor, RRT and respiratory-support episodes; KDIGO full, creatinine-only and urine-output stages;
norepinephrine equivalents; first-day SOFA components; daily fluid balance; ASE onsets). Built in parallel, identical
for any worker count; checked against the 72-hour cache (grid values and onsets) and against the tokens' own bins.

## Labels

**Programs** are canonical specs holding only label-affecting parameters (the M8 identity defect, design-review
correction 14). Targets: derived episodes, staged series, point events, thresholds at decile edges (exact under both
decodes; clinical cut points inside a bin are flagged variants), AND/OR composites, death and discharge; values,
remaining stay, stage maxima; functional heads; forecasts. Label types: binary within a horizon (or before the ICU
discharge), time to event with competing death, multi-horizon, ordinal, regression, in-hospital death, forecast.

**The label engine** reads event processes at many (stay, time) rows at once. Time to event is the primitive; the
horizon reduction is censoring-aware (an event by the horizon is 1; followed past it, or the stay ended by death or
discharge, 0; administratively censored before it, unknown). Every label carries its validity and reason, its time to
event, cause, censoring time, and its resolution time, which is always after its own time (a query's own history never
shows its target).

**Label sampling** (plan "Label sampling"): maximum-entropy sampling over the label grammar, calibrated to clinical task
attributes. The pool enumerates the catalog, F6 thresholds over every physiological channel x direction x edge x
sustained duration x horizon (and as times to the first crossing), functional heads, forecasts and multi-horizon
labels; measures each on a stratified panel (domain, label, horizon band, prevalence per time and per stay); clusters
label functions with Jaccard >= 0.9 on their positive sets (MinHash-LSH, exact check); and rakes the generated programs
of each family group toward P* = (1 - eps) tempered(P_clin, alpha) + eps uniform with weights within [0.2, 5] x their
base. Catalog programs are anchors with their natural mass. Episodes draw programs by these weights and draw half of
the functional heads fresh.

## Episodes

A recipe fixes the program, site and site variant, realism, cadence, context patients and their policy, queries and
their prediction times, descriptor, anonymization and stage. The family group is drawn by the study lead's mix (clinical
45%, thresholds 25%, values 15%, functional 15%) and retries stay inside it. An episode needs positive and negative
context trajectories (binary), two classes (ordinal) or values (regression); the cascade filter (a rank-AUROC screen,
then ExtraTrees) drops uninformative ones; the difficulty control redraws a perturbation that costs a probe more than
0.10 AUROC.

**Context selection and base rates (phase_f.2, QA #127).** The model reads an outcome's base rate from its context,
so a context's selection is either natural or declared:
- **Descriptor shown** (the bedside regime: the clinician names the outcome): a natural (random or retrieved) context
  of an event task is taken as drawn (`as_drawn: 1.0`). No case is required, and a rare outcome is not switched to an
  outcome-stratified context. A rare outcome's context often shows none, as a site's own past patients would.
- **Descriptor hidden:** the label function must be read from examples, so a natural context needs a positive and a
  negative trajectory, and a rare outcome's natural context is outcome-stratified (`rare_policy`). The hidden
  descriptor itself says so.
- **Outcome-stratified** contexts select at least max(2, N/4) positive stays while their queries stay natural.
- **The task block declares the selection** (`selection`):
  - `policy`;
  - `case_odds`: the odds of a positive stay's selection against another's, relative to the queries' (1 for natural
    contexts and for a tilted site's, whose tilt the queries share; about 17 for 4 of 16 context stays from 10
    positive among 500);
  - `min_cases`: the positive trajectories the selection guarantees.

  A model can then undo a stratified context's enrichment from the declared odds rather than from the twins' own
  prevalences.

## Task formats

Every task shares one skeleton:

```
episode
├── task       descriptor (or None), label type, horizon, bands, site, cadence, stage, anonymized,
│              selection {policy, case_odds, min_cases}
├── channels   global ids (or the episode-local map when anonymized), count
├── context    N complete trajectories + each one's label timeline at every cadence time
└── query      Q own trajectories up to their last prediction time + targets at their times + own labels
```

Trajectory arrays (ragged, with offsets `ev_off`, `iv_off`, `st_off`): `ev_t` int32 minutes since ICU admission,
`ev_c` int32 channel, `ev_v` float32 value (standardized when anonymized), `ev_z` float32 standardized value, `ev_bin`
uint8, `ev_f` uint8 (bit 0 value, bit 1 text, bits 2-4 infusion action, bit 5 thinned); `iv_c`, `iv_s`, `iv_e`, `iv_v`;
`st_k` int16 static key, `st_v`, `st_avail` int32 minutes. Context patients also carry `end_t` and `end_kind`.

Label arrays (offsets `lab_off`; the query's own history under `own_lab_*`): `lab_t`, `lab_y` (NaN where it does not
apply), `lab_valid`, `lab_reason`, `lab_t_event` hours, `lab_cause` (-1 none, 0 event, 1 death, 2 discharge, 3
administrative), `lab_t_censor` hours, `lab_t_resolved` minutes. Multi-horizon labels add `lab_yh` / `lab_mh` [n, H];
forecasts add `lab_yf` / `lab_mf` [n, H, C] (a grid of hourly values, or the next value and its delay). Since
phase_f.3 a forecast's context patients carry them at every cadence time, as every other type does (the row-level
model reads them as its training rows); its queries carry no own forecast labels (their past values are in their
trajectories). These arrays are present, empty and shaped, when no patient has rows. Label values are in channel
units, also in anonymized episodes: a consumer standardizes them per episode (the reference model does, from the
context's labels) so that units do not name the channel.

| Task type | Context carries | Query target | Encoding |
|---|---|---|---|
| Event within a horizon | label timeline | event in (t, t + h] | 0/1 |
| Landmark outcome | one label at the landmark | outcome after it | 0/1 |
| Event-triggered | labels at each trigger | outcome after the trigger | 0/1 or class |
| Time to event | the event process | (t_event, cause, t_censor) | hazard on a time grid |
| Multi-horizon | as time to event | y per horizon with validity | [n, H] |
| Ordinal | class timeline | class | 0..k-1 |
| Regression | value timeline | value | float |
| Forecast | complete trajectories + their values after each cadence time | own values after t | [n, H, C] + mask |
| Functional head | label timeline | as its label type | as its type |

At t_k a query sees events available by t_k, statics available by t_k, and its own labels resolved by t_k.

## The twin supply (S1b; built and smoke-tested, no pool produced in this phase)

`generate.supply.SupplyPlan` splits a generation budget across the study lead's four mechanisms (Q&A 10): base 55% (the
flattened dataset mix, the accepted sampler with round 3's raking), header strata 15% (dataset x sex x age band x unit
type x admission type x diagnosis group, strata with at least 20 training stays filled uniformly within a dataset; a
5-year age token drawn within the band by its natural share), eICU sites 10% (hospitals with at least 200 training
stays, uniformly) and goal enrichment 20% (ICU death, MCS, RRT, AKI stage 3, a first-day SOFA of 10 or more, an equal
share each). Conditioned cells are not raked (raking would undo the conditioning).

Weights: every twin's importance weight is the balance heuristic over the mixture (Veach & Guibas 1995; Owen & Zhou
2000), w(x) = 1 / sum over the cells c that could have drawn x of n_c / (N p(a_c)), from the cells' counts and the
training stays' marginals of the attributes each cell fixes (`tools/factory_store.py marginals`). A goal block follows
the prefix and header, which the model draws as usual, so a goal cell's ratio uses the goal's natural rate in the
twin's stratum. The base component bounds the weights; a base-only plan gives the bank's weights. A test draws twins
as the engine conditions them from a synthetic population and recovers its marginals within 1.2 points; within a
site the episode sampler draws patients by weight, so enrichment adds distinct rare-outcome patients without raising
prevalence.

Production: `generate.supply.produce` writes shards per part (resumable, partitioned across GPUs and engines, exact and
near duplicates and novelty filtered as in the bank; each twin's cell and request kept, its goal block removed);
`factory_store.py supply` appends them as a store segment, `factory_store.py weights` records the pool's weights for
its declared segments (a twin that missed a requested goal on its derived concepts is rejected), and a store opened
with those segments applies them. `tools/server/supply_produce.sh` runs ENGINES parts per GPU over every GPU of the
RTX PRO 6000 server (`DRY=1`: the parts' shards, twins, draws and hours, without the model).

## I/O

`EpisodeStream` (an `IterableDataset`): workers take the indices congruent to their id, so a run is reproducible and
resumable from any index; an index that never yields an episode (`NoEpisode`) is skipped. `collate` is the reference
batch layout. `write_shard` / `read_shard` / `dump`: NPZ shards with a manifest and checksums for validation and
evaluation sets.

## Tools

| Tool | Use |
|---|---|
| `tools/factory_store.py`, `tools/server/factory_stores.sh` | stores: the twin bank, real dev / test / control, real raw |
| `tools/factory_catalog.py` | the catalog's prevalences on stores |
| `tools/factory_pool.py` | the pool and its raking report |
| `tools/factory_bench.py` | throughput, acceptance, contract fuzz, determinism |
| `tools/factory_report.py` | every acceptance measure |
| `tools/factory_learnability.py` | zero-shot TabICLv2, probes, the switchable-context test |
| `tools/server/rollouts_bench.{py,sh}`, `tools/oracle_analysis.py` | the calibration benchmark |
| `tools/factory_profile.py` | where an episode's time and its redraws go |
| `tools/factory_screen.py` | the general-task screen: task meta-features and the real pool's coverage |
| `tools/factory_frontier.py` | the alignment-diversity frontier (alpha x eps) on a saved pool |
| `tools/factory_sets.py` | evaluation sets: episode dumps, context banks, full-trajectory query sets |
| `tools/server/factory_chain.sh` | bench, report, frontier, screen and the final report in one chain |
| `tools/supply_plan.py`, `tools/server/supply_produce.{py,sh}`, `tools/supply_report.py`, `tools/server/supply_smoke.sh` | the twin supply: plan, multi-GPU producer (dry run), the S1b smoke and its report |
| `factory_store.py marginals / supply / weights / index / generic` | training marginals, segment appends, pool weights, key-major copies, generic stores |
| `factory.consumer` | the reference consumer (a tiny random model reading every task type) |

## Results (host, 2026-10-07; aggregates only)

### S0 and S8: the calibration benchmark

1,400 anchors drawn (24 duplicates dropped: QA #123), 200 per dataset stratified by length of stay, half enriched
for an event within 48 hours; 20 rollouts each with MUC32_main, capped at each dataset's 72-hour token quantile
(5,411 s on the RTX 4080). Rollout 0 is each anchor's label, rollouts 1-19 estimate p*.

| Measure (47 binary catalog programs with at least 30 positives) | p10 | median | p90 |
|---|---|---|---|
| Oracle AUROC (p* for the label) | 0.766 | 0.848 | 0.907 |
| Its ceiling (labels redrawn from Bernoulli(p*)) | 0.879 | 0.938 | 0.972 |
| Probe AUROC (ExtraTrees on other twins' snapshots) | 0.624 | 0.702 | 0.866 |
| Probe's correlation with p* | 0.135 | 0.462 | 0.616 |
| Calibration slope of p* on the probe | 0.779 | 1.077 | 1.466 |

The S8 gate (programs with at least 30 positives scored) is met: 47 of 57. Rollout 0 and the mean of rollouts 1-19
agree (median label rate minus mean p*: 0.0001). The raking gap (the bank's own futures, raked to whole-stay margins,
against p* from the unraked model) is |own - p*| 0.009 at the median and 0.031 at p90, larger for treatments (RRT
onset 0.009-0.011 against 0.020-0.030). Weakest oracles: vasopressor onset within 4 hours (0.59) and lactate rise
(0.61). Trajectory labels are learnable to about 0.85 AUROC in the twin world, and tabular probes reach 0.70: the
gap is what a trajectory model can gain.

### S3: the first bench and what it changed

10,000 episodes on the development bank (13,416 twins) with the second pool, 12 workers: 13.65 episodes/s,
acceptance 0.395, 87 contract failures ("an event after the stay's end": QA #124, fixed), determinism 200 of 200,
family groups within 0.4 points of the study lead's mix. Profiling (`tools/factory_profile.py`) found costs that grow with
the pool, which the production pool (millions of twins) would have multiplied:

| Cost | Change |
|---|---|
| labels evaluated over every stay of a site | a cohort of the site's stays by weight (512, raised to expect 16 positive stays, at most 4,096) |
| a channel's events gathered by scattered indices, sorted again in every worker | key-major copies per segment (`evx_*`, `dvx_*`): a contiguous slice |
| composite keys, float copies and running sums recomputed per window | kept on the cached events (`labels.keyed`, `labels.finite`) |
| unbounded evaluator caches | LRU within a byte budget |
| one label evaluation per patient at materialization | one per episode part |
| a site drawn that cannot carry the program | sites from the pool's per-site support |
| a random context of a rare outcome without a case | outcome-stratified instead (recorded) |

After the changes, on the rebuilt pool (S3 gate, 10,000 episodes, 12 workers):

| Gate | Target | First bench | Now |
|---|---|---|---|
| Throughput | >= 150 stage-1 episodes/s per 16-core node | 13.7/s (12 workers) | 323/s (12 workers; 0.02 s per episode at the median, 0.075 s at p95) |
| Acceptance | >= 70% | 39.5% | 70.4% |
| Contract failures | 0 | 87 | 0 |
| Determinism (1 worker against 12) | identical | 200 of 200 | 200 of 200 |
| Family groups | the study lead's mix +-2 points | within 0.4 | 45.2 / 24.6 / 15.3 / 14.9 |
| Anonymized, descriptor shown | 25%, 50% | 25.5%, 51.9% | 25.5%, 49.7% |

Redraws per 10,000 accepted: a site that cannot compose the episode 3,831 (uninformative contexts 3,296: random
1,377, stratified 1,155, retrieved 764; no valid labels 438; too few stays 97), the difficulty cap 266, the cascade
filter 102. The first rare-outcome rule (outcome-stratified contexts whenever a natural one expects under one case)
moved the policy mix to 44% stratified; with one case required of a natural context and the switch below half a
case, a 5,000-episode bench gives random 47.5%, stratified 33.8%, retrieved 18.7% of contextual episodes (target
50 / 25 / 25), acceptance 72.4%, 219 episodes/s on 8 workers beside the S4 report. Retrieved contexts (the patients
most like the query) are the most often without a case, hence below their share.

### S1b: the twin-supply smoke (2,000 twins planned on the RTX 4080; deleted after the report)

Plan: base 1,100 (flattened mix), 300 in 42 header strata (the 6 rarest eligible per dataset), 200 over eICU sites,
400 goal-conditioned (ICU death, MCS, RRT, AKI stage 3, first-day SOFA >= 10); four engines on GPU 0, shards of 200.

| Gate | Result |
|---|---|
| Kept against planned | base 1,054 / 1,100, strata 280 / 300, sites 191 / 200, goals 361 / 400 (novelty drops: base 1.4%, conditioned 7.6%; no duplicates) |
| Strata and sites as requested | 100% and 100% |
| Weights recover natural marginals (+-2 points) | weighted twins against the base twins (the model's natural distribution under the accepted sampler), pooled at the natural dataset mix: largest gap 1.3 points over ICU death, RRT, MCS, IMV, vasopressor, AKI stage 3, SOFA >= 10, sex, age bands; unweighted, the enrichment shows (ICU death 7.3% against 4.2%). Against the training stays the largest gap is 2.9 points (age 18-45: 16.3% against 13.4%), which the base twins share (16.2%): a model gap, not a weighting one |
| Weights | median 0.94, p99 2.36; effective sample share 0.71 |
| Filters as in the bank | exact and near duplicates 0; novelty 1.4% (base) to 7.6% (conditioned; the bank 4.2%) |
| Segment appends byte-reproducible | yes (two builds, 12 and 4 workers) |
| Store append throughput | 1,886 twins and 4.95M events in 76 s on 12 workers: 7,400 twins per core-hour |
| Event goals kept in 100% of goal-conditioned twins | **no**: ICU death 54% (on the tokens and the derived concepts), RRT 90% on the tokens but 36% in the ICU's derived episodes, MCS 35% / 9% |
| Value goals' acceptance | AKI stage 3 21%, first-day SOFA >= 10 12% |

268 twins missed a requested goal and were rejected (the weights read goals on the derived concepts). The misses
are a property of the accepted model, which `tools/server/goal_diag.py` confirmed on 40 MIMIC twins per request:

| Request | Median / p90 tokens | At the 33.8k-token budget | Goal on the tokens | ICU death |
|---|---|---|---|---|
| none | 972 / 2,298 | 0 | - | 2 |
| ICU death | 33,778 / 33,791 | 24 | 16 (all 16 below the budget) | 16 |
| RRT | 3,297 / 14,565 | 1 | 39 | 10 |

MUC32_main saw goal blocks as conditioning in a quarter of its training sequences and barely raises a death's
probability for it; the obligation register then holds the stay open until the budget, where the grammar ends it
censored (the budget wins over the obligation). So event goals (ICU death, RRT, MCS) became **filter cells**: plain
draws of the dataset kept when their tokens carry the event (about quota / rate x 1.25 draws, at most 40 x quota),
whose conditional is the model's natural one (the weight ratio is then 1[x in c] / p(c | dataset)); per kept death
about 25 draws of ~700 tokens against ~56k tokens through conditioning. Value goals stay conditioned and are accepted
on their derived concepts. A model trained with goal tokens as targets (`TransformConfig.goal_targets`, prepared in
round 3) would make conditioning usable again.

**The smoke again with filter cells** (1,000 twins planned; the same tooling otherwise):

| Gate | Result |
|---|---|
| Kept against planned | base 523 / 550, strata 142 / 150, sites 95 / 100, goals 221 / 200 (filter cells keep every draw with the event) |
| Event goals kept | on the tokens 100% (ICU death 39 of 39, RRT 63 of 63, MCS 42 of 42); on the derived concepts ICU death 100%, RRT 89%, MCS 83% (the obligation sets and the derived episodes differ, QA #113; the rest rejected) |
| Value goals' acceptance (conditioned) | AKI stage 3 24%, first-day SOFA >= 10 15% |
| Filter cells' draws | 3,881 conditioned draws, 87% filtered out (about 1 / rate, as planned) |
| Strata and sites as requested | 100% and 100% |
| Weights recover natural marginals | largest gap to the base twins 1.1 points (unweighted: ICU death 10.6% and RRT 14.6% against 4.7% and 4.0%; weighted 5.4% and 5.1%); to the training stays 4.4 points (age 18-45, which the base twins share: 18.1% against 13.4%) |
| Weights | median 1.00, p99 2.90; effective sample share 0.63 |
| Filters, appends | no duplicates, novelty 1.0-1.6%; byte-reproducible; 12,950 twins per core-hour |

### Learnability with existing tools (no training)

300 twin episodes (development bank, rebuilt pool) and 300 real dev episodes (the catalog), each as a tabular snapshot
(context rows to fit, query rows to score: the last value in the day before t of the program's own channels and core
vitals and labs, and hours since admission). Medians over episodes:

| | ExtraTrees AUROC | Zero-shot TabICLv2 AUROC | TabICLv2 ECE |
|---|---|---|---|
| Twin episodes | 0.637 | 0.597 | 0.188 |
| Real dev episodes | 0.607 | 0.581 | 0.203 |

**The switchable-context test** (79 real dev episodes with a twin context of the same program, site, cadence and
size): real queries answered by TabICLv2 with the real context 0.666 median AUROC, with the twin context 0.670; the
median paired difference is 0.042 in favour of the real context, inside the episode-level noise (per-episode AUROCs
on 1-8 queries spread 0.42-0.78 between quartiles). Twin contexts carry the label function to real queries. The
snapshot view is a weak lower bound for a trajectory model (the oracle's ceiling is 0.85; regression R^2 is mostly
negative on these small episodes); it is used here to compare twins with real data, not to estimate accuracy.

### S6b: the pool

The development bank's pool (`data/factory/pools/bank_dev`): 68,686 programs (clinical catalog 42, threshold 48,040,
values 17,604, functional 3,000), built in 270 s on 16 workers. Raked to P* (alpha 0.7, eps 0.15, caps [0.02, 5]),
the distance of each raked group's weights to P* (total variation over the categories an axis applies to):

| Group | Domain | Horizon | Label type | Prevalence |
|---|---|---|---|---|
| Threshold (F6) | 0.000 | 0.000 | 0.013 | 0.001 |
| Values (F7, forecasts) | 0.000 | 0.000 | 0.034 | 0.000 |
| Functional heads | 0.000 | 0.000 | 0.000 | 0.000 |

Stage 1's own raking (without multi-horizon labels and forecasts beyond 24 hours): threshold 0.005 / 0.006 / 0.048 /
0.004, values 0.003 / 0.002 / 0.103 / 0.003, functional 0. The values group's label axis stays short of its target:
its times to event and ordinal labels are fewer than P* asks even after filling. 215 raked cells with target mass
have an effective size under 30, two thirds of them in the functional group (whose heads are also drawn fresh at
episode time, so its label space is not limited to the pool). Getting here took four defects out of the raking
(QA #125; free shares now follow the label targets; events never valid on the panel get no weight) and filling thin
cells (F6 severity bands, F7 values over every horizon band and to the end of the stay, F7 times to clinical events
and their AND/OR composites with threshold crossings, multi-horizon labels).

### The alignment-diversity frontier

The pool raked again on the grid, raked groups weighted by the study lead's mix:

| alpha | eps | KL to P_clin | Effective label functions |
|---|---|---|---|
| 0 | 0, 0.1, 0.2 | 0.118 | 15,461 |
| 0.5 | 0 / 0.1 / 0.2 | 0.032 / 0.039 / 0.047 | 22,105 / 21,509 / 20,787 |
| 0.7 | 0 / 0.1 / 0.2 | 0.011 / 0.017 / 0.025 | 23,470 / 23,179 / 22,686 |
| 1.0 | 0 | 0.000 | 23,683 |
| uniform base | | 0.174 | 41,701 |
| clinical catalog alone | | 0.116 | 76 |

On this pool alignment and diversity do not trade off over alpha: the generated programs' natural mass is closer to
P_clin than to a flattened target, so tempering costs effective label functions. Mixing in eps costs a little of
both; its role, a share for categories without clinical precedent, is not what these two measures see. The knee
(0.7, 0.15) stays the default; the training sweep sets it (Q&A 12).

### S4: the measurement suite (20,000 episodes on the development bank, the real dev split and the control)

After the S4 findings were acted on (acceptance-aware weights: each pool cell's weight divided by its measured
acceptance; families drawn by their raked mass; stage 1's own raking; site factors so episode use follows the bank's
dataset shares; a mass floor for catalog definitions among generated programs; functional heads drawn by P*):

| Gate | Target | Measured |
|---|---|---|
| Throughput, contract | >= 150 / s, 0 failures | 337 episodes/s (12 workers), 0 failures |
| Family groups | the study lead's mix +-2 points | 44.5 / 25.0 / 15.4 / 15.1 |
| Distinct programs | >= 150 | 9,161 |
| No program above 5% of its family | yes | generated families <= 1.3%; clinical families F1 36%, F3 36%, F5 26% (4-8 definitions each: structural) |
| Catalog reached | every program | 93% in 20,000 episodes |
| Label functions | median abs kappa <= 0.2; near-duplicates < 2% | 0.016; 0 (454 sampled programs, 289 effective) |
| Drawn programs against P* (raked groups) | TV <= 0.02 | threshold 0.043 / 0.028 / 0.067 / 0.023; values 0.066 / 0.086 / 0.162 / 0.066; functional 0.138 / 0.116 / 0.017 / 0.093 (domain / horizon / label / prevalence; the first S4 run had values horizon 0.36) |
| Probe AUROC deciles | spanning 0.55-0.95 | 0.27 ... 0.93 (p10 ... p90; per-episode AUROCs on 1-8 query patients are noisy, the lower deciles are mostly noise) |
| Clinical alignment | >= 80% of clinical programs within 1.5x the real-vs-real gap | 86.8% (33 of 38) |
| Strata reached | every stratum with >= 20 twins | 100% (138 strata) |
| Episode use per dataset | the bank's shares +-10% | eICU -4%, MIMIC +11%, HiRID +1%, NWICU -3%, SICdb -5%, AmsterdamUMCdb +7%, INSPIRE -11% |
| Reuse | Gini <= 0.3, max <= 5x mean | 0.324, 4.7x |
| Context Vendi against real contexts | >= 0.9 | median 0.946 (p10 0.83, p90 1.09) |
| Coverage of real trajectories | within the real-vs-real band | coverage 0.942 (control 0.952), density 1.020 (1.011), recall 0.789 (0.859) |
| Difficulty control | virtual sites and realism: median <= 0.03, p90 <= 0.10 | 0.000, 0.046 |
| Cross-site shift | 0.05-0.20 AUROC | not resolvable: the snapshot probe's paired differences have median 0.005 and spread -0.35 ... 0.50 over 86 episodes |
| Anonymized, descriptor shown, realism | 25 +-2%, 50%, 50% | 25.4%, 50.0%, 49.7% |
| Context policies | random 50, stratified 25, retrieved 25 | 46.9 / 34.5 / 18.6 (of contextual episodes; the rare-outcome rule) |
| Thinned context trajectories | <= 5% | 10.7% (the provisional budgets: 4,096 events per trajectory, 65,536 per context) |
| Prediction times by ICU day | spread by the cadences | day 1 47.6%, 2 20.9%, 3 10.7%, 4-7 13.2%, later 7.7% |

### S6: the general-task screen (no training)

Episodes of one task prior where it applies (threshold crossings, value targets and functional heads over each
store's own channels) from three pools: real tuning stays (291 scored episodes), twins (277) and generic trajectories
(381; 5,000 KernelSynth/TSMixup stays over 24 anonymous channels through the observation process). Each episode is
described by meta-features of its tabular snapshot (size, label type, prevalence and entropy, ExtraTrees and linear
probe scores and their difference, the best feature's association and its concentration); coverage is Naeem's on the
standardized meta-features.

| Coverage of the real pool | Precision | Recall | Density | Coverage |
|---|---|---|---|---|
| by twins | 0.986 | 0.942 | 1.069 | 0.966 |
| by generic episodes | 0.310 | 0.516 | 0.170 | 0.261 |
| by both | 0.594 | 0.945 | 0.548 | 0.979 |
| real against real (control) | 0.973 | 0.959 | 1.010 | 0.938 |

Twins cover the real task space as well as one half of the real pool covers the other. Generic episodes mostly lie
elsewhere (precision 0.31) and add 1.4 points of coverage; the 2.1% of real episodes neither covers are the hardest
to predict (probe median 0.16 against 0.35). The screen therefore records no coverage gap for the twins to fill and
sets no share: the generic source stays switchable, its share for training from scratch set by the sweep below.

**The training phase's first experiment (specified here, run then).** Generic share p in {0, 10, 25, 50}% of
episodes, 3 seeds each, at 20-25% of the training budget; scored on the real dev panel (the main criterion: AUROC,
log loss and calibration of the catalog's tasks on real dev episodes with real contexts, and with twin contexts) and
on a generic retention panel (an fev-bench subset: CRPS of short-horizon forecasts). The share with the best real
panel score is adopted unless another share within its seed spread scores better on retention.

### S9: the final report (100,000 episodes; the held-out split used once)

The same measures against the real held-out stays (`real_test`, the first and only use) and the control:

| Measure | Development (S4, dev split) | Final (held-out split) |
|---|---|---|
| Episodes, throughput, contract failures | 20,000, 337/s, 0 | 100,000, 343/s, 0 |
| Distinct programs, catalog reached | 9,161, 93% | 29,013, 100% |
| Drawn programs against P*: threshold | 0.043 / 0.028 / 0.067 / 0.023 | 0.036 / 0.019 / 0.067 / 0.017 |
| values | 0.066 / 0.086 / 0.162 / 0.066 | 0.075 / 0.079 / 0.163 / 0.046 |
| functional | 0.138 / 0.116 / 0.017 / 0.093 | 0.086 / 0.014 / 0.054 / 0.071 (fresh heads drawn by P*) |
| Median abs kappa, near-duplicates | 0.016, 0 | 0.016, 0 |
| Reuse Gini, max over mean | 0.324, 4.7 | 0.310, 4.4 |
| Episode use per dataset against the bank | -11% ... +11% | INSPIRE -11%, MIMIC +9%, AmsterdamUMCdb +9%, the others within 6% |
| Context Vendi against real contexts | 0.946 | 0.953 |
| Coverage of real trajectories (control) | 0.942 (0.952) | 0.905 (0.973) |
| Density (control) | 1.020 (1.011) | 0.952 (0.963) |
| Clinical programs within 1.5x the real-vs-real gap | 86.8% (33 of 38) | **63.2% (24 of 38)** |

The held-out clinical alignment misses its 80% target. Its control and held-out samples agree closely (ICU death
6.07% against 6.09%), so the band is narrow, and the 14 misses are the generator's systematic gaps, already seen in
the calibration benchmark and the bank's validation: fewer deaths (ICU death 4.2% of prediction times against 6.1%,
decompensation 1.9% against 2.8%, in-hospital death 7.1% against 8.9%), more KDIGO AKI (stage 1 within 24 hours 23.2%
against 20.2%, stage 2 14.3% against 12.5%), more hyperglycaemia (4.5% against 2.5%). They are properties of
MUC32_main, not of the factory; the development split's wider band let most of them pass. The coverage of held-out
trajectories (0.905 against the control's 0.973) points the same way.

The reference consumer (`tools/factory_consumer.py`, a 296k-parameter random model on the GPU, stage 2) read 200
production episodes of every task type (median 40,000-72,000 events each) with every output, loss and gradient
finite, after its raw magnitudes were squashed (the first run had 20 non-finite episodes: infusion rates, statics and
label values in channel units overflowed its attention).

### phase_f.2: context selection (QA #127-128; 20,000 episodes, dev split, after the S9 report)

Applying the study lead's principle (prevalence gaps are generator fidelity; the model reads base rates from its context)
showed that contexts could show more cases than their queries without saying so. The sampler now takes a shown
task's natural context as drawn and declares every context's selection ("Episodes"). The same measures, before
(`report_s4e.json`) and after (`report_f2.json`, acceptance per cell re-estimated):

| Measure | phase_f.1 | phase_f.2 |
|---|---|---|
| Real context rate inside the episodes' context-prevalence p10-p90 (35 binary clinical programs) | 30 | **35** |
| ... inside the p10-p90 of contexts taken as drawn (the bedside regime) | not measured | 35 |
| Context prevalence spread (p90 / p10) beyond the twin-to-real gap | 35 | 35 (p10 is 0 for the rarer events) |
| Event-task contexts without a case | none allowed | 15.0% (the rarest outcomes 20-24%, common ones 0-3%) |
| Stratified contexts' declared case odds (p10 / p50 / p90) | not declared | 1.8 / 7.4 / 29 |
| Descriptor shown | 50.4% | 50.4% (drawn once per index; 55.7% when redrawn per retry) |
| Context policies, all contextual episodes (random / stratified / retrieved) | 47 / 34 / 19 | 57 / 22 / 21 |
| ... event tasks at untilted sites, descriptor shown | — | 54 / 22 / 24 |
| ... event tasks at untilted sites, descriptor hidden | — | 35 / 51 / 14 (cases required; rare outcomes stratified) |
| Throughput (report run; bench stage 1; bench stage 2), contract failures | 315/s, 0 | 345/s; 368/s; 188/s; 0, 0, 0 |
| Acceptance (stage 1, stage 2); determinism (1 against 12 workers) | 70.4% | 71.8%, 74.7%; 200 of 200 each |
| Family groups; drawn programs against P*, values group's label axis | 44.5 / 24.9 / 15.4 / 15.1; 0.162 | 44.5 / 24.9 / 15.4 / 15.1; 0.131 |
| Probe AUROC deciles (p10 ... p90) | 0.27 ... 0.93 | 0.28 ... 0.94 (median 0.58 against 0.65: natural contexts hold fewer cases) |
| Context Vendi ratio, coverage of real (control), reuse Gini | 0.946, 0.942 (0.952), 0.327 | 0.947, 0.942 (0.952), 0.320 |

The remaining policy gap is structural. A tilted site's context (label shift, case mix) counts as random, because its
queries share the tilt. Labels without an outcome (values, classes, forecasts) have nothing to stratify on. A hidden
task needs cases.

## Acceptance

| Gate | Status |
|---|---|
| Family mix +-2 points | met (45.1 / 24.8 / 15.1 / 15.0) |
| >= 150 programs; every catalog program reached | met (29,013; 100% in 100,000 episodes) |
| No program above 5% of its family | met for generated families (<= 1.3%); the clinical families hold 4-8 definitions each, so each is 13-36% of its family: structural |
| Distributions against targets (TV <= 0.02 per axis) | the pool meets it (threshold, functional; values but its label axis, 0.034); drawn episodes come within 0.02-0.09 of P* on most axes, the values group's label axis at 0.16 (its times to event and ordinal labels compose less often than its forecasts) |
| Distinct label functions (median abs kappa <= 0.2; near-duplicates < 2%) | met (0.016; 0) |
| Alignment with diversity: cells with target mass at an effective size >= 30 | not met for 215 raked cells (two thirds functional, also drawn fresh); recorded in `pool.json` |
| Frontier reported | met (alignment and diversity do not trade off over alpha on this pool) |
| Difficulty spread (probe AUROC deciles 0.55-0.95) | the upper deciles reach 0.92 (phase_f.2, dev: 0.94); the lower ones sit below 0.5 because per-episode probes on 1-8 query patients are noisy, not because episodes are near chance |
| Clinical prevalence alignment (reported as generator fidelity, not a factory gate; see below) | 86.8% of clinical programs within 1.5x the real-vs-real gap on the dev split, 63.2% on the held-out split (the generator's death, AKI and glucose gaps) |
| Base rates read from context: episodes vary each clinical program's base rate beyond its twin-to-real gap, and cover the rate a real context shows | met for 35 of 35 binary clinical programs (context prevalence p90 / p10 2.8-14x against gaps of 1.0-1.6x). The rate a random real context shows lies inside the episodes' p10-p90 for 35 of 35 since phase_f.2 (30 before), and inside that of contexts taken as drawn for 35 of 35 (dev split) |
| Context selection declared (QA #127) | met since phase_f.2: every context carries its policy, case odds and guaranteed cases (the contract requires the odds); a shown task's natural context is taken as drawn |
| Twin contexts serve real queries (switchable-context test) | met (TabICLv2 0.670 with twin contexts, 0.666 with real ones) |
| Strata reached; reuse Gini <= 0.3, max <= 5x mean | strata met; Gini 0.31 (within 3% of the threshold), max 4.4x |
| Episode use per dataset +-10% | met but INSPIRE (-11%) |
| Context Vendi >= 0.9 of real | met (0.95) |
| Coverage of real trajectories in the real-vs-real band | dev met (0.942 against 0.952); held-out not (0.905 against 0.973) |
| Sites and views: virtual and realism cost (median <= 0.03, p90 <= 0.10) | met (0.000, 0.046) |
| Cross-site shift 0.05-0.20 AUROC | not measurable with the snapshot probe (paired differences median 0.03, spread -0.33 ... 0.50) |
| Anonymized 25 +-2%; descriptor 50%; realism 50% | met (25.0%, 50.2%, 49.7%) |
| Context policies at 50 / 25 / 25 | met where it applies: 54 / 22 / 24 for event tasks at untilted sites with the descriptor shown (phase_f.2, dev). The overall mix is 57 / 22 / 21 (47 / 34 / 19 in phase_f.1): a tilted site's context counts as random, labels without an outcome cannot be stratified, and a hidden task needs cases (stratified 51%) |
| Thinned context trajectories <= 5% | not met at the provisional budgets (10.7%); the architecture round sets the budgets |
| Contracts (0 failures on >= 100,000 episodes); determinism; throughput >= 150/s per 16-core node | met (0 of 100,000; identical with 1 and 12 workers; 343/s on 12 workers). phase_f.2: 0 failures in 35,000 more episodes (stage 1 and 2), 200 of 200 identical per stage, 368/s at stage 1, 188/s at stage 2 |
| Twin supply (S1b) | met with filter cells for event goals (100% on the tokens); value goals' acceptance reported; weights within 1.1 points of the base twins |
| Calibration benchmark (S8: programs with >= 30 positives scored) | met (47) |
| Test suite on the host | met: 830 passed, 65 skipped (the 2% subset's tokens are not on the host; vLLM tests run in its own environment), 0 failed (dd3f80e); 889 passed locally. phase_f.2 (ec62c6d): 833 passed, 65 skipped, 0 failed on the host; 894 passed locally |

### Prevalence mismatches and what the model learns (study lead, 2026-10-07)

A mismatch in clinical proportions between twins and real stays should not affect what the in-context model learns.
The episodes are built so that it does not: in every episode the context and the queries come from one world, so the
base rate the model must use is the one its context shows, and at the bedside the context comes from the deployment
population. What has to transfer is the label function, not the rate. The factory is therefore judged on capability
proxies rather than on matching prevalences:

- **Base rates are read, not memorized.**
  - **Spread.** Label-shift sites (positives enriched 2-10x), outcome-stratified contexts, case-mix tilts and hidden
    descriptors make each program's base rate vary across episodes. On the dev split (`report_s4e.json`, 20,000
    episodes) every one of the 35 binary clinical programs spans a context prevalence range (p90 / p10) of 2.8-14x.
    That is wider than its twin-to-real ratio (1.0-1.6x; ICU death 1.5x, decompensation 1.6x).
  - **The real rate, before phase_f.2.** The rate a random real context shows (every hour of every stay, as a
    context timeline counts them) lay inside the episodes' p10-p90 for 30 of 35. The 5 outside were the rarest
    events, whose real rate sat just under the episodes' p10 (decompensation 1.6% against 1.8%, NIPPV onset
    0.7-1.1% against 1.3-1.6%, hypoglycaemia 1.3% against 1.5%, RRT onset 3.7% against 4.2%), because every natural
    context had to hold one case.
  - **The same check exposed a second gap.** Outcome-stratified contexts enriched cases (median declared odds 7.4)
    while their queries stayed natural, and nothing told the model. The enrichment could only be undone from the
    twins' own prevalences, which is exactly where a twin-to-real gap would leak into what the model learns.
  - **The fix (phase_f.2, QA #127).** A shown task's natural context is now taken as drawn: 15% of event-task
    contexts hold no case, 20-24% for the rarest outcomes. Every context also declares its selection. The real rate
    now lies inside the episodes' p10-p90 for 35 of 35, and inside that of as-drawn contexts (the bedside regime)
    for 35 of 35.
- **The label function transfers.** Real queries answered with a twin context score as with a real one
  (switchable-context test), and twin and real episodes are about equally learnable from snapshots.

The twin-to-real prevalence gaps stay in the report as a measure of the generator's fidelity, for a later model
round.

## Open items

- The generator's gaps (deaths low, KDIGO AKI and hyperglycaemia high, held-out coverage 0.905) are the twin model's;
  the factory measures them as generator fidelity (they do not limit an in-context learner, which reads base rates from
  its context). A model round would address them; meanwhile the importance
  weights could tilt goal-filtered twins to restore death rates at the cost of effective sample size.
- Goal conditioning works for value goals only as conditioning with rejection (acceptance 15-24%); a model trained
  with goal tokens as targets (`TransformConfig.goal_targets`) would make conditioned enrichment efficient.
- The values group's label mix and the remaining thin cells need more time-to-event and ordinal programs that
  compose; the functional group's thin cells matter less (fresh heads).
- The cross-site shift probe needs larger query sets (or the trained model) to resolve 0.05-0.20 AUROC.
- Queries have their own event budget since phase_f.4 (`query_events`, 16,384 at stage 1; QA #129): thinning a
  query up to its last prediction time is not causal, so it is set where it almost never applies.
- Budgets (events per trajectory and per context) are provisional; at 4,096 / 65,536, 10.7% of context trajectories
  are thinned.
- **phase_f.2 was measured on the dev split only.** The held-out final report (phase_f.1) is not rerun, so the
  held-out split keeps its single use.
- **`as_drawn` (1.0) is a knob for the training phase.** A shown task's natural contexts without a case train the
  regime in which the descriptor alone names the outcome. A lower share keeps more cases in view, at the cost of
  base rates that are no longer the site's.

Evaluation samples were deleted after their runs (the S1b smokes' twins and stores, the S0 rollouts and their
decoded store); their aggregates are kept under `runs/supply/` and `runs/factory/` on the host.
