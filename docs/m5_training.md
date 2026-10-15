# M5: training code and the bounded local pilot

Scope (plan, milestone M5, stage S6): the training code and the server launch configuration. Locally there are
unit tests and a tiny-model overfit test and, at the user's request (2026-10-04), one bounded pilot. The pilot
validates data loading, loss masking, window-state continuity and the separation of prediction inputs from
target information. It also measures memory, throughput and validation NLL, to reassess the model size. Real
training runs on the server (M9); nothing here is a full-scale run.

## Design (`src/icutwin/model/`, `configs/train.yaml`)

| Part | Implementation | Plan reference |
|---|---|---|
| Model | HF `Qwen3ForCausalLM` from a config, trained from scratch: GQA, RoPE, context 8,192, tied embeddings (`presets.py`) | S6 |
| Presets | S: 6 layers, width 768 (61.3M parameters with the 12,277-token vocabulary, 51.9M without the embedding). M: 12 layers, 768 (113.3M). L: 16 layers, 2048 (about 1.0B). `tiny`, `pilot` and `pilot_wide` are local sanity sizes only | CoMET family |
| Samples | the stored S5 windows (never cut or merged), each with its stay's goal block (`data.py`) | S5 windows |
| Sampling | dataset temperature on tokens: a dataset's expected share of the training tokens is proportional to (its train tokens)^0.5. Windows are drawn whole, so a dataset is drawn with probability proportional to that share over its mean window length. Within a dataset, windows come in shuffled passes (each is seen equally often); each process takes a disjoint shard of every dataset | S6 data mixing |
| Transforms | goal block (fields in random order) after the header in 25% of samples; header fields shuffled; dataset and site `unknown` in 10%; each header field `unknown` with p = 0.05 | S5 any-order header, S6 goal block and DS dropout |
| Loss mask | no target on the first token, on the goal block (conditioning, masked at generation) or on a continuation window's pinned context (prefix, header, `[CONT]`, state block, `[TL]`: the sampler rebuilds it); everything else is a target, including the first window's prefix and header | S5/S6 |
| Loss | the mean over every target token of an optimizer step, across micro-batches and processes (`num_items_in_batch`, as the HF Trainer since its gradient-accumulation fix), not a mean of per-batch means | S6 |
| Batches | padded, length-bucketed (SDPA, any machine); flattened with per-sample position resets for FlashAttention-2 (padding-free packing) on the server. `task=parity` checks that packed windows give the hidden states of the same windows run alone (and that its negative control, packing without the resets, fails) | S6 throughput stack |
| Optimizer | AdamW (0.9 / 0.95, weight decay 0.1 on matrices), linear warmup, cosine decay to 10% of the peak, clipping 1.0, bf16 autocast. The scheduler steps once per optimizer step | CoMET recipe |
| Distribution | HF Accelerate: `configs/accelerate/ddp.yaml` for S and M (one model per GPU), `fsdp2.yaml` (FSDP2, Qwen3 decoder layers) for L. Only the main process writes files | S6 |
| Checkpoints | `save_pretrained` safetensors of the full weights (gathered from the shards under FSDP2; loadable by vLLM) plus optimizer, scheduler and RNG states of every process. The seeded data stream is replayed on resume, so a resumed run continues identically (tested) | S6 |
| Validation | teacher-forced NLL per dataset and per token family (class, events by domain) on the untransformed windows of the tuning split (all, or a fixed random subset of `eval_max_windows`) | S6 |

**Header-field dropout** is an addition to the plan's dataset dropout. A prospective input before a header
field's documentation window carries the field as `unknown` (M4, #79), and without such samples in training
that input would be out of distribution.

**Local attention kernel:** this machine's Windows PyTorch build has no FlashAttention kernel. For a batch
without a padding mask, transformers asks SDPA for native GQA, which then has only its math kernel: 12 GB for
one layer at 8,192 tokens. Locally the KV heads are therefore expanded first, transformers' own path for
masked batches, which keeps the memory-efficient kernel (`train.memory_efficient_sdpa`). The server uses
FlashAttention-2 and is unaffected.

## Pilot (2% subsets, 2026-10-04)

`icutwin.model.cli task=pilot` wrote `runs/train/subset/pilot/pilot_report.json` (gitignored) in about 25
minutes.

**Resources and limits.** These were checked before launch:
- **GPU:** RTX 4090 Laptop, 16.0 GB, of which 14.7 GB was free.
- **Memory and disk:** 20.8 GB of 31.8 GB RAM available; 515 GB free on D:.
- **Software:** PyTorch 2.13.0 (CUDA 13.0).

The pilot caps its GPU memory at 75% of the card, 12.0 GB (`torch.cuda.set_per_process_memory_fraction`). The
largest training run peaked at 8.6 GB allocated. A first attempt at 32,768 tokens per micro-batch ran out of
memory under the cap (fix 1 below).

### Checks

| Check | Result |
|---|---|
| Data loading | Every window of every split loads: train 6,865 windows (211 of them continuation windows), tuning 471 (10), held-out 1,288 (54). That is 11.8 million tokens; the longest window has 8,192. 0 windows exceed the context; 0 token ids fall outside the vocabulary |
| Transforms | 7,413 samples, all accepted by the grammar automaton: 2,000 train windows and every tuning window, with three random transforms each (goal block, header order, dataset and header-field dropout) |
| Loss masking | In all 7,413 samples, the labels equal the mask derived independently from the parser's structure (first token, goal block, a continuation window's pinned context). Target tokens by class: events 79.3%, intervals 17.9%, sub-tokens 1.1%, header 0.6%, prefix 0.5%, end block 0.4%, special 0.2% |
| Input/target separation | 0 goal tokens outside the masked goal block; 0 evaluation inputs that contain a goal token. S5 check A12: 0 failures at 6 h and 24 h in 8,349 stays |
| Window-state continuity | S5 check A11: 0 failures in 8,349 stays. In training, a continuation window's pinned context and state block are context, never targets (checked under Loss masking) |
| Round trip | 0 failures (S5 report) |
| Overfit | Tiny model (2.0M), one batch of 2,003 tokens, 300 steps: loss 9.43 → 0.010 |
| Padding-free packing | `attention_parity` (local SDPA; the server will use FlashAttention-2): 12 tuning windows packed into one row give the hidden states of the windows run alone (largest relative error 8e-7). The negative control, packing without position resets, differs by 1.48 |
| Data parallelism | Two CPU processes on gloo (`tests/ddp_worker.py`): the accumulated gradient equals the gradient of the mean over both processes' target tokens; the scheduler steps once per optimizer step; there is one metrics writer; the replicas stay equal; each process has its own shard and seed |

### Defects found and fixed

These came from the pilot's first attempt and from a review of the training loop with the server run in
mind. Every check and run in this section ran after all twelve fixes.

| # | Defect | Effect | Fix |
|---|---|---|---|
| 1 | 32,768 tokens per micro-batch | Out of memory under the 12 GB cap. The float32 logits over 12,277 tokens take 1.6 GB per copy, and the loss keeps several copies | 16,384 tokens per micro-batch with 2 accumulation steps (the step is still 32,768 tokens) |
| 2 | Temperature sampling was per window | Datasets were drawn with p ∝ tokens^0.5 per window. Windows average 742 tokens (NWICU) to 2,831 (AmsterdamUMCdb), so the token shares were off target: AmsterdamUMCdb 23.6% (intended 13.2%), MIMIC 25.2% (22.5%), SICdb 16.0% (13.4%), eICU 15.5% (24.3%), NWICU 3.6% (7.7%), INSPIRE 3.2% (6.0%) | Draw probability = the dataset's token share / its mean window length (tested on the sampled token shares) |
| 3 | Windows were drawn with replacement | Repeats followed chance (Poisson): at 4 expected passes, some windows are seen 8 or more times | Shuffled passes per dataset, so every window is seen equally often |
| 4 | Loss under accumulation | A mean of per-micro-batch means, so a token's weight depended on the size of its batch | The mean over the step's target tokens (`num_items_in_batch`) across micro-batches and processes. Tested: it equals the one-batch gradient |
| 5 | Every process drew the same data | Every process used the same seeded stream: N GPUs would train on one GPU's data | A seed per process and disjoint shards |
| 6 | Scheduler under several processes | Accelerate steps a prepared scheduler once per process. With 8 GPUs the cosine would complete in an eighth of the run and then climb again: about four full cycles from peak to floor and back | The scheduler steps once per optimizer step and is saved via `register_for_checkpointing` |
| 7 | Every process wrote files | Every process wrote the config, the metrics and the HF checkpoint; `save_pretrained` saved FSDP2 shards | Only the main process writes; full weights are gathered (`get_state_dict`) before `save_pretrained` |
| 8 | Evaluation subset | `eval_max_windows` took the first windows, which means only the first datasets | A fixed random subset |
| 9 | Loading at full scale | Every token became a Python int, in two copies: about 12 GB per process for eICU at full scale (an estimate) | Column-wise views into each shard's token array (the same samples, tested) |
| 10 | Memory-cap device | The cap was set on the current device, not on the process's own device | It is set on the process's device |
| 11 | Throughput figure | It included evaluation and checkpoint time | Training steps only |
| 12 | Server run config | 65,536 tokens × 8 accumulation steps × 8 GPUs = 4.2M tokens per step, for 20,000 steps: 84B tokens, about 180 epochs of the train split | Replaced by the next run's settings (below) |

Before fixes 2 to 4, the two pilot sizes reached 2.612 and 2.759 at the same learning rate. The larger model
lost, which is why the runs below compare each size at its best of a few rates.

### Memory and throughput

The `probe` runs five optimizer steps on one window longer than 7,000 tokens, in bf16 with padded SDPA (the
memory-efficient kernel), under the cap.

| Preset | Parameters (without the embedding) | Peak GB | Tokens/s |
|---|---|---|---|
| tiny | 2.0M (0.4M) | 1.31 | 285,146 |
| pilot | 7.1M (3.9M) | 1.85 | 170,276 |
| pilot_wide | 29.9M (23.6M) | 3.34 | 71,888 |
| S | 61.3M (51.9M) | 4.64 | 44,464 |
| M | 113.3M (103.8M) | 7.95 | 24,048 |

### Short training runs

Every run took 600 optimizer steps of 32,768 padded tokens (16,384 per micro-batch, 2 accumulation steps):
13.6 million tokens. That is 69% of the padded budget, or 1.45 passes over the 9.4-million-token train split
on average. Warmup was 30 steps, then a cosine decay to 10%, with the same seeded data in every run. The table
gives tuning NLL in nats per target token, over 568k target tokens. Throughput varied by up to 13% between
runs of the same model; the GPU ran at 86–88 °C.

| Run | Peak GB | Tokens/s | Step 150 | 300 | 450 | 600 |
|---|---|---|---|---|---|---|
| pilot (7.1M), 1e-3 | 5.35 | 147,361 | 4.000 | 3.084 | 2.736 | **2.606** |
| pilot, 5e-4 | 5.35 | 128,751 | 4.166 | 3.160 | 2.828 | 2.725 |
| pilot_wide (29.9M), 1e-3 | 8.64 | 48,680 | 3.711 | 3.091 | 2.810 | 2.681 |
| pilot_wide, 5e-4 | 8.64 | 48,234 | 3.851 | 3.041 | 2.714 | **2.596** |
| pilot_wide, 2.5e-4 | 8.64 | 47,915 | 4.077 | 3.093 | 2.793 | 2.694 |

Final tuning NLL per dataset, each size at its best rate:

| Run | AmsterdamUMCdb | eICU | HiRID | INSPIRE | MIMIC | NWICU | SICdb |
|---|---|---|---|---|---|---|---|
| pilot, 1e-3 | 2.262 | 2.678 | 2.874 | 2.255 | 2.637 | 2.871 | 2.384 |
| pilot_wide, 5e-4 | 2.287 | 2.659 | 2.903 | 2.180 | 2.630 | 2.838 | 2.338 |

By token family (pilot_wide at 5e-4):
- **Lowest:** hospital diagnoses 0.66, prefix 1.34, hospital outcome 1.36, intervals 1.65, assessments 1.90,
  vitals 2.07, respiratory 2.12.
- **Middle:** labs 3.35, header 3.50, output 4.66, procedures 4.90, input 5.00.
- **Highest:** intermittent doses 6.37, infusions 6.63, orders 7.71, microbiology 7.94.

What the runs show:
- **The pipeline trains.** Tuning NLL falls monotonically in every run, the train loss and the tuning NLL
  agree (2.57 vs 2.60 for the best run, so no overfitting at 1.45 passes), and no loss is non-finite.
- **The learning rate depends on width.** The best rate halves from width 256 to width 512 (1e-3 to 5e-4), as
  expected in the standard parameterization. Each width and batch needs its own rate; the pilot's rates do
  not carry over.
- **Size gains are small on this little data.** At its best rate, the 29.9M model beats the 7.1M model by
  only 0.010 nats (2.596 vs 2.606; 0.016 in the reproduction below, where reruns of one configuration differ
  by up to 0.006) after 13.6M tokens, which is 0.45 and 1.9 tokens per parameter. Parameters beyond the
  data-matched size buy little here. The subset is 50 times smaller than the corpus, so it shows the regime;
  it cannot rank S against M at full scale.
- **The hardest families have many codes and few examples each:** medications at 6.3–7.7 nats, plus
  microbiology, procedures and input/output. They should gain the most from full-scale data.

### Provenance of the reported NLLs (checked 2026-10-05)

All five runs were launched together (21:51, 2026-10-04) after fixes 1 to 11; the memory-cap device (fix 10)
was the last training-loop change before the launch. Later changes do not touch a training step: the two
batchers now end on a finite stream (training's stream is infinite), and the packing check, the smoke tooling,
`train`'s optional smoke instrumentation (off in these runs), the docs and the server config were added.

To confirm it, the two runs the size comparison rests on were rerun on the final code (this commit):

| Run | Data (tokens, target tokens) | Loss at step 1 | Largest difference of a logged step loss | Final tuning NLL, reported → rerun |
|---|---|---|---|---|
| pilot, 1e-3 | identical (13,629,503; 13,583,410) | identical (9.479256) | 1.2% | 2.6064 → 2.6126 |
| pilot_wide, 5e-4 | identical | identical (9.526536) | 0.5% | 2.5958 → 2.5967 |

The data stream and the initial loss are bit-identical. The rest is GPU run-to-run noise (bf16 kernels with
non-deterministic backward passes), up to 0.006 nats in the final NLL; the learning-rate differences above
(0.08 to 0.12 nats) are far larger.

## Model size

- **Token volume.** The full scale has about 0.47 billion train tokens and 0.59 billion over all splits (an
  estimate, the subsets × 50; the server's exposure report gives the actual counts). Per dataset: eICU 161M, MIMIC 139M, SICdb 49M, AmsterdamUMCdb 48M, HiRID 46M, NWICU 16M,
  INSPIRE 10M. The plan sized M for 1–3 billion tokens, so the measured volume falls below that range.
- **What the data allows.**
  - Compute-optimal training uses about 20 tokens per parameter (Chinchilla). CoMET's exponents, about 0.5
    for both parameters and data, imply a roughly constant ratio.
  - Repeated data is worth nearly as much as new data up to about 4 epochs (Muennighoff et al. 2023).
  - So 0.47B unique tokens support at most about 1.9B token passes, or about 95M parameters.
  - With α = 0.5, the small datasets are repeated more than average (per-dataset passes below).
- **The candidates at about 1.0B tokens.**
  - S (61.3M) gets 16 tokens per parameter.
  - M (113.3M) would get 9.
  - L (1.0B) would get 1, which this data cannot justify.
  - So S is the data-matched size and M is the scaling comparison at the same budget.
- **Memory and throughput.**
  - Memory is not a constraint on 80 GB GPUs. S takes 4.6 GB at full context here (8,192 tokens per
    micro-batch; 5.6 GB allocated in the smoke test) and ran out of the 12 GB cap at 16,384. That puts 65,536
    tokens per micro-batch at roughly 40–45 GB (an extrapolation; the server smoke test measures it): within
    an 80 GB GPU, not a 40–48 GB one (use 32,768 tokens and 2 accumulation steps there).
  - Time is not a constraint either. At the probe's 44k tokens/s on this laptop, 1.0B tokens would take about
    6 hours; that run is not done here. On the server it is a short run (to be measured).

## The candidate run

This is a candidate, not an approved run. It is configured in the `train:` section of `configs/train.yaml`;
the budget is set from the server's actual token counts and measurements (Server preflight) and needs the
user's approval before it starts.

| Setting | Value | Basis |
|---|---|---|
| Model | S, 61.3M | The data-matched size (Model size) |
| Distribution | DDP (`ddp.yaml`), 8 GPUs | S fits one GPU; FSDP2 is for L only and is not exercised here |
| Batch | 65,536 tokens per GPU and micro-batch, FlashAttention-2 packing, 1 accumulation step: 0.52M tokens per step (2 accumulation steps on 4 GPUs) | GPT-3 Small's batch at this width |
| Length | 2,000 steps: about 1.0B tokens (at most 1.05B), 2.2 passes over the train split on average | 16 tokens per parameter. Passes per dataset at 1.0B tokens (α = 0.5): eICU 1.5, MIMIC 1.6, SICdb 2.7, AmsterdamUMCdb 2.8, HiRID 2.8, NWICU 4.7, INSPIRE 6.1 |
| Optimizer | AdamW 0.9/0.95, weight decay 0.1, clipping 1.0, bf16; peak rate 6e-4, warmup 100 steps, cosine decay to 6e-5 | CoMET recipe. 6e-4 is GPT-3 Small's rate at width 768 and a 0.5M batch; the pilot's rates are for other widths and a 16× smaller batch |
| Data | α = 0.5 temperature on tokens, shuffled passes, goal block in 25%, header shuffling, dataset and site unknown in 10%, header field unknown with p = 0.05 | Plan S6 |
| Validation | Every 250 steps on the whole tuning split, with NLL per dataset and per token family; a checkpoint every 250 steps | The per-dataset curves drive the decision below |

After the run:
- **If the tuning NLL still falls at the end for every dataset:** train M at the same budget and in the same
  data order (M9's S-vs-M comparison). Keep the size with the lower tuning NLL per dataset, unless its
  train–tuning gap is larger.
- **If INSPIRE's or NWICU's tuning NLL turns up while the large datasets still improve:** cap their passes
  in the M9 ablation (as UniMax does) rather than lowering α for every dataset.
- **Either way:** the checkpoint with the lowest tuning NLL is the one M6 develops the sampler against. It
  must pass the privacy gates (M7) before any bank is generated.

**Not exercised locally:** FlashAttention-2 (there is no Windows build; `task=parity` covers it on the
server), FSDP2 and NCCL (Windows has neither; DDP's logic is tested on gloo), and multi-node runs.

## Prospective inputs (checked 2026-10-05)

A prediction input at a cutoff must not hold information recorded later. Three checks over every subset
stay (8,349; `encode.prospective`):

| Check | Scope | Result |
|---|---|---|
| Header fields | 8 fields × cutoffs 0, 1, 6, 12, 24, 48 and 72 h × 8,349 stays: 467,544 | Each field is `unknown` exactly when its documentation window (spec `available_within_hours`) ends after the cutoff, and as in the whole stay otherwise: 0 violations |
| Derived concepts | 8,349 stays at 6 h and 24 h, each built three times: with its derived concepts, with none, with extreme ones (every episode, KDIGO 3, SOFA 24) | The prospective inputs are identical (0 differences), while the goal block changes in all 8,349 (the control). 20 stays have KDIGO values computed with a later weight (MIMIC 11, eICU 7, HiRID 2); derived concepts reach only the goal block, which no prospective input has |
| Medication weights | 121,495 per-kg medication events | 6,245 come before their hospitalization's first charted weight, and 6,460 lie in infusion segments that began before it (NWICU 6,184 and 6,110). All were charted per kg at the source; 0 were converted with a later weight |

They are now part of the pipeline, so the full-scale run repeats them:
- The tokenizer's separation check (`validate`, A12) runs at 0, 1, 6 and 24 h and checks both directions of
  the header masking (subset rerun: 0 failures in all 8,349 stays).
- `icutwin_meds task=weights` (`meds.weights`) fails when a per-kg value before the first weight, or in a
  segment that began before it, was converted rather than charted per kg (subsets: 0 violations).
- `test_prospective_inputs_ignore_derived_concepts` pins the independence from derived concepts.

## Server preflight

**Status (2026-10-05): run on the AutoDL host the user designated** (results under *Server run* below). The
user confirmed that the host is approved under the data use agreements of all seven datasets.

### Steps

Each step starts only after the previous one has passed; the resource-heavy ones run one at a time.

1. **Resources and storage.** `python -m icutwin.model.cli task=resources` reports the GPUs, NCCL,
   FlashAttention-2 and disk; also check RAM and CPU count. The full ETL needs about 1 TB free:
   - the 31 GB of archives and up to about 350 GB of extracted CSV;
   - about 355 GB of stage outputs (the subsets take 7.1 GB, × 50);
   - DuckDB spill space.
2. **Data.** Run S1–S5 at full scale (`mode=full`):
   - `icutwin_stage task=stage`, then `task=mirror`;
   - `icutwin.clif.cli task=pipeline`, `task=qa`, `task=community`, `task=reconcile`;
   - `icutwin.meds.cli task=parity`, `task=build`, `task=derived`, `task=mimic_code`, `task=weights`;
   - `icutwin.tokenization.cli task=fit`, `task=encode`, `task=validate`.

   Every gate must pass: 0 hard validation errors, reviewed QA flags, 0 unexplained reconciliation rows, exact
   MEDS parity, 0 weight violations, and A5–A12 at full scale.
3. **Corpus and exposure.** `task=exposure mode=full` gives the actual train tokens per dataset and, for the
   configured run and other budgets, each dataset's sampled share and expected passes (`qa/full/exposure.json`).
4. **Kernel and packing.** `task=parity mode=full` with the run's attention (FlashAttention-2, or on the server
   torch's varlen FlashAttention: *Server run*) and preset. The largest per-token relative error
   of the last hidden states, packed against alone, must stay below 0.05 in bf16 (bf16 keeps 8 bits, about
   0.4% per operation, accumulated over the layers; float32: 1e-3). The leaking control must exceed ten times
   the tolerance (locally 1.48).
5. **Smoke test.** `accelerate launch --config_file src/icutwin/configs/accelerate/ddp.yaml -m
   icutwin.model.cli task=smoke mode=full`, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It runs
   50 steps of the configured run, checkpoints at 25 and resumes a second run from there
   (`icutwin.model.smoke`, report `runs/train/full/S/smoke/smoke_report.json`). Its checks and tolerances:

   | Check | Pass when |
   |---|---|
   | Token-normalized gradient | On every GPU, micro-batches of different sizes accumulate, through NCCL, to the gradient of the mean over all GPUs' target tokens: relative error < 1e-3 (float32, no TF32) |
   | Token-normalized loss | The first step's logged loss equals the mean token loss recomputed from the logits: relative error < 1e-3 |
   | Learning rate | Every step's rate is the warmup-cosine schedule's: relative error < 1e-6 |
   | Data partitioning | Every GPU draws from its own shard with its own seed; no window is drawn by two GPUs |
   | Resume | The resumed run draws exactly the same windows; step losses agree within 5e-3 and the final NLL within 2.5e-3 (relative) |
   | Checkpoint | The safetensors checkpoint loads as a plain HF model and gives the trained model's NLL: relative difference < 1e-3 |
   | Memory | Peak reserved memory ≤ 90% of every GPU |
   | Throughput | Tokens per second after the first 5 steps, and the configured run's time at that rate |

6. **Stop.** Report the results and the budget, and wait for approval of the concrete run configuration.

### Validated locally

- **Tests:** the smoke test's logic runs in the tests on two CPU processes over gloo, and its negative control
  (a resume that does not replay the data stream) fails as it should.
- **Smoke on this laptop.** One GPU; preset S; padded SDPA; 8,192 tokens × 4 accumulation steps; capped at
  12 GB. Every check passed:

  | Check | Result |
  |---|---|
  | Gradient | relative error 1.7e-6 |
  | First-step loss | relative error 5.8e-8 |
  | Learning rate | exact |
  | Resume | the same windows; step losses within 4e-5 |
  | Checkpoint NLL | within 1.4e-10 |
  | Memory | 5.6 GB allocated, 7.9 GB reserved |
  | Throughput | 45.7k tokens/s |
- **Not exercised locally:** NCCL, FlashAttention-2 and several GPUs. The server steps 4 and 5 cover them.

## Server run (2026-10-05)

### Host

| Item | Value |
|---|---|
| Host | AutoDL container (designated by the user; approved under the DUAs of all seven datasets) |
| GPUs | 4x NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB each (sm_120); driver 595.71 (CUDA 13.2); NCCL |
| CPU and memory | 208 cores, 1 TB RAM |
| Storage | everything on the 1.4 TB data disk `/root/autodl-tmp`: code, raw archives, data, environments, caches, temporary files, logs and runs (the 30 GB system disk holds nothing of the project) |
| Software | the project's lockfile (uv, Python 3.12, PyTorch 2.13 with CUDA 13.0); community ETL environments locked as locally (`docs/community_etls.md`, Linux host) |
| Access | SSH key only; code arrives as git bundles of the local repository (no remote) |

**Attention kernel.** No flash-attn release builds for sm_120 (Blackwell workstation and server parts), so the
packed path uses PyTorch's own variable-length FlashAttention (`torch.nn.attention.varlen.varlen_attn`, 2.13),
registered as the transformers attention implementation `varlen` (`model/attention.py`). It takes the same
`cu_seq_lens`/`max_length` arguments as flash-attn's varlen API (what `DataCollatorWithFlattening` produces),
causal, with native GQA. `task=parity` and the GPU tests check it against SDPA on the same windows.

### Speed optimization

Measured on the server before the run (synthetic packed windows with the subsets' length distribution; bf16):

| Configuration | Throughput | Step time | Peak memory |
|---|---|---|---|
| S, 1 GPU, padded SDPA, 65,536 tokens per batch | 116k tokens/s | | |
| S, 1 GPU, packed `varlen` | 220k tokens/s | | |
| S, 1 GPU, packed `varlen` + `torch.compile` (inductor, dynamic shapes) | 482k tokens/s | 133 ms | 18 GB |
| M, 1 GPU, padded SDPA / packed `varlen` | 63k / 119k tokens/s | | |
| **S, 4 GPUs DDP, `varlen` + compile, 131,072 tokens per GPU** (the run) | **1.74M tokens/s** (436k per GPU) | 297 ms | 43.4 GB |
| M, 4 GPUs, the same | 0.98M tokens/s (245k per GPU) | 528 ms | 68 GB |

- Packing (no padding tokens, one kernel call per batch) gives 1.9x and compile (fused normalizations, rotary
  embeddings and SwiGLU) another 2.2x: 4.2x per GPU over padded SDPA.
- DDP over four GPUs keeps 90% of the single-GPU rate per GPU at twice the micro-batch.
- Batch construction (sampling, transforms, packing) runs at 14.7M tokens/s on one core, so data loading is
  not on the critical path.
- At 1.74M tokens/s, 1.0B training tokens take about 10 minutes plus evaluations and checkpoints: compute does
  not constrain the budget; the data does (passes over the small datasets).
- The run uses 131,072 tokens per GPU and micro-batch with no accumulation: 0.52M tokens per optimizer step
  (the candidate's batch).

### Data at full scale

All seven datasets were staged, mirrored and run through S2-S5 on the host. The full-scale review found defects
the subsets could not show; the data was rebuilt with the fixes (r1)
and every MEDS step re-run with separate DuckDB spill directories (r2, #93).

| Gate | Result |
|---|---|
| CLIF 3.0 validation (clifpy) | 0 hard errors in all seven datasets |
| S3 QA flags | 343 flags. Reviewed individually: the 116 new at full scale and the 3 the fixes added (AmsterdamUMCdb volumes now given as amounts); the fixes removed 19. The 243 also raised on the subsets carry their subset review and were not re-reviewed at full-scale magnitude |
| MEDS parity (our ELF engine against CLIF_MEDS) | 0 differing events in all seven datasets |
| Weight causality (`meds.weights`) | 0 violations in all seven datasets |
| mimic-code parity | KDIGO 1.0 on every column; ventilation and norepinephrine-equivalent episodes identical |
| Tokenizer (A5-A12, all 409,297 stays) | 0 FSM or parser rejections, 0 round-trip, re-encoding, goal, continuity or separation failures |
| Reconciliation with the community ETLs | **Completed for all seven datasets after training** (2026-10-05). Training started while it was incomplete (MIMIC, NWICU and INSPIRE done; AmsterdamUMCdb being re-run; SICdb, HiRID and eICU not run), and the earlier statement that every discrepancy was explained was wrong. Every unpaired row now has a documented reason except the data defects of the snapshot, both in our adapters and fixed since: #95 (4 INSPIRE records) and #98 (HiRID infusion rates rounded to 6 decimals; 29 HiRID total-dose keys). #96, #97 and the other residuals were reconciliation artefacts (#99-#104). Several reasons attribute rows to a documented mechanism by group (same item and time, same patient) rather than proving each row |

Cohort: 409,297 ICU stays (MIMIC 89,201; eICU 199,120; HiRID 33,896; NWICU 27,680; SICdb 27,179; AmsterdamUMCdb
18,383; INSPIRE 13,838) with 460.8M MEDS events. The full-scale vocabulary has 25,566 tokens (subsets: 12,277),
so S has 71.5M parameters (19.6M of them the tied embedding).

### Corpus and exposure (`qa/full/exposure.json`)

| Dataset | Train windows | Train tokens | Natural share | Sampled share (alpha 0.5) | Passes at 1.034B tokens |
|---|---|---|---|---|---|
| eICU | 160,795 | 168.9M | 37.3% | 25.3% | 1.55 |
| MIMIC | 74,236 | 126.8M | 28.0% | 21.9% | 1.79 |
| AmsterdamUMCdb | 17,704 | 50.5M | 11.1% | 13.8% | 2.83 |
| HiRID | 28,296 | 42.8M | 9.4% | 12.7% | 3.08 |
| SICdb | 22,815 | 37.0M | 8.2% | 11.8% | 3.31 |
| NWICU | 22,274 | 15.7M | 3.5% | 7.7% | 5.09 |
| INSPIRE | 11,269 | 11.5M | 2.5% | 6.6% | 5.94 |
| **All** | 337,389 | **453.2M** | | | 2.28 on average |

**Budget.** B = min(20 tokens per parameter = 1.43B; the budget at which the most-repeated dataset reaches 6
passes = 1.04B; 4 passes over the train split on average = 1.81B) = 1.04B, set by INSPIRE. At the measured
517,182 tokens per step: 2,000 steps, 1.034B tokens, 14.5 tokens per parameter.

**The 6-pass limit is per model, not per experiment.** It bounds how often one training run (one model, trained from scratch) draws a dataset's train windows: S drew INSPIRE's 5.94 times on average and M 5.86 times (NWICU 5.09 and 5.02; `passes` are expected values of the temperature sampling, each window drawn the floor or ceiling of that). The experiment as a whole used INSPIRE's train windows in two runs (about 11.8 draws in all) and compared the two models on the tuning split once. No single model saw any dataset more than 6 times; the memorization and privacy assessment concerns the retained model M alone.

### Kernel, packing and the smoke test

- **Packing parity** (`task=parity`, preset S, `varlen`, bf16): 12 tuning windows packed in one row (8,796
  tokens); largest per-token relative error of the last hidden states against the windows alone 0.0163
  (tolerance 0.05); the leaking control 1.474 (more than ten times the tolerance).
- **Smoke test** (`task=smoke`, the run's configuration on 4 GPUs: 50 steps, checkpoint at 25, resumed run):

  | Check | Result |
  |---|---|
  | Token-normalized gradient (4 processes, NCCL) | relative error 1.1e-6 (tolerance 1e-3) |
  | First-step loss against the recomputed token mean | relative error 6.0e-8 |
  | Learning rate | exact (2e-16) |
  | Data partitioning | 4 disjoint shards (84,345-84,349 windows each), 0 windows drawn by two processes |
  | Resume | the same windows; step losses within 7.1e-6, final NLL within 1.9e-6 |
  | Checkpoint | safetensors loaded as a plain HF model: NLL within 7.4e-7 |
  | Memory | 63.0 GB allocated, 63.8 GB reserved per GPU (67% of 95 GB; the 25,566-token head adds to the benchmark's 43 GB) |
  | Throughput | 1.51M tokens/s (377k per GPU), 517,182 tokens per step: the configured run in about 12 minutes |

### The run (S, started 2026-10-05 09:29 server time)

Approved by the user ("Begin formal training after performing speed optimization on the cloud host"; "start
training as soon as the gates pass"). `tools/train.sh S_main 2000` on the host: the `train:` section of
`configs/train.yaml` unchanged (S, `varlen` packing, compile, 131,072 tokens per GPU and micro-batch, 4 GPUs, no
accumulation, AdamW 6e-4 with 100 warmup steps and cosine decay to 6e-5, alpha 0.5, goal block in 25%,
evaluation on the whole tuning split and a checkpoint every 250 steps), `max_steps` 2,000 from the budget above.
Run directory `runs/train/full/S_main` (config, metrics, checkpoints, `train_report.json`).

**Result (S_main).** 2,000 steps, 1.034B tokens (1.031B targets) in 12.0 minutes of training steps (1.43M tokens/s
including warm-up and compilation) and 14.4 minutes wall clock with 8 evaluations and 9 checkpoints; peak 63.1 GB
per GPU. Tuning NLL per target token (nats), the whole tuning split:

| Step | All | AmsterdamUMCdb | eICU | HiRID | INSPIRE | MIMIC | NWICU | SICdb |
|---|---|---|---|---|---|---|---|---|
| 250 | 2.059 | 1.714 | 2.055 | 2.286 | 1.920 | 2.182 | 2.435 | 1.767 |
| 500 | 1.830 | 1.524 | 1.815 | 2.013 | 1.724 | 1.956 | 2.186 | 1.575 |
| 1000 | 1.691 | 1.377 | 1.687 | 1.851 | 1.585 | 1.812 | 2.018 | 1.447 |
| 1500 | 1.637 | 1.322 | 1.639 | 1.781 | 1.538 | 1.757 | 1.946 | 1.397 |
| 1750 | 1.622 | 1.308 | 1.625 | 1.763 | 1.521 | 1.741 | 1.927 | 1.383 |
| 2000 | 1.614 | 1.301 | 1.618 | 1.755 | 1.515 | 1.733 | 1.919 | 1.377 |

Every dataset still improves over the last 250 steps (-0.005 to -0.009), INSPIRE (5.9 passes) and NWICU (5.1)
included, and the training loss on transformed samples (about 1.57 at the end) is below the tuning NLL by about
0.04: no sign of overfitting. By token family, measurement and outcome tokens are learned best (end-block
hospital diagnoses 0.12, CRRT 0.39, ADT 0.46, labs 1.96), medication events least (infusions 3.52, doses 3.47,
eICU orders 4.60, microbiology 5.71): many drugs, doses and timings per category.

By the decision rule above (the tuning NLL still falls for every dataset), M is trained at the same budget and
data order (`M_main`: 65,536 tokens per GPU and micro-batch with 2 accumulation steps, the same 0.52M tokens per
step, since M with the 25,566-token head would come close to the memory limit at 131,072).

**Result (M_main) and the size decision.** M (123.5M parameters) at the same budget and data order: 2,000 steps,
1.020B tokens (1.4% fewer than S: 65,536-token micro-batches pack with more slack), 20.9 minutes of training
steps at 0.81M tokens/s (25.5 minutes wall clock), peak 45.3 GB allocated per GPU.

| Tuning NLL at step 2000 | All | AmsterdamUMCdb | eICU | HiRID | INSPIRE | MIMIC | NWICU | SICdb |
|---|---|---|---|---|---|---|---|---|
| S (71.5M) | 1.614 | 1.301 | 1.618 | 1.755 | 1.515 | 1.733 | 1.919 | 1.377 |
| M (123.5M) | **1.582** | **1.271** | **1.588** | **1.712** | **1.490** | **1.699** | **1.879** | **1.350** |
| M - S | -0.032 | -0.030 | -0.031 | -0.043 | -0.025 | -0.034 | -0.040 | -0.026 |

- M is ahead from step 500 on, on every dataset, and still improves over its last 250 steps (-0.004 to -0.009).
- Train-tuning gap (mean training loss of steps 1910-2000 against the final tuning NLL): S 0.036, M 0.026.
- By token family M gains most where S is weakest: intermittent doses 3.47 -> 3.28, infusions 3.52 -> 3.37, eICU
  orders 4.60 -> 4.46, in-hospital death 0.646 -> 0.599, labs 1.96 -> 1.92.

**Decision (the rule above):** M has the lower tuning NLL on every dataset and no larger gap, so **M_main
(`runs/train/full/M_main/final`) is the checkpoint M6 develops the sampler against**; it must pass the privacy
gates (M7) before any bank is generated. Neither run shows the small datasets turning up, so no pass cap is
needed at this budget. More data passes are not the lever: the budget is set by INSPIRE's passes, not by compute
(a 1B-token run takes 15-25 minutes on this host). The next scaling step is the M9 ablation grid (UniMax-style
caps on the small datasets so the large ones can take more tokens; an intermediate width).

### Data defects found after training, and retraining (2026-10-05)

The reconciliation, completed after M was trained, found two data defects in the snapshot M was trained on
(#95 and #98 of the project's data-quality log;
#97 was a community merge). Changed token positions: 7 for #95 (two train stays), and 18,223 event tokens for
#98 (6,157 HiRID train stays; at most 25,836 with the interval tokens around missing events), together 4.0e-5
(at most 5.7e-5) of the 453.2M train tokens. The tuning split, used for validation only, has 1,298 changed tokens (356
stays), held-out 3,398 (1,139 stays). **Retraining is not necessary:** M stays the frozen candidate (no further
training, no deliverable synthetic data); the fixes take effect with the next INSPIRE and HiRID rebuild, and a
model trained after it is trained on the corrected snapshot.

## Second run: M32_main (2026-10-05)

The user asked for a retrained M with a larger context, chosen from actual needs (a larger model allowed if
needed), more extensive training and a more balanced number of passes per dataset. M_main stays frozen for
comparison. Three findings shaped the run: the stays the 8k windows cut, M_main's privacy gate (M7: P1 failed on
INSPIRE, drawn 5.9 times; the membership AUC rose with passes), and the data defects #95 and #98 (fixed and
rebuilt: r3).

### Context

Train stays of the r2 snapshot (the `ids` column: a stay's whole sequence), 327,662 stays:

| | Over 8k | Over 16k | Over 32k | Over 64k | p99 | Longest |
|---|---|---|---|---|---|---|
| Stays | 2.11% | 0.53% | 0.08% | < 0.01% | 12,135 | 242,498 |
| Tokens in those stays | 22.2% | 9.5% | 2.4% | 0.3% | | |
| AmsterdamUMCdb: stays / tokens | 11.3% / 59.2% | 4.6% / 36.5% | 1.05% / 13.2% | 0.04% / 1.1% | 33,085 | 133,858 |
| SICdb | 3.4% / 31.0% | 1.1% / 15.3% | 0.14% / 3.5% | 0 / 0.2% | 16,812 | 82,193 |

At 8k, stays holding 22% of the train tokens were cut into windows, and a window sees only its state summary of
what came before; these are the long, sick stays, most of AmsterdamUMCdb's data. **32,768** keeps 99.92% of stays
whole (97.6% of tokens); 64k would add 2.1% of tokens at a quadratic attention cost on the longest sequences and
twice the sampling length. Pinning prefix, header and state in every window costs 0.01-0.15% of tokens, so
windowing itself wasted little; what it lost was history. The window length is the grammar spec's
(`windows.max_tokens`); the model's context follows it (`presets.CONTEXT`, positions + 256 for the goal block),
with Qwen3's RoPE base for long contexts (1e6; M_main kept transformers' 10,000).

**Model size.** M's architecture (12 layers, width 768; 123.5M parameters with the 25.6k vocabulary). A longer
context adds no parameters with RoPE. A larger model is not justified by the data: about 453M unique train
tokens, so M is at or above the compute-optimal size for the budgets below, and M7 found the membership signal
growing with how often the model saw a dataset (a larger model memorizes more per pass).

### Passes

Mixing by target passes per dataset (`model.data.pass_shares`; share = passes x train tokens): 3 passes, 2.5 for
NWICU and INSPIRE (their per-pass AUC increase was the largest; at 3 passes INSPIRE's projected upper bound would
sit at the 0.60 limit). M_main drew the datasets 1.5 (eICU) to 5.9 (INSPIRE) times at alpha 0.5; M32_main draws
the large datasets about twice as often (eICU 1.53 -> 3) and the small ones less (INSPIRE 5.86 -> 2.5), about
1.35B tokens in all (M_main: 1.02B). The step budget is computed from the r3 exposure.

### Result (M32_main)

On the r3 tokens (`data/tokens/full_r3`: 452.95M train tokens, 327,662 stays in 327,936 windows): the packing parity check at 32k
passed (packed against unpacked logits within 0.019 relative), and the 50-step DDP smoke test with checkpoint and
resume passed; the step budget from the r3 exposure was 2,566 steps of 0.52M tokens. The run: 2,566 steps,
1.287B tokens (the packed batches held 95.7% of the step budget, so the realized passes are 2.87, and 2.39 for
NWICU and INSPIRE), 33 minutes of training steps at 0.65M tokens/s (38 minutes wall clock), peak 45.3 GB
allocated per GPU; frozen for the M7 gate at 14:42 server time.

| Tuning NLL, final | All | AmsterdamUMCdb | eICU | HiRID | INSPIRE | MIMIC | NWICU | SICdb |
|---|---|---|---|---|---|---|---|---|
| M_main (r2, 8k windows, step 2000) | 1.582 | 1.271 | 1.588 | 1.712 | 1.490 | 1.699 | 1.879 | 1.350 |
| M32_main (r3, 32k windows, step 2566) | 1.561 | 1.263 | 1.551 | 1.711 | 1.534 | 1.669 | 1.907 | 1.350 |

Not a like-for-like comparison (another token snapshot and vocabulary fit, whole stays instead of 8k windows,
other passes), but the direction is the expected one: the datasets drawn more often (eICU 1.53 -> 2.87 passes,
MIMIC 1.77 -> 2.87) improve, the two drawn less (INSPIRE 5.86 -> 2.39, NWICU 5.02 -> 2.39) give back 0.03-0.04
nats, which is the price of their membership signal (`docs/m7_privacy.md`: INSPIRE's P1 AUC 0.650 -> 0.540).
Tuning NLL fell at every evaluation (2.116 at step 250, 1.685 at 1,000, 1.616 at 1,500, 1.561 at the end).

**M7 gate: PASS** (P1-P4 pass under v1.1; P5 was inconclusive under v1.1, +0.004 [-0.017, 0.025], and passed
under amendment v2, adopted by the study lead before its run: -0.007 [-0.018, 0.005] on 20,000 stays per side;
`docs/m7_privacy.md`). M32_main is the cleared candidate.

## Third run: ML32_main (2026-10-06)

Round 2 of M7 (`docs/m7_authenticity.md`) found M32_main's twins valid, private and useful but distinguishable in
the smaller, densely charted datasets from their first day (under-fitting: tuning NLL still falling at the end,
membership AUCs 0.53-0.56), with passes bound by the privacy gate rather than by compute. The study lead chose the
recommended course: a larger model at the same passes. The candidate, its basis and the pre-registered adoption
rule are in `docs/m7_authenticity.md` ("Third run"); in short: preset `ML` (16 layers, width 1,024; 269.5M
parameters), peak rate 4e-4, goal-field subsets (#110), everything else M32_main's (`configs/train.yaml`), run by
`tools/server/ml32_run.sh` (parity and smoke at 32k, the run, the M7 gate, the authenticity audit).

**Result (ML32_main).** Parity and the 50-step DDP smoke test at 32k passed (peak 78.2 GB allocated of 95 GB per
GPU, 365k tokens/s). The run: 2,566 steps, 1.287B tokens, 61.7 minutes of training steps at 0.35M tokens/s (70
minutes wall clock). Tuning NLL against M32_main at the same steps:

| Step | M32_main | ML32_main |
|---|---|---|
| 250 | 2.116 | 2.150 |
| 500 | 1.840 | 1.856 |
| 1,000 | 1.685 | 1.687 |
| 1,500 | 1.616 | 1.618 |
| 2,000 | 1.578 | 1.580 |
| 2,566 (final) | **1.561** | **1.563** |

Final per dataset (M32_main / ML32_main): AmsterdamUMCdb 1.263 / 1.268, eICU 1.551 / 1.550, HiRID 1.711 / 1.715,
INSPIRE 1.534 / 1.532, MIMIC 1.669 / 1.671, NWICU 1.907 / 1.916, SICdb 1.350 / 1.353. The larger model caught up
by step 1,000 and never pulled ahead; the M7 P1 attack shows where its capacity went: members' NLL 0.007-0.013
lower than under M32_main, non-members' unchanged, membership AUC +0.010 to +0.016 per dataset (still PASS). Unlike
S -> M (same rate, 1.03B tokens: -0.03 nats everywhere), M -> ML at these privacy-bound passes buys memorization,
not fit to unseen stays; whether a higher peak rate would change that is untested. Not adopted (the pre-registered
rule, `docs/m7_authenticity.md`); M32_main stays the candidate.
