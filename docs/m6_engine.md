# M6: sampling engine (S7), status 2026-10-05

What is built, how it is tested, and what is not built yet. Generation of deliverable synthetic data (the bank)
waits for a candidate that passes the M7 privacy gate (`docs/m7_privacy.md`).

## Constrained decoding

- **Mask** (`tokenization.fsm.allowed`): every token the grammar automaton accepts in its current state, from the
  token classes as boolean vectors (prefix fields, header and goal fields not yet seen, group order, infusion state,
  sub-token levels, intervals, end-block stages). Tested against `Automaton.step` on every token along random walks
  drawn from the mask, and on real encoded stays and windows (`tests/test_fsm_mask.py`).
- **Generation rules** (`generate.constrain.generation_mask`): a new stay is a first window (no `[CONT]` or
  `[STATE]`); conditions (below); the budget rule (the end block once only it fits, and inside that reserve no
  interval after an interval, so `[END]` follows within one event (#106); a sample cut by it is marked).
- **Bounded checks** (`tests/test_liveness.py`, fixture and subset vocabularies): along walks drawn uniformly from
  the generation mask no state before `[EOS]` is dead, and from every visited state an adversary that always
  draws the most delaying token (week intervals first) still reaches `[EOS]` within the reserve.
- **Samplers:** the transformers reference (`GrammarProcessor`, a `LogitsProcessor`) and vLLM (`generate.vllm_grammar`,
  the V2 model runner's processor interface of vLLM 0.31: an automaton per request slot, fed from the step's input
  token, reading the runner's token history once per slot).
- **Evidence:** a random-initialized model, near uniform, samples only stays the automaton and the independent parser
  accept, each within its budget: transformers 64/64 and vLLM 300/300 on the full 25.6k vocabulary
  (`tests/test_constrain.py`, `tests/test_vllm_grammar.py`); unconstrained requests in the same vLLM batch are left
  alone; greedy decoding is identical between vLLM and the transformers reference (sampler parity).

## Dataset support and the header augmentation (from the quality pilot, 2026-10-05)

- **Support mask** (`generate.support`, the plan's dataset/site support layer): a twin conditioned on a dataset
  draws only tokens that dataset's training windows contain (`support.json` beside the vocabulary, built by
  `python -m icutwin.generate.support <tokens root>`); special, interval and goal tokens are always allowed and
  explicit conditions still win. The pilot found ventilator settings in INSPIRE twins that INSPIRE never charts.
  Tested: samples stay in the support, no dead state and completion within the reserve under it
  (`tests/test_support.py`).
- **Header augmentation undone** (`constrain.HeaderAugmentation`): training set each header field to unknown with
  probability 0.05, so the model emits unknown header values about 5 points too often, the pilot discriminator's
  strongest sign; in the header phase each unseen field's unknown token loses 0.05 x the field's probability and the
  field keeps its total (P_real = (P - r) / (1 - r)), at the rate in the model's freeze record.

## Conditions and goals

- `Conditions`: prefix fields (dataset, site), header fields drawn first (the header is trained in any order),
  and a goal block of requested outcomes after the header (conditioning, never generated).
- **Obligation registers** (event-type goals, plan S7): `require` token sets that must occur before the timeline
  closes, `require_end` in the end block, `forbid` tokens never drawn. `engine.obligations` maps goals: IMV,
  vasopressor (CLIF 3.0 vasoactives), RRT and MCS 'yes' require one of their events and 'no' forbids them; an ICU or
  hospital outcome requires its own token and forbids the others. The budget wins near the context's end and marks
  the sample forced (the plan's forced-ness metric). Tested with requirements, forbidden sets and the goal mapping.
- Conditioning only (model-dependent): the ICU length-of-stay band, the maximum KDIGO stage and the SOFA band.
- **Caveat (#110):** training wrote the goal block with all nine fields, never a subset, so a request with a few
  goals conditions the model on a block it never saw; partial goal conditioning waits for a run trained with goal
  fields dropped at random.

## Engine

`generate.engine.TwinEngine` (vLLM's environment, one GPU each): `sample(requests, seeds)` (conditioned stays from
`[BOS]`), `simulate(prefixes, n)` (Monte Carlo rollouts of prospective prefixes, used by the forecasting evaluation),
`decode` (`tokenization.decode`: events with times and values). `generate.vllm_audit` draws the M7 audit samples
(unconstrained, as the protocol states).

## Bank plan and filters (`generate.bank`)

- `GenerationPlan.from_mix`: quotas per dataset from the train stays' shares (natural, flattened p^a with a < 1,
  uniform, or explicit counts; largest remainders), and each twin's importance weight p_natural / p_plan.
- `exact_duplicates` (hash of the token sequence) and `near_duplicates` (MinHash over event 3-grams, banded LSH,
  estimated Jaccard >= 0.9): a later duplicate is dropped.
- `novelty`: a twin closer to its nearest training stay (cosine distance of the M7 TF-IDF representation) than the
  held-out stays' 5th percentile of its dataset is dropped.
- Tested on the subset's real stays (`tests/test_bank.py`): copies of training stays are all dropped, held-out
  stays are kept, one changed token is a near duplicate.

## Verification layer (b): MEDS (`generate.meds_out`)

Decoded stays become MEDS v0.4 rows by inverting the tokenizer's MEDS reader with the spec's code templates
(static prefix codes, the era and header fields at the ICU or hospital admission, the admission hour as the clock
time, derived GCS totals, end-block codes at the last time), and `validate` checks the DataSchema, the order by
subject and time, finite values and the codes against the reference `codes.parquet` and the fitted vocabulary
(tokenizer coarsenings such as `<domain>//OOV` counted apart). Evidence: real held-out stays of all seven datasets
(684 on the subset) go tokens -> decoded stay -> MEDS -> the tokenizer's reader -> the same tokens (the end block
compared as a set: a diagnosis without sub-tokens loses its canonical place), and their tables validate; random
model samples validate (`tests/test_meds_out.py`, `tests/test_constrain.py`).

## Verification layer (d): factory contracts (`factory.contracts`)

Every batch `ICUPrior.get_batch` returns is checked first (`check_batch` raises on any failure): shapes and dtypes,
finite values, 1 <= d <= H <= 100, rows within T, context within 0.1-0.9 of the rows, zero padding, no feature
constant over the context, classes numbered 0..k-1 (k <= 10) present in both the context and the query. Tested on
real batches (no failure) and on each corruption (`tests/test_factory.py`).

## Goal registers and interventions

- **ICU length-of-stay goal** (`engine.los_band`, `constrain.LosBand`): minutes since the ICU admission marker by
  the interval tokens' median gaps; the ICU discharge waits for the marker and the band's lower edge, no interval
  overshoots the upper edge, and the discharge is due before `[END]` (the budget still wins, marked forced).
  Random-model samples not forced all discharge within the band (`tests/test_constrain.py`).
- **Interventions** (`engine.Intervention`, `simulate(..., interventions=)`): do(treatment) writes the treatment's
  events as a time group at the cutoff (checked by the automaton), do(no treatment) withholds tokens from the
  continuation (`treatment_tokens`: vasopressor, IMV, RRT, MCS); interventional within the twin world only.

## Bank producer (`generate.producer`)

Seeded, filtered, resumable shards from a `GenerationPlan` (each twin its own seed; exact duplicates across the
bank, near duplicates and the novelty filter dropped; shards written atomically and the manifest after them; a
ring cap evicts the oldest shards), behind an interlock: `require_gate` refuses unless the M7 report is PASS for
the very weights sampled (`tests/test_producer.py`, with a stand-in sampler).

## Not built yet

Rejection sampling for value-type goals (KDIGO, SOFA); continuation windows for stays longer than the context (the
state summary from the registers; 0.08% of train stays exceed 32,768 tokens, and a sample reaching the budget is
closed by it); the verification layer (c), CLIFpy validation of synthetic CLIF (the MEDS -> CLIF writer exists
since 2026-10-07: `generate.clif_out`, written through `ClifWriter`'s CLIF 3.0 types and permissible values;
docs/m8_factory.md); exhaustive
bounded model checking over the automaton's abstract states and the XGrammar oracle; the engine's own static-cache
sampler.
