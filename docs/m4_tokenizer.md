# M4: medication vocabulary and the S5 tokenizer

Scope review (2026-10-04), decisions and acceptance criteria. Results are added under *Results* as the work
is validated. Subsets only; no full-scale or server run (M9), no model training (M5).

## Scope

From the plan (milestone M4, stage S5, *Structured-output guarantee*):
- **Tokenizer and grammar-spec compiler:** one spec (`configs/grammar/spec.yaml`) from which the vocabulary,
  encoder, decoder, FSM transition tables and an independent parser are compiled; the goal block.
- **Tests that must hold:** *no over-constraint* (the FSM accepts 100% of real encoded sequences, every
  dataset and split) and *round trip* (MEDS -> tokens -> MEDS reproduces the canonicalized input: values to
  their bin, times to their interval).

Decisions the M3 record deferred to M4:
- CLIF 3.0 medication vocabulary gaps (continuous electrolytes, clonidine and other infusions, common ward
  drugs), decided once for all datasets;
- presence-only infusions (NWICU, INSPIRE ward records) in the grammar;
- the blood-pressure site, which ELF codes drop;
- unit-bearing medication codes;
- eICU respiratory settings without a device;
- fluids as rate streams (where pumps are charted) or volumes (where intake is charted).

Constraints set for M4: standard CLIF terms stay distinct from local extensions; medication routes, units and
administration types are preserved.

Not in M4 (later milestones): model and training (M5); sampling engines, vLLM processors, sampler parity,
bounded model checking, differential tests against XGrammar, the distortion monitor (M6); evaluation (M7);
the ICL factory (M8); full-scale runs (M9).

## Findings of the review

1. **Medication records dropped for lack of a category for their administration type**: raw subset records
   whose name the shared lexicon resolves to an ingredient (non-systemic forms and line flushes excluded:
   they are dropped for their route), with no category for the record's administration type (orders: for
   either type). Names the lexicon does not resolve are not counted. M3 lexicon against the M4 lexicon with
   the registry (D1). The eICU and MIMIC infusion figures are the lexicon's view; their adapters map infusion
   items with their own maps.

   | Dataset | Records | M3: without a category | M4 |
   |---|---|---|---|
   | MIMIC eMAR (intermittent) | 616,507 | 270,939 | 19,223 |
   | MIMIC `inputevents` rates (continuous) | 129,293 | 6,860 | 479 |
   | NWICU (intermittent) | 136,840 | 63,503 | 4,545 |
   | INSPIRE ward (intermittent) | 202,943 | 63,178 | 2,751 |
   | HiRID intermittent / continuous | 44,380 / 330,586 | 10,283 / 21,954 | 751 / 1,363 |
   | SICdb intermittent / continuous | 70,456 / 59,487 | 22,864 / 6,129 | 2,638 / 1,458 |
   | AmsterdamUMCdb intermittent / continuous | 60,595 / 42,220 | 16,493 / 4,096 | 1,145 / 137 |
   | eICU infusions / orders | 87,528 / 146,385 | 666 / 22,189 | 174 / 1,059 |

   The most frequent ingredients have no CLIF 3.0 category in any medication table (metoprolol, aspirin,
   tramadol, calcium, amlodipine, clonidine, levetiracetam: all 7 datasets; famotidine, gabapentin,
   atorvastatin, ondansetron, enoxaparin, ...), or CLIF 3.0 lists them for one administration type only
   (potassium chloride infusions: 15,502 records in 5 datasets; potassium phosphate, hydrocortisone,
   atracurium infusions).
2. **Routes are lost in MEDS:** ELF codes are `MED_INT|MED_CON//<category>//<unit>//<action>`; CLIF_MEDS's
   config names `med_route_category` as an extra column but does not write it, and our build reads only the
   event columns. CLIF 3.0 `medication_orders` has no route category column at all.
3. **The blood-pressure site is lost in MEDS:** canonical vitals keep it (`meas_site_name`: arterial / cuff,
   about a third arterial), ELF codes are `VITAL//sbp|dbp|map`.
4. **Categorical assessment values are lost in MEDS (defect):** CLIF_MEDS's PA config maps `numerical_value`
   and `text_value` but not `categorical_value`, so categorical results (CAM-ICU: 11,762 negative, 3,489
   positive subset results; SBT pass/fail) become value-less events.
5. **Derivable fields:** GCS total is charted in MEDS (109,112 subset events, 93% with components at the same
   time); P/F, NEE, SOFA, BMI and anion gap are not MEDS events.
6. **Free text:** text values are source wording (device and position names, raw lab strings, location names)
   whose CLIF category is in the code; numeric lab results carry their raw string as text.

Found while implementing and validating (all fixed):

7. **Doses in counting units were handled three ways.** MIMIC, HiRID and eICU keep the count with its unit.
   NWICU dropped every dose whose prescription unit is not an amount: 9,587 subset administrations, so it had
   no inhaler doses at all. AmsterdamUMCdb kept the administration but not its count. Unit spellings also
   varied ('tablet'/'tab', 'ea'/'each').
8. **The cohort was not adult-only.** The plan's rule is "all adult ICU stays", but 16 subset stays of patients
   under 18 were MEDS subjects (eICU 12, SICdb 4). The tokenizer's age bands, which started at 18, exposed it.
9. **Era is released per stay by five of seven sources** (MIMIC and NWICU anchor year group, AmsterdamUMCdb
   year group, eICU discharge year, SICdb admission year), but no CLIF table carried it. The plan's prefix
   lists it.

## Decisions

### D1 Medication vocabulary

- **Kinds of category** (unchanged, enforced by strict harmonization):
  - *standard*: CLIF 3.0 permissible values of the table;
  - *consortium practice*: unprefixed values the official CLIF ETLs emit although the 3.0 list lacks them;
  - *local*: an existing CLIF 3.0 category used in an administration table that does not list it
    (`origin: local`; e.g. esomeprazole as an intermittent dose, potassium repletion as an infusion);
  - *extension concept*: `ext_<ingredient>`, a drug CLIF 3.0 has no category for in any medication table.
- **Selection rule** (on the raw subset records the shared lexicon resolves): an ingredient without a category
  for an administration type is added when (a) it has at least 300 records of that type across at least 2
  datasets (about 15,000 at full scale), or (b) for infusions, at least 100 infusion records. Not eligible:
  forms without a CLIF route (topical, ophthalmic, nasal, rectal, transdermal), line flushes and locks,
  antiseptics, regional anaesthesia mixtures, and groups that are not one ingredient (vitamins, probiotics).
  Everything below the rule stays dropped and counted.
- **One registry** (`src/icutwin/clif/medication_extensions.csv`): every medication category outside the CLIF
  3.0 list of its table, one row per (table, category): kind, lexicon ingredients, medication group, routes,
  units, CLIF 3.0 status, source, reason. The M3 declarations (consortium practice, the three local
  intermittent terms, `ext_albumin_20`, `ext_flucloxacillin`) moved there from `extensions.yaml`.
  `drugs.csv` names CLIF 3.0 terms only; the registry gives the extension categories to its ingredients, so a
  standard term can never be shadowed (tests). `medication_orders` accepts the union of both tables' additions.
- **Kinds are named consistently:** an ingredient with a CLIF 3.0 term in the other administration table uses
  that term (local), so one drug has one category name in both tables (`ext_alteplase` of M3 becomes
  `alteplase`, local continuous); `ext_` only where CLIF 3.0 has no term for the drug at all.
- **Administration types stay the source's:** categories only decide whether a record is kept, never its type,
  except for INSPIRE ward records, which state no type: there a local intermittent extension of a drug CLIF
  3.0 gives as an infusion (`untyped_iv`) types oral records only, so IV records stay infusions as in M3.
- **Routes:**
  - `inhaled` (a CLIF 3.0 continuous route) is a declared local extension of the intermittent route
    vocabulary, so nebulized and inhaled doses are kept (MIMIC IH/NEB/INHALATION, NWICU 'Inhalation', HiRID
    'inhal', AmsterdamUMCdb order category 70).
  - Routes are kept as the source states them. AmsterdamUMCdb's 'Injecties' order categories and INSPIRE's
    'iv' cover subcutaneous injections too, so subcutaneous drugs appear as `iv` there. The one rule that sets a route is NWICU's 'Injection': subcutaneous for insulin and
    prophylactic heparin, intravenous otherwise.
  - **Provenance (#78):** each route category says what it rests on, as `med_route_basis` in CLIF and
    `route_basis` on MEDS medication events (`clif/route_basis.csv`):
    - `reported`: a route the source names;
    - `reported_class`: a route class or order category;
    - `inferred`: the NWICU rule;
    - `assumed`: an infusion without a source route.

    `verified` is reserved: no source supplies a clinically verified route, so none is presented as one.
- **Units:**
  - A dose keeps its amount, or its count, in the unit the source states.
  - Counting units have one spelling across sources (`tab`, `cap`, `puff`, `neb`, `inh`, `drop`, `sachet`,
    `packet`, `vial`, `each`; `clif/units.py`); weight-based amounts keep their unit (`mg/kg`).
  - In S4, CLIF_MEDS's processors convert the units clifpy knows; the others stay as recorded.
  - NWICU orders without any dose unit stay excluded: they state no administration type (their products
    include IV fluid bags, drips and PCA syringes as well as bolus vials).
- Every added category is declared with its kind, the reason, its CLIF 3.0 status, its source (lexicon
  pattern) and its medication group. MEDS code descriptions and the token vocabulary mark every non-standard
  category with its kind.
- **Routes and units columns of the registry:** the observed profile. These are the CLIF route categories and
  canonical units of the category's subset records, each at least 1% of them, most frequent first. A category
  without subset records keeps the values its source states: argatroban and eptifibatide as intermittent doses
  (boluses whose subset records are charted without a dose unit or not at all).

### D2 Qualified MEDS codes (S4): route, blood-pressure site, categorical value

ELF codes stay the parent; one extension level is appended in our build (CLIF_MEDS's own configs, run by our
ELF engine, which reproduces CLIF_MEDS exactly; the parity check keeps comparing the unqualified codes):
- `MED_INT|MED_CON//<category>//<unit>//<action>//<route category>` (`UNK` where the source has no route);
- `MED_ORD//<category>//<unit>//start|stop//<route category>`, after `medication_orders` gains a declared
  extension column `med_route_category` (CLIF 3.0 admin tables have it, orders do not);
- `VITAL//sbp|dbp|map//<site>` (arterial, cuff; other vitals unchanged);
- `PA//<category>//<categorical value>` for categorical assessments (value normalized like CLIF categories).
`metadata/codes.parquet` gives each qualified code its ELF code as `parent_codes`, so ELF-only consumers can
collapse it.

Three build rules make each infusion code one causal stream, which the grammar's infusion state and prospective
use need (`docs/meds.md`, *Reference path and training MEDS*):
- **Causal units (#76):** a dose is converted only with a weight charted at or before it, and an infusion keeps
  the unit decided at its start until it stops. A weight charted mid-infusion changes nothing.
- **One stream per code (#75):** when a hospitalization's drug has several canonical streams (charted in two
  units) or several codes, its streams are summed again per code, the rule `canonical_continuous` applies to
  concurrent orders.
- **Window-start state (#77):** an infusion running at the stay's window start is a `going` event there, written
  from the events before the window.

Parity with CLIF_MEDS is computed on the reference path, without these rules, and is exact. The training MEDS
deliberately differs in the counted ways (`docs/meds.md`).

### D3 Tokens

- **Fused event tokens** (the plan's default, the tokenization benchmark's best): one token per event,
  `<code>` for value-less codes and `<code>|q<k>` for numeric codes, `k` the equal-frequency bin of the value
  (deciles per code, fitted on the train split, pooled over datasets; codes with fewer distinct values get
  one bin per value). Units, routes, actions and sites are in the code, so they stay in the token. A numeric
  code without a value (presence-only infusions, untestable GCS verbal) is `<code>|na`; a lab result that is
  not a number is `<code>|text`.
- **Time:**
  - Events less than 1 minute after the first event of their time group join that group. No interval token
    falls inside a group.
  - Interval tokens sit between groups: 1-5m, 5-15m, 15-30m, 30-60m, 1-2h, 2-4h, 4-8h, 8-12h, 12-24h, 1-3d,
    3-7d, and 7d+, which repeats once per whole 7 days before the token of the remainder.
  - Within a group, events are in canonical order: domain rank (the build's table order), then code, then
    value bin. One infusion's actions follow the spec's action order (start, going, dose change, verify,
    other, stop), not the alphabet.
  - Token ids are numbered in this order, so a group is in canonical order exactly when its events' first
    token ids do not decrease.
- **Rare codes:** a code seen in fewer than 10 train subjects becomes its domain's value-less
  `<DOMAIN>//OOV` token (`END//OOV` in the end block). Diagnosis and procedure sub-tokens below the threshold
  truncate the code.
- **Hierarchical codes** (ICD-9/10 diagnoses and procedures): a system token and 1-3 sub-tokens (3 characters,
  the next character, the rest), as in CoMET. Other code systems stay whole.
- **Free text** is not tokenized.
- **Derivable fields** are not generated: GCS total is dropped where its three components are charted at
  the same time, and kept where they are not (it is then the only GCS information).
- **Sequence layout:**
  - `[BOS]`, then the prefix, in fixed order:
    - dataset;
    - site (eICU hospitals with at least 10 train subjects, the others `other`; the dataset itself elsewhere);
    - sex;
    - age at hospital admission, in the plan's 5-year buckets from 18 (a 0-18 bucket keeps the tokenizer total
      without the adult cohort rule);
    - era (the admission period as the source releases it; `unknown` for HiRID and INSPIRE);
    - race;
    - ethnicity.
  - The header, each field once, in any order:
    - ICU unit type, admission source, surgical status and admission diagnosis group;
    - hospital admission type;
    - prior ICU stays (0, 1, 2+);
    - hospital time before the ICU (0-1 h, 1-24 h, 1-3 d, 3-7 d, 7 d+);
    - the ICU admission hour, where clocks are real (the plan's ADM_HOUR).
  - The optional goal block.
  - `[TL]` and the timeline: from the window start (48 h before ICU admission) to ICU discharge. The ICU
    admission is the marker token `ICU//admission` at its time, and the ICU outcome is the event
    `ICU//discharge//<outcome>` at ICU discharge.
  - `[END]` and the end block: hospital discharge category, death, then billing diagnoses in code order.
  - `[EOS]`.
  - A field the stay does not record is `unknown`.
- **Goal block** (optional, masked unless requested), delimited `[GOAL] ... [/GOAL]` after the header, fields
  in any order and any subset:
  - ICU outcome;
  - ICU length of stay (0-1, 1-2, 2-4, 4-7, 7-14, 14+ days);
  - hospital outcome;
  - ever IMV, vasopressor, RRT or MCS during the ICU stay (derived episodes; MCS events);
  - maximum KDIGO stage in the ICU stay;
  - first-day SOFA (bands of 2 points, 12+).
- **Long stays:** windows of at most 8,192 tokens. Each window keeps the prefix and header (and goal) pinned,
  then `[CONT]` and a state summary between `[STATE]` and `[/STATE]`. The summary holds the running infusions
  with their current rate bin (or `|na` for presence-only infusions), the last respiratory device and mode,
  and CRRT or MCS charted within 24 h. A window starts at a time-group boundary with the interval from the
  previous group. Only the last window holds the end block.
- An infusion running at the window start is a `going` event of the timeline's first group (S4, #77). No state
  is ever inferred from a later event, so a prediction input never names an infusion known only after its
  cutoff.
- **Prospective inputs** (`encode.prospective`): the stay as known at a cutoff after ICU admission. That is the
  timeline up to the cutoff (no goal, no end block) with every header field whose documentation window
  (`available_within_hours`, #79) ends after the cutoff set to `unknown`.

### D4 Grammar

An automaton with registers (`tokenization/fsm.py`) and an independent recursive-descent parser
(`tokenization/parser.py`) check the same rules with separate code. Both read their token classes from the
compiled vocabulary and their parameters from the spec. Enforced rules (`grammar.enforce`):
- **block_order:** `[BOS]`, the prefix fields in spec order, the header, an optional goal block, an optional
  `[CONT]`, an optional state block, `[TL]`, the timeline, `[END]`, the end block, `[EOS]`. A window that is
  not the stay's last ends inside the timeline, after an event.
- **field_once:** each header and goal field at most once; every header field before the timeline. In the end
  block, the hospital outcome and death at most once each, before the diagnoses, which follow in code order.
- **interval_between_groups:** an interval never opens a timeline (except a continuation window's) and never
  ends one. After an interval other than 7d+ comes an event.
- **group_order:** within a time group, events follow the canonical order (first token ids do not decrease).
  Hierarchical events are a system token followed by sub-tokens of levels 1, 2, 3 in order.
- **infusion_state:** a dose change or a stop (the spec's `continues` and `ends` actions) of an infusion key
  (drug, unit, route) only while the key runs. A start, a presence event (`going`) or a state token makes it
  run, and a stop ends it. `verify` and `other` are allowed either way.

A rule is enforced only if every real sequence satisfies it (no over-constraint). Measured, not enforced:
respiratory settings charted before any device of the stay (`respiratory_settings_without_device`). No rule
couples fluid rates and volumes. Generation-time layers belong to the sampling engine (M6): dataset and site
support masks, value coupling, goal obligations, budget and liveness.

### D5 Fitting, dequantization and storage

- **Fitted on the train split only:** the vocabulary, the value bins (equal-frequency deciles per code over
  the train values of all datasets; one bin per value for codes with at most 10 distinct values), the
  infusion state bins and the dequantization sketches. They are written with the spec's sha256 and a format
  version; encoding refuses a vocabulary fitted with another spec.
- **Dequantization (plan S5: from each bin's empirical distribution, per dataset):**
  - Per (code, dataset, bin) and per (interval, dataset), a sketch of the train values inside the bin:
    minimum, quartiles, maximum.
  - The decoder reads a sketch at its median by default (deterministic), or samples it through the inverse of
    its piecewise-linear distribution function.
  - A dataset without values in a bin uses the pooled sketch; a bin without any train value decodes to its
    lower edge.
  - Every sketch lies inside its bin, so a decoded stay, sampled or not, encodes back to the same tokens.
- **Prefix, header and goal bands** decode to their midpoint. An integer band of width one (KDIGO stage, the
  admission hour) decodes to its value, and an open last band to its lower edge.
- **Storage** (`data/tokens/<mode>/`):
  - `vocab.json`: the vocabulary, bins and sketches;
  - `vocabulary.parquet`: per token its class, field, code, bin and category kind;
  - `<dataset>/<split>.parquet`: one row per subject with `ids` (whole stay), `goal_ids` (with the goal
    block) and `windows`;
  - `index.json`: the spec and vocabulary sha256, and per shard its subjects, tokens, windows and sha256.
- **Validation:** `task=validate` writes `data/qa/<mode>/tokenizer_report.{json,md}` and fails on any
  rejection, round-trip, re-encoding, goal, continuity or separation failure. Two checks were added at the M4
  close:
  - **window continuity (A11):** the windows tile the whole sequence's timeline, and each continuation window's
    state names the infusions running and the last respiratory device and mode, replayed on the whole sequence;
  - **separation (A12):** at 6 h and 24 h after ICU admission the prospective input has no goal or end-block
    token and no header field documented later, the automaton accepts it, and it is unchanged when every event,
    outcome and goal value after the cutoff is changed. Extended in M5 (2026-10-05, `docs/m5_training.md`):
    also at 0 h and 1 h, and header fields documented by the cutoff must equal the stay's (subset rerun: 0
    failures).

### Deviations from the plan (S5)

| Plan | Implementation | Reason |
|---|---|---|
| Prefix static fields: sex, 5-year age bucket, era, height/weight buckets | sex, 5-year age bucket, era, race, ethnicity; no height/weight buckets | Weight and height are timeline events (`VITAL//weight_kg`, `height_cm`) with value bins; a prefix bucket would be a second, derivable copy. Race and ethnicity are CLIF patient statics |
| Era | the source's own period label (MIMIC/NWICU 3-year anchor group, AmsterdamUMCdb 7-year group, eICU and SICdb year); `unknown` for HiRID and INSPIRE | The sources release different granularities and two release none. MIMIC's is the patient's anchor year group, so a stay years from the anchor year is labelled by its patient's group |
| ICU and hospital outcome tokens in the end block | the ICU outcome is the timeline event at ICU discharge; the end block holds hospital outcome, death and diagnoses | The ICU outcome has a time, ICU discharge, inside the timeline |
| Goal tokens mixed into the header in random order | a delimited goal block after the header, fields in any order and subset | One block is masked or forced as a unit at generation (M6) and parses unambiguously. The training collator (M5) shuffles header and goal fields; the grammar accepts any order |
| Fused `CODE//Dk` token names | `<code>|q<k>` (and `|na`, `|text`) | `//` separates code levels; `|` keeps the value part apart from the code |
| Interval token ≥ 7 d repeatable | as planned; the remainder below 7 days follows as one token | — |
| Dequantization from the bins' empirical distributions | 5-point sketches per (code, dataset, bin) and per (interval, dataset); median or sampled | A sketch keeps `vocab.json` small; within-bin AR(1) rank persistence stays an S8-gated option |
| Ablations (unfused tokens, 20/30 bins, 5-10-5 bins, fixed header, time-RoPE) | not built | M9 ablation grid; `value_bins` is already a spec parameter |

## Acceptance criteria

| # | Criterion |
|---|---|
| A1 | Medication vocabulary: the rule, the added categories (kind, reason, CLIF 3.0 status, source, routes, units, group) and the dropped-record counts before and after are documented; strict harmonization, the validation gate (0 hard errors), the S3 QA review and the reconciliation gate (0 unexplained) pass after the change |
| A2 | Every MED_INT, MED_CON and MED_ORD event carries category, unit, administration action and route in its code; the share of unknown routes per dataset is reported; BP site and categorical assessment values are in their codes; `parent_codes` gives the ELF code |
| A3 | Standard CLIF categories are distinguishable from consortium-practice, local and `ext_` categories in CLIF tables, MEDS code descriptions and the token vocabulary |
| A4 | One spec compiles the vocabulary, encoder, decoder, FSM tables and parser; no second hand-written token list |
| A5 | No over-constraint: the FSM and the independent parser accept 100% of encoded subset sequences (7 datasets, all splits; with and without goal block; whole stays and windows) |
| A6 | Round trip: decoding reproduces the canonicalized MEDS of every subset stay (codes, value bins, interval sequence, order within a time), and encoding the decoded stay gives the same tokens |
| A7 | Fitted artifacts use the train split only (a test changes held-out values and gets identical artifacts) |
| A8 | Determinism: two runs give identical artifacts and shards |
| A9 | Goal tokens agree with the stay's outcomes and derived concepts for every stay |
| A10 | Reported per dataset: vocabulary size per token class, events without a token (by reason), tokens per stay, windows per stay |
| A11 | Window-state continuity (added at the M4 close): windows tile the timeline; every continuation window's state block equals the replayed state |
| A12 | Input/target separation (added at the M4 close): prospective inputs hold only what is known at their cutoff (D3, D5) |

## Results (2% subsets of all seven datasets, 2026-10-04)

All twelve criteria hold on the subsets. As in M3, this is subset-scale evidence; full-scale acceptance belongs
to M9.

The final runs:
- S2 to QA rebuilt for every dataset after the last S2 change, then MEDS and the tokenizer after the last S4 and
  S5 changes.
- Cohort: 8,349 stays (train 6,654, tuning 461, held-out 1,234), the sum of the seven MEDS cohorts: MIMIC 1,806,
  eICU 4,026, HiRID 699, SICdb 583, NWICU 574, AmsterdamUMCdb 385, INSPIRE 276. (An interim progress message gave
  8,345, an addition slip; no document carried it.)
- On this machine `fit` takes 54 s, `encode` 87 s and `validate` 435 s (11.8 million tokens).

| # | Result |
|---|---|
| A1 | Met. 163 registry categories (continuous: 1 consortium, 6 local, 15 concept; intermittent: 15 consortium, 19 local, 107 concept); 161 have subset records and carry their observed routes and units (argatroban and eptifibatide as intermittent doses have none). Dropped-record counts are in finding 1. Gates after every change: validation gate 0 hard errors in all 7 datasets; S3 QA 285 flags, 29 new and 1 resolved against M3, each reviewed; reconciliation 39 event bridges and 3 rate-stream comparisons with 0 unexplained rows; MEDS parity with CLIF_MEDS on the reference path exact in every domain of every dataset; mimic-code logic parity exact |
| A2 | Met. Every MED_CON, MED_INT and MED_ORD event has the route level and the event column `route_basis`. Unknown routes (`UNK`): SICdb doses 100% (the source states none), eICU orders 7.3%, INSPIRE infusions 7.7% and SICdb infusions 0.3% (regional-anaesthesia catheters: no CLIF route category); 0 elsewhere. Route basis of medication events: MIMIC 78% reported class (MetaVision order categories) and 22% reported (eMAR); AmsterdamUMCdb 57% reported class and 43% assumed (infusions); eICU 78% reported, 16% assumed, 6% none; HiRID 38% reported, 62% assumed; SICdb 52% assumed, 48% none; NWICU 50% reported, 39% assumed, 11% inferred; INSPIRE 53% reported class, 41% reported, 6% assumed. Every blood pressure has its site (arterial share: MIMIC 38%, AmsterdamUMCdb 95%, eICU 17%, HiRID 92%, SICdb 88%, NWICU 35%, INSPIRE 11%). Categorical assessments: MIMIC 22 codes, eICU 2. Every qualified code has its ELF code in `parent_codes` |
| A3 | Met. CLIF tables: the registry and `extensions.yaml` declare every non-standard term with its kind, and strict harmonization rejects undeclared ones. MEDS descriptions mark them. Token vocabulary: event tokens of standard categories 6,081, local 428, consortium practice 317, `ext_` concepts 1,967 |
| A4 | Met. `configs/grammar/spec.yaml` (sha256 `5d42b666…`) defines the token names, bands, header documentation windows, value bins, interval edges, windows, state, infusion actions and grammar rules. The vocabulary, encoder, decoder, automaton tables, parser and prospective inputs compile from it and the fitted vocabulary; no module holds a token list. The hash is in `vocab.json` and `index.json`, and encoding refuses a vocabulary fitted with another spec |
| A5 | Met. The automaton and the independent parser accept all 8,349 stays: whole sequences and all 8,624 windows, with and without the goal block. Before #75 they rejected 8 stays, and before #80 one, all on the infusion-state rule. Measured, not enforced: respiratory settings before any device (eICU 30,577, SICdb 13,178, MIMIC 2,647, HiRID 1,730, NWICU 1,402, INSPIRE 300, AmsterdamUMCdb 73 events) |
| A6 | Met. 0 round-trip failures. Encoding the decoded stay gives the same tokens with values at the bins' medians (0 failures) and with values and gaps sampled from the bins' empirical distributions (0 failures) |
| A7 | Met on real data. `fit` reads the train split only (6,654 stays). Refitting from MEDS copies whose tuning and held-out values were all changed and half their events dropped, or whose tuning and held-out shards were removed, gives a byte-identical `vocab.json` (sha256 `4f717f5c…`). That file holds the vocabulary, value bins, state bins and every dequantization sketch. The unit test does the same on synthetic data |
| A8 | Met. Three fits give a byte-identical `vocab.json`; two encodings give an identical `index.json` (vocabulary and 21 shard sha256) |
| A9 | Met. 0 goal mismatches: 9 fields x 8,349 stays, recomputed from MEDS and the derived files |
| A10 | Met. `data/qa/subset/tokenizer_report.{json,md}` |
| A11 | Met. Window continuity: for every stay the windows tile the timeline, and every continuation window's state block names the running infusions and the last respiratory device and mode, replayed on the whole sequence. Before #81 three MIMIC end-only windows lacked their state |
| A12 | Met. Separation: at 6 h and 24 h after ICU admission, for every stay, the prospective input has no goal or end-block token and no header field documented later. The automaton accepts it, and it is unchanged when every event, outcome and goal value after the cutoff is changed. At the data level: causal dose conversion (#76), window-start state from earlier events (#77), header fields from admission documentation (#79), derived-concept values computed with a later weight flagged; the remaining bounded look-aheads are listed in `docs/meds.md` |

Vocabulary: 12,277 tokens. By class: special 10, prefix 158, header 95, goal 50, interval 12, state 1,066, event
10,072, end 30, sub-token 784. This includes 58 infusion closure codes (#80). The site field has 92 eICU
hospitals, the six other datasets, `other` and `unknown`; the era field has 16 periods and `unknown`.

| Dataset | Stays | Tokens per stay: median (p90, p99, max) | Stays in more than one window (most windows) |
|---|---|---|---|
| MIMIC-IV | 1,806 | 942 (4,038; 13,611; 33,009) | 62 (5) |
| AmsterdamUMCdb | 385 | 861 (10,606; 27,930; 39,227) | 48 (5) |
| eICU | 4,026 | 644 (2,045; 7,231; 20,259) | 32 (3) |
| HiRID | 699 | 690 (3,133; 23,007; 27,998) | 25 (4) |
| SICdb | 583 | 702 (4,467; 18,850; 31,631) | 29 (4) |
| NWICU | 574 | 398 (1,565; 4,927; 7,459) | 0 |
| INSPIRE | 276 | 580 (1,554; 5,721; 12,445) | 2 (2) |

No window is oversized; every stay fits whole windows of 8,192 tokens.

Events without a token of their own:
- **GCS totals with their components:** MIMIC 42,595, eICU 54,635, AmsterdamUMCdb 3,999. The decoder derives
  them.
- **Codes seen in fewer than 10 train subjects** (their domain's OOV token): MIMIC 11,490, AmsterdamUMCdb 5,436,
  eICU 3,637, SICdb 3,377, HiRID 2,641, INSPIRE 1,770, NWICU 791. That is 0.1-0.5% of a dataset's events
  (INSPIRE 0.8%).
- **Truncated diagnosis or procedure detail:** MIMIC 4,448, NWICU 335, eICU 248, INSPIRE 231, SICdb 108.
- **Birth:** where it is an event; the age band carries it.

Scale: about 1,415 tokens per stay, so the full corpus (about 424k stays, or 50 times the subsets) gives roughly
0.6 billion tokens. The plan sized the default model (M) for 1-3 billion; M5 revisits the size
(`docs/m5_training.md`).

**Not covered by M4** (later milestones, unchanged):
- M5: header and goal shuffling, and goal dropout, in the training collator (now implemented, `icutwin.model`).
- M6: generation-time masks (dataset and site support, value coupling, goal obligations, budget, liveness),
  sampler parity, bounded model checking and XGrammar differential tests.
- M9: the ablation grid, and full-scale runs (needed before the M4 results count as full-scale acceptance).
