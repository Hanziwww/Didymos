# Materials: tokenization and the grammar

An ICU stay becomes one token sequence (`src/icutwin/tokenization`, spec `src/icutwin/configs/grammar/spec.yaml`,
vocabulary `data/tokens/full_r4/vocab.json`). Decisions and their reasons: `docs/m4_tokenizer.md` (D1-D5);
the ICU-day clock blocks: `docs/m7_refinement.md` amendment 1.

## Vocabulary (r4, fitted on the train split only)

| Class | Tokens | Content |
|---|---|---|
| Event | 19,774 | 4,434 event codes: 12,950 value tokens (`<code>\|q<k>`, equal-frequency deciles per code over all datasets' train values; codes with at most 10 distinct values one bin per value), 2,390 value-less (`\|na`), 2,390 non-numeric results (`\|text`), 2,011 flags, 29 domain OOV tokens (codes seen in fewer than 10 train subjects), 4 code-system tokens |
| Diagnosis and procedure sub-tokens | 3,057 | ICD-9/10 codes as a system token and 1-3 sub-tokens (3 characters, the next, the rest), as CoMET |
| State | 1,792 | the state summary of a continuation window: running infusions with their rate bin, the last respiratory device and mode, CRRT or MCS within 24 h |
| End block | 500 | hospital discharge category, death, billing diagnoses |
| Prefix | 270 | dataset, site (eICU hospitals with at least 10 train subjects), sex, 5-year age band, era, race, ethnicity |
| Header | 96 | ICU unit type, admission source, surgical status, admission diagnosis group, hospital admission type, prior ICU stays (0, 1, 2+), hospital time before the ICU (5 bands), ICU admission hour (where clocks are real) |
| Goal | 50 | ICU outcome, ICU length-of-stay band, hospital outcome, ever IMV / vasopressor / RRT / MCS, maximum KDIGO stage, first-day SOFA band (masked unless requested) |
| Clock | 40 | `[DAY]` marker, `DAY//n`, and each treatment's status (never / active / ended) for IMV, vasopressor and RRT, after every ICU-day crossing (written by the automaton; never a target) |
| Interval | 12 | 1-5 min, 5-15 min, 15-30 min, 30-60 min, 1-2 h, 2-4 h, 4-8 h, 8-12 h, 12-24 h, 1-3 d, 3-7 d, 7 d+ (repeated per whole week) |
| Special | 10 | `[BOS]`, `[EOS]`, `[PAD]`, `[TL]`, `[END]`, `[CONT]`, `[STATE]`/`[/STATE]`, `[GOAL]`/`[/GOAL]` |
| **Total** | **25,601** | |

Event codes by domain (top): medication orders 1,339; intermittent medications 979; continuous medications 765;
microbiology susceptibility 514 and organisms 255; procedures 190; laboratory 78; patient assessments 42;
respiratory support 39; MCS 31; outputs 25; invasive hemodynamics 19; inputs 15; vitals 14; CRRT 10.

## Sequence layout

```
[BOS] prefix (fixed order) header (any order) [GOAL] ... [/GOAL] (optional) [TL]
  time group: events in canonical order | interval token | time group | ... | [DAY] DAY//n STATUS//... | ...
[END] hospital discharge, death, diagnoses [EOS]
```

- **Time:** events within 1 minute of their group's first event join the group; interval tokens between groups;
  times decode from per-(interval, dataset) sketches of the train gaps.
- **Window:** from 48 hours before ICU admission to ICU discharge (marker `ICU//admission`, outcome
  `ICU//discharge//<alive|dead>` at its time). Up to 32,768 tokens per window; longer stays continue in windows that
  re-pin the prefix and header and open with a state summary (0.08% of train stays exceed one window).
- **Values decode** from 5-point sketches (minimum, quartiles, maximum) of the train values inside each bin per
  (code, dataset): at the median (labels) or sampled with within-bin rank persistence (features); a decoded stay
  encodes back to the same tokens.
- **Derived fields are never generated** (GCS total where its components are charted, BMI, P/F, SOFA, NEE): they
  are computed after decoding, so they cannot be inconsistent.
- **Prospective inputs:** the stay as known at a cutoff: the timeline up to it, no goal or end block, and header
  fields documented later (24 h for the diagnosis group and surgical status in eICU, HiRID, AmsterdamUMCdb and
  SICdb) set to unknown.

## Grammar (enforced only where every real sequence obeys it)

| Rule | Content |
|---|---|
| block_order | `[BOS]`, prefix in spec order, header, optional goal, optional `[CONT]` and state, `[TL]`, timeline, `[END]`, end block, `[EOS]` |
| field_once | each header and goal field at most once, all header fields before the timeline; hospital outcome and death at most once, before the diagnoses (in code order) |
| interval_between_groups | an interval never opens or ends a timeline; after an interval other than 7 d+ comes an event |
| group_order | canonical order within a time group; hierarchical events as system token then sub-tokens 1-3 |
| infusion_state | a dose change or stop of an infusion (drug, unit, route) only while it runs |
| clock | after an interval crossing an ICU-day boundary, the `[DAY]` block with the day and each treatment's status, computed from the registers |

## Acceptance (all 409,297 stays, `docs/m5_training.md` "Server run")

| Check | Result |
|---|---|
| Automaton and the independent parser accept every real stay | 0 rejections |
| Round trip (MEDS -> tokens -> decoded stay -> MEDS -> tokens) | 0 failures |
| Re-encoding, goal, window continuity, separation (no outcome or later header in a prospective input at 0, 1, 6, 24 h; invariance to every later event) | 0 failures |
