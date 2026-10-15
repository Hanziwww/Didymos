# Image-generation prompts for the paper's diagrams

Nine diagrams (the Fig. 1d timeline was removed, 2026-10-11):
- the five schematic panels of the main and supplementary figures, to redraw and swap in;
- the four diagrams of Supplementary Figs. 1 and 2 (model architecture, simulation, data processing, the token
  sequence), to draw from their Mermaid sources.

Each entry gives the slot, the message, what must stay exact, a prompt for the artwork and the labels to set.

## How to use the prompts

- **Text.** Image generators misspell small text. Each prompt asks for the artwork with blank label areas. Set the
  labels from the list afterwards in a vector editor: Arial (or Helvetica), 7 pt at final size, sentence case, black
  or dark grey (#4D4D4D). If your tool renders text reliably, paste the label list into the prompt instead.
- **Size.** Draw at the slot size (`slots.csv`, millimetres, width × height). Deliver either vector PDF or SVG with
  text kept as text, or a 600-dpi PNG or TIFF at the slot's pixel size. RGB.
- **Facts.** Numbers, names and counts in the label lists are data and must not change. Elements marked
  *illustrative* (token values, trajectories, probabilities) can be redrawn freely.
- **Panel letters and titles** stay in the figure code. The artwork fills the box below them.

## House style (append to every prompt)

> Flat vector scientific schematic for a Nature Portfolio journal, white background. Clean geometric shapes: rounded
> rectangles with thin 1-pt outlines, simple arrows, generous whitespace, elements aligned on a grid. Neutral boxes
> have a cool off-white fill (#F0F3F5) with a soft slate outline (#9299A0). Model and data boxes have a pale-blue
> fill (#E3EEF7) with a blue outline (#0072B2). There is one accent hue, blue (#0072B2), with tints #BBD7EC and
> #82B7E4 and a dark blue #005485 for emphasis. Neutrals are cool slates (#4E565E for secondary lines and text,
> #B8BFC4 for context lines), with black for text. Vermillion (#D55E00) marks only events. Colour-blind safe, no
> red against green. No 3D, shadows, gradients, textures, photographs, people, clip art or decorative icons.

The palette in full is `BLUE`, `SLATE` and `OUTCOME` in `tools/paper_figstyle.py`, as used by Figs. 1 and 3-8.

**Negative prompt (all):** text artefacts, invented labels, lorem ipsum, watermarks, logos, perspective, glow, dark
backgrounds, stock hospital imagery, robots, brains, glossy 3D.

---

## 1. Fig. 1a: from records to simulated futures (the hero panel)

- **Slot:** 180 × 62 mm (2.9 : 1). Reference: `reference/fig1a.png`, `vector/fig1a.svg`.
- **Message:** Didymos turns a harmonized ICU record into many simulated futures. Two risk estimates are read from
  the futures, and they feed the tools.
- **Exact:** every number and word in the boxes. **Illustrative:** the token values, the curves and the event dots
  (no real patient is shown).

**Prompt:**
> A wide horizontal scientific schematic, aspect 2.9 to 1, in two rows.
> - Top row, left to right: a light-grey rounded box with a blank label area, a short arrow, a second light-grey
>   rounded box, a short arrow, then a two-row strip of small rounded pill-shaped tokens. The pills are empty: some
>   with grey outlines, some with blue outlines, two filled pale blue.
> - Bottom row, left to right: a pale-blue rounded box with a small icon of stacked horizontal bars (a layer stack)
>   at its left edge, then an arrow. Next, a small line chart without axis text: a black history line on the left
>   half ends at a dashed vertical line in the middle. From it, a fan of about ten thin pale-blue lines spreads to the
>   right, one of them darker blue, with three small vermillion dots on the fan. Then an arrow, a light-grey rounded
>   box, an arrow, and a white rounded box with a blue outline.
>
> Leave all label areas empty. [house style]

**Labels:**
- Box 1: **Seven ICU databases** / 409,297 stays / five countries
- Box 2: **Harmonized and tokenized** / CLIF, then MEDS / 25,601 tokens
- Token pills, row 1: [BOS] · eICU · site 73 · female · age 70–74 · medical ICU · timeline
- Token pills, row 2: HR q6 · MAP q3 · SpO2 q8 · 5–15 min · norepinephrine start · rate q2 · 1–2 h · lactate q9
  - Outlines: context tokens grey; measurements blue; "norepinephrine start" and "rate q2" filled pale blue; intervals
    grey.
- Model box: **Didymos 1.0** / decoder, 12 layers / 123M parameters / context 32,768 tokens / trained once,
  privacy-bounded
- Chart: History · Simulated futures · −24 h · now · +24 h · legend "● an event in a future" (the dot vermillion)
- Estimates box: **Sampled estimate** / share of futures with the event / **Hazard estimate** / the model's step
  hazards, summed along each future
- Tools box (blue text): Forecasts · What-if · Synthetic cohorts · Explanations · Trial design

## 2. Fig. 1c: constrained sampling

- **Slot:** 86 × 48 mm (1.8 : 1). Reference: `reference/fig1c.png`, `vector/fig1c.svg`.
- **Message:** before each draw, the grammar and the database's support remove tokens that would be invalid. The
  rest are renormalized.
- **Exact:** the token names and the reasons for masking. **Illustrative:** the probabilities. A renormalized bar
  equals the model's probability ÷ (1 − the masked total).

**Prompt:**
> A compact horizontal bar diagram, aspect 1.8 to 1, eight rows.
> - In each row, a thin dark-grey outlined bar (the model's probability) is overlaid by a slightly longer solid blue
>   bar (after renormalization).
> - Rows 4, 5 and 7 instead show a short hatched light-grey bar with no blue bar, and a small vermillion tag to its
>   right.
> - A thin horizontal axis runs along the bottom with three ticks.
> - Below it, a legend row: an outlined bar, a solid blue bar and a hatched bar.
>
> Row labels on the left are blank. [house style]

**Labels:**
- Rows, top to bottom: MAP q4 · HR q5 · 5–15 min · norepinephrine rate up (tag: grammar) · ventilator PEEP (tag: not
  recorded here) · lactate q3 · [END] block (tag: grammar) · SpO2 q9
- Axis: 0 · 0.2 · 0.4 · Next-token probability (illustrative)
- Legend: Model · After masking, renormalized · Masked

## 4. Fig. 4a: design of the what-if validation

- **Slot:** 180 × 39 mm (4.6 : 1). Reference: `reference/fig4a.png`, `vector/fig4a.svg`.
- **Message:** a real record is cut 1 minute before a recorded treatment start. From the cut, Didymos simulates the
  response over a 6-hour window three times: with the treatment given, withheld, and withheld again with another
  seed (the null arm).
- **Exact:** the arm names. **Illustrative:** the event ticks and the lines.
- **Optional addition:** a thinner second record below the first for the propensity-matched non-starter at the same
  ICU hour, labelled "Matched non-starter, same ICU hour".

**Prompt:**
> A horizontal design schematic, aspect 4.6 to 1.
> - On the left, a long thin rectangle holds irregular vertical tick marks (events in a patient record). It ends at a
>   dashed vertical line just left of the centre.
> - Right of the line, a pale-blue shaded window holds three bundles of four or five thin, slightly wavy lines, all
>   starting from the same point on the dashed line:
>   - a bundle rising in blue;
>   - a flat bundle in dark grey;
>   - a flat dashed bundle in light grey, slightly lower.
> - At the right end of each bundle is a blank label area.
>
> [house style]

**Labels:** Real ICU record up to the cut · Cut · Give: the recorded start (blue) · Withhold · Null arm: second
withhold. Optional: Response window, 6 h.

## 5. Fig. 7a: the Didymos toolkit

- **Slot:** 180 × 38 mm (4.7 : 1). Reference: `reference/fig7a.png`, `vector/fig7a.svg`.
- **Message:** one model and seven tools. Every output carries its evidence and caveats.
- **Exact:** the tool names, descriptions and cross-references.

**Prompt:**
> A horizontal process schematic, aspect 4.7 to 1.
> - On the left, a light-grey rounded box, with an arrow to a pale-blue rounded box.
> - An arrow leads into a block of seven white rounded boxes with blue outlines, in two rows: four on top, three
>   below, the lower row offset by half a box.
> - An arrow leads out to a light-grey rounded box on the right.
>
> Each tool box has room for three short lines: a bold title, a description and a reference. Optionally, one simple
> flat line icon at the top left of each tool box, in this order:
> - a small line chart;
> - a fork with two branches;
> - a magnifier;
> - two overlapping circles;
> - a grid of dots;
> - a balance scale;
> - an open book.
>
> Leave the text areas blank. [house style]

**Labels:**
- Input: A record at an hour or a cohort specification
- Model (blue title): **Didymos 1.0** / simulated futures under the grammar and support masks
- Tools (title / description / reference):
  - Forecast / risks and curves / Figs 2, 3
  - What-if / give or withhold / Fig. 4
  - Explain / editing, knockout / Fig. 6
  - Similar patients / nearest stays / Supplementary
  - Cohort synthesizer / audited synthetic cohorts / Fig. 5
  - Trial designer / adjustment and enrichment / panels b, c
  - Teaching cases / synthetic decision points / panel d
- Output: JSON and an HTML card; evidence and caveats each

## 6. Supplementary Fig. 3: data and evaluation sets

- **Slot:** 180 × 112 mm (1.6 : 1), the whole figure. Reference: `reference/supp3.png`, `vector/supp3.svg`.
- **Message:** from seven databases to the splits, and from the held-out split to the anchor sets used for
  evaluation.
- **Exact:** every label, verbatim.

**Prompt:**
> A clean top-down flow chart, aspect 1.6 to 1, white background.
> - At the top centre, one wide light-grey rounded box.
> - Orthogonal connectors with right angles fan out to three light-grey rounded boxes in a row.
> - From the right-hand box, connectors fan out to a second row of three light-grey boxes.
> - From the first two boxes of that row, vertical connectors lead to two pale-blue boxes with blue outlines in a
>   third row.
>
> Every box has a blank area for a bold first line and two or three regular lines. [house style]

**Labels** (bold first line / regular lines):
- Top: **Seven ICU databases in five countries** / 409,297 adult ICU stays, harmonized and tokenized
- **Train: 327,662 stays (80%)** / Didymos's training (no task labels); comparators' training: up to 5,000 per
  database
- **Tuning: 20,453 stays (5%)** / model selection, early stopping; recalibration (100 rolling anchors per database)
- **Held-out: 61,182 stays (15%)** / evaluation only: a test store of 2,000 stays per database (14,000)
- **Rolling anchors: 3,500** / 500 per database at random ICU hours, enriched for events (inclusion weights); 32
  simulated futures each
- **24-hour landmark: 3,500** / up to 500 per database, in the ICU and alive at 24 h; 32 simulated futures each
- **48-hour landmark: 3,500** / up to 500 per database, in the ICU and alive at 48 h; 32 simulated futures each
- **Rolling anchors, follow-up: 1,050** (pale blue) / 150 per database of the rolling anchors; 128 simulated futures
  each
- **24-hour landmark, follow-up: 700** (pale blue) / every in-hospital death (346) and 354 others, weighted (a
  case-cohort sample); 128 simulated futures each

---

## 7. Supplementary Fig. 2a: the Didymos network and its training

- **Source of truth:** `mermaid/supp2a_architecture.mmd` (structure and every label); current render
  `mermaid/supp2a_architecture.svg`.
- **Suggested slot:** the left half of Supplementary Fig. 2, 88 × 170 mm (portrait). The figure may be up to
  180 × 225 mm.
- **Message:** a 12-block decoder-only transformer over event tokens, trained once on all seven databases.
- **Exact:** all numbers (layers, width, heads, sizes, parameters, steps, tokens, passes).

**Prompt:**
> A vertical transformer architecture diagram in the style of machine-learning papers, portrait, aspect about 0.5 to
> 1.
> - From top to bottom: an input strip of small token pills; an embedding block; then a large rounded container,
>   marked as repeated twelve times.
> - Inside the container, a vertical stack:
>   - a thin grey normalization bar;
>   - a pale-blue self-attention block with a small motif of twelve thin query lines sharing four key-value lines;
>   - a small circular addition node, with a residual skip connection drawn as a dotted bypass line around the block;
>   - another thin normalization bar;
>   - a pale-blue feed-forward block drawn as a trapezoid that widens then narrows;
>   - a second addition node with its own dotted residual bypass.
> - Below the container: a final normalization bar, an output block and a small bar chart of next-token
>   probabilities.
> - To the lower left, a separate white panel with a thin grey outline holds three stacked text blocks for training;
>   to the lower right, a small white note box.
>
> Blank label areas. [house style]

**Labels:** every node and container text of the Mermaid source, verbatim (line breaks at `<br/>`). In particular:
- Input: one ICU stay as a token sequence, up to 32,768 tokens; vocabulary of 25,601 tokens
- Token embedding, 25,601 × 768 (shared with the output head)
- Decoder block × 12:
  - RMSNorm (pre-norm)
  - Grouped-query self-attention, causal: 12 query heads, 4 key-value heads, head dimension 64; rotary positions
    (base 1,000,000); per-head q/k RMSNorm; full attention over the whole context
  - Add · RMSNorm · SwiGLU feed-forward 768 → 3,072 → 768 · Add · residual
- Final RMSNorm · Output head, 768 → 25,601 (the shared embedding) · Next-token probabilities, softmax over the
  vocabulary
- 123.5 million parameters: embedding 19.7M; decoder body 103.8M
- Training, once: the three blocks of the source

## 8. Supplementary Fig. 2b: simulating futures, and the two estimates

- **Source of truth:** `mermaid/supp2b_simulation.mmd`; current render `mermaid/supp2b_simulation.svg`.
- **Suggested slot:** the right half of Supplementary Fig. 2, 88 × 170 mm (portrait).
- **Message:** at every step, the model's probabilities pass through the grammar, the database's support and the
  token budget, are renormalized and sampled. The simulated futures give two risk estimates.

**Prompt:**
> A vertical process diagram, portrait.
> - At the top, two input boxes side by side: one with a solid outline, one with a dotted outline (the optional
>   what-if). Both feed a pale-blue model box.
> - Below, a rounded container holds four stacked filter steps, drawn as sieves that narrow a set of small candidate
>   token pills, with a small circular arrow showing that the step repeats.
> - Below that, a fan of thin simulated future trajectories.
> - Then two estimate boxes side by side, one light grey and one pale blue, converging on a final box.
>
> Blank label areas. [house style]

**Labels:** every node and container text of the Mermaid source, verbatim. These are the prompt, the what-if, the
model, the four constraint steps, simulated futures (32 or 128 per prediction time), the sampled estimate, the hazard
estimate and the uses.

## 9. Supplementary Fig. 1a: from source databases to model-ready sequences

- **Source of truth:** `mermaid/supp1a_data_processing.mmd`; current render `mermaid/supp1a_data_processing.svg`.
- **Suggested slot:** the top of Supplementary Fig. 1, 180 × 125 mm.
- **Message:** one pipeline harmonizes seven databases into one cohort with one sequence per stay, split by patient.
- **Exact:** the database names and versions, the stage contents, and the counts (409,297; 327,662; 20,453; 61,182;
  2,000 per database; 25,601 tokens).

**Prompt:**
> A top-down data pipeline diagram, aspect about 1.4 to 1.
> - At the top, seven small database cylinders in a row, each with a blank label below. Arrows merge them into a
>   vertical sequence of five stage boxes; the fourth and fifth are pale blue.
> - Two white side-note boxes with thin grey outlines:
>   - one feeds the fourth stage with a dotted arrow (uniform rules);
>   - one comes out of the fourth stage (derived concepts).
> - After a split box, three output boxes sit in a row.
> - Dotted arrows marked "labels" run from the derived-concepts note to the first and third output boxes.
>
> Blank label areas. [house style]

**Labels:** every node text of the Mermaid source, verbatim. Split the first node into the seven cylinders:
- MIMIC-IV 3.1 (USA)
- eICU-CRD 2.0 (USA, 208 hospitals)
- HiRID 1.1.1 (Switzerland)
- AmsterdamUMCdb 1.0.2 (Netherlands)
- SICdb 1.0.8 (Austria)
- NWICU 0.1.0 (USA)
- INSPIRE 1.4.2 (South Korea)

## 10. Supplementary Fig. 1b: one stay as one token sequence

- **Source of truth:** `mermaid/supp1b_sequence.mmd`; current render `mermaid/supp1b_sequence.svg`.
- **Suggested slot:** the bottom of Supplementary Fig. 1, 180 × 70 mm.
- **Message:** a stay becomes one grammar-checked sequence: a prefix and header, a timeline of time groups and
  intervals with daily clock blocks, and an end block.
- **Exact:** the block contents and the token counts (270; 96; 50; 19,774; 12 bins; 40; 500; 3,057; 32,768).

**Prompt:**
> A horizontal token-sequence diagram, aspect about 2.6 to 1, in three bands stacked vertically and joined by short
> arrows:
> - Band 1, a row of four blocks: a small start token, a prefix block, a header block, and an optional goal block
>   with a dotted outline.
> - Band 2, a long row: a timeline start token; a pale-blue time-group block holding several tiny token pills; a
>   narrow pale-blue interval token; another time-group block; a clock block; an ellipsis.
> - Band 3: an end block and a small end token.
> - Under the bands, a small white note box.
>
> Blank label areas. [house style]

**Labels:** every node and container text of the Mermaid source, verbatim.
