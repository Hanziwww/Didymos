# Diagrams: sources and briefs for redrawing

The paper's non-numerical diagrams, prepared for redrawing:
- the five schematic panels of the figures (Fig. 1d, the timeline, was removed on 2026-10-11);
- the model-architecture and data-processing diagrams of Supplementary Figs. 1 and 2, whose source is Mermaid.

Everything else in the figures is a data plot, drawn by code from the results.

## The diagrams

| # | Diagram | Where | Slot, mm (W × H) | Source of truth |
|---|---|---|---|---|
| 1 | From records to simulated futures | Fig. 1a | 180 × 62 | `tools/paper_fig1.py` |
| 2 | Constrained sampling (illustrative values) | Fig. 1c | 86 × 48 | `tools/paper_fig1.py` |
| 4 | Design of the what-if validation | Fig. 4a | 180 × 39 | `tools/paper_fig4.py` |
| 5 | The Didymos toolkit | Fig. 7a | 180 × 38 | `tools/paper_fig7.py` |
| 6 | Data and evaluation sets (flow chart) | Supplementary Fig. 3 | 180 × 112 | `tools/paper_figures_supp.py` |
| 7 | The Didymos network and its training | Supplementary Fig. 2a | suggested 88 × 170 | `supp2a_architecture.mmd` |
| 8 | Simulating futures, and the two estimates | Supplementary Fig. 2b | suggested 88 × 170 | `supp2b_simulation.mmd` |
| 9 | From source databases to model-ready sequences | Supplementary Fig. 1a | suggested 180 × 125 | `supp1a_data_processing.mmd` |
| 10 | One stay as one token sequence | Supplementary Fig. 1b | suggested 180 × 70 | `supp1b_sequence.mmd` |

The slots of diagrams 1-2 and 4-6 are the artwork area below each panel's letter and title in the code-drawn
figures, measured from the figure's top left (`tools/paper_diagrams.py`, `SLOTS`). The facts in diagrams 7-10 come
from `docs/paper/architecture.md`, `training.md`, `preprocessing.md`, `data.md` and `tokenization.md`; every number in
them is in those documents.

## Files

- `prompts.md`: one image-generation prompt per diagram, with the house style, the verbatim labels and what must stay
  exact.
- `*.mmd`: the Mermaid sources of Supplementary Figs. 1 and 2.
  - *Rendered:* by `python tools/paper_diagrams.py render` to `docs/paper/figures/supplementary/diagrams/`, as SVG
    with editable text, with `manifest.json`.
  - *Renderer:* mermaid 11.4.1 in headless Edge, locally; nothing goes to an online renderer.
- **The bundle:** `python tools/paper_diagrams.py bundle` writes `runs/paper/figures/diagram_bundle/` and a zip of it
  (git-ignored). It contains:
  - `reference/`: each slot cropped from the 600-dpi figure previews;
  - `vector/`: each slot cropped from the vector SVG, with text as text;
  - `figures/`: the whole figures, PDF and SVG, for context;
  - `slots.csv`: each slot's position and size in mm, and its pixel size at 600 dpi;
  - `labels.json`: every label in each slot, verbatim, read from the vector files;
  - `mermaid/`: the four Mermaid sources, their SVG renders and PNG previews;
  - this README and `prompts.md`.

## Mermaid notes

- Node labels must not begin with "1." and the like: Mermaid reads that as a Markdown list and drops the label. Stage
  labels therefore read "Stage 1: ...".
- Each source sets `htmlLabels: false` at both the top level and under `flowchart`. Without both, Mermaid 11 writes
  HTML labels (`foreignObject`), which vector editors do not open.
- Subgraph titles wrap at a fixed width, so they are kept short, and the detail sits in the nodes.

## The artwork in use (GPT-image, 2026-10-11)

The authors' GPT-image versions of diagrams 1, 2 and 4-10 are in the figures:
- *Placement:* `st.artwork`, cropped to the content (white border removed), at 600 dpi at print size.
- *Files:* the originals, their provenance and the credit rule, in `docs/paper/figures/artwork/gpt_image/`.
- *Credit:* "(GPT-image)" wherever they appear. The authors confirm their accuracy and scientific validity and assume
  full responsibility for them.
- *Scope:* the Mermaid sources stay as the structural record of Supplementary Figs. 1 and 2.
- *Fig. 1d:* the timeline was removed on 2026-10-11.

## Swapping artwork in

Deliver one file per slot. Either of:
- vector PDF or SVG, text as text, Arial 7 pt;
- PNG or TIFF, RGB, at 300 dpi or more at its printed width.

Keep the panel letter and title out of the artwork: the figure code sets them. The numbers must match `labels.json`
and the Mermaid sources exactly.

To place a PNG:
1. Save it as `docs/paper/figures/artwork/gpt_image/original/<slot>.png`, with its credit and sha256 in
   `provenance.json` there.
2. List the slot in `st.USE_ARTWORK` (`tools/paper_figstyle.py`). The white border is cropped (`st.artwork_box`),
   the panel takes the content's aspect ratio at its printed width, and the figure resizes to fit.
3. Rebuild the figures. A slot taken out of `st.USE_ARTWORK` falls back to the code-drawn schematic.
