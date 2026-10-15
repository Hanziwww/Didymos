# Schematic artwork (GPT-image)

**Credit.** These images were generated with GPT-image. They must be credited "(GPT-image)" wherever they appear: in
the figure legends (each panel is marked), the Methods' statement on AI use, and any reuse.

**The authors' statement** (2026-10-11): the authors confirm the accuracy and scientific validity of these images and
assume full responsibility for them.

## The images

| File (`original/`) | Where | Native pixels | Content box, px (left, top, right, bottom) | Kept pixels | Printed size | Native dpi in print |
|---|---|---|---|---|---|---|
| `fig1a.png` | Fig. 1a, from records to simulated futures | 1860 × 846 | 19, 11, 1849, 787 | 1830 × 776 | 180 × 76 mm | 258 |
| `fig1c.png` | Fig. 1c, constrained sampling | 1478 × 1064 | 59, 47, 1378, 950 | 1319 × 903 | 86 × 59 mm | 390 |
| `fig4a.png` | Fig. 4a, design of the what-if validation | 2172 × 724 | 24, 37, 2153, 619 | 2129 × 582 | 180 × 49 mm | 300 |
| `fig7a.png` | Fig. 7a, the Didymos toolkit | 1983 × 793 | 29, 112, 1960, 676 | 1931 × 564 | 180 × 53 mm | 272 |
| `supp1a.png` | Supplementary Fig. 1a, data processing | 1536 × 1024 | 44, 8, 1492, 1000 | 1448 × 992 | 180 × 123 mm | 204 |
| `supp1b.png` | Supplementary Fig. 1b, one stay as one token sequence | 1942 × 809 | 47, 95, 1899, 757 | 1852 × 662 | 180 × 64 mm | 261 |
| `supp2a.png` | Supplementary Fig. 2a, the Didymos network | 1086 × 1448 | 192, 14, 894, 1431 | 702 × 1417 | 63 × 127 mm | 283 |
| `supp2b.png` | Supplementary Fig. 2b, constrained simulation and the two estimates | 1199 × 1312 | 39, 5, 1188, 1298 | 1149 × 1293 | 113 × 127 mm | 258 |
| `supp3.png` | Supplementary Fig. 3, data and evaluation sets | 1536 × 1024 | 20, 10, 1516, 985 | 1496 × 975 | 180 × 117 mm | 211 |

The content boxes are computed from the originals by `st.artwork_box` and recorded in each figure's source data
(`crop_px`).

## Provenance

- *Originals:* `original/` holds the native PNGs exactly as received (package `Didymos_figures_EN_no_timeline.zip`,
  2026-10-11), renamed to their current figure numbers. The package used the old supplementary numbers: supp1 is now
  S3, supp15a/b are S2a/b, supp16a/b are S1a/b.
- *Content credentials:* each original keeps its C2PA content credentials (the `caBX` chunk).
- *Hashes:* `provenance.json` lists the package's sha256, each image's sha256 and size, and the authors' statement.
- *Generation records:* `generation/` holds the generation and edit prompts and the two visual review rounds that
  came with the package (the text as received, with its line endings normalized to LF).
- *Structure:* the facts in the artwork follow the Mermaid sources and the materials (`docs/paper/manuscript/
  diagrams/`).

## High-resolution versions

The figures place each image at its printed size at 600 dpi (`tools/paper_figstyle.py`, `artwork`).
- *White border:* each image is cropped to its content box (`st.artwork_box`), so the artwork starts under its panel
  letter and fills the panel's width:
  - *Content:* every row and column with at least 3 pixels more than 8 of 255 levels from the background. The
    background is the median colour of the outermost rows and columns (253-254 in every image, within 2 levels of
    white).
  - *Margin:* 6 native pixels are kept around the content for the anti-aliased edges.
  - *What is dropped:* background noise of a few levels and isolated specks (at most 1 pixel per image above the
    threshold).
  - *Test:* `tests/test_paper_figures.py` checks each box.
- *Resampling:* the cropped image is resampled with Lanczos. Nothing generative or learned is applied, so no detail is
  added; the effective detail stays that of the native image.
- *Native resolution:* cropping enlarges the content, so the native resolution in print is 204-390 dpi (the table
  above), and seven of the nine images are below 300 dpi before resampling. For sharper print, regenerate them
  larger: at least 2,126 pixels of content across a full-width (180 mm) image for 300 dpi.
- *Files:* the resampled pixels are embedded in the figure PDFs and SVGs. The originals are not modified.
- *Fallback:* a slot taken out of `st.USE_ARTWORK` falls back to the code-drawn schematic.

## Notes for submission

- **Policy.** Nature Portfolio's AI policy, read on 2026-10-11, is a risk framework: AI use is disclosed,
  accountability stays with the authors, and photorealistic images are not permitted. Confirm with the journal at
  submission that GPT-image schematics are acceptable.
- **Terminology.** The artwork labels the two estimates "sampled risk" and "hazard risk"; the paper calls them the
  sampled and hazard estimates, and the legends give both.
- **Style.** The artwork uses its own palette (teal and lavender) and typeface, unlike the data panels' Arial and
  Didymos blue. Lavender sits near TabPFN's purple in the comparator figures.
