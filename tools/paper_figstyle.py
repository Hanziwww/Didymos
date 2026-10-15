"""The paper's figure design system (npj Digital Medicine's figure rules).

* **Size:** figures drawn at print size, 180 mm wide (88 mm for half width), at most 225 mm tall. Axes are placed in
  millimetres (``Canvas.axes``), not by automatic layout.
* **Type:** Arial at one size (``PT``) for every text, panel letters bold at that size, no Greek letters, true minus
  signs, an italic *P* through mathtext in Arial.
* **Lines:** at least 1 pt.
* **Colour:** Okabe-Ito semantic colours (Didymos blue as the one accent), cividis and PuOr for scales.
* **Export:** vector PDF with TrueType fonts and no creation date (deterministic), SVG with the text kept as text, and
  a PNG preview.
* **Checks:** ``check`` lists every violation: font, size, line width, overlapping text, Greek glyphs, internal codes,
  panel letters.
* **Tools:** colour-blind and greyscale simulations of the previews (Machado et al. 2009); a source-data collector.
"""

from __future__ import annotations

import csv
import hashlib
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

MM = 1 / 25.4
FULL, HALF, MAX_H = 180.0, 88.0, 225.0
FONT, PT = "Arial", 7.0
LW_MIN, LW_DATA = 1.0, 1.5
MARKER = 4.0

# Okabe-Ito, by meaning
DIDYMOS = "#0072B2"
DIDYMOS_TINT = "#BBD7EC"
DIDYMOS_LIGHT = "#E3EEF7"
LGBM = "#E69F00"
TABPFN = "#CC79A7"
CLINICAL = "#000000"
LEARNER = "#B3B3B3"
BEDSIDE = "#BDBDBD"
PROBE = "#56B4E9"
GREEN = "#009E73"
FLAG = "#D55E00"
TEXT = "#000000"
TEXT2 = "#4D4D4D"
AXIS = "#333333"
REF = "#8C8C8C"
FILL = "#F0F0F0"
PALETTE = {"didymos": DIDYMOS, "didymos_tint": DIDYMOS_TINT, "didymos_light": DIDYMOS_LIGHT, "lgbm": LGBM,
           "tabpfn": TABPFN, "clinical": CLINICAL, "learner": LEARNER, "bedside": BEDSIDE, "probe": PROBE,
           "green": GREEN, "flag": FLAG, "text2": TEXT2, "axis": AXIS, "ref": REF, "fill": FILL, "white": "#FFFFFF"}

# The enhanced scheme of Figs. 1 and 3-8 (2026-10-11). The semantic colours above are unchanged, so Fig. 2 and the
# supplementary figures look as before and every comparator keeps its colour.
# - BLUE: Didymos's tonal ramp, one hue (OKLCH h 244) with monotone lightness, for ordered series (doses, amounts,
#   caps). 100 and 200 are the existing tints; 300-700 pass the dataviz validator's ordinal checks.
# - SLATE: cool neutrals at the same hue and low chroma, for context series, references, fills and outlines.
# - Distinct hues only where several of Didymos's outputs must be told apart: Okabe-Ito blue, bluish green and
#   vermillion (validated all-pairs: CVD separation 11.0, normal 18.7, contrast >= 3:1), with black as a neutral.
BLUE = {100: DIDYMOS_LIGHT, 200: DIDYMOS_TINT, 300: "#82B7E4", 400: "#4793CD", 500: DIDYMOS, 700: "#005485",
        900: "#053656"}
SLATE = {100: "#F0F3F5", 200: "#DFE2E6", 300: "#B8BFC4", 400: "#9299A0", 500: "#798188", 700: "#4E565E",
         900: "#21272D"}
VERMILLION = FLAG
OUTCOME = {"death": CLINICAL, "vasopressor": DIDYMOS, "imv": GREEN, "rrt": VERMILLION}
RING = {"mec": "white", "mew": 1.0}                    # a white ring that lifts a filled marker off its line

# The schematic artwork (2026-10-11): GPT-image PNGs, credited "(GPT-image)" wherever they appear; the authors confirm
# their accuracy and scientific validity and take full responsibility for them (docs/paper/figures/artwork/gpt_image/).
# A slot listed here draws its artwork; a slot taken out of the set falls back to the programmatic schematic.
ARTWORK = Path(__file__).resolve().parents[1] / "docs" / "paper" / "figures" / "artwork" / "gpt_image" / "original"
USE_ARTWORK = {"fig1a", "fig1c", "fig4a", "fig7a", "supp1a", "supp1b", "supp2a", "supp2b", "supp3"}
ART_DPI = 600
ART_CREDIT = "GPT-image"
ART_TOL, ART_MIN_PX, ART_PAD = 8, 3, 6  # the white-border crop: levels of 255, pixels per row or column, margin


@lru_cache(maxsize=None)
def artwork_box(name: str) -> tuple[int, int, int, int]:
    """The content box (left, top, right, bottom; native pixels) that crops the white border of artwork ``name``.

    Content is every row and column with at least ``ART_MIN_PX`` pixels more than ``ART_TOL`` levels from the
    background, the median colour of the outermost rows and columns; ``ART_PAD`` pixels are kept around it for the
    anti-aliased edges. Background noise (a few levels) and isolated specks fall outside."""
    from PIL import Image

    with Image.open(ARTWORK / f"{name}.png") as im:
        a = np.asarray(im.convert("RGB")).astype(int)
    bg = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), axis=0)
    ink = np.abs(a - bg).max(axis=2) > ART_TOL
    rows = np.flatnonzero(ink.sum(axis=1) >= ART_MIN_PX)
    cols = np.flatnonzero(ink.sum(axis=0) >= ART_MIN_PX)
    h, w = ink.shape
    return (max(int(cols[0]) - ART_PAD, 0), max(int(rows[0]) - ART_PAD, 0),
            min(int(cols[-1]) + 1 + ART_PAD, w), min(int(rows[-1]) + 1 + ART_PAD, h))


def artwork_height(name: str, width: float) -> float:
    """The height in mm of artwork ``name`` drawn ``width`` mm wide, at the aspect ratio of its content box."""
    x0, y0, x1, y1 = artwork_box(name)
    return width * (y1 - y0) / (x1 - x0)


def artwork_pixels(name: str, width: float, dpi: int = ART_DPI):
    """The artwork cropped to its content box, at ``dpi`` for its printed width: Lanczos resampling of the native PNG,
    nothing generative or learned, so no detail is added; the native file keeps its C2PA content credentials."""
    from PIL import Image

    with Image.open(ARTWORK / f"{name}.png") as im:
        im = im.convert("RGB").crop(artwork_box(name))
        tw = round(width / 25.4 * dpi)
        return im.resize((tw, round(tw * im.height / im.width)), Image.LANCZOS)


def artwork(cv: "Canvas", sd: "SourceData", name: str, x: float, y: float, width: float, panel: str) -> float:
    """Place artwork ``name``, cropped to its content box, with its top-left corner at (x, y) mm, ``width`` mm wide;
    returns its height (mm). The native file and its sha256 go into the figure's inputs, and the source data records
    the credit and the crop."""
    h = artwork_height(name, width)
    ax = cv.axes(x, y, width, h)
    img = artwork_pixels(name, width)
    ax.imshow(np.asarray(img), interpolation="none", aspect="auto")
    ax.set_axis_off()
    f = ARTWORK / f"{name}.png"
    sd.source(f)
    x0, y0, x1, y1 = artwork_box(name)
    sd.add(panel, "artwork", file=f"docs/paper/figures/artwork/gpt_image/original/{name}.png", credit=ART_CREDIT,
           crop_px=f"{x0},{y0},{x1},{y1}", width_mm=round(width, 2), height_mm=round(h, 2), printed_dpi=ART_DPI,
           pixels=f"{img.width}x{img.height}")
    return h

MINUS = "−"

ENDPOINT = {"decompensation@1440": "Death within 24 h", "imv_onset@1440": "New IMV within 24 h",
            "vasopressor_onset@1440": "New vasopressor within 24 h", "rrt_onset@2880": "New RRT within 48 h",
            "aki_2@2880": "AKI stage ≥2 within 48 h", "sepsis_ase@1440": "Sepsis within 24 h",
            "hypotension@240": "Hypotension within 4 h", "hyperkalemia_top@1440": "Hyperkalaemia within 24 h",
            "thrombocytopenia@2880": "Thrombocytopenia within 48 h", "icu_death@-1": "ICU death",
            "hospital_death@0": "In-hospital death", "remaining_los_gt_7d": "ICU stay over 7 days"}
ENDPOINT_SHORT = {"decompensation@1440": "Death, 24 h", "imv_onset@1440": "New IMV, 24 h",
                  "vasopressor_onset@1440": "New vasopressor, 24 h", "rrt_onset@2880": "New RRT, 48 h",
                  "aki_2@2880": "AKI stage ≥2, 48 h", "sepsis_ase@1440": "Sepsis, 24 h",
                  "hypotension@240": "Hypotension, 4 h", "hyperkalemia_top@1440": "Hyperkalaemia, 24 h",
                  "thrombocytopenia@2880": "Thrombocytopenia, 48 h", "icu_death@-1": "ICU death",
                  "hospital_death@0": "In-hospital death", "remaining_los_gt_7d": "ICU stay >7 days"}
PROGRAM = {"decompensation": "Death", "icu_death": "ICU death", "hospital_death": "In-hospital death",
           "remaining_los_gt_7d": "ICU stay >7 days", "circulatory_failure": "Circulatory failure",
           "hypoxemia": "Hypoxaemia", "aki_1": "AKI stage ≥1", "aki_2": "AKI stage ≥2", "aki_3": "AKI stage 3",
           "aki_2_creatinine": "AKI ≥2 (creatinine)", "aki_2_urine": "AKI ≥2 (urine)",
           "liver_dysfunction": "Liver dysfunction", "hyperglycaemia": "Hyperglycaemia",
           "hypoglycaemia": "Hypoglycaemia", "sepsis_ase": "Sepsis", "sepsis_ase_wo_lactate": "Sepsis without lactate",
           "imv_onset": "New IMV", "vasopressor_onset": "New vasopressor", "rrt_onset": "New RRT",
           "nippv_onset": "New NIV", "ventilatory_support": "New IMV or NIV",
           "discharge_alive": "Discharge alive", "hypotension": "Hypotension", "tachycardia": "Tachycardia",
           "tachypnea": "Tachypnoea", "fever": "Fever ≥38.0 °C", "fever_383": "Fever ≥38.3 °C",
           "low_gcs": "GCS below 9", "lactate_rise": "Lactate above 4", "acidemia": "Acidaemia",
           "hyperkalemia_top": "Hyperkalaemia", "hyperkalemia_55": "Potassium above 5.5",
           "hyponatremia": "Hyponatraemia", "thrombocytopenia": "Thrombocytopenia", "anemia": "Anaemia",
           "leukocytosis": "Leukocytosis"}
HORIZON = {60: "1 h", 240: "4 h", 480: "8 h", 720: "12 h", 1440: "24 h", 2880: "48 h"}
# the catalog's families (src/icutwin/configs/factory/catalog.yaml), by name, in display order
FAMILY = {"F1": "Mortality", "F2": "Organ failure", "F3": "Sepsis", "F4": "Intervention onsets", "F5": "Discharge",
          "F6": "Threshold crossings", "F7": "Remaining stay"}


def program_name(prog: str) -> str:
    """A program's display name: 'New IMV, 24 h'; stay-level outcomes without a horizon."""
    base, _, h = prog.partition("@")
    name = PROGRAM.get(base, base.replace("_", " "))
    return f"{name}, {HORIZON[int(h)]}" if h and int(h) in HORIZON else name


DATASETS = ("mimic", "eicu", "hirid", "aumc", "sicdb", "nwicu", "inspire")
DATASET = {"mimic": "MIMIC-IV", "eicu": "eICU", "hirid": "HiRID", "aumc": "AmsterdamUMCdb", "sicdb": "SICdb",
           "nwicu": "NWICU", "inspire": "INSPIRE"}
DATASET_SHORT = {**DATASET, "aumc": "AUMCdb"}
COUNTRY = {"mimic": "USA", "eicu": "USA", "hirid": "Switzerland", "aumc": "Netherlands", "sicdb": "Austria",
           "nwicu": "USA", "inspire": "South Korea"}
SCORE = {"sofa": "SOFA", "news": "NEWS", "mews": "MEWS", "qsofa": "qSOFA", "sirs": "SIRS", "sapsii_prob": "SAPS II",
         "oasis_prob": "OASIS", "lgbm_pooled": "LightGBM", "tabpfn_v3_5": "TabPFN-3.5", "twin": "Didymos",
         "apache_iva": "APACHE IVa"}
PREDICTOR = {**SCORE, "twin_current": "Didymos", "twin_e2": "Didymos, hazard estimate",
             "twin_128": "Didymos, sampled estimate", "twin_probe": "Didymos probe",
             "lgbm_per_dataset": "LightGBM per database", "logistic_pooled": "Logistic (L2)", "tabpfn_v2": "TabPFN v2",
             "logistic_l1": "Logistic (L1)", "logistic_enet": "Logistic (elastic net)", "cart": "Decision tree",
             "random_forest": "Random forest", "extra_trees": "Extra trees", "naive_bayes": "Naive Bayes",
             "knn": "k-nearest neighbours", "mlp": "Multilayer perceptron", "kdigo": "KDIGO stage",
             "creatinine": "Creatinine", "neg_map": "MAP (lower)", "neg_rox": "ROX (lower)", "shock_index": "Shock index",
             "lactate": "Lactate", "potassium": "Potassium", "neg_platelets": "Platelets (lower)", "sapsii": "SAPS II",
             "oasis": "OASIS"}

INTERNAL = re.compile(r"\b(MUC32\w*|E[0-2]|R128|L24x|A[1-5]|B[23]|C[1-4]|F[1-9]|V[1-5]b?|V3x|twin_\w+|lgbm_\w+)\b")
GREEK = re.compile("[Ͱ-Ͽἀ-῿]")


def setup() -> None:
    """The rcParams every figure uses."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": PT,
        "axes.titlesize": PT, "axes.labelsize": PT, "xtick.labelsize": PT, "ytick.labelsize": PT,
        "legend.fontsize": PT, "legend.title_fontsize": PT, "figure.titlesize": PT,
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 3.0,
        "axes.linewidth": LW_MIN, "axes.edgecolor": AXIS, "axes.labelcolor": TEXT, "text.color": TEXT,
        "axes.spines.top": False, "axes.spines.right": False, "axes.labelpad": 2.0,
        "xtick.major.width": LW_MIN, "ytick.major.width": LW_MIN, "xtick.minor.width": LW_MIN,
        "ytick.minor.width": LW_MIN, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "xtick.color": AXIS,
        "ytick.color": AXIS, "xtick.labelcolor": TEXT, "ytick.labelcolor": TEXT, "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": LW_DATA, "lines.markersize": MARKER, "lines.markeredgewidth": LW_MIN,
        "patch.linewidth": LW_MIN, "hatch.linewidth": LW_MIN, "grid.linewidth": LW_MIN, "boxplot.boxprops.linewidth":
        LW_MIN, "boxplot.whiskerprops.linewidth": LW_MIN, "boxplot.capprops.linewidth": LW_MIN,
        "boxplot.medianprops.linewidth": LW_MIN, "errorbar.capsize": 0,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.2,
        "legend.labelspacing": 0.3,
        "axes.unicode_minus": True, "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold", "mathtext.sf": FONT, "mathtext.default": "regular",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "didymos",
        "savefig.dpi": 600, "figure.dpi": 100, "savefig.transparent": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.compression": 6,
    })


class Canvas:
    """A figure of ``width`` x ``height`` mm whose axes are placed in millimetres from the top-left corner."""

    def __init__(self, width: float = FULL, height: float = 150.0):
        import matplotlib.pyplot as plt

        setup()
        self.w, self.h = float(width), float(height)
        self.fig = plt.figure(figsize=(self.w * MM, self.h * MM))
        self.letters: list[str] = []

    def axes(self, left: float, top: float, width: float, height: float, **kw):
        """Axes whose plotting area spans [left, left + width] x [top, top + height] mm (top-left origin)."""
        return self.fig.add_axes((left / self.w, 1 - (top + height) / self.h, width / self.w, height / self.h), **kw)

    def letter(self, x: float, y: float, letter: str, title: str | None = None):
        """A panel letter (bold) at (x, y) mm, its top-left corner, with an optional short subtitle beside it."""
        self.letters.append(letter)
        t = self.fig.text(x / self.w, 1 - y / self.h, letter, fontweight="bold", va="top", ha="left",
                          fontsize=PT, gid=f"letter_{letter}")
        if title:
            self.fig.text((x + 3.2) / self.w, 1 - y / self.h, title, va="top", ha="left", fontsize=PT, color=TEXT)
        return t

    def text(self, x: float, y: float, s: str, **kw):
        return self.fig.text(x / self.w, 1 - y / self.h, s, fontsize=PT, **kw)


def _p_round_up(p: float) -> str:
    """Two significant figures, rounded up (for bounds)."""
    e = int(np.floor(np.log10(p))) - 1
    v = np.ceil(p / 10 ** e - 1e-9) * 10 ** e
    return f"{v:.{max(0, -e)}f}"


def p_value(p: float | None, bound: bool = False) -> str:
    """The P value alone, for a column headed '$P$': '0.012', '<0.001', '0.41'; '≤0.0014' when ``p`` is a
    bound (no bootstrap resample on the other side of the null)."""
    if p is None:
        return "–"
    if p < 0.001:
        return "<0.001"
    if bound:
        return "≤" + _p_round_up(p)
    return f"{p:.2f}" if p >= 0.1 else f"{float(f'{p:.2g}'):g}"


def p_text(p: float | None, bound: bool = False) -> str:
    """'$P$ = 0.012', '$P$ < 0.001', '$P$ ≤ 0.0014' (mathtext italic P in Arial)."""
    if p is None:
        return ""
    v = p_value(p, bound)
    it = r"$\mathit{P}$"
    if v.startswith("<") or v.startswith("≤"):
        return f"{it} {v[0]} {v[1:]}"
    return f"{it} = {v}"


def num(x: float | None, d: int = 3, sign: bool = False) -> str:
    """A number with a true minus sign (and '+' when ``sign``)."""
    if x is None or not np.isfinite(x):
        return "–"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", MINUS)


def ci_text(lo: float | None, hi: float | None, d: int = 3) -> str:
    return f"{num(lo, d)} to {num(hi, d)}"


def clean(ax, left: bool = True, bottom: bool = True) -> None:
    """Spines left and bottom only (as kept), outward ticks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.tick_params(left=False, labelleft=False)
    if not bottom:
        ax.tick_params(bottom=False, labelbottom=False)


def refline(ax, value: float, axis: str = "x", **kw) -> None:
    kw = {"color": REF, "lw": LW_MIN, "ls": (0, (3, 2)), "zorder": 0, **kw}
    (ax.axvline if axis == "x" else ax.axhline)(value, **kw)


# ---- export --------------------------------------------------------------------------------------------------------

def git_sha() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - outside a checkout
        return "unknown"


def save(canvas: Canvas, name: str, out: Path, preview: Path | None = None, svg: bool = True) -> list[Path]:
    """PDF (and SVG) into ``out``; a 600-dpi PNG preview into ``preview``. Deterministic: no creation date."""
    out.mkdir(parents=True, exist_ok=True)
    files = []
    meta = {"Creator": "icutwin tools/paper_figures.py", "CreationDate": None, "Title": name}   # sha: manifest.json
    f = out / f"{name}.pdf"
    canvas.fig.savefig(f, format="pdf", metadata=meta)
    files.append(f)
    if svg:
        import io

        f = out / f"{name}.svg"
        buf = io.StringIO()
        canvas.fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": meta["Creator"]})
        with open(f, "w", encoding="utf-8", newline="\n") as fh:      # the repository is LF-only
            fh.write(buf.getvalue().replace("\r\n", "\n"))
        files.append(f)
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
        f = preview / f"{name}.png"
        canvas.fig.savefig(f, format="png", dpi=600, metadata={"Software": None})
        files.append(f)
    return files


# ---- checks ------------------------------------------------------------------------------------------------------

def _texts(fig):
    """The texts that are drawn: visible, non-empty, and for tick labels only those inside the view interval."""
    from matplotlib.text import Text

    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            eps = 1e-9 * max(1.0, abs(hi - lo))
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if not (lo - eps <= tick.get_loc() <= hi + eps):
                    hidden.update({id(tick.label1), id(tick.label2)})
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and id(t) not in hidden]


def check(canvas: Canvas, allow_overlap: tuple = ()) -> list[str]:
    """Every violation of the design system in a finished figure (an empty list when it complies)."""
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, issues = canvas.fig, []
    w, h = (fig.get_size_inches() / MM)
    if not (abs(w - FULL) < 0.5 or abs(w - HALF) < 0.5):
        issues.append(f"width {w:.1f} mm is neither {FULL} nor {HALF}")
    if h > MAX_H + 0.5:
        issues.append(f"height {h:.1f} mm over {MAX_H}")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = _texts(fig)
    for t in texts:
        s = t.get_text()
        if abs(t.get_fontsize() - PT) > 1e-6:
            issues.append(f"font size {t.get_fontsize()} for {s[:30]!r}")
        fam = t.get_fontname()
        if fam != FONT:
            issues.append(f"font {fam} for {s[:30]!r}")
        if GREEK.search(s):
            issues.append(f"Greek glyph in {s[:30]!r}")
        if INTERNAL.search(s.replace("$", "")):
            issues.append(f"internal code in {s[:40]!r}")
    for ln in fig.findobj(Line2D):
        if not ln.get_visible() or ln.get_linestyle() in ("None", "none", "", " "):
            if ln.get_visible() and ln.get_marker() not in (None, "None", "none", "", " ") and \
                    ln.get_markeredgewidth() and 0 < ln.get_markeredgewidth() < LW_MIN - 1e-6:
                issues.append(f"marker edge {ln.get_markeredgewidth()} pt")
            continue
        if 0 < ln.get_linewidth() < LW_MIN - 1e-6:
            issues.append(f"line {ln.get_linewidth()} pt ({ln.get_label()})")
    for c in fig.findobj(Collection):
        if not c.get_visible() or not len(c.get_paths()):
            continue
        lws = np.atleast_1d(c.get_linewidths())
        ec = c.get_edgecolors()
        has_edge = len(ec) and np.any(np.asarray(ec)[:, 3] > 0) if np.ndim(ec) == 2 else False
        if has_edge and np.any((lws > 0) & (lws < LW_MIN - 1e-6)):
            issues.append(f"collection line {lws.min()} pt")
    for p in fig.findobj(Patch):
        if not p.get_visible() or p is fig.patch:
            continue
        ec = p.get_edgecolor()
        if ec is not None and len(ec) == 4 and ec[3] > 0 and 0 < p.get_linewidth() < LW_MIN - 1e-6:
            issues.append(f"patch edge {p.get_linewidth()} pt")
    fb = fig.bbox
    tol = 0.3 * MM * fig.dpi                       # 0.3 mm
    for t in texts:
        bb = t.get_window_extent(renderer)
        if bb.x0 < fb.x0 - tol or bb.x1 > fb.x1 + tol or bb.y0 < fb.y0 - tol or bb.y1 > fb.y1 + tol:
            issues.append(f"outside the canvas: {t.get_text()[:30]!r}")
    boxes = []
    for t in texts:
        if t.get_gid() in allow_overlap or str(t.get_gid()).startswith("watermark"):
            continue
        bb = t.get_window_extent(renderer)
        if bb.width > 0 and bb.height > 0:
            boxes.append((t, bb.expanded(0.98, 0.9)))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                a, b = boxes[i][0].get_text(), boxes[j][0].get_text()
                issues.append(f"overlap: {a[:25]!r} / {b[:25]!r}")
    if canvas.letters != [chr(ord("a") + i) for i in range(len(canvas.letters))]:
        issues.append(f"panel letters out of order: {canvas.letters}")
    return issues


# ---- colour-blind and greyscale simulation -----------------------------------------------------------------------

MACHADO = {  # Machado, Oliveira & Fernandes (2009), severity 1.0, linear RGB
    "protanopia": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}


def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate(rgb: np.ndarray, kind: str) -> np.ndarray:
    """``rgb`` (..., 3) in [0, 1] as seen with ``kind`` (a Machado matrix, or 'greyscale')."""
    if kind == "greyscale":
        y = _lin(rgb) @ np.array([0.2126, 0.7152, 0.0722])
        return np.repeat(_srgb(y)[..., None], 3, axis=-1)
    return _srgb(_lin(rgb) @ MACHADO[kind].T)


def simulate_png(png: Path, out_dir: Path) -> list[Path]:
    from PIL import Image

    img = np.asarray(Image.open(png).convert("RGB"), float) / 255
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for kind in ("protanopia", "deuteranopia", "tritanopia", "greyscale"):
        f = out_dir / f"{png.stem}_{kind}.png"
        Image.fromarray((simulate(img, kind) * 255 + 0.5).astype(np.uint8)).save(f)
        files.append(f)
    return files


def _lab(rgb: np.ndarray) -> np.ndarray:
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = _lin(rgb) @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def distinguishable(colours: dict, min_delta: float = 10.0) -> dict:
    """The smallest CIE76 colour difference between any two of ``colours`` (name -> hex), normally and in each
    simulation; pairs below ``min_delta``."""
    from matplotlib.colors import to_rgb

    names = list(colours)
    rgb = np.array([to_rgb(colours[n]) for n in names])
    out = {}
    for kind in ("normal", "protanopia", "deuteranopia", "tritanopia", "greyscale"):
        lab = _lab(rgb if kind == "normal" else simulate(rgb, kind))
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = float(np.linalg.norm(lab[i] - lab[j]))
                pairs.append((d, names[i], names[j]))
        pairs.sort()
        out[kind] = {"min": pairs[0][0], "below": [(n1, n2, round(d, 1)) for d, n1, n2 in pairs if d < min_delta]}
    return out


# ---- source data ------------------------------------------------------------------------------------------------

class SourceData:
    """The numbers a figure plots, panel by panel, with their source files; written as one CSV per figure."""

    def __init__(self, figure: str):
        self.figure = figure
        self.rows: list[dict] = []
        self.inputs: dict[str, str] = {}

    def add(self, panel: str, series: str, **values) -> None:
        self.rows.append({"panel": panel, "series": series, **values})

    def source(self, *paths: Path) -> None:
        for p in paths:
            p = Path(p)
            if p.exists() and str(p) not in self.inputs:
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                self.inputs[str(p).replace("\\", "/")] = h.hexdigest()

    def write(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        keys = ["panel", "series"]
        for r in self.rows:
            keys += [k for k in r if k not in keys]
        f = out_dir / f"{self.figure}.csv"
        with open(f, "w", newline="\n", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
            w.writeheader()
            for r in self.rows:
                w.writerow({k: _cell(r.get(k)) for k in keys})
        return f


def _cell(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return repr(x) if np.isfinite(x) else ""
    if isinstance(x, (list, tuple)):
        return ";".join(str(_cell(v)) for v in x)
    return x
