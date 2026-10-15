"""Fig. 1: Didymos, a generative model of intensive care trained on seven international databases.

a, from records to simulated futures: seven databases are harmonized and tokenized, Didymos (a decoder) is trained
once, and conditioned on a patient's record it draws possible futures, read as two estimates and used by the tools.
Tokens and trajectories in a are illustrative. b, the seven databases, with their stays by split. c, constrained
sampling: the grammar and the database's support remove tokens before a draw (probabilities illustrative). Panels a and
c are the authors' schematic artwork (GPT-image) when st.USE_ARTWORK lists them; the code draws them otherwise.
"""

from __future__ import annotations

import numpy as np

import paper_figstyle as st
from paper_figdata import Inputs

W, H = st.FULL, 148.0
ORDER = ("eicu", "mimic", "hirid", "nwicu", "sicdb", "aumc", "inspire")


def _box(ax, x, y, w, h, fill=st.SLATE[100], edge=st.SLATE[400]):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", facecolor=fill,
                                edgecolor=edge, lw=1.0))


def _arrow(ax, x0, x1, y):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=6, color=st.SLATE[700], lw=1.0))


def _chips(ax, x, y, items, h=3.6):
    """Token chips from (x, y), left to right; ``items`` (text, fill, edge). Returns the end x."""
    for text, fill, edge in items:
        w = 1.32 * len(text) + 2.2
        _box(ax, x, y - h / 2, w, h, fill, edge)
        ax.text(x + w / 2, y, text, ha="center", va="center")
        x += w + 1.0
    return x


def pipeline(cv: st.Canvas, top: float) -> None:
    ax = cv.axes(0, top, W, 58)
    ax.set_xlim(0, W)
    ax.set_ylim(58, 0)
    ax.axis("off")
    # row 1: data, harmonization, the token sequence
    _box(ax, 0.5, 3, 26, 20)
    for i, (t, b) in enumerate((("Seven ICU", True), ("databases", True), ("409,297 stays", False),
                                ("five countries", False))):
        ax.text(13.5, 6.5 + i * 3.8, t, ha="center", va="center", fontweight="bold" if b else "normal")
    _arrow(ax, 27.0, 30.5, 13)
    _box(ax, 30.5, 3, 27, 20)
    for i, (t, b) in enumerate((("Harmonized", True), ("and tokenized", True), ("CLIF, then MEDS", False),
                                ("25,601 tokens", False))):
        ax.text(44.0, 6.5 + i * 3.8, t, ha="center", va="center", fontweight="bold" if b else "normal")
    _arrow(ax, 58.0, 61.5, 13)
    pre, ev, gap, tr = (st.SLATE[100], st.SLATE[300]), (st.BLUE[100], st.BLUE[500]), ("white", st.SLATE[300]), \
        (st.BLUE[200], st.BLUE[500])
    _chips(ax, 62.0, 6.0, [("[BOS]", *pre), ("eICU", *pre), ("site 73", *pre), ("female", *pre),
                           ("age 70–74", *pre), ("medical ICU", *pre), ("timeline", *pre)])
    _chips(ax, 62.0, 11.5, [("HR q6", *ev), ("MAP q3", *ev), ("SpO2 q8", *ev), ("5–15 min", *gap),
                            ("norepinephrine start", *tr), ("rate q2", *tr), ("1–2 h", *gap), ("lactate q9", *ev)])
    # row 2: the model, simulated futures, the two estimates, the tools
    _box(ax, 0.5, 30, 40, 26, st.DIDYMOS_LIGHT, st.DIDYMOS)
    for k in range(12):
        ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
            (3.0, 32.6 + k * 1.75), 9.5, 1.2, facecolor=st.BLUE[500] if k % 2 == 0 else st.BLUE[300],
            edgecolor="none"))
    for i, (t, b) in enumerate((("Didymos 1.0", True), ("decoder, 12 layers", False), ("123M parameters", False),
                                ("context 32,768 tokens", False), ("trained once,", False),
                                ("privacy-bounded", False))):
        ax.text(15.0, 33.6 + i * 3.7, t, ha="left", va="center", color=st.DIDYMOS if b else st.TEXT,
                fontweight="bold" if b else "normal")
    _arrow(ax, 41.0, 44.5, 43)
    _arrow(ax, 104.5, 108.0, 43)
    _box(ax, 108.0, 30, 40, 26)
    for i, (t, b) in enumerate((("Sampled estimate", True), ("share of futures with the event", False),
                                ("Hazard estimate", True), ("the model's step hazards,", False),
                                ("summed along each future", False))):
        ax.text(128.0, 34.0 + i * 4.3, t, ha="center", va="center", fontweight="bold" if b else "normal")
    _arrow(ax, 148.5, 152.0, 43)
    _box(ax, 152.0, 30, 27.5, 26, "white", st.DIDYMOS)
    for i, t in enumerate(("Forecasts", "What-if", "Synthetic cohorts", "Explanations", "Trial design")):
        ax.text(165.75, 34.0 + i * 4.3, t, ha="center", va="center", color=st.DIDYMOS)


def _smooth(x, k: int = 5):
    return np.convolve(np.pad(x, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]


def futures(cv: st.Canvas, x: float, y: float, w: float, h: float) -> None:
    """Illustrative simulated futures after the prediction time (smoothed random walks with a fixed seed)."""
    ax = cv.axes(x, y, w, h)
    rng = np.random.default_rng(11)
    th = np.linspace(-24, 0, 97)
    hist = 72 + 4 * np.sin(th / 6) + _smooth(np.cumsum(rng.normal(0, 0.35, len(th))))
    ax.plot(th, hist, color=st.SLATE[900], lw=1.5)
    tf = np.linspace(0, 24, 97)
    for k in range(10):
        drift = (k - 4.5) * 0.012
        path = hist[-1] + _smooth(np.cumsum(rng.normal(drift, 0.4, len(tf))), 4)
        path -= path[0] - hist[-1]
        ax.plot(tf, path, color=st.BLUE[700] if k == 0 else st.BLUE[300], lw=1.5 if k == 0 else 1.0,
                zorder=3 if k == 0 else 2)
        if k in (2, 6, 8):
            j = int(rng.integers(40, 90))
            ax.plot(tf[j], path[j], ls="none", marker="o", ms=3.8, color=st.VERMILLION, zorder=4, **st.RING)
    ax.axvline(0, color=st.SLATE[500], lw=1.0, ls=(0, (3, 2)))
    ax.set_xlim(-24, 24)
    ax.set_xticks([-24, 0, 24])
    ax.set_xticklabels([f"{st.MINUS}24 h", "now", "+24 h"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    cv.text(x, y - 1.2, "History", ha="left", va="bottom")
    cv.text(x + w, y - 1.2, "Simulated futures", ha="right", va="bottom", color=st.DIDYMOS)
    cv.text(x + w, y + h + 7.5, "● an event in a future", ha="right", va="center", color=st.FLAG)


def _sampling(cv: st.Canvas, y0: float, H: float) -> None:
    """Panel c drawn in code: the fallback when its artwork is not in st.USE_ARTWORK."""
    from matplotlib.patches import Patch

    toks = (("MAP q4", 0.22, "allowed"), ("HR q5", 0.18, "allowed"), ("5–15 min", 0.14, "allowed"),
            ("norepinephrine rate up", 0.12, "grammar"), ("ventilator PEEP", 0.10, "support"),
            ("lactate q3", 0.08, "allowed"), ("[END] block", 0.07, "grammar"), ("SpO2 q9", 0.05, "allowed"))
    cr = 3.6
    cx = cv.axes(126, y0 + 7.0, 30, len(toks) * cr)
    cx.set_ylim(len(toks) * cr, 0)
    cx.yaxis.set_visible(False)
    cx.spines["left"].set_visible(False)
    allowed = sum(p for _t, p, s in toks if s == "allowed")
    for i, (t, p, s) in enumerate(toks):
        yc = (i + 0.5) * cr
        cv.text(124.0, y0 + 7.0 + yc, t, ha="right", va="center")
        if s == "allowed":
            cx.barh(yc, p / allowed, height=cr * 0.62, color=st.DIDYMOS, edgecolor="none")
            cx.barh(yc, p, height=cr * 0.62, color="none", edgecolor=st.SLATE[900], lw=1.0)
        else:
            cx.barh(yc, p, height=cr * 0.62, color="white", edgecolor=st.SLATE[300], lw=1.0, hatch="////")
            cx.text(p + 0.02, yc, "grammar" if s == "grammar" else "not recorded here", ha="left", va="center",
                    color=st.VERMILLION)
    cx.set_xlim(0, 0.45)
    cx.set_xticks([0, 0.2, 0.4])
    cx.set_xticklabels(["0", "0.2", "0.4"])
    cx.set_xlabel("Next-token probability (illustrative)")
    cv.fig.legend([Patch(facecolor="white", edgecolor=st.SLATE[900], lw=1.0), Patch(facecolor=st.BLUE[500]),
                   Patch(facecolor="white", edgecolor=st.SLATE[300], hatch="////", lw=1.0)],
                  ["Model", "After masking, renormalized", "Masked"], loc="upper left", ncol=3,
                  bbox_to_anchor=(94 / W, 1 - (y0 + 7.0 + len(toks) * cr + 9.0) / H), columnspacing=0.9,
                  handletextpad=0.4, borderaxespad=0, handlelength=1.2)



def figure(inp: Inputs) -> tuple[st.Canvas, st.SourceData]:
    from matplotlib.patches import Patch

    sd = st.SourceData("fig1")
    t1 = inp.json("materials/table1.json", sd=sd)
    art_a, art_c = "fig1a" in st.USE_ARTWORK, "fig1c" in st.USE_ARTWORK
    y0 = 5.0 + st.artwork_height("fig1a", W) + 3.0 if art_a else 70.0          # panels b and c
    hc = st.artwork_height("fig1c", 86.0) if art_c else 0.0
    row = max(56.0, 5.0 + hc - 0.5) if art_c else 56.0                         # 1.5 mm below c's artwork
    H = y0 + row + 2.0
    cv = st.Canvas(W, H)
    cv.letter(0, 2.0, "a", "From records to simulated futures")
    if art_a:
        st.artwork(cv, sd, "fig1a", 0, 5.0, W, "a")
    else:
        pipeline(cv, 5.0)
        futures(cv, 47.0, 38.0, 56.0, 19.0)

    # ---- b: the databases --------------------------------------------------------------------------------------
    cv.letter(0, y0, "b", "Seven international databases")
    rh = (hc - 14.7) / len(ORDER) if art_c else 4.6       # rows spread so the legend ends level with c's artwork
    bx = cv.axes(36, y0 + 7.0, 40, len(ORDER) * rh)
    bx.set_ylim(len(ORDER) * rh, 0)
    bx.yaxis.set_visible(False)
    bx.spines["left"].set_visible(False)
    for i, d in enumerate(ORDER):
        x = t1["datasets"][d]
        sp = x["splits"]
        yc = (i + 0.5) * rh
        left = 0.0
        for key, col in (("train", st.BLUE[200]), ("tuning", st.SLATE[400]), ("held_out", st.BLUE[500])):
            bx.barh(yc, sp[key] / 1000, left=left, height=min(rh * 0.62, 2.9), color=col, edgecolor="none")
            left += sp[key] / 1000
        bx.text(left + 3, yc, f"{x['stays']:,}", ha="left", va="center")
        cv.text(34.0, y0 + 7.0 + yc, f"{st.DATASET_SHORT[d]} ({st.COUNTRY[d]})", ha="right", va="center")
        sd.add("b", d, stays=x["stays"], train=sp["train"], tuning=sp["tuning"], held_out=sp["held_out"],
               country=st.COUNTRY[d])
    bx.set_xlim(0, 230)
    bx.set_xticks([0, 50, 100, 150, 200])
    bx.set_xlabel("ICU stays (thousands)")
    cv.fig.legend([Patch(facecolor=st.BLUE[200]), Patch(facecolor=st.SLATE[400]), Patch(facecolor=st.BLUE[500])],
                  ["Train", "Tuning", "Held-out"], loc="upper left", ncol=3,
                  bbox_to_anchor=(36 / W, 1 - (y0 + 7.0 + len(ORDER) * rh + 9.0) / H), columnspacing=1.0,
                  handletextpad=0.4, borderaxespad=0, handlelength=1.2)

    # ---- c: constrained sampling ---------------------------------------------------------------------------------
    cv.letter(94, y0, "c", "Constrained sampling")
    if art_c:
        st.artwork(cv, sd, "fig1c", 94, y0 + 5.0, 86.0, "c")
    else:
        _sampling(cv, y0, H)

    return cv, sd
