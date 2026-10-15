"""The paper's supplementary figures, in the main figures' design system
(``tools/paper_figstyle.py``), from the local aggregates. Built by ``tools/paper_figures.py`` as ``supp3`` to
``supp16``, numbered in the order of the main figures; Supplementary Figs. 1-2 are schematics rendered from Mermaid
(``tools/paper_diagrams.py``).

With Fig. 1: S3 data and evaluation sets. With Fig. 2: S4 every predictor's AUROC; S5 ranks across endpoints; S6 AUROC
against the number of simulated futures, every endpoint; S7 per-database AUROC; S8 every scored endpoint; S9 the same
by clinical family, and label efficiency; S10 regression and time to event; S11 site prompts and leaving a database
out; S12 subgroups. With Fig. 3: S13 the recalibration maps. With Fig. 7: S14 the toolkit's conformance; S15
similarity; S16 the trial simulations' checks. (The planned what-if in raw units, attention diagnosis and prompted
privacy by database are in Figs. 4b, 6 and 5a; calibration by database is in Fig. 3b.) Single-panel figures carry no
panel letter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import paper_figstyle as st
from paper_figdata import CLINICAL, E2_PROGRAMS, GROUP, LEARNED, ROWS, Inputs, find

W = st.FULL
BEDSIDE = ("kdigo", "creatinine", "neg_map", "neg_rox", "shock_index", "lactate", "potassium", "neg_platelets")
SCORE_NAME = st.PREDICTOR


def _handle(**kw):
    from matplotlib.lines import Line2D

    return Line2D([], [], **{"ls": "none", **kw})


def _legend(cv: st.Canvas, handles, labels, x: float, y: float, ncol: int = 1, **kw):
    """A legend whose top-left corner is at (x, y) mm."""
    kw = {"loc": "upper left", "ncol": ncol, "bbox_to_anchor": (x / cv.w, 1 - y / cv.h), "columnspacing": 1.2,
          "handletextpad": 0.4, "borderaxespad": 0, **kw}
    return cv.fig.legend(handles, labels, **kw)


def _blocks(blocks, cols: int = 5, left: float = 13.0, top: float = 2.0, px: float = 33.5, py: float = 31.0,
            h: float = 22.0):
    """Small multiples in blocks, each under a bold header (none when its name is empty): the panels (item, x0, y0,
    column), the headers (name, y) and the bottom of the last axes, in mm."""
    pos, heads, y = [], [], top
    for name, items in blocks:
        if name:
            heads.append((name, y))
            y += 8.0
        else:
            y += 4.0
        for i, it in enumerate(items):
            pos.append((it, left + (i % cols) * px, y + (i // cols) * py, i % cols))
        y += ((len(items) - 1) // cols) * py + h + 7.0
    return pos, heads, y - 7.0


def _shared_labels(cv: st.Canvas, x_label: str, y_label: str, left: float, right: float, spans, bottom: float,
                   offset: float = 9.0) -> None:
    """One x label under a grid of small multiples, and a rotated y label beside each block of rows (spans: the
    blocks' (top, bottom) in mm)."""
    cv.text((left + right) / 2, bottom + 5.0, x_label, ha="center", va="top")
    for top, bot in spans:
        cv.text(left - offset, (top + bot) / 2, y_label, ha="center", va="center", rotation=90)


def _spans(pos, h: float = 22.0) -> list[tuple[float, float]]:
    """The (top, bottom) of each block of small multiples: runs of rows separated by a header."""
    tops = sorted({y0 for _it, _x, y0, _c in pos})
    spans, start, prev = [], tops[0], tops[0]
    for y in tops[1:]:
        if y - prev > 31.5:
            spans.append((start, prev + h))
            start = y
        prev = y
    spans.append((start, prev + h))
    return spans


def _rows_axes(cv: st.Canvas, left: float, top: float, width: float, n: int, rh: float):
    """Axes for n rows of rh mm starting at top (row i centred at top + (i + 0.5) rh); y in row units."""
    ax = cv.axes(left, top, width, n * rh)
    ax.set_ylim(n - 0.5, -0.5)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    return ax


def _row_y(top: float, rh: float, i: float) -> float:
    return top + (i + 0.5) * rh


def _endpoint(sub: str, prog: str) -> str:
    return st.ENDPOINT_SHORT[prog]


# ---- S1: data processing (schematic artwork) ------------------------------------------------------------------------

DIAGRAMS = Path(__file__).resolve().parents[1] / "docs" / "paper" / "manuscript" / "diagrams"


def data_processing(inp: Inputs):
    """From seven source databases to model-ready sequences (a), and one stay as one token sequence (b): the artwork
    (GPT-image), drawn from the Mermaid sources in docs/paper/manuscript/diagrams/."""
    sd = st.SourceData("supp1")
    ha, hb = st.artwork_height("supp1a", W), st.artwork_height("supp1b", W)
    yb = 5.0 + ha + 4.0
    cv = st.Canvas(W, yb + 5.0 + hb + 1.0)
    cv.letter(0, 2.0, "a", "Data processing")
    st.artwork(cv, sd, "supp1a", 0, 5.0, W, "a")
    cv.letter(0, yb, "b", "One stay as one token sequence")
    st.artwork(cv, sd, "supp1b", 0, yb + 5.0, W, "b")
    sd.source(*sorted(DIAGRAMS.glob("supp1*.mmd")))
    return cv, sd


# ---- S2: the network and constrained simulation (schematic artwork) --------------------------------------------------

def architecture(inp: Inputs):
    """The Didymos network (a), and constrained simulation with the two estimates (b), side by side at one height: the
    artwork (GPT-image), drawn from the Mermaid sources in docs/paper/manuscript/diagrams/."""
    sd = st.SourceData("supp2")
    ra, rb = st.artwork_height("supp2a", 1.0), st.artwork_height("supp2b", 1.0)    # height per mm of width
    gap = 4.0
    wa = (W - gap) * rb / (ra + rb)
    wb = W - gap - wa
    cv = st.Canvas(W, 5.0 + wa * ra + 1.0)
    cv.letter(0, 2.0, "a", "The Didymos network")
    st.artwork(cv, sd, "supp2a", 0, 5.0, wa, "a")
    cv.letter(wa + gap, 2.0, "b", "Constrained simulation and the two estimates")
    st.artwork(cv, sd, "supp2b", wa + gap, 5.0, wb, "b")
    sd.source(*sorted(DIAGRAMS.glob("supp2*.mmd")))
    return cv, sd


# ---- S3: data and evaluation sets ------------------------------------------------------------------------------------

def evaluation_sets(inp: Inputs):
    """The flow from the seven databases to the splits and the evaluation's anchor sets (counts from the materials):
    the artwork (GPT-image) when it is in use, the counts recorded in the source data either way."""
    sd = st.SourceData("supp3")
    t1 = inp.json("materials/table1.json", sd=sd)["datasets"]
    split = {s: sum(d["splits"][s] for d in t1.values()) for s in ("train", "tuning", "held_out")}
    stays = sum(d["stays"] for d in t1.values())
    a1 = {d: inp.json(f"a1/test_{d}/a1.json", sd=sd)["anchors"] for d in ("R", "L24", "L48")}
    f1 = {s: inp.json(f"f1/{s}/f2.json", sd=sd)["anchors"] for s in ("R128", "L24x")}
    deaths = inp.json("f1/L24x/f2.json")["f1"]["hospital_death@0"]["twin_128"]["positives"]
    per_db_test, per_db_train_rows, per_db_recal = 2000, 5000, 100      # the protocol's set sizes (data.md)
    if "supp3" in st.USE_ARTWORK:                                       # the artwork shows these counts
        cv = st.Canvas(W, st.artwork_height("supp3", W) + 1.0)
        st.artwork(cv, sd, "supp3", 0, 0.5, W, "flow")
    else:
        cv = _evaluation_sets_drawn(stays, split, a1, f1, deaths, per_db_test, per_db_train_rows, per_db_recal)
    sd.add("flow", "stays", n=stays, **{f"{k}_stays": v for k, v in split.items()})
    for d, n in a1.items():
        sd.add("flow", "pre-specified anchors", design=d, anchors=n)
    for s, n in f1.items():
        sd.add("flow", "follow-up anchors", design=s, anchors=n)
    sd.add("flow", "case-cohort in-hospital deaths", design="L24x", deaths=deaths)
    return cv, sd


def _evaluation_sets_drawn(stays, split, a1, f1, deaths, per_db_test, per_db_train_rows, per_db_recal):
    """S3 drawn in code: the fallback when its artwork is not in st.USE_ARTWORK."""
    from matplotlib.patches import FancyBboxPatch

    H = 112.0
    cv = st.Canvas(W, H)
    ax = cv.axes(0, 0, W, H)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    def box(x, y, w, h, lines, fill=st.FILL, edge=st.AXIS):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", facecolor=fill,
                                    edgecolor=edge, lw=1.0))
        lh = 3.6
        y0 = y + h / 2 - (len(lines) - 1) * lh / 2
        for i, s in enumerate(lines):
            ax.text(x + w / 2, y0 + i * lh, s, ha="center", va="center", fontweight="bold" if i == 0 else "normal",
                    color=st.TEXT if i == 0 else st.TEXT2)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": st.AXIS, "shrinkA": 0, "shrinkB": 0,
                                "mutation_scale": 6})

    def line(xs, ys):
        ax.plot(xs, ys, color=st.AXIS, lw=1.0, solid_capstyle="butt")

    bw, gap = 56.4, 4.5
    xs = [0.8, 0.8 + bw + gap, 0.8 + 2 * (bw + gap)]
    mid = [x + bw / 2 for x in xs]
    pct = {s: 100 * v / stays for s, v in split.items()}
    box(45, 2, 90, 13, ["Seven ICU databases in five countries",
                        f"{stays:,} adult ICU stays, harmonized and tokenized"])
    line([90, 90], [15, 18.5])
    line([mid[0], mid[2]], [18.5, 18.5])
    for m in mid:
        arrow(m, 18.5, m, 23)
    box(xs[0], 23, bw, 19, [f"Train: {split['train']:,} stays ({pct['train']:.0f}%)",
                            "Didymos's training (no task labels);",
                            f"comparators' training: up to {per_db_train_rows:,} per database"])
    box(xs[1], 23, bw, 19, [f"Tuning: {split['tuning']:,} stays ({pct['tuning']:.0f}%)",
                            "model selection, early stopping;",
                            f"recalibration ({per_db_recal} rolling anchors per database)"])
    box(xs[2], 23, bw, 19, [f"Held-out: {split['held_out']:,} stays ({pct['held_out']:.0f}%)",
                            "evaluation only: a test store of",
                            f"{per_db_test:,} stays per database ({7 * per_db_test:,})"])
    line([mid[2], mid[2]], [42, 46.5])
    line([mid[0], mid[2]], [46.5, 46.5])
    for m in mid:
        arrow(m, 46.5, m, 51)
    box(xs[0], 51, bw, 23, [f"Rolling anchors: {a1['R']:,}", "500 per database at random ICU hours,",
                            "enriched for events (inclusion weights);", "32 simulated futures each"])
    box(xs[1], 51, bw, 23, [f"24-hour landmark: {a1['L24']:,}", "up to 500 per database, in the ICU",
                            "and alive at 24 h;", "32 simulated futures each"])
    box(xs[2], 51, bw, 23, [f"48-hour landmark: {a1['L48']:,}", "up to 500 per database, in the ICU",
                            "and alive at 48 h;", "32 simulated futures each"])
    for m in mid[:2]:
        arrow(m, 74, m, 84)
    box(xs[0], 84, bw, 23, [f"Rolling anchors, follow-up: {f1['R128']:,}", "150 per database of the rolling anchors;",
                            "128 simulated futures each"], fill=st.DIDYMOS_LIGHT, edge=st.DIDYMOS)
    box(xs[1], 84, bw, 23, [f"24-hour landmark, follow-up: {f1['L24x']:,}",
                            f"every in-hospital death ({deaths}) and {f1['L24x'] - deaths} others,",
                            "weighted (a case-cohort sample);", "128 simulated futures each"],
        fill=st.DIDYMOS_LIGHT, edge=st.DIDYMOS)
    return cv


# ---- S4: every predictor's AUROC -------------------------------------------------------------------------------------

def predictor_matrix(inp: Inputs):
    """Every predictor's AUROC on the same anchors (amendment 10), with Didymos's hazard estimate where it applies."""
    import matplotlib.colors as mcolors
    from matplotlib import colormaps
    from matplotlib.patches import Rectangle

    sd = st.SourceData("supp4")
    comp = inp.json("f10/comparators.json", sd=sd)
    val = {}
    for sub, prog in ROWS:
        for n, x in comp["subsets"][sub][prog]["models"].items():
            val[(n, sub, prog)] = x
        if prog in E2_PROGRAMS[sub]:
            val[("twin_e2", sub, prog)] = inp.json(f"f1/{sub}/f2.json", sd=sd)["f2"][prog]["estimators"]["e2"]
    groups = [("twin_e2", "twin_128", "twin_probe"), LEARNED, CLINICAL, BEDSIDE]
    groups = [[n for n in g if any((n, s, p) in val for s, p in ROWS)] for g in groups]
    rh, gap, cw, cgap = 3.9, 1.6, 10.2, 2.0
    left, top = 43.0, 34.0
    ys, y = [], 0.0
    for g in groups:
        for n in g:
            ys.append((n, y))
            y += rh
        y += gap
    hm = y - gap
    colx = [j * cw + (cgap if ROWS[j][0] == "L24x" else 0.0) for j in range(len(ROWS))]
    wm = colx[-1] + cw
    H = top + hm + 3.0
    cv = st.Canvas(W, H)
    ax = cv.axes(left, top, wm, hm)
    ax.set_xlim(0, wm)
    ax.set_ylim(hm, 0)
    ax.axis("off")
    cmap, norm = colormaps["cividis"], mcolors.Normalize(0.5, 1.0)
    for n, y0 in ys:
        cv.text(left - 1.0, top + y0 + rh / 2, SCORE_NAME.get(n, n), ha="right", va="center")
        for j, (sub, prog) in enumerate(ROWS):
            x = val.get((n, sub, prog))
            if not x or x.get("auroc") is None:
                continue
            v = x["auroc"]
            c = cmap(norm(v))
            ax.add_patch(Rectangle((colx[j], y0), cw, rh, facecolor=c, edgecolor="white", lw=1.0))
            lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
            ax.text(colx[j] + cw / 2, y0 + rh / 2, f"{v:.2f}", ha="center", va="center",
                    color="white" if lum < 0.45 else st.TEXT)
            sd.add("matrix", SCORE_NAME.get(n, n), subset=sub, program=prog, auroc=v, n=x.get("n"),
                   events=x.get("positives"))
    for j, (sub, prog) in enumerate(ROWS):
        cv.text(left + colx[j] + cw / 2, top - 1.0, _endpoint(sub, prog), ha="center", va="bottom", rotation=90)
    for g in ("R128", "L24x"):
        js = [j for j, r in enumerate(ROWS) if r[0] == g]
        cv.text(left + colx[js[0]], 2.0, GROUP[g], ha="left", va="top", fontweight="bold")
    return cv, sd


# ---- S5: ranks across endpoints --------------------------------------------------------------------------------------

def _predictor_style(name: str):
    if name in ("twin_current", "twin"):
        return st.DIDYMOS, "o", 3.6
    if name == "tabpfn_v3_5":
        return st.TABPFN, "^", 3.4
    if name == "lgbm_pooled":
        return st.LGBM, "s", 3.0
    if name in ("sofa", "news", "mews", "qsofa", "sirs"):
        return st.CLINICAL, "D", 2.6
    return st.TEXT2, "o", 2.6


def _rank_panel(cv: st.Canvas, sd: st.SourceData, r: dict, top: float, rh: float, panel: str, k_max: int) -> float:
    """Mean ranks (95% CI) with every task's rank behind them, and Holm P for Didymos against each; returns the
    bottom of the axes (mm)."""
    comps = {c["comparator"]: c for c in r["comparisons"]}
    cols, ranks = r["names"], np.asarray(r["ranks"])          # the rank matrix's columns
    preds = r["predictors"]                                   # sorted by mean rank
    ax = _rows_axes(cv, 40, top, 112, len(preds), rh)
    cv.text(180.0, top - 2.0, r"$\mathit{P}$ (Holm)", ha="right", va="center", color=st.TEXT2)
    for i, p in enumerate(preds):
        n = p["predictor"]
        col, mk, ms = _predictor_style(n)
        y = _row_y(top, rh, i)
        cv.text(38.0, y, st.PREDICTOR.get(n, n), ha="right", va="center",
                fontweight="bold" if n in ("twin_current", "twin") else "normal")
        j = cols.index(n)
        task_ranks = ranks[:, j]
        jit = np.linspace(-0.25, 0.25, len(task_ranks)) if len(task_ranks) <= 20 else \
            np.random.default_rng(0).uniform(-0.28, 0.28, len(task_ranks))
        ax.plot(task_ranks, i + jit, ls="none", marker="o", ms=1.6, mfc="white", mec=st.LEARNER, mew=1.0, zorder=1)
        ax.plot(p["mean_rank_ci"], [i, i], color=col, lw=1.5, solid_capstyle="butt", zorder=2)
        ax.plot(p["mean_rank"], i, ls="none", marker=mk, ms=ms, color=col, zorder=3)
        c = comps.get(n)
        if c:
            cv.text(180.0, y, st.p_value(c["p_holm"]), ha="right", va="center")
        sd.add(panel, st.PREDICTOR.get(n, n), mean_rank=p["mean_rank"], ci_low=p["mean_rank_ci"][0],
               ci_high=p["mean_rank_ci"][1], firsts=p["firsts"], mean_auroc=p["mean_auroc"],
               didymos_wins=(c or {}).get("wins"), didymos_losses=(c or {}).get("losses"), p=(c or {}).get("p"),
               p_holm=(c or {}).get("p_holm"))
    ax.set_xlim(0.5, k_max + 0.5)
    ticks = [1, 5, 10, 15, 20] if k_max >= 10 else list(range(1, k_max + 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("Rank (1 = highest AUROC)")
    return top + len(preds) * rh


def ranks(inp: Inputs):
    """Ranks across endpoints (amendment 15): a, every predictor on the 12 primary endpoints; b, the predictors scored on
    all 165 endpoints of the pre-specified analysis."""
    sd = st.SourceData("supp5")
    sd.source(inp.stats_path)
    rk = inp.stats["ranks"]
    rp, rb = rk["primary"], rk["breadth"]
    rh_a, rh_b = 3.6, 4.6
    top_a = 10.0
    bottom_a = top_a + len(rp["predictors"]) * rh_a
    top_b = bottom_a + 24.0
    H = top_b + len(rb["predictors"]) * rh_b + 20.0
    cv = st.Canvas(W, H)
    cv.letter(0, 2.0, "a", f"Primary endpoints ({rp['tasks']}): mean rank among {len(rp['predictors'])} predictors")
    _rank_panel(cv, sd, rp, top_a, rh_a, "a", len(rp["predictors"]))
    cv.letter(0, top_b - 8.0, "b", f"Every scored endpoint ({rb['tasks']}): mean rank")
    _rank_panel(cv, sd, rb, top_b, rh_b, "b", len(rb["predictors"]))
    _legend(cv, [_handle(marker="o", ms=3.6, color=st.DIDYMOS), _handle(marker="^", ms=3.4, color=st.TABPFN),
                 _handle(marker="s", ms=3.0, color=st.LGBM), _handle(marker="D", ms=2.6, color=st.CLINICAL),
                 _handle(marker="o", ms=2.6, color=st.TEXT2),
                 _handle(marker="o", ms=1.6, mfc="white", mec=st.LEARNER, mew=1.0)],
            ["Didymos", "TabPFN-3.5", "LightGBM", "Clinical score", "Other predictor", "One endpoint"], 40.0, H - 8.0,
            ncol=6)
    return cv, sd


# ---- S6: AUROC against the number of simulated futures ---------------------------------------------------------------

def futures_curves(inp: Inputs):
    """AUROC against the number of simulated futures, every endpoint: the sampled estimate (the mean over random
    subsets of each size), the hazard estimate where it applies, and LightGBM on the same anchors."""
    import paper_fig2

    sd = st.SourceData("supp6")
    f2rows = {(r["subset"], r["program"]): r for r in paper_fig2.data(inp, st.SourceData("unused"))}
    blocks = [(GROUP[g], [r for r in ROWS if r[0] == g]) for g in ("R128", "L24x")]
    pos, heads, bottom = _blocks(blocks)
    H = bottom + 10.0
    cv = st.Canvas(W, H)
    for name, y in heads:
        cv.text(0, y, name, ha="left", va="top", fontweight="bold")
    ks = (1, 2, 4, 8, 16, 32, 64, 128)
    for (sub, prog), x0, y0, cx in pos:
        ax = cv.axes(x0, y0, 24, 22)
        lg = f2rows[(sub, prog)]["lgbm"]
        st.refline(ax, lg, axis="y", color=st.LGBM, ls=(0, (1, 1.3)))
        sd.add("panel", "LightGBM", subset=sub, program=prog, auroc=lg)
        a1 = inp.json(f"f1/{sub}/a1.json", sd=sd)["programs"].get(prog, {})
        cur = a1.get("rollout_curve") or {}
        e0 = [cur.get(str(k), {}).get("auroc") for k in ks]
        ax.plot(ks, [np.nan if v is None else v for v in e0], color=st.DIDYMOS, lw=1.0, ls=(0, (2.5, 1.5)),
                marker="o", ms=2.2, mfc="white", mec=st.DIDYMOS, mew=1.0)
        f2 = inp.json(f"f1/{sub}/f2.json")["f2"].get(prog)
        if f2:
            e2 = [f2["curve"][str(k)]["e2"] for k in ks]
            ax.plot(ks, e2, color=st.DIDYMOS, lw=1.5, marker="o", ms=2.2)
            for k, v in zip(ks, e2, strict=True):
                sd.add("panel", "hazard estimate", subset=sub, program=prog, futures=k, auroc=v)
        for k, v in zip(ks, e0, strict=True):
            sd.add("panel", "sampled estimate", subset=sub, program=prog, futures=k, auroc=v)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 8, 128])
        ax.set_xticklabels(["1", "8", "128"])
        ax.minorticks_off()
        ax.set_xlim(0.8, 160)
        ax.set_ylim(0.5, 1.0)
        ax.set_yticks([0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.5", "0.75", "1"] if cx == 0 else [])
        cv.text(x0, y0 - 1.0, _endpoint(sub, prog), ha="left", va="bottom")
    _shared_labels(cv, "Simulated futures per anchor", "AUROC", 13, 13 + 4 * 33.5 + 24, _spans(pos), bottom)
    _legend(cv, [_handle(color=st.DIDYMOS, lw=1.5, ls="-", marker="o", ms=2.2),
                 _handle(color=st.DIDYMOS, lw=1.0, ls=(0, (2.5, 1.5)), marker="o", ms=2.2, mfc="white",
                         mec=st.DIDYMOS, mew=1.0),
                 _handle(color=st.LGBM, lw=1.0, ls=(0, (1, 1.3)))],
            ["Hazard estimate", "Sampled estimate", "LightGBM"], 13 + 4 * 33.5, pos[-1][2] + 1.0)
    return cv, sd


# ---- S7: per-database AUROC ------------------------------------------------------------------------------------------

def per_database(inp: Inputs):
    """Per-database AUROC: Didymos's sampled estimate (128 futures) against LightGBM, on every primary endpoint."""
    sd = st.SourceData("supp7")
    comp = inp.json("f10/comparators.json", sd=sd)
    blocks = [(GROUP[g], [r for r in ROWS if r[0] == g]) for g in ("R128", "L24x")]
    pos, heads, bottom = _blocks(blocks)
    H = bottom + 10.0
    cv = st.Canvas(W, H)
    for name, y in heads:
        cv.text(0, y, name, ha="left", va="top", fontweight="bold")
    for (sub, prog), x0, y0, cx in pos:
        ax = cv.axes(x0, y0, 24, 22)
        m = comp["subsets"][sub][prog]["models"]
        tw, lg = m["twin_128"].get("datasets", {}), m["lgbm_pooled"].get("datasets", {})
        ax.plot([0.4, 1], [0.4, 1], color=st.REF, lw=1.0, ls=(0, (3, 2)), zorder=0)
        for d in st.DATASETS:
            if d in tw and d in lg and tw[d]["positives"] >= 5 and tw[d]["n"] - tw[d]["positives"] >= 5:
                ax.plot(lg[d]["auroc"], tw[d]["auroc"], ls="none", marker="o", ms=3.0, color=st.DIDYMOS,
                        clip_on=False, zorder=3)
                sd.add("panel", "per database", subset=sub, program=prog, dataset=d, didymos=tw[d]["auroc"],
                       lightgbm=lg[d]["auroc"], events=tw[d]["positives"], n=tw[d]["n"])
        ax.plot(m["lgbm_pooled"]["auroc"], m["twin_128"]["auroc"], ls="none", marker="o", ms=4.6, mfc="white",
                mec=st.TEXT, mew=1.0, clip_on=False, zorder=4)
        sd.add("panel", "all databases", subset=sub, program=prog, didymos=m["twin_128"]["auroc"],
               lightgbm=m["lgbm_pooled"]["auroc"])
        ax.set_xlim(0.4, 1)
        ax.set_ylim(0.4, 1)
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_xticklabels(["0.5", "0.75", "1"])
        ax.set_yticks([0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.5", "0.75", "1"] if cx == 0 else [])
        cv.text(x0, y0 - 1.0, _endpoint(sub, prog), ha="left", va="bottom")
    _shared_labels(cv, "LightGBM AUROC", "Didymos AUROC", 13, 13 + 4 * 33.5 + 24, _spans(pos), bottom)
    _legend(cv, [_handle(marker="o", ms=3.0, color=st.DIDYMOS),
                 _handle(marker="o", ms=4.6, mfc="white", mec=st.TEXT, mew=1.0),
                 _handle(color=st.REF, ls=(0, (3, 2)), lw=1.0)],
            ["One database", "All databases", "Equal AUROC"], 13 + 4 * 33.5, pos[-1][2] + 1.0)
    return cv, sd


# ---- S8: every scored endpoint ---------------------------------------------------------------------------------------

BREADTH_TITLE = {"R": "Rolling anchors", "L24": "24-hour landmark", "L48": "48-hour landmark"}


def _program_order(prog: str) -> tuple:
    base, _, h = prog.partition("@")
    names = list(st.PROGRAM)
    return (names.index(base) if base in names else len(names), int(h) if h else 0)


def breadth(inp: Inputs):
    """Every endpoint the pre-specified analysis scored (amendment 14, family 11): Didymos minus LightGBM with its 95%
    interval, by anchor set and the catalog's family; filled where the Benjamini-Hochberg q is at most 0.05."""
    sd = st.SourceData("supp8")
    sd.source(inp.stats_path)
    rows11 = inp.family(11)
    cols = []
    for design in ("R", "L24", "L48"):
        lines = []
        for f in st.FAMILY:
            sel = sorted((r for r in rows11 if r["design"] == design and r["catalog_family"] == f),
                         key=lambda r: _program_order(r["program"]))
            if sel:
                lines.append(("header", st.FAMILY[f], None))
                lines += [("row", st.program_name(r["program"]), r) for r in sel]
        cols.append((design, lines))
    rh, top = 2.7, 9.0
    n = max(len(lines) for _d, lines in cols)
    H = top + n * rh + 19.0
    cv = st.Canvas(W, H)
    for c, (design, lines) in enumerate(cols):
        x0 = c * 60.5
        cv.text(x0, 2.0, BREADTH_TITLE[design], ha="left", va="top", fontweight="bold")
        ax = _rows_axes(cv, x0 + 38.5, top, 19.0, len(lines), rh)
        st.refline(ax, 0.0)
        for i, (kind, lab, r) in enumerate(lines):
            y = _row_y(top, rh, i)
            if kind == "header":
                cv.text(x0, y, lab, ha="left", va="center", fontweight="bold")
                continue
            cv.text(x0 + 37.5, y, lab, ha="right", va="center")
            sig = r["q_bh"] <= 0.05
            col = (st.DIDYMOS if r["estimate"] > 0 else st.LGBM) if sig else st.TEXT2
            ax.plot(np.clip(r["ci"], -0.3, 0.3), [i, i], color=col if sig else st.LEARNER, lw=1.0,
                    solid_capstyle="butt")
            ax.plot(r["estimate"], i, ls="none", marker="o", ms=2.4, mfc=col if sig else "white", mec=col, mew=1.0)
            sd.add("panel", lab, design=design, program=r["program"], family=r["catalog_family"],
                   difference=r["estimate"], ci_low=r["ci"][0], ci_high=r["ci"][1], p=r["p"], q_bh=r["q_bh"],
                   p_holm=r["p_holm"], events=r["positives"], n=r["n"])
        ax.set_xlim(-0.3, 0.3)
        ax.set_xticks([-0.2, 0, 0.2])
        ax.set_xticklabels([st.num(-0.2, 1), "0", "0.2"])
        ax.set_xlabel("Difference, AUROC")
    _legend(cv, [_handle(color=st.DIDYMOS, lw=1.0, ls="-", marker="o", ms=2.4),
                 _handle(color=st.LGBM, lw=1.0, ls="-", marker="o", ms=2.4),
                 _handle(color=st.LEARNER, lw=1.0, ls="-", marker="o", ms=2.4, mfc="white", mec=st.TEXT2, mew=1.0)],
            ["Favours Didymos (FDR 5%)", "Favours LightGBM (FDR 5%)", "Not significant"], 0.0, H - 7.0, ncol=3)
    return cv, sd


# ---- S9: the breadth by clinical family, and label efficiency --------------------------------------------------------

LE_BINS = (("≤100", 100.0), ("≤300", 300.0), ("≤1,000", 1000.0), ("≤5,000", 5000.0), ("Not reached", None))


def _le_bin(r: dict) -> int:
    """The bin of a label-efficiency cell: the matching training size, or not reached."""
    if r["status"] == "not reached":
        return len(LE_BINS) - 1
    return next(i for i, (_l, hi) in enumerate(LE_BINS) if hi is not None and r["n_match"] <= hi + 1e-9)


def _breadth_panel(cv: st.Canvas, sd: st.SourceData, inp: Inputs, y0: float) -> None:
    """Didymos minus LightGBM on every program the pre-specified analysis scored, by catalog family; filled
    where the Benjamini-Hochberg q is at most 0.05; the family medians; all endpoints' median with its joint-bootstrap
    interval."""
    rows11 = inp.family(11)
    summ = inp.stats["breadth"]["groups"]["all"]
    groups = [(f, [r for r in rows11 if r["catalog_family"] == f]) for f in st.FAMILY]
    groups = [(f, sel) for f, sel in groups if sel]
    rh, top = 5.4, y0 + 10.0
    n = len(groups) + 1
    ax = cv.axes(30, top - rh / 2, 50, n * rh)
    ax.set_ylim(n - 0.5, -0.5)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    st.refline(ax, 0.0)
    cv.text(97.0, top - 5.0, "Ahead", ha="right", va="center", color=st.TEXT2)
    for i, (f, sel) in enumerate(groups):
        y = top + i * rh
        cv.text(28.0, y, st.FAMILY[f], ha="right", va="center")
        for j, r in enumerate(sorted(sel, key=lambda r: r["estimate"])):
            off = (-0.25, 0.0, 0.25)[j % 3]
            if r["q_bh"] <= 0.05:
                ax.plot(r["estimate"], i + off, ls="none", marker="o", ms=2.6, zorder=3,
                        color=st.DIDYMOS if r["estimate"] > 0 else st.LGBM)
            else:
                ax.plot(r["estimate"], i + off, ls="none", marker="o", ms=2.6, mfc="white", mec=st.LEARNER, mew=1.0,
                        zorder=2)
            sd.add("a", r["endpoint"], design=r["design"], program=r["program"], family=st.FAMILY[f],
                   difference=r["estimate"], ci_low=r["ci"][0], ci_high=r["ci"][1], p=r["p"], q_bh=r["q_bh"],
                   p_holm=r["p_holm"], events=r["positives"], n=r["n"])
        med = float(np.median([r["estimate"] for r in sel]))
        ax.plot([med, med], [i - 0.42, i + 0.42], color=st.TEXT, lw=1.5, zorder=4)
        cv.text(97.0, y, f"{sum(r['estimate'] > 0 for r in sel)} of {len(sel)}", ha="right", va="center")
    i, y = len(groups), top + len(groups) * rh
    cv.text(28.0, y, "All endpoints", ha="right", va="center", fontweight="bold")
    ax.plot(summ["median_ci"], [i, i], color=st.TEXT, lw=1.5, solid_capstyle="butt")
    ax.plot(summ["median"], i, ls="none", marker="D", ms=3.2, color=st.TEXT)
    cv.text(97.0, y, f"{summ['positive']} of {summ['programs']}", ha="right", va="center", fontweight="bold")
    sd.add("a", "all endpoints", median=summ["median"], median_ci_low=summ["median_ci"][0],
           median_ci_high=summ["median_ci"][1], positive=summ["positive"], programs=summ["programs"],
           share_positive=summ["share_positive"], share_ci_low=summ["share_positive_ci"][0],
           share_ci_high=summ["share_positive_ci"][1], fdr_favour_didymos=summ["bh_favour_didymos"],
           fdr_favour_lightgbm=summ["bh_favour_lightgbm"])
    ax.set_xlim(-0.15, 0.2)
    ax.set_xticks([-0.1, 0, 0.1, 0.2])
    ax.set_xticklabels([st.num(-0.1, 1), "0", "0.1", "0.2"])
    ax.set_xlabel("Didymos minus LightGBM, AUROC")
    from matplotlib.lines import Line2D

    ly = top + n * rh + 7.5
    cv.fig.legend([Line2D([], [], ls="none", marker="o", ms=2.6, color=st.DIDYMOS),
                   Line2D([], [], ls="none", marker="o", ms=2.6, color=st.LGBM),
                   Line2D([], [], ls="none", marker="o", ms=2.6, mfc="white", mec=st.LEARNER, mew=1.0),
                   Line2D([], [], ls="none", marker="|", ms=5.0, mew=1.5, color=st.TEXT)],
                  ["Favours Didymos (FDR 5%)", "Favours LightGBM (FDR 5%)", "Not significant", "Median"], loc="upper left", ncol=2,
                  bbox_to_anchor=(2.0 / W, 1 - ly / cv.h), columnspacing=1.2, handletextpad=0.4, borderaxespad=0,
                  handlelength=1.0)


def _labels_panel(cv: st.Canvas, sd: st.SourceData, inp: Inputs, y0: float) -> None:
    """The training size at which a LightGBM trained in one database matches Didymos there, per primary
    endpoint and database (amendment 14, item 12)."""
    from matplotlib import colormaps
    from matplotlib.patches import Patch, Rectangle

    le = inp.stats["label_efficiency"]
    keys = list(dict.fromkeys((r["design"], r["program"]) for r in le))
    rh, cw, top, x0 = 3.6, 6.8, y0 + 10.0, 131.0
    ax = cv.axes(x0, top - rh / 2, 7 * cw, len(keys) * rh)
    ax.set_xlim(0, 7 * cw)
    ax.set_ylim(len(keys) * rh, 0)
    ax.axis("off")
    cmap = colormaps["cividis"]
    shades = [cmap(v) for v in (0.95, 0.72, 0.5, 0.28, 0.04)]
    for i, (design, prog) in enumerate(keys):
        cv.text(x0 - 2.0, top + i * rh, st.ENDPOINT_SHORT[prog], ha="right", va="center")
        for j, ds in enumerate(st.DATASETS):
            r = find(le, design=design, program=prog, dataset=ds)
            if r is None:
                ax.text((j + 0.5) * cw, (i + 0.5) * rh, "–", ha="center", va="center", color=st.TEXT2)
                continue
            b = _le_bin(r)
            ax.add_patch(Rectangle((j * cw, i * rh), cw, rh, facecolor=shades[b], edgecolor="white", lw=1.0))
            sd.add("b", r["endpoint"], design=design, program=prog, dataset=ds, didymos_auroc=r["didymos"],
                   n_match=r["n_match"], status=r["status"], max_n=r["max_n"], bin=LE_BINS[b][0])
    yb = top - rh / 2 + len(keys) * rh + 1.0
    for j, ds in enumerate(st.DATASETS):
        cv.text(x0 + (j + 0.5) * cw, yb, st.DATASET_SHORT[ds], ha="center", va="top", rotation=90)
    cv.fig.legend([Patch(facecolor=c, edgecolor="none") for c in shades], [lab for lab, _hi in LE_BINS],
                  loc="upper left", ncol=3, bbox_to_anchor=(103.0 / W, 1 - (yb + 13.5) / cv.h), columnspacing=1.0,
                  handletextpad=0.4, borderaxespad=0, handlelength=1.2)


def families_and_labels(inp: Inputs):
    """a, every endpoint the pre-specified analysis scored, Didymos minus LightGBM by the catalog's families
    (amendment 14, family 11); b, the training size at which a LightGBM trained in one database matches Didymos there
    (amendment 14, item 12)."""
    sd = st.SourceData("supp9")
    sd.source(inp.stats_path)
    cv = st.Canvas(W, 84.0)
    cv.letter(0, 2.0, "a", "Every scored endpoint by clinical family")
    _breadth_panel(cv, sd, inp, 2.0)
    cv.letter(102, 2.0, "b", "Labelled stays needed to match Didymos")
    _labels_panel(cv, sd, inp, 2.0)
    return cv, sd


# ---- S10: regression and time to event -------------------------------------------------------------------------------

def regression_and_time_to_event(inp: Inputs):
    """Remaining ICU stay (error and interval coverage) and the time to event (concordance), the pre-specified
    analysis (32 futures)."""
    sd = st.SourceData("supp10")
    a1 = {d: inp.json(f"a1/test_{d}/a1.json", sd=sd)["programs"] for d in ("R", "L24", "L48")}
    H = 88.0
    cv = st.Canvas(W, H)
    lands = (("L24", "24-hour landmark"), ("L48", "48-hour landmark"))
    # a: error
    cv.letter(0, 2.0, "a", "Remaining ICU stay, error")
    ax = _rows_axes(cv, 34, 10, 48, 2, 7.0)
    for i, (d, lab) in enumerate(lands):
        r = a1[d]["remaining_los@0"]
        cv.text(32, _row_y(10, 7.0, i), f"{lab} ({r['n']:,})", ha="right", va="center")
        ax.plot([r["mae_median"], r["mae_baseline_mean"]], [i, i], color=st.LEARNER, lw=1.0, zorder=1)
        ax.plot(r["mae_median"], i, ls="none", marker="o", ms=3.8, color=st.DIDYMOS, zorder=3)
        ax.plot(r["mae_baseline_mean"], i, ls="none", marker="s", ms=3.2, color=st.TEXT2, zorder=3)
        ax.text(r["mae_median"] - 2.0, i - 0.32, f"{r['mae_median']:.1f}", ha="right", va="bottom")
        ax.text(r["mae_baseline_mean"] + 2.0, i - 0.32, f"{r['mae_baseline_mean']:.1f}", ha="left", va="bottom")
        sd.add("a", lab, anchors=r["n"], mae_didymos_h=r["mae_median"], mae_predicting_the_mean_h=r["mae_baseline_mean"],
               crps_h=r.get("crps"))
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 30, 60, 90])
    ax.set_xlabel("Mean absolute error (h)")
    _legend(cv, [_handle(marker="o", ms=3.8, color=st.DIDYMOS), _handle(marker="s", ms=3.2, color=st.TEXT2)],
            ["Didymos, median of simulated futures", "Predicting the mean"], 2.0, 36.0, ncol=1)
    # b: interval coverage
    cv.letter(92, 2.0, "b", "Remaining ICU stay, interval coverage")
    bx = cv.axes(104, 10, 26, 26)
    bx.plot([0.35, 1.0], [0.35, 1.0], color=st.REF, lw=1.0, ls=(0, (3, 2)), zorder=0)
    for (d, lab), mfc in zip(lands, (st.DIDYMOS, "white"), strict=True):
        cov = a1[d]["remaining_los@0"]["coverage"]
        nom = sorted(float(k) for k in cov)
        obs = [cov[k] for k in sorted(cov, key=float)]
        bx.plot(nom, obs, color=st.DIDYMOS, lw=1.0, marker="o", ms=3.2, mfc=mfc, mec=st.DIDYMOS, mew=1.0)
        for n_, o_ in zip(nom, obs, strict=True):
            sd.add("b", lab, nominal=n_, observed=o_)
    bx.set_xlim(0.35, 1.0)
    bx.set_ylim(0.35, 1.0)
    bx.set_xticks([0.5, 0.8, 0.95])
    bx.set_xticklabels(["0.5", "0.8", "0.95"])
    bx.set_yticks([0.4, 0.6, 0.8, 1.0])
    bx.set_yticklabels(["0.4", "0.6", "0.8", "1"])
    bx.set_xlabel("Nominal coverage")
    bx.set_ylabel("Observed coverage")
    _legend(cv, [_handle(color=st.DIDYMOS, lw=1.0, ls="-", marker="o", ms=3.2),
                 _handle(color=st.DIDYMOS, lw=1.0, ls="-", marker="o", ms=3.2, mfc="white", mec=st.DIDYMOS,
                         mew=1.0)],
            ["24-hour landmark", "48-hour landmark"], 136.0, 12.0)
    # c: concordance
    y0 = 50.0
    cv.letter(0, y0, "c", "Time to event, concordance")
    evs = (("death_time@0", "Death"), ("imv_time@0", "New IMV"), ("vasopressor_time@0", "New vasopressor"))
    designs = (("R", "Rolling anchors"), ("L24", "24-hour landmark"), ("L48", "48-hour landmark"))
    rh = 5.2
    for j, (d, lab) in enumerate(designs):
        x0 = 30 + j * 50.0
        cx = _rows_axes(cv, x0, y0 + 10, 42, len(evs), rh)
        cv.text(x0, y0 + 9.0, lab, ha="left", va="bottom")
        st.refline(cx, 0.5)
        for i, (p, el) in enumerate(evs):
            x = a1[d].get(p)
            if j == 0:
                cv.text(28, _row_y(y0 + 10, rh, i), el, ha="right", va="center")
            if not x or x.get("c_index_48h_share") is None:
                continue
            cx.plot(x["c_index_48h_share"], i, ls="none", marker="o", ms=3.8, color=st.DIDYMOS)
            cx.text(x["c_index_48h_share"] + 0.03, i, f"{x['c_index_48h_share']:.2f}", ha="left", va="center")
            sd.add("c", el, design=lab, c_index=x["c_index_48h_share"], events=x["events"], anchors=x["n"])
        cx.set_xlim(0.4, 1.0)
        cx.set_xticks([0.5, 0.75, 1.0])
        cx.set_xticklabels(["0.5", "0.75", "1"])
        cx.set_xlabel("Concordance index")
    return cv, sd


# ---- S11: site prompts and leaving a database out --------------------------------------------------------------------

def sites(inp: Inputs):
    """The site prompt (the follow-up rolling anchors, 32 futures): an unknown site or another database's against the
    known one; and supervised transfer (LightGBM trained without a database) beside label-free Didymos."""
    import matplotlib.colors as mcolors
    from matplotlib import colormaps
    from matplotlib.patches import Rectangle

    sd = st.SourceData("supp11")
    t = inp.json("f9/transfer_R.json", sd=sd)
    progs = [p for s_, p in ROWS if s_ == "R128" and p in t["site_prompts"]]
    rh, top = 4.6, 12.0
    H = 128.0
    cv = st.Canvas(W, H)
    cv.letter(0, 2.0, "a", "Site prompt")
    cv.text(46.0, top - 1.5, "Events/n", ha="right", va="bottom", color=st.TEXT2)
    cv.text(60.0, top - 1.5, "AUROC", ha="right", va="bottom", color=st.TEXT2)
    ax = _rows_axes(cv, 66, top, 68, len(progs), rh)
    st.refline(ax, 0.0)
    for i, p in enumerate(progs):
        x = t["site_prompts"][p]
        y = _row_y(top, rh, i)
        cv.text(31.0, y, st.ENDPOINT_SHORT[p], ha="right", va="center")
        cv.text(46.0, y, f"{x['positives']:,}/{x['anchors']:,}", ha="right", va="center", color=st.TEXT2)
        cv.text(60.0, y, f"{x['known']['auroc']:.2f}", ha="right", va="center")
        for key, off, mk, mfc in (("unknown_minus_known", -0.2, "o", "white"), ("swap_minus_known", 0.2, "^",
                                                                                 st.DIDYMOS)):
            d = x[key]
            ax.plot(d["ci"], [i + off] * 2, color=st.DIDYMOS, lw=1.0, solid_capstyle="butt", zorder=2)
            ax.plot(d["auroc_diff"], i + off, ls="none", marker=mk, ms=3.4, mfc=mfc, mec=st.DIDYMOS, mew=1.0,
                    color=st.DIDYMOS, zorder=3)
            sd.add("a", key, program=p, difference=d["auroc_diff"], ci_low=d["ci"][0], ci_high=d["ci"][1],
                   known_auroc=x["known"]["auroc"], events=x["positives"], anchors=x["anchors"])
    ax.set_xlim(-0.55, 0.2)
    ax.set_xticks([-0.5, -0.25, 0, 0.2])
    ax.set_xticklabels([st.num(-0.5, 1), st.num(-0.25, 2), "0", "0.2"])
    ax.set_xlabel("Difference in AUROC (95% CI)")
    _legend(cv, [_handle(color=st.DIDYMOS, lw=1.0, ls="-", marker="o", ms=3.4, mfc="white", mec=st.DIDYMOS,
                         mew=1.0),
                 _handle(color=st.DIDYMOS, lw=1.0, ls="-", marker="^", ms=3.4)],
            ["Site unknown", "Another database's token"], 139.0, top + 1.0)
    # b: leaving a database out
    yb = top + len(progs) * rh + 15.0
    cv.letter(0, yb, "b", "Leaving a database out")
    lodo = t["leave_one_dataset_out"]
    cw, rb = 8.6, 3.9
    tb = yb + 22.0
    mats = (("lgbm_lodo", "lgbm_within", "LightGBM without the database minus within it"),
            ("twin_zero_shot", "lgbm_lodo", "Didymos minus LightGBM without the database"))
    cmap, norm = colormaps["PuOr"], mcolors.Normalize(-0.3, 0.3)
    for k, (a_, b_, title) in enumerate(mats):
        x0 = 34.0 + k * (7 * cw + 8.0)
        mx = cv.axes(x0, tb, 7 * cw, len(progs) * rb)
        mx.set_xlim(0, 7 * cw)
        mx.set_ylim(len(progs) * rb, 0)
        mx.axis("off")
        cv.text(x0, yb + 6.0, title, ha="left", va="top")
        for j, d in enumerate(st.DATASETS):
            cv.text(x0 + (j + 0.5) * cw, tb - 1.0, st.DATASET_SHORT[d], ha="center", va="bottom", rotation=90)
        for i, p in enumerate(progs):
            for j, d in enumerate(st.DATASETS):
                x = lodo.get(p, {}).get(d)
                if not x:
                    continue
                v = x[a_] - x[b_]
                c = cmap(norm(v))
                mx.add_patch(Rectangle((j * cw, i * rb), cw, rb, facecolor=c, edgecolor="white", lw=1.0))
                mx.text((j + 0.5) * cw, (i + 0.5) * rb, "0.00" if round(v, 2) == 0 else st.num(v, 2, sign=True),
                        ha="center", va="center", color="white" if abs(v) > 0.2 else st.TEXT)
                if k == 0:
                    sd.add("b", "per database", program=p, dataset=d, lightgbm_without=x["lgbm_lodo"],
                           lightgbm_within=x["lgbm_within"], lightgbm_pooled=x["lgbm_pooled"],
                           didymos_label_free=x["twin_zero_shot"], events=x.get("positives"))
    for i, p in enumerate(progs):
        cv.text(32.0, tb + (i + 0.5) * rb, st.ENDPOINT_SHORT[p], ha="right", va="center")
    cb = cv.axes(34.0 + 2 * (7 * cw + 8.0) - 4.0, tb, 2.5, len(progs) * rb)
    grad = np.linspace(0.3, -0.3, 100)[:, None]
    cb.imshow(grad, cmap=cmap, norm=norm, aspect="auto", extent=(0, 1, -0.3, 0.3))
    cb.set_xticks([])
    cb.yaxis.tick_right()
    cb.set_yticks([-0.3, 0, 0.3])
    cb.set_yticklabels([st.num(-0.3, 1), "0", "0.3"])
    for s_ in ("top", "bottom", "left"):
        cb.spines[s_].set_visible(False)
    cb.spines["right"].set_visible(False)
    return cv, sd


# ---- S12: subgroups --------------------------------------------------------------------------------------------------

SUBGROUPS = (("sex", "Sex", (("female", "Female"), ("male", "Male"))),
             ("age_band", "Age, years", (("0", "18 to 44"), ("1", "45 to 64"), ("2", "65 to 79"), ("3", "80 and over"))),
             ("dataset", "Database", tuple((d, st.DATASET_SHORT[d]) for d in st.DATASETS)),
             ("admission_type", "Admission", (("ed", "Emergency department"), ("elective", "Elective"),
                                              ("direct", "Direct"), ("other", "Other"), ("UNK", "Not recorded"))),
             ("race", "Race", (("white", "White"), ("not recorded", "Not recorded"))))
SUB_ENDPOINTS = (("icu_death@-1", "ICU death"), ("hospital_death@0", "In-hospital death"),
                 ("remaining_los_gt_7d", "ICU stay >7 days"))


def _scored(m: dict) -> bool:
    tw = m.get("twin", {})
    return tw.get("scored") is not False and "auroc" in tw


def subgroups(inp: Inputs):
    """Subgroups (pre-specified, the 24-hour landmark, 32 futures): AUROC and observed over expected of Didymos and
    LightGBM for the three stay-level outcomes."""
    sd = st.SourceData("supp12")
    a5 = inp.json("a1/test_L24/compare.json", sd=sd)["a5"]
    lines = []
    for key, head, vals in SUBGROUPS:
        keep = [(v, lab) for v, lab in vals if any(_scored(a5[p].get(key, {}).get(v, {})) for p, _ in SUB_ENDPOINTS)]
        if keep:
            lines.append(("header", head, None))
            lines += [("row", lab, (key, v)) for v, lab in keep]
    rh = 3.1
    hp = len(lines) * rh
    top_a, top_b = 12.0, 12.0 + hp + 22.0
    H = top_b + hp + 19.0
    cv = st.Canvas(W, H)
    cv.letter(0, 2.0, "a", "Discrimination (AUROC)")
    cv.letter(0, top_b - 10.0, "b", "Calibration (observed over expected)")
    xs = (36.0, 84.0, 132.0)
    for top, metric in ((top_a, "auroc"), (top_b, "observed_over_expected")):
        for i, (kind, lab, _k) in enumerate(lines):
            if kind == "header":
                cv.text(0, _row_y(top, rh, i), lab, ha="left", va="center", fontweight="bold")
            else:
                cv.text(34.0, _row_y(top, rh, i), lab, ha="right", va="center")
        for x0, (p, plab) in zip(xs, SUB_ENDPOINTS, strict=True):
            ax = _rows_axes(cv, x0, top, 42, len(lines), rh)
            cv.text(x0, top - 1.5, plab, ha="left", va="bottom")
            for i, (kind, lab, k) in enumerate(lines):
                if kind != "row":
                    continue
                m = a5[p].get(k[0], {}).get(k[1], {})
                if not _scored(m):
                    continue
                tw, lg = m["twin"], m.get("lgbm_pooled", {})
                if lg.get(metric) is not None:
                    ax.plot(lg[metric], i + 0.18, ls="none", marker="s", ms=2.8, color=st.LGBM, zorder=2)
                ax.plot(tw[metric], i - 0.18, ls="none", marker="o", ms=3.2, color=st.DIDYMOS, zorder=3)
                if metric == "auroc":
                    sd.add("a+b", lab, endpoint=p, subgroup=k[0], value=k[1], events=tw["positives"], n=tw["n"],
                           didymos_auroc=tw["auroc"], lightgbm_auroc=lg.get("auroc"),
                           didymos_oe=tw["observed_over_expected"], lightgbm_oe=lg.get("observed_over_expected"))
            if metric == "auroc":
                ax.set_xlim(0.6, 1.0)
                ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
                ax.set_xticklabels(["0.6", "0.7", "0.8", "0.9", "1"])
                ax.set_xlabel("AUROC")
            else:
                ax.set_xscale("log")
                ax.set_xlim(0.45, 2.4)
                ax.set_xticks([0.5, 1, 2])
                ax.set_xticklabels(["0.5", "1", "2"])
                ax.minorticks_off()
                st.refline(ax, 1.0)
                ax.set_xlabel("Observed over expected")
    _legend(cv, [_handle(marker="o", ms=3.2, color=st.DIDYMOS), _handle(marker="s", ms=2.8, color=st.LGBM)],
            ["Didymos, sampled estimate (32 futures)", "LightGBM"], 36.0, top_b + hp + 13.0, ncol=2)
    return cv, sd


# ---- S13: the recalibration maps -------------------------------------------------------------------------------------

def recalibration_maps(inp: Inputs):
    """The recalibration maps in force (fitted on the tuning split): raw risk against recalibrated risk."""
    from icutwin.toolkit import recalibration as rc

    sd = st.SourceData("supp13")
    p = inp.path("recalibration/MUC32_main.json", toolkit=True)
    sd.source(p)
    maps = rc.load(p)
    progs = [q for s_, q in ROWS if s_ == "R128" and q in maps]
    pos, _heads, bottom = _blocks([("", progs)], cols=4, left=19.0, px=43.0)
    H = bottom + 10.0
    cv = st.Canvas(W, H)
    xs = np.logspace(-4, 0, 400)
    for prog, x0, y0, cx in pos:
        ax = cv.axes(x0, y0, 28, 22)
        ys, _ok = rc.apply(maps, prog, xs)
        ys = np.where(np.asarray(ys) > 0, ys, np.nan)
        ax.plot([1e-4, 1], [1e-4, 1], color=st.REF, lw=1.0, ls=(0, (3, 2)), zorder=0)
        ax.plot(xs, ys, color=st.DIDYMOS, lw=1.5)
        kind = maps[prog].get("kind", "isotonic")
        for xv, yv in zip(xs[::25], ys[::25], strict=True):
            sd.add("panel", prog, raw=float(xv), recalibrated=None if not np.isfinite(yv) else float(yv), kind=kind)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(1e-4, 1)
        ax.set_xticks([1e-4, 1e-2, 1])
        ax.set_xticklabels(["0.0001", "0.01", "1"])
        ax.set_yticks([1e-4, 1e-2, 1])
        ax.set_yticklabels(["0.0001", "0.01", "1"] if cx == 0 else [])
        ax.minorticks_off()
        cv.text(x0, y0 - 1.0, st.ENDPOINT_SHORT.get(prog, prog), ha="left", va="bottom")
    _shared_labels(cv, "Raw risk", "Recalibrated risk", 19, 19 + 3 * 43.0 + 28, _spans(pos), bottom, offset=13.0)
    _legend(cv, [_handle(color=st.DIDYMOS, lw=1.5, ls="-"), _handle(color=st.REF, lw=1.0, ls=(0, (3, 2)))],
            ["Map", "Identity"], 19 + 3 * 43.0, pos[-1][2] + 1.0)
    return cv, sd


# ---- S14: the toolkit's conformance ----------------------------------------------------------------------------------

def conformance(inp: Inputs):
    """Forecast's conformance: the toolkit's 32-future AUROC and O/E minus each of the paper's two runs on the same
    rolling anchors (the pre-specified run, 32 futures; the follow-up run, its 32-future curve and 128-future O/E),
    against the rollout-noise band (the spread expected between two independent runs). Not drawn, but checked: the
    per-measure comparisons with the band must reproduce the recorded ones."""
    sd = st.SourceData("supp14")
    v = inp.json("v1/v1_noise.json", toolkit=True, sd=sd)["programs"]
    adj = inp.json("v1/v1_adjusted.json", toolkit=True, sd=sd)["programs"]
    progs = [p for s_, p in ROWS if s_ == "R128" and p in v and p in adj]
    rh, top = 5.4, 14.0
    H = top + len(progs) * rh + 18.0
    cv = st.Canvas(W, H)
    cv.letter(0, 2.0, "a", "Discrimination (AUROC)")
    cv.letter(112, 2.0, "b", "Calibration (observed over expected)")
    cv.text(46.0, top - 1.5, "Events/n", ha="right", va="bottom", color=st.TEXT2)
    axes = {"auroc": _rows_axes(cv, 49, top, 56, len(progs), rh), "oe": _rows_axes(cv, 118, top, 56, len(progs), rh)}
    ref = {"auroc": {"a1": "auroc_a1", "f1": "auroc_f1_curve32"}, "oe": {"a1": "oe_a1", "f1": "oe_f1"}}
    runs = (("a1", "pre-specified run", -0.2, True), ("f1", "follow-up run", 0.2, False))
    for ax in axes.values():
        st.refline(ax, 0.0)
    for i, p in enumerate(progs):
        x, z = v[p], adj[p]
        y = _row_y(top, rh, i)
        cv.text(31.0, y, st.ENDPOINT_SHORT.get(p, p), ha="right", va="center")
        cv.text(46.0, y, f"{x['positives']:,}/{x['anchors']:,}", ha="right", va="center", color=st.TEXT2)
        within = {r: True for r, *_ in runs}
        for key, ax in axes.items():
            lim = z[f"{key}_limit"]
            ax.plot([-lim, lim], [i, i], color=st.DIDYMOS_TINT, lw=4.0, solid_capstyle="butt", zorder=1)
            for run, lab, off, filled in runs:
                d = x[f"{key}_difference"] if run == "a1" else z[f"{key}_toolkit"] - z[ref[key][run]]
                within[run] &= abs(d) <= lim               # checked against the record below, not drawn
                ax.plot(d, i + off, ls="none", marker="o", ms=3.4, mfc=st.DIDYMOS if filled else "white",
                        mec=st.DIDYMOS, mew=1.0, zorder=3)
                sd.add("a" if key == "auroc" else "b", p, run=lab, toolkit=z[f"{key}_toolkit"],
                       paper_run=z[ref[key][run]], difference=d, band_half_width=lim,
                       rollout_sd=x[f"sd_{key}_rollouts"], events=x["positives"], anchors=x["anchors"])
        for run, *_ in runs:
            if within[run] != z[f"within_{run}"]:
                raise ValueError(f"{p}: the conformance verdict against the {run} run differs from the recorded one")
    for key, ax in axes.items():
        ax.set_xlim(-0.17, 0.17)
        ax.set_xticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])
        ax.set_xticklabels(["", st.num(-0.1, 1), "", "0", "", "0.1", ""])
        ax.set_xlabel("Toolkit minus the paper's AUROC" if key == "auroc" else "Toolkit minus the paper's O/E")
    yl = top + len(progs) * rh + 12.0
    _legend(cv, [_handle(marker="o", ms=3.4, color=st.DIDYMOS),
                 _handle(marker="o", ms=3.4, mfc="white", mec=st.DIDYMOS, mew=1.0),
                 _handle(color=st.DIDYMOS_TINT, lw=4.0, ls="-")],
            ["Against the pre-specified run", "Against the follow-up run", "Rollout-noise band"], 49.0, yl, ncol=3)
    return cv, sd


# ---- S15: similarity -------------------------------------------------------------------------------------------------

def similarity(inp: Inputs):
    """Similar patients: the nearest neighbours' outcome as a predictor, by Didymos's state against simpler
    representations, against the number of neighbours."""
    sd = st.SourceData("supp15")
    c4 = inp.json("c4/c4b_R.json", sd=sd)["programs"]
    progs = [p for s_, p in ROWS if s_ == "R128" and p in c4]
    reps = (("state", "Didymos state", st.DIDYMOS, "o"), ("features", "Tabular features", st.TEXT, "^"),
            ("tfidf", "TF-IDF of events", st.TEXT2, "s"), ("random_init_state", "Untrained model", st.LEARNER, "D"))
    pos, _heads, bottom = _blocks([("", progs)])
    H = bottom + 10.0
    cv = st.Canvas(W, H)
    for p, x0, y0, cx in pos:
        ax = cv.axes(x0, y0, 24, 22)
        for key, lab, col, mk in reps:
            v = c4[p].get(key, {}).get("all")
            if v:
                ks = sorted(v, key=int)
                ax.plot([int(k) for k in ks], [v[k] for k in ks], color=col, lw=1.0, marker=mk, ms=2.4,
                        zorder=3 if key == "state" else 2)
                for k in ks:
                    sd.add("panel", lab, program=p, neighbours=int(k), auroc=v[k])
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 25, 50])
        ax.set_xticklabels(["5", "10", "25", "50"])
        ax.minorticks_off()
        ax.set_ylim(0.45, 1.0)
        ax.set_yticks([0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.5", "0.75", "1"] if cx == 0 else [])
        cv.text(x0, y0 - 1.0, st.ENDPOINT_SHORT.get(p, p), ha="left", va="bottom")
    _shared_labels(cv, "Neighbours", "AUROC, the neighbours' outcome share", 13, 13 + 4 * 33.5 + 24, _spans(pos),
                   bottom)
    _legend(cv, [_handle(color=c, lw=1.0, ls="-", marker=m, ms=2.4) for _k, _l, c, m in reps],
            [lab for _k, lab, _c, _m in reps], 13 + 4 * 33.5, pos[-1][2] + 1.0)
    return cv, sd


# ---- S16: the trial simulations' checks ------------------------------------------------------------------------------

TRIAL_ENDPOINTS = (("L24x", "icu_death@-1"), ("L24x", "hospital_death@0"), ("L24", "aki_2@2880"),
                   ("L24", "rrt_onset@2880"), ("L24", "vasopressor_onset@1440"), ("L24", "imv_onset@1440"))
TRIAL_SCORES = (("twin", ""), ("lgbm_pooled", ""), ("sapsii", ""), ("sofa", ""), ("oasis", ""), ("news", ""),
                ("twin", "eicu"), ("lgbm_pooled", "eicu"), ("apache_iva", "eicu"))


def _score_style(s: str):
    if s == "twin":
        return st.DIDYMOS, "o", 3.4
    if s == "lgbm_pooled":
        return st.LGBM, "s", 3.0
    return st.CLINICAL, "D", 2.8


def trial_checks(inp: Inputs):
    """The trial simulations (2,000 per cell): the adjusted analysis's type I error with Clopper-Pearson intervals, and
    the sample-size ratio at relative risks 0.8 and 0.7 with Monte Carlo intervals."""
    sd = st.SourceData("supp16")
    sd.source(inp.stats_path)
    trials = inp.stats.get("trials", [])
    lines = []
    for sub, prog in TRIAL_ENDPOINTS:
        lines.append(("header", st.ENDPOINT_SHORT[prog], None))
        for s, pop in TRIAL_SCORES:
            cells = {rr: f"{sub}|{prog}|{rr}" + (f"|{pop}" if pop else "") for rr in ("0.8", "0.7")}
            rows = {rr: find(trials, cell=c, score=s) for rr, c in cells.items()}
            if any(rows.values()):
                lab = SCORE_NAME.get(s, s) + (", eICU only" if pop else "")
                lines.append(("row", lab, (s, rows)))
    rh, top = 2.75, 12.0
    hp = len(lines) * rh
    H = top + hp + 24.0
    cv = st.Canvas(W, H)
    cv.letter(0, 2.0, "a", "Type I error of the adjusted analysis")
    cv.letter(110, 2.0, "b", "Sample size, adjusted over unadjusted")
    ax = _rows_axes(cv, 44, top, 56, len(lines), rh)
    bx = _rows_axes(cv, 116, top, 58, len(lines), rh)
    mc = 1.96 * np.sqrt(0.05 * 0.95 / 2000)
    ax.axvspan(0.05 - mc, 0.05 + mc, color=st.DIDYMOS_LIGHT, lw=0, zorder=0)
    st.refline(ax, 0.05)
    st.refline(bx, 1.0)
    for i, (kind, lab, k) in enumerate(lines):
        y = _row_y(top, rh, i)
        if kind == "header":
            cv.text(0, y, lab, ha="left", va="center", fontweight="bold")
            continue
        cv.text(42.0, y, lab, ha="right", va="center")
        s, rows = k
        col, mk, ms = _score_style(s)
        for rr, off, mfc in (("0.8", -0.2, col), ("0.7", 0.2, "white")):
            r = rows.get(rr)
            if not r:
                continue
            t1 = r["type1_adjusted"]
            ax.plot(t1["ci"], [i + off] * 2, color=col, lw=1.0, solid_capstyle="butt", zorder=2)
            ax.plot(t1["estimate"], i + off, ls="none", marker=mk, ms=ms, mfc=mfc, mec=col, mew=1.0, zorder=3)
            bx.plot(r["n_ratio_ci"], [i + off] * 2, color=col, lw=1.0, solid_capstyle="butt", zorder=2)
            bx.plot(r["n_ratio"], i + off, ls="none", marker=mk, ms=ms, mfc=mfc, mec=col, mew=1.0, zorder=3)
            sd.add("a+b", lab, cell=r["cell"], score=s, relative_risk=rr, type1_adjusted=t1["estimate"],
                   type1_ci_low=t1["ci"][0], type1_ci_high=t1["ci"][1], type1_unadjusted=r["type1_unadjusted"]["estimate"],
                   n_ratio=r["n_ratio"], n_ratio_ci_low=r["n_ratio_ci"][0], n_ratio_ci_high=r["n_ratio_ci"][1],
                   trials=r["simulations"])
    ax.set_xlim(0.025, 0.1)
    ax.set_xticks([0.03, 0.05, 0.07, 0.09])
    ax.set_xticklabels(["0.03", "0.05", "0.07", "0.09"])
    ax.set_xlabel("Type I error (95% CI)")
    bx.set_xlim(0.45, 1.1)
    bx.set_xticks([0.5, 0.75, 1.0])
    bx.set_xticklabels(["0.5", "0.75", "1"])
    bx.set_xlabel("Sample-size ratio (Monte Carlo 95% interval)")
    yl = top + hp + 13.0
    _legend(cv, [_handle(marker="o", ms=3.4, color=st.DIDYMOS), _handle(marker="s", ms=3.0, color=st.LGBM),
                 _handle(marker="D", ms=2.8, color=st.CLINICAL)],
            ["Didymos", "LightGBM", "Clinical score"], 0.0, yl, ncol=3)
    _legend(cv, [_handle(marker="o", ms=3.4, color=st.TEXT2),
                 _handle(marker="o", ms=3.4, mfc="white", mec=st.TEXT2, mew=1.0),
                 _handle(color=st.DIDYMOS_LIGHT, lw=4.0, ls="-")],
            ["Relative risk 0.8", "Relative risk 0.7", "Expected range at a true 0.05"], 76.0, yl, ncol=3)
    return cv, sd
