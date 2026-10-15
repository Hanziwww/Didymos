"""Fig. 7: the Didymos toolkit, and smaller trials with Didymos prognoses.

a, the toolkit (a schematic): a record or a cohort specification in, Didymos's simulated futures, seven tools, and
outputs that each carry their evidence and caveats. b, Trial designer, prognostic covariate adjustment: the sample
size an adjusted analysis needs relative to an unadjusted one (relative risk 0.8, 80% power, two-sided 0.05), with
Monte Carlo 95% intervals, for Didymos, LightGBM and the clinical scores; on eICU, APACHE IVa. c, enrichment: the
sample size when enrolling only the highest-risk fraction by each score. d, a synthetic patient through Forecast,
What-if and Explain (cleared for release by the study lead on 2026-10-11; ``case`` keeps the earlier internal
version).
"""

from __future__ import annotations

import json

import paper_figstyle as st
from paper_figdata import Inputs, find

W, H0 = st.FULL, 154.0                 # the height with the programmatic schematic in panel a
ENDPOINTS = (("L24x", "icu_death@-1"), ("L24x", "hospital_death@0"), ("L24", "aki_2@2880"), ("L24", "rrt_onset@2880"),
             ("L24", "vasopressor_onset@1440"), ("L24", "imv_onset@1440"))
CLIN = ("sapsii", "sofa", "oasis", "news")
CLIN_NAME = {"sapsii": "SAPS II", "sofa": "SOFA", "oasis": "OASIS", "news": "NEWS", "apache_iva": "APACHE IVa"}
TOOLS = (("Forecast", "risks and curves", "Figs 2, 3"),
         ("What-if", "give or withhold", "Fig. 4"),
         ("Explain", "editing, knockout", "Fig. 6"),
         ("Similar patients", "nearest stays", "Supplementary"),
         ("Cohort synthesizer", "audited synthetic cohorts", "Fig. 5"),
         ("Trial designer", "adjustment and enrichment", "panels b, c"),
         ("Teaching cases", "synthetic decision points", "panel d"))


def schematic(cv: st.Canvas, top: float) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    ax = cv.axes(0, top, W, 44)
    ax.set_xlim(0, W)
    ax.set_ylim(44, 0)
    ax.axis("off")

    def box(x, y, w, h, fill, edge=st.SLATE[400]):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", facecolor=fill,
                                    edgecolor=edge, lw=1.0))

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=6, color=st.SLATE[700], lw=1.0))

    box(0.5, 9, 24, 20, st.SLATE[100])
    for i, t in enumerate(("A record at", "an hour", "or a cohort", "specification")):
        ax.text(12.5, 13 + i * 3.8, t, ha="center", va="center", fontweight="bold" if i in (0, 1, 2) else "normal")
    arrow(25.0, 28.5, 19)
    box(28.5, 9, 22, 20, st.BLUE[100], st.BLUE[500])
    for i, t in enumerate(("Didymos 1.0", "simulated futures", "under the grammar", "and support masks")):
        ax.text(39.5, 13 + i * 3.8, t, ha="center", va="center", color=st.DIDYMOS if i == 0 else st.TEXT,
                fontweight="bold" if i == 0 else "normal")
    arrow(51.0, 54.5, 19)
    gx, bh, x0, x1 = 1.6, 12.5, 54.5, 154.0
    for i, (name, what, ev) in enumerate(TOOLS):
        row = 0 if i < 4 else 1
        n = 4 if row == 0 else 3
        bw = (x1 - x0 - (n - 1) * gx) / n
        col = i if row == 0 else i - 4
        x = x0 + col * (bw + gx)
        y = 3.5 + row * (bh + 2.5)
        box(x, y, bw, bh, "white", st.BLUE[500])
        ax.text(x + bw / 2, y + 3.2, name, ha="center", va="center", fontweight="bold")
        ax.text(x + bw / 2, y + 6.6, what, ha="center", va="center", color=st.SLATE[700])
        ax.text(x + bw / 2, y + 9.9, ev, ha="center", va="center", color=st.DIDYMOS)
    arrow(154.5, 157.5, 19)
    box(157.5, 9, 22, 20, st.SLATE[100])
    for i, t in enumerate(("JSON and an", "HTML card;", "evidence and", "caveats each")):
        ax.text(168.5, 13 + i * 3.8, t, ha="center", va="center")


CASE = "figures/internal/case.json"
CAUSES = (("death", "Death"), ("vasopressor", "Vasopressor"), ("imv", "IMV"), ("rrt", "RRT"))


def case(inp: Inputs) -> tuple[st.Canvas, st.SourceData]:
    """Fig. 7d on its own (internal): one teaching case's synthetic patient (``tools/server/paper_case_data.py``):
    history, Forecast, What-if and Explain. Not for release unless the study lead clears it."""
    sd = st.SourceData("fig7_case")
    p = inp.path(CASE)
    sd.source(p)
    c = json.loads(p.read_text(encoding="utf-8"))
    cv = st.Canvas(W, 84.0)
    cv.text(180, 1.5, "INTERNAL: a synthetic patient derived from credentialed data; not cleared for release",
            ha="right", va="top", color=st.FLAG)
    # history
    cv.letter(0, 6.0, "a", "History")
    ax = cv.axes(9, 15, 28, 36)
    h = c["hours"]
    for (name, x), col in zip(c["series"].items(), (st.DIDYMOS, st.TEXT2, st.LGBM, st.TABPFN), strict=False):
        if x["points"]:
            t, v = zip(*x["points"], strict=True)
            ax.plot(t, v, color=col, lw=1.5, marker="o", ms=2.4, label=f"{name} ({x['unit']})")
            for t_, v_ in x["points"]:
                sd.add("a", name, hours=t_, value=v_)
    ax.axvline(h, color=st.AXIS, lw=1.0, ls=(0, (3, 2)))
    ax.set_xlabel("Hours since ICU admission")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.25, -0.3), frameon=False, borderaxespad=0.0)
    # forecast
    cv.letter(42, 6.0, "b", "Forecast")
    fx = cv.axes(51, 15, 28, 36)
    inc = c["incidence"]
    for (key, lab), col in zip(CAUSES, (st.CLINICAL, st.DIDYMOS, st.LGBM, st.TABPFN), strict=True):
        if key in inc:
            y = [float("nan") if v is None else v for v in inc[key]["e2"]]
            fx.plot(inc["hours"], y, color=col, lw=1.5, label=lab)
            for hh, v in zip(inc["hours"], y, strict=True):
                sd.add("b", lab, hours=hh, cumulative_incidence=v)
    fx.set_xlim(0, 48)
    fx.set_xticks([0, 24, 48])
    fx.set_xlabel("Hours after the decision point")
    fx.set_ylabel("Cumulative incidence (hazard)")
    fx.legend(loc="upper left", bbox_to_anchor=(-0.25, -0.3), ncol=2, frameon=False, borderaxespad=0.0,
              columnspacing=1.0)
    # what-if
    cv.letter(86, 6.0, "c", "What-if, norepinephrine")
    wi = c["whatif"].get("norepinephrine", {})
    resp = (wi.get("responses") or {}).get("map", {})
    wx = cv.axes(104, 15, 20, 36)
    arms = [a for a in ("withhold", "give_low", "give_median", "give_high") if a in resp.get("arms", {})]
    for i, a in enumerate(arms):
        x = resp["arms"][a]
        wx.plot([x["q10"], x["q90"]], [i, i], color=st.DIDYMOS if a != "withhold" else st.TEXT2, lw=1.0)
        wx.plot(x["q50"], i, ls="none", marker="o", ms=3.4, color=st.DIDYMOS if a != "withhold" else st.TEXT2)
        cv.text(102.5, 15 + 36 * (i + 0.5) / len(arms), a.replace("give_", "give, ").replace("_", " "), ha="right",
                va="center")
        sd.add("c", a, map_q10=x["q10"], map_median=x["q50"], map_q90=x["q90"], futures=x["n_valid"])
    wx.set_ylim(len(arms) - 0.5, -0.5)
    wx.yaxis.set_visible(False)
    wx.spines["left"].set_visible(False)
    wx.set_xlabel("MAP, 6 h (mmHg)")
    # explain
    cv.letter(128, 6.0, "d", "Explain")
    gs = c["explain"]["groups"][:6][::-1]
    ex = cv.axes(158, 15, 19, 36)
    for i, g in enumerate(gs):
        col = st.DIDYMOS if g["edit"] < 0 else st.FLAG
        ex.plot([0, g["edit"]], [i, i], color=col, lw=1.5)
        ex.plot(g["edit"], i, ls="none", marker="o", ms=3.0, color=col)
        sd.add("d", g["group"], edit=g["edit"], knockout=g.get("knockout"))
    ex.set_yticks(range(len(gs)))
    ex.set_yticklabels([g["group"].replace("_", " ").replace("|", ", ") for g in gs])
    ex.axvline(0, color=st.REF, lw=1.0)
    ex.set_xlabel("Editing (logit)")
    return cv, sd


def figure(inp: Inputs) -> tuple[st.Canvas, st.SourceData]:
    from matplotlib.lines import Line2D

    sd = st.SourceData("fig7")
    v4 = inp.json("v4/v4.json", toolkit=True, sd=sd)
    trials = inp.stats.get("trials", [])
    sd.source(inp.stats_path)
    art = "fig7a" in st.USE_ARTWORK
    shift = 5.0 + st.artwork_height("fig7a", W) + 3.0 - 46.0 if art else 0.0     # panels b-d move below the artwork
    H = H0 + shift
    cv = st.Canvas(W, H)
    cv.letter(0, 2.0, "a", "The Didymos toolkit")
    if art:
        st.artwork(cv, sd, "fig7a", 0, 5.0, W, "a")
    else:
        schematic(cv, 5.0)

    def tr(cell, score):
        return find(trials, cell=cell, score=score)

    # ---- b: covariate adjustment ------------------------------------------------------------------------------
    y0 = 46.0 + shift
    cv.letter(0, y0, "b", "Trial designer: covariate adjustment")
    rows = [(d, p, f"{d}|{p}|0.8", False) for d, p in ENDPOINTS] + [("L24x", "icu_death@-1", "L24x|icu_death@-1|0.8|eicu",
                                                                      True)]
    top, rh = y0 + 10.0, 5.6
    ys = [top + i * rh + (2.0 if eicu else 0) for i, (_d, _p, _c, eicu) in enumerate(rows)]
    ylo, yhi = top - rh / 2, ys[-1] + rh / 2
    bx = cv.axes(36, ylo, 46, yhi - ylo)
    bx.set_ylim(yhi, ylo)
    bx.yaxis.set_visible(False)
    bx.spines["left"].set_visible(False)
    bx.set_xlim(0.5, 1.05)
    bx.set_xticks([0.5, 0.75, 1.0])
    bx.set_xticklabels(["0.5", "0.75", "1"])
    bx.set_xlabel("Sample size, adjusted over unadjusted")
    st.refline(bx, 1.0)
    cv.text(97.0, top - 6.0, "Didymos", ha="right", va="center", color=st.TEXT2)
    for (_d, prog, cell, eicu), y in zip(rows, ys, strict=True):
        c = v4["cells"][cell]
        lab = st.ENDPOINT_SHORT[prog] + (", eICU" if eicu else "")
        cv.text(34.0, y, lab, ha="right", va="center")
        others = [s for s in CLIN if s in c["scores"]]
        best = None
        if others:
            best = min(others, key=lambda s: c["scores"][s]["adjustment"]["n_ratio"])
        if eicu and "apache_iva" in c["scores"]:
            best = "apache_iva"
            others = []
        for s in others:
            if s != best:
                bx.plot(c["scores"][s]["adjustment"]["n_ratio"], y, ls="none", marker="|", ms=4.0, mew=1.0,
                        color=st.LEARNER, zorder=1)
        for s, col, mk, off in (("lgbm_pooled", st.LGBM, "s", -1.05), (best, st.CLINICAL, "D", 1.05),
                                ("twin", st.DIDYMOS, "o", 0.0)):
            if s is None or s not in c["scores"]:
                continue
            a = c["scores"][s]["adjustment"]
            t = tr(cell, s)
            if t:
                bx.plot(t["n_ratio_ci"], [y + off] * 2, color=col, lw=1.0, solid_capstyle="butt", zorder=2)
            bx.plot(a["n_ratio"], y + off, ls="none", marker=mk, ms=4.2 if s == "twin" else 3.2, color=col, zorder=3)
            sd.add("b", st.SCORE.get(s, CLIN_NAME.get(s, s)), cell=cell, n_ratio=a["n_ratio"],
                   ci_low=(t or {}).get("n_ratio_ci", [None])[0], ci_high=(t or {}).get("n_ratio_ci", [None, None])[1],
                   type1_adjusted=a["type1_adjusted"], type1_unadjusted=a["type1_unadjusted"])
        for s in others:
            if s != best:
                sd.add("b", CLIN_NAME[s], cell=cell, n_ratio=c["scores"][s]["adjustment"]["n_ratio"])
        cv.text(97.0, y, f"{(1 - c['scores']['twin']['adjustment']['n_ratio']) * 100:.0f}% fewer", ha="right",
                va="center")
    cv.fig.legend([Line2D([], [], color=st.DIDYMOS, lw=1.0, marker="o", ms=4.2),
                   Line2D([], [], color=st.LGBM, lw=1.0, marker="s", ms=3.2),
                   Line2D([], [], color=st.CLINICAL, lw=1.0, marker="D", ms=3.2),
                   Line2D([], [], ls="none", marker="|", ms=4.0, mew=1.0, color=st.LEARNER)],
                  ["Didymos", "LightGBM", "Best clinical score (eICU: APACHE IVa)", "Other clinical scores"],
                  loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1 - (yhi + 9.0) / H), columnspacing=1.0,
                  handletextpad=0.4, borderaxespad=0)
    t1 = [r["type1_adjusted"]["estimate"] for r in trials if r["cell"].split("|")[2] == "0.8"]
    if t1:
        inside = sum(0.04 <= x <= 0.06 for x in t1)
        sd.add("b", "type I error check, adjusted, relative risk 0.8", cells=len(t1), within_0_04_0_06=inside,
               minimum=min(t1), maximum=max(t1))

    # ---- c: enrichment ----------------------------------------------------------------------------------------
    cv.letter(104, y0, "c", "Trial designer: enrichment")
    fr = [1.0, 0.75, 0.5, 0.25]
    for i, (cell, title) in enumerate((("L24x|icu_death@-1|0.8", "ICU death"), ("L24|aki_2@2880|0.8",
                                                                                 "AKI stage ≥2, 48 h"))):
        cx = cv.axes(116 + i * 33.0, y0 + 12.0, 27.0, 30.0)
        c = v4["cells"][cell]
        best = min((s for s in CLIN if s in c["scores"]), key=lambda s: c["scores"][s]["enrichment"][-1]
                   ["n_vs_unenriched"])
        for s, col, mk, lw in [(s, st.LEARNER, None, 1.0) for s in CLIN if s != best] + \
                [(best, st.CLINICAL, "D", 1.5), ("lgbm_pooled", st.LGBM, "s", 1.5), ("twin", st.DIDYMOS, "o", 1.5)]:
            e = c["scores"][s]["enrichment"]
            yv = [x["n_vs_unenriched"] for x in e]
            cx.plot(fr, yv, color=col, lw=lw, marker=mk, ms=4.0 if mk else 0, zorder=3 if mk else 1,
                    **(st.RING if mk else {}))
            for f_, x in zip(fr, e, strict=True):
                sd.add("c", CLIN_NAME.get(s, st.SCORE.get(s, s)), cell=cell, fraction=f_,
                       n_vs_unenriched=x["n_vs_unenriched"], screened=x["screened"], event_rate=x["event_rate"])
        cx.set_xlim(1.05, 0.2)
        cx.set_xticks([1.0, 0.5, 0.25])
        cx.set_xticklabels(["100%", "50%", "25%"])
        cx.set_yscale("log")
        cx.set_ylim(0.15, 1.2)
        cx.set_yticks([0.2, 0.5, 1.0])
        cx.set_yticklabels(["0.2", "0.5", "1"] if i == 0 else [])
        cx.minorticks_off()
        cv.text(116 + i * 33.0, y0 + 10.6, title, ha="left", va="bottom")
        if i == 0:
            cx.set_ylabel("Sample size, relative")
    cv.text(116 + 30.0, y0 + 50.5, "Highest-risk fraction enrolled", ha="center", va="top")

    # ---- d: a synthetic patient (cleared for release by the study lead, 2026-10-11) ----------------------------------
    yd = y0 + 68.0
    cv.letter(0, yd, "d", "A synthetic patient through Forecast, What-if and Explain")
    p = inp.path(CASE)
    sd.source(p)
    case_panel(cv, sd, json.loads(p.read_text(encoding="utf-8")), yd + 8.0)
    return cv, sd


def _group_name(g: str) -> str:
    """A factor group's display name: 'labs|6-24h' -> 'Labs, 6–24 h'."""
    name, _, window = g.replace("_", " ").partition("|")
    name = {"statics": "Admission details", "fluids output": "Fluid output"}.get(name, name)
    window = window.replace("h", " h").replace("-", "–")
    return name[:1].upper() + name[1:] + (f", {window}" if window else "")


def case_panel(cv: st.Canvas, sd: st.SourceData, c: dict, top: float) -> None:
    """Panel d: a synthetic patient's history before the decision point, its forecast (cumulative incidence from the
    hazards), a what-if of norepinephrine on MAP, and the factor groups that move its risk of death within 24 h (the
    Didymos probe's log-odds, editing)."""
    hgt = 22.0
    # history
    cv.text(8, top - 1.2, "History", ha="left", va="bottom")
    ax = cv.axes(8, top, 30, hgt)
    h = c["hours"]
    hist = [(name, x) for name, x in c["series"].items() if x["points"]]
    ends = []
    for (name, x), col in zip(hist, (st.BLUE[500], st.SLATE[700], st.SLATE[500]), strict=False):
        t, v = zip(*x["points"], strict=True)
        ax.plot(t, v, color=col, lw=1.5, marker="o", ms=3.6, **st.RING)
        ends.append((v[-1], name, col))
        for t_, v_ in x["points"]:
            sd.add("d", f"history, {name} ({x['unit']})", hours=t_, value=v_)
    ax.axvline(h, color=st.SLATE[700], lw=1.0, ls=(0, (3, 2)))
    ax.set_xlim(0, h + 3.2)
    lo_, hi_ = ax.get_ylim()
    gap = 3.0 / hgt * (hi_ - lo_)                       # 3 mm between the labels
    ends.sort(key=lambda e: -e[0])
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = min(ys[i], ys[i - 1] - gap)
    for (_v, name, col), yy in zip(ends, ys, strict=True):     # labels right of the decision time
        ax.text(h + 0.2, yy, name, ha="left", va="center", color=col)
    ax.set_xticks([0, 2, 4])
    ax.set_xlabel("Hours in the ICU")
    # forecast
    cv.text(58, top - 1.2, "Forecast", ha="left", va="bottom")
    fx = cv.axes(58, top, 30, hgt)
    inc = c["incidence"]
    for (key, lab), col in ((c_, st.OUTCOME[c_[0]]) for c_ in CAUSES):     # one hue per outcome
        if key not in inc:
            continue
        y = [float("nan") if v is None else v for v in inc[key]["e2"]]
        fx.plot(inc["hours"], y, color=col, lw=1.5, label=lab)
        for hh, v in zip(inc["hours"], y, strict=True):
            sd.add("d", f"forecast, {lab}", hours=hh, cumulative_incidence=v)
    fx.legend(loc="center right", bbox_to_anchor=(1.02, 0.45), frameon=False, handlelength=1.2, borderaxespad=0.0,
              labelspacing=0.25)
    fx.set_xlim(0, 48)
    fx.set_xticks([0, 24, 48])
    fx.set_xlabel("Hours ahead")
    fx.set_ylabel("Cumulative risk")
    # what-if
    cv.text(110, top - 1.2, "What-if, norepinephrine", ha="left", va="bottom")
    resp = ((c["whatif"].get("norepinephrine") or {}).get("responses") or {}).get("map", {})
    arms = [a for a in ("withhold", "give_low", "give_median", "give_high") if a in resp.get("arms", {})]
    wx = cv.axes(126, top, 16, hgt)
    for i, a in enumerate(arms):
        x = resp["arms"][a]
        col = {"withhold": st.SLATE[700], "give_low": st.BLUE[300], "give_median": st.BLUE[500],
               "give_high": st.BLUE[700]}[a]                         # the dose as an ordinal tone
        wx.plot([x["q10"], x["q90"]], [i, i], color=col, lw=1.0, solid_capstyle="butt")
        wx.plot(x["q50"], i, ls="none", marker="o", ms=3.2, color=col)
        cv.text(124.5, top + hgt * (i + 0.5) / len(arms), {"withhold": "Withhold", "give_low": "Low dose",
                                                           "give_median": "Median dose",
                                                           "give_high": "High dose"}[a], ha="right", va="center")
        sd.add("d", f"what-if, {a}", map_q10=x["q10"], map_median=x["q50"], map_q90=x["q90"], futures=x["n_valid"])
    wx.set_ylim(len(arms) - 0.5, -0.5)
    wx.yaxis.set_visible(False)
    wx.spines["left"].set_visible(False)
    wx.set_xlabel("Mean MAP, 6 h (mmHg)")
    # explain
    cv.text(146, top - 1.2, "Explain: death within 24 h", ha="left", va="bottom")
    gs = c["explain"]["groups"][:6][::-1]
    ex = cv.axes(168, top, 11, hgt)
    for i, g in enumerate(gs):
        ex.plot([0, g["edit"]], [i, i], color=st.DIDYMOS, lw=1.5, solid_capstyle="butt")
        ex.plot(g["edit"], i, ls="none", marker="o", ms=2.8, color=st.DIDYMOS)
        cv.text(166.5, top + hgt * (len(gs) - i - 0.5) / len(gs), _group_name(g["group"]), ha="right", va="center")
        sd.add("d", f"explain, {g['group']}", edit=g["edit"], knockout=g.get("knockout"))
    ex.set_ylim(-0.5, len(gs) - 0.5)
    ex.yaxis.set_visible(False)
    ex.spines["left"].set_visible(False)
    ex.axvline(0, color=st.REF, lw=1.0)
    ex.set_xlabel("Log-odds")
