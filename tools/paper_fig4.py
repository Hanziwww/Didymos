"""Fig. 4: simulated treatment responses reproduce observational contrasts, and a null arm stays at chance.

a, the design (a schematic; lines illustrative). b, per pre-specified response: the observational contrast
(starters' real response minus their propensity-matched non-starters') and Didymos's version of it (the same pairs'
simulated factual futures), with 95% intervals and amendment 13's P for their difference (Holm). c, the null arm: the
share of prefixes whose two withhold arms differ in the expected direction, with exact 95% intervals. d, factual
accuracy: CRPS relative to persistence, and the coverage of the 80% interval. e, timing: the give-minus-withhold MAP
effect among starters, with the pre-specified cut (0.5-6 h before the start) and a cut 1 min before it.
"""

from __future__ import annotations

import numpy as np

import paper_figstyle as st
from paper_figdata import Inputs, find, p_holm

W, H0 = st.FULL, 149.5                 # the height with the programmatic schematic in panel a
ROWS = (  # by organ system: circulation, urine output, ventilation, renal replacement, glucose
    ("norepinephrine", "map", "Norepinephrine, MAP (mmHg)"), ("fluid_bolus", "map", "Fluid bolus, MAP (mmHg)"),
    ("fluid_bolus", "urine_output", "Fluid bolus, urine (mL)"), ("furosemide", "urine_output", "Furosemide, urine (mL)"),
    ("imv", "fio2_charted", "IMV, FiO2 records (n)"), ("imv", "spo2", "IMV, SpO2 (%)"),
    ("crrt", "potassium", "CRRT, potassium (mmol/L)"), ("crrt", "creatinine_24h", "CRRT, creatinine 24 h (mg/dL)"),
    ("insulin", "glucose", "Insulin, glucose (mg/dL)"))


def schematic(cv: st.Canvas, top: float) -> None:
    """Panel a: one prefix, the cut, three arms and the response window (drawn, not data)."""
    from matplotlib.patches import Rectangle

    ax = cv.axes(0, top, W, 34)
    ax.set_xlim(0, W)
    ax.set_ylim(34, 0)
    ax.axis("off")
    rng = np.random.default_rng(3)
    ax.add_patch(Rectangle((6, 15.5), 60, 3.0, facecolor=st.SLATE[100], edgecolor=st.SLATE[400], lw=1.0))
    for x in np.sort(rng.uniform(7, 65, 34)):
        ax.plot([x, x], [15.9, 18.1], color=st.SLATE[700], lw=1.0)
    ax.text(6, 13.0, "Real ICU record up to the cut", ha="left", va="bottom")
    ax.add_patch(Rectangle((68, 3.0), 50, 29.0, facecolor=st.BLUE[100], edgecolor="none", zorder=0))
    ax.plot([68, 68], [2.0, 32.0], color=st.SLATE[700], lw=1.0, ls=(0, (3, 2)))
    ax.text(68, 1.6, "Cut", ha="center", va="bottom")
    t = np.linspace(0, 1, 40)
    for yend, col, ls, lab in ((8.0, st.BLUE[500], "-", "Give: the recorded start"),
                               (17.0, st.SLATE[700], "-", "Withhold"),
                               (21.5, st.SLATE[400], (0, (2.5, 1.5)), "Null arm: second withhold")):
        for k in range(4):
            wob = 0.7 * np.sin(2 * np.pi * (t * (2 + k) + k / 3)) * t
            yy = 17.0 + (yend - 17.0) * (3 * t ** 2 - 2 * t ** 3) + wob + (k - 1.5) * 0.5 * t
            ax.plot(68 + 34 * t, yy, color=col, lw=1.0, ls=ls)
        ax.text(103.5, yend + (1.2 if yend > 20 else 0), lab, ha="left", va="center",
                color=st.SLATE[700] if col == st.SLATE[400] else col)


def _row_axes(cv, x, y, w, h):
    ax = cv.axes(x, y, w, h)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    return ax


def figure(inp: Inputs) -> tuple[st.Canvas, st.SourceData]:
    from matplotlib.lines import Line2D

    sd = st.SourceData("fig4")
    c1b = inp.json("f5/c1b.json", sd=sd)
    c1 = inp.json("c1/c1.json", sd=sd)
    fam6, fam7 = inp.family(6), inp.family(7)
    sd.source(inp.stats_path)
    art = "fig4a" in st.USE_ARTWORK
    shift = 5.0 + st.artwork_height("fig4a", W) + 3.0 - 45.5 if art else 0.0     # panel b moves below the artwork
    H = H0 + shift
    cv = st.Canvas(W, H)
    cv.letter(0, 2.0, "a", "Design of the what-if validation")
    if art:
        st.artwork(cv, sd, "fig4a", 0, 5.0, W, "a")
    else:
        schematic(cv, 6.0)

    # ---- b, c, d: one row per response ----------------------------------------------------------------------
    y0, rh = 52.0 + shift, 6.6
    cv.letter(0, y0 - 6.5, "b", "Contrast, observational and simulated")
    cv.letter(108, y0 - 6.5, "c", "Null arm")
    cv.letter(136, y0 - 6.5, "d", "Factual accuracy")
    ys = []
    y = y0 + 3.0
    for _key in ROWS:
        ys.append(y)
        y += rh
    ylo, yhi = y0 + 3.0 - rh / 2, ys[-1] + rh / 2
    cv.text(84.5, y0 - 1.2, "Obs. / Didymos", ha="right", va="center", color=st.TEXT2)
    cv.text(104.5, y0 - 1.2, r"$\mathit{P}$ (Holm)", ha="right", va="center", color=st.TEXT2)
    nx = cv.axes(110, ylo, 22, yhi - ylo)
    cx = cv.axes(137, ylo, 17, yhi - ylo)
    vx = cv.axes(160, ylo, 17, yhi - ylo)
    for ax in (nx, cx, vx):
        ax.set_ylim(yhi, ylo)
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
    nx.set_xlim(0.3, 0.7)
    nx.set_xticks([0.3, 0.5, 0.7])
    nx.set_xticklabels(["0.3", "0.5", "0.7"])
    nx.set_xlabel("Expected sign, share")
    st.refline(nx, 0.5)
    cx.set_xlim(0, 1.2)
    cx.set_xticks([0, 0.5, 1.0])
    cx.set_xticklabels(["0", "0.5", "1"])
    cx.set_xlabel("CRPS / persistence")
    st.refline(cx, 1.0)
    vx.set_xlim(0.5, 1.0)
    vx.set_xticks([0.5, 0.8, 1.0])
    vx.set_xticklabels(["0.5", "0.8", "1"])
    vx.set_xlabel("80% coverage")
    st.refline(vx, 0.8)
    for (tr, resp, label), y in zip(ROWS, ys, strict=True):
        x = c1b["factual"][tr][resp]
        o, t = x["observational_contrast"], x["twin_factual_contrast"]
        cv.text(36.0, y, label, ha="right", va="center")
        # b: an axis-free forest, scaled per row, with 0 marked
        lo = min(0.0, o["ci"][0], t["ci"][0])
        hi = max(0.0, o["ci"][1], t["ci"][1])
        pad = 0.12 * (hi - lo)
        ax = _row_axes(cv, 38.0, y - rh / 2 + 0.4, 30.0, rh - 0.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(1, -1)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        ax.axvline(0, color=st.REF, lw=1.0, ls=(0, (3, 2)), zorder=0)
        ax.plot(o["ci"], [-0.45, -0.45], color=st.CLINICAL, lw=1.0, solid_capstyle="butt")
        ax.plot(o["mean"], -0.45, ls="none", marker="s", ms=3.0, color=st.CLINICAL)
        ax.plot(t["ci"], [0.45, 0.45], color=st.DIDYMOS, lw=1.0, solid_capstyle="butt")
        ax.plot(t["mean"], 0.45, ls="none", marker="o", ms=3.4, color=st.DIDYMOS)
        d = 0 if abs(o["mean"]) >= 100 else (1 if abs(o["mean"]) >= 1 else 2)
        cv.text(84.5, y, f"{st.num(o['mean'], d)} / {st.num(t['mean'], d)}", ha="right", va="center")
        r6 = find(fam6, treatment=tr, response=resp)
        cv.text(104.5, y, st.p_value(*p_holm(r6)), ha="right", va="center")
        sd.add("b", "observational contrast", treatment=tr, response=resp, mean=o["mean"], ci_low=o["ci"][0],
               ci_high=o["ci"][1], pairs=o.get("pairs"))
        sd.add("b", "Didymos factual contrast", treatment=tr, response=resp, mean=t["mean"], ci_low=t["ci"][0],
               ci_high=t["ci"][1], pairs=t.get("pairs"), p_difference=(r6 or {}).get("p"),
               p_holm=(r6 or {}).get("p_holm"))
        # c: the null arm
        r7 = find(fam7, treatment=tr, response=resp)
        if r7:
            nx.plot([max(r7["ci"][0], 0.3), min(r7["ci"][1], 0.7)], [y, y], color=st.SLATE[700], lw=1.0,
                    solid_capstyle="butt")
            nx.plot(r7["estimate"], y, ls="none", marker="o", ms=3.2, mfc="white", mec=st.SLATE[700], mew=1.2)
            sd.add("c", "null arm, share with the expected sign", treatment=tr, response=resp, share=r7["estimate"],
                   ci_low=r7["ci"][0], ci_high=r7["ci"][1], k=r7["k"], n=r7["n"], p=r7.get("p"),
                   p_holm=r7.get("p_holm"))
        # d: accuracy of the factual futures (starters)
        f = x.get("factual_starter") or {}
        if f.get("crps") and f.get("crps_persistence"):
            ratio = f["crps"] / f["crps_persistence"]
            cx.plot(ratio, y, ls="none", marker="o", ms=3.4, color=st.DIDYMOS)
            vx.plot(f["coverage_80"], y, ls="none", marker="o", ms=3.4, color=st.DIDYMOS)
            sd.add("d", "factual starters", treatment=tr, response=resp, crps=f["crps"],
                   crps_persistence=f["crps_persistence"], crps_ratio=ratio, coverage_80=f["coverage_80"],
                   anchors=f.get("anchors"))
    cv.fig.legend([Line2D([], [], color=st.CLINICAL, lw=1.0, marker="s", ms=3.0),
                   Line2D([], [], color=st.DIDYMOS, lw=1.0, marker="o", ms=3.4)],
                  ["Observational contrast", "Didymos, simulated factual contrast"], loc="upper left", ncol=2,
                  bbox_to_anchor=(2.0 / W, 1 - (yhi + 3.0) / H), columnspacing=1.4, handletextpad=0.4,
                  borderaxespad=0)

    # ---- e: timing of the cut ------------------------------------------------------------------------------------
    ye = yhi + 12.0
    cv.letter(0, ye, "e", "Timing of the cut, MAP")
    ex = cv.axes(38, ye + 6.0, 52, 12.0)
    ex.set_ylim(2.6, -0.6)
    ex.yaxis.set_visible(False)
    ex.spines["left"].set_visible(False)
    st.refline(ex, 0.0)
    for i, (tr, lab) in enumerate((("norepinephrine", "Norepinephrine"), ("fluid_bolus", "Fluid bolus"))):
        pre = c1["treatments"][tr]["responses"]["map"]
        now = c1b["factual"][tr]["map"]["twin_contrast_starter"]
        yy = i * 1.6
        cv.text(36.0, ye + 6.0 + (yy + 0.6) / 3.2 * 12.0, lab, ha="right", va="center")
        ex.plot(pre["mean_effect_ci"], [yy, yy], color=st.SLATE[700], lw=1.0, solid_capstyle="butt")
        ex.plot(pre["mean_effect"], yy, ls="none", marker="o", ms=3.2, mfc="white", mec=st.SLATE[700], mew=1.2)
        ex.plot(now["ci"], [yy + 0.7, yy + 0.7], color=st.DIDYMOS, lw=1.0, solid_capstyle="butt")
        ex.plot(now["mean"], yy + 0.7, ls="none", marker="o", ms=3.4, color=st.DIDYMOS)
        sd.add("e", "pre-specified cut, 0.5-6 h before", treatment=tr, effect=pre["mean_effect"],
               ci_low=pre["mean_effect_ci"][0], ci_high=pre["mean_effect_ci"][1], prefixes=pre.get("scored"))
        sd.add("e", "cut 1 min before", treatment=tr, effect=now["mean"], ci_low=now["ci"][0], ci_high=now["ci"][1],
               anchors=now.get("anchors"))
    ex.set_xlim(-2.0, 2.0)
    ex.set_xticks([-2, -1, 0, 1, 2])
    ex.set_xticklabels([st.num(-2, 0), st.num(-1, 0), "0", "1", "2"])
    ex.set_xlabel("Give minus withhold, starters (mmHg)")
    cv.fig.legend([Line2D([], [], color=st.SLATE[700], lw=1.0, marker="o", ms=3.2, mfc="white", mec=st.SLATE[700],
                          mew=1.2),
                   Line2D([], [], color=st.DIDYMOS, lw=1.0, marker="o", ms=3.4)],
                  ["Cut 0.5–6 h before (pre-specified)", "Cut 1 min before"], loc="upper left", ncol=1,
                  bbox_to_anchor=(96.0 / W, 1 - (ye + 6.5) / H), handletextpad=0.4, borderaxespad=0)
    return cv, sd
