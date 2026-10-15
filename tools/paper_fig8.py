"""Fig. 8: the clinician study (amendment 12), and its template.

``figure``: the pre-specified analysis of the answers received for packets R01-R20 (protocol amendment 12; the
figure was added on 2026-10-11). It reads only the analysis's
pooled aggregates (``runs/paper/figures/internal/clinician_answers/aggregates.json``, extracted on the host without
any rater-level or item-level record), shows each module's pre-specified estimate with its interval and the recorded
verdict unchanged, and its legend states the data's status (authenticity attested by the study lead; the protocol's
export checks not run).

``template``: the same layout drawn on the analysis chain's simulated study (``tools/clinician_study_analysis.py
fake``, then ``tables`` and ``secondary``: no data, no ratings), kept internal.

a, module A (realism): each rater's accuracy, the pooled accuracy with its 90% interval against the equivalence
margins. b, module B1: twin minus real plausibility by treatment, with the non-inferiority margin. c, module B2: the
explanation methods' contrasts. d, module C: the ICU-death Brier score unaided, aided, and of the card and the
comparators. e, the questionnaire.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import paper_figstyle as st
from paper_figdata import Inputs

W, H = st.FULL, 112.0
SIMULATED = Path("runs/paper/figures/internal/clinician_simulated/analysis/secondary.json")
TREAT = {"norepinephrine": "Norepinephrine", "fluid_bolus": "Fluid bolus", "furosemide": "Furosemide",
         "imv": "IMV", "crrt": "CRRT", "insulin": "Insulin"}


def template(inp: Inputs, source: Path = SIMULATED) -> tuple[st.Canvas, st.SourceData]:
    import json

    from matplotlib.patches import Rectangle

    sd = st.SourceData("fig8_template")
    sd.source(source)
    s = json.loads(Path(source).read_text(encoding="utf-8"))
    simulated = "clinician_simulated" in str(source)
    cv = st.Canvas(W, H)
    if simulated:
        cv.fig.text(0.5, 0.5, "TEMPLATE – SIMULATED STUDY – NOT A RESULT", ha="center", va="center",
                    rotation=18, fontsize=st.PT, fontweight="bold", color="#F2C4A5", gid="watermark", zorder=0)

    # ---- a: realism ----------------------------------------------------------------------------------------------
    cv.letter(0, 2.0, "a", "Realism")
    a = s["A"]
    per = s["_internal"]["a_per_rater"]
    ax = cv.axes(10, 12, 40, 34)
    ax.add_patch(Rectangle((-0.5, 0.40), 20, 0.20, facecolor=st.DIDYMOS_LIGHT, edgecolor="none", zorder=0))
    st.refline(ax, 0.5, axis="y")
    acc = sorted(r["accuracy"] for r in per)
    ax.plot(np.arange(len(acc)), acc, ls="none", marker="o", ms=2.8, mfc="white", mec=st.TEXT2, mew=1.0)
    p = a["primary"]["accuracy_90"]
    ax.plot([len(acc) + 1.5] * 2, [p["lo"], p["hi"]], color=st.DIDYMOS, lw=1.5)
    ax.plot(len(acc) + 1.5, p["est"], ls="none", marker="D", ms=3.6, color=st.DIDYMOS)
    ax.set_xlim(-1, len(acc) + 3)
    ax.set_xticks([])
    ax.set_ylim(0.2, 0.8)
    ax.set_yticks([0.2, 0.4, 0.5, 0.6, 0.8])
    ax.set_ylabel("Accuracy, real or synthetic")
    ax.set_xlabel("Raters, sorted; pooled (90% CI)")
    cv.text(11, 54.0, "Shaded: the equivalence margins, 0.40–0.60", ha="left", va="center", color=st.TEXT2)
    for r in per:
        sd.add("a", "rater accuracy", rater=r["rater"], accuracy=r["accuracy"], n=r["n"])
    sd.add("a", "pooled accuracy", est=p["est"], lo=p["lo"], hi=p["hi"], level=p["level"])

    # ---- b: simulated responses ------------------------------------------------------------------------------------
    cv.letter(58, 2.0, "b", "Simulated responses")
    bt = s["B1"]["by_treatment"]
    names = [t for t in TREAT if t in bt]
    rows = [*names, "overall"]
    by0, brh = 12.0, 4.6
    bx = cv.axes(82, by0 - brh / 2, 30, len(rows) * brh)
    bx.set_ylim(by0 + len(rows) * brh - brh / 2, by0 - brh / 2)
    bx.yaxis.set_visible(False)
    bx.spines["left"].set_visible(False)
    bx.set_xlim(-2.0, 1.0)
    bx.set_xticks([-2, -1, -0.5, 0, 1])
    bx.set_xticklabels([st.num(-2, 0), st.num(-1, 0), st.num(-0.5, 1), "0", "1"])
    bx.set_xlabel("Twin minus real plausibility")
    st.refline(bx, 0.0)
    st.refline(bx, -0.5, color=st.FLAG)
    for i, t in enumerate(rows):
        y = by0 + i * brh
        x = s["B1"]["primary"]["twin_minus_real"] if t == "overall" else bt[t]["twin_minus_real"]
        cv.text(80.0, y, "Overall (insulin excluded)" if t == "overall" else TREAT[t], ha="right", va="center",
                fontweight="bold" if t == "overall" else "normal")
        bx.plot([max(x["lo"], -2.0), min(x["hi"], 1.0)], [y, y], color=st.DIDYMOS, lw=1.0, solid_capstyle="butt")
        bx.plot(x["est"], y, ls="none", marker="o", ms=3.4, color=st.DIDYMOS)
        sd.add("b", t, est=x["est"], lo=x["lo"], hi=x["hi"], items=x.get("items"))

    # ---- c: explanations -------------------------------------------------------------------------------------------
    cv.letter(120, 2.0, "c", "Explanations")
    b2 = s["B2"]["primary"]
    crow = (("edit_minus_random", "Editing minus random"), ("knockout_minus_random", "Knockout minus random"),
            ("edit_minus_knockout", "Editing minus knockout"))
    cx = cv.axes(152, 12.0 - 2.3, 26, len(crow) * 4.6)
    cx.set_ylim(12.0 + len(crow) * 4.6 - 2.3, 12.0 - 2.3)
    cx.yaxis.set_visible(False)
    cx.spines["left"].set_visible(False)
    cx.set_xlim(-0.5, 1.0)
    cx.set_xticks([-0.5, 0, 0.5, 1.0])
    cx.set_xticklabels([st.num(-0.5, 1), "0", "0.5", "1"])
    cx.set_xlabel("Relevance difference")
    st.refline(cx, 0.0)
    for i, (k, lab) in enumerate(crow):
        y = 12.0 + i * 4.6
        x = b2[k]
        cv.text(150.0, y, lab, ha="right", va="center")
        cx.plot([x["lo"], x["hi"]], [y, y], color=st.DIDYMOS, lw=1.0, solid_capstyle="butt")
        cx.plot(x["est"], y, ls="none", marker="o", ms=3.4, color=st.DIDYMOS)
        sd.add("c", k, est=x["est"], lo=x["lo"], hi=x["hi"])

    # ---- d: decision support -----------------------------------------------------------------------------------------
    cv.letter(58, 58.0, "d", "Decision support, ICU death")
    br = s["C"]["icu_death"]["brier"]
    drow = (("unaided", "Clinicians, unaided", st.TEXT2), ("aided", "Clinicians, with the card", st.DIDYMOS),
            ("twin_card", "The card alone", st.DIDYMOS), ("lgbm", "LightGBM", st.LGBM), ("sapsii", "SAPS II",
                                                                                         st.CLINICAL))
    dy0, drh = 68.0, 4.8
    dx = cv.axes(96, dy0 - drh / 2, 30, len(drow) * drh)
    dx.set_ylim(dy0 + len(drow) * drh - drh / 2, dy0 - drh / 2)
    dx.yaxis.set_visible(False)
    dx.spines["left"].set_visible(False)
    dx.set_xlim(0, 0.25)
    dx.set_xticks([0, 0.1, 0.2])
    dx.set_xticklabels(["0", "0.1", "0.2"])
    dx.set_xlabel("Brier score")
    for i, (k, lab, col) in enumerate(drow):
        y = dy0 + i * drh
        x = br[k]
        cv.text(94.0, y, lab, ha="right", va="center")
        dx.plot([x["lo"], x["hi"]], [y, y], color=col, lw=1.0, solid_capstyle="butt")
        dx.plot(x["est"], y, ls="none", marker="o", ms=3.4, color=col)
        sd.add("d", k, est=x["est"], lo=x["lo"], hi=x["hi"])
    pc = s["C"]["primary"]["brier_aided_minus_unaided"]
    cv.text(58, dy0 + len(drow) * drh + 9.0, f"Aided minus unaided: {st.num(pc['est'], 3, sign=True)} "
                                             f"({st.ci_text(pc['lo'], pc['hi'])})", ha="left", va="center")

    # ---- e: the questionnaire --------------------------------------------------------------------------------------
    cv.letter(134, 58.0, "e", "Usability and trust")
    q = s["questionnaire"]
    for i, (k, lab, rng_) in enumerate((("sus", "SUS (0–100)", (0, 100)), ("trust", "Trust (1–7)",
                                                                                 (1, 7)))):
        ex = cv.axes(146, 68.0 + i * 16.0, 30, 3.0)
        ex.yaxis.set_visible(False)
        ex.spines["left"].set_visible(False)
        ex.set_xlim(*rng_)
        x = q.get(k, {})
        if x:
            ex.plot(x["iqr"], [0.5, 0.5], color=st.DIDYMOS, lw=1.5, solid_capstyle="butt")
            ex.plot(x["median"], 0.5, ls="none", marker="o", ms=3.4, color=st.DIDYMOS)
            sd.add("e", k, median=x["median"], q25=x["iqr"][0], q75=x["iqr"][1], mean=x["mean"])
        ex.set_ylim(0, 1)
        cv.text(144.0, 69.5 + i * 16.0, lab, ha="right", va="center")
    cv.text(0, 104.0, "Template: the study's formal analysis (exports through collect, then the pre-specified models) "
                      "replaces these numbers; until then nothing here is a result.", ha="left", va="center",
            color=st.FLAG)
    return cv, sd


AGGREGATES = Path("runs/paper/figures/internal/clinician_answers/aggregates.json")


def _forest(cv: st.Canvas, x: float, top: float, w: float, n: int, rh: float, xlim: tuple, ticks: list, labels: list,
            xlabel: str):
    """Axes for n rows of rh mm (row i centred at top + (i + 0.5) rh), y in row units."""
    ax = cv.axes(x, top, w, n * rh)
    ax.set_ylim(n - 0.5, -0.5)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    return ax


def figure(inp: Inputs, source: Path = AGGREGATES) -> tuple[st.Canvas, st.SourceData]:
    """The clinician study, explanatory: estimates with their intervals against natural references (chance, zero),
    from the pre-specified analysis's pooled aggregates; the pre-registered decision rules are reported in the
    Methods, not drawn."""
    import json

    sd = st.SourceData("fig8")
    sd.source(source)
    g = json.loads(Path(source).read_text(encoding="utf-8"))
    s, lme = g["secondary"], g["primary_lme4"]
    cv = st.Canvas(W, 104.0)

    # ---- a: realism ------------------------------------------------------------------------------------------------
    cv.letter(0, 2.0, "a", "Realism: synthetic patients judged real")
    al, a = lme["A"]["primary"], s["A"]
    tw, re_ = a["accuracy_on_twins"], a["accuracy_on_real"]
    rows = (("Synthetic called real", 1 - tw["est"], 1 - tw["hi"], 1 - tw["lo"], st.DIDYMOS, "o"),
            ("Real called real", re_["est"], re_["lo"], re_["hi"], st.SLATE[700], "o"),
            ("Accuracy, real or synthetic", al["accuracy"], al["lo90"], al["hi90"], st.TEXT, "D"))
    top, rh = 10.0, 6.0
    ticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ax = _forest(cv, 40, top, 44, len(rows), rh, (0.2, 0.8), ticks, [st.num(t, 1) for t in ticks], "Share of calls")
    st.refline(ax, 0.5)
    for i, (lab, est, lo, hi, col, mk) in enumerate(rows):
        cv.text(38.0, top + (i + 0.5) * rh, lab, ha="right", va="center")
        ax.plot([lo, hi], [i, i], color=col, lw=1.0, solid_capstyle="butt")
        ax.plot(est, i, ls="none", marker=mk, ms=3.4, color=col)
        sd.add("a", lab, estimate=est, ci_low=lo, ci_high=hi, level=0.9 if mk == "D" else 0.95)
    sd.add("a", "raters and calls", raters=al["raters"], items=al["items"], calls=al["calls"])

    # ---- b: simulated responses ------------------------------------------------------------------------------------
    cv.letter(96, 2.0, "b", "Plausibility of simulated responses")
    bt, bl = s["B1"]["by_treatment"], lme["B1"]["primary"]["twin_minus_real"]
    names = [t for t in TREAT if t in bt and t != "insulin"]
    brows = [(TREAT[t], bt[t]["twin_minus_real"]["est"], bt[t]["twin_minus_real"]["lo"],
              bt[t]["twin_minus_real"]["hi"], False) for t in names]
    brows.append(("All treatments", bl["est"], bl["lo95"], bl["hi95"], True))
    btop, brh = 10.0, 5.0
    bx = _forest(cv, 130, btop, 46, len(brows), brh, (-2.2, 1.6), [-2, -1, 0, 1],
                 [st.num(-2, 0), st.num(-1, 0), "0", "1"], "Synthetic minus real (1–5 scale)")
    st.refline(bx, 0.0)
    for i, (lab, est, lo, hi, overall) in enumerate(brows):
        cv.text(128.0, btop + (i + 0.5) * brh, lab, ha="right", va="center",
                fontweight="bold" if overall else "normal")
        bx.plot([max(lo, -2.2), min(hi, 1.6)], [i, i], color=st.DIDYMOS, lw=1.5 if overall else 1.0,
                solid_capstyle="butt")
        bx.plot(est, i, ls="none", marker="D" if overall else "o", ms=3.4 if overall else 3.0, color=st.DIDYMOS)
        sd.add("b", lab, estimate=est, ci_low=lo, ci_high=hi)

    # ---- c: explanations -------------------------------------------------------------------------------------------
    y2 = 56.0
    cv.letter(0, y2, "c", "Relevance of explanatory factors")
    mr = s["B2"]["primary"]["mean_relevance"]
    crows = (("Counterfactual editing", mr["edit"], st.DIDYMOS), ("Attention knockout", mr["knockout"], st.DIDYMOS),
             ("Random factors", mr["random"], st.SLATE[700]))
    ctop, crh = y2 + 9.0, 6.0
    cx = _forest(cv, 40, ctop, 44, len(crows), crh, (1.0, 5.0), [1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"],
                 "Mean relevance (1–5 scale)")
    for i, (lab, x, col) in enumerate(crows):
        cv.text(38.0, ctop + (i + 0.5) * crh, lab, ha="right", va="center")
        cx.plot([x["lo"], x["hi"]], [i, i], color=col, lw=1.0, solid_capstyle="butt")
        cx.plot(x["est"], i, ls="none", marker="o", ms=3.2, color=col)
        sd.add("c", lab, mean_relevance=x["est"], ci_low=x["lo"], ci_high=x["hi"])

    # ---- d: decision support ---------------------------------------------------------------------------------------
    cv.letter(96, y2, "d", "Decision support: Brier score")
    src = (("unaided", "Without the card", st.SLATE[700], "o"), ("aided", "With the card", st.BLUE[500], "o"),
           ("twin_card", "The card alone", st.BLUE[700], "s"), ("lgbm", "LightGBM", st.LGBM, "s"),
           ("sapsii", "SAPS II", st.CLINICAL, "D"))
    dtop, drh = y2 + 9.0, 4.4
    for j, (outcome, title) in enumerate((("icu_death", "ICU death"), ("aki", "AKI"))):
        x0 = 130.0 + j * 26.0
        br = s["C"][outcome]["brier"]
        dx = _forest(cv, x0, dtop, 22, len(src), drh, (0.0, 0.32), [0, 0.1, 0.2, 0.3], ["0", "0.1", "0.2", "0.3"],
                     "Brier")
        cv.text(x0, dtop - 1.2, title, ha="left", va="bottom")
        for i, (key, lab, col, mk) in enumerate(src):
            if j == 0:
                cv.text(128.0, dtop + (i + 0.5) * drh, lab, ha="right", va="center")
            x = br.get(key)
            if not x:
                continue
            dx.plot([x["lo"], x["hi"]], [i, i], color=col, lw=1.0, solid_capstyle="butt")
            dx.plot(x["est"], i, ls="none", marker=mk, ms=3.0, color=col)
            sd.add("d", lab, outcome=outcome, brier=x["est"], ci_low=x["lo"], ci_high=x["hi"])
        dd = lme["C"]["primary"][outcome]["aided_minus_unaided"]
        yd = dtop + len(src) * drh + 13.0
        cv.text(x0, yd, f"{st.num(dd['est'], 3, sign=True)}, {st.p_text(dd['p'])}", ha="left", va="center")
        if j == 0:
            cv.text(128.0, yd, "With minus without the card", ha="right", va="center")
        sd.add("d", "with minus without the card", outcome=outcome, estimate=dd["est"], ci_low=dd["lo95"],
               ci_high=dd["hi95"], p=dd["p"])
    return cv, sd
