"""Paper P's supplementary figures for amendments 16 and 17 (the transfer to two databases never seen in training, the
48-hour landmark with the follow-up's settings, attention knockout against the generator, and adaptation to the new
databases), in the main figures' design system (``tools/paper_figstyle.py``). Built by ``tools/paper_figures.py``;
their numbers follow the order in which the manuscript first cites them (``si_sections/figures.tex``):

* ``supp_l48x`` (with Fig. 2): the 48-hour landmark with the follow-up's settings (16b; families 18 and 19);
* ``supp_knockout`` (with Fig. 6): attention knockout and editing against the generator, redrawn (16c; family 20);
* ``supp_transfer`` (with Fig. 9): zero-shot transfer to RICD and zigongICU on every 16a anchor (families 15 and 16,
  calibration, ranks);
* ``supp_adapt`` (with Fig. 9): the improved encoding, the training grid, forgetting and the membership check (17);
* ``supp_adapt_cal`` (with Fig. 9): calibration before and after recalibration, and label efficiency per pair (17);
* ``supp_adapt_gate`` (with Fig. 9): the adapted run's membership test on the seven databases at both sizes (17c).
"""

from __future__ import annotations

import json

import numpy as np

import paper_figstyle as st
from paper_figdata import Inputs, find
from paper_figures_supp import _handle, _legend, _row_y, _rows_axes

W = st.FULL
EXT = ("ricd", "zigong")
EXT_NAME = {"ricd": "RICD", "zigong": "zigongICU"}
DESIGN_NAME = {"R": "Rolling anchors", "L24e": "24-hour landmark"}
CLIN = ("sofa", "news", "mews", "qsofa", "sirs", "sapsii_prob", "oasis_prob", "ricd_recorded_sofa")
CLIN_NAME = {**st.SCORE, "ricd_recorded_sofa": "RICD's recorded SOFA", "sapsii_prob": "SAPS II", "oasis_prob": "OASIS"}
ORANGE_DARK = "#A86F00"
LOG_TICKS = (0.1, 0.2, 0.5, 1, 2, 5)


def _log_axis(ax, lo: float = 0.08, hi: float = 6.0) -> None:
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xticks(LOG_TICKS)
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5"])
    ax.minorticks_off()


def _ext_rows(designs: dict) -> list[tuple[str, str, str, dict]]:
    order = list(st.ENDPOINT_SHORT)
    out = []
    for design in ("R", "L24e"):
        for d in EXT:
            progs = designs.get(design, {}).get(d, {})
            for p in sorted(progs, key=order.index):
                out.append((design, d, p, progs[p]))
    return out


def _grouped_positions(rows) -> tuple[list[float], list[tuple[str, float]], float]:
    """Row positions in row units with a header row before each design: (positions, headers, total units)."""
    pos, heads, y, cur = [], [], 0.0, None
    for design, *_ in rows:
        if design != cur:
            if cur is not None:
                y += 0.3
            heads.append((DESIGN_NAME[design], y))
            y += 1.15
            cur = design
        pos.append(y)
        y += 1.0
    return pos, heads, y


def _row_axes(cv, left: float, top: float, width: float, units: float, rh: float):
    ax = cv.axes(left, top, width, units * rh)
    ax.set_ylim(units - 0.5, -0.5)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    return ax


# ---- 16b: the 48-hour landmark --------------------------------------------------------------------------------------

def l48x(inp: Inputs):
    """The 48-hour landmark (A1's test_L48 anchors, a case-cohort sample) with 128 futures, E2 and longer caps:
    Didymos's current estimate against every comparator, and its differences from LightGBM (family 18) and from the
    pre-specified 32-future estimate (family 19)."""
    sd = st.SourceData("supp_l48x")
    c = inp.json("a16/comparators_L48x.json", sd=sd)["subsets"]["L48x"]
    sd.source(inp.stats_path)
    f18, f19 = inp.family(18), inp.family(19)
    progs = [r["program"] for r in f18]
    learned = ("lgbm_per_dataset", "logistic_pooled", "logistic_l1", "logistic_enet", "cart", "random_forest",
               "extra_trees", "naive_bayes", "knn", "mlp", "tabpfn_v2", "twin_probe")
    rh, top = 6.0, 12.0
    cv = st.Canvas(W, 57.0)
    cv.letter(0, 2.0, "a", "Discrimination at the 48-hour landmark")
    cv.letter(120.0, 2.0, "b", "Didymos minus comparator")
    cv.text(40.0, top - 1.5, "Events/n", ha="right", va="bottom", color=st.TEXT2)
    cv.text(178.0, top - 1.5, r"$\mathit{P}$ (Holm)", ha="right", va="bottom", color=st.TEXT2)
    ax = _rows_axes(cv, 44.0, top, 70.0, len(progs), rh)
    bx = _rows_axes(cv, 124.0, top, 34.0, len(progs), rh)
    st.refline(bx, 0.0)
    for i, p in enumerate(progs):
        x = c[p]
        m = x["models"]
        r18, r19 = find(f18, program=p), find(f19, program=p)
        e2 = r18["comparison"].startswith("e2")
        cur = x["e2_auroc"] if e2 else m["twin_128"]["auroc"]
        pre = cur - r19["estimate"]
        y = _row_y(top, rh, i)
        cv.text(2.0, y, st.ENDPOINT_SHORT[p], ha="left", va="center")
        cv.text(40.0, y, f"{x['positives']}/{x['anchors']}", ha="right", va="center", color=st.TEXT2)
        for k in learned:
            if k in m:
                ax.plot(m[k]["auroc"], i, ls="none", marker="|", ms=5, mew=1.0, color=st.LEARNER, zorder=1)
        clin = {k: m[k]["auroc"] for k in CLIN if k in m}
        best = max(clin, key=clin.get)
        ax.plot([pre, cur], [i, i], color=st.BLUE[200], lw=1.5, solid_capstyle="butt", zorder=2)
        ax.plot(pre, i, ls="none", marker="o", ms=3.0, mfc="white", mec=st.DIDYMOS, mew=1.0, zorder=3)
        ax.plot(clin[best], i, ls="none", marker="D", ms=3.0, color=st.CLINICAL, zorder=3)
        ax.plot(m["lgbm_pooled"]["auroc"], i - 0.2, ls="none", marker="s", ms=3.0, color=st.LGBM, zorder=4)
        ax.plot(m["tabpfn_v3_5"]["auroc"], i + 0.2, ls="none", marker="^", ms=3.2, color=st.TABPFN, zorder=4)
        ax.plot(cur, i, ls="none", marker="o", ms=3.6, color=st.DIDYMOS, mfc=st.DIDYMOS if e2 else "white",
                mec=st.DIDYMOS, mew=1.5 if not e2 else 1.0, zorder=5)
        for row, off, mk in ((r18, -0.15, "s"), (r19, 0.15, "o")):
            col = st.LGBM if row is r18 else st.DIDYMOS
            bx.plot(row["ci"], [i + off] * 2, color=col, lw=1.0, solid_capstyle="butt", zorder=2)
            bx.plot(row["estimate"], i + off, ls="none", marker=mk, ms=3.0, color=col,
                    mfc="white" if row is r19 else col, mec=col, mew=1.0, zorder=3)
        cv.text(178.0, y - 1.2, st.p_value(r18["p_holm"], r18.get("p_holm_bound")), ha="right", va="center")
        cv.text(178.0, y + 1.2, st.p_value(r19["p_holm"], r19.get("p_holm_bound")), ha="right", va="center")
        sd.add("a", "Didymos", program=p, estimate="hazard" if e2 else "sampled", auroc=cur, pre_specified=pre,
               lightgbm=m["lgbm_pooled"]["auroc"], tabpfn_3_5=m["tabpfn_v3_5"]["auroc"], best_clinical=best,
               best_clinical_auroc=clin[best], events=x["positives"], anchors=x["anchors"])
        for row in (r18, r19):
            sd.add("b", row["comparison"], program=p, difference=row["estimate"], ci_low=row["ci"][0],
                   ci_high=row["ci"][1], p_holm=row["p_holm"])
    ax.set_xlim(0.6, 1.0)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"])
    ax.set_xlabel("AUROC")
    bx.set_xlim(-0.05, 0.15)
    bx.set_xticks([-0.05, 0, 0.05, 0.1, 0.15])
    bx.set_xticklabels([st.num(-0.05, 2), "0", "0.05", "0.10", "0.15"])
    bx.set_xlabel("Difference in AUROC (95% CI)")
    hs = [_handle(marker="o", ms=3.6, color=st.DIDYMOS), _handle(marker="o", ms=3.6, mfc="white", mec=st.DIDYMOS,
                                                                 mew=1.5),
          _handle(marker="o", ms=3.0, mfc="white", mec=st.DIDYMOS, mew=1.0), _handle(marker="s", ms=3.0, color=st.LGBM),
          _handle(marker="^", ms=3.2, color=st.TABPFN), _handle(marker="D", ms=3.0, color=st.CLINICAL),
          _handle(marker="|", ms=5, mew=1.0, color=st.LEARNER)]
    _legend(cv, hs, ["Hazard estimate", "Sampled estimate", "Pre-specified, 32 futures", "LightGBM", "TabPFN-3.5",
                     "Best clinical score", "Other learned models"], 2.0, 46.0, ncol=4)
    _legend(cv, [_handle(color=st.LGBM, lw=1.0, ls="-", marker="s", ms=3.0),
                 _handle(color=st.DIDYMOS, lw=1.0, ls="-", marker="o", ms=3.0, mfc="white", mec=st.DIDYMOS, mew=1.0)],
            ["Minus LightGBM", "Minus the pre-specified estimate"], 124.0, 46.0, ncol=1)
    return cv, sd


# ---- 16c: knockout against the generator ----------------------------------------------------------------------------

METHODS = (("edit", "Counterfactual editing", st.DIDYMOS, "o", st.DIDYMOS),
           ("knockout", "Attention knockout", st.GREEN, "D", st.GREEN),
           ("ig", "Integrated gradients", st.SLATE[500], "s", st.SLATE[500]),
           ("attn_rollout", "Attention rollout", st.SLATE[500], "^", "white"))


def knockout(inp: Inputs):
    """Agreement of each attribution method with the generator's own effects of each edit, on F4's prompts drawn
    again (48 futures each): the median Spearman per endpoint within its split-half ceiling, and per anchor
    knockout minus editing (family 20)."""
    import pandas as pd

    sd = st.SourceData("supp_knockout")
    c = inp.json("f4r/c2b.json", sd=sd)["programs"]
    kv_path = inp.path("f4r/knockout_vs_edit.parquet")
    sd.source(kv_path, inp.stats_path)
    kv = pd.read_parquet(kv_path)
    f20 = inp.family(20)
    progs = [p for p in st.ENDPOINT_SHORT if p in c]
    cv = st.Canvas(W, 61.0)
    rh, top = 7.0, 10.0
    cv.letter(0, 2.0, "a", "Agreement with the generator's effects")
    ax = _rows_axes(cv, 34.0, top, 54.0, len(progs), rh)
    for i, p in enumerate(progs):
        x = c[p]
        y = _row_y(top, rh, i)
        cv.text(32.0, y, st.ENDPOINT_SHORT[p], ha="right", va="center")
        ceil = x["reliability_sb_median"]
        ax.barh(i, ceil, height=0.62, color=st.SLATE[100], edgecolor=st.SLATE[300], lw=1.0, zorder=0)
        for k, (m, _lab, col, mk, face) in enumerate(METHODS):
            a = x["agreement"].get(m)
            if not a:
                continue
            ax.plot(a["spearman_median"], i + (k - 1.5) * 0.15, ls="none", marker=mk, ms=3.2, mfc=face, mec=col,
                    mew=1.0, zorder=3)
            sd.add("a", m, program=p, spearman_median=a["spearman_median"], anchors=a["anchors"],
                   sign_agreement=a.get("sign_agreement"), ceiling=ceil)
    st.refline(ax, 0.0)
    ax.set_xlim(-0.1, 0.65)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6"])
    ax.set_xlabel("Median Spearman with the generator's effects")
    from matplotlib.patches import Patch

    hs = [_handle(marker=mk, ms=3.2, mfc=face, mec=col, mew=1.0) for _m, _l, col, mk, face in METHODS] + \
        [Patch(facecolor=st.SLATE[100], edgecolor=st.SLATE[300], lw=1.0)]
    _legend(cv, hs, [lab for _m, lab, *_ in METHODS] + ["Split-half ceiling"], 2.0, 49.0, ncol=3)
    # b: per anchor, knockout minus editing
    cv.letter(100.0, 2.0, "b", "Per anchor, knockout minus editing")
    rows = progs + ["pooled"]
    bx = _rows_axes(cv, 124.0, top, 40.0, len(rows), rh * len(progs) / len(rows))
    st.refline(bx, 0.0)
    rr = rh * len(progs) / len(rows)
    cv.text(178.0, top - 1.5, r"$\mathit{P}$ (Holm)", ha="right", va="bottom", color=st.TEXT2)
    rng = np.random.default_rng(0)
    for i, p in enumerate(rows):
        g = kv if p == "pooled" else kv[kv["program"] == p]
        g = g[np.isfinite(g["rho_knockout"]) & np.isfinite(g["rho_edit"])]
        dif = (g["rho_knockout"] - g["rho_edit"]).to_numpy()
        y = top + (i + 0.5) * rr
        cv.text(122.0, y, "Pooled" if p == "pooled" else st.ENDPOINT_SHORT[p], ha="right", va="center",
                fontweight="bold" if p == "pooled" else "normal")
        bx.plot(dif, i + rng.uniform(-0.22, 0.22, len(dif)), ls="none", marker="o", ms=1.6, mfc="white",
                mec=st.SLATE[300], mew=1.0, zorder=1)
        q1, med, q3 = np.percentile(dif, [25, 50, 75])
        bx.plot([q1, q3], [i, i], color=st.GREEN, lw=2.0, solid_capstyle="butt", zorder=2)
        bx.plot(med, i, ls="none", marker="D", ms=3.4, color=st.GREEN, zorder=3)
        row = find(f20, program=p)
        pv = row.get("p") if p == "pooled" else row.get("p_holm")
        cv.text(178.0, y, st.p_value(pv), ha="right", va="center")
        sd.add("b", "knockout minus editing", program=p, anchors=len(dif), median=med, q1=q1, q3=q3,
               wins=row["wins"], ties=row["ties"], losses=row["losses"], p=pv)
    bx.set_xlim(-1.0, 1.0)
    bx.set_xticks([-1, -0.5, 0, 0.5, 1])
    bx.set_xticklabels([st.num(-1.0, 1), st.num(-0.5, 1), "0", "0.5", "1.0"])
    bx.set_xlabel("Difference in Spearman (median, interquartile range)")
    return cv, sd


# ---- 16a: zero-shot transfer on every anchor ------------------------------------------------------------------------

def transfer(inp: Inputs):
    """Zero-shot transfer to RICD and zigongICU on every 16a anchor: Didymos's AUROC against every comparator; its
    differences from transferred LightGBM (family 15) and the local cross-fitted LightGBM (family 16); O/E before and
    after a within-database recalibration; mean ranks (post hoc)."""
    sd = st.SourceData("supp_transfer")
    e = inp.json("ext/ext.json", sd=sd)
    rk = inp.json("ext/ranks.json", sd=sd)
    sd.source(inp.stats_path)
    f15, f16 = inp.family(15), inp.family(16)
    rows = _ext_rows(e["designs"])
    pos, heads, units = _grouped_positions(rows)
    rh, top = 4.0, 12.0
    H = top + units * rh + 92.0
    cv = st.Canvas(W, H)
    cv.letter(0, 2.0, "a", "Discrimination on every transfer anchor")
    cv.letter(118.0, 2.0, "b", "Didymos minus LightGBM")
    cv.text(58.0, top - 1.5, "Events/n", ha="right", va="bottom", color=st.TEXT2)
    ax = _row_axes(cv, 62.0, top, 52.0, units, rh)
    bx = _row_axes(cv, 122.0, top, 56.0, units, rh)
    st.refline(bx, 0.0)
    for name, y0 in heads:
        cv.text(0.0, top + (y0 + 0.4) * rh, name, ha="left", va="center", fontweight="bold")
    for (design, d, p, x), yy in zip(rows, pos, strict=True):
        pr = x["predictors"]
        t = pr["twin"]
        ty = top + (yy + 0.5) * rh
        cv.text(2.5, ty, st.ENDPOINT_SHORT[p], ha="left", va="center")
        cv.text(33.0, ty, EXT_NAME[d], ha="left", va="center", color=st.TEXT2)
        cv.text(58.0, ty, f"{t['positives']}/{t['n']}", ha="right", va="center", color=st.TEXT2)
        for k in ("logistic_pooled", "twin_probe", *[c for c in CLIN if c in pr]):
            if k in pr:
                ax.plot(pr[k]["auroc"], yy, ls="none", marker="|", ms=4.5, mew=1.0, color=st.LEARNER, zorder=1)
        lo, hi = t["auroc_ci"]
        ax.plot([lo, hi], [yy, yy], color=st.BLUE[200], lw=1.5, solid_capstyle="butt", zorder=2)
        ax.plot(pr["lgbm_local"]["auroc"], yy + 0.22, ls="none", marker="s", ms=2.8, color=ORANGE_DARK, zorder=3)
        ax.plot(pr["lgbm_pooled"]["auroc"], yy - 0.22, ls="none", marker="s", ms=2.8, mfc="white", mec=st.LGBM,
                mew=1.0, zorder=3)
        ax.plot(pr["tabpfn_v3_5"]["auroc"], yy, ls="none", marker="^", ms=3.0, color=st.TABPFN, zorder=4)
        ax.plot(t["auroc"], yy, ls="none", marker="o", ms=3.4, color=st.DIDYMOS, zorder=5)
        sd.add("a", "Didymos", design=design, dataset=d, program=p, auroc=t["auroc"], ci_low=lo, ci_high=hi,
               lightgbm_transferred=pr["lgbm_pooled"]["auroc"], lightgbm_local=pr["lgbm_local"]["auroc"],
               tabpfn_3_5=pr["tabpfn_v3_5"]["auroc"], events=t["positives"], n=t["n"])
        for fam, off, col, face in ((f15, -0.18, st.LGBM, "white"), (f16, 0.18, ORANGE_DARK, ORANGE_DARK)):
            r = find(fam, design=design, dataset=d, program=p)
            if not r:
                continue
            bx.plot(r["ci"], [yy + off] * 2, color=col, lw=1.0, solid_capstyle="butt", zorder=2)
            bx.plot(r["estimate"], yy + off, ls="none", marker="s", ms=2.8, mfc=face, mec=col, mew=1.0, zorder=3)
            sd.add("b", r["comparison"], design=design, dataset=d, program=p, difference=r["estimate"],
                   ci_low=r["ci"][0], ci_high=r["ci"][1], p_holm=r["p_holm"])
    ax.set_xlim(0.45, 1.0)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    ax.set_xlabel("AUROC")
    bx.set_xlim(-0.4, 0.25)
    bx.set_xticks([-0.4, -0.2, 0, 0.2])
    bx.set_xticklabels([st.num(-0.4, 1), st.num(-0.2, 1), "0", "0.2"])
    bx.set_xlabel("Difference in AUROC (95% CI)")
    ybot = top + units * rh
    hs = [_handle(marker="o", ms=3.4, color=st.DIDYMOS),
          _handle(marker="s", ms=2.8, mfc="white", mec=st.LGBM, mew=1.0),
          _handle(marker="s", ms=2.8, color=ORANGE_DARK), _handle(marker="^", ms=3.0, color=st.TABPFN),
          _handle(marker="|", ms=4.5, mew=1.0, color=st.LEARNER)]
    _legend(cv, hs, ["Didymos (95% CI)", "LightGBM, transferred", "LightGBM, local (cross-fitted)", "TabPFN-3.5",
                     "Other models and scores"], 2.0, ybot + 10.0, ncol=5)
    # c: calibration before and after a within-database recalibration
    yc = ybot + 19.0
    cv.letter(0, yc, "c", "Observed over expected, before and after recalibration")
    rows_c = [r for r in rows]
    n = len(rows_c)
    cw = 52.0
    cx = cv.axes(64.0, yc + 6.0, cw, 54.0)
    cx.set_ylim(n - 0.5, -0.5)
    cx.yaxis.set_visible(False)
    cx.spines["left"].set_visible(False)
    st.refline(cx, 1.0)
    rr = 54.0 / n
    for i, (design, d, p, x) in enumerate(rows_c):
        raw = x["predictors"]["twin"]["observed_over_expected"]
        rec = x["recalibrated"]["twin"]["observed_over_expected"]
        cv.text(62.0, yc + 6.0 + (i + 0.5) * rr, f"{st.ENDPOINT_SHORT[p]}, {EXT_NAME[d]}"
                + (" (L)" if design == "L24e" else ""), ha="right", va="center")
        cx.plot([raw, rec], [i, i], color=st.BLUE[200], lw=1.2, zorder=1)
        cx.plot(raw, i, ls="none", marker="o", ms=3.0, mfc="white", mec=st.DIDYMOS, mew=1.0, zorder=2)
        cx.plot(rec, i, ls="none", marker="o", ms=3.0, color=st.DIDYMOS, zorder=3)
        sd.add("c", "O/E", design=design, dataset=d, program=p, raw=raw, recalibrated=rec)
    _log_axis(cx, 0.03, 6.0)
    cx.set_xticks((0.05, 0.1, 0.2, 0.5, 1, 2, 5))
    cx.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
    cx.set_xlabel("O/E (log scale)")
    _legend(cv, [_handle(marker="o", ms=3.0, mfc="white", mec=st.DIDYMOS, mew=1.0),
                 _handle(marker="o", ms=3.0, color=st.DIDYMOS)], ["As transferred", "Recalibrated in the database"],
            2.0, yc + 64.5, ncol=2)
    # d: mean ranks among the predictors that never saw the database
    cv.letter(122.0, yc, "d", "Mean rank, transferred models")
    s = rk["transfer_only"]
    preds = s["predictors"]
    rr2 = 54.0 / len(preds)
    dx = cv.axes(150.0, yc + 6.0, 28.0, 54.0)
    dx.set_ylim(len(preds) - 0.5, -0.5)
    dx.yaxis.set_visible(False)
    dx.spines["left"].set_visible(False)
    name = {**st.PREDICTOR, "twin": "Didymos", "lgbm_pooled": "LightGBM", "logistic_pooled": "Logistic (L2)"}
    for i, x in enumerate(preds):
        k = x["predictor"]
        col = {"twin": st.DIDYMOS, "tabpfn_v3_5": st.TABPFN, "lgbm_pooled": st.LGBM, "twin_probe": st.PROBE}.get(
            k, st.CLINICAL if k in CLIN else st.SLATE[500])
        cv.text(148.5, yc + 6.0 + (i + 0.5) * rr2, name.get(k, k), ha="right", va="center",
                fontweight="bold" if k == "twin" else "normal")
        lo, hi = x["mean_rank_ci"]
        dx.plot([lo, hi], [i, i], color=col, lw=1.5, solid_capstyle="butt")
        dx.plot(x["mean_rank"], i, ls="none", marker="o", ms=3.0, color=col)
        sd.add("d", "mean rank", predictor=k, mean_rank=x["mean_rank"], ci_low=lo, ci_high=hi, firsts=x["firsts"])
    dx.set_xlim(0.5, len(preds) + 0.5)
    dx.set_xticks([1, 5, len(preds)])
    dx.set_xlabel("Rank (1 = highest AUROC)")
    return cv, sd


# ---- 17: the improved encoding, the grid, membership ----------------------------------------------------------------

DOMAIN = {"MED_CON": "Continuous infusions", "MED_INT": "Intermittent doses", "ADT": "Locations", "OUTPUT": "Outputs"}


def adaptation(inp: Inputs):
    """Amendment 17's adaptation: the share of events without a token before and after the mapping to the nearest
    kept code; the mapped encoding's AUROC against the original's (family 21, zero-shot); the training grid
    (held-out FT loss against the change on the seven databases); and the membership signal of the FT stays against
    the TEST stays."""
    sd = st.SourceData("supp_adapt")
    r = inp.json("ext17/ext17.json", sd=sd)
    gr = inp.json("ext17/grid.json", sd=sd)
    sd.source(inp.stats_path)
    ro = r["readouts"]
    oov = ro["oov"]
    fam21 = [x for x in inp.family(21) if x.get("arm") == "Z1"]
    cv = st.Canvas(W, 136.0)
    # a: OOV
    cv.letter(0, 2.0, "a", "Events without a vocabulary token")
    bars = [(d, dom) for d in EXT for dom in DOMAIN if oov["ext_r4"][d].get(dom, {}).get("share", 0) > 0.001]
    rh, top = 5.0, 10.0
    ax = _rows_axes(cv, 46.0, top, 40.0, len(bars), rh)
    for i, (d, dom) in enumerate(bars):
        a_, b_ = oov["ext_r4"][d][dom]["share"] * 100, oov["ext_r4b"][d].get(dom, {}).get("share", 0) * 100
        y = _row_y(top, rh, i)
        cv.text(44.0, y, f"{DOMAIN[dom]}, {EXT_NAME[d]}", ha="right", va="center")
        ax.barh(i - 0.18, a_, height=0.34, color=st.SLATE[300], lw=0)
        ax.barh(i + 0.18, max(b_, 0.4), height=0.34, color=st.DIDYMOS, lw=0)
        ax.text(a_ + 1.5, i, f"{a_:.1f}% to {b_:.1f}%", va="center", ha="left", fontsize=st.PT, color=st.TEXT2)
        sd.add("a", "OOV share", dataset=d, domain=dom, original=a_, mapped=b_)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of the domain's events (%)")
    from matplotlib.patches import Patch

    _legend(cv, [Patch(facecolor=st.SLATE[300]), Patch(facecolor=st.DIDYMOS)], ["Original encoding", "Mapped codes"],
            2.0, top + len(bars) * rh + 9.0, ncol=2)
    # b: mapped codes minus original (zero-shot, TEST anchors)
    cv.letter(96.0, 2.0, "b", "Mapped codes minus original, zero-shot")
    pairs = [(x["design"], x["dataset"], x["program"], x) for x in fam21]
    order = list(st.ENDPOINT_SHORT)
    pairs.sort(key=lambda t: (t[0] != "R", t[1], order.index(t[2])))
    rb = 3.2
    bx = _rows_axes(cv, 140.0, top, 30.0, len(pairs), rb)
    st.refline(bx, 0.0)
    for i, (design, d, p, x) in enumerate(pairs):
        y = top + (i + 0.5) * rb
        cv.text(138.0, y, f"{st.ENDPOINT_SHORT[p]}, {EXT_NAME[d]}" + (" (L)" if design == "L24e" else ""),
                ha="right", va="center")
        bx.plot(x["ci"], [i, i], color=st.BLUE[300], lw=1.0, solid_capstyle="butt")
        bx.plot(x["estimate"], i, ls="none", marker="o", ms=2.8, mfc="white", mec=st.BLUE[400], mew=1.0)
        sd.add("b", "mapped minus original", design=design, dataset=d, program=p, difference=x["estimate"],
               ci_low=x["ci"][0], ci_high=x["ci"][1], p_holm=x["p_holm"])
    bx.set_xlim(-0.15, 0.1)
    bx.set_xticks([-0.1, 0, 0.1])
    bx.set_xticklabels([st.num(-0.1, 1), "0", "0.1"])
    bx.set_xlabel("Difference in AUROC (95% CI)")
    # c: the grid
    yc = 66.0
    cv.letter(0, yc, "c", "The training grid")
    cx = cv.axes(16.0, yc + 7.0, 64.0, 48.0)
    seven = [k for k in gr["before"]["by_dataset"] if k not in EXT]
    for k, x in gr["cells"].items():
        fg = np.mean([x["forgetting"][d] for d in seven])
        chosen = x["run"] == gr["chosen"]
        mk = "o" if x["ft_passes"] == 1 else "s"
        col = st.BLUE[300] if x["lr"] < 5e-5 else st.DIDYMOS
        cx.plot(fg, x["ft_nll"], ls="none", marker=mk, ms=4.2, color=col, mec=st.BLUE[900] if chosen else col,
                mew=1.5 if chosen else 1.0, zorder=3)
        mant, ex = f"{x['lr']:.0e}".split("e")
        lab = rf"${mant}\times10^{{{int(ex)}}}$, {x['ft_passes']:g} pass" + ("es" if x["ft_passes"] > 1 else "")
        left = x["lr"] < 5e-5 and x["ft_passes"] > 1             # two cells sit side by side: one label on the left
        cx.annotate(lab + (" (chosen)" if chosen else ""), (fg, x["ft_nll"]), xytext=(-5, 0) if left else (5, 0),
                    textcoords="offset points", va="center", ha="right" if left else "left", fontsize=st.PT,
                    color=st.TEXT)
        sd.add("c", "grid cell", cell=k, lr=x["lr"], passes=x["ft_passes"], steps=x["steps"], ft_loss=x["ft_nll"],
               seven_change_mean=fg, chosen=chosen)
    cx.axhline(gr["before"]["ft_nll"], color=st.REF, lw=1.0, ls=(0, (3, 2)), zorder=0)
    cx.text(0.0005, gr["before"]["ft_nll"] - 0.03, "Before adaptation", va="top", ha="left", fontsize=st.PT,
            color=st.TEXT2)
    cx.set_xlim(0, 0.032)
    cx.set_xticks([0, 0.01, 0.02, 0.03])
    cx.set_xticklabels(["0", "0.01", "0.02", "0.03"])
    cx.set_ylim(1.0, 2.2)
    cx.set_xlabel("Change of the seven databases' loss (mean, nats per token)")
    cx.set_ylabel("Held-out FT loss (nats per token)")
    # d: membership of the FT stays
    cv.letter(96.0, yc, "d", "Membership signal of the FT stays")
    mem = ro["membership"]
    who = ((gr["chosen"], "Didymos, adapted", st.DIDYMOS, "white"), ("MUC32_main", "Didymos, zero-shot", st.SLATE[500],
                                                                      "white"))
    groups = ("ricd", "zigong", "pooled")
    dx = _rows_axes(cv, 130.0, yc + 7.0, 46.0, len(groups), 9.0)
    st.refline(dx, 0.5)
    for i, g in enumerate(groups):
        cv.text(128.0, yc + 7.0 + (i + 0.5) * 9.0, EXT_NAME.get(g, "Pooled"), ha="right", va="center",
                fontweight="bold" if g == "pooled" else "normal")
        for k, (run, lab, col, _f) in enumerate(who):
            x = mem[run]["groups"][g]
            y = i + (k - 0.5) * 0.36
            dx.plot(x["auc"][1:], [y, y], color=col, lw=1.0, solid_capstyle="butt")
            dx.plot(x["auc"][0], y, ls="none", marker="o", ms=3.2, color=col)
            sd.add("d", lab, group=g, auc=x["auc"][0], ci_low=x["auc"][1], ci_high=x["auc"][2],
                   members=x["members"], nonmembers=x["nonmembers"])
    dx.set_xlim(0.44, 0.62)
    dx.set_xticks([0.45, 0.5, 0.55, 0.6])
    dx.set_xticklabels(["0.45", "0.50", "0.55", "0.60"])
    dx.set_xlabel("AUC, FT stays against TEST stays (95% CI)")
    _legend(cv, [_handle(marker="o", ms=3.2, color=c_) for _r, _l, c_, _f in who], [lab for _r, lab, *_ in who],
            100.0, yc + 41.0, ncol=2)
    _legend(cv, [_handle(marker="o", ms=4.2, color=st.BLUE[300]), _handle(marker="o", ms=4.2, color=st.DIDYMOS),
                 _handle(marker="o", ms=4.2, color=st.SLATE[300]), _handle(marker="s", ms=4.2, color=st.SLATE[300])],
            [r"Learning rate $3\times10^{-5}$", r"$1\times10^{-4}$", "1 pass over FT", "3 passes"], 2.0, yc + 64.0,
            ncol=4)
    return cv, sd


# ---- 17: calibration and label efficiency per pair ------------------------------------------------------------------

def adapt_calibration(inp: Inputs):
    """O/E on the TEST anchors: zero-shot, adapted, adapted and recalibrated on the LABEL anchors; and per pair the
    local LightGBM's AUROC (median of 20 draws) against the number of LABEL stays, beside adapted Didymos's."""
    sd = st.SourceData("supp_adapt_cal")
    r = inp.json("ext17/ext17.json", sd=sd)
    rows = _ext_rows(r["designs"])
    pos, heads, units = _grouped_positions(rows)
    rh, top = 3.6, 10.0
    cv = st.Canvas(W, 196.0)
    cv.letter(0, 2.0, "a", "Observed over expected on the TEST anchors")
    ax = _row_axes(cv, 62.0, top, 70.0, units, rh)
    st.refline(ax, 1.0)
    for name, y0 in heads:
        cv.text(0.0, top + (y0 + 0.4) * rh, name, ha="left", va="center", fontweight="bold")
    styles = (("Z0", "Didymos, zero-shot", "o", "white", st.DIDYMOS), ("F1", "Didymos, adapted", "o", st.DIDYMOS,
                                                                       st.DIDYMOS),
              ("F1R", "Adapted, recalibrated on LABEL", "D", st.BLUE[700], st.BLUE[700]))
    for (design, d, p, x), yy in zip(rows, pos, strict=True):
        cal = x["calibration"]
        ty = top + (yy + 0.5) * rh
        desc = x["descriptive_only"]
        cv.text(2.5, ty, st.ENDPOINT_SHORT[p] + ("*" if desc else ""), ha="left", va="center",
                color=st.TEXT2 if desc else st.TEXT)
        cv.text(33.0, ty, EXT_NAME[d], ha="left", va="center", color=st.TEXT2)
        vals = [cal.get(k, {}).get("observed_over_expected") for k, *_ in styles]
        ok = [v for v in vals if v]
        if ok:
            ax.plot([min(ok), max(ok)], [yy, yy], color=st.BLUE[100], lw=1.5, zorder=1)
        for (k, lab, mk, face, col), v in zip(styles, vals, strict=True):
            if v:
                ax.plot(v, yy, ls="none", marker=mk, ms=3.0, mfc=face, mec=col, mew=1.0, zorder=3)
                sd.add("a", lab, design=design, dataset=d, program=p, observed_over_expected=v, descriptive=desc)
    _log_axis(ax, 0.01, 6.0)
    ax.set_xticks((0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5))
    ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
    ax.set_xlabel("O/E (log scale)")
    _legend(cv, [_handle(marker=mk, ms=3.0, mfc=face, mec=col, mew=1.0) for _k, _l, mk, face, col in styles],
            [lab for _k, lab, *_ in styles], 136.0, top + 2.0, ncol=1)
    # b: label efficiency, small multiples
    yb = top + units * rh + 14.0
    cv.letter(0, yb, "b", "A local LightGBM on n LABEL stays, against adapted Didymos (no labels)")
    cols, pw, ph, gx, gy = 5, 27.5, 21.0, 6.5, 11.0
    for i, (design, d, p, x) in enumerate(rows):
        c = r["curves"][design][d][p]
        x0 = 12.0 + (i % cols) * (pw + gx)
        y0 = yb + 8.0 + (i // cols) * (ph + gy)
        a_ = cv.axes(x0, y0, pw, ph)
        pts = sorted((int(k), v) for k, v in c["points"].items() if v["lgbm_auroc"] is not None)
        a_.axhline(c["f1_auroc"], color=st.DIDYMOS, lw=1.5, zorder=2)
        a_.plot([n for n, _ in pts], [v["lgbm_auroc"] for _, v in pts], color=st.LGBM, lw=1.5, marker="s", ms=2.4,
                zorder=3)
        a_.set_xscale("log")
        a_.set_xticks([25, 100, 400])
        a_.set_xticklabels(["25", "100", "400"])
        a_.minorticks_off()
        a_.set_xlim(18, 1100)
        a_.set_ylim(0.45, 1.0)
        a_.set_yticks([0.5, 0.75, 1.0])
        a_.set_yticklabels(["0.5", "0.75", "1.0"] if i % cols == 0 else [])
        desc = x["descriptive_only"]
        a_.set_title(f"{st.ENDPOINT_SHORT[p]}{'*' if desc else ''}\n{EXT_NAME[d]}"
                     + (", landmark" if design == "L24e" else ", rolling"), fontsize=st.PT, loc="left", pad=2.0,
                     color=st.TEXT2 if desc else st.TEXT)
        for n_, v in pts:
            sd.add("b", "label efficiency", design=design, dataset=d, program=p, n=n_, lightgbm_auroc=v["lgbm_auroc"],
                   adapted_auroc=c["f1_auroc"], status=c["status"], n_match=c["n_match"])
    nrow = (len(rows) - 1) // cols + 1
    ylast = yb + 8.0 + (nrow - 1) * (ph + gy) + ph
    cv.text(12.0 + (cols * (pw + gx) - gx) / 2, ylast + 6.5, "LABEL stays used for training (log scale)",
            ha="center", va="top")
    cv.text(3.5, yb + 8.0 + (nrow * (ph + gy) - gy) / 2, "AUROC", ha="center", va="center", rotation=90)
    _legend(cv, [_handle(color=st.DIDYMOS, lw=1.5, ls="-"), _handle(color=st.LGBM, lw=1.5, ls="-", marker="s", ms=2.4)],
            ["Didymos, adapted (no labels)", "Local LightGBM, median of 20 draws"], 12.0, ylast + 11.5, ncol=2)
    return cv, sd


# ---- 17c: the adapted run's membership test on the seven databases --------------------------------------------------

def adapt_gate(inp: Inputs):
    """The adapted run's membership test by loss on the seven training databases (P1): TPR at 1% FPR and AUC at the
    registered size (3,000 per side) and at amendment 17c's larger size, beside the base model's."""
    sd = st.SourceData("supp_adapt_gate")
    gr = inp.json("ext17/grid.json", sd=sd)
    audit = inp.root.parent / "audit"
    paths = {"base": audit / "m7_MUC32_main" / "p1.json",
             "reg": audit / f"m7_{gr['chosen']}" / "p1_cap3000.json",
             "large": audit / f"m7_{gr['chosen']}" / "p1.json"}
    for p_ in paths.values():
        sd.source(p_)
    p1 = {k: json.loads(v.read_text(encoding="utf-8"))["groups"] for k, v in paths.items()}
    groups = list(st.DATASETS) + ["pooled"]
    series = (("base", "Didymos before adaptation, 3,000 per side", st.SLATE[500], "o", st.SLATE[500]),
              ("reg", "Adapted, 3,000 per side", st.DIDYMOS, "o", "white"),
              ("large", "Adapted, up to 10,000 per side", st.DIDYMOS, "o", st.DIDYMOS))
    rh, top = 7.0, 11.0
    cv = st.Canvas(W, top + len(groups) * rh + 19.0)
    for k, (key, title, xl, lim, ticks) in enumerate(
            (("tpr_at_1pct_fpr", "True-positive rate at 1% false positives", "TPR at 1% FPR (95% CI)", (0, 0.04),
              (0, 0.01, 0.02, 0.03, 0.04)),
             ("auc", "Area under the curve", "AUC (95% CI)", (0.5, 0.6), (0.5, 0.55, 0.6)))):
        x0 = 30.0 + k * 78.0
        cv.letter(x0 - 30.0 if k == 0 else x0 - 4.0, 2.0, "ab"[k], title)
        ax = _rows_axes(cv, x0, top, 62.0, len(groups), rh)
        for i, g in enumerate(groups):
            if k == 0:
                cv.text(x0 - 2.0, _row_y(top, rh, i), st.DATASET_SHORT.get(g, "Pooled"), ha="right", va="center",
                        fontweight="bold" if g == "pooled" else "normal")
            for j, (s_, lab, col, mk, face) in enumerate(series):
                v = p1[s_][g][key]
                y = i + (j - 1) * 0.26
                ax.plot(v[1:], [y, y], color=col, lw=1.0, solid_capstyle="butt")
                ax.plot(v[0], y, ls="none", marker=mk, ms=3.0, mfc=face, mec=col, mew=1.0)
                sd.add("ab"[k], lab, group=g, statistic=key, estimate=v[0], ci_low=v[1], ci_high=v[2],
                       members=p1[s_][g]["members"])
        ax.set_xlim(*lim)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_xlabel(xl)
    _legend(cv, [_handle(marker=mk, ms=3.0, mfc=face, mec=col, mew=1.0) for _s, _l, col, mk, face in series],
            [lab for _s, lab, *_ in series], 2.0, top + len(groups) * rh + 11.0, ncol=3)
    return cv, sd
