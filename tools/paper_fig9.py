"""Fig. 9: Didymos at two hospitals it never saw, before and after adaptation without labels (Paper P amendment 17, on
amendment 16a's databases). From ``runs/paper/ext17/{split,grid,ext17}.json`` and amendment 17's statistics
(families 21 and 24 in ``runs/paper/stats/stats.json``).

a, the locked patient split of each database; b, next-token loss before and after adaptation (held-out FT patients of
the new databases, tuning split of the seven); c, AUROC on the locked TEST anchors; d, adapted minus zero-shot; e,
mean ranks over the TEST pairs; f, label efficiency of a local LightGBM against adapted Didymos.
"""

from __future__ import annotations

import numpy as np

import paper_figstyle as st
from paper_figdata import Inputs, find

W = st.FULL
EXT = ("ricd", "zigong")
EXT_NAME = {"ricd": "RICD", "zigong": "zigongICU"}
DESIGNS = (("R", "Rolling anchors"), ("L24e", "24-hour landmark"))
SLICES = (("test", "TEST, 50%", "locked; scoring only"), ("ft", "FT, 35%", "unlabelled continued training"),
          ("label", "LABEL, 15%", "recalibration and local models"))
SLICE_COLOUR = {"test": st.SLATE[700], "ft": st.BLUE[300], "label": "#F2C46D"}
ORANGE_DARK = "#A86F00"
# predictors of the ranks and the AUROC panel: (label, colour, marker, face)
STYLE = {
    "F1S": ("Adapted + local LightGBM", st.BLUE[700], "D", st.BLUE[700]),
    "F1": ("Didymos, adapted", st.DIDYMOS, "o", st.DIDYMOS),
    "Z0": ("Didymos, zero-shot", st.DIDYMOS, "o", "white"),
    "Z1": ("Didymos, mapped codes", st.BLUE[300], "o", "white"),
    "lgbm_local": ("Local LightGBM, 80% labelled", ORANGE_DARK, "s", ORANGE_DARK),
    "lgbm_ftlabel": ("Local LightGBM, 50% labelled", st.LGBM, "s", st.LGBM),
    "lgbm_label": ("Local LightGBM, 15% labelled", st.LGBM, "s", "white"),
    "tabpfn_v3_5": ("TabPFN-3.5, transferred", st.TABPFN, "^", st.TABPFN),
    "lgbm_pooled": ("LightGBM, transferred", st.SLATE[500], "s", st.SLATE[500]),
    "logistic_pooled": ("Logistic (L2), transferred", st.SLATE[500], "o", st.SLATE[500]),
    **{k: (v, st.CLINICAL, "D", st.CLINICAL) for k, v in (("sofa", "SOFA"), ("news", "NEWS"), ("mews", "MEWS"),
                                                          ("qsofa", "qSOFA"), ("sirs", "SIRS"))},
}


def _handle(**kw):
    from matplotlib.lines import Line2D

    return Line2D([], [], **{"ls": "none", **kw})


def _pairs(r: dict) -> list[tuple[str, str, str, dict]]:
    """The scored (design, database, program, result) rows in the figure's order: by design, database, endpoint."""
    order = list(st.ENDPOINT_SHORT)
    out = []
    for design, _ in DESIGNS:
        for d in EXT:
            progs = r["designs"].get(design, {}).get(d, {})
            for p in sorted(progs, key=order.index):
                out.append((design, d, p, progs[p]))
    return out


def figure(inp: Inputs):
    from matplotlib.patches import Rectangle

    sd = st.SourceData("fig9")
    sp = inp.json("ext17/split.json", sd=sd)
    gr = inp.json("ext17/grid.json", sd=sd)
    r = inp.json("ext17/ext17.json", sd=sd)
    sd.source(inp.stats_path)
    fam21 = [x for x in inp.family(21) if x.get("arm") == "F1"]
    H = 211.0
    cv = st.Canvas(W, H)

    # a: the locked split -------------------------------------------------------------------------------------------
    cv.letter(0, 2.0, "a", "Locked patient split of each new database")
    x0, bw, bh = 22.0, 64.0, 5.2
    for i, d in enumerate(EXT):
        y = 10.0 + i * 9.0
        pats = sp["patients"][d]
        tot = sum(pats.values())
        cv.text(x0 - 2.0, y + bh / 2, EXT_NAME[d], ha="right", va="center")
        cv.text(x0 - 2.0, y + bh / 2 + 3.4, f"{tot:,} patients", ha="right", va="center", color=st.TEXT2)
        ax = cv.axes(x0, y, bw, bh)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        left = 0.0
        for key, _lab, _use in SLICES:
            w = pats[key] / tot
            ax.add_patch(Rectangle((left, 0), w, 1, facecolor=SLICE_COLOUR[key], edgecolor="white", lw=1.0))
            ax.text(left + w / 2, 0.5, f"{pats[key]:,}", ha="center", va="center", fontsize=st.PT,
                    color="white" if key == "test" else st.TEXT)
            left += w
            sd.add("a", "patients", dataset=d, slice=key, patients=pats[key], stays=sp["stays"][d][key])
    for k, (key, lab, use) in enumerate(SLICES):
        y = 31.0 + k * 3.6
        ax = cv.axes(x0, y - 1.1, 2.6, 2.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=SLICE_COLOUR[key], edgecolor="none"))
        cv.text(x0 + 4.0, y, lab, ha="left", va="center")
        cv.text(x0 + 21.0, y, use, ha="left", va="center", color=st.TEXT2)

    # b: next-token loss before and after ---------------------------------------------------------------------------
    cv.letter(98.0, 2.0, "b", "Next-token loss, before and after adaptation")
    before = gr["before"]["by_dataset"]
    after = gr["cells"][next(k for k, v in gr["cells"].items() if v["run"] == gr["chosen"])]["by_dataset"]
    rows = [(d, f"{EXT_NAME[d]}, held-out FT") for d in EXT] + [(d, st.DATASET_SHORT[d]) for d in st.DATASETS]
    rh, top = 2.95, 9.5
    ypos = [i + (0.5 if i >= 2 else 0.0) for i in range(len(rows))]
    n_units = ypos[-1] + 1.0
    ax = cv.axes(130.0, top, 46.0, n_units * rh)
    ax.set_ylim(n_units - 0.5, -0.5)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    for (d, lab), yy in zip(rows, ypos, strict=True):
        b_, a_ = before[d], after[d]
        cv.text(128.5, top + (yy + 0.5) * rh, lab, ha="right", va="center")
        ax.plot([b_, a_], [yy, yy], color=st.BLUE[200], lw=1.5, solid_capstyle="butt", zorder=1)
        ax.plot(b_, yy, ls="none", marker="o", ms=3.2, mfc="white", mec=st.DIDYMOS, mew=1.0, zorder=2)
        ax.plot(a_, yy, ls="none", marker="o", ms=3.2, color=st.DIDYMOS, zorder=3)
        sd.add("b", "next-token loss", dataset=d, before=b_, after=a_, change=a_ - b_)
    ax.set_xlim(0.85, 2.25)
    ax.set_xticks([1.0, 1.5, 2.0])
    ax.set_xticklabels(["1.0", "1.5", "2.0"])
    ax.set_xlabel("Nats per token")
    cv.fig.legend([_handle(marker="o", ms=3.2, mfc="white", mec=st.DIDYMOS, mew=1.0),
                   _handle(marker="o", ms=3.2, color=st.DIDYMOS)], ["Before", "After"],
                  loc="upper right", bbox_to_anchor=(178.0 / cv.w, 1 - 5.5 / cv.h), ncol=2, handletextpad=0.4,
                  columnspacing=1.2, borderaxespad=0)

    # c and d: the TEST anchors -------------------------------------------------------------------------------------
    yc = 50.0
    cv.letter(0, yc, "c", "Discrimination on the locked TEST anchors")
    cv.letter(126.0, yc, "d", "Adapted minus zero-shot")
    pairs = _pairs(r)
    rh = 4.05
    top = yc + 9.0
    ypos, y = [], 0.0
    heads = []
    for k, (design, title) in enumerate(DESIGNS):
        heads.append((title, y, sum(1 for p in pairs if p[0] == design)))
        y += 1.15
        for p in pairs:
            if p[0] == design:
                ypos.append(y)
                y += 1.0
        y += 0.25
    n_units = y
    cv.text(62.0, top - 1.8, "Events/n", ha="right", va="bottom", color=st.TEXT2)
    cv.text(178.0, top - 1.8, r"$\mathit{P}$ (Holm)", ha="right", va="bottom", color=st.TEXT2)
    axc = cv.axes(66.0, top, 57.0, n_units * rh)
    axd = cv.axes(130.0, top, 32.0, n_units * rh)
    for a_ in (axc, axd):
        a_.set_ylim(n_units - 0.5, -0.5)
        a_.yaxis.set_visible(False)
        a_.spines["left"].set_visible(False)
    st.refline(axd, 0.0)
    for title, y0, _n in heads:
        cv.text(0.0, top + (y0 + 0.4) * rh, title, ha="left", va="center", fontweight="bold")
    for (design, d, p, x), yy in zip(pairs, ypos, strict=True):
        pr = x["predictors"]
        desc = x["descriptive_only"]
        ty = top + (yy + 0.5) * rh
        col = st.TEXT2 if desc else st.TEXT
        cv.text(2.5, ty, st.ENDPOINT_SHORT[p] + ("*" if desc else ""), ha="left", va="center", color=col)
        cv.text(33.0, ty, EXT_NAME[d], ha="left", va="center", color=st.TEXT2)
        f1 = pr["F1"]
        cv.text(62.0, ty, f"{f1['positives']}/{f1['n']}", ha="right", va="center", color=st.TEXT2)
        alpha = 0.55 if desc else 1.0
        z0, a1 = pr["Z0"]["auroc"], f1["auroc"]
        axc.plot([z0, a1], [yy, yy], color=st.BLUE[200], lw=1.5, solid_capstyle="butt", zorder=1, alpha=alpha)
        for key, off in (("lgbm_ftlabel", -0.22), ("tabpfn_v3_5", 0.22), ("F1S", 0.0), ("Z0", 0.0), ("F1", 0.0)):
            if key not in pr:
                continue
            lab, colr, mk, face = STYLE[key]
            ms = {"F1S": 2.9, "lgbm_ftlabel": 2.8, "tabpfn_v3_5": 3.0}.get(key, 3.4)
            axc.plot(pr[key]["auroc"], yy + off, ls="none", marker=mk, ms=ms, mfc=face, mec=colr, mew=1.0,
                     alpha=alpha, zorder={"F1": 5, "Z0": 4, "F1S": 6}.get(key, 3))
            sd.add("c", lab, design=design, dataset=d, program=p, auroc=pr[key]["auroc"],
                   ci_low=(pr[key].get("auroc_ci") or [None, None])[0],
                   ci_high=(pr[key].get("auroc_ci") or [None, None])[1], events=pr[key]["positives"],
                   n=pr[key]["n"], descriptive=desc)
        dz = x["arms_vs_z0"]["F1"]
        axd.plot(dz["ref_minus_ci"], [yy, yy], color=st.DIDYMOS, lw=1.0, solid_capstyle="butt", alpha=alpha, zorder=2)
        axd.plot(dz["ref_minus"], yy, ls="none", marker="o", ms=3.2, color=st.DIDYMOS, alpha=alpha, zorder=3)
        row = find(fam21, design=design, dataset=d, program=p)
        cv.text(178.0, ty, st.p_value(row.get("p_holm"), row.get("p_holm_bound")) if row else "–", ha="right",
                va="center", color=st.TEXT2 if not row else st.TEXT)
        sd.add("d", "adapted minus zero-shot", design=design, dataset=d, program=p, difference=dz["ref_minus"],
               ci_low=dz["ref_minus_ci"][0], ci_high=dz["ref_minus_ci"][1],
               p_holm=row.get("p_holm") if row else None, descriptive=desc)
    axc.set_xlim(0.5, 1.0)
    axc.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axc.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    axc.set_xlabel("AUROC")
    axd.set_xlim(-0.15, 0.3)
    axd.set_xticks([-0.1, 0, 0.1, 0.2, 0.3])
    axd.set_xticklabels([st.num(-0.1, 1), "0", "0.1", "0.2", "0.3"])
    axd.set_xlabel("Difference in AUROC (95% CI)")
    yl = top + n_units * rh + 10.0
    keys = ("F1", "Z0", "F1S", "lgbm_ftlabel", "tabpfn_v3_5")
    hs = [_handle(marker=STYLE[k][2], ms=3.2, mfc=STYLE[k][3], mec=STYLE[k][1], mew=1.0) for k in keys]
    cv.fig.legend(hs, [STYLE[k][0] for k in keys], loc="upper left", bbox_to_anchor=(2.0 / cv.w, 1 - yl / cv.h),
                  ncol=5, columnspacing=1.3, handletextpad=0.4, borderaxespad=0)

    # e: mean ranks -------------------------------------------------------------------------------------------------
    ye = yl + 9.0
    cv.letter(0, ye, "e", f"Mean rank over {r['ranks']['all']['tasks']} TEST pairs")
    ranks = r["ranks"]["all"]
    preds = ranks["predictors"]
    rh = 3.05
    top = ye + 6.5
    ax = cv.axes(38.0, top, 52.0, len(preds) * rh)
    ax.set_ylim(len(preds) - 0.5, -0.5)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    names = ranks.get("names") or r["ranks"]["names"]
    mat = np.array(r["ranks"]["auroc"], dtype=float)
    per_task = np.apply_along_axis(lambda v: len(v) + 1 - _rankdata(v), 1, mat)   # 1 = the highest AUROC
    for i, x in enumerate(preds):
        k = x["predictor"]
        lab, colr, mk, face = STYLE[k]
        bold = k in ("F1", "F1S")
        cv.text(36.5, top + (i + 0.5) * rh, lab, ha="right", va="center", fontweight="bold" if bold else "normal")
        j = names.index(k)
        jit = (np.arange(per_task.shape[0]) % 3 - 1) * 0.18
        ax.plot(per_task[:, j], i + jit, ls="none", marker="o", ms=1.6, mfc="white", mec=st.SLATE[300], mew=1.0,
                zorder=1)
        lo, hi = x["mean_rank_ci"]
        ax.plot([lo, hi], [i, i], color=colr, lw=1.5, solid_capstyle="butt", zorder=2)
        ax.plot(x["mean_rank"], i, ls="none", marker=mk, ms=3.2, mfc=face, mec=colr, mew=1.0, zorder=3)
        sd.add("e", lab, predictor=k, mean_rank=x["mean_rank"], ci_low=lo, ci_high=hi, firsts=x["firsts"],
               mean_auroc=x["mean_auroc"])
    ax.set_xlim(0.5, len(preds) + 0.5)
    ax.set_xticks([1, 5, 10, len(preds)])
    ax.set_xlabel("Rank (1 = highest AUROC)")

    # f: label efficiency -------------------------------------------------------------------------------------------
    cv.letter(98.0, ye, "f", "A local LightGBM on n labelled stays")
    axf = cv.axes(112.0, ye + 6.5, 64.0, len(preds) * rh - 2.0)
    st.refline(axf, 0.0, axis="y")
    diffs = {}
    for design, dbs in r["curves"].items():
        for d, progs in dbs.items():
            for p, c in progs.items():
                if r["designs"][design][d][p]["descriptive_only"]:
                    continue
                pts = sorted((int(k), v["lgbm_auroc"]) for k, v in c["points"].items() if v["lgbm_auroc"] is not None)
                xs = [n_ for n_, _ in pts]
                ys = [v - c["f1_auroc"] for _, v in pts]
                axf.plot(xs, ys, color=st.SLATE[300], lw=1.0, zorder=1)
                for n_, v in pts:
                    diffs.setdefault(n_, []).append(v - c["f1_auroc"])
                    sd.add("f", "LightGBM minus adapted Didymos", design=design, dataset=d, program=p, n=n_,
                           lightgbm_auroc=v, adapted_auroc=c["f1_auroc"], difference=v - c["f1_auroc"])
    ns = [k for k in sorted(diffs) if k <= 200]
    axf.plot(ns, [float(np.median(diffs[k])) for k in ns], color=st.LGBM, lw=2.0, marker="s", ms=3.0, zorder=3)
    for k in sorted(diffs):
        if k > 200:                                     # every LABEL stay: 466 (zigongICU) and 877 (RICD)
            axf.plot(k, float(np.median(diffs[k])), ls="none", marker="s", ms=3.0, color=st.LGBM, zorder=3)
    axf.set_xscale("log")
    axf.set_xticks([25, 50, 100, 200, 466, 877])
    axf.set_xticklabels(["25", "50", "100", "200", "466", "877"])
    axf.minorticks_off()
    axf.set_xlim(20, 1000)
    axf.set_ylim(-0.42, 0.12)
    axf.set_yticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1])
    axf.set_yticklabels([st.num(v, 1) if v else "0" for v in (-0.4, -0.3, -0.2, -0.1, 0, 0.1)])
    axf.set_xlabel("Labelled stays (LABEL; all = 877 RICD, 466 zigongICU)")
    axf.set_ylabel("LightGBM minus adapted Didymos (AUROC)")
    y0 = ye + 6.5 + (0.12 / 0.54) * (len(preds) * rh - 2.0)       # the zero line, in mm
    cv.text(113.5, y0 - 0.7, "Adapted Didymos, no labels", ha="left", va="bottom", color=st.TEXT2)
    cv.fig.legend([_handle(color=st.SLATE[300], lw=1.0, ls="-"), _handle(color=st.LGBM, lw=2.0, ls="-", marker="s",
                                                                      ms=3.0)],
                  ["One endpoint and database", "Median"], loc="upper left",
                  bbox_to_anchor=(113.5 / cv.w, 1 - (ye + 6.8) / cv.h), ncol=2, columnspacing=1.2,
                  handletextpad=0.4, borderaxespad=0)
    return cv, sd


def _rankdata(v):
    from scipy.stats import rankdata

    return rankdata(v)
