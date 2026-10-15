"""Fig. 2: Didymos forecasts intensive care outcomes without labels, level with or ahead of supervised models.

a, discrimination (AUROC) on the 13 primary endpoints: Didymos's current estimate (the hazard estimate where it
applies, otherwise the sampled estimate, from 128 simulated futures), its pre-specified value (32 futures, the sampled
estimate), LightGBM, TabPFN-3.5, the best clinical score and every other comparator. b, the paired differences from
LightGBM and TabPFN-3.5 with their 95% intervals and amendment 13's Holm-adjusted P (primary family). c, AUROC against
the number of simulated futures for the two estimates. d, the sampled estimate minus LightGBM in each dataset. e, the
follow-up minus the pre-specified analysis, with P.
"""

from __future__ import annotations

import numpy as np

import paper_figstyle as st
from paper_figdata import ANCHORS, CLINICAL, E2_PROGRAMS, GROUP, ROWS, Inputs, find, p_holm

W, H = st.FULL, 190.0


def data(inp: Inputs, sd: st.SourceData) -> list[dict]:
    comp = inp.json("f10/comparators.json", sd=sd)
    out = []
    for sub, prog in ROWS:
        f2 = inp.json(f"f1/{sub}/f2.json", sd=sd)
        e2 = prog in E2_PROGRAMS[sub]
        blk = f2["f2"][prog] if e2 else f2["f1"][prog]
        cur = blk["estimators"]["e2"] if e2 else blk["twin_128"]
        pre = "e2_minus_" if e2 else "twin_128_minus_"
        cmp_ = blk["comparators"]
        models = comp["subsets"][sub][prog]["models"]
        clin = {c: models[c]["auroc"] for c in CLINICAL if c in models}
        best = max(clin, key=clin.get) if clin else None
        tw, lg = models.get("twin_128", {}).get("datasets", {}), models.get("lgbm_pooled", {}).get("datasets", {})
        per_ds = {ds: tw[ds]["auroc"] - lg[ds]["auroc"] for ds in st.DATASETS
                  if ds in tw and ds in lg and tw[ds].get("positives", 0) >= 5
                  and tw[ds].get("n", 0) - tw[ds].get("positives", 0) >= 5}
        r = {"subset": sub, "program": prog, "hazard": e2, "auroc": cur["auroc"], "n": cur["n"],
             "positives": cur["positives"], "lgbm": cmp_["lgbm_pooled"]["auroc"],
             "tabpfn": (cmp_.get("tabpfn_v3_5") or {}).get("auroc"), "prespecified": cmp_["twin_a1_32"]["auroc"],
             "best_clinical": best, "best_clinical_auroc": clin.get(best),
             "others": {m: x["auroc"] for m, x in models.items() if m not in ("twin_128", "twin_probe")},
             "d_lgbm": blk["paired"].get(f"{pre}lgbm_pooled"), "d_tabpfn": blk["paired"].get(f"{pre}tabpfn_v3_5"),
             "d_pre": blk["paired"].get(f"{pre}twin_a1_32"), "per_dataset": per_ds}
        for fam, key in ((1, "p_lgbm"), (2, "p_tabpfn"), (3, "p_pre"), (4, "p_clinical")):
            r[key], r[f"{key}_bound"] = p_holm(find(inp.family(fam), subset=sub, program=prog))
        out.append(r)
        sd.add("a", "Didymos (current)", subset=sub, program=prog, estimate="hazard" if e2 else "sampled",
               auroc=r["auroc"], events=r["positives"], n=r["n"])
        sd.add("a", "Didymos (pre-specified, 32 futures, sampled)", subset=sub, program=prog, auroc=r["prespecified"])
        sd.add("a", "LightGBM", subset=sub, program=prog, auroc=r["lgbm"])
        sd.add("a", "TabPFN-3.5", subset=sub, program=prog, auroc=r["tabpfn"])
        sd.add("a", "best clinical score", subset=sub, program=prog, comparator=best, auroc=r["best_clinical_auroc"],
               p_holm_vs_didymos_sampled=r["p_clinical"],
               p_bound=r["p_clinical_bound"])
        for m, v in r["others"].items():
            sd.add("a", "comparator", subset=sub, program=prog, comparator=m, auroc=v)
        for key, lab, pk, panel in (("d_lgbm", "minus LightGBM", "p_lgbm", "b"),
                                    ("d_tabpfn", "minus TabPFN-3.5", "p_tabpfn", "b"),
                                    ("d_pre", "follow-up minus pre-specified", "p_pre", "e")):
            if r[key]:
                sd.add(panel, f"Didymos {lab}", subset=sub, program=prog, difference=r[key]["diff"],
                       ci_low=r[key]["ci"][0], ci_high=r[key]["ci"][1], p_holm=r[pk],
                       p_bound=r[f"{pk}_bound"])
        for ds, v in per_ds.items():
            sd.add("d", "sampled estimate minus LightGBM", subset=sub, program=prog, dataset=ds, difference=v)
    sd.source(inp.stats_path)
    return out


def _lines(rows: list, top: float, rh: float):
    """The vertical layout: a header line per group, then its rows; (kind, item, y) and the bottom."""
    out, y = [], top
    for g in ("R128", "L24x"):
        out.append(("header", g, y))
        y += rh
        for r in rows:
            if r["subset"] == g:
                out.append(("row", r, y))
                y += rh
    return out, y


def figure(inp: Inputs) -> tuple[st.Canvas, st.SourceData]:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    sd = st.SourceData("fig2")
    rows = data(inp, sd)
    cv = st.Canvas(W, H)

    # ---- a and b: one row per endpoint ------------------------------------------------------------------------
    top, rh = 13.5, 5.2
    lines, bottom = _lines(rows, top, rh)
    y_lo, y_hi = top - rh / 2, bottom - rh / 2
    ax_a = cv.axes(50, y_lo, 52, y_hi - y_lo)
    ax_b = cv.axes(108, y_lo, 28, y_hi - y_lo)
    for ax in (ax_a, ax_b):
        ax.set_ylim(y_hi, y_lo)
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
    ax_a.set_xlim(0.5, 1.0)
    ax_a.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_a.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    ax_a.set_xlabel("AUROC")
    ax_b.set_xlim(-0.10, 0.20)
    ax_b.set_xticks([-0.1, 0, 0.1, 0.2])
    ax_b.set_xticklabels([st.num(-0.1, 1), "0", "0.1", "0.2"])
    ax_b.set_xlabel("Difference in AUROC")
    st.refline(ax_b, 0.0)
    cv.letter(0, 2.0, "a", "Discrimination on held-out anchors")
    cv.letter(104, 2.0, "b", "Didymos minus comparator")
    cv.text(47, 8.6, "Events/n", ha="right", va="center", color=st.TEXT2)
    cv.text(161.5, 8.6, "Minus LightGBM (95% CI)", ha="right", va="center", color=st.TEXT2)
    cv.text(180, 8.6, r"$\mathit{P}$ (Holm)", ha="right", va="center", color=st.TEXT2)
    for kind, r, y in lines:
        if kind == "header":
            cv.text(0, y, f"{GROUP[r]} ({ANCHORS[r]:,} anchors)", va="center", ha="left", fontweight="bold")
            continue
        cv.text(33.5, y, st.ENDPOINT_SHORT[r["program"]], va="center", ha="right")
        cv.text(47, y, f"{r['positives']:,}/{r['n']:,}", va="center", ha="right", color=st.TEXT2)
        oth = [v for m, v in r["others"].items() if m not in ("lgbm_pooled", "tabpfn_v3_5", r["best_clinical"])]
        ax_a.plot(np.clip(oth, 0.5, 1.0), np.full(len(oth), y), ls="none", marker="|", ms=4.0, mew=1.0,
                  color=st.LEARNER, zorder=1)
        ax_a.plot(r["best_clinical_auroc"], y, ls="none", marker="D", ms=3.0, color=st.CLINICAL, zorder=3)
        ax_a.plot(r["lgbm"], y - 1.05, ls="none", marker="s", ms=3.2, color=st.LGBM, zorder=4)
        if r["tabpfn"] is not None:
            ax_a.plot(r["tabpfn"], y + 1.05, ls="none", marker="^", ms=3.6, color=st.TABPFN, zorder=4)
        ax_a.plot([r["prespecified"], r["auroc"]], [y, y], color=st.DIDYMOS_TINT, lw=1.5, zorder=2,
                  solid_capstyle="butt")
        ax_a.plot(r["prespecified"], y, ls="none", marker="o", ms=3.2, mfc="white", mec=st.DIDYMOS, mew=1.0,
                  zorder=3)
        if r["hazard"]:
            ax_a.plot(r["auroc"], y, ls="none", marker="o", ms=4.4, color=st.DIDYMOS, zorder=5)
        else:
            ax_a.plot(r["auroc"], y, ls="none", marker="o", ms=4.4, mfc="white", mec=st.DIDYMOS, mew=1.5, zorder=5)
        for key, col, mk, off in (("d_lgbm", st.LGBM, "s", -1.05), ("d_tabpfn", st.TABPFN, "^", 1.05)):
            d = r[key]
            if not d:
                continue
            ax_b.plot([max(d["ci"][0], -0.10), min(d["ci"][1], 0.20)], [y + off, y + off], color=col, lw=1.0,
                      zorder=2, solid_capstyle="butt")
            ax_b.plot(d["diff"], y + off, ls="none", marker=mk, ms=3.2 if mk == "s" else 3.6, color=col, zorder=3)
        d = r["d_lgbm"]
        cv.text(161.5, y, f"{st.num(d['diff'], 3, sign=True)} ({st.ci_text(*d['ci'])})", va="center", ha="right")
        cv.text(180, y, st.p_text(r["p_lgbm"], r["p_lgbm_bound"]), va="center", ha="right")

    # the legend, two rows aligned with a
    def handle(**kw):
        return Line2D([], [], ls="none", **kw)

    ly = bottom + 9.0
    cv.text(50, ly, "Didymos:", va="center", ha="left", color=st.TEXT2)
    cv.fig.legend([handle(marker="o", ms=4.4, color=st.DIDYMOS),
                   handle(marker="o", ms=4.4, mfc="white", mec=st.DIDYMOS, mew=1.5),
                   Line2D([], [], color=st.DIDYMOS_TINT, lw=1.5, marker="o", ms=3.2, mfc="white", mec=st.DIDYMOS,
                          mew=1.0)],
                  ["Hazard estimate", "Sampled estimate", "Pre-specified, 32 futures"], loc="center left",
                  ncol=3, bbox_to_anchor=(62 / W, 1 - ly / H), columnspacing=1.4, handletextpad=0.4,
                  borderaxespad=0, handlelength=1.4)
    cv.text(50, ly + 4.2, "Comparators:", va="center", ha="left", color=st.TEXT2)
    cv.fig.legend([handle(marker="s", ms=3.2, color=st.LGBM), handle(marker="^", ms=3.6, color=st.TABPFN),
                   handle(marker="D", ms=3.0, color=st.CLINICAL), handle(marker="|", ms=4.0, mew=1.0,
                                                                         color=st.LEARNER)],
                  ["LightGBM", "TabPFN-3.5", "Best clinical score", "Other learned models and scores"],
                  loc="center left", ncol=4, bbox_to_anchor=(67 / W, 1 - (ly + 4.2) / H), columnspacing=1.4,
                  handletextpad=0.4, borderaxespad=0, handlelength=1.0)

    # ---- c: AUROC against the number of simulated futures ------------------------------------------------------
    y0 = bottom + 20.0
    cv.letter(0, y0, "c", "Simulated futures and the two estimates")
    ks = (1, 2, 4, 8, 16, 32, 64, 128)
    for i, prog in enumerate(("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880")):
        cx, cy = 9.0 + (i % 2) * 26.5, y0 + 9.0 + (i // 2) * 24.5
        ax = cv.axes(cx, cy, 21.0, 15.0)
        blk = inp.json("f1/R128/f2.json")["f2"][prog]
        e2 = [blk["curve"][str(k)]["e2"] for k in ks]
        e0 = [blk["curve"][str(k)]["e0"] for k in ks]
        st.refline(ax, blk["comparators"]["lgbm_pooled"]["auroc"], axis="y", color=st.LGBM, ls=(0, (1, 1.3)))
        ax.plot(ks, e0, color=st.DIDYMOS, lw=1.0, ls=(0, (2.5, 1.5)), marker="o", ms=2.6, mfc="white",
                mec=st.DIDYMOS, mew=1.0)
        ax.plot(ks, e2, color=st.DIDYMOS, lw=1.5, marker="o", ms=2.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 8, 128])
        ax.set_xticklabels(["1", "8", "128"] if i // 2 else [])
        ax.minorticks_off()
        ax.set_ylim(0.6, 1.0)
        ax.set_yticks([0.6, 0.8, 1.0])
        ax.set_yticklabels([] if i % 2 else ["0.6", "0.8", "1.0"])
        cv.text(cx, cy - 1.2, st.ENDPOINT_SHORT[prog], ha="left", va="bottom")
        for k in ks:
            sd.add("c", "hazard estimate", subset="R128", program=prog, futures=k, auroc=blk["curve"][str(k)]["e2"])
            sd.add("c", "sampled estimate", subset="R128", program=prog, futures=k, auroc=blk["curve"][str(k)]["e0"])
        sd.add("c", "LightGBM", subset="R128", program=prog, auroc=blk["comparators"]["lgbm_pooled"]["auroc"])
    cv.text(1.5, y0 + 29.0, "AUROC", rotation=90, va="center", ha="center")
    cv.text(9.0 + 23.75, y0 + 53.5, "Simulated futures per anchor", ha="center", va="top")
    cv.fig.legend([Line2D([], [], color=st.DIDYMOS, lw=1.5, marker="o", ms=2.6),
                   Line2D([], [], color=st.DIDYMOS, lw=1.0, ls=(0, (2.5, 1.5)), marker="o", ms=2.6, mfc="white",
                          mec=st.DIDYMOS, mew=1.0),
                   Line2D([], [], color=st.LGBM, lw=1.0, ls=(0, (1, 1.3)))],
                  ["Hazard", "Sampled", "LightGBM"], loc="upper left", ncol=3,
                  bbox_to_anchor=(8.0 / W, 1 - (y0 + 58.0) / H), columnspacing=1.0, handletextpad=0.4,
                  handlelength=1.8, borderaxespad=0)

    # ---- d and e: rows shared ------------------------------------------------------------------------------------
    cv.letter(62, y0, "d", "By dataset")
    cv.letter(124, y0, "e", "Follow-up minus pre-specified")
    ry = 3.0
    dlines, dbottom = _lines(rows, y0 + 8.5, ry)
    ylo, yhi = y0 + 8.5 - ry / 2, dbottom - ry / 2
    hd = cv.axes(90.5, ylo, 28.0, yhi - ylo)
    hd.set_xlim(0, 7)
    hd.set_ylim(yhi, ylo)
    cmap = mcolors.LinearSegmentedColormap.from_list("puor", ["#5E3C99", "#B2ABD2", "#F7F7F7", "#FDB863", "#E66101"])
    norm = mcolors.TwoSlopeNorm(vmin=-0.2, vcenter=0.0, vmax=0.2)
    ea = cv.axes(125, ylo, 26.0, yhi - ylo)
    ea.set_ylim(yhi, ylo)
    ea.yaxis.set_visible(False)
    ea.spines["left"].set_visible(False)
    ea.set_xlim(-0.05, 0.15)
    ea.set_xticks([0, 0.1])
    ea.set_xticklabels(["0", "0.1"])
    ea.set_xlabel("Difference in AUROC")
    st.refline(ea, 0.0)
    cv.text(180, y0 + 5.0, r"$\mathit{P}$ (Holm)", ha="right", va="center", color=st.TEXT2)
    from matplotlib.patches import Rectangle

    for kind, r, y in dlines:
        if kind == "header":
            cv.text(62, y, GROUP[r], va="center", ha="left", fontweight="bold")
            continue
        cv.text(89.5, y, st.ENDPOINT_SHORT[r["program"]], ha="right", va="center")
        for j, ds in enumerate(st.DATASETS):
            v = r["per_dataset"].get(ds)
            if v is None:
                hd.text(j + 0.5, y, "–", ha="center", va="center", color=st.TEXT2)
            else:
                hd.add_patch(Rectangle((j, y - ry / 2), 1, ry, facecolor=cmap(norm(np.clip(v, -0.2, 0.2))),
                                       edgecolor="none"))
        d = r["d_pre"]
        ea.plot([max(d["ci"][0], -0.05), min(d["ci"][1], 0.15)], [y, y], color=st.DIDYMOS, lw=1.0,
                solid_capstyle="butt")
        if r["hazard"]:
            ea.plot(d["diff"], y, ls="none", marker="o", ms=3.2, color=st.DIDYMOS)
        else:
            ea.plot(d["diff"], y, ls="none", marker="o", ms=3.2, mfc="white", mec=st.DIDYMOS, mew=1.2)
        cv.text(180, y, st.p_text(r["p_pre"], r["p_pre_bound"]), ha="right", va="center")
    hd.set_xticks(np.arange(7) + 0.5)
    hd.set_xticklabels([st.DATASET_SHORT[d] for d in st.DATASETS], rotation=90)
    hd.set_yticks([])
    hd.tick_params(axis="x", length=0, pad=1.5)
    for s in hd.spines.values():
        s.set_visible(False)
    cb = cv.axes(90.5, yhi + 17.5, 28.0, 1.8)
    cv.fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb, orientation="horizontal", ticks=[-0.2, 0, 0.2])
    cb.set_xticklabels([st.num(-0.2, 1), "0", "0.2"])
    cb.tick_params(length=2.0)
    cb.set_xlabel("Sampled estimate minus LightGBM", labelpad=1.5)
    return cv, sd
