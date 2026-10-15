"""Fig. 5: synthetic patients are private, faithful and useful.

a, membership inference per dataset: the loss attack's AUC and its TPR at 1% FPR, for unconditional sampling (the
privacy gate's audit) and for prompted sampling (amendment 11's V3x), with 95% intervals against chance (AUC 0.5; TPR
0.01, a no-information attack's TPR at 1% FPR). b, membership signal against fit: membership AUC against the
member/non-member loss gap over the 9 checkpoints, per dataset, with amendment 13's exact permutation P. c, synthetic against real rates per dataset, with the real-against-
real control. d, eICU hospitals: rank correlation of hospital-level rates, synthetic against real and real against
real. e, train on synthetic, test on real: the ratio of gains per endpoint. f, augmenting a small site.
"""

from __future__ import annotations

import numpy as np

import paper_figstyle as st
from paper_figdata import Inputs, find, p_holm

W, H = st.FULL, 170.0
RATES = (("icu_death", "ICU death"), ("hosp_death", "In-hospital death"), ("ever_imv", "IMV"),
         ("ever_vasopressor", "Vasopressor"), ("ever_rrt", "RRT"))
HOSP = (("icu_death", "ICU death"), ("ever_imv", "IMV"), ("ever_vasopressor", "Vasopressor"),
        ("los_median", "Median stay"))


def _forest_axes(cv, x, y0, w, h, xlim, ticks, labels, xlabel):
    ax = cv.axes(x, y0, w, h)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    return ax


def figure(inp: Inputs) -> tuple[st.Canvas, st.SourceData]:
    from matplotlib.lines import Line2D

    sd = st.SourceData("fig5")
    cv = st.Canvas(W, H)
    a3 = inp.json("a3/a3.json", sd=sd)
    v3x = inp.json("v3x/v3x.json", toolkit=True, sd=sd)
    v3 = inp.json("v3/v3_final.json", toolkit=True, sd=sd)
    sd.source(inp.stats_path)

    # ---- a: membership inference ----------------------------------------------------------------------------
    cv.letter(0, 2.0, "a", "Membership inference")
    p1 = a3["checkpoints"]["MUC32_main"]["p1"]
    rows = [*st.DATASETS, "pooled"]
    top, rh = 13.0, 5.6
    ys = {d: top + i * rh + (1.5 if d == "pooled" else 0) for i, d in enumerate(rows)}
    ylo, yhi = top - rh / 2, ys["pooled"] + rh / 2
    ax_auc = _forest_axes(cv, 24, ylo, 34, yhi - ylo, (0.45, 0.65), [0.45, 0.5, 0.55, 0.6, 0.65],
                          ["0.45", "0.50", "0.55", "0.60", "0.65"], "AUC")
    ax_tpr = _forest_axes(cv, 63, ylo, 26, yhi - ylo, (0.0, 0.06), [0, 0.01, 0.03, 0.05],
                          ["0", "0.01", "0.03", "0.05"], "TPR at 1% FPR")
    for ax in (ax_auc, ax_tpr):
        ax.set_ylim(yhi, ylo)
    st.refline(ax_auc, 0.5, color=st.SLATE[400])       # chance
    st.refline(ax_tpr, 0.01, color=st.SLATE[400])      # chance: a no-information attack's TPR at 1% FPR
    for d in rows:
        y = ys[d]
        cv.text(22.0, y, st.DATASET_SHORT.get(d, "Pooled"), ha="right", va="center",
                fontweight="bold" if d == "pooled" else "normal")
        u = p1.get(d)
        pr = v3x["a_loss"].get(d)
        for x, filled, off in ((u, False, -0.95), (pr, True, 0.95)):
            if not x:
                continue
            auc, ci = x["auc"], x["auc_ci"]
            tpr, tci = (x["tpr_at_1pct_fpr"][0], x["tpr_at_1pct_fpr"][1:]) if "tpr_at_1pct_fpr" in x else \
                (x["tpr_1fpr"], x["tpr_ci"])
            kw = {"ls": "none", "marker": "o", "ms": 3.2, "mec": st.DIDYMOS, "mew": 1.2,
                  "mfc": st.DIDYMOS if filled else "white"}
            ax_auc.plot(ci, [y + off, y + off], color=st.DIDYMOS, lw=1.0, solid_capstyle="butt")
            ax_auc.plot(auc, y + off, **kw)
            ax_tpr.plot([max(tci[0], 0), min(tci[1], 0.06)], [y + off, y + off], color=st.DIDYMOS, lw=1.0,
                        solid_capstyle="butt")
            ax_tpr.plot(min(tpr, 0.06), y + off, **kw)
            sd.add("a", "prompted" if filled else "unconditional", dataset=d, auc=auc, auc_low=ci[0], auc_high=ci[1],
                   tpr=tpr, tpr_low=tci[0], tpr_high=tci[1])
    ext, ctl = v3["b_extraction"], v3["controls"]
    pref = ctl["private_reference_loss"]
    sd.add("a", "extraction, exact 32 tokens", members=ext["exact32_member"], nonmembers=ext["exact32_nonmember"],
           jaccard_auc=ext["jaccard"]["auc"])
    sd.add("a", "control, copying generator", exact32=ctl["copying_generator"]["exact32_member"])
    sd.add("a", "control, private reference (prompted loss)", auc=pref["auc"], auc_low=pref["auc_ci"][0],
           auc_high=pref["auc_ci"][1])
    cv.fig.legend([Line2D([], [], color=st.DIDYMOS, lw=1.0, marker="o", ms=3.2, mfc="white", mec=st.DIDYMOS, mew=1.2),
                   Line2D([], [], color=st.DIDYMOS, lw=1.0, marker="o", ms=3.2, mfc=st.DIDYMOS, mec=st.DIDYMOS),
                   Line2D([], [], color=st.SLATE[400], lw=1.0, ls=(0, (3, 2)))],
                  ["Unconditional sampling", "Prompted sampling", "Chance"], loc="upper left", ncol=3,
                  bbox_to_anchor=(0.0, 1 - (yhi + 10.0) / H), columnspacing=1.2, handletextpad=0.4, borderaxespad=0)

    # ---- b: membership signal against fit ---------------------------------------------------------------------
    cv.letter(118, 2.0, "b", "Membership signal aligns with fit")
    bx = cv.axes(130, 10, 46, 44)
    gaps, aucs = [], []
    for d in st.DATASETS:
        pts = sorted(a3["frontier"][d]["points"], key=lambda p: p["nll_gap"])
        g, a = [p["nll_gap"] for p in pts], [p["auc"] for p in pts]
        gaps += g
        aucs += a
        bx.plot(g, a, color=st.SLATE[300], lw=1.0, zorder=1)
        bx.plot(g, a, ls="none", marker="o", ms=2.4, mfc="white", mec=st.SLATE[500], mew=1.0, zorder=2)
        sel = [p for p in pts if p["run"] == "MUC32_main"]
        if sel:
            bx.plot(sel[0]["nll_gap"], sel[0]["auc"], ls="none", marker="o", ms=4.8, color=st.BLUE[500], zorder=3, **st.RING)
        for p in pts:
            sd.add("b", "checkpoint", dataset=d, run=p["run"], nll_gap=p["nll_gap"], auc=p["auc"])
    bx.set_xlabel("Member/non-member loss gap (nats)")
    bx.set_ylabel("Membership AUC")
    bx.set_xlim(0, max(gaps) * 1.08)
    bx.set_ylim(0.5, max(aucs) + 0.01)
    fam8 = inp.family(8)
    if fam8:
        r_lo, r_hi = min(r["estimate"] for r in fam8), max(r["estimate"] for r in fam8)
        p_max, bnd = max(r.get("p_holm") or 0 for r in fam8), any(r.get("p_holm_bound") for r in fam8)
        cv.text(130, 65.5, f"Spearman r {st.num(r_lo, 2)}–{st.num(r_hi, 2)}, Holm {st.p_text(p_max, bnd)}",
                ha="left", va="center")
        for r in fam8:
            sd.add("b", "Spearman per dataset", dataset=r["dataset"], r=r["estimate"], p=r.get("p"),
                   p_holm=r.get("p_holm"))
    cv.fig.legend([Line2D([], [], color=st.SLATE[300], lw=1.0, marker="o", ms=2.4, mfc="white", mec=st.SLATE[500],
                          mew=1.0),
                   Line2D([], [], ls="none", marker="o", ms=4.8, color=st.BLUE[500], **st.RING)],
                  ["Checkpoints of one dataset", "Didymos 1.0"], loc="upper left", ncol=1,
                  bbox_to_anchor=(132 / W, 1 - 11.0 / H), handletextpad=0.4, borderaxespad=0)

    # ---- c: synthetic against real rates ---------------------------------------------------------------------
    y0 = 76.0
    cv.letter(0, y0, "c", "Synthetic against real stays")
    c3 = inp.json("c3/c3.json", sd=sd)
    cx = cv.axes(10, y0 + 9, 38, 38)
    lo = 0.005
    cx.plot([lo, 1], [lo, 1], color=st.SLATE[400], lw=1.0, ls=(0, (3, 2)), zorder=0)
    for d in st.DATASETS:
        x = c3["datasets"][d]
        for key, lab in RATES:
            real, twin, ctl_ = x["real"][key]["mean"], x["twins"][key]["mean"], x["control"][key]["mean"]
            death = "death" in key
            cx.plot(max(real, lo), max(ctl_, lo), ls="none", marker="x", ms=2.6, mew=1.0, color=st.SLATE[400], zorder=1)
            cx.plot(max(real, lo), max(twin, lo), ls="none", marker="^" if death else "o", ms=4.2,
                    color=st.CLINICAL if death else st.BLUE[500], zorder=2, **st.RING)
            sd.add("c", lab, dataset=d, real=real, synthetic=twin, real_control=ctl_)
    for ax_ in (cx,):
        ax_.set_xscale("log")
        ax_.set_yscale("log")
        ax_.set_xlim(lo, 1)
        ax_.set_ylim(lo, 1)
        ax_.set_xticks([0.01, 0.1, 1])
        ax_.set_xticklabels(["0.01", "0.1", "1"])
        ax_.set_yticks([0.01, 0.1, 1])
        ax_.set_yticklabels(["0.01", "0.1", "1"])
        ax_.minorticks_off()
    cx.set_xlabel("Real stays, rate")
    cx.set_ylabel("Synthetic stays, rate")
    inter = inp.json("c3/c3b.json", sd=sd)["inter_dataset"]
    cv.fig.legend([Line2D([], [], ls="none", marker="o", ms=4.2, color=st.BLUE[500], **st.RING),
                   Line2D([], [], ls="none", marker="^", ms=4.2, color=st.CLINICAL, **st.RING),
                   Line2D([], [], ls="none", marker="x", ms=2.6, mew=1.0, color=st.SLATE[400])],
                  ["Treatments", "Deaths", "Real control"], loc="upper left", ncol=1,
                  bbox_to_anchor=(11.5 / W, 1 - (y0 + 9.5) / H), handletextpad=0.3, borderaxespad=0)
    sd.add("c", "between-dataset correlation", synthetic=inter["twins"]["r"], synthetic_low=inter["twins"]["ci"][0],
           synthetic_high=inter["twins"]["ci"][1], control=inter["control"]["r"])

    # ---- d: eICU hospitals -----------------------------------------------------------------------------------
    cv.letter(58, y0, "d", "eICU hospitals")
    hosp = inp.json("c3/c3b.json")["eicu_hospitals"]
    dy0, drh = y0 + 9.0, 5.4
    dys = [dy0 + i * drh for i in range(len(HOSP))]
    dx = _forest_axes(cv, 80, dy0 - drh / 2, 30, len(HOSP) * drh, (-0.2, 1.0), [0, 0.5, 1.0], ["0", "0.5", "1"],
                      "Rank correlation with real")
    dx.set_ylim(dy0 + len(HOSP) * drh - drh / 2, dy0 - drh / 2)
    st.refline(dx, 0.0)
    for (key, lab), y in zip(HOSP, dys, strict=True):
        cv.text(78.0, y, lab, ha="right", va="center")
        for side, filled, off in (("twins_vs_real", True, -0.9), ("control_vs_real", False, 0.9)):
            x = hosp[key][side]
            dx.plot([max(x["ci"][0], -0.2), min(x["ci"][1], 1.0)], [y + off, y + off],
                    color=st.DIDYMOS if filled else st.SLATE[700], lw=1.0, solid_capstyle="butt")
            dx.plot(x["spearman"], y + off, ls="none", marker="o", ms=3.2,
                    mfc=st.DIDYMOS if filled else "white", mec=st.DIDYMOS if filled else st.SLATE[700], mew=1.2)
            sd.add("d", side, metric=key, spearman=x["spearman"], ci_low=x["ci"][0], ci_high=x["ci"][1],
                   hospitals=x["hospitals"])
    cv.fig.legend([Line2D([], [], color=st.DIDYMOS, lw=1.0, marker="o", ms=3.2),
                   Line2D([], [], color=st.SLATE[700], lw=1.0, marker="o", ms=3.2, mfc="white", mec=st.SLATE[700],
                          mew=1.2)],
                  ["Synthetic vs real", "Real vs real"], loc="upper left", ncol=2,
                  bbox_to_anchor=(59.0 / W, 1 - (dy0 + len(HOSP) * drh + 9.0) / H), columnspacing=1.0,
                  handletextpad=0.3, borderaxespad=0)

    # ---- e: train on synthetic, test on real ---------------------------------------------------------------------
    cv.letter(118, y0, "e", "Train on synthetic, test on real")
    eps = []
    for design in ("L24", "R"):
        bl = inp.json(f"a1/test_{design}/baselines.json", sd=sd)
        for prog, x in bl["programs"].items():
            a4 = x.get("a4")
            if a4 and a4.get("pooled", {}).get("tstr"):
                eps.append((design, prog, a4))
    ey0, erh = y0 + 9.0, 4.2
    eys = [ey0 + i * erh for i in range(len(eps))]
    ex = _forest_axes(cv, 150, ey0 - erh / 2, 27, len(eps) * erh, (0.4, 1.4), [0.5, 1.0],
                      ["0.5", "1"], "Ratio of gains")
    ex.set_ylim(ey0 + len(eps) * erh - erh / 2, ey0 - erh / 2)
    st.refline(ex, 1.0)
    for (design, prog, a4), y in zip(eps, eys, strict=True):
        cv.text(148.0, y, f"{st.ENDPOINT_SHORT.get(prog, prog)}{' (L)' if design == 'L24' else ''}", ha="right",
                va="center")
        for ds, v in a4.get("datasets", {}).items():
            if v.get("tstr") and v.get("trtr") and v["trtr"] > 0.55:
                rg = (v["tstr"] - 0.5) / (v["trtr"] - 0.5)
                ex.plot(np.clip(rg, 0.42, 1.38), y, ls="none", marker="o", ms=1.8, mfc="white", mec=st.SLATE[400],
                        mew=1.0, zorder=1)
                sd.add("e", "ratio of gains, dataset", design=design, program=prog, dataset=ds, ratio=rg)
        pl = a4["pooled"]
        rg = (pl["tstr"] - 0.5) / (pl["trtr"] - 0.5)
        ex.plot(rg, y, ls="none", marker="o", ms=4.4, color=st.BLUE[500], zorder=2, **st.RING)
        sd.add("e", "ratio of gains, pooled", design=design, program=prog, tstr=pl["tstr"], trtr=pl["trtr"], ratio=rg,
               rows=pl.get("rows"))
    # ---- f: augmentation ---------------------------------------------------------------------------------------
    fy = y0 + 46.0
    cv.letter(58, fy, "f", "Augmenting a small site")
    fx = cv.axes(72, fy + 8.0, 38, 22)
    fam9 = inp.family(9)
    for design, ls, lab in (("L24", "-", "24-hour landmark"), ("R", (0, (2.5, 1.5)), "Rolling anchors")):
        aug = inp.json(f"f8/aug_{design}.json", sd=sd)
        for m, tone in ((1, st.BLUE[400]), (3, st.BLUE[700])):      # amount as an ordinal tone
            xs, ms_ = [], []
            for n in (100, 300, 1000):
                cells = [x["cells"][f"n{n}_m{m}"]["diff_mean"] for by in aug["programs"].values() for x in by.values()
                         if f"n{n}_m{m}" in x["cells"] and x.get("test_positives", 0) >= 20]
                if len(cells) >= 8:
                    xs.append(n)
                    ms_.append(float(np.mean(cells)))
                    sd.add("f", f"{lab}, +{m}x twins", real_stays=n, mean_gain=ms_[-1], cells=len(cells),
                           positive=int(np.sum(np.asarray(cells) > 0)))
            fx.plot(xs, ms_, color=tone, ls=ls, lw=1.5, marker="o", ms=4.2, mfc=tone, **st.RING)
    fx.set_xscale("log")
    fx.set_xticks([100, 300, 1000])
    fx.set_xticklabels(["100", "300", "1,000"])
    fx.minorticks_off()
    fx.set_ylim(-0.01, 0.07)
    fx.set_yticks([0, 0.02, 0.04, 0.06])
    fx.set_yticklabels(["0", "0.02", "0.04", "0.06"])
    st.refline(fx, 0.0, axis="y")
    fx.set_xlabel("Real stays at the site")
    fx.set_ylabel("AUROC gain")
    r9 = find(fam9, design="L24", cell="n100_m1")
    if r9:
        pw, bw = p_holm(r9, "p_wilcoxon_holm", "p_wilcoxon")
        fx.text(0.97, 0.97, f"{r9['positive']} of {r9['cells']} cells gain,", transform=fx.transAxes, ha="right",
                va="top")
        fx.text(0.97, 0.80, f"{st.p_text(pw, bw)} (Holm)", transform=fx.transAxes, ha="right", va="top")
    cv.fig.legend([Line2D([], [], color=st.BLUE[400], lw=1.5, marker="o", ms=4.2, **st.RING),
                   Line2D([], [], color=st.BLUE[700], lw=1.5, marker="o", ms=4.2, **st.RING),
                   Line2D([], [], color=st.SLATE[700], lw=1.5),
                   Line2D([], [], color=st.SLATE[700], lw=1.5, ls=(0, (2.5, 1.5)))],
                  ["+1× synthetic", "+3× synthetic", "Landmark", "Rolling"], loc="upper left", ncol=2,
                  bbox_to_anchor=(60.0 / W, 1 - (fy + 39.0) / H), columnspacing=0.8, handletextpad=0.3,
                  borderaxespad=0, handlelength=1.6)

    return cv, sd
