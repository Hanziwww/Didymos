"""Fig. 6: counterfactual editing and attention knockout reflect the trained model; attention rollout mirrors the
causal mask.

a, each attribution method's similarity to a random model (rank correlation of its group ranking between the trained
and a randomly initialized model; near 0, the ranking depends on training) against its agreement with counterfactual
editing, medians with interquartile ranges over 152 anchors; attention knockout, the toolkit's method, in blue. The
rule of amendment 9 is reported in the Methods, not drawn. b, the structural null:
rank correlation with the rollout of uniform causal attention, no model at all. c, the attention heads' entropy by
layer, trained and random. d, agreement with the generator's own edit effects, against the split-half ceiling.
"""

from __future__ import annotations

import numpy as np

import paper_figstyle as st
from paper_figdata import Inputs, find

W, H = st.FULL, 110.0
METHODS = (  # key, label; label position (data units) and alignment; None: labelled as a group
    ("rollout_sum", None, None),
    ("rollout_mean", None, None),
    ("norm_rollout_sum", None, None),
    ("rollout_minus_uniform_mean", "Rollout minus the null", ((0.13, -0.2), "left")),
    ("raw_last_mean", "Last-layer attention", ((0.06, 0.215), "left")),
    ("attn_x_grad_sum", "Attention × gradient, sums", ((0.66, 0.255), "left")),
    ("attn_x_grad_mean", "Attention × gradient, means", ((0.36, 0.40), "left")),
    ("grad_rollout_mean", "Gradient-weighted rollout", ((0.36, 0.47), "left")),
    ("knockout_sum", "Attention knockout", ((0.05, 0.71), "left")),
)
ROLLOUTS = ("Rollout: group sums,", "group means, norm-weighted")
GENERATOR = (("edit", "Editing", st.BLUE[500], "o", True), ("ig", "Integrated gradients", st.SLATE[700], "s", True),
             ("attn_x_grad", "Attention × gradient", st.SLATE[400], "D", True),
             ("attn_rollout", "Attention rollout", st.SLATE[700], "^", False))


def figure(inp: Inputs) -> tuple[st.Canvas, st.SourceData]:
    from matplotlib.lines import Line2D

    sd = st.SourceData("fig6")
    diag = inp.json("f4/attn_diag/attn_diag.json", sd=sd)
    pool = diag["pooled"]
    sd.source(inp.stats_path)
    cv = st.Canvas(W, H)

    # ---- a: sanity against agreement ---------------------------------------------------------------------------
    cv.letter(0, 2.0, "a", "Model dependence and agreement")
    ax = cv.axes(12, 10, 60, 72)
    st.refline(ax, 0.0, axis="y")
    for key, lab, place in METHODS:
        s, e = pool[f"sanity_{key}"], pool[f"edit_{key}"]
        mine = key == "knockout_sum"                    # the toolkit's method; the others in one neutral tone
        col = st.BLUE[500] if mine else st.SLATE[700]
        light = st.BLUE[200] if mine else st.SLATE[300]
        ax.plot([s["q25"], s["q75"]], [e["median"]] * 2, color=light, lw=1.0, zorder=1, solid_capstyle="butt")
        ax.plot([s["median"]] * 2, [e["q25"], e["q75"]], color=light, lw=1.0, zorder=1, solid_capstyle="butt")
        ax.plot(s["median"], e["median"], ls="none", marker="o", ms=4.0, color=col, zorder=3)
        if place:
            (lx, ly), ha = place
            ax.text(lx, ly, lab, ha=ha, va="center", color=col, zorder=4)
            if abs(ly - e["median"]) > 0.06:
                ax.plot([s["median"], lx - 0.01], [e["median"], ly], color=st.SLATE[300], lw=1.0, zorder=2)
        sd.add("a", lab or f"rollout ({key})", sanity_median=s["median"], sanity_q25=s["q25"],
               sanity_q75=s["q75"], agreement_median=e["median"], agreement_q25=e["q25"],
               agreement_q75=e["q75"], anchors=s["n"])
    ax.text(1.08, 0.135, ROLLOUTS[0], ha="right", va="center", color=st.SLATE[700])
    ax.text(1.08, 0.075, ROLLOUTS[1], ha="right", va="center", color=st.SLATE[700])
    ax.plot([0.95, 0.95], [0.045, -0.055], color=st.SLATE[300], lw=1.0, zorder=2)
    ax.set_xlim(-0.3, 1.1)
    ax.set_ylim(-0.35, 0.9)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75])
    ax.set_yticklabels([st.num(-0.25, 2), "0", "0.25", "0.50", "0.75"])
    ax.set_xlabel("Similarity to a random model's ranking")
    ax.set_ylabel("Agreement with counterfactual editing")
    r10 = find(inp.family(10), program="pooled")
    if r10:
        cv.text(12, 92.5, f"Knockout minus rollout: {st.num(r10['estimate'], 2, sign=True)}, Wilcoxon "
                          f"{st.p_text(r10['p'])}", ha="left", va="center")
        sd.add("a", "knockout minus rollout (agreement), Wilcoxon", median=r10["estimate"], anchors=r10["n"],
               p=r10["p"])

    # ---- b: the structural null --------------------------------------------------------------------------------
    cv.letter(86, 2.0, "b", "The structural null")
    rows = (("Rollout, trained", pool["uniform_rollout_sum"], st.SLATE[700]),
            ("Rollout, random", pool["runiform_rollout_sum"], st.SLATE[700]),
            ("Knockout, trained", pool["uniform_knockout_sum"], st.BLUE[500]),
            ("Knockout, random", pool["runiform_knockout_sum"], st.BLUE[500]))
    by0, brh = 13.0, 5.2
    bx = cv.axes(112, by0 - brh / 2, 26, len(rows) * brh)
    bx.set_ylim(by0 + len(rows) * brh - brh / 2, by0 - brh / 2)
    bx.yaxis.set_visible(False)
    bx.spines["left"].set_visible(False)
    bx.set_xlim(-0.4, 1.05)
    bx.set_xticks([0, 0.5, 1.0])
    bx.set_xticklabels(["0", "0.5", "1"])
    bx.set_xlabel("Correlation with the null")
    st.refline(bx, 0.0)
    for i, (lab, x, col) in enumerate(rows):
        y = by0 + i * brh
        cv.text(110.0, y, lab, ha="right", va="center")
        bx.plot([x["q25"], x["q75"]], [y, y], color=col, lw=1.0, solid_capstyle="butt")
        bx.plot(x["median"], y, ls="none", marker="o", ms=3.6, color=col)
        sd.add("b", lab, median=x["median"], q25=x["q25"], q75=x["q75"], anchors=x["n"])

    # ---- c: entropy by layer -------------------------------------------------------------------------------------
    cv.letter(142, 2.0, "c", "Attention heads")
    cx = cv.axes(150, 10, 28, 26)
    layers = np.arange(1, 13)
    for key, col, lab in (("trained_entropy_head", st.BLUE[500], "Trained"), ("random_entropy_head", st.SLATE[700],
                                                                              "Random")):
        v = np.asarray(pool[key], float)
        cx.plot(layers, v, color=col, lw=1.5, marker="o", ms=2.4)    # dense: no ring
        cx.text(7.0 if lab == "Random" else 9.5, v[6] + 0.025 if lab == "Random" else 0.70, lab, ha="center",
                va="bottom" if lab == "Random" else "top", color=col)
        for li, x in zip(layers, v, strict=True):
            sd.add("c", f"head entropy, {lab.lower()} model", layer=int(li), normalized_entropy=float(x))
    cx.set_xlim(0.5, 12.5)
    cx.set_xticks([1, 6, 12])
    cx.set_ylim(0.6, 1.0)
    cx.set_yticks([0.6, 0.8, 1.0])
    cx.set_yticklabels(["0.6", "0.8", "1.0"])
    cx.set_xlabel("Layer")
    cx.set_ylabel("Entropy / maximum")
    fm, rm = pool["trained_mass_first"], pool["trained_raw_mass_first"]
    sd.add("c", "rollout mass on the first token (trained)", median=fm["median"])
    sd.add("c", "last-layer attention on the first token (trained)", median=rm["median"])
    sd.add("c", "rollout mass on the first token (random)", median=pool["random_mass_first"]["median"])

    # ---- d: agreement with the generator ------------------------------------------------------------------------
    cv.letter(86, 58.0, "d", "Agreement with the generator's own effects")
    c2b = inp.json("f4/c2b.json", sd=sd)["programs"]
    progs = list(c2b)
    dy0, drh = 68.0, 5.6
    dx = cv.axes(112, dy0 - drh / 2, 66, len(progs) * drh)
    dx.set_ylim(dy0 + len(progs) * drh - drh / 2, dy0 - drh / 2)
    dx.yaxis.set_visible(False)
    dx.spines["left"].set_visible(False)
    dx.set_xlim(-0.2, 0.6)
    dx.set_xticks([0, 0.2, 0.4, 0.6])
    dx.set_xticklabels(["0", "0.2", "0.4", "0.6"])
    dx.set_xlabel("Rank correlation with the generator (median over anchors)")
    st.refline(dx, 0.0)
    for i, prog in enumerate(progs):
        y = dy0 + i * drh
        x = c2b[prog]
        cv.text(110.0, y, st.ENDPOINT_SHORT.get(prog, prog), ha="right", va="center")
        ceil = x["reliability_sb_median"]
        dx.plot([ceil, ceil], [y - 1.9, y + 1.9], color=st.CLINICAL, lw=1.5, solid_capstyle="butt")
        for j, (key, _lab, col, mk, filled) in enumerate(GENERATOR):
            v = x["agreement"][key]["spearman_median"]
            dx.plot(v, y + (j - 1.5) * 0.9, ls="none", marker=mk, ms=3.0, mfc=col if filled else "white", mec=col,
                    mew=1.0)
            sd.add("d", key, program=prog, agreement=v, ceiling=ceil, anchors=x["agreement"][key]["anchors"])
    hs = [Line2D([], [], ls="none", marker=mk, ms=3.0, mfc=col if filled else "white", mec=col, mew=1.0)
          for _k, _l, col, mk, filled in GENERATOR]
    hs.append(Line2D([], [], color=st.CLINICAL, lw=1.5))
    cv.fig.legend(hs, [g[1] for g in GENERATOR] + ["Ceiling (split-half)"], loc="upper left", ncol=3,
                  bbox_to_anchor=(86.0 / W, 1 - (dy0 + len(progs) * drh + 9.5) / H), handletextpad=0.4,
                  borderaxespad=0, columnspacing=1.0, labelspacing=0.3)
    return cv, sd
