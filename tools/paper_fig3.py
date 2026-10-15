"""Fig. 3: Didymos's step hazards are calibrated; free generation delays deaths, and correction largely restores
calibration.

a, the death hazard of the simulated futures over the real futures', by hour since the prediction time (rolling anchors
and the 24-hour landmark), beside alive discharge; the step calibration along real futures. b, observed over expected
(O/E) before and after the recalibration maps in force, on V1's 32-rollout draws, with 95% intervals and amendment 13's
P (O/E = 1, Holm). c, reliability before and after recalibration. d, the landmark's ICU death under the pre-specified
token cap and four times it. e, ICU death by stay length in the Didymos synthetic cohort and in real stays.
"""

from __future__ import annotations

import numpy as np

import paper_figstyle as st
from paper_figdata import Inputs, find, p_holm

W, H = st.FULL, 130.0
MAPS = (("imv_onset@1440", "logistic"), ("vasopressor_onset@1440", "logistic"), ("rrt_onset@2880", "logistic"),
        ("decompensation@1440", "isotonic"), ("aki_2@2880", "isotonic"), ("hypotension@240", "isotonic"),
        ("hyperkalemia_top@1440", "isotonic"))
E2 = ("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880")
LOS = (("0-24h", "<1"), ("24-72h", "1–3"), ("72-168h", "3–7"), ("168-infh", ">7"))
TIMES = "×"


def v1_arrays(inp: Inputs, prog: str, sd: st.SourceData):
    """V1's 32-rollout estimate for ``prog`` on the rolling anchors, with A1's labels and weights, raw and
    recalibrated (as amendment 11's V2 and V2b evaluated them)."""
    import pandas as pd

    from icutwin.toolkit import recalibration as rc

    maps_p = inp.path("recalibration/MUC32_main.json", toolkit=True)
    sd.source(maps_p, inp.path("a1/test_R/predictions.parquet"), inp.path("v1/forecast/e2.parquet", toolkit=True),
              inp.path("v1/forecast/predictions.parquet", toolkit=True))
    maps = rc.load(maps_p)
    a1 = pd.read_parquet(inp.path("a1/test_R/predictions.parquet"))
    lab = a1[a1["program"] == prog][["anchor", "label", "weight"]]
    if prog in E2:
        src = pd.read_parquet(inp.path("v1/forecast/e2.parquet", toolkit=True))
        src = src[src["program"] == prog].rename(columns={"e2": "p"})
    else:
        src = pd.read_parquet(inp.path("v1/forecast/predictions.parquet", toolkit=True))
        src = src[src["program"] == prog]
    x = lab.merge(src[["anchor", "p"]], on="anchor")
    x = x[np.isfinite(x["label"]) & np.isfinite(x["p"])]
    y, p, w = x["label"].to_numpy(), x["p"].to_numpy(), x["weight"].to_numpy()
    pc, _ = rc.apply(maps, prog, p)
    return y, p, pc, w


EDGES = (0.0, 0.005, 0.02, 0.05, 0.15, 0.4, 1.0)


def reliability(y, p, w, edges=EDGES, min_n: int = 20) -> list[tuple[float, float, int]]:
    """Bins of predicted risk on fixed edges (a share of 32 rollouts is often exactly 0), those with at least
    ``min_n`` anchors: (mean predicted, observed rate, anchors), weighted."""
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        m = (p >= lo) & ((p < hi) if i < len(edges) - 2 else (p <= hi))
        if m.sum() >= min_n:
            out.append((float(np.average(p[m], weights=w[m])), float(np.average(y[m], weights=w[m])), int(m.sum())))
    return out


def figure(inp: Inputs) -> tuple[st.Canvas, st.SourceData]:
    from matplotlib.lines import Line2D

    sd = st.SourceData("fig3")
    cv = st.Canvas(W, H)
    open_ = {"mfc": "white", "mew": 1.0}

    # ---- a: hourly hazards, simulated over real --------------------------------------------------------------
    cv.letter(0, 2.0, "a", "Death hazard, simulated over real")
    ax = cv.axes(12, 10, 52, 42)
    styles = {"R128": ("-", "rolling"), "L24x": ((0, (3.0, 1.6)), "landmark")}
    ends = []
    for sub, (ls, lab) in styles.items():
        hh = inp.json(f"f1/{sub}/f2.json", sd=sd)["f3"]["hourly_hazards"]
        for cause, col in (("death", st.BLUE[500]), ("alive", st.SLATE[700])):
            sim, real = np.asarray(hh[cause]["rollouts"], float), np.asarray(hh[cause]["real"], float)
            blocks = [(b * 6 + 3, sim[b * 6:(b + 1) * 6].sum() / real[b * 6:(b + 1) * 6].sum()) for b in range(8)
                      if real[b * 6:(b + 1) * 6].sum() > 0]
            bx_, by_ = zip(*blocks, strict=True)
            mk = {"marker": "o" if cause == "death" else "s", "ms": 2.8, "mec": col}
            mk.update({"mfc": col, "ms": 4.0, **st.RING} if sub == "R128" else open_)
            ax.plot(bx_, by_, color=col, ls=ls, lw=1.5, **mk)
            ends.append((by_[-1], f"{'Death' if cause == 'death' else 'Discharge alive'}, {lab}", col))
            for x_, r_ in blocks:
                sd.add("a", f"{cause}, {lab}", hours_since_prediction=x_, ratio_simulated_over_real=r_)
            for i in range(48):
                sd.add("a", f"{cause}, {lab}, hourly", hour=i + 0.5, simulated_hazard=float(sim[i]),
                       real_hazard=float(real[i]))
    ax.set_yscale("log")
    ax.set_ylim(0.2, 2.5)
    ax.set_yticks([0.25, 0.5, 1, 2])
    ax.set_yticklabels(["0.25", "0.5", "1", "2"])
    ax.minorticks_off()
    st.refline(ax, 1.0, axis="y")
    ax.set_xlim(0, 48)
    ax.set_xticks([0, 12, 24, 36, 48])
    ax.set_xlabel("Hours since the prediction time")
    ax.set_ylabel("Hazard, simulated over real")
    ends.sort(key=lambda e: -e[0])
    ys = np.log10([e[0] for e in ends])
    for i in range(1, len(ys)):                 # keep the end labels 0.1 decades apart
        ys[i] = min(ys[i], ys[i - 1] - 0.1)
    for (_, lab, col), yy in zip(ends, ys, strict=True):
        ax.text(49.0, 10 ** yy, lab, ha="left", va="center", color=col, clip_on=False)
    d1r = inp.json("f3/D1_R/f2.json", sd=sd)["f3"]["decompensation@1440"]["d1"]["pooled"]
    d1l = inp.json("f3/D1_L24/f2.json", sd=sd)["f3"]["icu_death@-1"]["d1"]["pooled"]
    d2l = inp.json("f1/L24x/f2.json")["f3"]["icu_death@-1"]["d2"]
    sd.add("a", "step calibration, death within 24 h, rolling", o_e=d1r["o_e"], ci_low=d1r["o_e_ci"][0],
           ci_high=d1r["o_e_ci"][1])
    sd.add("a", "step calibration, ICU death, landmark", o_e=d1l["o_e"], ci_low=d1l["o_e_ci"][0],
           ci_high=d1l["o_e_ci"][1])
    sd.add("a", "simulated over real death hazard until ICU discharge, landmark", ratio=d2l["drift_ratio"])

    # ---- b: O/E before and after recalibration ---------------------------------------------------------------
    cv.letter(100, 2.0, "b", "Recalibration on held-out anchors")
    fam = inp.family(5)
    top, rh = 13.0, 6.0
    rows, y = [], top
    for kind in ("logistic", "isotonic"):
        rows.append(("header", kind, y))
        y += rh
        for prog, k in MAPS:
            if k == kind:
                rows.append(("row", prog, y))
                y += rh
    ylo, yhi = top - rh / 2, y - rh / 2
    bx = cv.axes(129, ylo, 26, yhi - ylo)
    bx.set_xscale("log")
    bx.set_xlim(0.3, 3.0)
    bx.set_xticks([0.5, 1, 2])
    bx.set_xticklabels(["0.5", "1", "2"])
    bx.minorticks_off()
    bx.set_ylim(yhi, ylo)
    bx.yaxis.set_visible(False)
    bx.spines["left"].set_visible(False)
    bx.set_xlabel("Observed over expected")
    st.refline(bx, 1.0)
    cv.text(180, 4.8, r"$\mathit{P}$ for O/E = 1 (Holm)", ha="right", va="center", color=st.TEXT2)
    cv.text(168.0, 8.8, "Raw", ha="right", va="center", color=st.TEXT2)
    cv.text(180, 8.8, "Recal.", ha="right", va="center", color=st.TEXT2)
    for kind, item, yv in rows:
        if kind == "header":
            cv.text(100, yv, f"{item.capitalize()} map", ha="left", va="center", fontweight="bold")
            continue
        raw, cal = find(fam, program=item, comparison="oe_raw"), find(fam, program=item, comparison="oe_recalibrated")
        cv.text(127.5, yv, st.ENDPOINT_SHORT[item], ha="right", va="center")
        for r, filled, off in ((raw, False, -0.9), (cal, True, 0.9)):
            lo, hi = r["ci"]
            bx.plot([max(lo, 0.3), min(hi, 3.0)], [yv + off, yv + off], color=st.DIDYMOS, lw=1.0,
                    solid_capstyle="butt")
            bx.plot(min(max(r["estimate"], 0.3), 3.0), yv + off, ls="none", marker="o", ms=3.4,
                    mfc=st.DIDYMOS if filled else "white", mec=st.DIDYMOS, mew=1.2)
        cv.text(168.0, yv, st.p_value(*p_holm(raw)), ha="right", va="center")
        cv.text(180, yv, st.p_value(*p_holm(cal)), ha="right", va="center")
        for r, lab in ((raw, "raw"), (cal, "recalibrated")):
            sd.add("b", lab, program=item, o_e=r["estimate"], ci_low=r["ci"][0], ci_high=r["ci"][1], p=r.get("p"),
                   p_holm=r.get("p_holm"), anchors=r.get("n"), events=r.get("positives"))
    cv.fig.legend([Line2D([], [], color=st.DIDYMOS, lw=1.0, marker="o", ms=3.4, mfc="white", mec=st.DIDYMOS,
                          mew=1.2),
                   Line2D([], [], color=st.DIDYMOS, lw=1.0, marker="o", ms=3.4, mfc=st.DIDYMOS, mec=st.DIDYMOS)],
                  ["Raw", "Recalibrated"], loc="upper left", ncol=2, bbox_to_anchor=(100 / W, 1 - (yhi + 9.5) / H),
                  columnspacing=1.2, handletextpad=0.4, borderaxespad=0)

    # ---- c: reliability ----------------------------------------------------------------------------------------
    y0 = 82.0
    cv.letter(0, y0, "c", "Reliability")
    lo = 0.002
    minis = (("vasopressor_onset@1440", "New vasopressor"), ("aki_2@2880", "AKI stage ≥2"))
    for i, (prog, title) in enumerate(minis):
        cx = 10.0 + i * 24.0
        ax = cv.axes(cx, y0 + 12.0, 19.0, 19.0)
        y, p, pc, w = v1_arrays(inp, prog, sd)
        ax.plot([lo, 1], [lo, 1], color=st.REF, lw=1.0, ls=(0, (3, 2)), zorder=0)
        for vals, filled, lab in ((reliability(y, p, w), False, "raw"), (reliability(y, pc, w), True, "recalibrated")):
            xs = [min(max(v[0], lo), 1) for v in vals]
            ys = [min(max(v[1], lo), 1) for v in vals]
            ax.plot(xs, ys, color=st.DIDYMOS, lw=1.5 if filled else 1.0, ls="-" if filled else (0, (2.5, 1.5)),
                    marker="o", ms=2.8, mfc=st.DIDYMOS if filled else "white", mec=st.DIDYMOS, mew=1.0,
                    clip_on=False)
            for v in vals:
                sd.add("c", lab, program=prog, mean_predicted=v[0], observed=v[1], anchors=v[2])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, 1)
        ax.set_ylim(lo, 1)
        ax.set_xticks([0.01, 0.1, 1])
        ax.set_xticklabels(["0.01", "0.1", "1"])
        ax.set_yticks([0.01, 0.1, 1])
        ax.set_yticklabels(["0.01", "0.1", "1"] if i == 0 else [])
        ax.minorticks_off()
        cv.text(cx, y0 + 10.6, title, ha="left", va="bottom")
    cv.text(1.5, y0 + 21.5, "Observed", rotation=90, ha="center", va="center")
    cv.text(10.0 + 21.5, y0 + 37.0, "Predicted", ha="center", va="top")
    cv.fig.legend([Line2D([], [], color=st.DIDYMOS, lw=1.0, ls=(0, (2.5, 1.5)), marker="o", ms=2.8, mfc="white",
                          mec=st.DIDYMOS, mew=1.0),
                   Line2D([], [], color=st.DIDYMOS, lw=1.5, marker="o", ms=2.8)],
                  ["Raw", "Recalibrated"], loc="upper left", ncol=2, bbox_to_anchor=(9.0 / W, 1 - (y0 + 42.0) / H),
                  columnspacing=1.2, handletextpad=0.4, borderaxespad=0)

    # ---- d: the landmark's ICU death and the token cap ----------------------------------------------------------
    cv.letter(60, y0, "d", "ICU death and rollout length")
    blk = inp.json("f1/L24x/f2.json")["f2"]["icu_death@-1"]
    capped = (inp.json("a1/test_L24/a1.json", sd=sd)["censored_rollouts"],
              inp.json("f1/L24x/a1.json", sd=sd)["censored_rollouts"])
    dx = cv.axes(72, y0 + 12.0, 26, 22)
    au = {}
    for est, filled, lab in (("e2", True, "Hazard"), ("e0", False, "Sampled")):
        oe = (blk["at_a1_cap"][est]["observed_over_expected"], blk["estimators"][est]["observed_over_expected"])
        au[lab] = (blk["at_a1_cap"][est]["auroc"], blk["estimators"][est]["auroc"])
        dx.plot([0, 1], oe, color=st.DIDYMOS, lw=1.5 if filled else 1.0, ls="-" if filled else (0, (2.5, 1.5)),
                marker="o", ms=3.4, mfc=st.DIDYMOS if filled else "white", mec=st.DIDYMOS, mew=1.2)
        dx.text(-0.15, oe[0], lab, ha="right", va="center", color=st.DIDYMOS)
        for i_, (o_, a_) in enumerate(zip(oe, au[lab], strict=True)):
            sd.add("d", lab, cap="pre-specified" if i_ == 0 else "four times", o_e=o_, auroc=a_)
    dx.set_xlim(-1.05, 1.25)
    dx.set_ylim(0.8, 1.5)
    dx.set_yticks([0.8, 1.0, 1.2, 1.4])
    dx.set_yticklabels(["0.8", "1.0", "1.2", "1.4"])
    st.refline(dx, 1.0, axis="y")
    dx.set_xticks([0, 1])
    dx.set_xticklabels([f"1{TIMES} cap", f"4{TIMES} cap"])
    dx.set_ylabel("Observed over expected")
    sd.add("d", "capped futures", pre_specified=capped[0], four_times=capped[1])

    # ---- e: the synthetic cohort's deaths by stay length --------------------------------------------------------
    cv.letter(118, y0, "e", "Deaths in synthetic stays")
    gap = inp.json("c3/c3b.json", sd=sd)["death_gap"]
    ex = cv.axes(130, y0 + 12.0, 46, 22)
    for side, col, mk, off in (("real", st.CLINICAL, "s", -0.13), ("twins", st.DIDYMOS, "o", 0.13)):
        pooled = []
        for b, (key, _lab) in enumerate(LOS):
            num_ = den = 0.0
            for ds, x in gap[side].items():
                v = x["by_los"].get(key)
                if not v:
                    continue
                k = x["stays"] * v["share"]
                num_, den = num_ + k * v["icu_death"], den + k
                ex.plot(b + off, v["icu_death"] * 100, ls="none", marker=mk, ms=2.2, mfc="white",
                        mec=st.SLATE[400] if side == "real" else st.BLUE[300], mew=1.0,
                        zorder=1)
                sd.add("e", f"{side} by dataset", dataset=ds, stay_length=key, icu_death=v["icu_death"],
                       share=v["share"], stays=x["stays"])
            pooled.append(num_ / den if den else np.nan)
            sd.add("e", f"{side} pooled", stay_length=key, icu_death=pooled[-1])
        ex.plot(np.arange(4) + off, np.asarray(pooled) * 100, color=col, lw=1.5, marker=mk, ms=4.8, zorder=3,
                label="Real stays" if side == "real" else "Synthetic stays", **st.RING)
    ex.set_xticks(range(4))
    ex.set_xticklabels([lab for _k, lab in LOS])
    ex.set_xlim(-0.5, 3.5)
    ex.set_xlabel("ICU stay (days)")
    ex.set_ylabel("ICU death (%)")
    ex.set_ylim(0, 25)
    ex.set_yticks([0, 10, 20])
    ex.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.0)
    return cv, sd
