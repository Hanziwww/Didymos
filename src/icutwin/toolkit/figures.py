"""Paper figures from toolkit results (``docs/toolkit.md``): the case figure (Paper P's planned Figure 6: a twin with its
history, risk curves, a what-if and its attributions), PNG and PDF with matplotlib. Figures show twins only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def case_figure(session, forecast: dict, whatif: dict | None = None, explain: dict | None = None, path=None,
                response: str | None = None) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from icutwin.toolkit import inputs
    from icutwin.toolkit.card import CAUSE_NAMES, SERIES, label

    p = forecast["patient"]
    if p["kind"] != "twin":
        raise ValueError("paper figures show twins only")
    pat = inputs.load(session, p["ref"])
    stay = pat.stay(session.vocab)
    h = float(forecast["anchor"]["hours"])
    n_pan = 2 + (whatif is not None) + (explain is not None)
    fig, axes = plt.subplots(1, n_pan, figsize=(4.2 * n_pan, 3.4))
    ax = axes[0]
    t0 = stay.icu_in
    for i, (name, codes, _u) in enumerate(SERIES[:3]):
        pts = [((t - t0).total_seconds() / 3600, float(num)) for t, _r, code, num, _x in stay.timeline
               if code in codes and num is not None and np.isfinite(num) and max(0, h - 48) <= (t - t0).total_seconds()
               / 3600 <= h]
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, lw=1, label=name, color=f"C{i}")
    ax.axvline(h, color="k", ls="--", lw=0.8)
    ax.set_xlabel("hours since ICU admission")
    ax.set_title("a  History", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    inc = forecast["outputs"].get("incidence") or {}
    for c in ("death", "imv", "vasopressor", "rrt"):
        if c in inc:
            e2 = np.array([np.nan if x is None else x for x in inc[c]["e2"]], float)
            ax.plot(inc["hours"], e2, label=CAUSE_NAMES[c])
    ax.set_xlabel("hours after the prediction time")
    ax.set_ylabel("cumulative incidence (E2)")
    ax.set_title("b  Forecast", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    k = 2
    if whatif is not None:
        ax = axes[k]
        k += 1
        resp = response or next((r for r, x in whatif["outputs"]["responses"].items() if x["pre_specified"]), None)
        x = whatif["outputs"]["responses"][resp]
        labs, cs, lo, hi = [], [], [], []
        for arm, c in x["contrasts"].items():
            labs.append(arm.replace("give_", "give "))
            cs.append(np.nan if c["difference"] is None else c["difference"])
            lo.append(np.nan if c["ci"][0] is None else c["ci"][0])
            hi.append(np.nan if c["ci"][1] is None else c["ci"][1])
        y = np.arange(len(labs))
        if x.get("aa_band"):
            ax.axvspan(-x["aa_band"], x["aa_band"], color="C0", alpha=0.12, label="A/A band")
        ax.errorbar(cs, y, xerr=[np.array(cs) - np.array(lo), np.array(hi) - np.array(cs)], fmt="o", color="C0")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(y, labs)
        ax.set_xlabel(f"{resp}: give minus withhold")
        ax.set_title(f"c  What-if ({whatif['settings']['treatment']})", loc="left", fontsize=10)
    if explain is not None:
        ax = axes[k]
        g = explain["outputs"]["groups"][:8][::-1]
        ax.barh([r["group"] for r in g], [r["edit"] for r in g], color=["C1" if r["edit"] > 0 else "C0" for r in g])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("probe logit change (editing)")
        ax.set_title(f"d  {label(explain['settings']['program'])}", loc="left", fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    path = Path(path or session.paths.out / "figures" / "case")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        f = path.with_suffix(f".{ext}")
        fig.savefig(f, dpi=200)
        out.append(f)
    plt.close(fig)
    return out
