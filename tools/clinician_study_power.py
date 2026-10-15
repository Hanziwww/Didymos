"""Paper P's clinician study (``docs/paper/clinician_study_protocol.md``): the pre-specified power simulation that fixes
the study's sizes. Every assumption is a named argument; the defaults are the protocol's.

* A (realism): calls correct with logit P = mu + rater + item (normal intercepts, SDs ``--a-rater-sd``,
  ``--a-item-sd``), the true pooled accuracy set by mu; the analysis is the pooled accuracy with a two-way
  cluster-robust interval (raters and items, Cameron-Gelbach-Miller), and equivalence when the 90% interval lies
  within 0.40-0.60.
* B1 (plausibility): ratings 1-5 = item mean + rater effect + the source effect (twin minus real) + noise, rounded
  and clipped; each rater sees every item once, the source counterbalanced; the analysis is the mean of per-item
  contrasts (twin minus real) with its item-level interval; non-inferiority when the one-sided 97.5% lower bound
  exceeds -0.5.
* B2 (explanations): three methods counterbalanced; the editing-minus-random contrast as B1's, a two-sided test.
* C (decision support): each case a true ICU-death risk on the logit scale, its outcome, the twin's risk (that logit
  plus noise, ``--c-twin-noise``), the comparator card's (``--c-comp-noise``) and each rater's unaided risk (more
  noise, a rater bias); each case is shown with the twin's card to half the raters and the comparator's to the other
  half (counterbalanced); the aided risk moves toward the card's by a weight of advice drawn per estimate (Beta, mean
  ``--c-woa``). Analyses: the twin card's mean per-estimate squared-error difference (aided minus unaided) with a
  two-way cluster-robust interval (raters and cases); the twin-specific part, the same difference minus that of the
  counterfactual blend toward the comparator's risk (paired). Both two-sided. Averaging any second opinion with
  independent errors lowers the Brier score: the counterfactual blend separates the advice's quality from that.
  ``--c-control-arm`` simulates the rejected design with a real comparator card for half the raters.

    python tools/clinician_study_power.py --sims 2000 --out runs/toolkit/clinician_power.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm


def twoway_se(y: np.ndarray, r: np.ndarray, c: np.ndarray) -> float:
    """The standard error of mean(y), clustered two ways (by r and by c; one observation per pair)."""
    e = y - y.mean()
    n = len(y)

    def v(g):
        s = np.bincount(g, weights=e)
        return float(np.sum(s ** 2)) / n ** 2

    vr, vc, vi = v(r), v(c), float(np.sum(e ** 2)) / n ** 2
    return float(np.sqrt(max(vr + vc - vi, vi)))


def logit(p):
    return np.log(p / (1 - p))


def sim_a(rng, raters: int, items: int, acc: float, sd_r: float, sd_i: float) -> bool:
    """One module-A study: True when equivalence to chance (0.40-0.60) is shown."""
    # mu so that the marginal accuracy is ``acc`` (the logistic-normal mean, by a probit approximation)
    s2 = sd_r ** 2 + sd_i ** 2
    mu = logit(acc) * np.sqrt(1 + np.pi * s2 / 8)
    u, w = rng.normal(0, sd_r, raters), rng.normal(0, sd_i, items)
    r, c = np.repeat(np.arange(raters), items), np.tile(np.arange(items), raters)
    y = (rng.random(raters * items) < 1 / (1 + np.exp(-(mu + u[r] + w[c])))).astype(float)
    m, se = y.mean(), twoway_se(y, r, c)
    z = norm.ppf(0.95)
    return bool(m - z * se > 0.40 and m + z * se < 0.60)


def contrast_items(rng, raters: int, items: int, effect: float, sd_item: float, sd_rater: float, sd_noise: float,
                   arms: int) -> tuple[float, float]:
    """Per-item contrast (arm 1 minus arm 0) of 1-5 ratings with ``arms`` counterbalanced versions; its mean and
    item-level standard error."""
    base = 3.2 + rng.normal(0, sd_item, items)
    rb = rng.normal(0, sd_rater, raters)
    diffs = []
    for i in range(items):
        version = (np.arange(raters) + i) % arms               # counterbalanced across raters
        x = base[i] + rb + np.where(version == 1, effect, 0.0) + rng.normal(0, sd_noise, raters)
        x = np.clip(np.round(x), 1, 5)
        a1, a0 = x[version == 1], x[version == 0]
        if len(a1) and len(a0):
            diffs.append(a1.mean() - a0.mean())
    d = np.array(diffs)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def sim_c(rng, raters: int, cases: int, deaths: float, twin_noise: float, rater_noise: float, woa: float,
          comp_noise: float, arm: bool = False) -> tuple:
    """One module-C study. The protocol's design (``arm`` False): every case with the twin's card; the twin-specific
    part of the gain by the counterfactual blend, each estimate's own weight of advice applied to the comparator's
    risk instead of the twin's (paired on the same estimates). The rejected control-arm design (``arm`` True): each
    case with the twin's card for half the raters and the comparator's for the other half (counterbalanced), the
    contrast by per-case differences. Returns the z statistic and mean of the twin card's Brier difference (aided
    minus unaided), the z statistic and mean of the twin-minus-comparator contrast, and the AUCs of the unaided
    raters, the twin and the comparator."""
    y = np.zeros(cases)
    y[: int(round(deaths * cases))] = 1.0
    rng.shuffle(y)
    true = np.where(y == 1, rng.normal(0.4, 1.2, cases), rng.normal(-2.6, 1.2, cases))   # case-cohort mix
    twin = 1 / (1 + np.exp(-(true + rng.normal(0, twin_noise, cases))))
    comp = 1 / (1 + np.exp(-(true + rng.normal(0, comp_noise, cases))))
    bias = rng.normal(0, 0.4, raters)
    r, c = np.repeat(np.arange(raters), cases), np.tile(np.arange(cases), raters)
    un = 1 / (1 + np.exp(-(true[c] + bias[r] + rng.normal(0, rater_noise, raters * cases))))
    a, b = woa * 4.0, (1 - woa) * 4.0
    w = rng.beta(a, b, raters * cases)
    if arm:
        is_twin = (r + c) % 2 == 0                              # counterbalanced: each case gets both cards
        advice = np.where(is_twin, twin[c], comp[c])
        d = ((1 - w) * un + w * advice - y[c]) ** 2 - (un - y[c]) ** 2
        dt = d[is_twin]
        z_twin = float(dt.mean() / twoway_se(dt, r[is_twin], c[is_twin]))
        per_case = np.array([d[(c == k) & is_twin].mean() - d[(c == k) & ~is_twin].mean() for k in range(cases)])
        z_diff = float(per_case.mean() / (per_case.std(ddof=1) / np.sqrt(cases)))
        return z_twin, float(dt.mean()), z_diff, float(per_case.mean()), auc(y[c], un), auc(y, twin), auc(y, comp)
    d = ((1 - w) * un + w * twin[c] - y[c]) ** 2 - (un - y[c]) ** 2
    cf = ((1 - w) * un + w * comp[c] - y[c]) ** 2 - (un - y[c]) ** 2    # the same shift toward the comparator
    spec = d - cf
    return (float(d.mean() / twoway_se(d, r, c)), float(d.mean()), float(spec.mean() / twoway_se(spec, r, c)),
            float(spec.mean()), auc(y[c], un), auc(y, twin), auc(y, comp))


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata

    rk = rankdata(s)
    n1 = y.sum()
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raters", type=int, nargs="+", default=[12, 16, 20])
    ap.add_argument("--a-items", type=int, default=60)
    ap.add_argument("--a-acc", type=float, nargs="+", default=[0.50, 0.52, 0.55])
    ap.add_argument("--a-rater-sd", type=float, default=0.3)
    ap.add_argument("--a-item-sd", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--b1-items", type=int, default=20)
    ap.add_argument("--b1-effect", type=float, nargs="+", default=[0.0, -0.2])
    ap.add_argument("--b2-items", type=int, default=20)
    ap.add_argument("--b2-effect", type=float, nargs="+", default=[0.4, 0.6])
    ap.add_argument("--b-item-sd", type=float, default=0.6)
    ap.add_argument("--b-rater-sd", type=float, default=0.4)
    ap.add_argument("--b-noise-sd", type=float, default=0.9)
    ap.add_argument("--c-cases", type=int, default=48)
    ap.add_argument("--c-deaths", type=float, default=0.25)
    ap.add_argument("--c-twin-noise", type=float, default=0.8)
    ap.add_argument("--c-rater-noise", type=float, nargs="+", default=[1.6, 2.2])
    ap.add_argument("--c-woa", type=float, nargs="+", default=[0.2, 0.3])
    ap.add_argument("--c-comp-noise", type=float, nargs="+", default=[0.8, 1.4, 2.2],
                    help="the comparator card's noise: as good as the twin, worse, as noisy as a rater")
    ap.add_argument("--c-control-arm", action="store_true",
                    help="the rejected design: half the raters see each case with the comparator's card")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    rep = {"args": vars(a), "A": [], "B1": [], "B2": [], "C": []}
    for n in a.raters:
        for sd_i in a.a_item_sd:
            for acc in a.a_acc:
                p = np.mean([sim_a(rng, n, a.a_items, acc, a.a_rater_sd, sd_i) for _ in range(a.sims)])
                rep["A"].append({"raters": n, "items": a.a_items, "item_sd": sd_i, "true_accuracy": acc,
                                 "power_equivalence": float(p)})
        for eff in a.b1_effect:
            ok = []
            for _ in range(a.sims):
                m, se = contrast_items(rng, n, a.b1_items, eff, a.b_item_sd, a.b_rater_sd, a.b_noise_sd, 2)
                ok.append(m - norm.ppf(0.975) * se > -0.5)
            rep["B1"].append({"raters": n, "items": a.b1_items, "true_difference": eff,
                              "power_noninferiority": float(np.mean(ok))})
        for eff in a.b2_effect:
            ok = []
            for _ in range(a.sims):
                m, se = contrast_items(rng, n, a.b2_items, eff, a.b_item_sd, a.b_rater_sd, a.b_noise_sd, 3)
                ok.append(abs(m / se) > norm.ppf(1 - 0.025 / 2))              # Holm: the first test at 0.025
            rep["B2"].append({"raters": n, "items": a.b2_items, "true_difference": eff,
                              "power_superiority": float(np.mean(ok))})
        for rn in a.c_rater_noise:
            for woa in a.c_woa:
                for cn in a.c_comp_noise:
                    zt, mt, zd, md, au_r, au_t, au_c = zip(*[sim_c(rng, n, a.c_cases, a.c_deaths, a.c_twin_noise,
                                                                   rn, woa, cn, a.c_control_arm)
                                                             for _ in range(a.sims)], strict=True)
                    rep["C"].append({"raters": n, "cases": a.c_cases, "twin_noise": a.c_twin_noise,
                                     "comparator_noise": cn, "rater_noise": rn, "weight_of_advice": woa,
                                     "auroc_unaided": float(np.mean(au_r)), "auroc_twin": float(np.mean(au_t)),
                                     "auroc_comparator": float(np.mean(au_c)),
                                     "design": "control arm" if a.c_control_arm else "counterfactual blend",
                                     "twin_card_brier_difference": float(np.mean(mt)),
                                     "power_twin_card": float(np.mean(np.abs(zt) > 1.96)),
                                     "twin_minus_comparator": float(np.mean(md)),
                                     "power_twin_minus_comparator": float(np.mean(np.abs(zd) > 1.96))})
    text = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8", newline="\n")
    for k in ("A", "B1", "B2", "C"):
        for row in rep[k]:
            print(k, json.dumps(row))


if __name__ == "__main__":
    main()
