"""Paper P's follow-up materials (``docs/paper/results_followup.md``): amendment 7's experiments F1-F9, amendment 9's
attention diagnosis, amendment 10's comparators and amendment 11's toolkit validation (``docs/p1_paper_protocol.md``),
every table rendered from the aggregate JSON files under ``runs/paper`` (and ``runs/toolkit``) with its source and
annotations; with ``--tables``, every table also as CSV (``paper_results.export_tables``). The pre-registered results
(``docs/paper/results.md``) are not touched.

    python tools/paper_results_followup.py --root runs/paper --out docs/paper/results_followup.md \\
        --tables docs/paper/tables/followup
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_results import DS, DSN, NAMES, ci, export_tables, f, load, table  # noqa: E402

NAMES = {**NAMES, "remaining_los_gt_7d": "ICU stay > 7 days"}
EST = {"e0": "E0 (A1's share)", "e1": "E1 (Aalen-Johansen)", "e2": "E2 (hybrid)"}


def nm(p):
    return NAMES.get(p, p)


F_SUBSETS = (("R128", "f1/R128"), ("L24x", "f1/L24x"))


def f1_section(root: Path, subsets=F_SUBSETS, title: str = "## F1. Rollout count to 128 (accuracy, inference only)") -> str:
    s = [title, ""]
    sub = load(root / "f1/subsets/subsets.json")
    if sub and subsets == F_SUBSETS:
        s += [f"Anchors: R128 {sub['R128']['anchors']} (150 per dataset of test_R, inclusion weights kept); L24x "
              f"{sub['L24x']['anchors']} ({sub['L24x']['cases']} in-hospital deaths and {sub['L24x']['controls']} "
              f"controls weighted {f(sub['L24x']['control_weight'], 2)}: a case-cohort sample). Left out as continued "
              f"context windows: {sub['R128'].get('excluded_continued_window', 0)} (R), "
              f"{sub['L24x'].get('excluded_continued_window_or_label', 0)} (L24).", ""]
    for name, src in subsets:
        a1 = load(root / src / "a1.json")
        f2 = load(root / src / "f2.json")
        if not a1 and not f2:
            continue
        s += [f"### {name} (`runs/paper/{src}`)", ""]
        if a1:
            s += [f"{a1['anchors']} anchors x {a1['rollouts']} rollouts; {f(a1['censored_rollouts'] * 100, 1)}% end at "
                  f"the token cap.", ""]
        if f2 and f2.get("f1"):
            rows = []
            for p, x in f2["f1"].items():
                c, pd_ = x["comparators"], x["paired"]
                d32 = pd_.get("twin_128_minus_twin_a1_32") or {}
                dl = pd_.get("twin_128_minus_lgbm_pooled") or {}
                dt = pd_.get("twin_128_minus_tabpfn_v3_5") or {}
                rows.append([nm(p), x["twin_128"].get("n"), x["twin_128"].get("positives"), f(x["twin_128"].get("auroc")),
                             f(c.get("twin_a1_32", {}).get("auroc")), f(c.get("lgbm_pooled", {}).get("auroc")),
                             f(c.get("tabpfn_v3_5", {}).get("auroc")), f(c.get("twin_probe", {}).get("auroc")),
                             f(d32.get("diff")) + f" ({ci(d32.get('ci'))})" if d32 else "-",
                             f(dl.get("diff")) + f" ({ci(dl.get('ci'))})" if dl else "-",
                             f(dt.get("diff")) + f" ({ci(dt.get('ci'))})" if dt else "-",
                             f(x["twin_128"].get("observed_over_expected"), 2)])
            s += [table(["Endpoint", "Anchors", "Positives", "Twin, 128 rollouts", "Twin, A1's 32", "LightGBM",
                         "TabPFN-3.5", "Twin probe", "128 - A1's 32 (95% CI)", "128 - LightGBM", "128 - TabPFN-3.5",
                         "O/E (128)"], rows), ""]
        if a1:
            rows = []
            for p, x in a1["programs"].items():
                cur = x.get("rollout_curve")
                if cur:
                    rows.append([nm(p)] + [f(cur.get(str(k), {}).get("auroc")) for k in (1, 2, 4, 8, 16, 32, 64, 128)])
            if rows:
                s += ["AUROC against the number of rollouts (E0; the mean over 20 random subsets of each size):", "",
                      table(["Endpoint", "1", "2", "4", "8", "16", "32", "64", "128"], rows), ""]
    s += ["Annotations: the same anchors throughout; A1's 32-rollout Twin is an independent replicate (other seeds) of "
          "the same estimator. Weighted metrics (R: the design's inclusion weights; L24x: the case-cohort weights).", ""]
    return "\n".join(s)


def f2_section(root: Path, subsets=F_SUBSETS, title: str = "## F2. Estimators from the model's step hazards") -> str:
    s = [title, ""]
    for name, src in subsets:
        f2 = load(root / src / "f2.json")
        if not f2 or not f2.get("f2"):
            continue
        s += [f"### {name}", ""]
        rows, crows, srows = [], [], []
        for p, x in f2["f2"].items():
            e = x["estimators"]
            pd_ = x["paired"]
            d20 = pd_.get("e2_minus_e0") or {}
            dl = pd_.get("e2_minus_lgbm_pooled") or {}
            dt = pd_.get("e2_minus_tabpfn_v3_5") or {}
            rows.append([nm(p), e["e2"].get("n"), e["e2"].get("positives")] +
                        [f(e[k].get("auroc")) for k in ("e0", "e1", "e2")] +
                        [f(e[k].get("observed_over_expected"), 2) for k in ("e0", "e1", "e2")] +
                        [f(d20.get("diff")) + f" ({ci(d20.get('ci'))})" if d20 else "-",
                         f(dl.get("diff")) + f" ({ci(dl.get('ci'))})" if dl else "-",
                         f(dt.get("diff")) + f" ({ci(dt.get('ci'))})" if dt else "-"])
            for k in ("e0", "e2"):
                crows.append([nm(p), EST[k]] + [f(x["curve"].get(str(n), {}).get(k)) for n in (1, 2, 4, 8, 16, 32, 64,
                                                                                               128)])
            sc = x.get("self_check", {})
            srows.append([nm(p), f(sc.get("events"), 0), f(sc.get("compensator"), 1), f(sc.get("o_e"), 3)])
            if "at_a1_cap" in x:
                cut = x["at_a1_cap"]
                rows.append([nm(p) + ", at A1's cap", cut["e2"].get("n"), cut["e2"].get("positives")] +
                            [f(cut[k].get("auroc")) for k in ("e0", "e1", "e2")] +
                            [f(cut[k].get("observed_over_expected"), 2) for k in ("e0", "e1", "e2")] + ["-", "-", "-"])
        s += [table(["Endpoint", "Anchors", "Positives", "AUROC E0", "E1", "E2", "O/E E0", "E1", "E2",
                     "E2 - E0 (95% CI)", "E2 - LightGBM", "E2 - TabPFN-3.5"], rows), "",
              "AUROC against the number of rollouts, E0 and E2 on the same random subsets:", "",
              table(["Endpoint", "Estimator", "1", "2", "4", "8", "16", "32", "64", "128"], crows), "",
              "The self-check (sampled events against summed hazards within the rollouts; 1 expected):", "",
              table(["Endpoint", "Sampled events", "Summed hazards", "Ratio"], srows), ""]
    raw = load(root / "f1/R128/f2_raw.json") if subsets == F_SUBSETS else None
    cur = load(root / "f1/R128/f2.json")
    if raw and cur and raw.get("f2") and cur.get("f2"):
        rows = []
        for p in cur["f2"]:
            if p in raw["f2"]:
                a_, b_ = raw["f2"][p], cur["f2"][p]
                rows.append([nm(p), f(a_["estimators"]["e2"].get("auroc")), f(b_["estimators"]["e2"].get("auroc")),
                             f(a_["estimators"]["e2"].get("observed_over_expected"), 2),
                             f(b_["estimators"]["e2"].get("observed_over_expected"), 2),
                             f(a_["self_check"].get("o_e"), 3), f(b_["self_check"].get("o_e"), 3)])
        s += ["### R128: the first (raw) and the corrected (renormalized) hazards (amendment 8)", "",
              table(["Endpoint", "E2 AUROC, raw", "renormalized", "E2 O/E, raw", "renormalized",
                     "Self-check, raw", "renormalized"], rows), ""]
    chk = load(root / "f1/R128/hazard_check.json") if subsets == F_SUBSETS else None
    if chk:
        rows = [[DSN.get(ds, ds)] + [f(x.get(c, {}).get("ratio"), 3) + " / " + f(x.get(c, {}).get("ratio_normalized"), 3)
                                     for c in ("death", "alive", "imv", "vasopressor", "rrt")]
                for ds, x in chk.items()]
        s += ["### The hazards against the sampler's exact probabilities (`runs/paper/f1/R128/hazard_check.json`)", "",
              table(["Dataset", "Death (raw / renormalized)", "Alive discharge", "IMV", "Vasopressor", "RRT"], rows), "",
              "Ratio of the exact masked probability (grammar and support, as the sampler draws) to the extracted "
              "hazard, summed over a 20% sample of the timeline positions of 40 rollouts per dataset (sequences of at "
              "most 6,000 tokens); 1 means exact. 'Raw': the first scoring; 'renormalized': amendment 8's.", ""]
    s += [] if subsets != F_SUBSETS else [
          "Annotations: E0 = A1's share of positive rollouts among those whose outcome is known; E1 = Aalen-Johansen over "
          "the sampled events (competing causes, the cap censors); E2 = E1 with each rollout's events replaced by the "
          "model's hazard increments (Rao-Blackwellization). Treatment events are the first token of the engine's "
          "sets, death the ICU discharge 'dead', on the labels' decode clock. 'At A1's cap': the L24x rollouts "
          "censored where A1's (four times shorter) cap would have stopped them.", ""]
    return "\n".join(s)


def f3_section(root: Path, subsets=F_SUBSETS, title: str = "## F3. Mortality: where the death deficit comes from") -> str:
    s = [title, ""]
    rows = []
    for d in ("R", "L24") if subsets == F_SUBSETS else ():
        r = load(root / f"f3/D1_{d}/f2.json")
        if not r:
            continue
        for p, x in r["f3"].items():
            if "d1" not in x:
                continue
            d1 = x["d1"]
            rows.append([d, nm(p), f(d1["pooled"]["observed"], 1), f(d1["pooled"]["expected"], 1),
                         f(d1["pooled"]["o_e"], 2) + (f" ({ci(d1['pooled']['o_e_ci'], 2)})"
                                                      if d1["pooled"].get("o_e_ci") else "")]
                        + [f(d1.get(ds, {}).get("o_e"), 2) for ds in DS])
    if rows:
        s += ["### D1. Step calibration along the real futures (all test_R and test_L24 anchors)", "",
              table(["Design", "Endpoint", "Observed", "Expected (summed hazards)", "O/E (95% CI)"] + [DSN[x] for x in DS],
                    rows),
              "", "Annotation: the model's next-token probabilities of each event summed along each anchor's real future "
                  "(the compensator) against the events observed; O/E near 1 means calibrated step by step. Weighted "
                  "(R inclusion weights). Real futures that continue a context window are left out (4 R, 5 L24).", ""]
    rows, hrows = [], []
    for name, src in subsets:
        r = load(root / src / "f2.json")
        if not r:
            continue
        for p, x in r["f3"].items():
            if "d2" in x:
                d2 = x["d2"]
                rows.append([name, nm(p), f(d2["observed"], 1), f(d2["real_compensator"], 1),
                             f(d2["rollout_compensator"], 1), f(d2.get("drift_ratio"), 2)])
        hh = r["f3"].get("hourly_hazards")
        if hh:
            for c in ("death", "alive"):
                ro, re_ = np.array(hh[c]["rollouts"]), np.array(hh[c]["real"])
                hrows.append([name, "death" if c == "death" else "alive discharge"] +
                             [f(ro[a:b].mean() / max(re_[a:b].mean(), 1e-12), 2) for a, b in ((0, 6), (6, 12), (12, 24),
                                                                                            (24, 48)) if b <= len(ro)])
        mpt = r["f3"].get("minutes_per_token")
        if mpt:
            s += [f"Minutes per token ({name}): rollouts {f(mpt['rollouts_median'], 2)}, real futures "
                  f"{f(mpt.get('real_median'), 2)} (medians).", ""]
    if rows:
        s += ["### D2. Generation drift (the same anchors' rollouts against their real futures)", "",
              table(["Subset", "Endpoint", "Observed", "Real futures' summed hazards", "Rollouts' summed hazards (mean)",
                     "Drift ratio"], rows), ""]
    if hrows:
        s += ["Hourly hazard of the rollouts over the real futures' (paths still at risk), by hours after the anchor:", "",
              table(["Subset", "Cause", "0-6 h", "6-12 h", "12-24 h", "24-48 h"], hrows), ""]
    rows = []
    for name, src in subsets:
        r = load(root / src / "f2.json")
        if r and "d3" in r["f3"]:
            d3 = r["f3"]["d3"]
            for p in ("hospital_death@0", "icu_death@-1", "decompensation@1440"):
                if p in d3:
                    rows.append([name, nm(p), f(d3[p].get("0"), 3), f(d3[p].get("1"), 3)])
    if rows:
        s += ["### D3. Capped rollouts by the real outcome (mean share per anchor)", "",
              table(["Subset", "Endpoint", "Real outcome 0", "Real outcome 1"], rows), ""]
    c3 = load(root / "c3/c3b.json") if subsets == F_SUBSETS else None
    if c3:
        rows = []
        for ds in DS:
            t, r_ = c3["death_gap"]["twins"].get(ds), c3["death_gap"]["real"].get(ds)
            if not t or not r_:
                continue
            rows.append([DSN[ds], f(r_["end_kind"]["dead"], 3), f(t["end_kind"]["dead"], 3),
                         f(t["end_kind"]["censored"] + t["end_kind"]["unknown"], 3), f(t.get("icu_death_ended"), 3)] +
                        [f(r_["by_los"][k]["icu_death"], 3) + " / " + f(t["by_los"][k]["icu_death"], 3)
                         for k in r_["by_los"]])
        s += ["### D4. The bank's death gap (C3's twins against real held-out stays)", "",
              table(["Dataset", "ICU death, real", "ICU death, twins", "Twins censored", "Twins' ICU death among ended"]
                    + [f"ICU death, stay {k.replace('-infh', '+ h')} (real / twins)"
                       for k in c3["death_gap"]["real"]["mimic"]["by_los"]], rows), ""]
    if subsets == F_SUBSETS:
        s += ["Pre-specified reading: D1 near 1 with a drift ratio below 1 means the deficit is made in generation; D1 "
              "near the rollouts' O/E means per-step miscalibration; a gap between the caps means evaluation censoring.", ""]
    return "\n".join(s)


def f4_section(root: Path) -> str:
    r = load(root / "f4/c2b.json")
    if not r:
        return ""
    s = ["## F4. C2: attributions against the generator's own edit effects (`runs/paper/f4/c2b.json`)", ""]
    rows, srows = [], []
    for p, x in r["programs"].items():
        ag = x["agreement"]
        rows.append([nm(p), x["anchors"], f(x["reliability_split_half_median"], 2), f(x["reliability_sb_median"], 2)] +
                    [f(ag[m]["spearman_median"], 2) + " / " + f(ag[m]["top3_overlap_mean"], 2)
                     for m in ("edit", "ig", "attn_rollout", "attn_x_grad")] +
                    [f(ag["edit"]["sign_agreement"], 2) + f" (n={ag['edit']['signed_groups']})",
                     f(ag["ig"]["sign_agreement"], 2) + f" (n={ag['ig']['signed_groups']})"])
        sa = x["sanity"]
        srows.append([nm(p)] + [f(sa[f"{k}_{m}"]["spearman_median"], 2) for k in ("random", "perm")
                                for m in ("edit", "ig", "attn_rollout", "attn_x_grad")])
    s += [table(["Endpoint", "Anchors", "Reliability (split half)", "Ceiling (Spearman-Brown)",
                 "Probe editing: rho / top-3", "IG", "Attention rollout", "Attention x grad",
                 "Sign agreement, editing", "IG"], rows), "",
          "Sanity checks (Adebayo et al. 2018): the rank correlation of each method with itself after randomizing the "
          "model's weights (probe kept) or the probe's training labels; low means the method depends on what was learned:",
          "", table(["Endpoint", "Random weights: editing", "IG", "Attention rollout", "Attn x grad",
                     "Permuted labels: editing", "IG", "Attention rollout", "Attn x grad"], srows), "",
          "Annotations: 40 anchors per endpoint drawn from the upper half of probe risk (valid labels); every event group "
          "edited, 48 rollouts per prompt; the generator's risk is the rollouts' mean summed hazard (death, IMV) or "
          "positive share (AKI, hypotension) over the horizon after the prompt's end. Agreement is bounded by the "
          "reliability ceiling.", ""]
    return "\n".join(s)


A9_METHODS = (("rollout", "sum", "Attention rollout (C2, F4: group sums)"),
              ("rollout", "mean", "Attention rollout, group means"),
              ("rollout_minus_uniform", "sum", "Rollout less the structural null, sums"),
              ("rollout_minus_uniform", "mean", "Rollout less the null, means"),
              ("rollout_lift", "mean", "Rollout over the null (lift), means"),
              ("raw_last", "sum", "Last layer's attention, sums"),
              ("raw_last", "mean", "Last layer's attention, means"),
              ("norm_rollout", "sum", "Norm-weighted rollout (Kobayashi et al. 2020), sums"),
              ("norm_rollout", "mean", "Norm-weighted rollout, means"),
              ("attn_x_grad", "sum", "Attention x gradient (C2, F4: sums)"),
              ("attn_x_grad", "mean", "Attention x gradient, means"),
              ("grad_rollout", "sum", "Gradient-weighted rollout (Chefer et al. 2021), sums"),
              ("grad_rollout", "mean", "Gradient-weighted rollout, means"),
              ("knockout", "sum", "Attention knockout (Geva et al. 2023)"),
              ("uniform", "sum", "The structural null (no model), sums"))


def a9_section(root: Path) -> str:
    r = load(root / "f4/attn_diag/attn_diag.json")
    if not r:
        return ""
    p = r["pooled"]

    def md(k, nd=2, d=None):
        x = (d or p).get(k)
        return "-" if not x else f"{f(x['median'], nd)} ({f(x['q25'], nd)} to {f(x['q75'], nd)})"

    def m1(k, d=None):
        x = (d or p).get(k)
        return "-" if not x else f(x["median"], 2)

    def layers(k):
        x = p.get(k) or []
        return "-" if not x else f"{f(min(x), 2)}-{f(max(x), 2)}"

    s = ["## Amendment 9. Why attention rollout fails the randomization check, and remedies "
         "(`runs/paper/f4/attn_diag/attn_diag.json`; exploratory)", "",
         f"F4's anchors with prompts of at most 4,096 tokens ({p['anchors']} of 160), the trained model and F4's random "
         "initialization (seed 7, the trained probe kept). Medians over anchors of per-anchor Spearman correlations over "
         "the event groups (interquartile range in brackets). The structural null is the rollout of uniform causal "
         "attention, which involves no model. The rollout and attention x gradient computed here reproduce F4's stored "
         f"scores (median correlation {m1('check_rollout_trained')} trained, {m1('check_rollout_random')} random).", "",
         "(a) Diagnosis:", ""]
    rows = [["Rollout, trained against random (group sums: F4's check)", md("sanity_rollout_sum")],
            ["... with the groups' token counts held fixed (partial)", md("sanity_given_size_rollout_sum")],
            ["Rollout, trained against random, token level", md("tok_sanity_rollout")],
            ["Trained rollout against the structural null (group sums)", md("uniform_rollout_sum")],
            ["Random rollout against the structural null (group sums)", md("runiform_rollout_sum")],
            ["Total variation from the null, trained / random rollout",
             md("trained_tv_rollout_uniform") + " / " + md("random_tv_rollout_uniform")],
            ["Rollout against the groups' token counts, trained / random",
             md("size_rollout_sum") + " / " + md("rsize_rollout_sum")],
            ["Rollout mass on the first token ([BOS], in no group), trained / random",
             md("trained_mass_first") + " / " + md("random_mass_first")],
            ["Rollout mass on the statics, trained / random / null",
             md("trained_mass_statics") + " / " + md("random_mass_statics") + " / " + md("uniform_mass_statics")],
            ["Largest over smallest group, in tokens", md("size_max_over_min", 0)],
            ["Last layer's attention, trained against random: tokens / group sums / group means",
             m1("tok_sanity_raw_last") + " / " + m1("sanity_raw_last_sum") + " / " + m1("sanity_raw_last_mean")],
            ["Last token's attention entropy per head (share of the maximum), range over the 12 layers: trained / "
             "random", layers("trained_entropy_head") + " / " + layers("random_entropy_head")],
            ["Probe editing (the reference) against the token counts: signed / absolute",
             md("edit_size") + " / " + md("editabs_size")]]
    s += [table(["Measure", "Median (IQR)"], rows), ""]
    rows = []
    for meth, agg, label in A9_METHODS:
        tag = f"{meth}_{agg}"
        san, ed = (p.get(f"sanity_{tag}") or {}).get("median"), (p.get(f"edit_{tag}") or {}).get("median")
        ok = "-" if meth == "uniform" or san is None or ed is None else ("yes" if san <= 0.3 and ed > 0 else "no")
        rows.append([label, md(f"sanity_{tag}"), m1(f"sanity_given_size_{tag}"), md(f"edit_{tag}"),
                     m1(f"edit_given_size_{tag}"), m1(f"editabs_{tag}"), m1(f"size_{tag}"), m1(f"uniform_{tag}"), ok])
    s += ["(b) Methods (pooled over the four endpoints). A method passes if its sanity correlation is at most 0.3 and its "
          "agreement with editing is positive (amendment 9):", "",
          table(["Method", "Trained against random (sanity; want low)", "Sanity, token counts fixed",
                 "Agreement with probe editing", "Editing, token counts fixed", "With absolute editing",
                 "With token counts", "With the null", "Passes"], rows), ""]
    rows = []
    for prog, d in r["programs"].items():
        rows.append([nm(prog), d["anchors"]] + [m1(f"{k}_{tag}", d) for tag in
                                                ("rollout_sum", "rollout_minus_uniform_mean", "grad_rollout_mean",
                                                 "attn_x_grad_mean", "knockout_sum") for k in ("sanity", "edit")])
    s += ["(c) By endpoint (sanity / agreement with editing):", "",
          table(["Endpoint", "Anchors", "Rollout (C2): sanity", "editing", "Rollout less null, means: sanity",
                 "editing", "Gradient-weighted rollout, means: sanity", "editing", "Attn x grad, means: sanity",
                 "editing", "Knockout: sanity", "editing"], rows), "",
          "Annotations: editing is the trained model's probe-level counterfactual (C2, F4's `attrib.jsonl`), signed (> 0: "
          "the group raised the risk); unsigned methods are also compared with its absolute value. Knockout blocks "
          "attention from every other token to the group's tokens in all 12 layers and reads the probe's logit with the "
          "group's positions left out of the mean pooling. Agreement with the generator (F4's ceiling 0.13-0.54) is not "
          "re-estimated here.", ""]
    return "\n".join(s)


A10_NAMES = {"twin": "Twin (A1: 32 rollouts, share)", "twin_128": "Twin, 128 rollouts (share, E0)",
             "tabpfn_v3_5": "TabPFN-3.5", "tabpfn_v2": "TabPFN v2", "lgbm_pooled": "LightGBM, pooled",
             "lgbm_per_dataset": "LightGBM, per dataset", "random_forest": "Random forest",
             "extra_trees": "Extra trees", "cart": "Decision tree (CART)", "logistic_pooled": "Logistic regression, L2",
             "logistic_l1": "Logistic regression, L1 (LASSO)", "logistic_enet": "Logistic regression, elastic net",
             "naive_bayes": "Naive Bayes", "knn": "k-nearest neighbours", "mlp": "Multilayer perceptron",
             "twin_probe": "Twin probe (linear, frozen states)", "sofa": "SOFA", "sapsii_prob": "SAPS II",
             "oasis_prob": "OASIS", "news": "NEWS", "mews": "MEWS", "qsofa": "qSOFA", "sirs": "SIRS",
             "shock_index": "Shock index", "lactate": "Lactate", "neg_rox": "ROX index (lower worse)",
             "neg_map": "MAP (lower worse)", "kdigo": "KDIGO stage", "creatinine": "Creatinine",
             "potassium": "Potassium", "neg_platelets": "Platelets (lower worse)",
             "apache_icu_mort": "APACHE IVa, predicted ICU mortality (eICU)",
             "apache_hosp_mort": "APACHE IVa, predicted hospital mortality (eICU)",
             "apache_icu_los": "APACHE IVa, predicted ICU stay (eICU)", "saps3_prob": "SAPS 3 (SICdb)",
             "asa": "ASA class (INSPIRE)"}
A10_GROUPS = (("Tabular foundation models", ("tabpfn_v3_5", "tabpfn_v2")),
              ("Gradient boosting", ("lgbm_pooled", "lgbm_per_dataset")),
              ("Other trees", ("random_forest", "extra_trees", "cart")),
              ("Linear", ("logistic_pooled", "logistic_l1", "logistic_enet", "twin_probe")),
              ("Other learners", ("naive_bayes", "knn", "mlp")),
              ("Clinical scores", ("sofa", "sapsii_prob", "oasis_prob", "news", "mews", "qsofa", "sirs")),
              ("Bedside indicators", ("shock_index", "lactate", "neg_rox", "neg_map", "kdigo", "creatinine",
                                      "potassium", "neg_platelets")))
A10_COVERAGE = {"heart_rate_last": "Heart rate", "sbp_last": "Systolic pressure", "map_last": "MAP",
                "respiratory_rate_last": "Respiratory rate", "spo2_last": "SpO2", "temp_c_last": "Temperature",
                "gcs_last": "GCS", "platelet_count_last": "Platelets", "bilirubin_total_last": "Bilirubin",
                "creatinine_last": "Creatinine", "lactate_last": "Lactate", "wbc_last": "WBC",
                "pco2_arterial_last": "PaCO2", "urine_ml": "Urine output", "pf_vent_min": "PaO2/FiO2 under support",
                "dev_last": "Respiratory device"}


def a10_section(root: Path) -> str:
    r = load(root / "f10/comparators.json")
    if not r:
        return ""
    cov = (load(root / "f10/clinical/clinical.json") or {}).get("coverage", {})
    s = ["## Amendment 10. More comparators: clinical scores, linear models, trees and other learners "
         "(`runs/paper/f10/comparators.json`)", "",
         "Every comparator on A1's held-out anchors (3,500 per design) and on F1's subsets. The clinical scores "
         "come from the canonical CLIF tables (the harmonized raw values), with only data recorded or resulted by "
         "the prediction time, from the 24 hours before it. The learned baselines use A1's features and training rows. "
         "AUROC weighted as A1; intervals by a bootstrap over stays within datasets (1,000 resamples shared by all "
         "predictors, so differences are paired). On the anchors the Twin scored.", ""]
    if cov.get("L24"):
        ds = [d for d in DS if d in cov["L24"]]
        rows = [[label] + [f(cov["L24"][d].get(k), 2) for d in ds] for k, label in A10_COVERAGE.items()]
        s += ["(a) Inputs observed in the 24 hours before the 24-hour landmark (share of anchors; a missing component "
              "scores 0):", "", table(["Input"] + [DSN[d] for d in ds], rows), ""]
    chk = load(root / "f10/clinical/clinical_check.json")
    if chk:
        rows = [[DSN.get(d, d), x["anchors"], f(x.get("correlation")), f(x["identical"], 2), f(x["mean_abs_diff"], 2),
                 f(x["median_window"], 1), f(x["median_day1"], 1)] for d, x in chk.items()]
        s += ["A check of the windowed SOFA: at the 24-hour landmark its window nearly coincides with the first ICU "
              "day, so it should match the pipeline's first-day SOFA (`meds.derived`, the same CLIF tables). INSPIRE's "
              "window also holds data from before ICU admission (the operating room):", "",
              table(["Dataset", "Anchors", "Correlation", "Identical", "Mean absolute difference", "Median, window",
                     "Median, first day"], rows), ""]

    def matrix(block: dict, ref: str, title: str, e2: bool = False):
        progs = list(block)
        rows, drows = [], []
        if e2:
            rows.append(["Twin", "Twin, current estimator (E2, F2)"] +
                        [f(block[p].get("e2_auroc")) for p in progs])
        for model in [ref]:
            rows.append(["Twin", A10_NAMES[model]] + [f(block[p]["models"].get(model, {}).get("auroc")) for p in progs])
        for group, models in A10_GROUPS:
            for model in models:
                cells = [block[p]["models"].get(model) for p in progs]
                if not any(cells):
                    continue
                rows.append([group, A10_NAMES[model]] + [f(c.get("auroc")) if c else "-" for c in cells])
                drows.append([group, A10_NAMES[model]] + [
                    (f(c.get("ref_minus")) + " (" + ci(c.get("ref_minus_ci")) + ")") if c else "-" for c in cells])
        head = ["Group", "Predictor"] + [nm(p) for p in progs]
        return ([f"{title}, AUROC (positives / anchors: " +
                 "; ".join(f"{nm(p)} {block[p]['positives']} / {block[p]['anchors']}" for p in progs) + "):", "",
                 table(head, rows), "", f"{title}: the Twin minus each predictor, AUROC (95% CI):", "",
                 table(head, drows), ""])

    for design, title in (("L24", "(b) 24-hour landmark"), ("L48", "(c) 48-hour landmark"),
                          ("R", "(d) Rolling anchors")):
        if r["designs"].get(design):
            s += matrix(r["designs"][design], "twin", title)
    for sub, title in (("R128", "(e) R128 (the current Twin)"), ("L24x", "(f) L24x (the current Twin)")):
        if r["subsets"].get(sub):
            s += matrix(r["subsets"][sub], "twin_128", title, e2=True)
    rows = []
    for design in ("L24", "L48"):
        for p, x in r["designs"].get(design, {}).items():
            for name, nv in x.get("native", {}).items():
                sc, tw, lg = nv.get(name, {}), nv.get("twin", {}), nv.get("lgbm_pooled", {})
                if (sc.get("positives") or 0) < 10:              # an AUROC on a handful of events says little
                    continue
                rows.append([design, nm(p), A10_NAMES[name], sc.get("n"), sc.get("positives"), f(sc.get("auroc")),
                             f(tw.get("auroc")), f(lg.get("auroc")),
                             f(sc.get("ref_minus")) + " (" + ci(sc.get("ref_minus_ci")) + ")",
                             f(sc.get("observed_over_expected"), 2) if name != "asa" and name != "apache_icu_los"
                             else "-"])
    if rows:
        s += ["(g) The datasets' own admission scores, on their datasets' anchors (the Twin and LightGBM on the same "
              "anchors; endpoints with fewer than 10 positives left out):", "", table(["Design", "Endpoint", "Score", "Anchors", "Positives", "Score AUROC", "Twin",
                                      "LightGBM", "Twin - score (95% CI)", "Score O/E"], rows), ""]
    for design, prog, title in (("L24", "icu_death@-1", "(i) By dataset: ICU death at the 24-hour landmark"),
                                ("R", "decompensation@1440", "(j) By dataset: death within 24 h, rolling anchors")):
        x = r["designs"].get(design, {}).get(prog)
        if not x:
            continue
        ds = [d for d in DS if any(d in (m.get("datasets") or {}) for m in x["models"].values())]
        rows = []
        for model in ("twin", "tabpfn_v3_5", "lgbm_pooled", "random_forest", "logistic_pooled", "logistic_l1", "mlp",
                      "sapsii_prob", "oasis_prob", "sofa", "news", "mews", "qsofa"):
            c = x["models"].get(model)
            if c:
                rows.append([A10_NAMES[model]] + [f((c.get("datasets") or {}).get(d, {}).get("auroc")) for d in ds])
        pos = {d: (x["models"]["twin"].get("datasets") or {}).get(d, {}).get("positives") for d in ds}
        s += [f"{title} (AUROC; positives: " + ", ".join(f"{DSN[d]} {pos[d]}" for d in ds) + "):", "",
              table(["Predictor"] + [DSN[d] for d in ds], rows), ""]
    rows = []
    for design in ("L24", "L48"):
        for p, x in r["designs"].get(design, {}).items():
            for name in ("sapsii_prob", "oasis_prob"):
                c = x["models"].get(name)
                if c and p in ("icu_death@-1", "hospital_death@0"):
                    rows.append([design, nm(p), A10_NAMES[name], f(c.get("auroc")), f(c.get("observed_over_expected"), 2),
                                 f(c.get("calibration_slope"), 2), f(c.get("brier"))])
    if rows:
        s += ["(h) Calibration of the published mortality equations (pooled over datasets):", "",
              table(["Design", "Endpoint", "Score", "AUROC", "O/E", "Calibration slope", "Brier"], rows), ""]
    s += ["Annotations: clinical scores need no training; a component not charted in the window scores 0, so scores "
          "are weaker where inputs are missing (table a; NWICU and SICdb have no GCS, NWICU no urine output or blood "
          "gases). SAPS II leaves out its chronic-disease items. Bedside indicators set a missing value to normal. "
          "Learned baselines: A1's features of real training stays; settings chosen on the tuning store. E2's "
          "per-anchor values were not kept, so its row has no paired difference; the subsets' differences are "
          "against the 128-rollout share (E0). Descriptive: no multiplicity adjustment.", ""]
    return "\n".join(s)


def f5_section(root: Path) -> str:
    r = load(root / "f5/c1b.json")
    dz = load(root / "f5/design.json")
    if not r:
        return ""
    s = ["## F5. C1: validation of the what-if rollouts (`runs/paper/f5/c1b.json`)", ""]
    rows = [[t, resp, x["prefixes"], f(x["share_expected_sign"], 2), f(x["mean"], 3), ci(x["mean_ci"], 3), f(x["sd"], 3)]
            for t, rr in r["aa"].items() for resp, x in rr.items()]
    s += ["### (a) The null of C1's read-outs (two withhold arms, seeds 7 and 8)", "",
          table(["Treatment", "Response", "Prefixes", "Share of 'expected sign' (null 0.5)", "Mean (null 0)", "95% CI",
                 "SD across prefixes"], rows), ""]
    if dz:
        rows = [[t, x.get("eligible_starters"), x.get("matched"), f(x.get("logit_starters_mean"), 2),
                 f(x.get("logit_nonstarters_mean"), 2)] for t, x in dz["treatments"].items() if "matched" in x]
        s += ["### (b) The factual design: matching", "",
              table(["Treatment", "Eligible starters", "Matched pairs", "Mean logit, starters", "Mean logit, non-starters"],
                    rows), ""]
    rows, frows = [], []
    for t, rr in r["factual"].items():
        for resp, x in rr.items():
            o, tw, ts, tn = (x["observational_contrast"], x["twin_factual_contrast"], x["twin_contrast_starter"],
                             x["twin_contrast_nonstarter"])
            rows.append([t, resp, "+" if x["expected_sign"] > 0 else "-", o["pairs"], f(o["mean"], 2) + f" ({ci(o['ci'], 2)})",
                         f(tw["mean"], 2) + f" ({ci(tw['ci'], 2)})", f(ts["mean"], 2) + f" ({ci(ts['ci'], 2)})",
                         f(ts["share_expected_sign"], 2), f(tn["mean"], 2) + f" ({ci(tn['ci'], 2)})"])
            for grp in ("starter", "nonstarter"):
                fc = x[f"factual_{grp}"]
                frows.append([t, resp, grp, fc["anchors"], f(fc["coverage_80"], 2), f(fc["crps"], 2),
                              f(fc["crps_persistence"], 2), f(fc["pit_ks"], 2)])
    s += [table(["Treatment", "Response", "Expected", "Pairs", "Observational contrast (real starters - matched)",
                 "Twin's version (starters' give - matches' withhold)", "Twin give - withhold, starters",
                 "Share expected sign", "Twin give - withhold, non-starters"], rows), "",
          "The factual check (each factual arm's 16 rollouts against the real response):", "",
          table(["Treatment", "Response", "Group", "Anchors", "80% interval coverage", "CRPS",
                 "Persistence error", "PIT KS"], frows), "",
          "Annotations: starters are cut 1 minute before their first start; non-starters are matched within dataset at "
          "the same ICU hour (or within 3-6 h) on the propensity of starting within 6 h (LightGBM on A1's features), "
          "caliper 0.2 SD of the logit. Contrasts carry the data's confounding by indication; the comparison is whether "
          "the twin reproduces the observational contrast, not whether either is causal.", ""]
    return "\n".join(s)


def f6_section(root: Path) -> str:
    r = load(root / "c3/c3b.json")
    if not r:
        return ""
    s = ["## F6. C3: hospitals and intervals (`runs/paper/c3/c3b.json`)", ""]
    rows = []
    for k, x in r["eicu_hospitals"].items():
        t, c = x["twins_vs_real"], x["control_vs_real"]
        rows.append([k, t["hospitals"], f(t["spearman"], 2) + f" ({ci(t['ci'], 2)})",
                     c["hospitals"], f(c["spearman"], 2) + f" ({ci(c['ci'], 2)})"])
    s += [table(["Hospital-level rate", "Hospitals (twins)", "Spearman twins vs real (95% CI)", "Hospitals (control)",
                 "Spearman control vs real"], rows), "",
          f"Inter-dataset structure: twins {f(r['inter_dataset']['twins']['r'], 2)} "
          f"({ci(r['inter_dataset']['twins']['ci'], 2)}), real control {f(r['inter_dataset']['control']['r'], 2)} "
          f"({ci(r['inter_dataset']['control']['ci'], 2)}).", "",
          "Annotations: eICU hospitals with at least 30 held-out stays and 30 twins (control: 30 train stays); the real "
          "control's correlation is the ceiling sampling noise allows.", ""]
    return "\n".join(s)


def f7_section(root: Path) -> str:
    s = []
    for d in ("L24", "R"):
        r = load(root / f"c4/c4b_{d}.json")
        if not r:
            continue
        if not s:
            s = ["## F7. C4: neighbours as a predictor (`runs/paper/c4/c4b_<design>.json`)", ""]
        rows = []
        for p, x in r["programs"].items():
            for scope in ("all", "within_dataset"):
                rows.append([nm(p), scope.replace("_", " "), x["positives"]] +
                            [f(x.get(k, {}).get(scope, {}).get("10")) for k in r["representations"]] +
                            [f(x["state"][scope].get(str(k))) for k in (5, 25, 50)])
        s += [f"### {d}", "", table(["Endpoint", "Neighbours", "Positives"] + [f"{k} (k=10)" for k in r["representations"]]
                                    + ["state k=5", "state k=25", "state k=50"], rows), ""]
    if s:
        s += ["Annotations: leave-one-out kNN risk (the weighted mean label of the k most similar anchors of other stays), "
              "its AUROC; 'random init state' is the same architecture with random weights (a negative control).", ""]
    return "\n".join(s)


def _eq(x):
    if x is None:
        return "-"
    return x if isinstance(x, str) else f"{x:,.0f}"


def f8_section(root: Path) -> str:
    s = []
    for d in ("L24", "R"):
        r = load(root / f"f8/aug_{d}.json")
        if not r:
            continue
        if not s:
            s = ["## F8. Augmentation with repetitions (`runs/paper/f8/aug_<design>.json`)", ""]
        rows = []
        for p, byd in r["programs"].items():
            for ds, x in byd.items():
                for n in (100, 300, 1000):
                    base = x["cells"].get(f"n{n}_m1")
                    if not base:
                        continue
                    m3 = x["cells"].get(f"n{n}_m3", {})
                    rows.append([nm(p), DSN.get(ds, ds), x["test_positives"], n, f(base["auroc_real"]),
                                 f(base["diff_mean"]) + f" ({ci(base['diff_reps_ci'])}; {ci(base['diff_test_boot_ci'])})",
                                 f(m3.get("diff_mean")) + f" ({ci(m3.get('diff_reps_ci'))}; "
                                 f"{ci(m3.get('diff_test_boot_ci'))})" if m3 else "-",
                                 _eq(base.get("real_equivalent_n")), _eq(m3.get("real_equivalent_n")) if m3 else "-",
                                 f(x["cells"].get("twins_1000", {}).get("auroc"))])
        summ = []
        for n in (100, 300, 1000):
            for m in (1, 3):
                xs = [x["cells"][f"n{n}_m{m}"] for byd in r["programs"].values() for x in byd.values()
                      if f"n{n}_m{m}" in x["cells"] and x["test_positives"] >= 20]
                if xs:
                    dm = np.array([x["diff_mean"] for x in xs])
                    excl = sum(1 for x in xs if x["diff_test_boot_ci"][0] is not None and x["diff_test_boot_ci"][0] > 0)
                    summ.append([n, f"{m} x n", len(xs), f(float(dm.mean())), f(float(np.median(dm))),
                                 f"{int((dm > 0).sum())}/{len(xs)}", f"{excl}/{len(xs)}"])
        if summ:
            s += [f"### {d}: summary over datasets and endpoints (cells with at least 20 test positives)", "",
                  table(["Real stays n", "Twins added", "Cells", "Mean gain", "Median gain", "Cells with a gain",
                         "Cells whose test-anchor interval excludes 0"], summ), ""]
        s += [f"### {d}", "", table(["Endpoint", "Dataset", "Test positives", "Real stays n", "AUROC, n real",
                                     "+ n twins: difference (over 20 draws; over test anchors)", "+ 3n twins",
                                     "Real equivalent of n + n twins", "of n + 3n twins", "1,000 twins alone"], rows), ""]
    if s:
        s += ["Annotations: LightGBM per dataset (A1's features and settings, early stopping on the dataset's tuning "
              "rows), tested on the dataset's held-out A1 anchors. Two 95% intervals: over the 20 draws of the real "
              "stays (how much one small site's gain varies), and a paired bootstrap over test anchors of the "
              "draw-averaged predictions (the uncertainty of the expected gain). The real equivalent interpolates the "
              "real-only curve (100-4,000 stays, 10 draws each). Few test positives make a dataset's row noise.", ""]
    return "\n".join(s)


def f9_section(root: Path) -> str:
    s = []
    for d in ("R", "L24"):
        r = load(root / f"f9/transfer_{d}.json")
        if not r:
            continue
        if not s:
            s = ["## F9. Transfer to new institutions, characterized (`runs/paper/f9/transfer_<design>.json`)", ""]
        s += [f"### {d}", ""]
        sp = r.get("site_prompts")
        if sp:
            rows = []
            for p, x in sp.items():
                um, sm = x.get("unknown_minus_known", {}), x.get("swap_minus_known", {})
                sh = x.get("swap_shift", {})
                rows.append([nm(p), x["anchors"], f(x["known"].get("auroc")), f(x["unknown"].get("auroc")),
                             f(x["swap"].get("auroc")), f(um.get("auroc_diff")) + f" ({ci(um.get('ci'))})",
                             f(sm.get("auroc_diff")) + f" ({ci(sm.get('ci'))})",
                             f(x["known"].get("observed_over_expected"), 2), f(x["unknown"].get("observed_over_expected"), 2),
                             f(x["swap"].get("observed_over_expected"), 2), f(sh.get("slope"), 2), f(sh.get("corr"), 2)])
            s += ["(a) Site prompts (R128, 32 rollouts each):", "",
                  table(["Endpoint", "Anchors", "AUROC known", "unknown", "swapped", "unknown - known", "swapped - known",
                         "O/E known", "unknown", "swapped", "Swap shift: slope", "corr"], rows), ""]
        nov = r.get("novelty")
        if nov:
            rows = [[nm(p)] + [f(x[q]["twin"]) + " / " + f(x[q]["lgbm"]) for q in ("q1", "q2", "q3", "q4", "q5")]
                    for p, x in nov["programs"].items()]
            s += ["(b) AUROC by the anchor's novelty quintile (q1 most familiar; Twin / LightGBM):", "",
                  table(["Endpoint", "q1", "q2", "q3", "q4", "q5"], rows), ""]
        fam = r.get("eicu_familiarity")
        if fam:
            rows = [[nm(p)] + [f(x[k]["twin"]) + " / " + f(x[k]["lgbm"]) + f" ({x[k]['positives']})"
                               for k in ("volume_low", "volume_mid", "volume_high")]
                    for p, x in fam["programs"].items()]
            s += ["eICU hospitals by their number of train stays (tertiles; Twin / LightGBM, positives). Every eICU "
                  "hospital has its own site token in this vocabulary, so no anchor comes from a hospital the model "
                  "knows only as 'unknown':", "",
                  table(["Endpoint", "Low train volume", "Mid", "High"], rows), ""]
        lo = r.get("leave_one_dataset_out")
        if lo:
            rows = []
            for p, byd in lo.items():
                for ds, x in byd.items():
                    rows.append([nm(p), DSN.get(ds, ds), x.get("positives"), f(x.get("lgbm_within")),
                                 f(x.get("lgbm_pooled")), f(x.get("lgbm_lodo")), f(x.get("probe_pooled")),
                                 f(x.get("probe_lodo")), f(x.get("twin_zero_shot"))])
            s += ["(c) Leave one dataset out (AUROC on that dataset's held-out anchors):", "",
                  table(["Endpoint", "Dataset", "Positives", "LightGBM within", "pooled", "without the dataset",
                         "Probe pooled", "Probe without", "Twin (zero-shot)"], rows), ""]
    if s:
        s += ["Annotations: the Twin saw every dataset's training stays, so none of this is transfer of the generative "
              "model to a new site (that needs a model trained without a dataset). (a) shows how much the predictions "
              "depend on the site tokens; (c) what a supervised model loses at a site without labels.", ""]
    return "\n".join(s)


A11_NAMES = {"twin": "Twin", "sofa": "SOFA", "sapsii": "SAPS II", "oasis": "OASIS", "news": "NEWS",
             "lgbm_pooled": "LightGBM", "apache_iva": "APACHE IVa"}


def a11_section(root: Path) -> str:
    """Amendment 11: the application toolkit's validation (``runs/toolkit/v1`` ... ``v5``; V3 also in the audit)."""
    tk = root.parent / "toolkit"
    v1, v2, v3 = load(tk / "v1/v1.json"), load(tk / "v2/v2.json"), load(tk / "v3/v3.json")
    v4, v5 = load(tk / "v4/v4.json"), load(tk / "v5/v5.json")
    if not any((v1, v2, v3, v4, v5)):
        return ""
    s = ["## Amendment 11. The application toolkit's validation", "",
         "The toolkit (`src/icutwin/toolkit`, `docs/toolkit.md`) run on the RTX 5090 host; its rollouts are new draws "
         "(seeded per request), so V1 compares estimates within sampling error.", ""]
    if v1:
        adj, noise = load(tk / "v1/v1_adjusted.json"), load(tk / "v1/v1_noise.json")
        s += [f"### V1. Conformance (`runs/toolkit/v1/`): the original rule "
              f"{'passed' if v1['pass'] else 'not passed'}"
              + (f"; the added check {'passed' if noise['pass'] else 'not passed'}" if noise else "")
              + (f"; the study lead's adjusted criterion {'passed' if adj['pass'] else 'not passed'}"
                 if adj else ""), "",
              "Verdicts in the order they were made (amendment 11: the original rule fixed before the run; the added "
              "rollout-noise check and the study lead's adjusted criterion post hoc, after V1's result).", ""]
        if adj:
            rows = [[nm(p), f(x["auroc_toolkit"]), f(x["auroc_a1"]), f(x["auroc_f1_curve32"]), f(x["auroc_limit"]),
                     f(x["oe_toolkit"], 2), f(x["oe_a1"], 2), f(x["oe_f1"], 2), f(x["oe_limit"], 2),
                     "yes" if x["original_rule"] else "no", "yes" if x["added_check"] else "no",
                     ("A1, F1" if x["within_a1"] and x["within_f1"] else "A1" if x["within_a1"] else
                      "F1" if x["within_f1"] else "neither")]
                    for p, x in adj["programs"].items()]
            s += ["The toolkit against both paper runs on the same anchors (A1: 32 rollouts; F1: 128, its AUROC curve "
                  "at 32 rollouts and its O/E). Limit: 1.96 x sqrt(2) x the toolkit's rollout-bootstrap SD.", "",
                  table(["Endpoint", "Toolkit AUROC", "A1", "F1 (32)", "Limit", "Toolkit O/E", "A1 O/E", "F1 O/E",
                         "Limit", "Original rule", "Added check", "Within the limit of"], rows), ""]
        rows = [[nm(p), x["anchors"], f(x["auroc_a1"]) + f" ({ci(x['auroc_a1_ci'])})", f(x["auroc_toolkit"]),
                 f(x["oe_a1"], 2) + f" ({ci(x['oe_a1_ci'], 2)})", f(x["oe_toolkit"], 2), "yes" if x["pass"] else "no"]
                for p, x in v1["forecast"].items()]
        s += ["The toolkit's forecast on F1's R128 anchors (32 rollouts, A1's caps) against A1's 32-rollout Twin on the "
              "same anchors:", "", table(["Endpoint", "Anchors", "A1 AUROC (95% CI)", "Toolkit AUROC",
                                          "A1 O/E (95% CI)", "Toolkit O/E", "Within"], rows), ""]
        rows = [[nm(p), x["anchors"], f(x["auroc_e2_toolkit"]), f(x["f2_curve_32"]), "yes" if x["pass"] else "no"]
                for p, x in v1["e2"].items()]
        s += ["E2 at 32 rollouts against F2's curve (within 0.02):", "",
              table(["Endpoint", "Anchors", "Toolkit E2 AUROC", "F2 at k = 32", "Within"], rows), ""]
        rows = [[k, x["pairs"], f(x["factual_contrast"], 2) + f" ({ci(x['factual_ci'], 2)})",
                 f(x["f5"]["mean"], 2) + f" ({ci(x['f5']['ci'], 2)})", x["aa_prefixes"],
                 f(x["aa_mean"], 2) + f" ({ci(x['aa_ci'], 2)})", "yes" if x["pass"] else "no"]
                for k, x in v1["whatif"].items()]
        if rows:
            s += ["F5's factual contrast (starters' give minus matched non-starters' withhold) and the A/A null:", "",
                  table(["Treatment: response", "Pairs", "Toolkit (95% CI)", "F5's Twin (95% CI)", "A/A prefixes",
                         "A/A mean (95% CI)", "Within"], rows), ""]
    if v2:
        rows = []
        adopted = set(v2.get("adopted") or [])
        for k, x in v2["results"].items():
            prog, n = k.split("|")
            r, c = x["raw"], x.get("recalibrated") or {}
            rows.append([nm(prog), n, x["anchors"], f(r["observed_over_expected"], 2) + " → " +
                         f(c.get("observed_over_expected"), 2), f(r["calibration_slope"], 2) + " → " +
                         f(c.get("calibration_slope"), 2), f(r["ece"]) + " → " + f(c.get("ece")),
                         f(r["brier"], 4) + " → " + f(c.get("brier"), 4), f(r["auroc"]) + " → " + f(c.get("auroc")),
                         "yes" if x.get("auroc_within_0005") else "no", "yes" if prog in adopted else "no"])
        s += ["### V2. Recalibration (`runs/toolkit/v2/v2.json`)", "",
              "Isotonic maps from the tuning split's R anchors (E2 for the four hazard programs, the share otherwise), "
              "applied to V1's draws (A1's labels); raw → recalibrated. A map is adopted by the toolkit where AUROC "
              "stayed within 0.005 at both 32 and 16 rollouts.", "",
              table(["Endpoint", "Rollouts", "Anchors", "O/E", "Calibration slope", "ECE", "Brier", "AUROC",
                     "AUROC within 0.005", "Adopted"], rows), "",
              "Annotations: the isotonic steps tie risks, so AUROC can fall (most for the rarer endpoints); a map fixes "
              "O/E only where the tuning split's miscalibration matches the held-out anchors' (death within 24 h: raw "
              "O/E 1.98 on tuning against 1.71 here, so its map overcorrects).", ""]
    v2b = load(tk / "v2/v2b.json")
    if v2b:
        rows = []
        for prog, x in v2b["programs"].items():
            for n, e in x["evaluation"].items():
                r, c = e["raw"], e["recalibrated"]
                rows.append([nm(prog), f(x["fit"]["a"], 2), f(x["fit"]["b"], 2), n, e["anchors"],
                             f(r["observed_over_expected"], 2) + " → " + f(c["observed_over_expected"], 2),
                             f(r["ece"]) + " → " + f(c["ece"]), f(r["brier"], 4) + " → " + f(c["brier"], 4),
                             f(r["auroc"]) + " → " + f(c["auroc"]), "yes" if x["adopted"] else "no"])
        s += ["### V2b. Logistic recalibration of the treatment starts (`runs/toolkit/v2/v2b.json`)", "",
              "Post hoc (the study lead's decision after V2, recorded before it ran): logit(p') = a + b logit(p) on "
              "E2, fitted on V2's tuning anchors; adopted when b > 0 and, at 32 and 16 rollouts, O/E is closer to 1 "
              "and ECE no larger. Raw → recalibrated on V1's draws:", "",
              table(["Endpoint", "a", "b", "Rollouts", "Anchors", "O/E", "ECE", "Brier", "AUROC", "Adopted"], rows), "",
              f"Maps in force: {', '.join(nm(p) for p in v2b['maps_in_force'])}.", ""]
    v3f = load(tk / "v3/v3_final.json")
    if v3f:
        v3 = v3f
    if v3:
        rows = []
        for d, x in v3["a_loss"].items():
            rows.append([f"(a) loss, {DSN.get(d, d)}", x["members"], x["nonmembers"],
                         f(x["auc"]) + f" ({ci(x['auc_ci'])})", f(x["tpr_1fpr"]) + f" ({ci(x['tpr_ci'])})", x["status"]])
        for part, key in (("(b) extraction", "b_extraction"), ("(c) intervened", "c_intervened")):
            x = v3.get(key) or {}
            if x:
                j = x["jaccard"]
                rows.append([f"{part}: first 32 tokens", "", "", f"{f(x['exact32_member'])} vs "
                             f"{f(x['exact32_nonmember'])} (difference {f(x['exact32_difference'])})", "",
                             x["exact32_status"]])
                rows.append([f"{part}: event Jaccard", j["members"], j["nonmembers"],
                             f(j["auc"]) + f" ({ci(j['auc_ci'])})", "", x["jaccard_status"]])
        c = v3["controls"]
        rows.append(["Control: copying generator", "", "", f"difference {f(c['copying_generator']['exact32_difference'])}",
                     "", c["copying_generator"]["status"]])
        pr = c["private_reference_loss"]
        rows.append(["Control: private reference (loss)", pr["members"], pr["nonmembers"],
                     f(pr["auc"]) + f" ({ci(pr['auc_ci'])})", f(pr["tpr_1fpr"]) + f" ({ci(pr['tpr_ci'])})",
                     pr["status"]])
        title = (f"### V3. Privacy of prompted and intervened sampling: {v3['status']}"
                 + (f" (V3 at 500 per side: {v3['status_v3']}; the loss test at P1's sizes, V3x, in its place)"
                    if v3f else "") + f" (controls valid: {'yes' if v3['controls_valid'] else 'no'})")
        s += [title, "", table(["Test", "Members", "Non-members", "AUC or rate (95% CI)", "TPR at 1% FPR (95% CI)",
                                "Status"], rows), "",
              "Annotations: P1's rule (PASS: AUC upper bound at most 0.60 and TPR upper bound at most 0.03; FAIL: a "
              "lower bound above 0.60 or 0.05); P2's limit for the exact match (member minus non-member at most 0.01). "
              "Written also to `runs/audit/m7_MUC32_main/prompted.json`.", ""]
        gate = load(root.parent / "audit" / "m7_MUC32_main" / "report.json") or {}
        rec = gate.get("conditional_and_counterfactual_sampling")
        if rec:
            exc = ", ".join(f"{DSN.get(d, d)} {x['status'].lower()} (TPR at 1% FPR {f(x['tpr_1fpr'])}, "
                            f"{ci(x['tpr_ci'])}; {x['members']} per side, every eligible held-out stay)"
                            for d, x in (rec.get("exceptions") or {}).items())
            s += [f"The M7 gate report records conditional and counterfactual sampling as {rec['status']} "
                  f"(recorded {rec['recorded']}, the study lead's decision; the rule's verdict "
                  f"{rec.get('rule_verdict', rec['status'])}" + (f"; stated: {exc}" if exc else "") + ").", ""]
        if v3f and v3f.get("a_loss_v3"):
            rows = [[DSN.get(d, d), x["members"], x["nonmembers"], f(x["auc"]) + f" ({ci(x['auc_ci'])})",
                     f(x["tpr_1fpr"]) + f" ({ci(x['tpr_ci'])})", x["status"]] for d, x in v3f["a_loss_v3"].items()]
            s += ["V3's own loss test at up to 500 per side (replaced by V3x's above, the study lead's decision):", "",
                  table(["Dataset", "Members", "Non-members", "AUC (95% CI)", "TPR at 1% FPR (95% CI)", "Status"],
                        rows), ""]
    if v4:
        chk = v4["checks_passed"]
        s += ["### V4. Trial designer (`runs/toolkit/v4/v4.json`)", "",
              f"Checks: type I error within 0.04-0.06 (both analyses) in {chk['type1_in_0.04_0.06'][0]} of "
              f"{chk['type1_in_0.04_0.06'][1]} score cells; unadjusted simulated size within 5% of the analytic in "
              f"{chk['unadjusted_within_5pct'][0]} of {chk['unadjusted_within_5pct'][1]}.", ""]
        for rr in (0.8, 0.7):
            rows = []
            for k, o in v4["cells"].items():
                if abs(o["rr"] - rr) > 1e-9:
                    continue
                for name, sc in o["scores"].items():
                    e = {x["fraction"]: x for x in sc["enrichment"]}
                    adj = sc.get("adjustment") or {}
                    rows.append([nm(o["endpoint"]) + f" ({o['design']}{', eICU' if k.endswith('eicu') else ''})",
                                 A11_NAMES.get(name, name), f(sc.get("auroc")), f(o["population"]["event_rate"]),
                                 f(e.get(1.0, {}).get("n_total"), 0), f(e.get(0.5, {}).get("n_total"), 0),
                                 f(e.get(0.25, {}).get("n_total"), 0), f(adj.get("n_ratio"), 2),
                                 f(adj.get("type1_unadjusted")), f(adj.get("type1_adjusted"))])
            s += [f"Relative risk {rr:g} (sizes in patients, both arms; 80% power, two-sided 0.05):", "",
                  table(["Endpoint", "Score", "AUROC", "Event rate", "N, all", "N, top 50%", "N, top 25%",
                         "N adjusted / unadjusted", "Type I (unadjusted)", "Type I (adjusted)"], rows), ""]
        s += ["Annotations: L24x is F1's case-cohort sample (deaths enriched), so its event rates and sizes are not "
              "population values; the scores' comparison holds within it. L24's anchors are a natural sample. The "
              "eICU rows restrict to eICU's anchors (APACHE IVa's own data). Type I errors outside 0.04-0.06 mostly "
              "come from Monte Carlo noise (2,000 trials: SE 0.005) and from the sparse eICU subsets, where a "
              "near-perfect score strains the Wald test.", ""]
    if v5:
        rows = [[k, o["cases"], o["candidates"], sum(o["meets_objective"]), "no" if any(o["external_references"].values())
                 else "yes"] for k, o in v5["objectives"].items()]
        s += ["### V5. Teaching cases (`runs/toolkit/v5/`)", "",
              table(["Objective", "Cases", "Candidates", "Meeting the objective", "Self-contained pages"], rows), "",
              f"Novelty: {v5['novelty'] if isinstance(v5['novelty'], str) else json.dumps(v5['novelty'])}.", ""]
    return "\n".join(s)


EXT_DB = {"ricd": "RICD", "zigong": "zigongICU"}
EXT_DESIGN = {"R": "Rolling anchors (R)", "L24e": "The 24-hour landmark, case-cohort (L24e)"}
EXT_COLS = (("lgbm_pooled", "LightGBM, transferred"), ("logistic_pooled", "Logistic, transferred"),
            ("tabpfn_v3_5", "TabPFN-3.5, transferred"), ("twin_probe", "Didymos probe"),
            ("lgbm_local", "LightGBM, local (cross-fitted)"))
LEARNER = {"logistic_l1": "L1 logistic", "logistic_enet": "Elastic-net logistic", "cart": "CART",
           "random_forest": "Random forest", "extra_trees": "Extra trees", "naive_bayes": "Naive Bayes", "knn": "kNN",
           "mlp": "MLP"}
CLIN = {"sofa": "SOFA", "news": "NEWS", "mews": "MEWS", "qsofa": "qSOFA", "sirs": "SIRS", "sapsii_prob": "SAPS II",
        "oasis_prob": "OASIS", "ricd_recorded_sofa": "RICD's recorded SOFA"}


def a16a_section(root: Path) -> str:
    g, r = load(root / "ext/gate.json"), load(root / "ext/ext.json")
    if not g:
        return ""
    s = ["## 16a. Transfer to two databases never seen in training (RICD, zigongICU)", "",
         "Amendment 16a. The databases were harmonized with new adapters and encoded with MUC32_main's frozen "
         "vocabulary; prompts carry the unknown database and site tokens. The pre-run gate (`docs/paper/"
         "external_gate.md`, recorded before any future was drawn) fixed the anchors and the endpoints scored: "
         f"rolling anchors R ({g['designs']['R']['kept']} kept of {g['designs']['R']['anchors']}) and the 24-hour "
         f"landmark's case-cohort sample L24e ({g['designs']['L24e']['anchors']}), 128 futures per anchor.", ""]
    if not r:
        return "\n".join(s)
    chk = [load(root / f"ext/refit_check_{d}.json") for d in ("R", "L24")]
    if all(chk):
        s += ["The transferred models' refit (A1's recipe and seed, westc) reproduces A1's stored predictions on A1's "
              f"test anchors: LightGBM exactly (largest difference {f(max(c['by_model']['lgbm_pooled'] for c in chk), 3)}); "
              "logistic regression to the solver's numerics (AUROC within "
              f"{f(max(c['logistic_auroc_abs_diff']['max'] for c in chk), 4)} on every program, median "
              f"{f(max(c['logistic_auroc_abs_diff']['median'] for c in chk), 4)}).", ""]
    for design in [d for d in ("R", "L24e") if d in r["designs"]]:
        dbs = r["designs"][design]
        rows, crows, rrows = [], [], []
        for d, progs in dbs.items():
            for p, x in progs.items():
                pr = x["predictors"]
                t = pr["twin"]
                cells = [nm(p), EXT_DB[d], t["n"], t["positives"], x["estimate"],
                         f(t.get("auroc")) + f" ({ci(t.get('auroc_ci'))})", f(t.get("observed_over_expected"), 2),
                         f(x.get("internal_auroc"))]
                for m, _ in EXT_COLS:
                    c = pr.get(m)
                    cells.append(f(c.get("auroc")) if c else "-")
                rows.append(cells)
                dcells = [nm(p), EXT_DB[d]]
                for m in ("lgbm_pooled", "lgbm_local", "tabpfn_v3_5"):
                    c = pr.get(m)
                    dcells.append(f(c.get("ref_minus")) + f" ({ci(c.get('ref_minus_ci'))})" if c and "ref_minus" in c
                                  else "-")
                sv = {k: v for k, v in (x.get("sampled_vs_scores") or {}).items() if k != "twin_sampled" and v.get("auroc")}
                if sv:
                    best = max(sv, key=lambda k: sv[k]["auroc"])
                    dcells.append(f"{CLIN.get(best, best)}: {f(sv[best]['auroc'])}; sampled minus it "
                                  f"{f(sv[best].get('ref_minus'))} ({ci(sv[best].get('ref_minus_ci'))})")
                else:
                    dcells.append("-")
                crows.append(dcells)
                rc = x.get("recalibrated", {})
                rrows.append([nm(p), EXT_DB[d], f(t.get("observed_over_expected"), 2),
                              f(rc.get("twin", {}).get("observed_over_expected"), 2), f(rc.get("twin", {}).get("auroc")),
                              f(pr.get("lgbm_pooled", {}).get("observed_over_expected"), 2),
                              f(rc.get("lgbm_pooled", {}).get("observed_over_expected"), 2)])
        s += [f"### {EXT_DESIGN.get(design, design)}", "",
              table(["Endpoint", "Database", "Anchors", "Positives", "Didymos estimate", "Didymos AUROC (95% CI)",
                     "Didymos O/E", "Didymos AUROC, internal (follow-up)"] + [n for _, n in EXT_COLS], rows), "",
              "Paired differences (Didymos minus each; 95% CI) and the best clinical score:", "",
              table(["Endpoint", "Database", "minus LightGBM, transferred", "minus LightGBM, local",
                     "minus TabPFN-3.5", "Best clinical score (AUROC; the sampled estimate minus it)"], crows), "",
              "Calibration before and after a logistic recalibration within the database (cross-fitted by patient, "
              "5 folds):", "",
              table(["Endpoint", "Database", "Didymos O/E", "after recalibration", "Didymos AUROC after",
                     "LightGBM (transferred) O/E", "after recalibration"], rrows), ""]
    s += ["Annotations: weighted metrics (R: the design's inclusion weights; L24e: the case-cohort weights); intervals "
          "from 1,000 resamples of stays, paired differences from the same resamples. Didymos's estimate is the hazard "
          "estimate (E2) for death within 24 h, new vasopressor and ICU death, the sampled estimate otherwise; the "
          "internal AUROC is the same estimator on the follow-up anchors of the seven databases (R128, L24x), shown "
          "for reference. Transferred models were trained on the seven databases with A1's recipe (a new database's "
          "one-hot columns are all zero); the local LightGBM was trained on the database's own stays, cross-fitted by "
          "patient. Endpoints not listed are not defined in that database (the gate's rule).", ""]
    return "\n".join(s)


def a16b_section(root: Path) -> str:
    sub = (load(root / "f1/subsets/subsets.json") or {}).get("L48x")
    if not sub and not load(root / "f1/L48x/a1.json"):
        return ""
    s = ["## 16b. The 48-hour landmark with the follow-up's settings (L48x)", ""]
    if sub:
        s += [f"Anchors: {sub['anchors']} of A1's test_L48 ({sub['cases']} in-hospital deaths and {sub['controls']} "
              f"controls weighted {f(sub['control_weight'], 2)}: a case-cohort sample, seed 16); left out as continued "
              f"context windows: {sub.get('excluded_continued_window_or_label', 0)}. 128 futures per anchor at four "
              "times A1's caps.", ""]
    one = (("L48x", "f1/L48x"),)
    body = [f1_section(root, one, "### Rollout count to 128 (as F1)"), f2_section(root, one, "### Estimators (as F2)"),
            f3_section(root, one, "### Mortality (as F3)")]
    s += [b for b in body if b.count("\n") > 2]
    cmp_ = (load(root / "a16/comparators_L48x.json") or {}).get("subsets", {}).get("L48x")
    if cmp_:
        rows = []
        learners = ("logistic_l1", "logistic_enet", "cart", "random_forest", "extra_trees", "naive_bayes", "knn", "mlp")
        for p, x in cmp_.items():
            m = x["models"]

            def au(k):
                return f(m[k]["auroc"]) + (f" ({f(m[k].get('ref_minus'))})" if "ref_minus" in m[k] else "") \
                    if k in m else "-"
            best_l = max((k for k in learners if k in m), key=lambda k: m[k]["auroc"], default=None)
            best_c = max((k for k in m if k in CLIN), key=lambda k: m[k]["auroc"], default=None)
            rows.append([nm(p), x["anchors"], x["positives"], f(m["twin_128"]["auroc"]), f(x.get("e2_auroc")),
                         au("lgbm_pooled"), au("tabpfn_v3_5"), au("twin_probe"),
                         f"{LEARNER.get(best_l, best_l)}: {au(best_l)}" if best_l else "-",
                         f"{CLIN.get(best_c, best_c)}: {au(best_c)}" if best_c else "-"])
        s += ["### Every comparator on L48x (`runs/paper/a16/comparators_L48x.json`)", "",
              table(["Endpoint", "Anchors", "Positives", "Didymos, 128 (sampled)", "Didymos E2", "LightGBM",
                     "TabPFN-3.5", "Didymos probe", "Best of the 8 learners", "Best clinical score"], rows),
              "", "AUROC (the sampled 128-future estimate minus it, paired). TabPFN-3.5 and the eight learners of "
                  "amendment 10 for death within 24 h were fitted for this subset with their recipes; the other cells "
                  "are A1's stored predictions, reindexed.", ""]
    s += ["Annotations: as F1-F3 at L24x; the comparators are A1's stored predictions on the same anchors unless "
          "stated.", ""]
    return "\n".join(s)


def a16c_section(root: Path) -> str:
    r, old = load(root / "f4r/c2b.json"), load(root / "f4/c2b.json")
    if not r:
        return ""
    s = ["## 16c. Attention knockout against the generator (`runs/paper/f4r/c2b.json`)", "",
         "F4's 160 anchors and 2,698 prompts drawn again (48 futures each, seed 16, F4's caps); every group's effect "
         "recomputed as F4 did. Knockout (amendment 9's mask and score) on all 160 anchors.", ""]
    rows, prow = [], []
    for p, x in r["programs"].items():
        ag = x["agreement"]
        o = (old or {}).get("programs", {}).get(p, {}).get("agreement", {}).get("edit", {})
        rows.append([nm(p), x["anchors"], f(x["reliability_sb_median"], 2)] +
                    [f(ag[m]["spearman_median"], 2) + " / " + f(ag[m]["top3_overlap_mean"], 2) if m in ag else "-"
                     for m in ("edit", "knockout")] +
                    [f(ag[m]["sign_agreement"], 2) + f" (n={ag[m]['signed_groups']})" if m in ag else "-"
                     for m in ("edit", "knockout")] +
                    [f(o.get("spearman_median"), 2) if o else "-"])
        kv = x.get("knockout_vs_edit")
        if kv:
            prow.append([nm(p), kv["anchors"], f(kv["knockout_median"], 2), f(kv["edit_median"], 2),
                         f(kv["difference_median"], 2)])
    s += [table(["Endpoint", "Anchors", "Ceiling (Spearman-Brown)", "Editing: rho / top-3", "Knockout: rho / top-3",
                 "Sign agreement, editing", "Knockout", "F4's editing rho (first draw)"], rows), ""]
    if prow:
        s += ["Per anchor on the groups both score (median rho with the redrawn effects):", "",
              table(["Endpoint", "Anchors", "Knockout", "Editing", "Knockout minus editing"], prow), ""]
    s += ["Annotations: agreement is bounded by the reliability ceiling (split halves of the 48 futures, "
          "Spearman-Brown corrected). The statistics (family 20) are in `docs/paper/tables/manuscript/stats_family20.csv`.",
          ""]
    return "\n".join(s)


A17_ARMS = (("Z0", "Z0: 16a"), ("Z1", "Z1: back-off"), ("F1", "F1: adapted"), ("F1S", "F1+S: stacked"),
            ("lgbm_label", "LightGBM, LABEL"), ("lgbm_ftlabel", "LightGBM, FT+LABEL"), ("tabpfn_v3_5", "TabPFN-3.5"))


def a17_section(root: Path) -> str:
    r = load(root / "ext17/ext17.json")
    sp = load(root / "ext17/split.json")
    if not sp:
        return ""
    s = ["## 17. Adapting Didymos to RICD and zigongICU", "",
         "Amendment 17 (2026-10-12, the study lead's request; fixed before any adapted model was trained or any new "
         "future drawn). Each database's patients split once (seed 17): TEST 50% (locked; amendment 16a's anchors "
         "there), FT 35% (unlabelled continued training) and LABEL 15% (the few-label methods). Labels are amendment "
         "16a's for every arm.", ""]
    rows = [[EXT_DB[d], sp["patients"][d].get(k, 0), sp["stays"][d].get(k, 0)] for d in EXT_DB
            for k in ("test", "ft", "label")]
    s += [table(["Database", "Patients", "Stays"], [[f"{x[0]}, {k}", x[1], x[2]] for x, k in
                                                   zip(rows, ["TEST", "FT", "LABEL"] * 2)]), ""]
    if not r:
        return "\n".join(s)
    ro = r.get("readouts", {})
    oov = ro.get("oov") or {}
    if oov.get("ext_r4") and oov.get("ext_r4b"):
        doms = sorted({k for d in EXT_DB for k in oov["ext_r4"][d]})
        orow = [[EXT_DB[d], dom, f(oov["ext_r4"][d][dom]["share"] * 100, 1),
                 f(oov["ext_r4b"][d].get(dom, {}).get("share", 0) * 100, 1)]
                for d in EXT_DB for dom in doms if dom in oov["ext_r4"][d] and oov["ext_r4"][d][dom]["share"] > 0]
        s += ["### The improved encoding (Z1's back-off)", "",
              table(["Database", "Domain", "Events to OOV, 16a (%)", "after the back-off (%)"], orow), ""]
    gr = ro.get("grid")
    if gr:
        before = gr["before"]
        seven = [k for k in before["by_dataset"] if k not in EXT_DB]
        grow = [[x["run"], f"{x['lr']:g}", f"{x['ft_passes']:g}", x["steps"], f(x["ft_nll"], 4),
                 f(np.mean([x["forgetting"][k] for k in seven]), 4), f(max(x["forgetting"][k] for k in seven), 4),
                 "chosen" if x["run"] == gr["chosen"] else ""] for x in gr["cells"].values()]
        s += ["### F1: the grid (next-token loss, nats per token)", "",
              f"MUC32_main before adaptation: held-out FT loss {f(before['ft_nll'], 4)}.", "",
              table(["Run", "Learning rate", "Passes over FT", "Steps", "Held-out FT loss",
                     "Seven databases' tuning loss, change (mean)", "change (largest)", ""], grow), ""]
    audit = root.parent / "audit" / f"m7_{gr['chosen']}" if gr else None
    p1a, p1b = (load(audit / f"{n}.json") if audit else None for n in ("p1_cap3000", "p1"))
    rep, ex = (load(audit / f"{n}.json") if audit else None for n in ("report", "owner_exemption"))
    if p1a and p1b and rep:
        def tp(g):
            return f(g["tpr_at_1pct_fpr"][0], 4) + f" ({ci(g['tpr_at_1pct_fpr'][1:], 4)})"
        prow = [[DSN.get(d, d) if d != "pooled" else "Pooled", p1a["groups"][d]["members"], tp(p1a["groups"][d]),
                 p1b["groups"][d]["members"], tp(p1b["groups"][d]), f(p1b["groups"][d]["auc"][0])]
                for d in p1b["groups"] if d in p1a["groups"]]
        s += ["### F1: the M7 gate", "", "The M7 gate of the chosen run, P1 (membership by loss on the seven databases) at the registered size and "
              "at amendment 17c's larger size, TPR at 1% FPR against the 0.03 limit:", "",
              table(["Database", "Per side", "TPR (95% CI)", "Per side, 17c", "TPR, 17c (95% CI)", "AUC, 17c"], prow), "",
              f"P2-P5 as run: {', '.join(f'{k} {rep[k]}' for k in ('P2', 'P3', 'P4', 'P5'))}; the gate "
              f"{rep['gate']} after 17c (one group's upper bound above the limit)" +
              ("; F1's futures were drawn under the study lead's exemption (amendment 17d: these futures only, deleted "
               "after scoring; no twins or other synthetic data from this run)." if ex else "."), ""]
    mem = ro.get("membership") or {}
    if mem:
        mrow = [[who, EXT_DB.get(d, d), g["members"], g["nonmembers"], f(g["auc"][0]) + f" ({ci(g['auc'][1:])})",
                 f(g["mean_nll_members"], 4), f(g["mean_nll_nonmembers"], 4)]
                for who, p1 in mem.items() if p1 for d, g in p1["groups"].items()]
        s += ["### F1: membership of its own FT stays", "",
              "Membership of the FT stays against the TEST stays (loss attack, P1's method):", "",
              table(["Model", "Database", "FT stays", "TEST stays", "AUC (95% CI)", "Mean loss, FT", "Mean loss, TEST"],
                    mrow), ""]
    for design in [d for d in ("R", "L24e") if d in r["designs"]]:
        rows, drows, crows = [], [], []
        for d, progs in r["designs"][design].items():
            for p, x in progs.items():
                pr = x["predictors"]
                tag = " (descriptive)" if x["descriptive_only"] else ""
                rows.append([nm(p) + tag, EXT_DB[d], x["anchors"], x["positives"], x["label_positives"]] +
                            [f(pr[m]["auroc"]) + (f" ({ci(pr[m].get('auroc_ci'))})" if m == "F1" else "")
                             if m in pr else "-" for m, _ in A17_ARMS])
                av = x["arms_vs_z0"]
                drows.append([nm(p) + tag, EXT_DB[d]] +
                             [f(av[a].get("ref_minus")) + f" ({ci(av[a].get('ref_minus_ci'))})"
                              if av.get(a) and "ref_minus" in av[a] else "-" for a in ("Z1", "F1", "F1R")] +
                             [f(pr[c].get("ref_minus")) + f" ({ci(pr[c].get('ref_minus_ci'))})"
                              if c in pr and "ref_minus" in pr[c] else "-" for c in ("lgbm_ftlabel", "tabpfn_v3_5")])
                cal = x.get("calibration", {})
                rc = (x.get("recalibration") or {}).get("F1") or {}
                crows.append([nm(p) + tag, EXT_DB[d]] + [f(cal.get(c, {}).get("observed_over_expected"), 2)
                                                         for c in ("Z0", "Z1", "F1", "Z1R", "F1R", "F1S")] +
                             ["intercept" if rc.get("intercept_only") else ("intercept, slope" if rc else "-")])
        s += [f"### {EXT_DESIGN.get(design, design)}: the TEST anchors", "",
              table(["Endpoint", "Database", "Anchors", "Positives", "LABEL positives"] + [n for _, n in A17_ARMS],
                    rows), "",
              "Paired differences in AUROC (95% CI):", "",
              table(["Endpoint", "Database", "Z1 minus Z0", "F1 minus Z0", "F1+R minus Z0",
                     "F1+R minus LightGBM, FT+LABEL", "F1+R minus TabPFN-3.5"], drows), "",
              "Calibration (O/E) before and after recalibration on the LABEL anchors:", "",
              table(["Endpoint", "Database", "Z0", "Z1", "F1", "Z1+R", "F1+R", "F1+S", "F1's map"], crows), ""]
    cur = r.get("curves") or {}
    lrows = []
    for design, dbs in cur.items():
        for d, progs in dbs.items():
            for p, c in progs.items():
                pts = c["points"]
                lrows.append([EXT_DESIGN.get(design, design).split(" (")[-1].rstrip(")"), nm(p), EXT_DB[d],
                              f(c.get("f1_auroc"))] +
                             [f(pts[n]["lgbm_auroc"]) if n in pts else "-" for n in ("25", "50", "100", "200")] +
                             [f(pts[str(c["label_stays"])]["lgbm_auroc"]) + f" (n={c['label_stays']})",
                              c["status"] + (f", {f(c['n_match'], 0)}" if c.get("n_match") else "")])
    if lrows:
        s += ["### Label efficiency (LightGBM on n LABEL stays, median AUROC of 20 draws, against F1+R)", "",
              table(["Design", "Endpoint", "Database", "F1+R", "n=25", "n=50", "n=100", "n=200", "all LABEL",
                     "LightGBM reaches F1+R at n"], lrows), ""]
    rk = r.get("ranks") or {}
    if rk.get("all"):
        s += ["### Ranks over the TEST pairs (family 24)", ""]
    for key, title in (("all", "every predictor"),
                       ("without_local", "without the three local LightGBMs; F1+S keeps its LABEL-trained part")):
        if rk.get(key):
            sm = rk[key]
            s += [f"Mean ranks by AUROC over the {sm['tasks']} TEST pairs with at least 10 positives ({title}):", "",
                  table(["Predictor", "Mean rank (95% CI)", "Firsts", "Mean AUROC"],
                        [[x["predictor"], f(x["mean_rank"], 2) + f" ({ci(x.get('mean_rank_ci'), 2)})", x["firsts"],
                          f(x["mean_auroc"])] for x in sm["predictors"]]), "",
                  f"Friedman P {f(sm['friedman']['p'], 4)}. F1 against each (Wilcoxon over the pairs, Holm):", "",
                  table(["Comparator", "F1 higher / lower / tied", "Median AUROC difference (F1 minus)", "P (Holm)"],
                        [[c["comparator"], f"{c['wins']} / {c['losses']} / {c['ties']}", f(c["median_difference"]),
                          f(c.get("p_holm"), 3)] for c in sm.get("comparisons", [])]), ""]
    al = ro.get("alignment") or {}
    if al:
        s += ["Labels under the improved encoding against 16a's (paper_a1's alignment control, lowest label agreement "
              "over programs): " + "; ".join(f"{k} {f(v.get('min'), 3)}" for k, v in al.items()) + ".", ""]
    s += ["Annotations: weighted metrics on the TEST anchors (R: 16a's inclusion weights; L24e: the case-cohort "
          "weights recomputed within TEST); intervals from 1,000 resamples of stays, paired differences from the same "
          "resamples. Each arm's estimate is the hazard estimate (E2) on 16a's hazard endpoints, the 128-future share "
          "otherwise. Z0 is 16a's prediction; Z1 the same model on the back-off encoding; F1 the adapted model. R: "
          "logistic recalibration on the LABEL anchors (the intercept alone below 10 LABEL positives), which keeps the "
          "ranks; S: stacking with the LABEL-only LightGBM. Endpoints with fewer than 10 TEST positives are "
          "descriptive. The statistics (families 21-24) are in "
          "`docs/paper/tables/manuscript/stats_family21-23.csv` and `stats_family24.csv`.", ""]
    return "\n".join(s)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    head = ["# Materials: results of the follow-up experiments (amendments 7-11, 16 and 17)", "",
            "Experiments F1-F9 of `docs/p1_paper_protocol.md` amendment 7 (2026-10-09), fixed before they were run; "
            "amendment 9's exploratory diagnosis of attention (2026-10-10, after F4); amendment 10's comparators "
            "(2026-10-10: clinical scores, linear, tree and other learners, fixed before any was computed); and "
            "amendment 11's validation of the application toolkit (2026-10-10, fixed before it ran). The "
            "pre-registered results are in `docs/paper/results.md` and unchanged. Every table is computed from the "
            "aggregate JSON files under `runs/paper/` and `runs/toolkit/` on the host and rendered by "
            "`tools/paper_results_followup.py`. Held-out split throughout, except amendment 16a, where every stay of the "
            "two external databases is a test stay. Amendment 16 (2026-10-11, the study lead's request; fixed before "
            "it ran): 16a transfer to RICD and zigongICU, 16b the 48-hour landmark with the follow-up's settings, 16c "
            "attention knockout against the generator. Amendment 17 (2026-10-12, the study lead's request; fixed "
            "before any adapted model was trained): adapting Didymos to RICD and zigongICU on a locked patient split.",
            ""]
    body = [x for x in (f1_section(root), f2_section(root), a10_section(root), f3_section(root), f4_section(root),
                        a9_section(root), f5_section(root), f6_section(root), f7_section(root), f8_section(root),
                        f9_section(root), a11_section(root), a16a_section(root), a16b_section(root),
                        a16c_section(root), a17_section(root)) if x]
    text = export_tables("\n".join(head + body), Path(a.tables) if a.tables else None)
    Path(a.out).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {a.out}" + (f" and its tables to {a.tables}" if a.tables else ""))


if __name__ == "__main__":
    main()
