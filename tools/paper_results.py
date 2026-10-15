"""Paper P's results materials (``docs/paper/results.md``): every experiment's aggregates rendered as tables with
their source files and annotations (``docs/p1_paper_protocol.md``). Reads the aggregate JSON tree of ``runs/paper``
(and ``runs/eval/<run>/calibration_masked.json`` for the teacher-forced rates); writes Markdown and, with
``--tables``, every table as CSV (``export_tables``).

    python tools/paper_results.py --root runs/paper --eval runs/eval --out docs/paper/results.md \\
        --tables docs/paper/tables/results
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import PRIMARY  # noqa: E402

DESIGNS = {"L24": "24-hour landmark", "L48": "48-hour landmark", "R": "rolling times (enriched, weighted)"}
NAMES = {"icu_death@-1": "ICU death", "hospital_death@0": "In-hospital death", "remaining_los_gt_7d": "ICU stay > 7 days",
         "decompensation@1440": "Decompensation, 24 h", "imv_onset@1440": "New IMV, 24 h",
         "vasopressor_onset@1440": "New vasopressor, 24 h", "rrt_onset@2880": "New RRT, 48 h",
         "aki_2@2880": "AKI stage >= 2, 48 h", "sepsis_ase@1440": "Sepsis (ASE), 24 h",
         "hypotension@240": "Hypotension, 4 h", "hyperkalemia_top@1440": "Hyperkalaemia (top bin), 24 h",
         "thrombocytopenia@2880": "Thrombocytopenia, 48 h"}
DS = ("mimic", "eicu", "hirid", "aumc", "sicdb", "nwicu", "inspire")
DSN = {"mimic": "MIMIC-IV", "eicu": "eICU", "hirid": "HiRID", "aumc": "AmsterdamUMCdb", "sicdb": "SICdb", "nwicu": "NWICU",
       "inspire": "INSPIRE"}


def f(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{d}f}"


def ci(c, d=3):
    return f"{f(c[0], d)} to {f(c[1], d)}" if c and c[0] is not None else "-"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


TABLES: list = []                                    # every table rendered, for export_tables


def table(head: list, rows: list) -> str:
    TABLES.append(([str(c) for c in head], [[str(c) for c in r] for r in rows]))

    def cells(xs):                                    # a literal | (e.g. in a group name) would split the cell
        return "| " + " | ".join(str(c).replace("|", "\\|") for c in xs) + " |"

    out = [f"<!-- table {len(TABLES) - 1} -->", cells(head), "|" + "---|" * len(head)]
    out += [cells(r) for r in rows]
    return "\n".join(out)


def slug(s: str, n: int) -> str:
    s = re.sub(r"\(\s*\)", "", re.sub(r"`[^`]*`", "", s))
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:n].rstrip("_")


def export_tables(text: str, out_dir: Path | None) -> str:
    """The rendered document without its table markers; with ``out_dir``, also every table as CSV, the cells as
    rendered (``<section>_<nn>_<context>.csv``: the section's code, the table's order in it, and its subsection and
    lead-in line, else the section's title), and ``index.csv`` (file, section, subsection, the line before the table,
    rows). The directory's earlier CSV files are replaced."""
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.csv"):
            old.unlink()
    h2 = h3 = caption = ""
    count: dict = {}
    index, keep = [], []
    for line in text.split("\n"):
        m = re.fullmatch(r"<!-- table (\d+) -->", line)
        if not m:
            keep.append(line)
            if line.startswith("## "):
                h2, h3, caption = line[3:], "", ""
            elif line.startswith("### "):
                h3, caption = line[4:], ""
            elif line and not line.startswith("|"):
                caption = line
            continue
        if out_dir is None:
            continue
        head, rows = TABLES[int(m.group(1))]
        code = slug(h2.split(".")[0], 20) or "doc"
        count[code] = count.get(code, 0) + 1
        lead = caption if caption.endswith(":") else ""             # a lead-in line, not a paragraph of numbers
        context = slug(f"{h3} {lead}", 48) or slug(h2.split(".", 1)[-1], 48) or "table"
        name = f"{code}_{count[code]:02d}_{context}.csv"
        with open(out_dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")          # LF on every platform, as the repository's files
            w.writerow(head)
            w.writerows(rows)
        index.append([name, h2, h3, caption, len(rows)])
        caption = ""
    if out_dir is not None:
        with open(out_dir / "index.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["file", "section", "subsection", "line_before", "rows"])
            w.writerows(index)
    return "\n".join(keep)


def a1_section(root: Path) -> str:
    s = ["## A1. Prediction by sampling (held-out split)", ""]
    for d, title in DESIGNS.items():
        a1 = load(root / f"a1/test_{d}/a1.json")
        cmp_ = load(root / f"a1/test_{d}/compare.json")
        if not a1:
            continue
        s += [f"### {title} (`runs/paper/a1/test_{d}`)", "",
              f"{a1['anchors']} anchors x {a1['rollouts']} rollouts; {f(a1['censored_rollouts'] * 100, 1)}% of rollouts "
              f"end at the token cap; alignment control (bin-edge programs) {f(a1['alignment_control_min'])}.", ""]
        rows = []
        for k in PRIMARY[d]:
            tw = a1["programs"].get(k, {}).get("pooled", {})
            c = (cmp_ or {}).get("a1", {}).get(k, {})
            rows.append([NAMES.get(k, k), tw.get("n"), tw.get("positives"), f(tw.get("auroc")), ci(tw.get("auroc_ci")),
                         f(c.get("twin")), f(c.get("tabpfn_v3_5")), f(c.get("tabpfn_v2")), f(c.get("lgbm_pooled")),
                         f(c.get("lgbm_per_dataset")), f(c.get("logistic_pooled")), f(c.get("twin_probe")),
                         f(c.get("twin_minus_lgbm")), ci(c.get("twin_minus_lgbm_ci"))])
        s += [table(["Endpoint", "Anchors", "Positives", "Twin AUROC", "95% CI", "Twin (common anchors)",
                     "TabPFN-3.5", "TabPFN v2", "LightGBM pooled", "LightGBM per dataset", "Logistic", "Twin probe",
                     "Twin - LightGBM", "95% CI (paired)"], rows), ""]
        au = [v["pooled"]["auroc"] for v in a1["programs"].values() if "pooled" in v and v["pooled"].get("auroc")]
        if au:
            s += [f"All {len(au)} binary catalog programs of this design: Twin AUROC median {f(float(np.median(au)))}, "
                  f"10th-90th percentile {f(float(np.percentile(au, 10)))}-{f(float(np.percentile(au, 90)))}.", ""]
        rows = []
        for k in PRIMARY[d]:
            x = a1["programs"].get(k, {})
            rows.append([NAMES.get(k, k)] + [f(x.get("datasets", {}).get(ds, {}).get("auroc")) for ds in DS])
        s += ["Twin AUROC by dataset:", "", table(["Endpoint"] + [DSN[x] for x in DS], rows), ""]
        rows = []
        for k in PRIMARY[d]:
            cur = a1["programs"].get(k, {}).get("rollout_curve")
            if cur:
                rows.append([NAMES.get(k, k)] + [f(cur.get(str(n), {}).get("auroc")) for n in (1, 2, 4, 8, 16, 32)])
        if rows:
            s += ["AUROC against the number of rollouts:", "",
                  table(["Endpoint", "1", "2", "4", "8", "16", "32"], rows), ""]
        reg = [(k, v) for k, v in a1["programs"].items() if "mae_median" in v or "c_index_48h_share" in v]
        if reg:
            rows = [[k, v.get("n"), f(v.get("mae_median")), f(v.get("mae_baseline_mean")), f(v.get("crps")),
                     f((v.get("coverage") or {}).get("0.8")), f(v.get("c_index_48h_share"))] for k, v in reg]
            s += ["Regression and time-to-event programs (Twin ensembles):", "",
                  table(["Program", "Anchors", "MAE (median)", "MAE (predict the mean)", "CRPS", "80% coverage",
                         "C-index (48 h share)"], rows), ""]
        bl = load(root / f"a1/test_{d}/baselines.json")
        if bl:
            rows = []
            for k in PRIMARY[d]:
                eff = bl["programs"].get(k, {}).get("label_efficiency", {})
                tw = a1["programs"].get(k, {}).get("datasets", {})
                for ds in DS:
                    if ds in eff:
                        e = eff[ds]
                        rows.append([NAMES.get(k, k), DSN[ds], f(tw.get(ds, {}).get("auroc"))] +
                                    [f(e.get(n)) for n in ("100", "300", "1000", "3000")] + [f(e[max(e, key=int)])])
            if rows:
                s += ["Label efficiency: LightGBM per dataset trained on n train stays, against the Twin (zero-shot):", "",
                      table(["Endpoint", "Dataset", "Twin", "n=100", "300", "1,000", "3,000", "all (up to 5,000)"],
                            rows), ""]
    s += ["Annotations: the Twin is zero-shot (MUC32_main's 32 rollouts per anchor, no real labels); every other model is "
          "trained on real train stays (TabPFN on a stratified sample, the probe on the model's frozen states). The "
          "rolling design is enriched (half its anchors precede an event) and its metrics are inclusion-weighted. "
          "'Common anchors' are those every model scored; the paired difference resamples stays within datasets. SOFA "
          "(first day) is in `baselines.json` for the death endpoints.", ""]
    return "\n".join(s)


def a2_section(root: Path, ev: Path) -> str:
    s = ["## A2. Calibration and uncertainty", ""]
    for d, title in DESIGNS.items():
        c = load(root / f"a1/test_{d}/compare.json")
        if not c:
            continue
        rows = []
        for k in PRIMARY[d]:
            x = c["a2"].get(k)
            if not x:
                continue
            raw, rec = x["raw"], x.get("recalibrated", {})
            days = x.get("by_icu_day", {})
            rows.append([NAMES.get(k, k), f(raw.get("observed_over_expected"), 2), f(raw.get("calibration_slope"), 2),
                         f(raw.get("ece")), f(raw.get("brier")), f(rec.get("observed_over_expected"), 2),
                         f(rec.get("ece")), ", ".join(f"d{k2}: {f(v['observed_over_expected'], 2)}"
                                                      for k2, v in days.items())])
        s += [f"### {title}", "", table(["Endpoint", "O/E", "Slope", "ECE", "Brier", "O/E recalibrated",
                                         "ECE recalibrated", "O/E by ICU day of the anchor"], rows), ""]
    tf = load(ev / "MUC32_main/calibration_masked.json")
    a1r = load(root / "a1/test_R/a1.json")
    if tf:
        rows = []
        for ev_, prog in (("imv", "imv_onset@1440"), ("vasopressor", "vasopressor_onset@1440"),
                          ("rrt", "rrt_onset@2880"), ("icu_death", "decompensation@1440")):
            o = sum(tf[ds][ev_]["after_24h"]["observed"] for ds in tf)
            e = sum(tf[ds][ev_]["after_24h"]["expected"] for ds in tf)
            ro = (a1r or {}).get("programs", {}).get(prog, {}).get("pooled", {}).get("observed_over_expected")
            rows.append([ev_, f(o / e if e else None, 2), NAMES[prog], f(ro, 2)])
        s += ["### Teacher-forced against rollout calibration (exposure bias)", "",
              table(["Event", "Teacher-forced O/E after 24 h (pooled, masked)", "Rollout endpoint", "Rollout O/E (R)"],
                    rows), "",
              "The teacher-forced O/E sums the model's next-event probability of each event over every eligible "
              "position after the 24-h landmark (whole remaining stay); the rollout O/E is over the endpoint's horizon "
              "from rolling anchors, so the death row compares related, not identical, quantities.", ""]
    for d in ("L24", "L48"):
        a1 = load(root / f"a1/test_{d}/a1.json")
        disp = (a1 or {}).get("remaining_los_dispersion")
        if disp:
            s += [f"Remaining ICU stay ({d}) dispersion: " + ", ".join(f"{k} {f(v) if isinstance(v, float) else v}"
                                                                    for k, v in disp.items()), ""]
    s += ["Annotations: O/E above 1 means the Twin under-predicts. Recalibration is isotonic per endpoint, fitted on the "
          "tuning-split dry run of the same design (A2 of the protocol) and applied to the held-out anchors; it is "
          "reported beside the raw Twin, never instead of it.", ""]
    return "\n".join(s)


def a3_section(root: Path) -> str:
    a3 = load(root / "a3/a3.json")
    if not a3:
        return ""
    s = ["## A3. Privacy: the gate and the fit-privacy frontier (`runs/paper/a3/a3.json`)", ""]
    ad = a3["checkpoints"][a3["adopted"]]
    rows = [[g, f(x["auc"]), ci(x["auc_ci"]), f((x.get("tpr_at_1pct_fpr") or [None])[0]), x["status"]]
            for g, x in ad["p1"].items()]
    s += [f"P1 membership inference of {a3['adopted']} (gate {ad.get('gate')}):", "",
          table(["Group", "AUC", "95% CI", "TPR at 1% FPR", "Status"], rows), ""]
    rows = []
    for run, c in sorted(a3["checkpoints"].items(), key=lambda z: z[1].get("tuning_nll") or 9):
        rows.append([run, c["optimizer"], f(c.get("tuning_nll"), 4), f(c["p1"].get("pooled", {}).get("auc")),
                     f(max((x["auc"] for g, x in c["p1"].items() if g != "pooled"), default=None)), c["p1_status"]])
    s += ["Checkpoints (the frontier):", "",
          table(["Checkpoint", "Optimizer", "Tuning NLL", "P1 AUC pooled", "P1 AUC, worst dataset", "P1 status"], rows),
          "", "Rank correlation across checkpoints, per dataset:", "",
          table(["Dataset", "AUC vs member/non-member NLL gap", "AUC vs held-out NLL"],
                [[g, f(x["spearman_auc_vs_gap"], 2), f(x["spearman_auc_vs_heldout_nll"], 2)]
                 for g, x in a3["frontier"].items()]), "",
          "Annotations: AdamW and Muon checkpoints lie on one curve: the membership signal follows the fit reached, "
          "not the optimizer; MUC32_main's per-dataset passes were set on this frontier (AUC 0.549-0.555 in every "
          "dataset). The audit's samples are unconstrained by protocol (worst case).", ""]
    return "\n".join(s)


def a4_section(root: Path) -> str:
    s = ["## A4. Utility of synthetic data (`baselines.json` per design)", ""]
    for d, title in DESIGNS.items():
        bl = load(root / f"a1/test_{d}/baselines.json")
        if not bl:
            continue
        rows, aug = [], []
        for k in PRIMARY[d]:
            a4 = bl["programs"].get(k, {}).get("a4")
            if not a4:
                continue
            p = a4.get("pooled", {})
            rows.append([NAMES.get(k, k), p.get("rows"), f(p.get("tstr")), f(p.get("trtr")),
                         f((p.get("tstr") - 0.5) / (p.get("trtr") - 0.5), 2) if p.get("tstr") and p.get("trtr") else "-"]
                        + [f(a4.get("datasets", {}).get(ds, {}).get("tstr")) + " / "
                           + f(a4.get("datasets", {}).get(ds, {}).get("trtr")) for ds in DS])
            for ds, x in a4.get("augmentation", {}).items():
                aug.append([NAMES.get(k, k), DSN[ds], f(x.get("real_300")), f(x.get("real_300_plus_twins_300")),
                            f(x.get("real_300_plus_twins_900")), f(x.get("real_all"))])
        if rows:
            s += [f"### {title}", "", table(["Endpoint", "Rows", "TSTR", "TRTR", "Ratio of gains"] +
                                            [f"{DSN[x]} TSTR/TRTR" for x in DS], rows), ""]
        if aug:
            s += ["Small-site augmentation (AUROC on that dataset's held-out anchors):", "",
                  table(["Endpoint", "Dataset", "300 real", "300 real + 300 twins", "300 real + 900 twins",
                         "all real"], aug), ""]
    s += ["Annotations: LightGBM on the A1 features; TSTR trained on the bank's twins, TRTR on as many real train rows; "
          "ratio of gains = (TSTR - 0.5) / (TRTR - 0.5). '-' in an augmentation arm: the bank has fewer eligible twins of "
          "that dataset than the arm needs. The cross-dataset matrices are in `baselines.json`.", ""]
    return "\n".join(s)


def a5_section(root: Path) -> str:
    s = ["## A5. Subgroups (`compare.json` per design)", ""]
    for d, title in DESIGNS.items():
        c = load(root / f"a1/test_{d}/compare.json")
        if not c:
            continue
        rows = []
        for k, attrs in c["a5"].items():
            for attr, vals in attrs.items():
                for val, m in vals.items():
                    tw, lg = m.get("twin", {}), m.get("lgbm_pooled", {})
                    if tw.get("scored") is False and lg.get("scored") is False:
                        continue
                    rows.append([NAMES.get(k, k), attr, val, tw.get("n"), tw.get("positives"), f(tw.get("auroc")),
                                 f(tw.get("observed_over_expected"), 2), f(lg.get("auroc"))])
        if rows:
            s += [f"### {title}", "", table(["Endpoint", "Attribute", "Value", "Anchors", "Positives", "Twin AUROC",
                                             "Twin O/E", "LightGBM AUROC"], rows), ""]
    s += ["Annotations: subgroups with fewer than 30 positives or negatives are not scored; race is recorded in MIMIC, "
          "eICU and NWICU only. Age bands: 0 = 18-44, 1 = 45-64, 2 = 65-79, 3 = 80+.", ""]
    return "\n".join(s)


def b_section(root: Path, ev: Path) -> str:
    s = ["## B2. Unknown site (`runs/paper/a1/test_L24/compare.json`)", ""]
    c = load(root / "a1/test_L24/compare.json")
    if c and c.get("b2"):
        rows = [[NAMES.get(k, k), x["anchors"], f(x["known"].get("auroc")), f(x["unknown"].get("auroc")),
                 f(x["known"].get("observed_over_expected"), 2), f(x["unknown"].get("observed_over_expected"), 2)]
                for k, x in c["b2"].items() if k in PRIMARY["L24"]]
        s += [table(["Endpoint", "Anchors", "AUROC, site known", "AUROC, site unknown", "O/E known", "O/E unknown"], rows),
              "", "Annotations: the same anchors (B2's subset of the L24 anchors) with the prompt's dataset and site tokens set to "
              "unknown; the rollouts are decoded with their stay's real dataset and site. A proxy only: no dataset was "
              "held out of training.", ""]
    s += ["## B3. Ablations across checkpoints (inference only)", ""]
    a3 = load(root / "a3/a3.json") or {"checkpoints": {}}
    rows = []
    for run in ("M32_main", "ML32_main", "MC32_main", "MCN32_main", "MU32_main", "MUH32_main", "MUF32_main", "MUC32_main"):
        pr = load(root / (f"b3/{run}/probe.json" if run != "MUC32_main" else "a1/test_L24/probe.json"))
        tf = load(ev / f"{run}/calibration_masked.json")
        c = a3["checkpoints"].get(run, {})
        oe = {}
        if tf:
            for e in ("imv", "rrt"):
                o = sum(tf[ds][e]["after_24h"]["observed"] for ds in tf)
                x = sum(tf[ds][e]["after_24h"]["expected"] for ds in tf)
                oe[e] = o / x if x else None
        p = (pr or {}).get("programs", {})
        rows.append([run, c.get("optimizer"), f(c.get("tuning_nll"), 4), f(c.get("p1", {}).get("pooled", {}).get("auc")),
                     f(p.get("icu_death@-1", {}).get("pooled", {}).get("auroc")),
                     f(p.get("imv_onset@1440", {}).get("pooled", {}).get("auroc")),
                     f(p.get("rrt_onset@2880", {}).get("pooled", {}).get("auroc")), f(oe.get("imv"), 2),
                     f(oe.get("rrt"), 2)])
    s += [table(["Checkpoint", "Optimizer", "Tuning NLL", "P1 AUC pooled", "Probe AUROC ICU death (L24)",
                 "Probe AUROC new IMV 24 h", "Probe AUROC new RRT 48 h", "Teacher-forced O/E IMV after 24 h",
                 "Teacher-forced O/E RRT after 24 h"], rows), "",
          "Annotations: M32 (32k context, AdamW), ML32 (2.2x larger), MC32 (ICU-day clock blocks), MCN32 (+ input noise), "
          "MU32/MUH32 (Muon, two rates), MUF32/MUC32 (Muon, passes on the privacy frontier; MUC32 adopted). M32 and "
          "ML32 read r3 tokens; ML32 has no teacher-forced audit. S and S_main (older tokens) are not included.", ""]
    return "\n".join(s)


def c_section(root: Path) -> str:
    s = ["## C1. What-if rollouts (`runs/paper/c1/c1.json`)", ""]
    c1 = load(root / "c1/c1.json")
    if c1:
        rows = []
        for t, x in c1["treatments"].items():
            for r, y in x["responses"].items():
                d = y.get("dose_response")
                rows.append([t, r, "+" if y["expected_sign"] > 0 else "-", y["scored"],
                             f(y.get("share_expected_sign"), 2), f(y.get("mean_effect")), ci(y.get("mean_effect_ci")),
                             f(y.get("withhold_mean"), 2),
                             "; ".join(f"{k[5:]} {f(v['mean_response'], 1)}" for k, v in sorted(d.items())) if d else "-"])
        s += [table(["Treatment", "Response", "Expected", "Prefixes", "Share expected sign", "Mean effect (give - withhold)",
                     "95% CI", "Withhold mean", "Dose arms (mean response)"], rows), "",
              "Annotations: 16 rollouts per arm and prefix; response windows 6 h (creatinine 24 h); urine output without "
              "records above 2,000 mL (amendment 6). Effects are twin-world: they inherit the data's confounding by "
              "indication (norepinephrine starts accompany falling pressure), so they are not treatment-effect "
              "estimates.", ""]
    s += ["## C2. Attribution (`runs/paper/c2/test_<design>/c2.json`)", ""]
    for d in ("L24", "R"):
        c2 = load(root / f"c2/test_{d}/c2.json")
        if not c2:
            continue
        rows, rows2 = [], []
        for p, x in c2["programs"].items():
            fa, ag = x["faithfulness"], x["agreement"]
            rows.append([NAMES.get(p, p), x["anchors"]] + [f(fa.get(m, {}).get("mean_drop")) for m in
                         ("edit", "ig", "attn_rollout", "attn_x_grad", "random")] +
                        [f(ag.get(m, {}).get("spearman_median"), 2) + " / " + f(ag.get(m, {}).get("top3_overlap_mean"), 2)
                         for m in ("ig", "attn_rollout", "attn_x_grad")] + [f(x.get("ig_completeness_relative_error_median"), 2)])
            top = sorted(x["groups"].items(), key=lambda z: -z[1]["top3_share"])[:6]
            rows2.append([NAMES.get(p, p), "; ".join(f"{g} ({f(v['top3_share'], 2)}, {f(v['mean_edit_effect'], 2)})"
                                                    for g, v in top)])
        s += [f"### {DESIGNS[d]}", "",
              table(["Endpoint", "Anchors", "Deletion drop: editing", "IG", "Attention rollout", "Attention x grad",
                     "Random", "IG vs editing (rho / top-3)", "Rollout vs editing", "Attn x grad vs editing",
                     "IG completeness error"], rows), "",
              "Groups most often among the top 3 (share of anchors, mean edit effect on the probe's logit):", "",
              table(["Endpoint", "Groups"], rows2), ""]
        rc = c2.get("rollout_check")
        if rc:
            s += [table(["Endpoint", "Anchors", "Sign agreement probe vs rollouts", "Rank correlation (median)"],
                        [[NAMES.get(p, p), x["anchors"], f(x["sign_agreement"], 2), f(x["spearman_median"], 2)]
                         for p, x in rc.items()]), ""]
    s += ["Annotations: the target is the Twin probe's logit; deletion drop = mean fall of the logit when a method's "
          "top 1-5 groups are removed (higher = more faithful), against a random order. Attention methods cover prompts "
          "up to 4,096 tokens. Anchors: the 100 with the highest probe risk plus 100 at random per endpoint. Rollout "
          "check: the first 50 anchors per endpoint, original and its 3 highest and 3 lowest edits rolled out 16 times "
          "each; an anchor is scored when its original and at least 4 edits each have at least 8 rollouts with a valid "
          "label (eligible, horizon reached), so the scored anchors are few. "
          "Attributions are causal within the model only.", ""]
    c3 = load(root / "c3/c3.json")
    if c3:
        s += ["## C3. Virtual cohorts (`runs/paper/c3/c3.json`)", ""]
        keys = ("age", "female", "icu_death", "hosp_death", "ever_imv", "ever_vasopressor", "ever_rrt", "stay_lt_1h",
                "events_per_hour", "labs_per_day")
        rows = []
        for ds, x in c3["datasets"].items():
            for src in ("real", "twins", "control"):
                y = x[src]
                rows.append([DSN.get(ds, ds), src, y["stays"]] +
                            [f(y.get(k, {}).get("mean", y.get(k, {}).get("median")), 3) for k in keys] +
                            [f(y["los_h"]["median"], 1)])
        s += [table(["Dataset", "Set", "Stays", "Age", "Female", "ICU death", "Hospital death", "IMV", "Vasoactive",
                     "RRT", "Stay < 1 h", "Events / h (median)", "Labs / day (median)", "LOS h (median)"], rows), "",
              f"Inter-dataset structure (correlation of profile distances with real): twins "
              f"{f(c3['inter_dataset']['twins_vs_real'], 2)}, real control {f(c3['inter_dataset']['control_vs_real'], 2)}. "
              f"eICU hospitals, total variation from real: twins {f(c3['eicu_hospitals']['twins_vs_real_tv'])}, control "
              f"{f(c3['eicu_hospitals']['control_vs_real_tv'])}. Rare strata: {c3['rare_strata']['strata']} strata, "
              f"share twins {f(c3['rare_strata']['share_twins'])} against real {f(c3['rare_strata']['share_real'])}, all "
              f"reached: {c3['rare_strata']['strata_reached_by_twins']}; their ICU death twins "
              f"{f(c3['rare_strata']['icu_death_twins'])} against real {f(c3['rare_strata']['icu_death_real'])}.", "",
              "Annotations: twins from the internal bank (round-3 sampler), real = held-out stays, control = real train "
              "stays against the same held-out stays. The death gap is the generator's known one.", ""]
    c4 = load(root / "c4/c4.json")
    if c4:
        s += ["## C4. Patient similarity (`runs/paper/c4/c4.json`)", "",
              table(["Endpoint", "Prevalence", "Positive neighbours of positives: model state", "TF-IDF",
                     "Lift: model state", "TF-IDF"],
                    [[NAMES.get(p, p), f(x["prevalence"]), f(x["state"]["positive_neighbours_of_positives"]),
                      f(x["tfidf"]["positive_neighbours_of_positives"]), f(x["state"]["lift"], 2),
                      f(x["tfidf"]["lift"], 2)] for p, x in c4["programs"].items()]), "",
              f"Annotations: {c4['anchors']} L24 anchors, {c4['k']} nearest other stays by cosine.", ""]
    return "\n".join(s)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables", default=None)
    a = ap.parse_args(argv)
    root, ev = Path(a.root), Path(a.eval)
    head = ["# Materials: results of the supplementary experiments", "",
            "Every table is computed from the aggregate JSON files under `runs/paper/` on the host (protocol: "
            "`docs/p1_paper_protocol.md`, amendments 1-6), rendered by `tools/paper_results.py`. Held-out split "
            "unless stated; the tuning-split dry runs are not results.", ""]
    body = [a1_section(root), a2_section(root, ev), a3_section(root), a4_section(root), a5_section(root),
            b_section(root, ev), c_section(root)]
    text = export_tables("\n".join(head + body), Path(a.tables) if a.tables else None)
    Path(a.out).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {a.out}" + (f" and its tables to {a.tables}" if a.tables else ""))


if __name__ == "__main__":
    main()
