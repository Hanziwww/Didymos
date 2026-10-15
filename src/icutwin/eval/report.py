"""The S8 evaluation report (plan S8: ``runs/<id>/eval/report.html``): one static HTML page of aggregates from the
privacy gate (``eval.cli`` out_dir: p1, p2, synthetic, report), the quality pilot (``eval.quality`` report: fidelity
and diversity per dataset and pooled, with their controls, and utility) and twin forecasting (``eval.forecast``
scores). Aggregates only: no stay, sample or record appears in it.

    python -m icutwin.eval.report --privacy <gate dir> --quality <report.json> --forecast <scores.json> --out <html>
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

CSS = """body{font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;color:#1d1d1f}
h1{font-size:22px}h2{font-size:18px;margin-top:28px;border-bottom:1px solid #ddd}h3{font-size:15px}
table{border-collapse:collapse;margin:8px 0 16px;font-size:13px}td,th{border:1px solid #ddd;padding:3px 8px;text-align:right}
th{background:#f4f4f6}td:first-child,th:first-child{text-align:left}.PASS{color:#1a7f37;font-weight:600}
.FAIL{color:#c62828;font-weight:600}.PENDING,.INCONCLUSIVE{color:#9a6700;font-weight:600}.flag{color:#c62828}
.muted{color:#6e6e73}"""


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.3f}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"
    if isinstance(x, (list, tuple)) and len(x) == 3 and all(isinstance(v, (int, float)) for v in x):
        return f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"
    if isinstance(x, (list, tuple)):
        return ", ".join(_fmt(v) for v in x)
    s = html.escape(str(x))
    return f'<span class="{s}">{s}</span>' if s in ("PASS", "FAIL", "PENDING", "INCONCLUSIVE") else s


def table(rows: list[dict], cols: list[str] | None = None, flagged: dict | None = None) -> str:
    """Rows of dicts as a table; ``flagged``: row index -> the columns to mark."""
    if not rows:
        return '<p class="muted">none</p>'
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    out = ["<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>"]
    for i, r in enumerate(rows):
        marks = (flagged or {}).get(i, ())
        out.append("<tr>" + "".join(f'<td class="{"flag" if c in marks else ""}">{_fmt(r.get(c))}</td>'
                                     for c in cols) + "</tr>")
    return "".join(out) + "</table>"


def _read(path: Path | None):
    return json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else None


def privacy_section(d: Path) -> str:
    rep, p1, p2, syn = (_read(d / f"{n}.json") for n in ("report", "p1", "p2", "synthetic"))
    out = ["<h2>Privacy gate (M7 protocol)</h2>"]
    if rep:
        out.append(table([{"test": k, "status": v} for k, v in rep.items() if isinstance(v, str)]))
    if p1:
        out.append("<h3>P1 loss-based membership inference, per dataset</h3>")
        out.append(table([{"dataset": k, "members": g["members"], "nonmembers": g["nonmembers"], "AUC": g["auc"],
                           "TPR at 1% FPR": g["tpr_at_1pct_fpr"], "status": g["status"]}
                          for k, g in p1["groups"].items()]))
    if p2:
        out.append("<h3>P2 prefix-prompted exact continuation</h3>")
        out.append(table([{"setting": s, "dataset": k, "exact (members)": g["exact_members"],
                           "exact (non-members)": g["exact_nonmembers"], "difference": g["difference"],
                           "status": g["status"]} for s, rows in p2["settings"].items() for k, g in rows.items()]))
    if syn:
        out.append(f'<h3>P3-P5 on the audit samples</h3><p class="muted">samples {syn["samples"]}, grammar-valid '
                   f'{syn["grammar_valid_share"]:.4f}</p>')
        rows = []
        for g in ("model", "private_reference", "copying_generator"):
            rows.append({"set": g, "P3": syn["p3"][g]["status"], "share below DCR q05":
                         syn["p3"][g]["share_below_dcr_q05"], "P4": syn["p4"][g]["status"],
                         "P4 AUC": syn["p4"][g].get("auc"), "P5": syn["p5"][g].get("status")})
        out.append(table(rows))
    return "".join(out)


def harness_rows(h: dict, keys: list[str], diversity: list[str]) -> tuple[list, dict]:
    rows, marks = [], {}
    sets = [("positive control", {"metrics": h["positive_control"], "flags": []})]
    sets += [(f"control: {k}", x) for k, x in h["negative_controls"].items()]
    sets += [(f"candidate: {k}", x) for k, x in h["candidates"].items()]
    for i, (name, x) in enumerate(sets):
        m = x["metrics"]
        r = {"set": name, **{k: m.get(k) for k in keys}, **{k: (m.get("diversity") or {}).get(k) for k in diversity},
             "flags": len(x["flags"])}
        rows.append(r)
        marks[i] = set(x["flags"])
    return rows, marks


FIDELITY = ["discriminator_auc", "code_jsd", "value_jsd", "length_ks", "los_ks", "death_rate_diff", "coverage",
            "recall"]
DIVERSITY = ["vendi_ratio", "rare_code_recall", "distinct_ratio", "duplicate_rate", "nn_ratio", "copy_rate",
             "site_confusion_diff", "mmd_corr"]


def quality_section(q: dict) -> str:
    out = ["<h2>Quality pilot: fidelity and diversity</h2>",
           '<p class="muted">Each metric against held-out stays; the positive control (real training stays) sets '
           "the band, marked cells are flagged against it; every negative control must be flagged.</p>"]
    if "pooled" in q:
        rows, marks = harness_rows(q["pooled"], FIDELITY, DIVERSITY)
        out.append("<h3>Pooled</h3>" + table(rows, ["set", *FIDELITY, *DIVERSITY, "flags"], marks))
        out.append(f'<p>controls detected: {_fmt(q["pooled"].get("controls_detected"))}</p>')
    per = []
    for ds, rep in (q.get("datasets") or {}).items():
        c = rep["candidates"]["model"]
        per.append({"dataset": ds, "grammar valid": rep.get("grammar_valid"),
                    **{k: c["metrics"].get(k) for k in FIDELITY[:4]},
                    **{k: (c["metrics"].get("diversity") or {}).get(k) for k in ("vendi_ratio", "copy_rate")},
                    "flags": ", ".join(c["flags"]) or "none"})
    if per:
        out.append("<h3>Per dataset (the model's samples)</h3>" + table(per))
    u = q.get("utility")
    if u:
        out.append(f"<h2>Utility: TSTR against TRTR ({u['landmark_hours']} h landmark)</h2>")
        rows = []
        for t, base in u["trtr"].items():
            r = {"task": t, "TRTR AUROC": base["auroc"], "test prevalence": base["test_prevalence"]}
            for name, res in u["sets"].items():
                r[f"{name} AUROC"] = res[t]["auroc"]
                r[f"{name} ratio"] = res[t]["utility_ratio"]
            rows.append(r)
        out.append(table(rows))
    return "".join(out)


def forecast_section(f: dict) -> str:
    """``eval.forecast.score_dir``'s output: per dataset, the binary outcomes and the ICU stay's dispersion."""
    out = ["<h2>Twin forecasting (CoMET protocol)</h2>"]
    per = f if all(isinstance(v, dict) and "death" in v for v in f.values()) else {"all": f}
    rows, disp = [], []
    for ds, scores in per.items():
        for k, v in scores.items():
            if k == "icu_los_hours":
                disp.append({"dataset": ds, **{m: v.get(m) for m in ("stays", "pit_ks", "crps", "spread_skill",
                                                                      "rollouts_censored")}})
            elif isinstance(v, dict) and "stays" in v:
                rows.append({"dataset": ds, "outcome": k, **{m: v.get(m) for m in (
                    "stays", "prevalence", "mean_predicted", "auroc", "auprc", "ece", "observed_over_expected")}})
    out.append(table(rows))
    if disp:
        out.append("<h3>ICU length of stay: dispersion of the rollouts</h3>" + table(disp))
    return "".join(out)


def render(title: str, privacy: Path | None = None, quality: Path | None = None, forecast: Path | None = None) -> str:
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
             f"<style>{CSS}</style></head><body><h1>{html.escape(title)}</h1>"
             '<p class="muted">Aggregates only. Evaluation samples are not deliverables and are deleted after their '
             "report.</p>"]
    if privacy and privacy.exists():
        parts.append(privacy_section(privacy))
    q = _read(quality)
    if q:
        parts.append(quality_section(q))
    f = _read(forecast)
    if f:
        parts.append(forecast_section(f))
    return "".join(parts) + "</body></html>"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="icutwin evaluation report")
    ap.add_argument("--privacy", type=Path)
    ap.add_argument("--quality", type=Path)
    ap.add_argument("--forecast", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(render(a.title, a.privacy, a.quality, a.forecast), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
