"""The S8 HTML report renders the gate, harness, utility and forecast aggregates it is given, marks flagged cells,
and holds nothing but aggregates."""

from __future__ import annotations

import json
from pathlib import Path

from icutwin.eval import report


def _harness(flagged: list) -> dict:
    m = {"discriminator_auc": 0.55, "code_jsd": 0.01, "coverage": 0.9, "diversity": {"vendi_ratio": 1.0, "copy_rate": 0.05}}
    return {"positive_control": m, "controls_detected": True,
            "negative_controls": {"collapsed": {"metrics": {**m, "coverage": 0.1}, "flags": ["coverage"]}},
            "candidates": {"model": {"metrics": m, "flags": flagged}}}


def test_report_renders_every_section(tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    gate.mkdir()
    (gate / "report.json").write_text(json.dumps({"P1": "PASS", "P5": "INCONCLUSIVE", "gate": "INCONCLUSIVE"}))
    (gate / "p1.json").write_text(json.dumps({"groups": {"mimic": {"members": 3, "nonmembers": 3, "auc": [0.56, 0.54, 0.58],
                                                                   "tpr_at_1pct_fpr": [0.01, 0.0, 0.02], "status": "PASS"}}}))
    quality = tmp_path / "q.json"
    quality.write_text(json.dumps({
        "pooled": _harness([]),
        "datasets": {"eicu": {"grammar_valid": 1.0, "candidates": {"model": {"metrics": _harness([])["positive_control"],
                                                                             "flags": ["code_jsd"]}}}},
        "utility": {"landmark_hours": 24, "trtr": {"death": {"auroc": 0.8, "test_prevalence": 0.1}},
                    "sets": {"model": {"death": {"auroc": 0.75, "utility_ratio": 0.83}}}}}))
    forecast = tmp_path / "f.json"
    forecast.write_text(json.dumps({"mimic": {"death": {"stays": 10, "prevalence": 0.1, "auroc": 0.7, "ece": 0.02},
                                              "icu_los_hours": {"stays": 9, "pit_ks": 0.1, "crps": 12.0,
                                                                "spread_skill": 0.9, "rollouts_censored": 0.02}}}))
    html = report.render("t", gate, quality, forecast)
    for s in ("Privacy gate", "INCONCLUSIVE", "Pooled", "control: collapsed", "Per dataset", "TSTR", "Twin forecasting",
              'class="flag"', "dispersion", "mimic"):
        assert s in html, s
    assert html.count("<table>") >= 7
    out = tmp_path / "r.html"
    report.main(["--privacy", str(gate), "--quality", str(quality), "--forecast", str(forecast), "--out", str(out)])
    assert out.read_text(encoding="utf-8") == html.replace("<title>t</title>", "<title>icutwin evaluation report</title>") \
        .replace("<h1>t</h1>", "<h1>icutwin evaluation report</h1>")
