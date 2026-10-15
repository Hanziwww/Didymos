"""The clinician study's pre-specified analysis runs end to end on a simulated study (no data): exports joined and
checked, tidy tables with the 80% completeness rule, the bootstrap and per-item cross-checks, and, where R is
installed, the lme4 primary models agreeing with them."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    import clinician_study_analysis as ca

    d = tmp_path_factory.mktemp("clin")
    ca.B = 60
    ca.fake(d, raters=12)
    key = json.loads((d / "key.json").read_text())
    res = json.loads((d / "responses.json").read_text())
    t = ca.tables(res["rows"], key)
    (d / "analysis").mkdir()
    for name, df in t.items():
        df.to_csv(d / "analysis" / f"{name}.csv", index=False, lineterminator="\n")
    return d, res, t, ca.secondary({k: pd.read_csv(d / "analysis" / f"{k}.csv")
                                    for k in ("a", "b1", "b2", "c", "q", "profile")})


def test_join_and_tables(study) -> None:
    _d, res, t, _s = study
    assert {(p["file"], p["problem"]) for p in res["problems"]} == {
        ("R01_progress.json", "superseded by a more complete export"), ("tampered.json", "checksum mismatch")}
    assert len(t["a"]) == 12 * 80 and len(t["b1"]) == 12 * 24 and len(t["b2"]) == 12 * 20
    prof = t["profile"].set_index("rater")
    assert prof.loc["R12", "answered_C"] == pytest.approx(30 / 48) and prof.loc["R01", "answered_C"] == 1.0
    assert set(t["c"].loc[t["c"]["complete"] == 0, "rater"]) == {"R12"}
    c = t["c"]
    assert c["unaided"].between(0, 1).all() and c["aided"].between(0, 1).all()
    w = c.drop_duplicates("case").groupby(["dataset", "label"])["weight"].first()
    assert w[("mimic", 1.0)] == pytest.approx(600 / 6) and w[("eicu", 0.0)] == pytest.approx(6500 / 18)
    assert t["q"]["sus"].between(0, 100).all() and t["q"]["trust"].between(1, 7).all()


def test_secondary_cross_checks(study) -> None:
    _d, _res, t, s = study
    a = t["a"][t["a"]["complete"] == 1]
    assert s["A"]["primary"]["accuracy_90"]["est"] == pytest.approx(a["correct"].mean())
    ci = s["A"]["primary"]["accuracy_90"]
    assert ci["lo"] < ci["est"] < ci["hi"] and ci["level"] == 0.9
    b1 = s["B1"]["primary"]["twin_minus_real"]
    assert b1["items"] == 20 and b1["lo"] < b1["est"] < b1["hi"]
    c = s["C"]["icu_death"]
    assert c["counterfactual_blend"]["lgbm"]["twin_replay_minus_unaided"]["est"] == pytest.approx(
        c["brier_aided_minus_unaided"]["est"], abs=2e-4)      # the replay reproduces the aided estimates
    assert 0 <= c["weight_of_advice"]["mean"]["est"] <= 1
    assert s["C"]["primary"]["raters"] == 11 and s["C"]["sensitivity_all_raters"]["raters"] == 12


def test_helpers() -> None:
    import clinician_study_analysis as ca

    y, p = np.array([0, 0, 1, 1, 0, 1]), np.array([0.1, 0.4, 0.35, 0.8, 0.4, 0.4])
    from sklearn.metrics import roc_auc_score

    assert ca.wauc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert ca.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    q = rng.uniform(0.02, 0.98, 4000)
    slope, citl = ca.calibration(rng.random(4000) < q, q, np.ones(4000))
    assert slope == pytest.approx(1, abs=0.15) and citl == pytest.approx(0, abs=0.1)


def _rscript() -> str | None:
    for c in (shutil.which("Rscript"), "C:/Program Files/R/R-4.3.1/bin/Rscript.exe", "/root/autodl-tmp/r-env/bin/Rscript"):
        if c and Path(c).exists():
            return c
    return None


def test_lme4_models_agree(study) -> None:
    r = _rscript()
    if r is None:
        pytest.skip("R")
    d, _res, _t, s = study
    p = subprocess.run([r, str(TOOLS / "clinician_study_models.R"), str(d / "analysis"), "19"], capture_output=True,
                       text=True)
    if "there is no package" in p.stderr:
        pytest.skip("lme4")
    assert p.returncode == 0, p.stderr[-2000:]
    m = json.loads((d / "analysis" / "primary_lme4.json").read_text())
    assert abs(m["A"]["primary"]["accuracy"] - s["A"]["primary"]["accuracy_90"]["est"]) < 0.02
    assert m["B1"]["primary"]["twin_minus_real"]["est"] == pytest.approx(s["B1"]["primary"]["twin_minus_real"]["est"],
                                                                         abs=0.05)
    assert m["C"]["primary"]["icu_death"]["aided_minus_unaided"]["est"] == pytest.approx(
        s["C"]["primary"]["brier_aided_minus_unaided"]["est"], abs=2e-3)
    assert m["B2"]["primary"]["edit_minus_random"]["p_holm"] >= m["B2"]["primary"]["edit_minus_random"]["p"]


def test_pilot_set_and_archives(tmp_path) -> None:
    """The short pilot set takes five cases from a build's P01 (one real and one synthetic practice sheet, editing's
    factors, a twin course that is not insulin, a case asking both outcomes) with a card readable under its own key;
    the archives hold one zip per packet in number order, intact."""
    import hashlib
    import io
    import zipfile
    from argparse import Namespace

    import clinician_study as cs

    from icutwin.toolkit import rating

    d = tmp_path
    (d / "packets").mkdir()
    salt, sheet = "s", "<div>sheet</div>"
    items = {"t1": {"module": "practice", "kind": "twin"}, "t2": {"module": "practice", "kind": "real"},
             "b21": {"module": "B2"}, "b22": {"module": "B2"},
             "b11": {"module": "B1", "treatment": "insulin"}, "b12": {"module": "B1", "treatment": "insulin"},
             "b13": {"module": "B1", "treatment": "crrt"}, "c1": {"module": "C"}, "c2": {"module": "C"}}
    key = {"build": "", "items": items, "raters": {}}
    man = {}
    for k, name in enumerate(("R01", "R02", "P01")):
        ck = rating.card_key(name, salt)
        outs = [{"key": "icu_death", "label": "d"}, {"key": "aki", "label": "a"}]
        mods = [{"id": "T", "title": "Practice", "kind": "realism", "practice": True, "intro": "",
                 "items": [{"id": i, "html": sheet} for i in ("t1", "t2")]},
                {"id": "B2", "title": "B2", "kind": "factors", "intro": "", "break_after": True,
                 "items": [{"id": i, "html": sheet, "risk": "40%", "outcome": "o", "factors": []} for i in ("b21", "b22")]},
                {"id": "B1", "title": "B1", "kind": "course", "intro": "",
                 "items": [{"id": i, "html": sheet, "lead": "", "direction": ""} for i in ("b11", "b12", "b13")]},
                {"id": "C", "title": "C", "kind": "reader", "intro": "",
                 "items": [{"id": "c1", "html": sheet, "outcomes": outs[:1], "card": rating.obfuscate("card one", ck)},
                           {"id": "c2", "html": sheet, "outcomes": outs, "card": rating.obfuscate("card two 37%", ck)}]}]
        page = rating.build_packet({"title": "t", "intro": "", "packet": name, "build": "bx", "cardkey": ck, "modules": mods})
        (d / "packets" / f"icu_case_review_{name}.html").write_text(page, encoding="utf-8", newline="\n")
        man[name] = {"file": f"icu_case_review_{name}.html", "bytes": len(page),
                     "sha256": hashlib.sha256(page.encode()).hexdigest()}
        key["raters"][name] = {"b2_method": {"b21": ("random", "edit")[k % 2], "b22": ("edit", "knockout")[k % 2]},
                               "b1_version": {"b11": "twin", "b12": "real", "b13": ("real", "twin")[k % 2] if k < 2
                                              else "twin"}}
    key["build"] = "bx"
    (d / "key.json").write_text(json.dumps(key))
    (d / "items.json").write_text(json.dumps({"salt": salt}))
    (d / "packets" / "manifest.json").write_text(json.dumps({"build": "bx", "packets": man}))
    cs.pilot(d, Namespace(copies=3))
    m5 = json.loads((d / "pilot5" / "manifest.json").read_text())
    assert sorted(m5["packets"]) == ["S01", "S02", "S03"] and m5["build"] != "bx"
    page = (d / "pilot5" / "icu_case_review_S02.html").read_text(encoding="utf-8")
    P = cs.packet_data(page)
    got = {m["id"]: [it["id"] for it in m["items"]] for m in P["modules"]}
    assert sorted(got["A"]) == ["t1", "t2"] and got["B2"] == ["b22"] and got["B1"] == ["b13"] and got["C"] == ["c2"]
    assert sum(len(v) for v in got.values()) == 5 and not any(m.get("break_after") for m in P["modules"])
    assert P["profile"] is False and "profile" not in cs.packet_data((d / "packets" / "icu_case_review_P01.html")
                                                                     .read_text(encoding="utf-8"))
    assert "card two" not in page
    assert rating.deobfuscate(P["modules"][3]["items"][0]["card"], rating.card_key("S02", salt)) == "card two 37%"
    cs.archive(d, Namespace())
    z = zipfile.ZipFile(d / "archives" / "icu_case_review_bx_distribution.zip")
    names = z.namelist()
    assert [n for n in names if n.startswith("raters/")] == ["raters/R01.zip", "raters/R02.zip"]
    assert "pilot/P01.zip" in names and "distribution_log.csv" in names and "README_center_leads.txt" in names
    inner = zipfile.ZipFile(io.BytesIO(z.read("raters/R02.zip")))
    assert sorted(inner.namelist()) == ["icu_case_review_R02.html", "instructions.txt"]
    assert hashlib.sha256(inner.read("icu_case_review_R02.html")).hexdigest() == man["R02"]["sha256"]
    assert "pilot_feedback_form.txt" in zipfile.ZipFile(io.BytesIO(z.read("pilot/P01.zip"))).namelist()
    log = z.read("distribution_log.csv").decode().splitlines()
    assert log[1].startswith("1,R01,raters/R01.zip,") and log[3].startswith("3,P01,pilot/P01.zip,")
    z5 = zipfile.ZipFile(d / "archives" / f"icu_case_review_{m5['build']}_pilot5.zip")
    assert sorted(n for n in z5.namelist() if n.endswith(".zip")) == ["S01.zip", "S02.zip", "S03.zip"]


def test_case_sheet_layout_rules() -> None:
    """The case sheet's texts keep to their zones: names truncated to the gutter (the full name in a tooltip), values
    in the plot written only where they fit and change, FiO2 in percent."""
    from icutwin.toolkit import casesheet as cs

    long = "piperacillin tazobactam sodium bicarbonate"
    shown, cut = cs.fit(long, cs.L - 14)
    assert cut and shown.endswith("…") and cs.text_width(shown) <= cs.L - 14
    assert cs.fit("PEEP (cmH2O)", cs.L - 14) == ("PEEP (cmH2O)", False)
    assert "<title>" in cs._gutter(10, long) and "<title>" not in cs._gutter(10, "propofol")
    assert cs.text_width("0.05 mcg/kg/min", "rate") > cs.text_width("5", "rate")
    vals = [(cs.L + k * 3.0, "40") for k in range(10)] + [(cs.L + 40, "60"), (cs.L + 41, "70"), (cs.W - cs.R, "100")]
    kept = cs._spread(vals, cs.L, cs.W - cs.R, "val")
    xs = [(x - cs.text_width(s, "val") / 2, x + cs.text_width(s, "val") / 2) for x, s in kept]
    assert all(b[0] >= a[1] for a, b in zip(xs, xs[1:])) and xs[0][0] >= cs.L and xs[-1][1] <= cs.W - cs.R
    assert [s for _x, s in kept][:2] == ["40", "60"]           # repeats dropped, a crowded change skipped
    assert cs._fio2(0.4) == pytest.approx(40) and cs._fio2(60.0) == 60.0
