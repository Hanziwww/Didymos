"""The application toolkit (``icutwin.toolkit``) on the CPU: a random-initialized tiny model on the full r4 vocabulary,
real held-out token sequences as patients (skipped without the full r4 tokens), a stand-in gate for the tiny
weights. The patients and anchors are the paper's prompts; the job's requests round-trip; the reference sampler draws
grammatical continuations under interventions; the teacher's hazards give proper E0/E2 curves; a forecast runs end to
end into a valid result; the toolkit's copies of the paper's definitions agree with the paper scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

R = Path(__file__).resolve().parents[1]
T = R / "data" / "tokens" / "full_r4"
STAY = "split:mimic/held_out/mimic:38983722"


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    if not (T / "vocab.json").exists() or not (T / "mimic" / "held_out.parquet").exists():
        pytest.skip("the full r4 tokens")
    import torch
    from transformers import Qwen3ForCausalLM

    from icutwin.generate import producer
    from icutwin.model.presets import qwen3_config
    from icutwin.tokenization.vocab import Vocab
    from icutwin.toolkit.session import Paths, Session

    v = Vocab.load(T / "vocab.json")
    d = tmp_path_factory.mktemp("toolkit")
    torch.manual_seed(0)
    Qwen3ForCausalLM(qwen3_config("tiny", len(v.tokens), v.id("[PAD]"), v.id("[BOS]"), v.id("[EOS]"),
                                  max_positions=8192)).save_pretrained(d / "hf")
    audit = d / "audit"
    audit.mkdir()
    (audit / "report.json").write_text(json.dumps({"gate": "PASS", "amendment": "test",
                                                   "not_in_v1": {"conditional_and_counterfactual_sampling": "PENDING"}}))
    (audit / "p1.json").write_text(json.dumps({"model": {"weights_sha256": producer.weights_sha256(d / "hf")}}))
    paths = Paths(R, T, d / "hf", audit, d / "out", R / "runs" / "paper", R / "data" / "bank" / "MUC32_main_bank",
                  R / "data" / "factory" / "stores" / "bank_dev", {})
    return Session(paths, backend="hf", device="cpu", workers=1)


def test_patient_anchor_is_the_papers_prompt(session) -> None:
    from icutwin.eval import bench

    v = session.vocab
    p = session.patient(STAY)
    assert p.kind == "held_out" and p.dataset == "mimic" and p.real
    a = p.at(v, 6.0)
    assert a.prompt == [int(x) for x in bench.prompt(p.ids, v, p.stay_id, 6.0)]
    assert a.t_min == bench.anchor_time(a.prompt, v, p.stay_id) and 0 < a.t_min <= 360
    with pytest.raises(ValueError):
        p.at(v, p.los_hours(v) + 1)
    u = session.patient(STAY, unknown_site=True).at(v, 6.0)
    assert u.prompt == [int(x) for x in bench.unknown_site(a.prompt, v)] and u.prompt != a.prompt


def test_job_requests_round_trip_and_stage(session, tmp_path: Path) -> None:
    from icutwin.toolkit import stage
    from icutwin.toolkit.jobs import Job, SimRequest

    v = session.vocab
    reqs = [SimRequest(f"r{i}", (1, 2, 3), 3, 10 + i, 50 if i else None, (5,) if i == 1 else (), (7, 8) if i == 2 else ())
            for i in range(3)]
    job = Job(tmp_path / "j")
    job.write_requests(reqs)
    assert job.requests() == reqs

    class Stub:
        def simulate(self, group):
            return {r.id: [[v.id("[END]"), v.id("[EOS]")]] * r.n for r in group}

    info = stage.run(job, Stub(), batch_rollouts=4)
    assert info["batches"] == 3 and info["written"] == 3
    out = job.rollouts()
    assert set(out) == {"r0", "r1", "r2"} and all(len(x) == 3 for x in out.values())
    assert stage.run(job, Stub(), batch_rollouts=4)["written"] == 0           # resumed: nothing redrawn


def test_reference_sampler_draws_grammatical_continuations(session) -> None:
    from icutwin.generate.engine import treatment_tokens
    from icutwin.tokenization.fsm import Automaton
    from icutwin.toolkit.backends import HfSampler
    from icutwin.toolkit.jobs import SimRequest

    v, t = session.vocab, session.tables
    a = session.patient(STAY).at(v, 6.0)
    hold = tuple(sorted(treatment_tokens(v, "vasopressor")))
    s = HfSampler(session.paths.model, T / "vocab.json", device="cpu")
    res = s.simulate([SimRequest("plain", tuple(a.prompt), 3, 1, 60), SimRequest("hold", tuple(a.prompt), 3, 2, 60,
                                                                                 withhold=hold)])
    for rid, conts in res.items():
        assert len(conts) == 3
        for c in conts:
            aut = Automaton(t)
            assert all(aut.step(tok) for tok in a.prompt + c), aut.violation
            assert len(c) <= 60
            if rid == "hold":
                assert not set(c) & set(hold)


def test_teacher_hazards_give_proper_curves(session) -> None:
    from icutwin.toolkit import forecast
    from icutwin.toolkit.backends import HfSampler
    from icutwin.toolkit.jobs import SimRequest
    from icutwin.toolkit.teacher import LONG_EDGES, HazardInput

    v = session.vocab
    a = session.patient(STAY).at(v, 6.0)
    conts = HfSampler(session.paths.model, T / "vocab.json").simulate([SimRequest("x", tuple(a.prompt), 4, 3, 80)])["x"]
    paths = session.teacher.hazards([HazardInput(np.asarray(a.prompt), conts, "mimic", a.patient.stay_id, 80)])[0]
    assert len(paths) == 4
    for p in paths:
        assert all(np.all(np.isfinite(x)) and np.all(x >= 0) for x in p.inc.values())
    s = forecast.e2_summary(paths, LONG_EDGES)
    for c in forecast.CAUSES:
        e2 = np.array(s["curves"][c]["e2"], float)
        ok = np.isfinite(e2)
        assert ok.any() and np.all((e2[ok] >= 0) & (e2[ok] <= 1))
        assert np.all(np.diff(e2[ok]) >= -1e-9)                               # a cumulative incidence
    assert 0 <= s["censored_share"] <= 1


def test_forecast_end_to_end(session) -> None:
    from icutwin.toolkit import schema

    v = session.vocab
    caps = {d: 80 for d in ("mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire")}
    res = session.forecast(STAY, 6.0, n=4, caps=caps, observed=True)
    schema.validate(res)
    assert res["tool"] == "forecast" and res["patient"]["kind"] == "held_out" and res["anchor"]["hours"] == 6.0
    out = res["outputs"]
    assert out["design"] == "R" and out["risks"] and len(out["incidence"]["hours"]) == 48
    r = out["risks"]["decompensation@1440"]
    assert r["estimator"] in ("e2", "share") and r["paper_oe"] == 1.77 and not r["calibrated"]
    assert res["provenance"]["gate"] == "PASS" and res["provenance"]["gate_not_in_v1"]
    assert res["evidence"]["refs"] and any("recalibration" in c for c in res["caveats"])
    again = session.forecast(STAY, 6.0, n=4, caps=caps, observed=True,
                             job_dir=Path(res["settings"]["job"]))            # resumed from the job's stages
    assert again["outputs"]["risks"].keys() == out["risks"].keys()
    json.dumps(res)                                                          # JSON-safe
    assert v.id("[EOS]") >= 0


def test_toolkit_copies_match_the_paper_scripts() -> None:
    sys.path.insert(0, str(R / "tools"))
    sys.path.insert(0, str(R / "tools" / "server"))
    import paper_a1
    import paper_c1
    import paper_whatif

    from icutwin.toolkit import labels, treatments

    assert treatments.TREATMENTS == paper_whatif.TREATMENTS and treatments.BINS == paper_whatif.BINS
    assert treatments.RESPONSES == paper_c1.RESPONSES and treatments.CAP == paper_c1.CAP
    if (R / "data" / "factory" / "stores" / "real_dev" / "channels.json").exists():
        from icutwin.factory import programs as fpr
        from icutwin.factory.store import Channels

        ch = Channels.load(R / "data" / "factory" / "stores" / "real_dev" / "channels.json")
        for p in fpr.catalog(ch):
            for d in ("L24", "L48", "R"):
                assert labels.applies(p, d) == paper_a1.applies(p, d)


CAPS = {d: 60 for d in ("mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire")}


@pytest.fixture(scope="module")
def probed(session, tmp_path_factory):
    """The session with stand-in Twin probes of the tiny model's width (b + w . [last, mean])."""
    import copy

    d = tmp_path_factory.mktemp("paper")
    width = 2 * session.teacher.model.config.hidden_size
    rng = np.random.default_rng(0)
    for design in ("L24", "L48", "R"):
        (d / "a1" / f"test_{design}").mkdir(parents=True)
        np.savez(d / "a1" / f"test_{design}" / "probe_models.npz",
                 **{k: rng.normal(0, 0.05, width + 1) for k in ("decompensation_at_1440", "icu_death_at_-1")})
    s = copy.copy(session)
    s.paths = copy.copy(session.paths)
    s.paths.paper = d
    return s


def test_whatif_end_to_end(session) -> None:
    from icutwin.toolkit import schema

    res = session.whatif(STAY, 6.0, "norepinephrine", n=3, caps=CAPS)
    schema.validate(res)
    o = res["outputs"]
    assert {"withhold", "withhold_b"} <= set(o["arms"])
    given = [a for a in o["arms"] if a.startswith("give_")]
    assert given or o["dropped"]                                             # given, or rejected with a reason
    m = o["responses"]["map"]
    assert m["pre_specified"] and m["evidence"] == "reproduced" and m["paper"]["observational"] == -3.3
    assert o["responses"]["glucose"]["evidence"] == "exploratory"           # not norepinephrine's response
    assert res["evidence"]["framing"].startswith("Observationally faithful")


def test_explain_edit_and_knockout(probed) -> None:
    from icutwin.toolkit import schema

    res = probed.explain(STAY, 6.0, "decompensation@1440")
    schema.validate(res)
    g = res["outputs"]["groups"]
    assert g and all(np.isfinite(x["edit"]) for x in g) and any(x["knockout"] is not None for x in g)
    assert abs(g[0]["edit"]) >= abs(g[-1]["edit"])                          # ranked by the size of the edit
    assert 0 < res["outputs"]["probe"]["risk"] < 1 and res["outputs"]["agreement_edit_knockout"]["groups"] >= 3
    with pytest.raises(KeyError):
        probed.explain(STAY, 6.0, "no_such_program@60")


def test_similar_index_and_query(session, tmp_path: Path) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    from icutwin.toolkit import similar

    t = pq.read_table(T / "mimic" / "held_out.parquet", columns=["stay_id"]).slice(0, 40).column(0).to_pylist()
    v = session.vocab
    rows = []
    for sid in t:
        p = session.patient(f"split:mimic/held_out/{sid}")
        if p.los_hours(v) > 7 and len(p.at(v, 6.0).prompt) < 4000:
            rows.append({"anchor": f"{sid}@6", "twin": sid, "dataset": "mimic", "hours": 6.0, "weight": 1.0})
        if len(rows) == 12:
            break
    (tmp_path / "anchors").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "anchors" / "anchors.parquet")
    lab = pd.DataFrame([{"anchor": r["anchor"], "program": prog, "label": float(i % 2)}
                        for i, r in enumerate(rows) for prog in ("icu_death@-1", "imv_onset@1440")])
    lab.to_parquet(tmp_path / "labels.parquet")
    similar.build_index(session, "unit", tmp_path / "anchors", tmp_path / "labels.parquet")
    res = session.similar(f"split:mimic/held_out/{rows[0]['twin']}", 6.0, index="unit", k=6, min_cell=1)
    o = res["outputs"]
    assert o["excluded_own_stay"] == 1 and o["k"] == 6
    r = o["rates"]["icu_death@-1"]
    assert r["neighbours_with_label"] == 6 and (r["rate"] is None or 0 <= r["rate"] <= 1)
    strict = session.similar(f"split:mimic/held_out/{rows[0]['twin']}", 6.0, index="unit", k=6, min_cell=10)
    assert strict["outputs"]["rates"]["icu_death@-1"].get("suppressed")       # small cells never shown


def test_cohort_select_and_export(session) -> None:
    from icutwin.toolkit import schema

    if not (session.paths.bank_store / "manifest.json").exists():
        pytest.skip("the bank's store")
    res = session.cohort({"n": 4, "mode": "select", "datasets": {"mimic": 0.5, "eicu": 0.5},
                          "filters": {"vasopressor": "yes"}, "export": ["clif", "meds"], "seed": 0})
    schema.validate(res)
    o = res["outputs"]
    assert o["n"] == 4 and o["datasets"] == {"mimic": 2, "eicu": 2} and o["conformance"]["vasopressor"] == 1.0
    exp = Path(o["export_dir"])
    assert list((exp / "clif").glob("clif_*.parquet")) and (exp / "meds" / "data.parquet").exists()
    assert "INTERNAL" in (exp / "README.txt").read_text()


def test_trial_designer_statistics() -> None:
    from icutwin.toolkit import trial

    assert abs(trial.n_two_proportions(0.2, 0.8) - 1447.0) < 1.0
    rng = np.random.default_rng(1)
    x = rng.normal(size=4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-(-1.5 + 1.5 * x)))).astype(float)
    null = trial.simulate(y, x, 400, 1.0, 2000, 0.05, rng)
    assert 0.03 <= null["unadjusted"] <= 0.07 and 0.03 <= null["adjusted"] <= 0.07     # type I error
    alt = trial.simulate(y, x, 400, 0.6, 1000, 0.05, rng)
    assert alt["adjusted"] > alt["unadjusted"]                               # a prognostic covariate adds power


def test_trial_designer_on_kept_predictions(session) -> None:
    from icutwin.toolkit import schema

    if not (session.paths.paper / "a1" / "test_L24" / "predictions.parquet").exists():
        pytest.skip("the paper's kept predictions")
    res = session.trial({"design": "L24", "endpoint": "icu_death@-1", "simulations": 200,
                         "scores": ["twin", "sofa", "lgbm_pooled"], "population": {"datasets": ["mimic", "eicu"]}})
    schema.validate(res)
    o = res["outputs"]
    assert o["population"]["anchors"] > 500 and set(o["population"]["datasets"]) == {"mimic", "eicu"}
    tw = o["scores"]["twin"]
    assert 0.5 < tw["auroc"] < 1 and len(tw["enrichment"]) == 4
    e = {x["fraction"]: x for x in tw["enrichment"]}
    assert e[0.25]["event_rate"] > e[1.0]["event_rate"] and e[0.25]["n_total"] < e[1.0]["n_total"]
    assert 0.0 <= tw["adjustment"]["type1_adjusted"] <= 0.15


def test_cards_are_self_contained(session, probed) -> None:
    from icutwin.toolkit import card

    caps = CAPS
    results = [session.forecast(STAY, 6.0, n=4, caps=caps), session.whatif(STAY, 6.0, "norepinephrine", n=3, caps=caps),
               probed.explain(STAY, 6.0, "decompensation@1440")]
    for r in results:
        h = card.render(r, session)
        assert h.startswith("<!doctype html>") and "http://" not in h and "https://" not in h and "<script src" not in h
        assert "real stay: individual data" in h
    assert "<svg" in card.render(results[0], session)


def test_teaching_case_and_drill(probed, tmp_path: Path) -> None:
    if not (probed.paths.bank_store / "manifest.json").exists():
        pytest.skip("the bank's store")
    res = probed.teach({"filters": {"vasopressor": "yes"}, "datasets": ["mimic", "eicu", "hirid"]},
                       treatment="norepinephrine", cases=1, n=3, pool=2, pick="random", out_dir=tmp_path / "t")
    files = res["outputs"]["cases"][0]["files"]
    learner = (tmp_path / "t" / files[1]).read_text(encoding="utf-8")
    teacher = (tmp_path / "t" / files[2]).read_text(encoding="utf-8")
    assert "class=\"hidden\"" in learner and "Reveal" in learner and "class=\"hidden\"" not in teacher
    assert "synthetic patient" in learner and "https://" not in learner + teacher
    assert "Discussion prompts" in teacher and (tmp_path / "t" / "drill.html").exists()
    case = json.loads((tmp_path / "t" / files[0]).read_text(encoding="utf-8"))
    assert case["patient"].startswith("bank:") and case["forecast"]["patient"]["kind"] == "twin"


def test_v3_statistics() -> None:
    """V3's membership statistics: no signal stays PASS, a strong signal FAILs; extraction finds an exact copy."""
    sys.path.insert(0, str(R / "tools" / "server"))
    import toolkit_v3

    rng = np.random.default_rng(0)
    same = toolkit_v3.mia(rng.normal(size=3000), rng.normal(size=3000), rng, boot=200)
    assert same["status"] == "PASS" and abs(same["auc"] - 0.5) < 0.03
    leak = toolkit_v3.mia(rng.normal(1.0, 1, 3000), rng.normal(size=3000), rng, boot=200)
    assert leak["status"] == "FAIL" and leak["auc"] > 0.7
    if not (T / "vocab.json").exists():
        return
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    ev = [i for i, m in enumerate(v.meta) if m.get("cls") == "event"][:40]
    fut = ev[:36]
    rows = [{"id": "m", "future": fut}, {"id": "n", "future": fut}]
    conts = {"m": [list(fut), ev[4:40]], "n": [ev[1:37]]}
    ex = toolkit_v3.extraction(rows, conts, v, given=False)
    assert ex["m"]["exact32"] and not ex["n"]["exact32"] and ex["m"]["jaccard"] == 1.0 and ex["n"]["jaccard"] < 1.0


def test_cohort_sampling_under_conditions(session) -> None:
    from icutwin.toolkit import schema

    if not (session.paths.bank_store / "manifest.json").exists():
        pytest.skip("the bank's store")
    res = session.cohort({"n": 2, "mode": "sample", "datasets": {"mimic": 1.0}, "filters": {"sex": "female"},
                          "export": [], "seed": 1, "max_new": 80}, max_oversample=1)
    schema.validate(res)
    o = res["outputs"]
    assert o["drawn"] == 2 and o["n"] <= 2 and o["conformance"]["sex"] in (None, 1.0)
    assert res["provenance"]["gate"] == "PASS"


def test_bank_exemplars_figure_and_blinded_card(probed, tmp_path: Path) -> None:
    from icutwin.toolkit import card, figures, similar

    if not probed.paths.bank.exists():
        pytest.skip("the bank")
    similar.build_bank_index(probed, "bank_unit", hours=(6.0,), max_twins=6)
    twin = next(iter(np.load(similar.index_path(probed, "bank_unit"))["twin"]))
    p = probed.patient(f"bank:{twin}")
    a = p.at(probed.vocab, 6.0)
    ex = similar.exemplars(probed, probed.teacher.states([a.prompt])[1][0] / 1.0, "bank_unit", 3, 6.0)
    assert ex and ex[0]["twin"] == str(twin) and len({e["twin"] for e in ex}) == len(ex)
    f = probed.forecast(p, 6.0, n=4, caps=CAPS)
    w = probed.whatif(p, 6.0, "imv", n=3, caps=CAPS)
    e = probed.explain(p, 6.0, "decompensation@1440")
    paths = figures.case_figure(probed, f, w, e, tmp_path / "case")
    assert all(x.exists() and x.stat().st_size > 1000 for x in paths) and {x.suffix for x in paths} == {".png", ".pdf"}
    with pytest.raises(ValueError):
        figures.case_figure(probed, probed.forecast(STAY, 6.0, n=4, caps=CAPS), path=tmp_path / "real")
    b = card.blinded(probed, p, 6.0)
    assert "<svg" in b and twin not in b and "synthetic" not in b and "mimic" not in b.lower()


def test_case_sheet_parity_and_packet_blinding(session) -> None:
    """The clinician study's case sheet renders real and synthetic stays alike, with no identifier; the packet holds
    no plaintext card and its checksum is the page's (the value a browser computed for this export)."""
    import pyarrow.parquet as pq

    from icutwin.factory import twins
    from icutwin.toolkit import casesheet, rating

    v = session.vocab
    real = session.patient(STAY)
    if not session.paths.bank.exists():
        pytest.skip("the bank")
    tw = pq.read_table(sorted(session.paths.bank.glob("part*/mimic_*.parquet"))[0], columns=["twin", "ids"]).slice(0, 1)
    tw = tw.to_pylist()[0]
    sheets = [casesheet.sheet(twins.decode_features(ids, v, sid, 0, None), 24.0, start=0.0, end=True)
              for ids, sid in ((real.ids, real.stay_id), (tw["ids"], tw["twin"]))]
    for h in sheets:
        assert "Vital signs" in h and "Laboratory values" in h and "hours since ICU admission" in h
        assert "mimic" not in h.lower() and "eicu" not in h.lower() and "http" not in h
    assert real.stay_id.split(":")[1] not in sheets[0] and tw["twin"] not in sheets[1]
    key = rating.card_key("R01", "salt")
    card = "<h2>forecast: secret 37%</h2>"
    page = rating.build_packet({"title": "t", "intro": "<p>i</p>", "packet": "R01", "build": "b", "cardkey": key,
                                "modules": [{"id": "C", "title": "C", "kind": "reader", "intro": "", "items": [
                                    {"id": "c1", "html": sheets[0], "outcomes": [{"key": "icu_death", "label": "d"}],
                                     "card": rating.obfuscate(card, key)}]}]})
    assert "secret 37%" not in page and "http://" not in page and "https://" not in page and "<script src" not in page
    assert rating.deobfuscate(rating.obfuscate("é ✓ 37%", key), key) == "é ✓ 37%"
    browser = json.loads(  # a demo packet's export as Chrome wrote it; the page computed check 83c54c22
        '{"responses":{"T":[{"item":"t1","order":0,"call":"synthetic","confidence":4,"cue":"","shown":1791688438887,'
        '"answered":1791688468371}],"A":[{"item":"a1","order":0,"call":"real","confidence":3,"cue":"",'
        '"shown":1791688468373,"answered":1791688468374},{"item":"a2","order":1,"call":"real","confidence":3,'
        '"cue":"","shown":1791688468374,"answered":1791688468375}],"B2":[{"item":"b21","order":0,"relevance":3,'
        '"irrelevant":[true,false,false],"shown":1791688468376,"answered":1791688468377}],"B1":[{"item":"b11",'
        '"order":0,"plausibility":3,"direction":"unsure","shown":1791688468378,"answered":1791688468379}],'
        '"C":[{"item":"c1","order":0,"unaided":{"icu_death":10,"aki":11},"aided":{"icu_death":20,"aki":21},'
        '"changed":"yes","shown":1791688468379,"locked":1791688468380,"answered":1791688468381}]},'
        '"questionnaire":{"sus":[3,3,3,3,3,3,3,3,3,3],"trust":[3,3,3,3,3,3],"shown":1791688468382,'
        '"answered":1791688468383}}')
    assert rating.check(browser) == "83c54c22"


def test_logistic_recalibration_keeps_ranks() -> None:
    from icutwin.eval import bench
    from icutwin.toolkit import recalibration as rc

    rng = np.random.default_rng(3)
    p = rng.beta(0.5, 8, 20000)
    lg = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    y = (rng.random(len(p)) < 1 / (1 + np.exp(-(0.9 + 0.8 * lg)))).astype(float)
    m = rc.fit_logistic(y, p)
    assert m["kind"] == "logistic" and abs(m["a"] - 0.9) < 0.15 and abs(m["b"] - 0.8) < 0.08
    q, ok = rc.apply({"x": m}, "x", p)
    assert ok and np.all(np.diff(q[np.argsort(p)]) >= 0)                     # strictly increasing map
    assert abs(bench.auroc_fast(y, q) - bench.auroc_fast(y, p)) < 1e-9         # AUROC kept
    assert abs(np.mean(q) / np.mean(y) - 1) < 0.05                             # calibrated in the large
    iso = rc.fit(y, p)
    assert iso["kind"] == "isotonic" and rc.apply({"x": {k: v for k, v in iso.items() if k != "kind"}}, "x", 0.1)[1]
