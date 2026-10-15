"""Trajectory episodes (S3, ``factory.episodes``, ``factory.trajectories``, the episode contract) on a store built from
real held-out token sequences standing in for twins (skipped without them): recipes are a function of their index,
every materialized episode passes the contract, a query never sees what is not yet available (events, intervals,
header fields documented later), anonymization is consistent, and the family groups follow the configured mix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from icutwin.factory import contracts, episodes, store as fst, trajectories as ftr

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "full_r4"


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    if not (T / "vocab.json").exists() or not (T / "eicu" / "held_out.parquet").exists():
        pytest.skip("the full r4 tokens")
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    rows = []
    for ds in ("mimic", "eicu", "aumc"):
        t = pq.read_table(T / ds / "held_out.parquet", columns=["stay_id", "ids"]).slice(0, 60).to_pylist()
        rows += [{"stay_id": r["stay_id"], "ids": r["ids"], "split": "train"} for r in t]
    root = tmp_path_factory.mktemp("ep")
    fst.build(rows, T / "vocab.json", root / "s", "seg_0000", {"kind": "test"}, workers=1, chunk=90)
    st = fst.Store(root / "s")
    st.channels.stats = fst.channel_stats(st)
    return v, st


def test_recipes_are_a_function_of_their_index(setup) -> None:
    v, st = setup
    a, b = episodes.Sampler(st, v, seed=7), episodes.Sampler(st, v, seed=7)
    ra = [a.recipe(i) for i in (3, 1, 2)]
    rb = [b.recipe(i) for i in (1, 2, 3)]
    assert ra[0] == rb[2] and ra[1] == rb[0] and ra[2] == rb[1]
    assert episodes.Sampler(st, v, seed=8).recipe(1) != rb[0]


def test_contract_fuzz(setup) -> None:
    v, st = setup
    s = episodes.Sampler(st, v, seed=0)
    kinds, anon, ctx = set(), 0, []
    for i in range(80):
        ep = s.episode(i)
        fails = contracts.episode_failures(ep)
        assert not fails, (i, fails)
        kinds.add(ep["task"]["label"])
        anon += ep["task"]["anonymized"]
        ctx.append(len(ep["recipe"]["context"]))
    assert {"binary", "regression"} <= kinds and 0 < anon < 80 and max(ctx) > 4


def test_queries_see_only_what_is_available(setup) -> None:
    v, st = setup
    rd = ftr.Reader(st, v)
    i = next(k for k, r in enumerate(st.stays.to_dict("records"))
             if r["dataset"] == "eicu" and r.get("adm_dx_group") not in (None, "UNK") and r["los_min"] > 1600)
    keys = {j: k for k, j in rd.keys.items()}
    early, late = rd.trajectory(i, until=600), rd.trajectory(i, until=1500)
    has = lambda t: any(keys[int(k)].startswith("adm_dx_group=") for k in t["st_k"])  # noqa: E731
    assert not has(early) and has(late)                 # eICU documents the diagnosis group within 24 h
    assert early["ev_t"].max() <= 600 and early["iv_e"].max(initial=0) <= 600 and "end_t" not in early
    full = rd.trajectory(i)
    assert full["end_t"] == st.los[i] and len(full["ev_t"]) >= len(late["ev_t"])
    thin = rd.trajectory(i, budget=200)
    assert len(thin["ev_t"]) <= 200 and (thin["ev_f"] & ftr.THINNED).any()


def test_anonymized_episodes(setup) -> None:
    v, st = setup
    s = episodes.Sampler(st, v, seed=0)
    for i in range(200):
        r = s.recipe(i)
        if r["anonymize"] is not None:
            ep = s.materialize(r)
            n = ep["channels"]["n"]
            c = np.concatenate([ep["context"]["ev_c"], ep["query"]["ev_c"]])
            assert c.max() < n and len(ep["channels"]["global_ids"]) == n
            np.testing.assert_array_equal(ep["query"]["ev_v"], ep["query"]["ev_z"])
            if ep["task"]["descriptor"]:
                chans = [t["channel"] for t in ep["task"]["descriptor"]["terms"] if "channel" in t]
                assert all(x is None or x < n for x in chans)
            return
    pytest.fail("no anonymized episode in 200 recipes")


def test_case_odds() -> None:
    assert episodes.case_odds(500, 10, 16, 0) == 1.0                  # natural
    assert episodes.case_odds(20, 5, 20, 4) == 1.0                    # every candidate taken
    assert abs(episodes.case_odds(500, 10, 16, 4) - (0.4 + 0.6 * 12 / 496) / (12 / 496)) < 1e-9
    assert episodes.case_odds(10, 4, 4, 4) == 1000.0                  # only cases: capped


def test_context_selection_is_declared(setup) -> None:
    """With the descriptor shown, a natural context of an event task is taken as drawn (no case required); a hidden
    task's natural context shows a case; an outcome-stratified one declares the odds of a case's selection and the
    cases it selected; the task block carries the selection, and an as-drawn context without a case materializes."""
    v, st = setup
    s = episodes.Sampler(st, v, seed=3)
    drawn = 0
    for i in range(160):
        r = s.recipe(i)
        prog = episodes._program(r["program"])
        event = prog.label in ("binary", "hosp_death", "tte", "multihorizon")
        if r["as_drawn"]:
            drawn += 1
            assert r["descriptor"] and event and r["policy"] in ("random", "retrieved")
            assert r["case_odds"] == 1.0 and r["min_cases"] == 0
        elif r["policy"] == "stratified":
            assert r["case_odds"] >= 1.0 and r["min_cases"] >= 1
        elif r["context"] and event:                   # a hidden task's context, or a tilted site's: cases needed
            assert r["case_odds"] == 1.0 and r["min_cases"] >= 1
        if i < 40 or (r["as_drawn"] and r["filter"].get("one_class")):
            ep = s.materialize(r)
            assert ep["task"]["selection"] == {"policy": r["policy"], "case_odds": r["case_odds"],
                                               "min_cases": r["min_cases"]}
            assert not contracts.episode_failures(ep), contracts.episode_failures(ep)
    assert drawn > 0


def test_family_mix(setup) -> None:
    v, st = setup
    s = episodes.Sampler(st, v, seed=1)
    fams = [s.recipe(i)["program"]["family"] for i in range(300)]
    groups = {g: sum(f in x["families"] for f in fams) / len(fams) for g, x in s.groups.items()}
    share = {g: x["share"] for g, x in s.groups.items()}
    total = sum(share.values())
    for g in groups:
        assert abs(groups[g] - share[g] / total) < 0.08, (g, groups[g], share[g] / total)


def test_own_labels_never_show_the_target(setup) -> None:
    v, st = setup
    s = episodes.Sampler(st, v, seed=5)
    for i in range(40):
        ep = s.episode(i)
        q, r = ep["query"], ep["recipe"]
        oo = q["own_lab_off"]
        for k, rq in enumerate(r["queries"]):
            ot = q["own_lab_t"][oo[k]:oo[k + 1]]
            tr = q["own_lab_t_resolved"][oo[k]:oo[k + 1]]
            for tk in rq["times"]:
                visible = ot[tr <= tk]
                assert (visible < tk).all()            # the label at t_k itself is never visible at t_k


def test_forecast_and_multihorizon_formats(setup) -> None:
    from icutwin.factory import pool, programs as fpr

    v, st = setup
    progs = pool.fc_candidates(st.channels, 12, 0) + pool.mh_candidates(fpr.catalog(st.channels), 12, 0)
    s = episodes.Sampler(st, v, stage=2, seed=2, programs=progs)
    seen = set()
    for i in range(40):
        ep = s.episode(i)
        assert not contracts.episode_failures(ep), contracts.episode_failures(ep)
        lab = ep["task"]["label"]
        seen.add(lab)
        q = ep["query"]
        if lab == "forecast":                # phase_f.3: the context carries its forecast labels like any other type
            c = ep["context"]
            assert q["lab_yf"].shape[0] == len(q["lab_t"]) and q["lab_mf"].any()
            assert c["lab_yf"].shape == (len(c["lab_t"]), *q["lab_yf"].shape[1:])
            assert len(c["lab_t"]) > 0 or not ep["recipe"]["context"]
            tg = ep["recipe"]["program"]["spec"]["target"]
            assert q["lab_yf"].shape[2] == len(tg[1])
        if lab == "multihorizon":
            assert q["lab_yh"].shape == (len(q["lab_t"]), len(ep["task"]["bands"]))
    assert seen == {"forecast", "multihorizon"}


def test_site_variants_and_realism(setup) -> None:
    v, st = setup
    s = episodes.Sampler(st, v, seed=11)
    kinds, losses = {}, []
    for i in range(120):
        r = s.recipe(i)
        k = r["site_variant"]["kind"]
        kinds[k] = kinds.get(k, 0) + 1
        if r["difficulty_loss"] is not None:
            losses.append(r["difficulty_loss"])
            assert r["difficulty_loss"] <= s.cfg["difficulty"]["cap"] + 1e-9
        if k == "cross_site":
            ids = st.stays.set_index("stay_id")["dataset"]
            assert {ids[x] for x in r["context"]} <= {r["site"]}
            assert {ids[q["stay"]] for q in r["queries"]} <= {r["query_site"]}
        if i < 30:
            ep = s.materialize(r)
            assert not contracts.episode_failures(ep), contracts.episode_failures(ep)
    assert set(kinds) <= {"plain", "virtual", "label_shift", "cross_site"} and kinds.get("plain", 0) > 20


def test_context_banks_and_full_trajectory_queries(setup) -> None:
    """A context bank holds complete trajectories with the program's whole label timeline and no queries; a
    full-trajectory query set queries each stay at every cadence time and has no context; both pass the contract."""
    from icutwin.factory import programs as fpr

    v, st = setup
    s = episodes.Sampler(st, v, seed=0)
    prog = next(p for p in fpr.catalog(st.channels) if p.label == "binary" and p.name.startswith("tachycardia"))
    site = str(s.site_names[0])
    bank = s.context_bank(prog, ("rolling", 60), site, 12)
    ctx = bank["context"]
    assert len(bank["recipe"]["context"]) == len(ctx["ev_off"]) - 1 > 2 and not bank["recipe"]["queries"]
    rows = s.cadence_rows(("rolling", 60), [s.index[x] for x in bank["recipe"]["context"]])
    assert len(ctx["lab_t"]) == len(rows)                                   # every cadence time of each stay
    assert not contracts.episode_failures(bank)
    stays = list(s.by_site[site][:5])
    q = s.full_queries(prog, ("rolling", 60), stays)
    assert len(q["context"]["ev_off"]) == 1 and len(q["recipe"]["queries"]) == len(stays)
    assert len(q["query"]["lab_t"]) == len(s.cadence_rows(("rolling", 60), stays))
    assert not contracts.episode_failures(q)
