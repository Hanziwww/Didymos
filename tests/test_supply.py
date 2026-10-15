"""The twin supply (Phase F, S1b; ``generate.supply``): the plan's cells and shares, the balance-heuristic weights
(which recover the natural marginals when the cells draw as the engine does; a base-only plan gives the bank's
weights), and the producer and store append on real held-out token sequences standing in for twins (no model, no
synthetic data): shards with cells, requests and no goal block, resumable and partitioned; a segment appended
byte-reproducibly; weights recorded for the store's segments and applied when it is opened with them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from icutwin.generate import bank, supply

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "full_r4"
SEXES = ("female", "male")
AGES = ("20-25", "50-55", "70-75", "85-90")
UNITS = ("medical_icu", "surgical_icu")


def _population(seed: int = 0, n: int = 6000) -> list[dict]:
    """Synthetic stays: two datasets (eICU with three sites), strata, and goal labels that depend on the stratum."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        d = "eicu" if rng.random() < 0.7 else "mimic"
        site = f"eicu_{rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])}" if d == "eicu" else "mimic"
        a = {"sex": str(rng.choice(SEXES)), "age": str(rng.choice(AGES, p=[0.2, 0.3, 0.35, 0.15])),
             "unit_type": str(rng.choice(UNITS, p=[0.7, 0.3])), "admission_type": "ed",
             "adm_dx_group": "medical_sepsis"}
        a["age_band"] = supply.coarse_age(a["age"])
        old = a["age_band"] in ("65-80", "80+")
        dead = rng.random() < (0.25 if old else 0.06) * (1.5 if d == "mimic" else 1.0)
        rrt = rng.random() < (0.12 if a["unit_type"] == "medical_icu" else 0.04)
        out.append({"id": i, "dataset": d, "site": site, **a, "stratum": supply.stratum_key(a),
                    "goals": {"icu_outcome": "dead" if dead else "alive", "rrt": "yes" if rrt else "no"}})
    return out


def _marginals(pop: list[dict]) -> dict:
    m: dict = {"datasets": {}, "sites": {}, "strata": {}, "goals": {}}
    for x in pop:
        d = x["dataset"]
        m["datasets"][d] = m["datasets"].get(d, 0) + 1
        m["sites"][x["site"]] = m["sites"].get(x["site"], 0) + 1
        s = m["strata"].setdefault(d, {}).setdefault(x["stratum"], {"n": 0, "age": {}, "goals": {}})
        s["n"] += 1
        s["age"][x["age"]] = s["age"].get(x["age"], 0) + 1
        for f, lab in x["goals"].items():
            g = m["goals"].setdefault(d, {}).setdefault(f, {})
            g[lab] = g.get(lab, 0) + 1
            sg = s["goals"].setdefault(f, {})
            sg[lab] = sg.get(lab, 0) + 1
    return m


def _draw(pop, plan: supply.SupplyPlan, rng) -> list[dict]:
    """Twins as the engine draws them: base from the dataset; strata and sites with their fields fixed; a conditioned
    goal cell keeps the natural prefix and header (a stratum drawn from the dataset) and draws the rest given the goal;
    a filter cell keeps the natural draws with the goal (its quota of them)."""
    by: dict = {}
    for x in pop:
        for key in (("d", x["dataset"]), ("s", x["dataset"], x["stratum"], x["age"]), ("h", x["site"])):
            by.setdefault(key, []).append(x)
    out = []
    reqs = [(k, r) for k, r in plan.requests() if plan.cells[k].mode != "filter"]
    for k, c in enumerate(plan.cells):
        if c.mode == "filter":
            f, (lab,) = c.goal
            pick = [x for x in by[("d", c.dataset)] if x["goals"][f] == lab]
            out += [{**pick[rng.integers(len(pick))], "cell": c.name} for _ in range(c.quota)]
    for k, r in reqs:
        c = plan.cells[k]
        if c.mechanism == "base":
            pick = by[("d", c.dataset)]
        elif c.mechanism == "strata":
            pick = by[("s", c.dataset, c.stratum, r["prefix"]["age"])]
        elif c.mechanism == "site":
            pick = by[("h", r["site"])]
        else:
            f, lab = next(iter(r["goals"].items()))
            for _ in range(1000):
                x0 = by[("d", c.dataset)][rng.integers(len(by[("d", c.dataset)]))]
                pick = [x for x in by[("s", c.dataset, x0["stratum"], x0["age"])] if x["goals"][f] == lab]
                if pick:
                    break
        x = pick[rng.integers(len(pick))]
        out.append({**x, "cell": c.name})
    return out


def test_plan_cells_and_shares() -> None:
    pop = _population()
    m = _marginals(pop)
    plan = supply.SupplyPlan.build(m, 1000, name="t", strata_min=20, site_min=100, goal_min=20,
                                   goal_cells=(("icu_outcome", ("dead",)), ("rrt", ("yes",))))
    mech = plan.mechanisms()
    assert sum(mech.values()) == 1000
    assert len(plan.requests()) == sum(p["n"] for c in plan.cells for p in c.parts) > 1000     # filter cells' draws
    for k, share in supply.SHARES.items():
        assert abs(mech[k] / 1000 - share) <= 0.002, (k, mech)
    for c in plan.cells:
        if c.mode == "filter":                                              # about quota / rate draws
            assert c.goal[0] in supply.EVENT_GOALS and sum(p["n"] for p in c.parts) > c.quota > 0
            assert not any(p["goals"] for p in c.parts)
        else:
            assert sum(p["n"] for p in c.parts) == c.quota > 0
        if c.mechanism == "strata":
            sex, band, unit, _adm, _dx = c.stratum.split("|")
            assert all(p["prefix"]["sex"] == sex and supply.coarse_age(p["prefix"]["age"]) == band
                       and p["header"]["unit_type"] == unit for p in c.parts)
        if c.mechanism == "site":
            assert c.dataset == "eicu" and c.site in ("eicu_1", "eicu_2", "eicu_3") and m["sites"][c.site] >= 100
    again = supply.SupplyPlan.from_json(json.loads(json.dumps(plan.to_json())))
    assert again.requests() == plan.requests()
    small = supply.SupplyPlan.build(m, 1000, max_strata=2, site_min=10**9, goal_cells=())
    assert small.mechanisms().get("site", 0) == 0 and small.mechanisms().get("goal", 0) == 0
    assert sum(small.mechanisms().values()) == 1000                            # their budget goes to the base
    assert len({c.stratum for c in small.cells if c.mechanism == "strata" and c.dataset == "eicu"}) <= 2


def test_weights_recover_the_natural_marginals() -> None:
    pop = _population()
    m = _marginals(pop)
    plan = supply.SupplyPlan.build(m, 20000, name="t", strata_min=20, site_min=100, goal_min=20,
                                   goal_cells=(("icu_outcome", ("dead",)), ("rrt", ("yes",))))
    twins = _draw(pop, plan, np.random.default_rng(1))
    counts: dict = {}
    for x in twins:
        counts[x["cell"]] = counts.get(x["cell"], 0) + 1
    w = plan.weights(twins, m, counts=counts, shrink=0.0)
    assert np.isclose(w.mean(), 1.0) and (w > 0).all()

    def share(pred, weights=None):
        y = np.array([pred(x) for x in twins], dtype=float)
        return float(np.average(y, weights=weights))

    def truth(pred):
        return float(np.mean([pred(x) for x in pop]))

    checks = {"eicu": lambda x: x["dataset"] == "eicu",
              "site_3": lambda x: x["site"] == "eicu_3",
              "dead": lambda x: x["goals"]["icu_outcome"] == "dead",
              "dead_young": lambda x: x["goals"]["icu_outcome"] == "dead" and x["age_band"] == "18-45",
              "rrt": lambda x: x["goals"]["rrt"] == "yes",
              "old": lambda x: x["age_band"] == "80+"}
    gaps = {k: abs(share(f, w) - truth(f)) for k, f in checks.items()}
    raw = {k: abs(share(f) - truth(f)) for k, f in checks.items()}
    assert max(gaps.values()) < 0.012, gaps
    assert raw["dead"] > 0.03 and raw["rrt"] > 0.02, raw                  # the enrichment the weights undo
    mimic = [x for x in twins if x["dataset"] == "mimic"]                   # within a dataset (the factory's sites)
    wm = w[[x["dataset"] == "mimic" for x in twins]]
    dead_m = float(np.average([x["goals"]["icu_outcome"] == "dead" for x in mimic], weights=wm))
    assert abs(dead_m - np.mean([x["goals"]["icu_outcome"] == "dead" for x in pop if x["dataset"] == "mimic"])) < 0.015


def test_base_only_plan_gives_the_bank_weights() -> None:
    pop = _population()
    m = _marginals(pop)
    plan = supply.SupplyPlan.build(m, 900, shares={"base": 1.0})
    ref = bank.GenerationPlan.from_mix(m["datasets"], 900, "flattened", 0.5)
    assert plan.counts() == {f"base/{d}": q for d, q in ref.quotas.items()}
    attrs = [{"dataset": d, "site": None, "stratum": "", "goals": {}} for d in ("eicu", "mimic")]
    w = plan.weights(attrs, m)
    assert np.isclose(w[0] / w[1], ref.weights["eicu"] / ref.weights["mimic"], rtol=2e-3)   # quotas: whole twins


# -- the producer and the store append ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tokens():
    if not (T / "vocab.json").exists() or not (T / "mimic" / "held_out.parquet").exists():
        pytest.skip("the full r4 tokens")
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    stays = {ds: [r["ids"] for r in pq.read_table(T / ds / "held_out.parquet", columns=["ids"]).slice(0, 40)
                  .to_pylist()] for ds in ("mimic", "eicu")}
    return v, stays


def _stand_in(v, stays):
    """A held-out stay by seed; a request with goals gets their block after its header, as the engine writes it."""
    from icutwin.model.data import layout

    lay = layout(v)
    head = 1 + lay.n_prefix + lay.n_header

    def sample(requests, seeds):
        out = []
        for r, s in zip(requests, seeds, strict=True):
            ids = list(stays[r.dataset][(s * 7919) % len(stays[r.dataset])])
            if r.goals:
                block = [v.id("[GOAL]"), *(v.id(f"GOAL//{f}//{lab}") for f, lab in r.goals.items()), v.id("[/GOAL]")]
                ids = ids[:head] + block + ids[head:]
            out.append(ids)
        return out
    return sample


def _plan(v) -> supply.SupplyPlan:
    m = supply.marginals(T, ["mimic", "eicu"], "held_out", v=v)
    return supply.SupplyPlan.build(m, 48, name="t", strata_min=1, max_strata=2, site_min=1, goal_min=1,
                                   goal_cells=(("icu_outcome", ("dead",)), ("kdigo_max", ("3+",))))


def test_produce_shards_resume_and_parts(tokens, tmp_path: Path) -> None:
    from icutwin.generate.engine import Request

    v, stays = tokens
    plan = _plan(v)
    event = np.array([m.get("cls") == "event" for m in v.meta])
    kw = {"seed": 2, "shard_size": 10, "event": event, "gate": {"gate": "PASS"}, "model": {"name": "stand-in"},
          "v": v, "request": lambda r: Request(**r)}
    calls = []
    sample = _stand_in(v, stays)

    def counted(reqs, seeds):
        calls.append(len(seeds))
        return sample(reqs, seeds)

    man = supply.produce(counted, plan, tmp_path / "one", **kw)
    assert man["complete"] and sum(s["drawn"] for s in man["shards"]) == len(plan.requests())
    rows = [r for s in man["shards"] for r in pq.read_table(tmp_path / "one" / f"{s['name']}.parquet").to_pylist()]
    assert len(rows) == sum(s["kept"] for s in man["shards"]) > 0
    goal_tok = {v.id("[GOAL]"), v.id("[/GOAL]")}
    for r in rows:
        assert not goal_tok & set(r["ids"])                                  # the twin keeps its own tokens
        c = json.loads(r["conditions"])
        cell = next(x for x in plan.cells if x.name == r["cell"])
        assert c["dataset"] == r["dataset"] == cell.dataset
        if c.get("filter"):                                                # kept on its event
            assert r["goal_met"] is True and cell.mode == "filter"
        else:
            assert r["goal_met"] is None                                    # value goals: on derived concepts
    n = len(calls)
    supply.produce(counted, plan, tmp_path / "one", **kw)                     # resumes: nothing left to draw
    assert len(calls) == n
    names = {s["name"] for s in man["shards"]}
    split = set()
    for p in range(2):
        mp = supply.produce(sample, plan, tmp_path / "two" / f"part{p}", part=p, parts=2, **kw)
        split |= {s["name"] for s in mp["shards"]}
        assert all(json.loads((tmp_path / "two" / f"part{p}" / "manifest.json").read_text())["part"] == [p, 2]
                   for _ in [0])
    assert split == names
    with pytest.raises(ValueError):
        supply.produce(sample, plan, tmp_path / "one", **{**kw, "seed": 3})


def test_supply_segment_and_weights(tokens, tmp_path: Path) -> None:
    import sys

    from icutwin.factory import store as fst
    from icutwin.generate.engine import Request

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import factory_store as cli

    v, stays = tokens
    plan = _plan(v)
    plan.save(tmp_path / "plan.json")
    (tmp_path / "marg.json").write_text(json.dumps(supply.marginals(T, ["mimic", "eicu"], "held_out", v=v)))
    event = np.array([m.get("cls") == "event" for m in v.meta])
    supply.produce(_stand_in(v, stays), plan, tmp_path / "run" / "part0", seed=0, shard_size=16, event=event,
                   gate={"gate": "PASS"}, model={}, v=v, request=lambda r: Request(**r))
    for out in ("a", "b"):
        cli.main(["supply", "--tokens", str(T), "--shards", str(tmp_path / "run"), "--out", str(tmp_path / out),
                  "--segment", "seg_0000", "--workers", "1", "--chunk", "20"])
    ma, mb = (json.loads((tmp_path / x / "manifest.json").read_text()) for x in ("a", "b"))
    assert ma["segments"][0]["files"] == mb["segments"][0]["files"]           # byte-reproducible
    st = fst.Store(tmp_path / "a")
    assert not st.weighted and {"cell", "conditions"} <= set(st.stays.columns)
    cli.main(["weights", "--out", str(tmp_path / "a"), "--plan", str(tmp_path / "plan.json"),
              "--marginals", str(tmp_path / "marg.json")])
    st = fst.Store(tmp_path / "a")
    assert st.weighted and np.isfinite(st.stays["weight"]).all()
    attrs = supply.store_attributes(st)
    met = supply.goal_check(attrs)
    goal = [i for i, a in enumerate(attrs) if json.loads(a["conditions"]).get("goals")]
    assert goal and all((st.stays["split"].iat[i] == "rejected") == (met[i] is False) for i in goal)
    ok = st.stays["split"] != "rejected"
    assert np.isclose(st.stays.loc[ok, "weight"].mean(), 1.0)
    assert not fst.Store(tmp_path / "a", segments=["seg_0000"]).stays.empty


def test_request_fixes_prefix_fields(tokens) -> None:
    from icutwin.generate.engine import Request, conditions

    v, _stays = tokens
    c = conditions(v, Request(dataset="mimic", prefix={"sex": "female", "age": "70-75"},
                              header={"unit_type": "medical_icu"}, goals={"rrt": "yes"}))
    assert c.prefix == {0: v.id("PREFIX//dataset//mimic"), 2: v.id("PREFIX//sex//female"),
                        3: v.id("PREFIX//age//70-75")}
    assert c.header == {v.id("HEADER//unit_type//medical_icu")} and c.goal == {v.id("GOAL//rrt//yes")}
    assert c.require                                                           # the RRT goal's obligation
