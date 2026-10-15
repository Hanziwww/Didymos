"""Constrained decoding (S7): a random-initialized model, whose logits are near uniform, samples only sequences the
grammar accepts when the automaton masks its logits (validity rests on the masks alone)."""

from __future__ import annotations

import numpy as np
import pytest
from test_tokenization import SPLITS, _write_meds

from icutwin.generate.constrain import GrammarProcessor
from icutwin.tokenization.fsm import Automaton, accepts, compile_tables, mask_tables
from icutwin.tokenization.parser import parse
from icutwin.tokenization.stays import iter_stays
from icutwin.tokenization.vocab import fit


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory):
    import torch
    from transformers import Qwen3ForCausalLM

    from icutwin.model.presets import qwen3_config

    meds = _write_meds(tmp_path_factory.mktemp("constrain") / "meds", SPLITS)
    v = fit(iter_stays(meds, "train"), min_subjects=1)
    t = compile_tables(v)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(qwen3_config("tiny", len(v.tokens), v.id("[PAD]"), v.id("[BOS]"), v.id("[EOS]"))).eval()
    return v, t, mask_tables(t, len(v.tokens)), model


def _sample(v, t, m, model, n: int, max_new: int, seed: int, cond=None, plan=None, support=None):
    import torch
    from transformers import GenerationConfig, LogitsProcessorList

    from icutwin.generate.constrain import NONE

    torch.manual_seed(seed)
    proc = GrammarProcessor(t, m, v.id("[PAD]"), max_length=1 + max_new, conditions=cond or NONE, goal_plan=plan,
                            support=support)
    cfg = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=max_new,
                           eos_token_id=v.id("[EOS]"), pad_token_id=v.id("[PAD]"))
    ids = torch.full((n, 1), v.id("[BOS]"), dtype=torch.long)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=cfg,
                             logits_processor=LogitsProcessorList([proc]))
    seqs = []
    for row in out.cpu().numpy():
        stop = np.flatnonzero(row == v.id("[EOS]"))
        seqs.append(row[:stop[0] + 1].tolist() if len(stop) else row.tolist())
    return seqs, proc


def test_random_model_samples_only_grammatical_stays(setup) -> None:
    v, t, m, model = setup
    seqs, proc = _sample(v, t, m, model, n=64, max_new=200, seed=0)
    assert proc.violations == 0
    ended = 0
    for s in seqs:
        if s[-1] == v.id("[EOS]"):
            ended += 1
            assert accepts(t, s)[0], accepts(t, s)
            parse(s, v)                  # the independent parser (raises on a violation)
        else:
            assert accepts(t, s, whole=False)[0]
    assert ended == len(seqs)          # the budget rule ends every stay within max_new tokens
    assert all(v.id("[CONT]") not in s and v.id("[GOAL]") not in s for s in seqs)


def test_conditions_fix_prefix_header_and_goal_block(setup) -> None:
    from icutwin.generate.constrain import Conditions
    from icutwin.model.data import layout

    v, t, m, model = setup
    lay = layout(v)
    ds = next(i for i, x in enumerate(v.tokens) if x.startswith("PREFIX//dataset//") and not x.endswith("unknown"))
    head = next(i for i, x in enumerate(v.tokens) if x.startswith("HEADER//") and v.meta[i].get("cls") == "header"
                and not x.endswith("unknown"))
    goals = sorted({x for x in t.goal}, key=lambda i: v.tokens[i])
    by_field = {}
    for g in goals:
        by_field.setdefault(t.goal[g], g)
    goal = frozenset(list(by_field.values())[:2])
    cond = Conditions(prefix={lay.dataset_pos - 1: ds}, header=frozenset({head}), goal=goal)
    seqs, proc = _sample(v, t, m, model, n=32, max_new=200, seed=1, cond=cond)
    assert proc.violations == 0
    for s in seqs:
        assert accepts(t, s)[0]
        parse(s, v)
        assert s[lay.dataset_pos] == ds and s[1 + lay.n_prefix] == head      # the forced header field comes first
        a, b = s.index(v.id("[GOAL]")), s.index(v.id("[/GOAL]"))
        assert set(s[a + 1:b]) == set(goal) and s[b + 1] == v.id("[TL]")
    assert Conditions.from_args(cond.to_args()) == cond


def test_obligations_require_and_forbid(setup) -> None:
    """Required token sets must be drawn before the timeline (or end block) closes; forbidden tokens never are."""
    from icutwin.generate.constrain import Conditions

    v, t, m, model = setup
    vitals = frozenset(i for i, x in enumerate(v.tokens) if x.startswith("VITAL//") and v.meta[i].get("cls") == "event")
    labs = frozenset(i for i, x in enumerate(v.tokens) if x.startswith("LAB//") and v.meta[i].get("cls") == "event")
    hosp = [i for i, x in enumerate(v.tokens) if x.startswith("HOSP//discharge_category//") and v.meta[i].get("cls") == "end"]
    cond = Conditions(require=(vitals,), require_end=(frozenset(hosp[:1]),), forbid=labs | frozenset(hosp[1:]))
    seqs, proc = _sample(v, t, m, model, n=32, max_new=400, seed=2, cond=cond)
    forced = sum(o.forced for o in proc.obligations)
    for s, o in zip(seqs, proc.obligations):
        assert accepts(t, s)[0]
        assert not labs & set(s) and not set(hosp[1:]) & set(s)
        if not o.forced:
            end = s.index(v.id("[END]"))
            assert vitals & set(s[:end]) and hosp[0] in s[end:]
    assert forced < len(seqs)


def test_drawn_goal_block_binds_the_stay(setup) -> None:
    """Goal-first sampling (round 3, ``draw_goals``): the model draws a whole goal block after the header (every goal
    field once), and the stay keeps what binds (every binding adopted here): a treatment drawn 'no' never occurs in
    the ICU, the ICU outcome is the drawn one, a 'yes' is conditioning (no stay is held open for it);
    ``strip_goal`` returns the stay without the block and the drawn labels; a goal its dataset's support could not
    keep is never drawn."""
    from icutwin.eval.forecast import Reader
    from icutwin.generate.constrain import Conditions
    from icutwin.generate.engine import goal_plan, strip_goal

    v, t, m, model = setup
    plan, rd = goal_plan(v), Reader(v)
    gid = {v.tokens[i]: i for i in t.goal}
    assert Conditions.from_args(Conditions(draw_goals=True).to_args()).draw_goals
    seqs, proc = _sample(v, t, m, model, n=48, max_new=600, seed=7, cond=Conditions(draw_goals=True), plan=plan)
    assert proc.violations == 0
    fields = set(t.goal.values())
    for s, o in zip(seqs, proc.obligations):
        assert accepts(t, s)[0]
        parse(s, v)
        a, b = s.index(v.id("[GOAL]")), s.index(v.id("[/GOAL]"))
        assert {t.goal[g] for g in s[a + 1:b]} == fields and b - a - 1 == len(fields)
        bare, drawn = strip_goal(s, v)
        assert v.id("[GOAL]") not in bare and len(bare) == len(s) - len(fields) - 2 and set(drawn) == fields
        icu = set(bare[bare.index(plan.marker):]) if plan.marker in bare else set()    # the ICU part
        for k in ("imv", "vasopressor", "rrt", "mcs"):
            if drawn[k] == "no":
                assert not icu & plan.forbid_icu[gid[f"GOAL//{k}//no"]], (k, drawn)
        if drawn["icu_outcome"] != "unknown":
            assert {v.tokens[x] for x in bare if x in rd.dis} <= {f"ICU//discharge//{drawn['icu_outcome']}"}
    assert sum(not o.forced for o in proc.obligations) > len(seqs) // 2    # no stay held open by a 'yes'
    # a goal that could not be kept is never drawn: 'imv: yes' where the vocabulary has no IMV device record (this
    # fixture), 'vasopressor: yes' where the dataset's support lacks every vasopressor
    vaso = plan.feasible[gid["GOAL//vasopressor//yes"]][0]
    assert not plan.feasible[gid["GOAL//imv//yes"]][0] and vaso
    assert all(strip_goal(s, v)[1]["imv"] != "yes" for s in seqs)
    support = np.ones(len(v.tokens), dtype=bool)
    support[list(vaso)] = False

    class _Sup:
        def mask(self, _tok):
            return support

    seqs, _proc = _sample(v, t, m, model, n=24, max_new=300, seed=8, cond=Conditions(draw_goals=True), plan=plan,
                          support=_Sup())
    drawn = [strip_goal(s, v)[1] for s in seqs]
    assert all(d["vasopressor"] != "yes" for d in drawn) and any(d["vasopressor"] == "no" for d in drawn)


def test_goal_plan_binds_as_the_labels_are_derived(setup) -> None:
    """The goal plan follows the goal labels' derivation (``tokenization.stays``; checked on the full tokens by
    ``tools/server/check_goalfirst.py``, real stays with their own goal blocks): a treatment binds in the ICU only,
    vasopressors are the derived episode's drugs, an ICU outcome only forbids the other discharge, a hospital outcome
    requires its token in the end block, and the length-of-stay, KDIGO and SOFA bands bind nothing."""
    from icutwin.generate.engine import GOAL_VASOPRESSORS, goal_plan
    from icutwin.meds.derived import VASOPRESSORS

    v, t, _m, _model = setup
    assert set(GOAL_VASOPRESSORS) == set(VASOPRESSORS)
    plan = goal_plan(v)
    discharge = {v.tokens[i] for i, mt in enumerate(v.meta)
                 if mt.get("cls") == "event" and (mt.get("code") or "").startswith("ICU//discharge//")}
    bound = set(plan.forbid_icu) | set(plan.require_end) | set(plan.forbid)
    for g in t.goal:
        _x, f, label = v.tokens[g].split("//", 2)
        if f in ("imv", "vasopressor", "rrt", "mcs") and label in ("yes", "no"):
            assert (g in plan.forbid_icu) == (label == "no") and (g in plan.feasible) == (label == "yes")
            assert g not in plan.forbid and g not in plan.require_end      # 'yes' is conditioning
        elif f == "icu_outcome" and label != "unknown":
            assert g in plan.forbid and g not in plan.require_end
            assert {v.tokens[x] for x in plan.forbid[g]} == discharge - {f"ICU//discharge//{label}"}
        elif f == "hosp_outcome" and label != "unknown":
            assert {v.tokens[x] for x in plan.require_end[g][0]} == {f"HOSP//discharge_category//{label}"}
        else:
            assert g not in bound, v.tokens[g]
    names = {v.tokens[x].split("//")[1] for g in plan.forbid_icu for x in plan.forbid_icu[g]
             if v.tokens[g] == "GOAL//vasopressor//no"}
    assert names and names <= set(GOAL_VASOPRESSORS)


def test_goal_bindings_are_adopted_per_dataset(setup, tmp_path) -> None:
    """``goal_bindings``: counts per dataset and goal label of the training stays and those breaking the binding;
    a binding is adopted where at most 0.1% of at least 100 labelled stays break it; a drawn goal binds only in the
    datasets that adopt it (the stay's prefix dataset token)."""
    from test_model import _tokens

    from icutwin.generate import goal_bindings as gb
    from icutwin.generate.constrain import Conditions, Obligations
    from icutwin.generate.engine import goal_plan
    from icutwin.tokenization.vocab import Vocab

    tokens = _tokens(tmp_path)
    tv = Vocab.load(tokens / "vocab.json")
    counts = gb.compute(tokens, ["mimic"], tv)["mimic"]
    assert set(counts) == {tv.tokens[g] for g in gb.rules(tv)} and any(n for n, _bad in counts.values())
    assert all(0 <= bad <= n for n, bad in counts.values())
    v = setup[0]                                     # a vocabulary with the ICU admission marker
    rl = gb.rules(v)
    no = next(g for g in rl if v.tokens[g].endswith("//no") and rl[g][1])
    ds = next(i for i, m in enumerate(v.meta) if m.get("cls") == "prefix" and m.get("value") == "mimic")
    data = {"threshold": 0.001, "min_stays": 100, "datasets": {"mimic": {v.tokens[no]: [1000, 1], "x": [0, 0]}}}
    assert gb.adopted(v, data)[ds] == frozenset({no})
    data["datasets"]["mimic"][v.tokens[no]] = [1000, 2]            # 0.2% break it: not adopted
    assert gb.adopted(v, data)[ds] == frozenset()
    marker = next(i for i, m in enumerate(v.meta) if (m.get("code") or "") == "ICU//admission")
    for adopt, binds in ((frozenset({no}), True), (frozenset(), False)):
        obl = Obligations(Conditions(draw_goals=True), goal_plan(v, {ds: adopt}))
        for tok in (ds, no, marker):
            obl.feed(tok)
        assert obl.in_icu and (obl.forbid_icu == rl[no][1]) == binds and bool(obl.forbid_icu) == binds


def test_goal_obligations_map_to_their_events() -> None:
    from pathlib import Path

    from icutwin.generate.engine import obligations
    from icutwin.tokenization.vocab import Vocab

    T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset" / "vocab.json"
    if not T.exists():
        pytest.skip("the subset's vocabulary")
    v = Vocab.load(T)
    req, req_end, forbid = obligations(v, {"imv": "yes", "vasopressor": "no", "hosp_outcome": "home",
                                           "icu_outcome": "alive", "kdigo_max": "2-3"})
    names = lambda s: {v.tokens[i].split("|")[0] for i in s}  # noqa: E731
    assert names(req[0]) == {"RESP//device_category//imv"} and names(req[1]) == {"ICU//discharge//alive"}
    assert names(req_end[0]) == {"HOSP//discharge_category//home"}
    assert any(n.startswith("MED_CON//norepinephrine//") and n.endswith("//start//iv") for n in names(forbid))
    assert "ICU//discharge//dead" in names(forbid) and "HOSP//discharge_category//expired" in names(forbid)
    assert not any(n.startswith("MED_CON//propofol") for n in names(forbid))


def test_random_model_samples_pass_the_meds_layer(setup, tmp_path) -> None:
    """Layer (b) on samples whose validity rests on the masks alone: decoded and written as MEDS, they pass the
    DataSchema, order and code checks against the fixture's codes."""
    import pyarrow.parquet as pq

    from icutwin.generate import meds_out as mo
    from icutwin.tokenization.decode import decode

    v, t, m, model = setup
    seqs, _proc = _sample(v, t, m, model, n=64, max_new=300, seed=3)
    root = _write_meds(tmp_path / "meds", SPLITS)
    known = mo.known_codes(v=v) | {c for f in root.glob("data/*/0.parquet")
                                   for c in pq.read_table(f, columns=["code"]).column(0).to_pylist()}
    known.add("ICU//admission//UNK")       # an unknown unit type's ICU admission (the real references carry it)
    table = mo.table([decode(s, v) for s in seqs], list(range(len(seqs))))
    rep = mo.validate(table, known)
    assert rep["passed"], rep["failures"]
    assert rep["subjects"] == len(seqs)


def test_icu_los_goal_bounds_the_discharge(setup) -> None:
    """The LOS register: with a requested ICU length-of-stay band, every sample not forced by the budget has its
    ICU discharge inside the band by the interval tokens' median gaps (the decoder's deterministic reading)."""
    from icutwin.eval.forecast import Reader
    from icutwin.generate.engine import Request, conditions

    v, t, m, model = setup
    rd = Reader(v)
    for band, (lo, hi) in (("24-48", (24, 48)), ("0-24", (0, 24))):
        if f"GOAL//icu_los//{band}" not in v.index:
            pytest.skip("the fixture vocabulary has no such goal")
        cond = conditions(v, Request(goals={"icu_los": band}))
        assert cond.los is not None and cond.los.lo == lo * 60
        seqs, proc = _sample(v, t, m, model, n=24, max_new=600, seed=5, cond=cond)
        free = [(s, o) for s, o in zip(seqs, proc.obligations) if not o.forced]
        assert free, band
        for s, _o in free:
            assert accepts(t, s)[0]
            los = rd.outcomes(s)
            minutes, start, end = 0.0, None, None
            for tok in s:
                minutes += rd.minutes.get(tok, 0.0)
                if start is None and tok in rd.adm:
                    start = minutes
                elif start is not None and tok in rd.dis:
                    end = minutes
                    break
            assert end is not None and lo * 60 <= end - start < hi * 60, (band, end - start, los)


def test_header_augmentation_is_undone(setup) -> None:
    """Training set each header field to unknown with probability r; the inverse takes r x the field's mass from its
    unknown token, keeps the field's total, and leaves seen fields and other phases alone. With a random model, the
    unknown share of each field in samples falls from about rate + (1 - rate) u to about u."""
    import torch

    from icutwin.generate.constrain import HeaderAugmentation

    v, t, m, model = setup
    undo = HeaderAugmentation(v, 0.05)
    a = Automaton(t)
    for tok in [v.id("[BOS]"), *[next(iter(sorted(f))) for f in t.prefix]]:
        assert a.step(tok)
    assert a.phase == "header"
    rng = np.random.default_rng(0)
    logits = rng.normal(size=len(v.tokens)).astype(np.float32)
    mask = np.zeros(len(v.tokens), dtype=bool)
    mask[list(t.header)] = True
    logits[~mask] = -np.inf
    out = undo(logits, a)
    p0 = np.exp(logits[mask] - logits[mask].max())
    p0 = p0 / p0.sum()
    full0 = np.zeros(len(v.tokens))
    full0[mask] = p0
    p1 = np.exp(out - out[np.isfinite(out)].max())
    p1 = p1 / p1.sum()
    for f, u in undo.unknown.items():
        toks = undo.fields[f]
        q0, q1 = full0[toks].sum(), p1[toks].sum()
        assert abs(q0 - q1) < 1e-6                                      # the field keeps its mass
        share0, share1 = full0[u] / q0, p1[u] / q1
        assert abs(share1 - max(share0 - 0.05, 0) / 0.95) < 1e-6        # P_real = (P - r) / (1 - r)
    a.step(next(iter(t.header)))
    tl = Automaton(t)
    assert np.array_equal(undo(logits, tl), logits)                     # other phases untouched
    seqs, _proc = _sample(v, t, m, model, n=48, max_new=40, seed=9)
    proc = GrammarProcessor(t, m, v.id("[PAD]"), max_length=41, header_augmentation=undo)
    from transformers import GenerationConfig, LogitsProcessorList

    torch.manual_seed(9)
    cfg = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=40,
                           eos_token_id=v.id("[EOS]"), pad_token_id=v.id("[PAD]"))
    ids = torch.full((48, 1), v.id("[BOS]"), dtype=torch.long)
    with torch.no_grad():
        out_ids = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=cfg,
                                 logits_processor=LogitsProcessorList([proc]))
    unknown = set(undo.unknown.values())
    before = np.mean([sum(int(x) in unknown for x in s) for s in seqs])
    after = np.mean([sum(int(x) in unknown for x in s.tolist()) for s in out_ids])
    assert after < before and proc.violations == 0
