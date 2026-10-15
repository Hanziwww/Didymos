"""Interventions for rollouts (S7 ``simulate``, plan *Prediction-task generation* 4) on the 2% subset's real token
sequences (skipped without them): do(start norepinephrine) writes a grammatical time group at the cutoff,
do(no vasopressor) withholds every vasoactive start from the continuation, and an intervention the prefix cannot
take (a dose change of an infusion that does not run) is refused."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from icutwin.generate.constrain import Conditions
from icutwin.generate.engine import Intervention, intervene, rollout_requests, treatment_tokens
from icutwin.tokenization.fsm import Automaton, compile_tables

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"


@pytest.fixture(scope="module")
def setup():
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's tokens")
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    t = compile_tables(v)
    prefixes = []
    for ids in pq.read_table(T / "mimic" / "held_out.parquet", columns=["ids"]).column(0).to_pylist()[:40]:
        a = Automaton(t)
        cut = None
        for k, tok in enumerate(ids):           # the prefix: up to an event right before an interval, mid-stay
            if a.phase == "tl" and a.last == "event" and tok in t.interval and k > len(ids) // 3:
                cut = k
                break
            a.step(tok)
        if cut is not None:
            prefixes.append(ids[:cut])
    return v, t, prefixes


def _start(v, drug: str) -> int:
    return next(i for i, m in enumerate(v.meta) if m.get("kind") == "bin" and m.get("code", "").startswith(
        f"MED_CON//{drug}//") and m.get("action") == "start")


def test_do_and_withhold(setup) -> None:
    v, t, prefixes = setup
    assert len(prefixes) > 20
    norepi = _start(v, "norepinephrine")
    give = Intervention(give=(norepi,))
    for pre in prefixes:
        prompt = intervene(pre, give, v, t)
        assert prompt[:len(pre)] == list(pre) and prompt[len(pre)] == v.id("INT//1-5m") and prompt[-1] == norepi
    vaso = treatment_tokens(v, "vasopressor")
    assert norepi in vaso and len(vaso) > 10
    withhold = Intervention(withhold=vaso)
    prompts, args = rollout_requests(prefixes[:3], 4, seed=2, context=33024, v=v, t=t, interventions=withhold)
    assert len(prompts) == 12 and prompts[0] == list(prefixes[0])
    assert len({a["seed"] for a in args}) == 12
    cond = Conditions.from_args(args[0]["icutwin"]["conditions"])
    assert cond.forbid == vaso and args[0]["icutwin"]["max_new"] == 33024 - len(prefixes[0]) - 1
    _p, plain = rollout_requests(prefixes[:2], 2, seed=0, context=33024, v=v, t=t)
    assert "conditions" not in plain[0]["icutwin"]


def test_refuses_what_the_prefix_cannot_take(setup) -> None:
    v, t, prefixes = setup
    pre = prefixes[0]
    a = Automaton(t)
    for tok in pre:
        a.step(tok)
    change = next(i for i, m in enumerate(v.meta) if m.get("kind") == "bin" and m.get("action") == "dose_change"
                  and m.get("key") not in a.running)
    with pytest.raises(ValueError):
        intervene(pre, Intervention(give=(change,)), v, t)
