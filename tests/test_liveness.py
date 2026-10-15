"""Bounded checks of the constrained sampler's liveness (plan *Structured-output guarantee* 3 and 5: every reachable
state has an allowed token and a path to [EOS] within the reserved budget).

* No dead state: along walks drawn uniformly from the generation mask (the worst case of a near-uniform model),
  every state before [EOS] allows at least one token.
* Completion within the reserve: from every state such a walk visits, an adversary that always draws the token
  that delays the end the most (a repeatable week interval first, then other intervals, sub-tokens, events; in the
  end block anything but [EOS]) still reaches [EOS] when the budget rule takes over at ``END_RESERVE`` tokens left.
Run on the fixture vocabulary and, when present, on the 2% subset's real one."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from test_tokenization import SPLITS, _write_meds

from icutwin.generate.constrain import END_RESERVE, NONE, generation_mask
from icutwin.tokenization.fsm import Automaton, compile_tables, mask_tables
from icutwin.tokenization.stays import iter_stays
from icutwin.tokenization.vocab import Vocab, fit

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"


def _vocabs(tmp: Path):
    meds = _write_meds(tmp / "meds", SPLITS)
    out = [fit(iter_stays(meds, "train"), min_subjects=1)]
    if (T / "vocab.json").exists():
        out.append(Vocab.load(T / "vocab.json"))
    return out


def _delay_rank(v: Vocab, t, tok: int) -> int:
    """Lower is drawn first by the adversary."""
    if tok == t.week:
        return 0
    if tok in t.interval:
        return 1
    if tok in t.chunk_level:
        return 2
    if tok == v.id("[EOS]") or tok == v.id("[END]"):
        return 9
    return 3


def _states(v: Vocab, t, m, rng, walks: int, max_len: int):
    """Automata along uniform walks from [BOS] under the generation mask (no budget)."""
    out = []
    for _ in range(walks):
        a = Automaton(t)
        a.step(v.id("[BOS]"))
        for _k in range(max_len):
            mask = generation_mask(a, m, None, NONE)
            ok = np.flatnonzero(mask)
            assert len(ok) or a.phase == "done", f"dead state in phase {a.phase}"
            if a.phase == "done":
                break
            out.append(copy.deepcopy(a))
            assert a.step(int(rng.choice(ok))), a.violation
    return out


@pytest.mark.parametrize("which", [0, 1])
def test_no_dead_state_and_completion_within_the_reserve(which: int, tmp_path: Path) -> None:
    vocabs = _vocabs(tmp_path)
    if which >= len(vocabs):
        pytest.skip("the subset's vocabulary")
    v = vocabs[which]
    t = compile_tables(v)
    m = mask_tables(t, len(v.tokens))
    rng = np.random.default_rng(which)
    states = _states(v, t, m, rng, walks=40, max_len=120)
    timeline = [a for a in states if a.phase in ("tl", "end")]
    assert len(timeline) > 100
    # make the adversary's favourite position common: states right after a week interval
    for a in timeline[:200]:
        if a.phase == "tl" and a.last in ("event", "week"):
            b = copy.deepcopy(a)
            if b.step(t.week):
                timeline.append(b)
    worst = 0
    for a in timeline:
        a = copy.deepcopy(a)
        left = END_RESERVE
        used = 0
        while a.phase != "done":
            mask = generation_mask(a, m, left, NONE)
            ok = np.flatnonzero(mask)
            assert len(ok), f"dead state under the budget rule in phase {a.phase}, last {a.last}"
            tok = int(min(ok, key=lambda x: (_delay_rank(v, t, int(x)), int(x))))
            assert a.step(tok), a.violation
            used += 1
            left -= 1
            assert left >= 0, f"budget exhausted in phase {a.phase} (last {a.last})"
        worst = max(worst, used)
    assert worst <= END_RESERVE + 1
