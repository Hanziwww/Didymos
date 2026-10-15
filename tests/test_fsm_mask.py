"""The grammar automaton's token mask (S7 constrained decoding) against its own ``step``: on real encoded stays and
on random walks drawn from the mask, a token is allowed exactly when ``step`` accepts it."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from test_tokenization import SPLITS, _write_meds

from icutwin.tokenization.encode import encode
from icutwin.tokenization.fsm import Automaton, accepts, allowed, compile_tables, mask_tables
from icutwin.tokenization.stays import iter_stays
from icutwin.tokenization.vocab import fit


@pytest.fixture(scope="module")
def grammar(tmp_path_factory: pytest.TempPathFactory):
    meds = _write_meds(tmp_path_factory.mktemp("fsm") / "meds", SPLITS)
    v = fit(iter_stays(meds, "train"), min_subjects=1)
    seqs = []
    for split in SPLITS:
        for s in iter_stays(meds, split):
            seqs.append(encode(s, v).ids)
            seqs.append(encode(s, v, goal=True).ids)
            seqs += encode(s, v, max_tokens=28).windows          # continuation windows with state blocks
    t = compile_tables(v)
    return v, t, mask_tables(t, len(v.tokens)), seqs


def _check(a: Automaton, m, rng: np.random.Generator, n: int = 0) -> np.ndarray:
    """The mask in ``a``'s state, checked against ``step`` on copies (every token when n == 0, else n of each kind)."""
    mask = allowed(a, m)
    toks = np.arange(m.n)
    if n:
        toks = np.r_[rng.choice(np.flatnonzero(mask), min(n, int(mask.sum())), replace=False) if mask.any() else [],
                     rng.choice(np.flatnonzero(~mask), min(n, int((~mask).sum())), replace=False)].astype(int)
    for tok in toks:
        b = copy.deepcopy(a)
        assert b.step(int(tok)) == bool(mask[tok]), (a.phase, int(tok), bool(mask[tok]), b.violation)
    return mask


def test_real_sequences_take_allowed_tokens_only(grammar) -> None:
    v, t, m, seqs = grammar
    rng = np.random.default_rng(0)
    for ids in seqs:
        assert accepts(t, ids, whole=False)[0]
        a = Automaton(t)
        for pos, tok in enumerate(ids):
            mask = _check(a, m, rng, n=20) if pos % 5 == 0 else allowed(a, m)
            assert mask[tok], (pos, v.tokens[tok], a.phase)
            assert a.step(tok)


def test_random_walks_agree_with_step_on_every_token(grammar) -> None:
    _v, t, m, _seqs = grammar
    rng = np.random.default_rng(1)
    for _walk in range(12):
        a = Automaton(t)
        for _ in range(120):
            mask = _check(a, m, rng)              # all tokens
            if not mask.any():                    # [EOS] taken: nothing may follow
                assert a.phase == "done"
                break
            assert a.step(int(rng.choice(np.flatnonzero(mask))))
