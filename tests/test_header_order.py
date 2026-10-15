"""Canonical header order of generated stays (round 2): the any-order header sorted into the spec's order,
losslessly (real stays unchanged, shuffled headers restored, grammar kept)."""

from __future__ import annotations


def test_header_order_is_canonical_and_lossless() -> None:
    """Generated headers come in any order; HeaderOrder sorts them into the spec's order: real stays unchanged,
    shuffled ones restored, the grammar still accepts, a malformed sample left as it is."""
    from pathlib import Path

    import numpy as np
    import pyarrow.parquet as pq
    import pytest

    from icutwin.generate.engine import HeaderOrder
    from icutwin.tokenization.fsm import accepts, compile_tables
    from icutwin.tokenization.vocab import Vocab

    root = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"
    if not (root / "vocab.json").exists():
        pytest.skip("the subset's tokens")
    v = Vocab.load(root / "vocab.json")
    t = compile_tables(v)
    order = HeaderOrder(v)
    rng = np.random.default_rng(0)
    a = 1 + order.lay.n_prefix
    b = a + order.lay.n_header
    for ids in pq.read_table(root / "eicu" / "held_out.parquet", columns=["ids"]).column(0).to_pylist()[:50]:
        assert order(ids) == list(ids)                         # real stays: already canonical
        shuffled = list(ids)
        head = shuffled[a:b]
        rng.shuffle(head)
        shuffled[a:b] = head
        assert accepts(t, shuffled)[0] and order(shuffled) == list(ids)
    assert order([v.id("[BOS]")]) == [v.id("[BOS]")]
