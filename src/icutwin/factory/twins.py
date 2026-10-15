"""Twins as the factory decodes them (plan Phase F, Factory design 1): a twin's clock and values come from generators
keyed by its id, so it decodes the same way wherever it is read (the stay store, the oracle's anchors, evaluation
sets), whatever its position in a list.

* ``clock_rng``: the generator of a twin's gaps (``tokenization.decode`` draws interval tokens within their bins).
* ``value_rng``: the generator of its values (drawn within their bins, with the real within-bin rank persistence).
* ``decode_labels`` / ``decode_features``: the twin on its clock with values at their bins' medians (labels: drawn
  values inflate change- and ratio-based labels such as KDIGO, docs/m8_factory.md) or drawn (features). The two
  decodes share the clock because gaps and values use separate generators.
"""

from __future__ import annotations

import random

from icutwin.tokenization.decode import decode
from icutwin.tokenization.stays import Stay


def clock_rng(twin_id: str, seed: int = 0) -> random.Random:
    return random.Random(f"clock:{seed}:{twin_id}")


def value_rng(twin_id: str, seed: int = 0) -> random.Random:
    return random.Random(f"value:{seed}:{twin_id}")


def decode_labels(ids, v, twin_id: str, seed: int = 0, clock_id: str | None = None) -> Stay:
    """The labels' decode: gaps drawn on the twin's clock (``clock_id``'s when given: the rollouts of one anchor share
    their prompt's clock), values at their bins' medians."""
    s = decode([int(x) for x in ids], v, clock_rng(clock_id or twin_id, seed), value_rng=None)
    s.stay_id = twin_id
    return s


def decode_features(ids, v, twin_id: str, seed: int = 0, persistence: dict | None = None,
                    clock_id: str | None = None) -> Stay:
    """The features' decode: the same clock, values drawn within their bins (with rank persistence)."""
    s = decode([int(x) for x in ids], v, clock_rng(clock_id or twin_id, seed), persistence, value_rng(twin_id, seed))
    s.stay_id = twin_id
    return s
