"""The reference consumer (S9, ``factory.consumer``) on episodes of every task type from a store built from real
held-out token sequences standing in for twins (skipped without them): a forward pass and a masked loss with finite
values and gradients for binary, time-to-event, ordinal, regression, in-hospital death, forecast and multi-horizon
episodes, and a query's output at a prediction time unchanged when its later events change (the format carries
availability times a consumer can mask by)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from icutwin.factory import episodes, pool  # noqa: E402
from icutwin.factory import programs as fpr  # noqa: E402
from icutwin.factory import store as fst  # noqa: E402

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "full_r4"


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    if not (T / "vocab.json").exists() or not (T / "eicu" / "held_out.parquet").exists():
        pytest.skip("the full r4 tokens")
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    rows = []
    for ds in ("mimic", "eicu"):
        t = pq.read_table(T / ds / "held_out.parquet", columns=["stay_id", "ids"]).slice(0, 70).to_pylist()
        rows += [{"stay_id": r["stay_id"], "ids": r["ids"], "split": "train"} for r in t]
    root = tmp_path_factory.mktemp("cons")
    fst.build(rows, T / "vocab.json", root / "s", "seg_0000", {"kind": "test"}, workers=1, chunk=70)
    st = fst.Store(root / "s")
    st.channels.stats = fst.channel_stats(st)
    return v, st


def _episodes(v, st) -> dict:
    """One episode per label type (stage 2 carries multi-horizon labels)."""
    cat = fpr.catalog(st.channels)
    progs = cat + pool.fc_candidates(st.channels, 6, 0) + pool.mh_candidates(cat, 6, 0)
    out = {}
    for stage in (1, 2):
        s = episodes.Sampler(st, v, stage=stage, seed=3, programs=progs)
        for i in range(400):
            try:
                ep = s.episode(i)
            except episodes.NoEpisode:
                continue
            out.setdefault(ep["task"]["label"], ep)
            if len(out) == 7:
                return out
    return out


def test_forward_and_loss_on_every_task_type(setup) -> None:
    from icutwin.factory.consumer import Consumer

    v, st = setup
    eps = _episodes(v, st)
    assert set(eps) == {"binary", "tte", "ordinal", "regression", "hosp_death", "forecast", "multihorizon"}, sorted(eps)
    m = Consumer(len(st.channels), d=16)
    for label, ep in eps.items():
        outs = m(ep)
        assert len(outs) == len(ep["query"]["ev_off"]) - 1
        for o in outs:
            assert torch.isfinite(o["out"]).all(), label
        loss = m.loss(ep, outs)
        assert torch.isfinite(loss), label
        m.zero_grad()
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), label


def test_a_query_state_ignores_its_later_events(setup) -> None:
    from icutwin.factory.consumer import Consumer

    v, st = setup
    eps = _episodes(v, st)
    ep = eps["binary"]
    q = ep["query"]
    k = 0
    lo = np.asarray(q["lab_off"])
    t1 = int(np.asarray(q["lab_t"])[lo[0]])                       # the first query's first prediction time
    m = Consumer(len(st.channels), d=16).eval()
    with torch.no_grad():
        before = m(ep)[k]["out"][0].clone()
        a, b = int(q["ev_off"][k]), int(q["ev_off"][k + 1])
        later = np.flatnonzero(np.asarray(q["ev_t"])[a:b] > t1) + a
        q2 = {**q, "ev_z": q["ev_z"].copy(), "ev_c": q["ev_c"].copy()}
        q2["ev_z"][later] = 9.0
        q2["ev_c"][later] = 1
        after = m({**ep, "query": q2})[k]["out"][0]
    assert len(later) > 0 and torch.allclose(before, after)
