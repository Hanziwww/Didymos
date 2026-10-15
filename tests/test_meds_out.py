"""MEDS output of decoded stays (S7 verification layer b), on the 2% subset's real token sequences (skipped without
them): every decoded stay's MEDS rows pass the DataSchema, order and code checks, and the tokenizer's own MEDS
reader (``build_stay``) turns them back into the same tokens (the writer inverts the reader); corrupted tables fail."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from icutwin.generate import meds_out as mo

ROOT = Path(__file__).resolve().parents[1] / "data"
T = ROOT / "tokens" / "subset"
DATASETS = ("mimic", "eicu", "hirid", "aumc", "sicdb", "nwicu", "inspire")


@pytest.fixture(scope="module")
def decoded():
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's token shards")
    from icutwin.tokenization.decode import decode
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    out = []
    for ds in DATASETS:
        ids = pq.read_table(T / ds / "held_out.parquet", columns=["ids"]).column(0).to_pylist()[:150]
        out += [(ds, x, decode(x, v)) for x in ids]
    return v, out


def _restay(rows: list, ds: str):
    from icutwin.tokenization.stays import build_stay

    adm = [r[1] for r in rows if r[2].startswith("ICU//admission//")]
    dis = [r[1] for r in rows if r[2].startswith("ICU//discharge//")]
    icu_in = adm[0]
    icu_out = dis[0] if dis else max(r[1] for r in rows if r[1] is not None)
    clock = any(r[2] == "ICU//admission//" or r[2].startswith("ICU//admission//") for r in rows)
    meta = {"subject_id": rows[0][0], "stay_id": "x", "dataset": ds, "split": "held_out", "icu_in": icu_in,
            "icu_out": icu_out, "clock_valid": clock}
    return build_stay(meta, [(t, c, n, x) for _s, t, c, n, x in rows])


def _end_events(ids: list, v) -> list:
    """The end block's events (an end token and its sub-tokens), sorted: a diagnosis without its sub-tokens
    decodes to its system alone, whose canonical place in the block depended on the unseen code."""
    at = ids.index(v.id("[END]"))
    evs = []
    for t in ids[at + 1:-1]:
        if v.meta[t].get("cls") == "chunk" and evs:
            evs[-1].append(t)
        else:
            evs.append([t])
    return sorted(evs)


def test_rows_round_trip_through_the_tokenizer(decoded) -> None:
    from icutwin.tokenization.encode import encode

    v, stays = decoded
    same, diffs = 0, []
    for k, (ds, ids, s) in enumerate(stays):
        rows = mo.rows(s, k)
        s2 = _restay(rows, ds)
        s2.clock_valid = s.header.get("admission_hour") is not None
        if s2.clock_valid is False:
            s2.header["admission_hour"] = None
        again = encode(s2, v).ids
        ids = list(ids)
        cut, cut2 = ids.index(v.id("[END]")), again.index(v.id("[END]"))
        if again[:cut2] == ids[:cut] and _end_events(again, v) == _end_events(ids, v):
            same += 1
        else:
            j = next((i for i, (a, b) in enumerate(zip(again, ids)) if a != b), min(len(again), len(ids)))
            diffs.append((ds, j, v.tokens[ids[j]] if j < len(ids) else None, v.tokens[again[j]] if j < len(again) else None))
    assert same == len(stays), (same, len(stays), diffs[:10])


def test_tables_validate_and_corruptions_fail(decoded) -> None:
    _v, stays = decoded
    known = mo.known_codes(ROOT / "meds" / "subset")
    if not known:
        pytest.skip("the subset's MEDS metadata")
    t = mo.table([s for _ds, _ids, s in stays], list(range(len(stays))))
    rep = mo.validate(t, known)
    assert rep["passed"], rep["failures"]
    assert rep["subjects"] == len(stays)
    import pyarrow as pa

    shuffled = t.take(pa.array(list(range(t.num_rows))[::-1]))
    assert any("sorted" in f for f in mo.validate(shuffled, known)["failures"])
    bad = t.set_column(2, "code", pa.array(["LAB//not_a_code"] * t.num_rows, pa.string()))
    assert any("outside the reference" in f for f in mo.validate(bad, known)["failures"])
    first = mo.rows(stays[0][2], 0)
    assert first[0][1] is None and all(isinstance(r[1], datetime) for r in first if r[1] is not None)
