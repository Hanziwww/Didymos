"""S5 tokenizer (M4 D3-D5): grammar acceptance, round trip, windows, train-only fitting, determinism."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from icutwin.tokenization import timeline as tl
from icutwin.tokenization.decode import decode
from icutwin.tokenization.encode import encode
from icutwin.tokenization.fsm import accepts, compile_tables
from icutwin.tokenization.parser import ParseError, parse
from icutwin.tokenization.stays import iter_stays
from icutwin.tokenization.validate import Report, canonical_view, validate_stay
from icutwin.tokenization.vocab import fit

T = datetime(2100, 1, 10, 8)


def _stay_events(sid: int, ds: str, shift: float = 0.0, extra: bool = True) -> list[tuple]:
    h = lambda x: T + timedelta(hours=x)  # noqa: E731
    ev = [(None, "DATASET//" + ds, None, None), (None, f"SITE//{ds}//{ds}", None, None),
          (None, "PATIENT//sex//female", None, "F"), (None, "PATIENT//race//white", None, "W"),
          (h(-30), "HOSP//age_charted", 64.0 + sid, None), (h(-30), "HOSP//admission_type//ed", None, "EW"),
          (h(0), "ICU//admission//medical_icu", None, None), (h(0), "ICU//admission_source//ed", None, None),
          (h(0), "ICU//surgical_status//nonsurgical", None, None), (h(0), "ICU//adm_dx_group//sepsis", None, None),
          (h(0), "ICU//stay_seq", 1.0, None), (h(0), "ICU//era//2011-2013", None, None),
          (h(-2), "VITAL//heart_rate", 90.0 + sid + shift, None), (h(0.5), "VITAL//heart_rate", 100.0 + sid, None),
          (h(0.5), "VITAL//sbp//arterial", 110.0 + 2 * sid, None),
          (h(0.5 + 10 / 3600), "LAB//sodium//mmol/l//bmp", 138.0 + sid, None),   # 10 s later: the same group
          (h(1), "MED_CON//norepinephrine//mcg/kg/min//start//iv", 0.05 * (sid + 1), None),
          (h(1 + 30 / 3600), "MED_CON//norepinephrine//mcg/kg/min//dose_change//iv", 0.08 * (sid + 1), None),
          (h(3), "MED_CON//norepinephrine//mcg/kg/min//stop//iv", 0.0, None),
          (h(4), "LAB//troponin_i//ng/l//misc", 15.0 if sid == 2 else None, None if sid == 2 else "UNABLE TO REPORT"),
          (h(4), "PA//gcs_eye", 3.0, None), (h(4), "PA//gcs_motor", 5.0, None), (h(4), "PA//gcs_verbal", 4.0, None),
          (h(4), "PA//gcs_total", 12.0, None),   # derivable: not tokenized
          (h(30), "ICU//discharge//alive", None, None),
          (h(80), "HOSP//discharge_category//home", None, "HOME"),
          (h(80), "HOSP_DX//ICD10CM//I10", None, None), (h(80), "HOSP_DX//ICD10CM//A419", None, None),
          (h(80), "HOSP_DX//AMSTERDAM_APACHE//13116-1", None, None)]   # a system that is not character-hierarchical
    if extra:
        ev += [(h(200 + 24 * 9), "MEDS_DEATH", None, None)]
    if sid % 2:
        ev += [(h(2), "MED_INT//ext_metoprolol//mcg//given//enteral", 25000.0, None),
               (h(2), "PROC//ICD10PCS//0BH17EZ", None, None)]
    return [(sid, *e) for e in ev]


def _write_meds(root: Path, stays: dict[str, list[int]], shift: float = 0.0) -> Path:
    meta = []
    for split, sids in stays.items():
        part = [r for sid in sids for r in _stay_events(sid, "mimic", shift if split == "held_out" else 0.0)]
        table = pa.table({"subject_id": pa.array([r[0] for r in part], pa.int64()),
                          "time": pa.array([r[1] for r in part], pa.timestamp("us")),
                          "code": [r[2] for r in part],
                          "numeric_value": pa.array([r[3] for r in part], pa.float32()),
                          "text_value": [r[4] for r in part]})
        (root / "data" / split).mkdir(parents=True, exist_ok=True)
        pq.write_table(table, root / "data" / split / "0.parquet")
        for sid in sids:
            meta.append({"subject_id": sid, "stay_id": f"mimic:{sid}", "dataset": "mimic", "patient_id": sid,
                         "hospitalization_id": sid, "icu_in": T, "icu_out": T + timedelta(hours=30),
                         "window_start": T - timedelta(hours=48), "split": split, "clock_valid": True})
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(meta), root / "metadata" / "stays.parquet")
    return root


SPLITS = {"train": [1, 2, 3, 4], "tuning": [5], "held_out": [6, 7]}


@pytest.fixture()
def meds(tmp_path: Path) -> Path:
    return _write_meds(tmp_path / "meds", SPLITS)


def _fit(meds: Path):
    return fit(iter_stays(meds, "train"), min_subjects=1)


def test_every_stay_is_accepted_and_round_trips(meds: Path) -> None:
    v = _fit(meds)
    tables = compile_tables(v)
    rep = Report()
    for split in SPLITS:
        for s in iter_stays(meds, split):
            validate_stay(s, v, tables, rep)
    summary = rep.summary()
    assert rep.stays == 7
    assert not summary["fsm_rejected"] and not summary["parser_rejected"], summary["examples"]
    assert summary["round_trip_failed"] == 0 and summary["reencode_failed"] == 0, summary["examples"]
    assert summary["sampled_reencode_failed"] == 0, summary["examples"]
    assert summary["continuity_failed"] == 0 and summary["separation_failed"] == 0, summary["examples"]
    assert summary["events_without_own_token"]["derivable (decoder derives it)"] == 7


def test_sampled_dequantization_stays_in_its_bins(meds: Path) -> None:
    """D5: values and gaps drawn from the bins' empirical distributions (per dataset) bin back to the same
    tokens; without a generator the decoder takes the bins' medians."""
    import random

    from icutwin.tokenization.decode import read_sketch
    from icutwin.tokenization.vocab import bin_of
    assert read_sketch([1.0, 2.0, 3.0, 4.0, 5.0], None) == 3.0
    assert read_sketch([1.0, 2.0, 3.0, 4.0, 5.0], 0.0) == 1.0 and read_sketch([1.0, 2.0, 3.0, 4.0, 5.0], 0.375) == 2.5
    v = _fit(meds)
    for s in iter_stays(meds, "held_out"):
        ids = encode(s, v).ids
        median, draws = decode(ids, v), [decode(ids, v, random.Random(seed)) for seed in range(5)]
        for d in [median, *draws]:
            assert encode(d, v).ids == ids
            for _t, _r, code, num, _x in d.timeline:
                if num is not None and code in v.codes and v.codes[code]["kind"] == "numeric":
                    assert v.codes[code]["first"] + bin_of(num, v.codes[code]["edges"]) in ids


def test_persistent_draws_and_a_shared_clock(meds: Path) -> None:
    """The within-bin rank persistence keeps every value in its bin (a re-encode gives the tokens back) and, at rho
    near 1, repeats a code's rank while its bin repeats; a decode with its own value generator, or with values at
    the medians (``value_rng=None``), keeps the clock its gap generator draws; the ICU window is the marker to the
    discharge."""
    import random
    from types import SimpleNamespace

    from icutwin.tokenization.decode import _persistent_rank, load_persistence

    toy = SimpleNamespace(meta=[{"kind": "bin", "code": "X", "bin": 2}, {"kind": "bin", "code": "X", "bin": 3}])

    def steps(rho: float) -> float:   # mean rank change between consecutive draws in one bin
        last, g = {}, random.Random(0)
        u = [_persistent_rank((0,), toy, g, {"X": rho}, last) for _ in range(400)]
        return sum(abs(a - b) for a, b in zip(u, u[1:], strict=False)) / 399

    assert steps(0.99) < 0.2 * steps(0.0) and 0.25 < steps(0.0) < 0.42      # independent: E|U1 - U2| = 1/3
    last = {}
    _persistent_rank((0,), toy, random.Random(0), {"X": 0.99}, last)
    _persistent_rank((1,), toy, random.Random(0), {"X": 0.99}, last)          # another bin: a fresh draw
    assert last["X"][0] == 3
    v = _fit(meds)
    for s in list(iter_stays(meds, "held_out"))[:20]:
        ids = encode(s, v).ids
        assert encode(decode(ids, v, random.Random(1), {c: 0.99 for c in v.codes}), v).ids == ids
        a = decode(ids, v, random.Random(7), value_rng=random.Random(8))
        b = decode(ids, v, random.Random(7), value_rng=None)
        assert [t for t, *_ in a.timeline] == [t for t, *_ in b.timeline]
        assert (a.icu_in, a.icu_out) == (b.icu_in, b.icu_out)
        assert [x for *_, x, _ in b.timeline] == [x for *_, x, _ in decode(ids, v).timeline]   # medians
        marker = next(t for t, _r, c, *_ in a.timeline if c == "ICU//admission")
        assert a.icu_in == marker and a.icu_out >= marker
    p = meds.parent / "rank_persistence.json"
    p.write_text('{"per_code": {"VITAL//heart_rate": {"pairs": 300, "rho": 0.4}}, '
                 '"by_domain": {"VITAL": {"codes": 1, "median": 0.4}, "LAB": {"codes": 1, "median": NaN}}}')
    r = load_persistence(p, v)
    assert r.get("VITAL//heart_rate") == 0.4 and all(0 <= x <= 0.99 for x in r.values())
    assert all(r[c] == 0.0 for c in r if c.startswith("LAB//"))          # an undefined median falls back to 0


def test_band_decoding() -> None:
    from icutwin.tokenization.decode import _band_value
    assert _band_value("60-65") == 62.5 and _band_value("90+") == 90.0   # midpoint; an open band's lower edge
    assert _band_value("2-3") == 2.0 and _band_value("23+") == 23.0      # an integer band of width one: its value


def test_canonical_view_keeps_groups_values_and_order(meds: Path) -> None:
    v = _fit(meds)
    s = next(iter_stays(meds, "train"))
    view = canonical_view(s, v)
    groups = view[3]
    # the sodium 10 s after the vitals joins their group; group order is (domain, code)
    codes = [[c for c, _b in evs] for _ints, evs in groups]
    assert ["VITAL//heart_rate", "VITAL//sbp//arterial", "LAB//sodium//mmol/l//bmp"] in codes
    # the start and the dose change 30 s later share a group: start first (the spec's action order)
    assert ["MED_CON//norepinephrine//mcg/kg/min//start//iv",
            "MED_CON//norepinephrine//mcg/kg/min//dose_change//iv"] in codes
    assert ("LAB//troponin_i//ng/l//misc", "text") in [e for _i, evs in groups for e in evs]
    assert view[4][0][0] == "HOSP//discharge_category//home" and view[4][1][0] == "MEDS_DEATH"
    # prefix: the plan's 5-year age bucket and the era, a stay event at ICU admission
    assert {"PREFIX//age//65-70", "PREFIX//era//2011-2013"} <= set(view[0])
    dec = decode(encode(s, v).ids, v)
    assert canonical_view(dec, v) == view


def test_long_stays_are_windowed_with_state(meds: Path) -> None:
    v = _fit(meds)
    tables = compile_tables(v)
    s = next(iter_stays(meds, "train"))
    enc = encode(s, v, max_tokens=24)
    assert len(enc.windows) > 1
    for k, w in enumerate(enc.windows):
        ok, why = accepts(tables, w, whole=k == len(enc.windows) - 1)
        assert ok, why
        parse(w, v, whole=k == len(enc.windows) - 1)
    toks = [[v.tokens[i] for i in w] for w in enc.windows]
    running = [w for w in toks if "[CONT]" in w and any(t.startswith("STATE//MED_CON//norepinephrine") for t in w)]
    assert running, "a window opening while the infusion runs declares it"
    from icutwin.tokenization.validate import _continuity
    assert _continuity(enc.ids, enc.windows, v, tables) is None      # windows tile the timeline; states replay
    broken = [w[:-1] if k == 1 else w for k, w in enumerate(enc.windows)]
    assert _continuity(enc.ids, broken, v, tables) == "windows do not tile the timeline"


def test_prospective_inputs_hold_only_what_is_known_at_the_cutoff(meds: Path) -> None:
    """Separation: a prediction input at a cutoff has the timeline up to it, no goal or outcome, and header fields
    whose documentation may come later (spec available_within_hours, e.g. eICU's APACHE diagnosis) unknown."""
    from icutwin.tokenization.encode import prospective
    v = _fit(meds)
    s = next(iter_stays(meds, "train"))
    early, late = prospective(s, v, 0.6), prospective(s, v, 30)
    assert "HEADER//adm_dx_group//sepsis" in [v.tokens[t] for t in early]          # MIMIC: known at admission
    names = lambda ids: [v.tokens[t] for t in ids]  # noqa: E731
    assert "[END]" not in names(late) and "[GOAL]" not in names(late) and names(late)[-1] != "[EOS]"
    infusion = lambda ids: sum(t.startswith("MED_CON//") for t in names(ids))  # noqa: E731
    assert infusion(early) == 0 and infusion(late) == 3                               # started at 1 h
    s.dataset = "eicu"                                                                # APACHE form: within 24 h
    assert "HEADER//adm_dx_group//unknown" in names(prospective(s, v, 6))
    assert "HEADER//adm_dx_group//sepsis" in names(prospective(s, v, 24))


def test_prospective_inputs_ignore_derived_concepts(meds: Path) -> None:
    """Derived concepts (treatment episodes, KDIGO, SOFA; KDIGO's urine criteria may use a weight charted later,
    flagged ``weight_charted_later``) reach only the goal block: whatever their values, the prospective input is
    the same, while the goal block follows them."""
    from icutwin.tokenization.encode import prospective
    from icutwin.tokenization.stays import _naive, build_stay, load_events, load_stays
    v = _fit(meds)
    meta = {m["subject_id"]: m for m in load_stays(meds).iter_rows(named=True)}
    (sid,), part = next(iter(load_events(meds, "train").group_by(["subject_id"], maintain_order=True)))
    m = {**meta[sid], "icu_in": _naive(meta[sid]["icu_in"]), "icu_out": _naive(meta[sid]["icu_out"])}
    rows = part.select("time", "code", "numeric_value", "text_value").rows()
    plain = build_stay(m, rows, {})
    sid = plain.stay_id
    extreme = build_stay(m, rows, {"episodes": {sid: [(e, m["icu_in"], m["icu_out"]) for e in ("imv", "vasopressor", "rrt")]},
                                   "kdigo": {sid: [(m["icu_in"], 3)]}, "sofa_24h": {sid: 24}})
    goal = lambda s: [v.tokens[t] for t in encode(s, v, goal=True).ids if v.meta[t].get("cls") == "goal"]  # noqa: E731
    assert "GOAL//kdigo_max//3+" in goal(extreme) and "GOAL//kdigo_max//unknown" in goal(plain)
    for cutoff in (0, 1, 6, 24, 72):
        assert prospective(plain, v, cutoff) == prospective(extreme, v, cutoff)


def _bad(v, ids, whole=True):
    tables = compile_tables(v)
    ok, _ = accepts(tables, ids, whole=whole)
    with pytest.raises(ParseError):
        parse(ids, v, whole=whole)
    return ok


def test_grammar_rejects_broken_sequences(meds: Path) -> None:
    v = _fit(meds)
    s = next(iter_stays(meds, "train"))
    ids = encode(s, v).ids
    tl_at = ids.index(v.id("[TL]"))
    first_int = next(i for i, t in enumerate(ids) if v.meta[t].get("cls") == "interval")
    hdr = next(i for i, t in enumerate(ids) if v.meta[t].get("cls") == "header")
    start = next(i for i, t in enumerate(ids) if v.meta[t].get("action") == "start")
    end_at = ids.index(v.id("[END]"))
    hosp, death = ids.index(v.id("HOSP//discharge_category//home")), ids.index(v.id("MEDS_DEATH"))
    apache = ids.index(v.id("HOSP_DX//AMSTERDAM_APACHE//13116-1"))
    assert ids[apache + 1] == v.id("HOSP_DX//ICD10CM")   # canonical order: code order, then [EOS]
    cases = {
        "interval first in the timeline": ids[:tl_at + 1] + [ids[first_int]] + ids[tl_at + 1:],
        "header field twice": ids[:hdr + 1] + [ids[hdr]] + ids[hdr + 1:],
        "dose change without a start": ids[:start] + ids[start + 1:],
        "interval before [END]": ids[:end_at] + [ids[first_int]] + ids[end_at:],
        "death before the hospital outcome": ids[:hosp] + [ids[death], ids[hosp]] + ids[death + 1:],
        # a whole-code diagnosis (a system that is not character-hierarchical) after an ICD-10 one
        "diagnoses out of order": ids[:apache] + ids[apache + 1:-1] + [ids[apache], ids[-1]],
        "no [EOS]": ids[:-1],
    }
    for name, bad in cases.items():
        assert not _bad(v, bad), name


def test_fitting_reads_the_train_split_only(tmp_path: Path) -> None:
    """A7: changing held-out values leaves every fitted artifact identical."""
    a = _write_meds(tmp_path / "a", SPLITS)
    b = _write_meds(tmp_path / "b", SPLITS, shift=37.0)
    fit(iter_stays(a, "train"), min_subjects=1).save(tmp_path / "a.json")
    fit(iter_stays(b, "train"), min_subjects=1).save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_fitting_and_encoding_are_deterministic(meds: Path, tmp_path: Path) -> None:
    """A8: two runs give identical artifacts."""
    _fit(meds).save(tmp_path / "1.json")
    _fit(meds).save(tmp_path / "2.json")
    assert (tmp_path / "1.json").read_bytes() == (tmp_path / "2.json").read_bytes()
    v = _fit(meds)
    assert [encode(s, v).ids for s in iter_stays(meds, "held_out")] == \
        [encode(s, v).ids for s in iter_stays(meds, "held_out")]


def test_vocabulary_marks_category_kinds(meds: Path) -> None:
    """A3: tokens of non-standard CLIF categories carry their kind."""
    v = _fit(meds)
    kind = {v.tokens[i].split("|")[0]: m.get("category_kind") for i, m in enumerate(v.meta) if m.get("code")}
    assert kind["MED_INT//ext_metoprolol//mcg//given//enteral"] == "concept"
    assert kind["MED_CON//norepinephrine//mcg/kg/min//start//iv"] == "standard"


def test_interval_tokens_cover_gaps_by_week_and_rest() -> None:
    assert tl.labels()[0] == "1-5m" and tl.labels()[-1] == "7d+"
    week = len(tl.labels()) - 1
    assert [k for k, _m in tl.split_gap(3)] == [0]
    assert [k for k, _m in tl.split_gap(2 * 10080 + 90)] == [week, week, tl.labels().index("1-2h")]
    assert tl.split_gap(10080) == [(week, 10080.0)]


def test_infusion_keys_have_every_grammar_action(meds: Path) -> None:
    """Vocabulary closure: a kept infusion key has a token for each action the infusion-state rule reads, seen in
    train or not, so a held-out 'going' event (an infusion running at the window start) is never OOV."""
    from icutwin.tokenization.vocab import infusion_code, infusion_key
    v = _fit(meds)
    key = "MED_CON//norepinephrine//mcg/kg/min//iv"
    assert infusion_key(infusion_code(key, "going")) == (key, "going")
    going = infusion_code(key, "going")                       # never in the fixture's train split
    assert v.codes[going]["kind"] == "numeric" and v.codes[going]["edges"] == v.state_keys[key]["edges"]
    assert v.fitted_on["infusion_closure_codes"] >= 1
    s = next(iter_stays(meds, "held_out"))
    s.timeline = [e for e in s.timeline if "start" not in e[2]]   # the infusion now runs from the window start
    first = min(t for t, _r, c, *_ in s.timeline if c.startswith("MED_CON"))
    s.timeline.append((s.timeline[0][0], 0, going, 0.05, None))
    s.timeline.sort(key=lambda e: (e[0], e[1], e[2]))
    ids = encode(s, v).ids
    ok, why = accepts(compile_tables(v), ids)
    assert ok, why
    assert first > s.timeline[0][0]


def test_parallel_encode_and_validate_match_one_process(tmp_path: Path) -> None:
    """fit, encode and validate over chunks of subjects in parallel processes give the one-process vocabulary and
    shards (byte-identical) and the same report."""
    from icutwin.tokenization.cli import encode_all, fit_vocab, validate_all

    meds_root, nd = tmp_path / "meds", tmp_path / "no_derived"
    _write_meds(meds_root / "mimic", SPLITS)
    one, many = tmp_path / "one", tmp_path / "many"
    v = fit_vocab(meds_root, nd, ["mimic"], one)
    fit_vocab(meds_root, nd, ["mimic"], many, workers=2, chunk_stays=1)                  # merged parts: the same vocab
    assert (one / "vocab.json").read_bytes() == (many / "vocab.json").read_bytes()
    splits = ["train", "tuning", "held_out"]
    a = encode_all(v, meds_root, nd, ["mimic"], splits, one)
    b = encode_all(v, meds_root, nd, ["mimic"], splits, many, workers=2, chunk_stays=1)
    assert a["shards"] == b["shards"] and sum(s["subjects"] for s in a["shards"].values()) > 2
    assert validate_all(v, meds_root, nd, ["mimic"], splits) == validate_all(
        v, meds_root, nd, ["mimic"], splits, workers=2, chunk_stays=1, vocab_path=one / "vocab.json")
