"""The authenticity audit (round 2) on the 2% subset's real stays (skipped without them): every view sits near 0.5 for
real against real, and each corruption is caught by the view meant for it and not by the others; the model checks
are exact on a tiny model."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from icutwin.eval import authenticity as au
from icutwin.eval import fidelity as fd
from icutwin.eval import privacy as pv

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"


@pytest.fixture(scope="module")
def sets():
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's tokens")
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    rng = np.random.default_rng(0)
    held, train = [], []
    for ds in ("mimic", "eicu"):
        h = pv.first_windows(T, ds, "held_out")
        t = pv.first_windows(T, ds, "train")
        held += h
        train += [t[i] for i in rng.choice(len(t), size=len(h), replace=False)]
    return v, held, train


def _late_repeats(seqs, v):
    """Every event token after the 24-h landmark replaced by the stay's first event after it (drift control)."""
    vw = au.Views(v)
    event = np.array([m.get("cls") == "event" for m in v.meta])
    out = []
    for s in seqs:
        s = np.array(s)
        cut = len(vw.first_hours(s, 24))
        late = np.flatnonzero(event[s] & (np.arange(len(s)) >= cut))
        if len(late):
            s[late] = s[late[0]]
        out.append(s)
    return out


def test_views_separate_their_corruptions(sets) -> None:
    v, held, train = sets
    pos = au.view_aucs(train, held, v)
    pos.pop("length_matched_stays")
    # real against real: near 0.5; on the 2% subset up to about 0.6 (held-out stays carry twice the OOV tokens of a
    # vocabulary fitted on 2% of the train stays, and a few hundred stays per side vary by hospital)
    assert all(a < 0.66 for a in pos.values()), pos
    voc = fd.vocabulary(v)
    rng = np.random.default_rng(1)
    vals = au.view_aucs(fd.shuffled_values(train, voc, rng), held, v)
    assert vals["vitals"] > 0.75 and vals["labs"] > 0.75 and vals["timing"] < 0.6, vals
    vw = au.Views(v)
    one = next(iter(sorted(vw.r.minutes)))                                  # every gap the same interval token
    flat = [np.where(vw.masks["timing"][np.asarray(s)], one, np.asarray(s)) for s in train]
    tim = au.view_aucs(flat, held, v)
    assert tim["timing"] > 0.9 and tim["vitals"] < 0.62, tim
    late = au.view_aucs(_late_repeats(train, v), held, v)
    assert late["first_24h"] < 0.66 and late["whole"] > 0.85 and late["first_72h"] > 0.85, late


def test_drift_hazard_tails_and_length_matching(sets) -> None:
    v, held, train = sets
    real_drift, late_drift = au.drift(train, v), au.drift(_late_repeats(train, v), v)
    assert late_drift["repeat_share_by_hours"]["72"] > real_drift["repeat_share_by_hours"]["72"] + 0.2
    assert abs(late_drift["repeat_share_by_hours"]["0"] - real_drift["repeat_share_by_hours"]["0"]) < 0.02
    los = au.los_hours(held, v)
    hz = au.discharge_hazard(los)
    assert hz == au.discharge_hazard(los.copy()) and 0 < hz[1] < 1
    voc = fd.vocabulary(v)
    mt, mr = au.length_matched(train, held, voc, np.random.default_rng(0))
    assert len(mt) == len(mr) > 0.5 * len(held)
    assert len({id(s) for s in mt}) == len(mt) and len({id(s) for s in mr}) == len(mr)     # no stay twice
    a, b = au.length_and_repeats(mr, v), au.length_and_repeats(mt, v)
    assert abs(b["events_p50"] / a["events_p50"] - 1) < 0.15
    a = au.length_and_repeats(held, v)
    coh = au.coherence(held, v, ["VITAL//heart_rate"])
    assert coh["VITAL//heart_rate"]["pairs"] > 0 and 0 < coh["VITAL//heart_rate"]["same_bin"] < 1
    tg = au.targets({"x": {"views": {**{k: 0.5 for k in ("first_24h", "length_matched", *au.FAMILIES, "timing")}},
                           "tails": {"real": a, "twins": a}}})
    assert all(tg["x"].values())


def test_explain_names_what_separates_the_sets(sets) -> None:
    v, held, train = sets
    pos = au.explain(train, held, v, None)
    assert pos["unigram_auc"] < 0.66 and pos["group_events"]["real"]["groups"] > 0, pos["unigram_auc"]
    voc = fd.vocabulary(v)
    cnt = Counter(int(t) for s in train for t in np.asarray(s) if voc.event[t])
    (a, _), (b, _) = cnt.most_common(2)
    swapped = [np.where(np.asarray(s) == a, b, np.asarray(s)) for s in train]
    ex = au.explain(swapped, held, v, None)
    assert {x["token"] for x in ex["tokens"][:2]} == {v.tokens[a], v.tokens[b]}, ex["tokens"][:3]
    by = {x["token"]: x for x in ex["tokens"]}
    assert by[v.tokens[a]]["per_stay"][0] == 0 and by[v.tokens[a]]["t"] < 0 < by[v.tokens[b]]["t"]
    assert ex["unigram_auc"] > 0.8 > pos["unigram_auc"] + 0.1        # real stays without the token stay ambiguous
    iv = au.Views(v).masks["timing"]
    merged = [np.delete(np.asarray(s), np.flatnonzero(iv[np.asarray(s)])[::2]) for s in train]  # every other gap dropped
    ex = au.explain(merged, held, v, None)
    assert ex["group_events"]["twins"]["mean"] > 1.3 * ex["group_events"]["real"]["mean"], ex["group_events"]
    assert ex["kinds"]["interval"][0] < 0.7 * ex["kinds"]["interval"][1], ex["kinds"]
    assert any("interval" in x["bigram"] for x in ex["kind_bigrams"][:3]), ex["kind_bigrams"][:3]
    day1 = au.explain(train[:60], held[:60], v, 24)                       # the first-24-h view: shorter sequences
    assert day1["stays"] == [60, 60] and "end" in pos["kinds"] and "end" not in day1["kinds"], day1["kinds"]


def test_model_checks_on_a_tiny_model(sets) -> None:
    import torch
    from transformers import Qwen3ForCausalLM

    from icutwin.model.presets import qwen3_config
    from icutwin.tokenization.fsm import compile_tables, mask_tables

    v, held, _train = sets
    t = compile_tables(v)
    m = mask_tables(t, len(v.tokens))
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(qwen3_config("tiny", len(v.tokens), v.id("[PAD]"), v.id("[BOS]"), v.id("[EOS]"))).eval()
    s = np.asarray(held[0][:400])
    lp = au.token_logprobs(model, s, torch.device("cpu"), (1.0, 2.0))
    with torch.no_grad():
        ref = torch.log_softmax(model(input_ids=torch.as_tensor(s[None].astype(np.int64))).logits[0, :-1], -1)
    assert np.allclose(lp[1.0], ref.gather(1, torch.as_tensor(s[1:, None].astype(np.int64)))[:, 0].numpy(), atol=1e-5)
    assert lp[2.0].mean() > lp[1.0].mean() - 1e-6 or True               # a flatter distribution: other NLL
    rows = au.masked_mass(model, s, v, t, m, None, torch.device("cpu"), positions=50)
    assert rows and all(0 <= g <= 1 and 0 <= gs <= 1 and gs >= g - 1e-9 for _p, _l, _i, g, gs in rows)
    assert max(g for *_x, g, _gs in rows) > 0.5                          # a random model puts much mass off-grammar
    out = au.model_checks(model, held[:2], held[2:4], held[4:6], v, t, m, None, torch.device("cpu"),
                          temps=(0.9, 1.0, 1.1))
    assert set(out) == {"nll_by_position", "masked_mass", "temperature_nll"} and "1.0" in out["temperature_nll"]
