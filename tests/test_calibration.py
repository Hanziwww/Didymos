"""Teacher-forced hazard calibration (round 3, step 0a): on a toy process whose next-token probabilities are known,
the expected counts match the observed ones for every outcome (observed over expected near 1, overall and by ICU
day), a model off by a factor is caught, censoring and new onset are handled, and the model wrapper sums the softmax
over a set."""

from __future__ import annotations

import numpy as np

from icutwin.eval import calibration as cb

BOS, ADM, ALIVE, DEAD, A, X, H1, END = range(8)
Q_DIS, Q_A, P_DEAD = 0.02, 0.01, 0.2
TG = cb.Targets(adm={ADM}, dis={ALIVE, DEAD}, dead={DEAD}, end=END,
                sets={"imv": {90}, "vasopressor": {91}, "rrt": {A}}, minutes={H1: 60.0})


def _stays(n: int, seed: int = 0, cap: int = 400) -> list:
    """After the admission, each step: discharge with Q_DIS (dead with P_DEAD), else A with Q_A, else an hour
    interval or another event; a stay reaching ``cap`` tokens ends without a discharge (censored)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ids = [BOS, X, H1, ADM]
        while len(ids) < cap:
            if rng.random() < Q_DIS:
                ids.append(DEAD if rng.random() < P_DEAD else ALIVE)
                break
            ids.append(A if rng.random() < Q_A else (H1 if rng.random() < 0.5 else X))
        out.append(ids + [END])
    return out


def _true(ids: list, scale: float = 1.0) -> dict:
    n = len(ids) - 1
    return {"imv": np.zeros(n), "vasopressor": np.zeros(n), "rrt": np.full(n, scale * (1 - Q_DIS) * Q_A),
            "icu_discharge": np.full(n, Q_DIS), "icu_death": np.full(n, P_DEAD * Q_DIS)}


def test_a_calibrated_model_has_observed_over_expected_near_one() -> None:
    stays = _stays(3000)
    acc = cb.Accumulator()
    for ids in stays:
        acc.add(cb.stay_hazards(ids, _true(ids), TG))
    s = acc.summary()
    assert s["stays"] == 3000
    for k in ("rrt", "icu_discharge", "icu_death"):
        assert s[k]["all"]["observed"] > 300, (k, s[k]["all"])
        assert 0.9 < s[k]["all"]["o_e"] < 1.1, (k, s[k]["all"])
        lo, hi = s[k]["all"]["o_e_ci"]
        assert lo < 1 < hi, (k, s[k]["all"])
        for day in ("1", "2"):
            if s[k][day]["observed"] >= 100:
                assert 0.8 < s[k][day]["o_e"] < 1.2, (k, day, s[k][day])
    assert s["imv"]["all"]["o_e"] is None and s["imv"]["all"]["observed"] == 0
    # a model that doubles the treatment's probability is caught
    acc2 = cb.Accumulator()
    for ids in stays:
        acc2.add(cb.stay_hazards(ids, _true(ids, scale=2.0), TG))
    assert 0.4 < acc2.summary()["rrt"]["all"]["o_e"] < 0.6


def test_risk_windows_new_onset_and_censoring() -> None:
    p = lambda ids: {k: np.ones(len(ids) - 1) for k in cb.OUTCOMES}  # noqa: E731  each prediction counts 1
    # admission at 3; A at 6 ends the treatment's risk (predictions of tokens 4..6); the alive discharge at 8 ends
    # the discharge's and death's (tokens 4..8)
    ids = [BOS, X, H1, ADM, X, X, A, X, ALIVE, END]
    r = cb.stay_hazards(ids, p(ids), TG)
    assert r["rrt"][0].sum() == 3 and r["rrt"][1].sum() == 1
    assert r["icu_discharge"][0].sum() == 5 and r["icu_discharge"][1].sum() == 1
    assert r["icu_death"][0].sum() == 5 and r["icu_death"][1].sum() == 0
    # an event before the admission: not new onset, the stay is not at risk for it
    pre = [BOS, A, H1, ADM, X, ALIVE, END]
    assert "rrt" not in cb.stay_hazards(pre, p(pre), TG) and "icu_discharge" in cb.stay_hazards(pre, p(pre), TG)
    # censored: no discharge, at risk up to and including the timeline's end
    cens = [BOS, X, ADM, X, X, END]
    r = cb.stay_hazards(cens, p(cens), TG)
    assert r["icu_discharge"][0].sum() == 3 and r["icu_discharge"][1].sum() == 0
    assert cb.stay_hazards([BOS, X, END], p([BOS, X, END]), TG) == {}          # no admission: nothing


def test_masked_probs_follow_the_generation_mask() -> None:
    """Under the generation mask (grammar and the dataset's support) a uniform model gives each set its share of
    the allowed tokens at the picked positions, NaN elsewhere (the subset's tokens; skipped without them)."""
    from pathlib import Path

    import pytest
    import torch
    from transformers import Qwen3ForCausalLM

    from icutwin.eval import privacy as pv
    from icutwin.eval.forecast import Reader
    from icutwin.generate import support as gs
    from icutwin.generate.constrain import NONE, generation_mask
    from icutwin.model.presets import qwen3_config
    from icutwin.tokenization.fsm import Automaton, compile_tables, mask_tables
    from icutwin.tokenization.vocab import Vocab

    T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's tokens")
    v = Vocab.load(T / "vocab.json")
    tg = cb.targets(Reader(v))
    t, sup = compile_tables(v), gs.Support(v, gs.compute(T, ["mimic"]))
    m = mask_tables(t, len(v.tokens))
    model = Qwen3ForCausalLM(qwen3_config("tiny", len(v.tokens), v.id("[PAD]"), v.id("[BOS]"), v.id("[EOS]"))).eval()
    with torch.no_grad():
        model.model.embed_tokens.weight.zero_()             # tied output layer: every logit 0, a uniform model
    ids = [int(x) for x in pv.first_windows(T, "mimic", "held_out")[0]]
    adm, _dis, limit = cb.risk_span(ids, tg)
    pick = np.arange(adm, limit)[::7]
    index = {k: torch.as_tensor(sorted(tg.sets[k])) for k in cb.TREATMENTS}
    p = cb.masked_probs(model, ids, index, t, m, sup, torch.device("cpu"), pick)
    assert np.isnan(p["imv"][adm + 1]) and not np.isnan(p["imv"][pick]).any()
    a, dtok = Automaton(t), None
    for pos, tok in enumerate(ids):
        if pos - 1 in set(pick.tolist()):
            g = generation_mask(a, m, None, NONE, None, sup.mask(dtok))
            for k in cb.TREATMENTS:
                want = g[sorted(tg.sets[k])].sum() / g.sum()
                assert abs(p[k][pos - 1] - want) < 1e-5, (k, pos)
        if a.phase == "prefix" and a.i == 0:
            dtok = tok
        a.step(tok)


def test_token_qa_adds_up_to_the_calibration() -> None:
    """The treatment QA's expected first events, by token and by position, sum to the calibration's expected count
    over the same stays, and its observed ones to the calibration's (the subset's tokens; skipped without them)."""
    from pathlib import Path

    import pytest
    import torch
    from transformers import Qwen3ForCausalLM

    from icutwin.eval import privacy as pv
    from icutwin.eval.forecast import Reader
    from icutwin.model.presets import qwen3_config
    from icutwin.tokenization.vocab import Vocab

    T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's tokens")
    v = Vocab.load(T / "vocab.json")
    tg = cb.targets(Reader(v))
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(qwen3_config("tiny", len(v.tokens), v.id("[PAD]"), v.id("[BOS]"), v.id("[EOS]"))).eval()
    dev = torch.device("cpu")
    index = {k: torch.as_tensor(sorted(tg.sets[k])) for k in cb.TREATMENTS}
    index.update(icu_discharge=torch.as_tensor(sorted(tg.dis)), icu_death=torch.as_tensor(sorted(tg.dead)))
    qa, acc = cb.TokenQA(tg, v), cb.Accumulator()
    for ids in pv.first_windows(T, "mimic", "held_out")[:15]:
        qa.add(model, ids, dev)
        acc.add(cb.stay_hazards(ids, cb.set_probs(model, ids, index, dev), tg))
    s, q = acc.summary(), qa.summary(top=10_000)
    for k in cb.TREATMENTS:
        e_tok = sum(x["expected"] for x in q[k]["tokens"])
        e_pos = sum(x["expected"] for x in q[k]["by_position"].values())
        assert abs(e_tok - s[k]["all"]["expected"]) < 1e-3 * max(1.0, s[k]["all"]["expected"]), k
        assert abs(e_pos - e_tok) < 1e-3 * max(1.0, e_tok), k
        o_tok = sum(x["observed"] for x in q[k]["tokens"]) + q[k]["observed_tokens_not_listed"]
        assert o_tok == s[k]["all"]["observed"] == sum(x["observed"] for x in q[k]["by_position"].values()), k
        cells = [c for row in q[k]["by_day_position"].values() for c in row.values()]
        assert abs(sum(c["expected"] for c in cells) - e_tok) < 1e-3 * max(1.0, e_tok), k
        assert sum(c["observed"] for c in cells) == o_tok, k
        for d, row in q[k]["by_day_position"].items():        # the day margin is the calibration's by-day count
            assert abs(sum(c["expected"] for c in row.values()) - s[k][d]["expected"]) < 1e-3 * max(1.0, e_tok), (k, d)


def test_set_probs_sums_the_softmax_over_a_set() -> None:
    import torch
    from transformers import Qwen3ForCausalLM

    from icutwin.model.presets import qwen3_config

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(qwen3_config("tiny", 40, 0, 1, 2)).eval()
    ids = [1, 5, 9, 7, 30, 12, 3]
    index = {"a": torch.tensor([5, 9]), "b": torch.tensor([30])}
    p = cb.set_probs(model, ids, index, torch.device("cpu"))
    with torch.no_grad():
        ref = torch.softmax(model(input_ids=torch.tensor([ids])).logits[0, :-1].float(), -1)
    assert np.allclose(p["a"], ref[:, [5, 9]].sum(-1).numpy(), atol=1e-5)
    assert np.allclose(p["b"], ref[:, 30].numpy(), atol=1e-5) and len(p["a"]) == len(ids) - 1
