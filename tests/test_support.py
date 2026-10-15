"""The dataset support mask (S7 constrained decoding) on the 2% subset's real vocabulary and train tokens (skipped
without them): a sample conditioned on a dataset draws only that dataset's training tokens, no state becomes dead
and the end is still reached within the reserve under the mask."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from icutwin.generate import support as gs
from icutwin.generate.constrain import END_RESERVE, NONE, Conditions, GrammarProcessor, generation_mask
from icutwin.tokenization.fsm import Automaton, compile_tables, mask_tables

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"
DATASETS = ("mimic", "eicu", "hirid", "aumc", "sicdb", "nwicu", "inspire")


@pytest.fixture(scope="module")
def setup():
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's tokens")
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    t = compile_tables(v)
    sup = gs.Support(v, gs.compute(T, list(DATASETS)))
    return v, t, mask_tables(t, len(v.tokens)), sup


def test_support_sets(setup) -> None:
    v, t, m, sup = setup
    sizes = {ds: int(sup.masks[ds].sum()) for ds in DATASETS}
    assert all(500 < n < len(v.tokens) for n in sizes.values()) and sizes["inspire"] < sizes["mimic"]
    ins = sup.mask(v.id("PREFIX//dataset//inspire"))
    assert ins is not None and sup.mask(v.id("PREFIX//dataset//unknown")) is None
    assert ins[v.id("[EOS]")]
    # interval tokens follow each dataset's training windows (#111: INSPIRE charts no gap under 5 minutes)
    by = gs.compute(T, list(DATASETS))
    iv = [i for i, mt in enumerate(v.meta) if mt.get("cls") == "interval"]
    for ds in DATASETS:
        mk, has = sup.masks[ds], set(by[ds])
        assert all(mk[i] == (i in has) for i in iv) and any(mk[i] for i in iv), ds


@pytest.mark.parametrize("ds", ["inspire", "nwicu", "hirid"])
def test_walks_under_support(setup, ds: str) -> None:
    v, t, m, sup = setup
    rng = np.random.default_rng(1)
    dtok = v.id(f"PREFIX//dataset//{ds}")
    mask_ds = sup.mask(dtok)
    for _ in range(25):
        a = Automaton(t)
        a.step(v.id("[BOS]"))
        a.step(dtok)
        states = []
        for _k in range(150):
            if a.phase == "done":
                break
            mk = generation_mask(a, m, None, NONE, None, mask_ds)
            ok = np.flatnonzero(mk)
            assert len(ok), f"dead state in {a.phase}"
            assert mask_ds[ok].all()
            states.append(copy.deepcopy(a))
            assert a.step(int(rng.choice(ok)))
        for s in states[::10]:                      # completion within the reserve, a delaying adversary
            b, left = copy.deepcopy(s), END_RESERVE
            while b.phase != "done":
                ok = np.flatnonzero(generation_mask(b, m, left, NONE, None, mask_ds))
                assert len(ok)
                pick = min(ok, key=lambda x: (0 if x == t.week else 1 if x in t.interval else 2, int(x)))
                if b.phase != "tl" or not (set(ok) & t.interval):
                    pick = int(ok[-1]) if v.id("[EOS]") not in set(ok) else v.id("[EOS]")
                assert b.step(int(pick))
                left -= 1
                assert left >= 0


def test_random_model_samples_stay_in_support(setup) -> None:
    import torch
    from transformers import GenerationConfig, LogitsProcessorList, Qwen3ForCausalLM

    from icutwin.model.presets import qwen3_config

    v, t, m, sup = setup
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(qwen3_config("tiny", len(v.tokens), v.id("[PAD]"), v.id("[BOS]"), v.id("[EOS]"))).eval()
    dtok = v.id("PREFIX//dataset//inspire")
    cond = Conditions(prefix={0: dtok})
    proc = GrammarProcessor(t, m, v.id("[PAD]"), max_length=201, conditions=cond, support=sup)
    cfg = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=200,
                           eos_token_id=v.id("[EOS]"), pad_token_id=v.id("[PAD]"))
    ids = torch.full((8, 1), v.id("[BOS]"), dtype=torch.long)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=cfg,
                             logits_processor=LogitsProcessorList([proc]))
    mk = sup.mask(dtok)
    for row in out.tolist():
        row = [x for x in row if x != v.id("[PAD]")]
        assert row[1] == dtok and all(mk[x] for x in row[2:]), [v.tokens[x] for x in row if not mk[x]][:5]
    assert proc.violations == 0
