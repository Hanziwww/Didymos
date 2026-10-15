"""The twin engine with vLLM (its own environment, a GPU; skipped elsewhere): a random-initialized tiny model on the
full vocabulary samples conditioned stays that carry their conditions, pass the automaton and the parser, and
decode to stays."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("vllm")
R = Path(__file__).resolve().parents[1]
VOCAB = Path(os.environ.get("ICUTWIN_TEST_VOCAB", R / "data/tokens/full/vocab.json"))


def test_engine_samples_conditioned_valid_stays() -> None:
    import torch
    from transformers import Qwen3ForCausalLM

    if not torch.cuda.is_available() or not VOCAB.exists():
        pytest.skip("a GPU and the full-scale vocabulary")
    from icutwin.generate.engine import Request, TwinEngine, conditions
    from icutwin.model.presets import qwen3_config
    from icutwin.tokenization.fsm import accepts
    from icutwin.tokenization.parser import parse
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(VOCAB)
    d = Path(tempfile.mkdtemp())
    torch.manual_seed(0)
    Qwen3ForCausalLM(qwen3_config("tiny", len(v.tokens), v.id("[PAD]"), v.id("[BOS]"), v.id("[EOS]"),
                                  max_positions=4096)).save_pretrained(d / "tiny")
    eng = TwinEngine(d / "tiny", VOCAB, gpu_memory_utilization=0.3, dtype="float32")
    sex = next(t.split("//")[-1] for t in v.tokens if t.startswith("HEADER//") and "unknown" not in t)
    field_name = next(t.split("//")[1] for t in v.tokens if t.startswith("HEADER//") and "unknown" not in t)
    goal = next(t for t in v.tokens if t.startswith("GOAL//") and "unknown" not in t)
    reqs = [Request(dataset="hirid"), Request(dataset="eicu", header={field_name: sex}),
            Request(goals={goal.split("//")[1]: goal.split("//")[2]})] * 20
    seqs = eng.sample(reqs, list(range(len(reqs))), max_new=300)
    for r, s in zip(reqs, seqs):
        assert accepts(eng.t, s)[0], accepts(eng.t, s)
        parse(s, v)
        c = conditions(v, r)
        assert all(s[1 + i] == tok for i, tok in c.prefix.items())
        assert c.header <= set(s) and c.goal <= set(s)
        stay = eng.decode(s)
        assert stay is not None
