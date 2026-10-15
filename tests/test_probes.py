"""Linear probes (S8 *Representations*) on the 2% subset's held-out stays with a tiny random-initialized model
(skipped without the subset): the pipeline runs end to end, embeddings have one row per window, and the TF-IDF
baseline carries signal where the outcome has enough positives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from icutwin.eval import probes as pb
from icutwin.eval import utility as ut
from icutwin.eval.forecast import Reader

T = Path(__file__).resolve().parents[1] / "data" / "tokens" / "subset"


def test_probes_run_end_to_end() -> None:
    if not (T / "vocab.json").exists():
        pytest.skip("the subset's tokens")
    import torch
    from transformers import Qwen3ForCausalLM

    from icutwin.eval import privacy as pv
    from icutwin.model.presets import qwen3_config
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(T / "vocab.json")
    held = pv.first_windows(T, "eicu", "held_out") + pv.first_windows(T, "mimic", "held_out")
    w, lab = ut.examples(held, Reader(v), 24.0)
    w = [x[:1500] for x in w[:240]]
    lab = {k: y[:240] for k, y in lab.items()}
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(qwen3_config("tiny", len(v.tokens), v.id("[PAD]"), v.id("[BOS]"), v.id("[EOS]"))).eval()
    last, mean = pb.embed(model, w[:10], torch.device("cpu"))
    assert last.shape == mean.shape and last.shape[0] == 10 and np.isfinite(last).all()
    rep = pb.evaluate(model, w, lab, len(v.tokens), torch.device("cpu"))
    assert set(rep) == set(ut.TASKS)
    assert rep["death"]["tfidf_baseline"] is None or rep["death"]["tfidf_baseline"] > 0.6
