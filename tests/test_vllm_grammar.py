"""vLLM (its own environment, a GPU; skipped elsewhere): the grammar processor under vLLM with a random-initialized tiny
model on the full vocabulary. Every constrained sample must be accepted by the automaton and the independent parser
and end within its budget; unconstrained requests in the same batch are left alone; greedy decoding equals the
transformers reference (same masks, same argmax)."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np

import pytest

pytest.importorskip("vllm")
R = Path(__file__).resolve().parents[1]
VOCAB = Path(os.environ.get("ICUTWIN_TEST_VOCAB", R / "data/tokens/full/vocab.json"))
os.environ["ICUTWIN_VOCAB"] = str(VOCAB)


def test_vllm_grammar_processor() -> None:
    import torch

    if not torch.cuda.is_available() or not VOCAB.exists():
        pytest.skip("a GPU and the full-scale vocabulary")
    from transformers import GenerationConfig, LogitsProcessorList, Qwen3ForCausalLM
    from vllm import LLM, SamplingParams

    from icutwin.generate.constrain import GrammarProcessor
    from icutwin.generate.vllm_grammar import GrammarLogitsProcessor
    from icutwin.model.presets import qwen3_config
    from icutwin.tokenization.fsm import accepts, compile_tables, mask_tables
    from icutwin.tokenization.parser import parse
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(VOCAB)
    t, m = compile_tables(v), None
    m = mask_tables(t, len(v.tokens))
    bos, eos, pad = v.id("[BOS]"), v.id("[EOS]"), v.id("[PAD]")
    d = Path(tempfile.mkdtemp())
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(qwen3_config("tiny", len(v.tokens), pad, bos, eos, max_positions=4096))
    model.save_pretrained(d / "tiny")
    llm = LLM(model=str(d / "tiny"), skip_tokenizer_init=True, dtype="float32", max_model_len=4096,
              gpu_memory_utilization=0.3, logits_processors=[GrammarLogitsProcessor], seed=0,
              enable_prefix_caching=False)
    max_new = 400
    con = [SamplingParams(temperature=1.0, top_p=1.0, top_k=0, max_tokens=max_new, seed=i, stop_token_ids=[eos],
                          detokenize=False, extra_args={"icutwin": {"max_new": max_new}}) for i in range(300)]
    free = [SamplingParams(temperature=1.0, top_p=1.0, top_k=0, max_tokens=50, seed=1000 + i, stop_token_ids=[eos],
                           detokenize=False) for i in range(20)]
    outs = llm.generate([{"prompt_token_ids": [bos]}] * len(con + free), con + free, use_tqdm=False)
    seqs = [[bos, *o.outputs[0].token_ids] for o in outs]
    bad, ended = 0, 0
    for s in seqs[:len(con)]:
        ok, why = accepts(t, s)
        ended += s[-1] == eos
        if not ok:
            bad += 1
            print("rejected:", why)
            continue
        parse(s, v)
    free_valid = sum(accepts(t, s, whole=False)[0] for s in seqs[len(con):])
    print(json.dumps({"constrained": len(con), "rejected": bad, "ended": ended,
                      "mean_len": float(np.mean([len(s) for s in seqs[:len(con)]])),
                      "unconstrained_grammatical": free_valid, "unconstrained": len(free)}))
    assert bad == 0 and ended == len(con)
    assert free_valid < len(free)   # a random model unconstrained breaks the grammar: the processor left them alone
    # greedy parity with the transformers reference: same masks, same argmax
    g = [SamplingParams(temperature=0.0, max_tokens=120, stop_token_ids=[eos], detokenize=False,
                        extra_args={"icutwin": {"max_new": 120}})]
    vg = [bos, *llm.generate([{"prompt_token_ids": [bos]}], g, use_tqdm=False)[0].outputs[0].token_ids]
    model = model.to("cuda").eval()
    proc = GrammarProcessor(t, m, pad, max_length=121)
    cfg = GenerationConfig(do_sample=False, max_new_tokens=120, eos_token_id=eos, pad_token_id=pad)
    with torch.no_grad():
        hg = model.generate(input_ids=torch.tensor([[bos]], device="cuda"), generation_config=cfg,
                            logits_processor=LogitsProcessorList([proc]))[0].tolist()
    hg = hg[:hg.index(eos) + 1] if eos in hg else hg
    print("greedy parity:", vg == hg, len(vg), len(hg))
    assert vg == hg
