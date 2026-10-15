"""The toolkit's samplers (``docs/toolkit.md``): rollouts of prompts (``simulate``) and new twins (``sample``), both
grammar-constrained, ancestral at temperature 1 without truncation, as the paper drew them.

* ``VllmSampler``: ``generate.engine.TwinEngine`` (vLLM, Linux, one GPU; ``toolkit.stage`` runs it in vLLM's
  environment). All requests of a call go to one ``generate``; each request's rollouts carry their own seeds
  (``generate.engine.rollout_requests``), so a request's draws do not depend on its batch.
* ``HfSampler``: the reference, with transformers and the grammar's reference processor
  (``generate.constrain.GrammarProcessor``: the same masks, the dataset's support, the budget rule; the header
  augmentation undone for new twins, as the vLLM processor does). One request at a time (its n rollouts share one
  prompt length); seeded per request. For tests and small jobs.

Both return, per request id, its continuations: the intervention's tokens (the gap and the given events, when given)
followed by the sampled tokens to ``[EOS]`` (as ``TwinEngine.simulate``); for new twins the whole sequence.
"""

from __future__ import annotations

from pathlib import Path


def _intervention(r):
    from icutwin.generate.engine import Intervention

    if not r.give and not r.withhold:
        return None
    return Intervention(give=tuple(int(x) for x in r.give), withhold=frozenset(int(x) for x in r.withhold), gap=r.gap)


class VllmSampler:
    def __init__(self, model_dir, vocab_path, gpu_memory_utilization: float = 0.85, seed: int = 0):
        from icutwin.generate.engine import TwinEngine

        self.eng = TwinEngine(model_dir, vocab_path, gpu_memory_utilization=gpu_memory_utilization, seed=seed)

    def simulate(self, reqs: list) -> dict:
        from vllm import SamplingParams

        from icutwin.generate.engine import rollout_requests

        e = self.eng
        eos = e.v.id("[EOS]")
        prompts, params, owner = [], [], []
        for r in reqs:
            ps, args = rollout_requests([list(r.prompt)], r.n, r.seed, e.context, e.v, e.t, _intervention(r), r.max_new)
            for p, a in zip(ps, args, strict=True):
                prompts.append(p)
                params.append(SamplingParams(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, max_tokens=a["max_tokens"],
                                             seed=a["seed"], stop_token_ids=[eos], detokenize=False,
                                             extra_args={"icutwin": a["icutwin"]}))
                owner.append(r)
        outs = e.llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        res: dict = {}
        for p, r, o in zip(prompts, owner, outs, strict=True):
            res.setdefault(r.id, []).append(p[len(r.prompt):] + list(o.outputs[0].token_ids))
        return res

    def sample(self, reqs: list) -> dict:
        from icutwin.generate.engine import Request

        rq = [Request(r.dataset, r.site, dict(r.header), dict(r.goals), dict(r.prefix)) for r in reqs]
        caps = {r.max_new for r in reqs}
        out: dict = {}
        for cap in sorted(caps, key=lambda c: -1 if c is None else c):          # one token budget per call
            idx = [k for k, r in enumerate(reqs) if r.max_new == cap]
            seqs = self.eng.sample([rq[k] for k in idx], [reqs[k].seed for k in idx], max_new=cap)
            out.update({reqs[k].id: s for k, s in zip(idx, seqs, strict=True)})
        return out


class HfSampler:
    def __init__(self, model_dir, vocab_path, device: str = "cpu", dtype: str | None = None):
        import json

        import torch
        from transformers import AutoModelForCausalLM

        from icutwin.generate.support import Support
        from icutwin.tokenization.fsm import compile_tables, mask_tables
        from icutwin.tokenization.vocab import Vocab

        self.model_dir = Path(model_dir)
        self.v = Vocab.load(Path(vocab_path))
        self.t = compile_tables(self.v)
        self.m = mask_tables(self.t, len(self.v.tokens))
        sup = Path(vocab_path).parent / "support.json"
        self.support = Support.load(self.v, sup) if sup.exists() else None
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        dt = getattr(torch, dtype) if dtype else (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=dt).to(self.device).eval()
        self.context = int(json.loads((self.model_dir / "config.json").read_text())["max_position_embeddings"])

    def _generate(self, prompt: list, n: int, seed: int, budget: int, cond, undo=None) -> list:
        import numpy as np
        import torch
        from transformers import GenerationConfig, LogitsProcessorList

        from icutwin.generate.constrain import GrammarProcessor

        v = self.v
        proc = GrammarProcessor(self.t, self.m, v.id("[PAD]"), max_length=len(prompt) + budget, conditions=cond,
                                header_augmentation=undo, support=self.support)
        cfg = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=budget,
                               eos_token_id=v.id("[EOS]"), pad_token_id=v.id("[PAD]"), use_cache=True)
        torch.manual_seed(int(seed) % (2 ** 63))
        ids = torch.tensor([list(prompt)] * n, dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=cfg,
                                      logits_processor=LogitsProcessorList([proc]))
        res = []
        for row in out[:, len(prompt):].cpu().numpy():
            stop = np.flatnonzero(row == v.id("[EOS]"))
            res.append([int(x) for x in (row[:stop[0] + 1] if len(stop) else row)])
        return res

    def simulate(self, reqs: list) -> dict:
        from icutwin.generate.constrain import NONE, Conditions
        from icutwin.generate.engine import intervene

        out = {}
        for r in reqs:
            iv = _intervention(r)
            prompt = intervene(list(r.prompt), iv, self.v, self.t) if iv is not None else list(r.prompt)
            budget = self.context - len(prompt) - 1
            if r.max_new is not None:
                budget = min(budget, int(r.max_new))
            cond = Conditions(forbid=frozenset(int(x) for x in r.withhold)) if r.withhold else NONE
            given = prompt[len(r.prompt):]
            out[r.id] = [given + c for c in self._generate(prompt, r.n, r.seed, budget, cond)]
        return out

    def sample(self, reqs: list) -> dict:
        from icutwin.generate import engine as en
        from icutwin.generate.constrain import HeaderAugmentation

        order, undo = en.HeaderOrder(self.v), HeaderAugmentation(self.v, en.header_unknown_rate(self.model_dir))
        bos = self.v.id("[BOS]")
        out = {}
        for r in reqs:
            cond = en.conditions(self.v, en.Request(r.dataset, r.site, dict(r.header), dict(r.goals), dict(r.prefix)))
            budget = (r.max_new or self.context - 1)
            out[r.id] = order([bos, *self._generate([bos], 1, r.seed, budget, cond, undo)[0]])
        return out
