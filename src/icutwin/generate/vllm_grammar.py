"""The grammar mask as a vLLM logits processor (plan S7: vLLM with per-request state; the V2 model runner's
interface, vLLM >= 0.31).

A request takes part through ``SamplingParams(extra_args={"icutwin": {"max_new": N, "conditions": {...}}})``
(``Conditions.to_args()``):
an automaton per request slot follows the request's tokens, and the logits of the tokens
:func:`icutwin.generate.constrain.generation_mask` does not allow are set to -inf, as the transformers reference
(:class:`icutwin.generate.constrain.GrammarProcessor`) does. Requests without the key are not touched. The
vocabulary is ``$ICUTWIN_VOCAB`` (vocab.json); the processor is registered with
``LLM(..., logits_processors=[GrammarLogitsProcessor])``.

Each step's rows carry the token fed at their input position; the automaton consumes the request's tokens up to
it. A slot's first step (or the first after a preemption, which re-adds the request) reads the tokens before it
from the runner's token history once. No speculative decoding: one logits row per request.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext, LogitsProcessor, LogitsProcRequestState

from icutwin.generate.constrain import Conditions, HeaderAugmentation, Obligations, generation_mask
from icutwin.generate.engine import goal_plan
from icutwin.generate.goal_bindings import adopted
from icutwin.generate.support import Support
from icutwin.tokenization.fsm import Automaton, compile_tables, mask_tables
from icutwin.tokenization.vocab import Vocab

KEY = "icutwin"


class _Slot:
    __slots__ = ("a", "fed", "max_new", "cond", "obl", "dataset_token")

    def __init__(self, a: Automaton, max_new: int | None, cond: Conditions, plan: dict | None = None):
        self.a, self.fed, self.max_new, self.cond, self.obl = a, 0, max_new, cond, Obligations(cond, plan)
        self.dataset_token = None

    def feed(self, tokens) -> None:
        for tok in tokens:
            if self.a.phase == "prefix" and self.a.i == 0:
                self.dataset_token = int(tok)
            if self.a.phase == "done" or self.a.violation is not None or not self.a.step(int(tok)):
                break
            self.obl.feed(int(tok))
        self.fed += len(tokens)


class GrammarLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, req_states: LogitsProcRequestState):
        self.rs = req_states
        v = Vocab.load(Path(os.environ["ICUTWIN_VOCAB"]))
        self.t = compile_tables(v)
        self.m = mask_tables(self.t, len(v.tokens))
        self.eos = v.id("[EOS]")
        self.slots: dict[int, _Slot] = {}
        # the model's training-time header augmentation, undone in the header phase (constrain.HeaderAugmentation)
        self.undo = HeaderAugmentation(v, float(os.environ.get("ICUTWIN_HEADER_UNKNOWN", "0") or 0))
        sup = os.environ.get("ICUTWIN_SUPPORT")
        self.support = Support.load(v, Path(sup)) if sup and Path(sup).exists() else None
        # what a drawn goal binds (goal-first sampling): the bindings each dataset adopts, beside the vocabulary
        # (generate.goal_bindings); without the file a drawn goal is conditioning only
        gb = Path(os.environ["ICUTWIN_VOCAB"]).parent / "goal_bindings.json"
        self.plan = goal_plan(v, adopted(v, json.loads(gb.read_text(encoding="utf-8"))) if gb.exists() else {})

    @classmethod
    def validate_params(cls, sampling_params) -> None:
        cfg = (getattr(sampling_params, "extra_args", None) or {}).get(KEY)
        if cfg is not None and not isinstance(cfg, dict):
            raise ValueError(f"extra_args['{KEY}'] must be a dict")

    def add_request(self, req_idx: int, sampling_params) -> bool:
        cfg = (getattr(sampling_params, "extra_args", None) or {}).get(KEY)
        if cfg is None:
            self.slots.pop(req_idx, None)
            return False
        self.slots[req_idx] = _Slot(Automaton(self.t), int(cfg.get("max_new") or 0) or None,
                                    Conditions.from_args(cfg.get("conditions")), self.plan)
        return True

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        slots = ctx.idx_mapping_np
        if logits.shape[0] != len(slots):
            raise ValueError("the grammar processor needs one logits row per request (no speculative decoding)")
        active = [(r, int(s)) for r, s in enumerate(slots) if int(s) in self.slots]
        if not active:
            return logits
        pos = ctx.pos.cpu().numpy()
        inp = ctx.input_ids.cpu().numpy()
        masks = np.zeros((len(active), logits.shape[1]), dtype=bool)
        for k, (r, s) in enumerate(active):
            st, p = self.slots[s], int(pos[r])
            if st.fed < p:                                   # the prompt (or a resumed request's tokens)
                st.feed(self.rs.all_token_ids.gpu[s, st.fed:p].cpu().tolist())
            if st.fed == p:
                st.feed([int(inp[r])])
            if st.a.phase == "done" or st.a.violation is not None:
                masks[k, self.eos] = True
                continue
            generated = p + 1 - int(self.rs.prompt_len.np[s])
            left = None if st.max_new is None else st.max_new - generated - 1
            sup = self.support.mask(st.dataset_token) if self.support is not None else None
            masks[k, :self.m.n] = generation_mask(st.a, self.m, left, st.cond, st.obl, sup)
            if not masks[k].any():
                masks[k, self.eos] = True
        rows = torch.tensor([r for r, _s in active], device=logits.device)
        keep = torch.from_numpy(masks).to(logits.device)
        logits[rows] = logits[rows].masked_fill(~keep, float("-inf"))
        if self.undo.rate > 0:
            for k, (r, s) in enumerate(active):
                a = self.slots[s].a
                if a.phase == "header" and a.violation is None:
                    row = logits[r, :self.m.n].float().cpu().numpy()
                    logits[r, :self.m.n] = torch.from_numpy(self.undo(row, a)).to(logits.device, logits.dtype)
        return logits
