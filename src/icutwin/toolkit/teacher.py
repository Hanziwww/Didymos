"""The toolkit's teacher-forced passes of the gated model (``docs/toolkit.md``): the computations of the paper's host
scripts as functions of in-memory sequences, with the same definitions.

* ``states``: the last layer's hidden state at a prompt's last token and its mean over the prompt
  (``tools/server/paper_states.py``; the probes' features, the similarity index).
* ``hazards``: per rollout, the step hazards of the five causes (death, alive discharge, first IMV, vasopressor and
  RRT) renormalized as the sampler draws (the grammar's timeline rules, the dataset's support, the running infusions:
  ``eval.hazards.masked_parts`` / ``normalized``; zero before a clock block), summed per time bin while at risk on the
  labels' decode clock of the anchor (``eval.hazards.summarize``), with the capped rollout's last free position
  (``tools/server/paper_hazards.py``). ``eval.hazards.estimate`` turns an anchor's paths into E0, E1 and E2.
* ``probe_logits``: the Twin probe, b + w . [last state, mean state] (``tools/paper_probe.py``'s models).
* ``knockout``: attention knockout (amendment 9; Geva et al. 2023): attention from every position outside a group to
  the group's tokens blocked in every layer, and the probe logit's fall, both logits pooling the positions outside
  the group (``tools/server/paper_attn_diag.py``).

On a GPU in bfloat16 with SDPA attention (eager for the knockout's additive masks), as the paper's passes; on the CPU
in float32 (tests).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

END_RESERVE = 32
LONG_EDGES = np.r_[np.arange(0, 49), np.arange(54, 721, 6)].astype(np.float64)     # hourly to 48 h, 6-hourly to 30 d
SHORT_EDGES = np.arange(0, 49).astype(np.float64)


@dataclass
class HazardInput:
    """One anchor's sequences for ``Teacher.hazards``: the prompt, its continuations (``given[k]`` intervention
    tokens at the start of continuation k), the dataset (the clock's sketches), the clock id and the token cap the
    rollouts were drawn with (None: an uncapped sequence, such as a real future)."""
    prompt: np.ndarray
    conts: list
    dataset: str | None
    clock_id: str
    cap: int | None = None
    given: list = field(default_factory=list)


class Teacher:
    def __init__(self, model_dir, v, tokens_root, device: str = "cuda", tokens_per_batch: int = 65536,
                 chunk: int = 2048):
        import torch
        from transformers import AutoModelForCausalLM

        from icutwin.eval import hazards as hz
        from icutwin.generate.support import Support
        from icutwin.model.data import layout

        self.model_dir, self.v = Path(model_dir), v
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=self.dtype,
                                                          attn_implementation="sdpa").to(self.device).eval()
        self._eager = None
        self.tpb, self.chunk = tokens_per_batch, chunk
        self.g, self.nz, self.lay = hz.grammar(v), hz.normalizer(v), layout(v)
        sup = Path(tokens_root) / "support.json"
        self.support = Support.load(v, sup) if sup.exists() else None
        self.context = int(json.loads((self.model_dir / "config.json").read_text())["max_position_embeddings"])
        self._sup: dict = {}

    @property
    def eager(self):
        if self._eager is None:
            from transformers import AutoModelForCausalLM

            self._eager = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=self.dtype,
                                                               attn_implementation="eager").to(self.device).eval()
        return self._eager

    def _packs(self, lengths: np.ndarray):
        """Batches of sequence indices by length, at most ``tokens_per_batch`` padded tokens each."""
        order = np.argsort(lengths, kind="stable")
        k = 0
        while k < len(order):
            n = 1
            while k + n < len(order) and (n + 1) * lengths[order[k + n]] <= self.tpb:
                n += 1
            yield order[k:k + n]
            k += n

    def _batch(self, seqs: list, idx):
        import torch

        width = int(max(len(seqs[i]) for i in idx))
        x = torch.zeros((len(idx), width), dtype=torch.long)
        att = torch.zeros((len(idx), width), dtype=torch.long)
        for r, i in enumerate(idx):
            x[r, :len(seqs[i])] = torch.as_tensor(np.asarray(seqs[i], dtype=np.int64))
            att[r, :len(seqs[i])] = 1
        return x.to(self.device), att.to(self.device)

    # -- states and probes ----------------------------------------------------------------------------------------
    def states(self, prompts: list) -> tuple[np.ndarray, np.ndarray]:
        """(last, mean) float32 [n, d]: the last layer's state at each prompt's last token and its mean."""
        import torch

        lengths = np.array([len(p) for p in prompts])
        d = self.model.config.hidden_size
        last, mean = np.zeros((len(prompts), d), np.float32), np.zeros((len(prompts), d), np.float32)
        with torch.no_grad():
            for idx in self._packs(lengths):
                x, att = self._batch(prompts, idx)
                h = self.model(input_ids=x, attention_mask=att, output_hidden_states=True).hidden_states[-1]
                for r, i in enumerate(idx):
                    m = int(att[r].sum())
                    last[i] = h[r, m - 1].float().cpu().numpy()
                    mean[i] = h[r, :m].float().mean(0).cpu().numpy()
        return last, mean

    def probe_logits(self, prompts: list, coef: np.ndarray) -> np.ndarray:
        last, mean = self.states(prompts)
        return np.hstack([last, mean]).astype(np.float64) @ coef[1:].astype(np.float64) + float(coef[0])

    # -- hazards --------------------------------------------------------------------------------------------------
    def _support_of(self, seq):
        import torch

        tok = int(seq[self.lay.dataset_pos])
        if tok not in self._sup:
            m = self.support.mask(tok) if self.support is not None else None
            self._sup[tok] = torch.as_tensor(m if m is not None else np.ones(len(self.v.tokens), bool),
                                             device=self.device).float()
        return self._sup[tok]

    def hazards(self, items: list, edges: np.ndarray = LONG_EDGES) -> list[list]:
        """Per item (``HazardInput``) the ``eval.hazards.Path`` of each continuation (module docstring)."""
        import torch

        from icutwin.eval import hazards as hz
        from icutwin.factory import twins

        g, nz, v = self.g, self.nz, self.v
        W = self.model.get_output_embeddings().weight
        sets_dev = {c: torch.as_tensor(g.sets[c], device=self.device) for c in hz.CAUSES}
        clocks = []
        for it in items:                                 # the prompt's clock draws once, the state shared
            p = np.asarray(it.prompt, dtype=np.int64)
            ck = hz.Clock(v, it.dataset)
            rng = twins.clock_rng(str(it.clock_id), 0)
            tlp = int(np.flatnonzero(p == g.tl)[0]) if g.tl in p else 0
            mins = ck.minutes(p, rng, tlp)
            clocks.append((ck, rng.getstate(), float(mins[-1])))
        seqs, owner = [], []
        for a, it in enumerate(items):
            for k, c in enumerate(it.conts):
                seqs.append(np.concatenate([np.asarray(it.prompt, np.int64), np.asarray(c, np.int64)]))
                owner.append((a, k))
        out = [[None] * len(it.conts) for it in items]
        lengths = np.array([len(s) for s in seqs])
        with torch.no_grad():
            for idx in self._packs(lengths):
                x, att = self._batch(seqs, idx)
                hid = self.model.model(input_ids=x, attention_mask=att).last_hidden_state
                for r, i in enumerate(idx):
                    a, k = owner[i]
                    it, s = items[a], seqs[i]
                    P, n_s = len(it.prompt), len(s)
                    pos = np.arange(P - 1, n_s - 1)
                    gf = torch.as_tensor(hz.group_first(s, g)[pos], device=self.device)
                    acc: dict = {}
                    sup = self._support_of(s)
                    for c0 in range(0, len(pos), self.chunk):
                        h = hid[r, torch.as_tensor(pos[c0:c0 + self.chunk], device=self.device)]
                        for key, val in hz.masked_parts(h @ W.T, nz, sets_dev, sup, gf[c0:c0 + self.chunk]).items():
                            acc.setdefault(key, []).append(val.float().cpu().numpy())
                    parts = {key: np.concatenate(val) for key, val in acc.items()}
                    per = hz.normalized(parts, s[pos], nz, hz.CAUSES, hz.running_state(s, pos, nz)) if len(pos) \
                        else {cz: np.zeros(0) for cz in hz.CAUSES}
                    nxt_clock = g.clock[s[pos + 1]]
                    haz = {}
                    for cz in hz.CAUSES:
                        h_ = np.asarray(per[cz], dtype=np.float64)
                        h_[nxt_clock] = 0.0
                        full = np.zeros(n_s)
                        full[pos] = h_
                        haz[cz] = full
                    ck, state, m_prompt = clocks[a]
                    rng = random.Random()
                    rng.setstate(state)
                    c = s[P:]
                    mins = np.r_[np.zeros(P), ck.minutes(c, rng, 0)] + m_prompt
                    mins[:P] = m_prompt
                    hours = (mins - m_prompt) / 60.0
                    stop = n_s - 1
                    ends = np.flatnonzero(c == g.end)
                    if len(ends):
                        stop = min(stop, P + int(ends[0]) - 1)
                    if it.cap is not None:                   # the sampler's budget: the reserve's start
                        p0 = P + (it.given[k] if it.given else 0)
                        max_new = min(int(it.cap), self.context - p0 - 1)
                        stop = min(stop, p0 + max_new - END_RESERVE - 3)
                    out[a][k] = hz.summarize(s, P - 1, haz, hours, edges, g, stop_pos=stop)
                del hid
        return out

    # -- knockout -------------------------------------------------------------------------------------------------
    def knockout(self, prompt: list, coef: np.ndarray, positions: dict) -> dict:
        """Per group (``positions``: group -> token positions), the probe logit's fall under attention knockout."""
        import torch

        m = self.eager
        n = len(prompt)
        w = torch.as_tensor(coef[1:], dtype=torch.float32, device=self.device)
        b = float(coef[0])
        neg = torch.finfo(self.dtype).min
        ids = torch.as_tensor([list(prompt)], device=self.device)

        def logit(h, keep=None):
            pool = h.mean(0) if keep is None else h[keep].mean(0)
            return b + float((torch.cat([h[-1], pool]) * w).sum())

        out = {}
        with torch.no_grad():
            h_ref = m(input_ids=ids, output_hidden_states=True).hidden_states[-1][0].float()
            causal = torch.ones(n, n, dtype=torch.bool, device=self.device).tril()
            for gname, pos in positions.items():
                pos = [q for q in pos if q != n - 1]
                if not pos:
                    continue
                keep = torch.ones(n, dtype=torch.bool, device=self.device)
                keep[pos] = False
                allowed = causal & ~(keep[:, None] & ~keep[None, :])
                mask = torch.zeros(n, n, dtype=self.dtype, device=self.device).masked_fill(~allowed, neg)
                hk = m(input_ids=ids, attention_mask=mask[None, None], output_hidden_states=True).hidden_states[-1][0]
                out[gname] = logit(h_ref, keep) - logit(hk.float(), keep)
        return out


def estimates(paths: list, cause: str, horizon: float, edges: np.ndarray = LONG_EDGES) -> dict:
    """E0, E1, E2 and the censored share of one anchor's rollouts (``eval.hazards.estimate``)."""
    from icutwin.eval import hazards as hz

    first = np.array([p.first[cause] for p in paths])
    end_hour = np.array([p.end_hour for p in paths])
    end_kind = np.array([p.end_kind for p in paths]).astype(int)
    if cause == "death":
        ev, ot = np.stack([p.inc["death"] for p in paths]), np.stack([p.inc["alive"] for p in paths])
    elif cause == "alive":
        ev, ot = np.stack([p.inc["alive"] for p in paths]), np.stack([p.inc["death"] for p in paths])
    else:
        ev, ot = np.stack([p.inc[cause] for p in paths]), np.stack([p.comp[cause] for p in paths])
    if cause == "alive":                                    # a death competes with an alive discharge
        oth = np.where(end_kind == hz.END_DEAD, end_hour, np.inf)
        first = np.where(end_kind == hz.END_ALIVE, end_hour, np.inf)
        stop = np.minimum(first, end_hour)
        h = horizon if np.isfinite(horizon) else float(edges[-1])
        return {"e0": hz.e0(first, end_hour, end_kind, h),
                "e1": hz.aj(edges, stop, hz.event_counts(first, edges), hz.event_counts(oth, edges), h),
                "e2": hz.aj(edges, stop, ev, ot, h),
                "censored_before": float(np.mean((end_kind == hz.END_CENSORED) & (end_hour < h)))}
    return hz.estimate(cause, horizon, edges, first, end_hour, end_kind, ev, ot)
