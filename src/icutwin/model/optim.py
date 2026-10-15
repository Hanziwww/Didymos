"""Round 3, amendment 2 (``docs/m7_refinement.md``): Muon for the hidden weight matrices with AdamW for the rest, as
one optimizer, so the learning-rate schedule, gradient accumulation and Accelerate's checkpoints see a single torch
optimizer.

Muon (Jordan et al. 2024; ``torch.optim.Muon``) is momentum SGD whose update of each weight matrix is orthogonalized
by Newton-Schulz iterations: every direction of the update moves at the same rate, so directions with small
gradients, such as those of rare events, learn as fast as frequent ones. Moonshot's scaling (``match_rms_adamw``:
0.2 x sqrt(max(rows, columns))) matches the update's RMS to AdamW's, so an AdamW recipe's learning rate and weight
decay carry over (Liu et al. 2025, "Muon is scalable for LLM training"). The embeddings (tied with the output layer)
and the norms stay on AdamW, as in every Muon recipe. Data parallel only: Muon needs each matrix whole (no FSDP
shards)."""

from __future__ import annotations

import torch


class MuonAdamW(torch.optim.Optimizer):
    """``torch.optim.Muon`` over ``muon`` (2-D hidden matrices) and ``torch.optim.AdamW`` over ``adamw`` (parameter
    groups), exposed as one optimizer: ``param_groups`` are the two optimizers' own groups, so a scheduler's learning
    rates reach both; ``state_dict`` nests theirs."""

    def __init__(self, muon: list, adamw: list, lr: float, weight_decay: float, betas=(0.9, 0.95), eps: float = 1e-8,
                 momentum: float = 0.95, adjust_lr_fn: str = "match_rms_adamw", fused: bool | None = None):
        self.muon = torch.optim.Muon(muon, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True,
                                     adjust_lr_fn=adjust_lr_fn)
        self.adamw = torch.optim.AdamW(adamw, lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
                                       fused=fused)
        params = [p for o in (self.muon, self.adamw) for g in o.param_groups for p in g["params"]]
        super().__init__(params, {"lr": lr})
        self._link()

    def _link(self) -> None:
        """The groups a scheduler reads and writes: the inner optimizers' own (renewed by ``load_state_dict``)."""
        self.param_groups = self.muon.param_groups + self.adamw.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adamw.step()
        return loss

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self._link()


def split_parameters(model) -> tuple[list, list, list]:
    """(hidden matrices for Muon, other matrices, vectors): the embeddings (and an untied output layer) are matrices
    that stay on AdamW."""
    hidden, other, vectors = [], [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "embed" not in name and "lm_head" not in name:
            hidden.append(p)
        elif p.ndim >= 2:
            other.append(p)
        else:
            vectors.append(p)
    return hidden, other, vectors
