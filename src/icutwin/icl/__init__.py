"""The bedside in-context model (plan Phase G, "TrajPFN"): factory episodes in, predictive distributions out.

* ``batch``: episodes (``factory.episodes`` "Task formats") as model inputs: tokens, time patches, rows, targets.
* ``model``: the token embedder, patch pooling, the causal temporal encoder, the task encoder, the in-context
  transformer and the heads.
* ``heads``: losses and scores of every label type (Bernoulli, categorical, bar distribution, competing-risk hazard).
* ``train`` / ``eval``: the learning confirmation's training loop and its pre-registered checks.
"""
