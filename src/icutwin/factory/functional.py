"""Functional heads (plan Phase F, "Label sampling" 2): synthetic labels with clinical attributes. A random function
(TabICLv2's families: linear, quadratic, a small MLP, a product, axis-aligned trees, random Fourier features as a GP
draw) of window features of a few physiological channels. The window lies in the future, (t + a, t + b], so the label
is a prediction and is as uncertain given the history as clinical outcomes are, the generator's future being
stochastic. A present window, (t - L, t], makes a nowcast, flagged and capped.

``features``: per row and channel, the channel's aggregate in the window, standardized with the store's statistics,
missing as 0 (the median) plus the count of observed channels. ``score``: the function, from its seed alone.
"""

from __future__ import annotations

import numpy as np

KINDS = ("linear", "quadratic", "mlp", "product", "tree", "rff")


def features(evaluator, rows, channels: tuple, aggs: tuple, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """(X [n, k] standardized with missing as 0, observed count [n])."""
    from icutwin.factory import labels as fl

    st = evaluator.store
    cols, seen = [], np.zeros(len(rows), dtype=np.int64)
    for ch, agg in zip(channels, aggs):
        y, cnt = fl.window_values(fl.finite(evaluator.events(ch)), rows, a, b, agg)
        s = st.channels.stats.get(ch) or {}
        med = s.get("median", np.nanmedian(y) if np.isfinite(y).any() else 0.0)
        spread = (s.get("p95", 1.0) - s.get("p05", 0.0)) / 3.29 if s else (np.nanstd(y) or 1.0)
        z = (y - med) / (spread if spread > 0 else 1.0)
        cols.append(np.where(cnt > 0, np.clip(z, -6, 6), 0.0))
        seen += cnt > 0
    return np.stack(cols, 1), seen


def score(kind: str, seed: int, x: np.ndarray) -> np.ndarray:
    """The random function of family ``kind`` drawn from ``seed``, applied to the rows of ``x``."""
    rng = np.random.default_rng(seed)
    n, k = x.shape
    if kind == "linear":
        return x @ rng.normal(size=k)
    if kind == "quadratic":
        w, q = rng.normal(size=k), rng.normal(size=(k, k)) / np.sqrt(k)
        return x @ w + np.einsum("ni,ij,nj->n", x, q, x)
    if kind == "mlp":
        h = 16
        w1, b1 = rng.normal(size=(k, h)) / np.sqrt(k), rng.normal(size=h)
        w2 = rng.normal(size=h) / np.sqrt(h)
        act = np.tanh if rng.random() < 0.5 else (lambda u: np.maximum(u, 0))
        return act(x @ w1 + b1) @ w2
    if kind == "product":
        i, j = rng.choice(k, size=2, replace=k < 2)
        return x[:, i] * x[:, j] + 0.3 * x @ rng.normal(size=k)
    if kind == "tree":
        out = np.zeros(n)
        for _ in range(4):                                     # a few depth-3 axis-aligned trees
            leaf = np.zeros(n, dtype=np.int64)
            for d in range(3):
                f, cut = rng.integers(k), rng.normal(scale=0.7)
                leaf = leaf * 2 + (x[:, f] > cut)
            out += rng.normal(size=8)[leaf]
        return out
    if kind == "rff":
        m = 32
        w, phase = rng.normal(size=(k, m)) / rng.uniform(0.5, 2.0), rng.uniform(0, 2 * np.pi, size=m)
        return np.cos(x @ w + phase) @ rng.normal(size=m) / np.sqrt(m)
    raise ValueError(kind)


def draw(rng: np.random.Generator, channels: list[str], prevalence: float, horizon: int, nowcast: bool = False,
         first: str | None = None) -> tuple:
    """A functional head's target: (``fn``, kind, seed, channels, aggregates, a, b, prevalence) over 2-5 channels
    (``first`` leading, then the others in name order: the first sets the head's domain); the window (t, t + horizon]
    (a nowcast: (t - horizon, t])."""
    k = int(rng.integers(2, 6))
    rest = sorted(rng.choice(channels, size=min(k - (first is not None), len(channels)), replace=False).tolist())
    chans = tuple(([first] if first is not None else []) + rest)
    aggs = tuple(str(rng.choice(["last", "min", "max", "mean"])) for _ in chans)
    a, b = (-horizon, 0) if nowcast else (0, horizon)
    return ("fn", str(rng.choice(KINDS)), int(rng.integers(1 << 31)), chans, aggs, a, b, float(prevalence))
