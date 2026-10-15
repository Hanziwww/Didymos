"""Twin-level calibration (round 3, ``docs/m7_refinement.md``): the twins raked to the training stays' margins, the
ICU length band (round 2, ``generate.los``) and new-onset IMV, vasopressor and RRT after the 24-hour landmark
(``eval.utility``'s tasks: 'yes', 'no', or 'na' for a stay not in the ICU at 24 hours or treated before). It
generalizes ``los.resample`` from one margin to several.

* ``rake``: iterative proportional fitting (Deming and Stephan): the weights closest to uniform in Kullback-Leibler
  divergence whose weighted shares match every margin; a category the pool lacks cannot be matched and is reported.
* ``draw``: n twins without replacement with inclusion probabilities proportional to the weights (capped at 1;
  systematic sampling in a random order), so each category's expected count is n times its target share.

This corrects the population's margins, not the model: a twin is kept or not, never changed. ``margins`` measures
the training stays' shares (an aggregate, written once beside the vocabulary: ``calibration.json``).

Round 3, amendment 6: with ``events=True`` a further margin, the events per stay (the harness's count,
``eval.fidelity``) in bands cut at the training stays' 50th, 75th, 90th, 95th and 99th percentiles (``events_edges``
beside the shares), so the twins' length tails (A4's 90th and 99th percentiles) follow the training stays' too.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from icutwin.generate import los

TREATMENTS = ("imv", "vasopressor", "rrt")
MARGINS = ("los", *TREATMENTS)
EVENT_QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99)


def event_mask(v) -> np.ndarray:
    """The harness's event tokens (``eval.fidelity.vocabulary``: coded events; clock blocks are not events)."""
    return np.array([m.get("cls") == "event" and m.get("kind") is not None for m in v.meta])


def event_band(count: int, edges) -> str:
    return f"e{int(np.searchsorted(np.asarray(edges), count, side='right'))}"


def features(seqs: list, rd, hours: float = 24.0, edges=None) -> list[dict]:
    """Per stay, its category in each margin (with ``edges``, also its events-per-stay band)."""
    from icutwin.eval import utility as ut

    ev = event_mask(rd.v) if edges is not None else None
    out = []
    for x in seqs:
        ids = [int(t) for t in x]
        f = {"los": los.band(rd.outcomes(ids)["icu_los_hours"])}
        cut = ut.landmark(ids, rd, hours)
        before = set(ids[:cut]) if cut is not None else set()
        after = rd.outcomes(ids, after=cut) if cut is not None else {}
        for k in TREATMENTS:
            f[k] = "na" if cut is None or rd.sets[k] & before else ("yes" if after[k] else "no")
        if ev is not None:
            f["events"] = event_band(int(ev[np.asarray(ids, dtype=np.int64)].sum()), edges)
        out.append(f)
    return out


def margins(tokens_root: Path, datasets: list, per_dataset: int = 20_000, seed: int = 0, events: bool = False) -> dict:
    """Per dataset and margin, the share of its training stays (a sample of up to ``per_dataset`` first windows) in
    each category; with ``events``, the events-per-stay bands and their edges (``events_edges``) too."""
    from icutwin.eval import privacy as pv
    from icutwin.eval.forecast import Reader
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(Path(tokens_root) / "vocab.json")
    rd = Reader(v)
    rng = np.random.default_rng(seed)
    out = {}
    for ds in datasets:
        w = pv.first_windows(Path(tokens_root), ds, "train")
        pick = rng.choice(len(w), size=min(per_dataset, len(w)), replace=False)
        sample = [w[i] for i in pick]
        edges = None
        if events:
            ev = event_mask(v)
            counts = np.array([int(ev[np.asarray(s, dtype=np.int64)].sum()) for s in sample])
            edges = [float(x) for x in np.quantile(counts, EVENT_QUANTILES)]
        feats = features(sample, rd, edges=edges)
        names = (*MARGINS, "events") if events else MARGINS
        out[ds] = {m: {k: c / len(feats) for k, c in sorted(Counter(f[m] for f in feats).items())} for m in names}
        if events:
            out[ds]["events_edges"] = edges
    return out


def rake(feats: list[dict], targets: dict, iters: int = 500, tol: float = 1e-6, base=None,
         caps: tuple | None = None) -> tuple[np.ndarray, dict]:
    """Weights (mean 1) whose weighted category shares match ``targets`` ({margin: {category: share}}) over the
    categories the pool has (their targets renormalized); returns (weights, report). ``base``: the weights to stay
    closest to (Kullback-Leibler; default uniform); ``caps`` (lo, hi): bounded raking, every weight kept within
    [lo, hi] times its base (the margins may then be met only approximately: the report's error says how well)."""
    n = len(feats)
    b0 = np.ones(n) if base is None else np.asarray(base, dtype=np.float64) * n / np.sum(base)
    w = b0.copy()
    cats = {m: np.array([f[m] for f in feats]) for m in targets}
    unmatched, goal = {}, {}
    for m, t in targets.items():
        have = {k: s for k, s in t.items() if s > 0 and (cats[m] == k).any()}
        unmatched[m] = {k: s for k, s in t.items() if s > 0 and k not in have}
        total = sum(have.values())
        goal[m] = {k: s / total for k, s in have.items()}
    err = np.inf
    for _ in range(iters):
        for m, t in goal.items():                    # one margin at a time, all its categories at once
            total = w.sum()
            factor = {k: s / (w[cats[m] == k].sum() / total) for k, s in t.items() if w[cats[m] == k].sum() > 0}
            for k, f in factor.items():
                w[cats[m] == k] *= f
            w[~np.isin(cats[m], list(t))] = 0.0                  # categories without a target share
            if caps is not None:              # keep the scale (IPF is scale-free; clipping must not drift it)
                w = _bounded(w * n / w.sum(), b0, caps)
        err = max(abs(w[cats[m] == k].sum() / w.sum() - s) for m, t in goal.items() for k, s in t.items())
        if err < tol:
            break
    w = w * n / w.sum()
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return w, {"max_margin_error": float(err), "effective_size": ess,
               "unmatched": {m: u for m, u in unmatched.items() if u}}


def _bounded(w: np.ndarray, b0: np.ndarray, caps: tuple, rounds: int = 50) -> np.ndarray:
    """``w`` clipped to [lo, hi] times its base relative to the current total (repeated: clipping moves the total)."""
    for _ in range(rounds):
        on = w > 0
        if not on.any():
            return w
        scale = w.sum() / b0[on].sum()
        clipped = np.where(on, np.clip(w, caps[0] * b0 * scale, caps[1] * b0 * scale), 0.0)
        if np.allclose(clipped, w, rtol=1e-9, atol=0):
            return clipped
        w = clipped
    return w


def draw(weights: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` indices without replacement, each included with probability proportional to its weight (capped at 1,
    the rest rescaled): systematic sampling over a random order."""
    w = np.asarray(weights, dtype=np.float64)
    n = min(n, int((w > 0).sum()))
    pi = np.zeros_like(w)
    certain = np.zeros(len(w), dtype=bool)
    for _ in range(len(w)):
        rest = n - certain.sum()
        free = ~certain & (w > 0)
        pi[free] = rest * w[free] / w[free].sum()
        pi[certain] = 1.0
        over = free & (pi > 1)
        if not over.any():
            break
        certain |= over
    order = rng.permutation(len(w))
    cum = np.cumsum(pi[order])
    points = rng.uniform() + np.arange(n)
    at = np.minimum(np.searchsorted(cum, points, side="right"), len(order) - 1)   # round-off at the very end
    return np.unique(order[at])


def resample(seqs: list, rd, targets: dict, n: int, rng: np.random.Generator) -> tuple[list, dict]:
    """``n`` of ``seqs`` drawn so that their margins follow ``targets`` (one dataset's ``margins``; with
    ``events_edges``, the events-per-stay margin too). Returns (indices, report)."""
    edges = targets.get("events_edges")
    targets = {m: t for m, t in targets.items() if m != "events_edges"}
    feats = features(seqs, rd, edges=edges)
    w, rep = rake(feats, targets)
    keep = draw(w, n, rng)
    kept = [feats[i] for i in keep]
    achieved = {m: {k: c / len(kept) for k, c in sorted(Counter(f[m] for f in kept).items())} for m in targets}
    pool = {m: {k: c / len(feats) for k, c in sorted(Counter(f[m] for f in feats).items())} for m in targets}
    return keep.tolist(), {"generated": len(seqs), "kept": len(keep), **rep, "pool": pool, "achieved": achieved,
                           "target": targets}


def save(tokens_root: Path, datasets: list, per_dataset: int = 20_000, events: bool = False) -> Path:
    path = Path(tokens_root) / ("calibration_events.json" if events else "calibration.json")
    path.write_text(json.dumps(margins(tokens_root, datasets, per_dataset, events=events), indent=1), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    """``python -m icutwin.generate.calibrate <tokens root> [--events] [datasets...]``: write ``calibration.json``
    (with ``--events``: ``calibration_events.json``, the events-per-stay margin added)."""
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    events = "--events" in args
    args = [a for a in args if a != "--events"]
    datasets = args[1:] or ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
    print("wrote", save(Path(args[0]), datasets, events=events))


if __name__ == "__main__":
    main()
