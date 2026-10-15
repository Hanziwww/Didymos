"""The ICU length-of-stay marginal of the twins (round 2, ``docs/m7_authenticity.md``): the model discharges too few
stays on days 1-2, so its twins' length tail is too long. ``resample`` corrects the marginal at the twin level: twins
are drawn unconditionally (oversampled), each falls in a length band (the goal block's bands, read with the
interval tokens' median gaps as everywhere else), and per band as many twins are kept as the dataset's training
stays put there; the model's stay given its length is untouched and no token distribution is truncated (the plan's
twin-level rejection sampling, applied to one marginal).

``bands`` measures the training stays' band shares (an aggregate; written once beside the vocabulary,
``los_bands.json``). Conditioning on a band through the goal block is not used: training always wrote all nine goal
fields, so a block with the band alone is unseen by the model.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

EDGES_H = (0, 24, 48, 96, 168, 336)
LABELS = ("0-24", "24-48", "48-96", "96-168", "168-336", "336+")


def band(hours: float | None) -> str:
    """The length band of an ICU stay (hours), or 'none' for a stay without an ICU discharge."""
    if hours is None or not np.isfinite(hours):
        return "none"
    return LABELS[int(np.searchsorted(EDGES_H, hours, side="right")) - 1]


def bands(tokens_root: Path, datasets: list, per_dataset: int = 20_000, seed: int = 0) -> dict:
    """Per dataset, the share of its training stays (a sample of at most ``per_dataset`` first windows) in each band."""
    from icutwin.eval import privacy as pv
    from icutwin.eval.forecast import Reader
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(Path(tokens_root) / "vocab.json")
    r = Reader(v)
    rng = np.random.default_rng(seed)
    out = {}
    for ds in datasets:
        w = pv.first_windows(Path(tokens_root), ds, "train")
        pick = rng.choice(len(w), size=min(per_dataset, len(w)), replace=False)
        c = Counter(band(r.outcomes(w[i])["icu_los_hours"]) for i in pick)
        out[ds] = {k: c[k] / len(pick) for k in (*LABELS, "none")}
    return out


def resample(seqs: list, hours: list, shares: dict, n: int, rng: np.random.Generator) -> tuple[list, dict]:
    """``n`` of ``seqs`` (twins with their length in ``hours``) so that their bands follow ``shares``; a band with too
    few twins gives all it has and the shortfall is filled at random from the rest. Returns (indices, report)."""
    lab = np.array([band(h) for h in hours])
    keys = [k for k in (*LABELS, "none") if shares.get(k, 0) > 0]
    p = np.array([shares[k] for k in keys])
    p = p / p.sum()
    want = np.floor(p * n).astype(int)
    want[np.argsort(-(p * n - want))[: n - want.sum()]] += 1
    chosen, short = [], {}
    for k, q in zip(keys, want):
        pool = np.flatnonzero(lab == k)
        take = min(q, len(pool))
        chosen += rng.choice(pool, size=take, replace=False).tolist()
        if take < q:
            short[k] = int(q - take)
    rest = np.setdiff1d(np.arange(len(seqs)), chosen)
    fill = sum(short.values())
    if fill:
        chosen += rng.choice(rest, size=min(fill, len(rest)), replace=False).tolist()
    return chosen, {"generated": len(seqs), "kept": len(chosen), "want": dict(zip(keys, want.tolist())),
                    "available": {k: int((lab == k).sum()) for k in (*LABELS, "none")}, "shortfall": short}


def save(tokens_root: Path, datasets: list, per_dataset: int = 20_000) -> Path:
    path = Path(tokens_root) / "los_bands.json"
    path.write_text(json.dumps(bands(tokens_root, datasets, per_dataset), indent=1), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    """``python -m icutwin.generate.los <tokens root> [datasets...]``: write ``los_bands.json``."""
    import sys

    args = argv if argv is not None else sys.argv[1:]
    datasets = args[1:] or ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
    print("wrote", save(Path(args[0]), datasets))


if __name__ == "__main__":
    main()
