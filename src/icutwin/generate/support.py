"""The dataset support mask of constrained decoding (plan *Structured-output guarantee* 3, "Dataset/site support
mask"): a twin of a dataset draws only tokens that dataset's training data contains, which keeps each source's schema
(no ventilator settings in a dataset that charts none, no infusion rates where only presence is recorded). The
pilot found the model mixing other datasets' patterns into INSPIRE twins without it.

``compute`` counts each dataset's train windows once (``<tokens root>/<dataset>/train.parquet``) and ``save`` writes
``support.json`` beside the vocabulary (token ids per dataset: an aggregate of the vocabulary, no stay).
``Support`` gives the boolean mask of a dataset, and the dataset of an automaton (its prefix's dataset token; no mask
for the unconditional mode). Explicit conditions still win: a requested token is drawn even if unsupported.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compute(tokens_root: Path, datasets: list) -> dict:
    import pyarrow.parquet as pq

    out = {}
    for ds in datasets:
        col = pq.read_table(Path(tokens_root) / ds / "train.parquet", columns=["windows"]).column(0).combine_chunks()
        flat = col.values.values.to_numpy(zero_copy_only=False)
        out[ds] = np.unique(flat).astype(int).tolist()
    return out


def save(tokens_root: Path, datasets: list) -> Path:
    path = Path(tokens_root) / "support.json"
    path.write_text(json.dumps(compute(tokens_root, datasets)), encoding="utf-8")
    return path


class Support:
    def __init__(self, v, by_dataset: dict):
        n = len(v.tokens)
        # interval tokens follow the support as any token: a dataset's charting granularity is part of its schema
        # (INSPIRE records no gap under 5 minutes; exempting intervals let its twins draw them)
        always = np.array([m.get("cls") in ("special", "goal", "clock") for m in v.meta])   # clock: written
        self.masks = {}
        for ds, toks in by_dataset.items():
            mk = always.copy()
            mk[np.asarray(toks, dtype=np.int64)] = True
            self.masks[ds] = mk
        self.dataset_of = {i: m.get("value") for i, m in enumerate(v.meta)
                           if m.get("cls") == "prefix" and m.get("field") == "dataset"}
        self.n = n

    @classmethod
    def load(cls, v, path: Path):
        return cls(v, json.loads(Path(path).read_text(encoding="utf-8")))

    def mask(self, dataset_token: int | None) -> np.ndarray | None:
        """The mask of the dataset whose prefix token is ``dataset_token`` (None: not read yet, or the
        unconditional mode)."""
        if dataset_token is None:
            return None
        return self.masks.get(self.dataset_of.get(int(dataset_token)))


def main(argv: list[str] | None = None) -> None:
    """``python -m icutwin.generate.support <tokens root> [datasets...]``: write ``support.json``."""
    import sys

    args = argv if argv is not None else sys.argv[1:]
    root = Path(args[0])
    datasets = args[1:] or ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
    print("wrote", save(root, datasets))


if __name__ == "__main__":
    main()
