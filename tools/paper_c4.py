"""Paper P, C4: patient similarity (``docs/p1_paper_protocol.md`` C4). L24 held-out anchors: the 10 nearest other
anchors by the twin model's state (cosine of the mean last-layer state, ``tools/server/paper_states.py``) against by
TF-IDF of the prompt's tokens; for ICU death, new IMV within 24 h and new RRT within 48 h, the share of a positive
anchor's neighbours that are positive (and of a negative's that are negative), against the prevalence (chance), and
the lift of the first over chance. Neighbours come from other stays only. Writes ``<out>/c4.json``.

    python tools/paper_c4.py --states runs/paper/states/test_L24 --store data/factory/stores/real_test \\
        --tokens data/tokens/full_r4 --out runs/paper/c4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import _sha  # noqa: E402

PROGRAMS = ("icu_death@-1", "imv_onset@1440", "rrt_onset@2880")


def neighbours(x: np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's k nearest other rows by cosine similarity."""
    z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    out = np.zeros((len(z), k), dtype=np.int64)
    for a in range(0, len(z), 1024):
        s = z[a:a + 1024] @ z.T
        for r in range(s.shape[0]):
            s[r, a + r] = -np.inf
        out[a:a + 1024] = np.argsort(-s, axis=1)[:, :k]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("states", "store", "tokens", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args(argv)
    import pyarrow.parquet as pq
    from sklearn.feature_extraction.text import TfidfTransformer

    from icutwin.eval import baselines as bl
    from icutwin.eval import bench
    from icutwin.eval.utility import _counts
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst
    from icutwin.tokenization.vocab import Vocab

    an = pq.read_table(Path(a.states) / "anchors.parquet").to_pandas()
    z = np.load(Path(a.states) / "states.npz")
    x_state = z["mean"].astype(np.float32)
    v = Vocab.load(Path(a.tokens) / "vocab.json")
    rows = {r["stay_id"]: r for r in bench.real_rows(a.tokens, "held_out", set(an["twin"].astype(str)))}
    keep = an["twin"].astype(str).isin(rows).to_numpy()
    an, x_state = an[keep].reset_index(drop=True), x_state[keep]
    prompts = [bench.prompt(rows[str(s)]["ids"], v, str(s), float(h)) for s, h in zip(an["twin"], an["hours"],
                                                                                       strict=True)]
    x_tfidf = TfidfTransformer(sublinear_tf=True).fit_transform(_counts(prompts, len(v.meta))).toarray()
    st = fst.Store(Path(a.store))
    idx = {str(s): i for i, s in enumerate(st.stays["stay_id"].tolist())}
    r = fl.Rows(np.array([idx[str(s)] for s in an["twin"]], dtype=np.int64), an["t_min"].to_numpy(dtype=np.int64))
    ev = fpr.Evaluator(st)
    cat = {f"{p.name}@{p.horizon}": p for p in fpr.catalog(st.channels)}
    nb = {"state": neighbours(x_state, a.k), "tfidf": neighbours(x_tfidf.astype(np.float32), a.k)}
    rep = {"git": _sha(), "anchors": int(len(an)), "k": a.k, "programs": {}}
    for name in PROGRAMS:
        y = bl.labels(st, cat[name], r, ev)
        ok = np.isfinite(y)
        prev = float(np.nanmean(y))
        res = {"anchors": int(ok.sum()), "prevalence": prev}
        for kind, n in nb.items():
            yn = y[n]
            pos = ok & (y == 1)
            neg = ok & (y == 0)
            p_pos = float(np.nanmean(yn[pos])) if pos.any() else None
            p_neg = float(np.nanmean(1 - yn[neg])) if neg.any() else None
            res[kind] = {"positive_neighbours_of_positives": p_pos, "negative_neighbours_of_negatives": p_neg,
                         "lift": p_pos / prev if p_pos is not None and prev > 0 else None}
        rep["programs"][name] = res
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "c4.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
