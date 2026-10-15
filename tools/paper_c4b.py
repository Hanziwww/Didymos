"""Paper P amendment 7, F7 (``docs/p1_paper_protocol.md``): C4's neighbours as a predictor. For one design's held-out
anchors (``tools/server/paper_states.py``'s states, A1's stored labels): the leave-one-out kNN risk of every primary
endpoint (the inclusion-weighted mean label of the k nearest anchors of other stays, k = 5, 10, 25, 50) and its
AUROC, by representation: the model's state (mean state, cosine), TF-IDF of the prompt, A1's baseline features
(standardized, missing as 0 with indicators, cosine) and, as a negative control, the state of a randomly initialized
model of the same architecture (``--random-states``). Also with neighbours restricted to the anchor's own dataset.
Writes ``<out>/c4b_<design>.json`` (aggregates).

    python tools/paper_c4b.py --design L24 --states runs/paper/states/test_L24 \\
        --random-states runs/paper/states/test_L24_random --a1 runs/paper/a1/test_L24 --out runs/paper/c4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import PRIMARY, _sha  # noqa: E402

KS = (5, 10, 25, 50)


def sims(x: np.ndarray) -> np.ndarray:
    z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    return (z @ z.T).astype(np.float32)


def knn_risk(s: np.ndarray, y: np.ndarray, w: np.ndarray, block: np.ndarray, k: int) -> np.ndarray:
    """Per anchor, the weighted mean label of its k most similar anchors with a label, not blocked (same stay or
    another dataset)."""
    s = np.where(block | ~np.isfinite(y)[None, :], -np.inf, s)
    nb = np.argpartition(-s, kth=min(k, s.shape[1] - 1), axis=1)[:, :k]
    yy, ww = y[nb], w[nb]
    ok = np.isfinite(yy) & np.isfinite(np.take_along_axis(s, nb, axis=1))
    return np.where(ok.any(1), np.nansum(np.where(ok, yy * ww, 0), 1) / np.maximum(np.sum(np.where(ok, ww, 0), 1),
                                                                                    1e-12), np.nan)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("design", "states", "a1", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--random-states", default=None)
    ap.add_argument("--store", default="data/factory/stores/real_test")
    ap.add_argument("--train", default="data/factory/stores/real_train")
    ap.add_argument("--tokens", default="data/tokens/full_r4")
    a = ap.parse_args(argv)
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfTransformer

    from icutwin.eval import baselines as bl
    from icutwin.eval import bench
    from icutwin.eval.utility import _counts
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst
    from icutwin.tokenization.vocab import Vocab

    an = pd.read_parquet(Path(a.states) / "anchors.parquet")
    z = np.load(Path(a.states) / "states.npz")
    assert list(z["anchor"]) == an["anchor"].tolist()
    reps = {"state": z["mean"].astype(np.float32)}
    if a.random_states:
        zr = np.load(Path(a.random_states) / "states.npz")
        assert list(zr["anchor"]) == an["anchor"].tolist()
        reps["random_init_state"] = zr["mean"].astype(np.float32)
    v = Vocab.load(Path(a.tokens) / "vocab.json")
    rows = {r["stay_id"]: r for r in bench.real_rows(a.tokens, "held_out", set(an["twin"].astype(str)))}
    prompts = [bench.prompt(rows[str(s)]["ids"], v, str(s), float(h)) for s, h in zip(an["twin"], an["hours"],
                                                                                       strict=True)]
    reps["tfidf"] = TfidfTransformer(sublinear_tf=True).fit_transform(_counts(prompts, len(v.meta))).toarray() \
        .astype(np.float32)
    st, tr = fst.Store(Path(a.store)), fst.Store(Path(a.train))
    idx = {str(s): i for i, s in enumerate(st.stays["stay_id"].tolist())}
    r = fl.Rows(np.array([idx[str(s)] for s in an["twin"]], dtype=np.int64), an["t_min"].to_numpy(dtype=np.int64))
    fz = bl.Featurizer(bl.feature_channels(tr, 64)).fit_levels(tr)
    xf = fz(st, r, fpr.Evaluator(st)).astype(np.float64)
    miss = ~np.isfinite(xf)
    mu, sd = np.nanmean(xf, 0), np.nanstd(xf, 0)
    xf = np.where(miss, 0.0, (xf - mu) / np.where(sd > 0, sd, 1))
    reps["features"] = np.hstack([xf, miss[:, miss.any(0)].astype(float)]).astype(np.float32)
    pr = pd.read_parquet(Path(a.a1) / "predictions.parquet")
    w = an["weight"].to_numpy(dtype=float)
    ds = an["dataset"].to_numpy()
    stay = an["twin"].astype(str).to_numpy()
    same_stay = stay[:, None] == stay[None, :]
    other_ds = ds[:, None] != ds[None, :]
    sim = {k: sims(x) for k, x in reps.items()}
    rep = {"git": _sha(), "design": a.design, "anchors": int(len(an)), "representations": list(reps), "programs": {}}
    for prog in PRIMARY[a.design]:
        y = pr[pr["program"] == prog].set_index("anchor")["label"].reindex(an["anchor"]).to_numpy(dtype=float)
        if np.nansum(y) < 10:
            continue
        res = {"anchors": int(np.isfinite(y).sum()), "positives": int(np.nansum(y))}
        for kind, s in sim.items():
            for scope, block in (("all", same_stay), ("within_dataset", same_stay | other_ds)):
                for k in KS:
                    p = knn_risk(s, y, w, block, k)
                    ok = np.isfinite(y) & np.isfinite(p)
                    res.setdefault(kind, {}).setdefault(scope, {})[str(k)] = bench.auroc_or_none(y[ok], p[ok], w[ok])
        rep["programs"][prog] = res
        print(json.dumps({prog: {kind: res[kind]["all"]["10"] for kind in sim}}), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"c4b_{a.design}.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
