"""Paper P, the Twin probe (``docs/p1_paper_protocol.md`` A1; the differentiable target of C2): a logistic regression
per program on the frozen model's states at train-split anchors (``tools/server/paper_states.py``), scored at the
held-out anchors of the same design. Features: the last token's and the mean state, standardized; L2 strength chosen
by 5-fold cross-validated log loss among C = 0.01, 0.1, 1. Labels from the stores (``eval.baselines.labels``).
Writes ``<out>/probe_predictions.parquet`` (host only), ``<out>/probe.json`` (aggregates) and
``<out>/probe_models.npz`` (each program's scaler and coefficients, for C2's gradients).

    python tools/paper_probe.py --train-states runs/paper/states/train_L24 --train-store data/factory/stores/real_train \\
        --test-states runs/paper/states/test_L24 --test-store data/factory/stores/real_test --design L24 \\
        --out runs/paper/a1/test_L24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import _sha, applies, key  # noqa: E402


def load(d: Path):
    import pyarrow.parquet as pq

    z = np.load(d / "states.npz")
    an = pq.read_table(d / "anchors.parquet").to_pandas()
    x = np.hstack([z["last"].astype(np.float32), z["mean"].astype(np.float32)])
    assert list(z["anchor"]) == an["anchor"].tolist()
    return an, x


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("train-states", "train-store", "test-states", "test-store", "design", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--programs", nargs="*", default=None, help="amendment 7: only these programs")
    ap.add_argument("--permute-seed", type=int, default=None,
                    help="amendment 7 (F4's sanity check): the training labels permuted with this seed")
    a = ap.parse_args(argv)
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from icutwin.eval import baselines as bl
    from icutwin.eval import bench
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst

    tr_an, x_tr = load(Path(a.train_states))
    te_an, x_te = load(Path(a.test_states))
    tr, te = fst.Store(Path(a.train_store)), fst.Store(Path(a.test_store))

    def rows(store, an):
        idx = {str(s): i for i, s in enumerate(store.stays["stay_id"].tolist())}
        ok = an["twin"].astype(str).isin(idx).to_numpy()
        return ok, fl.Rows(np.array([idx[str(s)] for s in an["twin"][ok]], dtype=np.int64),
                           an["t_min"].to_numpy(dtype=np.int64)[ok])

    ok_tr, r_tr = rows(tr, tr_an)
    ok_te, r_te = rows(te, te_an)
    x_tr, x_te = x_tr[ok_tr], x_te[ok_te]
    te_an = te_an[ok_te].reset_index(drop=True)
    w_te = te_an["weight"].to_numpy(dtype=np.float64)
    ev_tr, ev_te = fpr.Evaluator(tr), fpr.Evaluator(te)
    preds, models, rep = [], {}, {"design": a.design, "git": _sha(), "train_anchors": int(len(r_tr)),
                                  "permute_seed": a.permute_seed,
                                  "test_anchors": int(len(r_te)), "programs": {}}

    def lab(store, ev, prog, r):
        if prog == "remaining_los_gt_7d":
            rem = (store.los[r.stay] - r.t) / 60.0
            cens = np.isin(store.end_kind[r.stay], ("censored", "unknown")) & (rem <= 168)
            return np.where(rem > 168, 1.0, np.where(cens, np.nan, 0.0))
        return bl.labels(store, prog, r, ev)

    progs = [(key(p), p) for p in fpr.catalog(tr.channels) if applies(p, a.design) and p.label in ("binary",
                                                                                                    "hosp_death")]
    if a.design in ("L24", "L48"):
        progs.append(("remaining_los_gt_7d", "remaining_los_gt_7d"))
    if a.programs:
        progs = [(n_, p_) for n_, p_ in progs if n_ in a.programs]
    for name, prog in progs:
        y_tr, y_te = lab(tr, ev_tr, prog, r_tr), lab(te, ev_te, prog, r_te)
        m_tr = np.isfinite(y_tr)
        if a.permute_seed is not None:              # labels shuffled among the training rows (Adebayo et al. 2018)
            y_tr[m_tr] = np.random.default_rng(a.permute_seed).permutation(y_tr[m_tr])
        if min(np.nansum(y_tr), (1 - y_tr[m_tr]).sum()) < 20 or np.nansum(y_te) < 5:
            continue
        cv = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                          {"logisticregression__C": [0.01, 0.1, 1.0]}, scoring="neg_log_loss",
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=4)
        cv.fit(x_tr[m_tr], y_tr[m_tr])
        p = cv.predict_proba(x_te)[:, 1]
        m = np.isfinite(y_te)
        rep["programs"][name] = {"C": cv.best_params_["logisticregression__C"],
                                 "pooled": bench.binary_metrics(y_te[m], p[m], w_te[m]),
                                 "datasets": {d: bench.binary_metrics(y_te[m & (te_an["dataset"] == d).to_numpy()],
                                                                      p[m & (te_an["dataset"] == d).to_numpy()],
                                                                      w_te[m & (te_an["dataset"] == d).to_numpy()])
                                              for d in sorted(set(te_an["dataset"]))}}
        sc, lr = cv.best_estimator_[0], cv.best_estimator_[1]
        w_eff = lr.coef_[0] / sc.scale_                       # logit = b + w . [last, mean] on raw states
        models[name] = np.concatenate([[lr.intercept_[0] - float(np.dot(w_eff, sc.mean_))], w_eff])
        for i in np.flatnonzero(m):
            preds.append({"anchor": te_an["anchor"][i], "dataset": te_an["dataset"][i], "program": name,
                          "model": "twin_probe", "p": float(p[i])})
        print(json.dumps({"program": name, "auroc": rep["programs"][name]["pooled"].get("auroc")}), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_parquet(out / "probe_predictions.parquet", index=False)
    np.savez_compressed(out / "probe_models.npz", **{k.replace("@", "_at_"): v for k, v in models.items()})
    (out / "probe.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
