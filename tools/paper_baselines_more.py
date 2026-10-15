"""Paper P amendment 10 (``docs/p1_paper_protocol.md``): the learned baselines added to A1's, on one A1 design's
held-out anchors. They are trained as A1's LightGBM (``tools/paper_baselines.py``): A1's features
(``eval.baselines.Featurizer``) of real training stays' rows, the tuning store's rows for any choice.

* linear: logistic regression with an L1 (LASSO; liblinear) or elastic-net (l1 ratio 0.5; saga, at most 100 epochs,
  warm-started along the C path) penalty, C in {0.01, 0.1, 1};
* trees: a decision tree (CART; depth 3, 5 or 8; at least 50 rows per leaf), random forest and extra trees (300
  trees, square-root features per split, at least 5 rows per leaf);
* others: Gaussian naive Bayes, k-nearest neighbours (k 25 or 100, distance-weighted; one exact search per program
  for the larger k), a multilayer perceptron (hidden layers of 128 and 64 units, early stopping on 10% of the rows).

Every model gets median imputation with missing indicators and standardization, fitted on the training rows. Where a
setting is listed, the tuning rows pick it by AUROC. Endpoints: A1's primary ones; at L24 also death within 24 h and
the treatment starts (F2's L24x endpoints). ``--workers`` runs the fits (one per program, model and setting; one per
program for kNN and for the elastic net's C path) in forked processes that share the features.

Writes ``<out>/ml_predictions.parquet`` (anchor, dataset, program, model, p; host only) and ``<out>/ml_baselines.json``
(per program and model: the chosen setting, its tuning AUROC, seconds).

    python tools/paper_baselines_more.py --design-dir runs/paper/a1/test_R --train data/factory/stores/real_train \\
        --val data/factory/stores/real_dev --test data/factory/stores/real_test --out runs/paper/f10/ml_R
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import PRIMARY, _sha, applies, key  # noqa: E402

EXTRA = {"L24": ("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880")}
C_GRID = (0.01, 0.1, 1.0)


def candidates(name: str, n_jobs: int):
    """The model's settings (dict) and a constructor of each."""
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    forest = dict(n_estimators=300, max_features="sqrt", min_samples_leaf=5, n_jobs=n_jobs, random_state=0)
    return {
        "logistic_l1": [({"C": c}, lambda c=c: LogisticRegression(C=c, l1_ratio=1.0, solver="liblinear",
                                                                  random_state=0)) for c in C_GRID],
        "logistic_enet": [({"C": c}, lambda c=c: LogisticRegression(C=c, l1_ratio=0.5, solver="saga", max_iter=100,
                                                                    tol=1e-3, random_state=0)) for c in C_GRID],
        "cart": [({"max_depth": d}, lambda d=d: DecisionTreeClassifier(max_depth=d, min_samples_leaf=50,
                                                                       random_state=0)) for d in (3, 5, 8)],
        "random_forest": [({}, lambda: RandomForestClassifier(**forest))],
        "extra_trees": [({}, lambda: ExtraTreesClassifier(**forest))],
        "naive_bayes": [({}, GaussianNB)],
        "mlp": [({}, lambda: MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, validation_fraction=0.1,
                                           max_iter=200, random_state=0))],
    }[name]


MODELS = ("logistic_l1", "logistic_enet", "cart", "random_forest", "extra_trees", "naive_bayes", "knn", "mlp")


_SHARED: dict = {}                                    # the features and labels, set before the workers fork


def jobs_of(programs: list, models) -> list:
    """The fits: one per program, model and setting; one per program for kNN (one search) and the elastic net (its
    warm-started C path). The slow ones first."""
    jobs = []
    for prog in programs:
        for model in models:
            if model in ("knn", "logistic_enet"):
                jobs.append((prog, model, None))
            else:
                jobs += [(prog, model, i) for i in range(len(candidates(model, 1)))]
    order = {m: k for k, m in enumerate(("logistic_enet", "logistic_l1", "random_forest", "extra_trees", "knn", "mlp",
                                         "cart", "naive_bayes"))}
    return sorted(jobs, key=lambda j: order[j[1]])


def _fit(job):
    """One fit (``jobs_of``) on ``_SHARED``: (program, model, [(setting, tuning AUROC, test risks)], seconds)."""
    from icutwin.eval import bench

    prog, model, idx = job
    z = _SHARED
    y_tr, y_va = z["labels"][prog]
    ok_tr, ok_va = np.isfinite(y_tr), np.isfinite(y_va)
    xf, yf = z["z_tr"][ok_tr], y_tr[ok_tr].astype(int)
    xv, yv = z["z_va"][ok_va], y_va[ok_va].astype(int)
    t1 = time.time()
    out = []
    if model == "knn":
        s = knn_scores(xf, yf, {"val": xv, "test": z["z_te"]}, n_jobs=z["n_jobs"])
        out = [({"k": int(k)}, bench.auroc_or_none(yv, s["val"][k]), s["test"][k]) for k in s["val"]]
    else:
        warm = None
        for i, (setting, make) in enumerate(candidates(model, z["n_jobs"])):
            if idx is not None and i != idx:
                continue
            m = make()
            if model == "logistic_enet" and warm is not None:       # warm start along the C path
                m.set_params(warm_start=True)
                m.coef_, m.intercept_ = warm.coef_.copy(), warm.intercept_.copy()
            m.fit(xf, yf)
            warm = m
            out.append((setting, bench.auroc_or_none(yv, m.predict_proba(xv)[:, 1]), m.predict_proba(z["z_te"])[:, 1]))
    return prog, model, out, time.time() - t1


def knn_scores(x_fit, y_fit, queries: dict, ks=(25, 100), n_jobs: int = 8) -> dict:
    """Distance-weighted k-nearest-neighbour risks of each query set for every k (one exact search for the largest)."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=max(ks), algorithm="brute", n_jobs=n_jobs).fit(x_fit)
    out = {}
    for name, xq in queries.items():
        dist, idx = nn.kneighbors(xq)
        w = 1.0 / np.maximum(dist, 1e-9)
        lab = y_fit[idx]
        out[name] = {k: (w[:, :k] * lab[:, :k]).sum(1) / w[:, :k].sum(1) for k in ks}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("design-dir", "train", "val", "test", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--per-stay", type=int, default=4)
    ap.add_argument("--n-jobs", type=int, default=14)
    ap.add_argument("--workers", type=int, default=1, help="parallel fits (forked processes sharing the features)")
    ap.add_argument("--linear-workers", type=int, default=4, help="of --workers, those for the logistic fits")
    ap.add_argument("--programs", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    a = ap.parse_args(argv)
    import pandas as pd
    import pyarrow.parquet as pq
    from sklearn.impute import MissingIndicator, SimpleImputer
    from sklearn.pipeline import make_pipeline, make_union
    from sklearn.preprocessing import StandardScaler

    from icutwin.eval import baselines as bl
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst

    t0 = time.time()
    dd = Path(a.design_dir)
    design = json.loads((dd / "design.json").read_text())["design"]
    anchors = pq.read_table(dd / "anchors.parquet").to_pandas().drop_duplicates("anchor").reset_index(drop=True)
    tr, va, te = fst.Store(Path(a.train)), fst.Store(Path(a.val)), fst.Store(Path(a.test))
    qidx = {str(s): i for i, s in enumerate(te.stays["stay_id"].tolist())}
    anchors = anchors[anchors["twin"].astype(str).isin(qidx)].reset_index(drop=True)
    rows_te = fl.Rows(np.array([qidx[str(s)] for s in anchors["twin"]], dtype=np.int64),
                      anchors["t_min"].to_numpy(dtype=np.int64))
    fz = bl.Featurizer(bl.feature_channels(tr, a.channels)).fit_levels(tr)       # as tools/paper_baselines.py
    ev_tr, ev_va, ev_te = fpr.Evaluator(tr), fpr.Evaluator(va), fpr.Evaluator(te)
    rows_tr = bl.training_rows(tr, design, a.per_stay, seed=0)
    rows_va = bl.training_rows(va, design, 2, seed=1)
    x_tr, x_va, x_te = fz(tr, rows_tr, ev_tr), fz(va, rows_va, ev_va), fz(te, rows_te, ev_te)
    prep = make_pipeline(make_union(SimpleImputer(strategy="median", keep_empty_features=True),
                                    MissingIndicator(features="all")), StandardScaler()).fit(x_tr)
    z_tr, z_va, z_te = (prep.transform(x).astype(np.float32) for x in (x_tr, x_va, x_te))
    print(json.dumps({"design": design, "train_rows": len(rows_tr), "val_rows": len(rows_va),
                      "test_anchors": len(rows_te), "features": int(z_tr.shape[1]),
                      "setup_s": round(time.time() - t0, 1)}), flush=True)

    def lab(store, ev, prog, rows):
        if prog == "remaining_los_gt_7d":                 # as tools/paper_baselines.py
            rem = (store.los[rows.stay] - rows.t) / 60.0
            cens = np.isin(store.end_kind[rows.stay], ("censored", "unknown")) & (rem <= 168)
            return np.where(rem > 168, 1.0, np.where(cens, np.nan, 0.0))
        return bl.labels(store, prog, rows, ev)

    want = list(a.programs or (*PRIMARY.get(design, ()), *EXTRA.get(design, ())))
    progs = {key(p): p for p in fpr.catalog(tr.channels) if applies(p, design) and p.label in ("binary", "hosp_death")}
    progs["remaining_los_gt_7d"] = "remaining_los_gt_7d"
    preds, rep = [], {"design": design, "git": _sha(), "features": int(z_tr.shape[1]), "programs": {}}
    labels = {}
    for name in want:
        y_tr, y_va = lab(tr, ev_tr, progs[name], rows_tr), lab(va, ev_va, progs[name], rows_va)
        ok_tr, ok_va = np.isfinite(y_tr), np.isfinite(y_va)
        if len(np.unique(y_tr[ok_tr])) == 2 and len(np.unique(y_va[ok_va])) == 2:
            labels[name] = (y_tr, y_va)
            rep["programs"][name] = {"train_rows": int(ok_tr.sum()), "train_positives": int(np.nansum(y_tr)),
                                     "models": {}}
    _SHARED.update(z_tr=z_tr, z_va=z_va, z_te=z_te, labels=labels, n_jobs=a.n_jobs)
    out = Path(a.out)
    fits = out / "fits"                                  # one file per finished fit: a rerun resumes
    fits.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    def path_of(job):
        prog, model, idx = job
        return fits / f"{prog.replace('@', '_at_')}__{model}__{'all' if idx is None else idx}.pkl"

    def record(r, job, save=True):
        prog, model, out_, sec = r
        results.setdefault((prog, model), []).append((out_, sec))
        if save:
            tmp = path_of(job).with_suffix(".partial")
            tmp.write_bytes(pickle.dumps(r))
            tmp.replace(path_of(job))
        print(json.dumps({"program": prog, "model": model, "settings": [o[0] for o in out_],
                          "tuning_auroc": [o[1] for o in out_], "seconds": round(sec, 1), "saved": save}), flush=True)

    jobs = []
    for j in jobs_of(list(labels), a.models):
        if path_of(j).exists():
            record(pickle.loads(path_of(j).read_bytes()), j, save=False)
        else:
            jobs.append(j)
    if a.workers > 1:                                    # forked workers share the features (copy on write); the
        import multiprocessing as mp                     # linear fits (liblinear copies the rows) get a pool of their own
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context("fork")
        linear = [j for j in jobs if j[1].startswith("logistic")]
        other = [j for j in jobs if not j[1].startswith("logistic")]
        with ProcessPoolExecutor(max_workers=max(1, a.linear_workers), mp_context=ctx) as ex_l, \
                ProcessPoolExecutor(max_workers=max(1, a.workers - a.linear_workers), mp_context=ctx) as ex_o:
            futs = {ex_l.submit(_fit, j): j for j in linear} | {ex_o.submit(_fit, j): j for j in other}
            for f in as_completed(futs):
                record(f.result(), futs[f])
    else:
        for j in jobs:
            record(_fit(j), j)
    for (prog, model), parts in sorted(results.items()):
        cands = [o for out_, _s in parts for o in out_]
        setting, auc_val, p = max(cands, key=lambda o: o[1] if o[1] is not None else -1.0)
        for i in range(len(anchors)):
            preds.append({"anchor": anchors["anchor"][i], "dataset": anchors["dataset"][i], "program": prog,
                          "model": model, "p": float(p[i])})
        rep["programs"][prog]["models"][model] = {"setting": setting, "tuning_auroc": auc_val,
                                                  "seconds": round(sum(s for _o, s in parts), 1)}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_parquet(out / "ml_predictions.parquet", index=False)
    rep["seconds"] = round(time.time() - t0, 1)
    (out / "ml_baselines.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
