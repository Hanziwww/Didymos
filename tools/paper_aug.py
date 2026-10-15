"""Paper P amendment 7, F8 (``docs/p1_paper_protocol.md``): augmentation with repetitions. For one A1 design and
every dataset: LightGBM as A1's per-dataset baseline (A1's features and hyperparameters, early stopping on the
dataset's tuning rows), trained on n real train stays (n = 100, 300, 1,000) alone and with m x n of the dataset's
training twins (m = 1, 3), and on twins alone (300, 1,000, 3,000), 20 repetitions (seeds 7-26); tested on the
dataset's held-out A1 anchors (inclusion weights). Real stays alone at more sizes (100 to 4,000; 10 repetitions)
give the curve on which each augmented cell's "real equivalent" is read (interpolated). Read-outs per dataset,
endpoint and cell: the mean AUROC; the paired difference (augmented minus the same real draw alone) with the
percentile interval over repetitions and a paired bootstrap over test anchors of the repetition-averaged predictions.
Writes ``<out>/aug_<design>.json`` (aggregates).

    python tools/paper_aug.py --design-dir runs/paper/a1/test_R --train data/factory/stores/real_train \\
        --val data/factory/stores/real_dev --test data/factory/stores/real_test --twins data/factory/stores/bank_dev \\
        --out runs/paper/f8 --processes 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import _sha  # noqa: E402

ENDPOINTS = {"L24": ("icu_death@-1", "hospital_death@0", "remaining_los_gt_7d"),
             "L48": ("icu_death@-1", "hospital_death@0", "remaining_los_gt_7d"),
             "R": ("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880", "aki_2@2880")}
NS, MS, TWIN_ONLY = (100, 300, 1000), (1, 3), (300, 1000, 3000)
REAL_CURVE = (100, 200, 300, 600, 1000, 2000, 3000, 4000)
G: dict = {}                                       # the shared arrays (forked workers read them)


def fit_auc(tr_idx, tw_idx, task):
    from icutwin.eval import baselines as bl
    from icutwin.eval import bench

    x = np.vstack([G["x_tr"][tr_idx], G["x_tw"][tw_idx]]) if len(tw_idx) else G["x_tr"][tr_idx]
    y = np.r_[G["y_tr"][task][tr_idx], G["y_tw"][task][tw_idx]] if len(tw_idx) else G["y_tr"][task][tr_idx]
    if len(np.unique(y)) < 2:
        return None, None
    d = G["cur_ds"]
    mv = G["ok_va"][task] & (G["ds_va"] == d)
    m = bl.fit_lgbm(x, y, G["x_va"][mv], G["y_va"][task][mv], n_jobs=1)
    mt = G["ok_te"][task] & (G["ds_te"] == d)
    p = m.predict_proba(G["x_te"][mt])[:, 1]
    return bench.auroc_or_none(G["y_te"][task][mt], p, G["w_te"][mt]), p


def run_cell(args):
    task, d, kind, n, m, reps = args
    G["cur_ds"] = d
    real_st, tw_st = G["real_stays"][(task, d)], G["tw_stays"][(task, d)]
    out = []
    for rep in range(reps):
        rng = np.random.default_rng(7 + rep)
        if kind == "real" and n > len(real_st):
            return task, d, kind, n, m, None
        if kind in ("real", "aug"):
            if n > len(real_st):
                return task, d, kind, n, m, None
            rs = rng.choice(real_st, size=n, replace=False)
        else:
            rs = np.array([], dtype=np.int64)
        k_tw = m * n if kind == "aug" else (n if kind == "twins" else 0)
        if k_tw > len(tw_st):
            return task, d, kind, n, m, None
        ts = rng.choice(tw_st, size=k_tw, replace=False) if k_tw else np.array([], dtype=np.int64)
        tr_idx = np.flatnonzero(np.isin(G["st_tr"], rs) & G["ok_tr"][task] & (G["ds_tr"] == d))
        tw_idx = np.flatnonzero(np.isin(G["st_tw"], ts) & G["ok_tw"][task] & (G["ds_tw"] == d))
        if kind == "twins":
            tr_idx = np.array([], dtype=np.int64)
            x_auc = fit_twins_only(tw_idx, task)
            out.append(x_auc)
            continue
        out.append(fit_auc(tr_idx, tw_idx, task))
    return task, d, kind, n, m, out


def fit_twins_only(tw_idx, task):
    from icutwin.eval import baselines as bl
    from icutwin.eval import bench

    y = G["y_tw"][task][tw_idx]
    if len(np.unique(y)) < 2:
        return None, None
    d = G["cur_ds"]
    mv = G["ok_va"][task] & (G["ds_va"] == d)
    m = bl.fit_lgbm(G["x_tw"][tw_idx], y, G["x_va"][mv], G["y_va"][task][mv], n_jobs=1)
    mt = G["ok_te"][task] & (G["ds_te"] == d)
    p = m.predict_proba(G["x_te"][mt])[:, 1]
    return bench.auroc_or_none(G["y_te"][task][mt], p, G["w_te"][mt]), p


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("design-dir", "train", "val", "test", "twins", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--curve-reps", type=int, default=10)
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(argv)
    import multiprocessing as mp

    import pyarrow.parquet as pq

    from icutwin.eval import baselines as bl
    from icutwin.eval import bench
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst

    t0 = time.time()
    dd = Path(a.design_dir)
    design = json.loads((dd / "design.json").read_text())["design"]
    anchors = pq.read_table(dd / "anchors.parquet").to_pandas().drop_duplicates("anchor").reset_index(drop=True)
    tr, va, te, tw = (fst.Store(Path(x)) for x in (a.train, a.val, a.test, a.twins))
    tw_stays = np.flatnonzero((tw.stays["split"] == "train").to_numpy()) if "split" in tw.stays else None
    qidx = {str(s): i for i, s in enumerate(te.stays["stay_id"].tolist())}
    anchors = anchors[anchors["twin"].astype(str).isin(qidx)].reset_index(drop=True)
    rows_te = fl.Rows(np.array([qidx[str(s)] for s in anchors["twin"]], dtype=np.int64),
                      anchors["t_min"].to_numpy(dtype=np.int64))
    fz = bl.Featurizer(bl.feature_channels(tr, 64)).fit_levels(tr)
    ev = {k: fpr.Evaluator(s) for k, s in (("tr", tr), ("va", va), ("te", te), ("tw", tw))}
    rows = {"tr": bl.training_rows(tr, design, 4, seed=0), "va": bl.training_rows(va, design, 2, seed=1),
            "te": rows_te, "tw": bl.training_rows(tw, design, 4, seed=2, stays=tw_stays)}
    stores = {"tr": tr, "va": va, "te": te, "tw": tw}
    for k in rows:
        G[f"x_{k}"] = fz(stores[k], rows[k], ev[k])
        G[f"ds_{k}"] = stores[k].stays["dataset"].to_numpy()[rows[k].stay]
        G[f"st_{k}"] = rows[k].stay
        G[f"y_{k}"], G[f"ok_{k}"] = {}, {}
    G["w_te"] = anchors["weight"].to_numpy(dtype=np.float64)
    G["ds_te"] = anchors["dataset"].to_numpy()
    cat = {f"{p.name}@{p.horizon}": p for p in fpr.catalog(tr.channels)}

    def lab(store, e, prog, r):
        if prog == "remaining_los_gt_7d":
            rem = (store.los[r.stay] - r.t) / 60.0
            cens = np.isin(store.end_kind[r.stay], ("censored", "unknown")) & (rem <= 168)
            return np.where(rem > 168, 1.0, np.where(cens, np.nan, 0.0))
        return bl.labels(store, cat[prog], r, e)

    tasks = [t for t in ENDPOINTS[design] if t == "remaining_los_gt_7d" or t in cat]
    for task in tasks:
        for k in rows:
            y = lab(stores[k], ev[k], task, rows[k])
            G[f"y_{k}"][task], G[f"ok_{k}"][task] = y, np.isfinite(y)
    G["real_stays"], G["tw_stays"] = {}, {}
    dsets = sorted(set(G["ds_te"]))
    for task in tasks:
        for d in dsets:
            G["real_stays"][(task, d)] = np.unique(G["st_tr"][G["ok_tr"][task] & (G["ds_tr"] == d)])
            G["tw_stays"][(task, d)] = np.unique(G["st_tw"][G["ok_tw"][task] & (G["ds_tw"] == d)])
    print(json.dumps({"design": design, "features": int(G["x_tr"].shape[1]), "setup_s": round(time.time() - t0, 1)}),
          flush=True)
    jobs = []
    for task in tasks:
        for d in dsets:
            for n in NS:
                jobs.append((task, d, "real", n, 0, a.reps))
                for m in MS:
                    jobs.append((task, d, "aug", n, m, a.reps))
            for n in TWIN_ONLY:
                jobs.append((task, d, "twins", n, 0, a.reps))
            for n in REAL_CURVE:
                if n not in NS:
                    jobs.append((task, d, "real", n, 0, a.curve_reps))
    with mp.get_context("fork").Pool(a.processes) as pool:
        results = pool.map(run_cell, jobs, chunksize=1)
    cells: dict = {}
    for task, d, kind, n, m, out in results:
        cells[(task, d, kind, n, m)] = out
    rep = {"design": design, "git": _sha(), "reps": a.reps, "programs": {}}
    for task in tasks:
        rep["programs"][task] = {}
        for d in dsets:
            mt = G["ok_te"][task] & (G["ds_te"] == d)
            y, w = G["y_te"][task][mt], G["w_te"][mt]
            if y.sum() < 5:
                continue
            row = {"test_anchors": int(mt.sum()), "test_positives": int(y.sum()),
                   "real_stays": int(len(G["real_stays"][(task, d)])), "twin_stays": int(len(G["tw_stays"][(task, d)])),
                   "cells": {}}
            curve = {}
            for n in sorted(set(NS) | set(REAL_CURVE)):
                out = cells.get((task, d, "real", n, 0))
                if out:
                    au = [x[0] for x in out if x[0] is not None]
                    if au:
                        curve[n] = float(np.mean(au))
            row["real_curve"] = {str(k): v_ for k, v_ in curve.items()}
            for n in NS:
                base = cells.get((task, d, "real", n, 0))
                for m in MS:
                    aug = cells.get((task, d, "aug", n, m))
                    if not base or not aug:
                        continue
                    pairs = [(x[0], b[0]) for x, b in zip(aug, base, strict=True) if x[0] is not None and b[0] is not None]
                    if not pairs:
                        continue
                    diff = np.array([p_ - q_ for p_, q_ in pairs])
                    pa = np.mean([x[1] for x in aug if x[1] is not None], axis=0)
                    pb = np.mean([b[1] for b in base if b[1] is not None], axis=0)
                    ci_t = bench.bootstrap(lambda yy, aa, bb, ww: (bench.auroc_or_none(yy, aa, ww) or np.nan)
                                           - (bench.auroc_or_none(yy, bb, ww) or np.nan),
                                           np.arange(len(y)), y, pa, pb, w, n=a.boot)
                    mean_aug = float(np.mean([p_ for p_, _q in pairs]))
                    xs = sorted(curve)
                    equiv = None
                    if xs and mean_aug >= curve[xs[0]]:
                        ys_ = [curve[k] for k in xs]
                        if mean_aug > max(ys_):
                            equiv = f">{xs[-1]}"
                        else:
                            j = next(i for i in range(len(xs)) if ys_[i] >= mean_aug)
                            equiv = float(xs[j]) if j == 0 else float(np.interp(mean_aug, ys_[j - 1:j + 1], xs[j - 1:j + 1]))
                    row["cells"][f"n{n}_m{m}"] = {"auroc_aug": mean_aug, "auroc_real": float(np.mean([q_ for _p, q_ in pairs])),
                                                  "diff_mean": float(diff.mean()),
                                                  "diff_reps_ci": [float(np.percentile(diff, 2.5)),
                                                                   float(np.percentile(diff, 97.5))],
                                                  "diff_test_boot_ci": list(ci_t), "reps": len(pairs),
                                                  "real_equivalent_n": equiv}
            for n in TWIN_ONLY:
                out = cells.get((task, d, "twins", n, 0))
                if out:
                    au = [x[0] for x in out if x[0] is not None]
                    if au:
                        row["cells"][f"twins_{n}"] = {"auroc": float(np.mean(au)), "reps": len(au)}
            rep["programs"][task][d] = row
        print(json.dumps({task: {d: {k: x.get("diff_mean") for k, x in r["cells"].items() if k.startswith("n")}
                                 for d, r in rep["programs"][task].items()}}), flush=True)
    rep["seconds"] = round(time.time() - t0, 1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"aug_{design}.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
