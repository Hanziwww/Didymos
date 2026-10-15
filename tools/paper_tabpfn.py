"""Paper P, an added baseline (``docs/p1_paper_protocol.md`` amendments 4-5): TabPFN, run locally (no data leaves the
host): v2 from Prior Labs' open weights, or TabPFN-3.5 from a local checkpoint (``--model-path``, ``--all-features``,
a larger ``--context``), on one A1 design's held-out anchors, for the primary endpoints.

The features are ``eval.baselines``' minus the 6-hour windows (TabPFN v2 takes at most 500 features): for each of
the 64 channels the last value, mean, minimum, maximum and count over 24 hours and the hours since its last value,
the treatment states, age, hours in the ICU, statics and dataset. Training rows: the real train store's rows of the
design, a stratified sample of at most 10,000 (TabPFN v2's context; positives kept up to half of it). Writes
``<out>/<name>_predictions.parquet`` (host only; ``tools/paper_compare.py`` reads it) and ``<out>/<name>.json``.

Needs ``PYTHONPATH`` with the tabpfn package and ``TABPFN_MODEL_VERSION=v2``.

    python tools/paper_tabpfn.py --design-dir runs/paper/a1/test_R --train data/factory/stores/real_train \\
        --test data/factory/stores/real_test --out runs/paper/a1/test_R
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import PRIMARY, _sha  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("design-dir", "train", "test", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--estimators", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-path", default=None, help="a local checkpoint (TabPFN-3.5: a licensed local copy)")
    ap.add_argument("--name", default="tabpfn_v2", help="the predictions' model name and file prefix")
    ap.add_argument("--all-features", action="store_true", help="keep the 6-hour windows (TabPFN-3.5)")
    ap.add_argument("--programs", nargs="+", default=None,
                    help="the programs to fit (default: the design's primary endpoints; amendment 16 adds others)")
    a = ap.parse_args(argv)
    import pandas as pd
    import pyarrow.parquet as pq
    from tabpfn import TabPFNClassifier

    from icutwin.eval import baselines as bl
    from icutwin.eval import bench
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst

    t0 = time.time()
    dd = Path(a.design_dir)
    design = json.loads((dd / "design.json").read_text())["design"]
    anchors = pq.read_table(dd / "anchors.parquet").to_pandas().drop_duplicates("anchor").reset_index(drop=True)
    tr, te = fst.Store(Path(a.train)), fst.Store(Path(a.test))
    qidx = {str(s): i for i, s in enumerate(te.stays["stay_id"].tolist())}
    anchors = anchors[anchors["twin"].astype(str).isin(qidx)].reset_index(drop=True)
    rows_te = fl.Rows(np.array([qidx[str(s)] for s in anchors["twin"]], dtype=np.int64),
                      anchors["t_min"].to_numpy(dtype=np.int64))
    w_te = anchors["weight"].to_numpy(dtype=np.float64)
    fz = bl.Featurizer(bl.feature_channels(tr, 64)).fit_levels(tr)
    names = fz.names()
    cols = list(range(len(names))) if a.all_features else [k for k, n in enumerate(names) if "|360" not in n]
    ev_tr, ev_te = fpr.Evaluator(tr), fpr.Evaluator(te)
    rows_tr = bl.training_rows(tr, design, 4, seed=0)
    x_tr, x_te = fz(tr, rows_tr, ev_tr)[:, cols], fz(te, rows_te, ev_te)[:, cols]
    cat = {f"{p.name}@{p.horizon}": p for p in fpr.catalog(tr.channels)}
    rng = np.random.default_rng(0)
    preds, rep = [], {"design": design, "git": _sha(), "features": len(cols), "context": a.context,
                      "estimators": a.estimators, "model": a.name, "model_path": a.model_path, "programs": {}}
    for name in (a.programs or PRIMARY.get(design, ())):
        if name == "remaining_los_gt_7d":
            def lab(store, r):
                rem = (store.los[r.stay] - r.t) / 60.0
                cens = np.isin(store.end_kind[r.stay], ("censored", "unknown")) & (rem <= 168)
                return np.where(rem > 168, 1.0, np.where(cens, np.nan, 0.0))
            y_tr, y_te = lab(tr, rows_tr), lab(te, rows_te)
        else:
            y_tr, y_te = bl.labels(tr, cat[name], rows_tr, ev_tr), bl.labels(te, cat[name], rows_te, ev_te)
        ok = np.flatnonzero(np.isfinite(y_tr))
        pos, neg = ok[y_tr[ok] == 1], ok[y_tr[ok] == 0]
        n_pos = min(len(pos), a.context // 2)
        pick = np.concatenate([rng.choice(pos, n_pos, replace=False),
                               rng.choice(neg, min(len(neg), a.context - n_pos), replace=False)])
        t1 = time.time()
        kw = {"model_path": a.model_path} if a.model_path else {}
        clf = TabPFNClassifier(device=a.device, n_estimators=a.estimators, ignore_pretraining_limits=True, **kw)
        clf.fit(x_tr[pick], y_tr[pick].astype(int))
        p = clf.predict_proba(x_te)[:, 1]
        # the sample's prevalence differs from the train rows' when positives were capped: correct the odds
        pi_s, pi = y_tr[pick].mean(), y_tr[ok].mean()
        odds = p / np.maximum(1 - p, 1e-9) * (pi / (1 - pi)) / (pi_s / (1 - pi_s))
        p = odds / (1 + odds)
        m = np.isfinite(y_te)
        rep["programs"][name] = {"train_rows": int(len(pick)), "train_positives": int(n_pos),
                                 "pooled": bench.binary_metrics(y_te[m], p[m], w_te[m]),
                                 "seconds": round(time.time() - t1, 1)}
        for i in np.flatnonzero(m):
            preds.append({"anchor": anchors["anchor"][i], "dataset": anchors["dataset"][i], "program": name,
                          "model": a.name, "p": float(p[i])})
        print(json.dumps({"program": name, "auroc": rep["programs"][name]["pooled"].get("auroc"),
                          "seconds": rep["programs"][name]["seconds"]}), flush=True)
    out = Path(a.out)
    pd.DataFrame(preds).to_parquet(out / f"{a.name}_predictions.parquet", index=False)
    rep["seconds"] = round(time.time() - t0, 1)
    (out / f"{a.name}.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
