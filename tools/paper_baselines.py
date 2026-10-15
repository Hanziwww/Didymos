"""Paper P, the supervised baselines and the synthetic-data utility (``docs/p1_paper_protocol.md`` A1, A4) on one
A1 design's held-out anchors (``tools/server/paper_rollouts.py``'s ``anchors.parquet``).

* A1: LightGBM (pooled with the dataset as a feature, and per dataset for the primary endpoints), logistic regression
  (pooled) and, for death at the landmarks, the first-day SOFA, all trained on a real train store's rows of the same
  design (``eval.baselines``), early stopping on the tuning store, scored at the anchors' prediction times.
* label efficiency: LightGBM per dataset on n = 100, 300, 1,000, 3,000 train stays and all (primary endpoints).
* A4: LightGBM trained on twins (the bank store, pooled and per dataset) against the same on real train stays; the
  cross-dataset matrix (trained on one dataset, tested on each); the small-site augmentation (INSPIRE, NWICU: 300 real
  stays alone, plus 300 or 900 of the dataset's twins, and all real stays).

Writes ``<out>/baseline_predictions.parquet`` (host only) and ``<out>/baselines.json`` (aggregates).

    python tools/paper_baselines.py --design-dir runs/paper/a1/test_R --train data/factory/stores/real_train \\
        --val data/factory/stores/real_dev --test data/factory/stores/real_test \\
        --twins data/factory/stores/bank_dev --out runs/paper/a1/test_R
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import PRIMARY, _sha, applies, key  # noqa: E402

SMALL = ("inspire", "nwicu")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("design-dir", "train", "val", "test", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--twins", default=None)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--per-stay", type=int, default=4)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--skip-a4", action="store_true")
    a = ap.parse_args(argv)
    import pandas as pd
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
    tr, va, te = fst.Store(Path(a.train)), fst.Store(Path(a.val)), fst.Store(Path(a.test))
    tw = fst.Store(Path(a.twins)) if a.twins and not a.skip_a4 else None
    if tw is not None and "split" in tw.stays:                # the bank's training twins only
        tw_stays = np.flatnonzero((tw.stays["split"] == "train").to_numpy())
    else:
        tw_stays = None
    qidx = {str(s): i for i, s in enumerate(te.stays["stay_id"].tolist())}
    anchors = anchors[anchors["twin"].astype(str).isin(qidx)].reset_index(drop=True)
    rows_te = fl.Rows(np.array([qidx[str(s)] for s in anchors["twin"]], dtype=np.int64),
                      anchors["t_min"].to_numpy(dtype=np.int64))
    ds_te, w_te = anchors["dataset"].to_numpy(), anchors["weight"].to_numpy(dtype=np.float64)
    fz = bl.Featurizer(bl.feature_channels(tr, a.channels)).fit_levels(tr)
    ev_tr, ev_va, ev_te = fpr.Evaluator(tr), fpr.Evaluator(va), fpr.Evaluator(te)
    rows_tr = bl.training_rows(tr, design, a.per_stay, seed=0)
    rows_va = bl.training_rows(va, design, 2, seed=1)
    x_tr, x_va, x_te = fz(tr, rows_tr, ev_tr), fz(va, rows_va, ev_va), fz(te, rows_te, ev_te)
    ds_tr = tr.stays["dataset"].to_numpy()[rows_tr.stay]
    ds_va = va.stays["dataset"].to_numpy()[rows_va.stay]
    print(json.dumps({"train_rows": len(rows_tr), "val_rows": len(rows_va), "test_anchors": len(rows_te),
                      "features": x_tr.shape[1], "setup_s": round(time.time() - t0, 1)}), flush=True)
    if tw is not None:
        rows_tw = bl.training_rows(tw, design, a.per_stay, seed=2, stays=tw_stays)
        ev_tw = fpr.Evaluator(tw)
        x_tw = fz(tw, rows_tw, ev_tw)
        ds_tw = tw.stays["dataset"].to_numpy()[rows_tw.stay]
        st_tw = rows_tw.stay
    preds, rep = [], {"design": design, "git": _sha(), "features": fz.names()[:5] + ["..."],
                      "n_features": int(x_tr.shape[1]), "programs": {}}

    def lab(store, ev, prog, rows):
        if prog == "remaining_los_gt_7d":
            rem = (store.los[rows.stay] - rows.t) / 60.0
            cens = np.isin(store.end_kind[rows.stay], ("censored", "unknown")) & (rem <= 168)
            return np.where(rem > 168, 1.0, np.where(cens, np.nan, 0.0))
        return bl.labels(store, prog, rows, ev)

    def record(name, model, p, extra=None):
        for i in np.flatnonzero(np.isfinite(p)):
            preds.append({"anchor": anchors["anchor"][i], "dataset": ds_te[i], "program": name, "model": model,
                          "p": float(p[i]), **(extra or {})})

    def metrics(y, p, mask=None):
        m = np.isfinite(y) & np.isfinite(p) & (np.ones(len(y), bool) if mask is None else mask)
        return bench.binary_metrics(y[m], p[m], w_te[m]) if m.sum() else {"n": 0}

    def a4(name, prog):
        """TSTR against TRTR, cross-dataset, small-site augmentation (one primary endpoint)."""
        y_tw = lab(tw, ev_tw, prog, rows_tw)
        y_tr = lab(tr, ev_tr, prog, rows_tr)
        y_va = lab(va, ev_va, prog, rows_va)
        y_te = lab(te, ev_te, prog, rows_te)
        ok_tw, ok_tr, ok_va = np.isfinite(y_tw), np.isfinite(y_tr), np.isfinite(y_va)
        out = {"pooled": {}, "datasets": {}, "cross": {}, "augmentation": {}}

        def fit_score(x, y, mask_test=None):
            if len(np.unique(y)) < 2:
                return None
            mdl = bl.fit_lgbm(x, y, x_va[ok_va], y_va[ok_va])
            mt = np.isfinite(y_te) & (np.ones(len(y_te), bool) if mask_test is None else mask_test)
            return bench.auroc_or_none(y_te[mt], mdl.predict_proba(x_te[mt])[:, 1], w_te[mt])

        # pooled: as many twins as real rows
        n = min(int(ok_tw.sum()), int(ok_tr.sum()))
        rng = np.random.default_rng(0)
        itw = rng.choice(np.flatnonzero(ok_tw), size=n, replace=False)
        itr = rng.choice(np.flatnonzero(ok_tr), size=n, replace=False)
        out["pooled"] = {"rows": n, "tstr": fit_score(x_tw[itw], y_tw[itw]), "trtr": fit_score(x_tr[itr], y_tr[itr])}
        dsets = sorted(set(ds_te))
        for d in dsets:
            mtw, mtr = np.flatnonzero(ok_tw & (ds_tw == d)), np.flatnonzero(ok_tr & (ds_tr == d))
            k = min(len(mtw), len(mtr))
            if k < 50:
                continue
            itw, itr = rng.choice(mtw, size=k, replace=False), rng.choice(mtr, size=k, replace=False)
            out["datasets"][d] = {"rows": k, "tstr": fit_score(x_tw[itw], y_tw[itw], ds_te == d),
                                  "trtr": fit_score(x_tr[itr], y_tr[itr], ds_te == d)}
            out["cross"][d] = {}
            if len(np.unique(y_tw[itw])) == 2 and len(np.unique(y_tr[itr])) == 2:
                m_tw = bl.fit_lgbm(x_tw[itw], y_tw[itw], x_va[ok_va], y_va[ok_va])
                m_tr = bl.fit_lgbm(x_tr[itr], y_tr[itr], x_va[ok_va], y_va[ok_va])
                for d2 in dsets:
                    mt = np.isfinite(y_te) & (ds_te == d2)
                    out["cross"][d][d2] = {
                        "twins": bench.auroc_or_none(y_te[mt], m_tw.predict_proba(x_te[mt])[:, 1], w_te[mt]),
                        "real": bench.auroc_or_none(y_te[mt], m_tr.predict_proba(x_te[mt])[:, 1], w_te[mt])}
        for d in SMALL:
            real_st = np.unique(rows_tr.stay[ok_tr & (ds_tr == d)])
            tw_st = np.unique(st_tw[ok_tw & (ds_tw == d)])
            if len(real_st) < 300 or len(tw_st) < 300:
                continue
            r300 = np.isin(rows_tr.stay, rng.choice(real_st, 300, replace=False)) & ok_tr & (ds_tr == d)
            res = {"real_300": fit_score(x_tr[r300], y_tr[r300], ds_te == d),
                   "real_all": fit_score(x_tr[ok_tr & (ds_tr == d)], y_tr[ok_tr & (ds_tr == d)], ds_te == d)}
            for k in (300, 900):
                if len(tw_st) < k:
                    continue
                t_k = np.isin(st_tw, rng.choice(tw_st, k, replace=False)) & ok_tw & (ds_tw == d)
                res[f"real_300_plus_twins_{k}"] = fit_score(np.vstack([x_tr[r300], x_tw[t_k]]),
                                                            np.r_[y_tr[r300], y_tw[t_k]], ds_te == d)
            out["augmentation"][d] = res
        return out

    progs = [p for p in fpr.catalog(tr.channels) if applies(p, design) and p.label in ("binary", "hosp_death")]
    names = [(key(p), p) for p in progs] + ([("remaining_los_gt_7d", "remaining_los_gt_7d")]
                                            if design in ("L24", "L48") else [])
    primary = PRIMARY.get(design, ())
    for name, prog in names:
        t1 = time.time()
        y_tr, y_va, y_te = lab(tr, ev_tr, prog, rows_tr), lab(va, ev_va, prog, rows_va), lab(te, ev_te, prog, rows_te)
        ok_tr, ok_va = np.isfinite(y_tr), np.isfinite(y_va)
        if len(np.unique(y_tr[ok_tr])) < 2 or np.nansum(y_te) < 5:
            continue
        res = {"train_rows": int(ok_tr.sum()), "train_positives": int(np.nansum(y_tr)), "models": {}}
        m = bl.fit_lgbm(x_tr[ok_tr], y_tr[ok_tr], x_va[ok_va], y_va[ok_va])
        p = m.predict_proba(x_te)[:, 1]
        record(name, "lgbm_pooled", p)
        res["models"]["lgbm_pooled"] = {"pooled": metrics(y_te, p),
                                        "datasets": {d: metrics(y_te, p, ds_te == d) for d in sorted(set(ds_te))}}
        lr = bl.fit_logistic(x_tr[ok_tr], y_tr[ok_tr])
        p_lr = lr.predict_proba(x_te)[:, 1]
        record(name, "logistic_pooled", p_lr)
        res["models"]["logistic_pooled"] = {"pooled": metrics(y_te, p_lr)}
        if design in ("L24", "L48") and name in ("icu_death@-1", "hospital_death@0"):
            s = bl.sofa24(te, rows_te)
            res["models"]["sofa24"] = {"pooled": {"auroc": bench.auroc_or_none(y_te[np.isfinite(y_te) & np.isfinite(s)],
                                                                             s[np.isfinite(y_te) & np.isfinite(s)])}}
        if name in primary:
            per = np.full(len(y_te), np.nan)
            eff: dict = {}
            for d in sorted(set(ds_te)):
                mt, mv = ok_tr & (ds_tr == d), ok_va & (ds_va == d)
                if len(np.unique(y_tr[mt])) < 2:
                    continue
                md = bl.fit_lgbm(x_tr[mt], y_tr[mt], x_va[mv], y_va[mv])
                per[ds_te == d] = md.predict_proba(x_te[ds_te == d])[:, 1]
                # label efficiency: n train stays of this dataset
                stays_d = np.unique(rows_tr.stay[mt])
                rng = np.random.default_rng(0)
                order = rng.permutation(stays_d)
                eff[d] = {}
                for n in (100, 300, 1000, 3000, len(stays_d)):
                    if n > len(stays_d):
                        continue
                    sub = mt & np.isin(rows_tr.stay, order[:n])
                    if len(np.unique(y_tr[sub])) < 2:
                        continue
                    mn = bl.fit_lgbm(x_tr[sub], y_tr[sub], x_va[mv], y_va[mv])
                    pn = mn.predict_proba(x_te[ds_te == d])[:, 1]
                    yd, wd = y_te[ds_te == d], w_te[ds_te == d]
                    okd = np.isfinite(yd)
                    eff[d][str(n)] = bench.auroc_or_none(yd[okd], pn[okd], wd[okd])
            record(name, "lgbm_per_dataset", per)
            res["models"]["lgbm_per_dataset"] = {"pooled": metrics(y_te, per),
                                                 "datasets": {d: metrics(y_te, per, ds_te == d)
                                                              for d in sorted(set(ds_te))}}
            res["label_efficiency"] = eff
            if tw is not None:
                res["a4"] = a4(name, prog)
        res["seconds"] = round(time.time() - t1, 1)
        rep["programs"][name] = res
        print(json.dumps({"program": name, "lgbm_auroc": res["models"]["lgbm_pooled"]["pooled"].get("auroc"),
                          "seconds": res["seconds"]}), flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_parquet(out / "baseline_predictions.parquet", index=False)
    rep["seconds"] = round(time.time() - t0, 1)
    (out / "baselines.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
