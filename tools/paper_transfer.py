"""Paper P amendment 7, F9 (``docs/p1_paper_protocol.md``): transfer to new institutions, characterized (a model
without a dataset would need training; out of scope).

* (a) site prompts on R128 (``paper_a1.py``'s predictions of three design directories, 32 rollouts each): the site as
  known (F1's first 32), unknown, and swapped to another dataset: AUROC and O/E per primary endpoint, paired against
  the known site; and how far the swap moves the predicted risk toward the other dataset's practice: per anchor, the
  change in predicted risk against the two datasets' difference in real prevalence (A1's held-out R anchors,
  weighted), summarized by the slope and correlation across anchors;
* (b) familiarity: the Twin's (A1, 32 rollouts) and LightGBM's AUROC by the anchor's novelty (the cosine distance of
  its state to the nearest train-split anchor's state, quintiles), and in eICU by its hospital's number of train stays
  (tertiles) and whether the hospital has its own site token;
* (c) supervised transfer: LightGBM (A1's features) and the Twin probe (logistic on the frozen states) trained on six
  datasets' train anchors and tested on the seventh's held-out anchors, against the same trained on all seven
  (pooled) and within the dataset, beside the zero-shot Twin.

Writes ``<out>/transfer_<design>.json`` (aggregates).

    python tools/paper_transfer.py --design R --a1 runs/paper/a1/test_R --states runs/paper/states \\
        --known runs/paper/f9/R128_first32 --unknown runs/paper/f9/R_unknown --swap runs/paper/f9/R_swap \\
        --out runs/paper/f9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import PRIMARY, _sha  # noqa: E402


def load_states(d: Path):
    import pandas as pd

    z = np.load(d / "states.npz")
    an = pd.read_parquet(d / "anchors.parquet")
    assert list(z["anchor"]) == an["anchor"].tolist()
    return an, np.hstack([z["last"].astype(np.float32), z["mean"].astype(np.float32)])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("design", "a1", "states", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--known", default=None)
    ap.add_argument("--unknown", default=None)
    ap.add_argument("--swap", default=None)
    ap.add_argument("--tokens", default="data/tokens/full_r4")
    ap.add_argument("--train", default="data/factory/stores/real_train")
    ap.add_argument("--val", default="data/factory/stores/real_dev")
    ap.add_argument("--test", default="data/factory/stores/real_test")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(argv)
    import pandas as pd

    from icutwin.eval import baselines as bl
    from icutwin.eval import bench
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst

    a1 = Path(a.a1)
    rep = {"git": _sha(), "design": a.design}
    prim = PRIMARY[a.design]
    an = pd.read_parquet(a1 / "anchors.parquet").drop_duplicates("anchor").reset_index(drop=True)
    pr = pd.read_parquet(a1 / "predictions.parquet")
    lab = {p: g.set_index("anchor")["label"] for p, g in pr.groupby("program")}
    tw32 = {p: g.set_index("anchor")["p"] for p, g in pr.groupby("program")}
    base = pd.read_parquet(a1 / "baseline_predictions.parquet")
    lgb = {p: g[g["model"] == "lgbm_pooled"].drop_duplicates("anchor").set_index("anchor")["p"]
           for p, g in base.groupby("program")}
    w_of = an.set_index("anchor")["weight"]
    ds_of = an.set_index("anchor")["dataset"]

    def auc(x, prog, preds):
        y = lab[prog].reindex(x).to_numpy(float)
        p = preds.reindex(x).to_numpy(float)
        w = w_of.reindex(x).to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(p)
        return bench.auroc_or_none(y[ok], p[ok], w[ok]), int(ok.sum()), int(np.nansum(y[ok]))

    # (a) site prompts
    if a.known and a.unknown and a.swap:
        runs = {k: pd.read_parquet(Path(d) / "predictions.parquet") for k, d in
                (("known", a.known), ("unknown", a.unknown), ("swap", a.swap))}
        sw_an = pd.read_parquet(Path(a.swap) / "anchors.parquet").drop_duplicates("anchor").set_index("anchor")
        prev = {p: {d: float(np.average(g["label"], weights=g["weight"])) for d, g in
                    pr[pr["program"] == p].dropna(subset=["label"]).groupby("dataset")} for p in prim}
        res = {}
        for prog in prim:
            pp = {k: r[r["program"] == prog].set_index("anchor")["p"] for k, r in runs.items()}
            common = sorted(set.intersection(*(set(x.index) for x in pp.values())) & set(lab[prog].dropna().index))
            if len(common) < 30:
                continue
            y = lab[prog].reindex(common).to_numpy(float)
            w = w_of.reindex(common).to_numpy(float)
            stays = an.set_index("anchor").loc[common, "twin"].astype(str).to_numpy()
            dsx = ds_of.reindex(common).to_numpy()
            row = {"anchors": len(common), "positives": int(y.sum())}
            for k, p_ in pp.items():
                row[k] = bench.binary_metrics(y, p_.reindex(common).to_numpy(float), w)
            for k in ("unknown", "swap"):
                p1, p0 = pp[k].reindex(common).to_numpy(float), pp["known"].reindex(common).to_numpy(float)
                row[f"{k}_minus_known"] = {
                    "auroc_diff": (bench.auroc_or_none(y, p1, w) or np.nan) - (bench.auroc_or_none(y, p0, w) or np.nan),
                    "ci": list(bench.bootstrap(lambda yy, aa, bb, ww: (bench.auroc_or_none(yy, aa, ww) or np.nan)
                                               - (bench.auroc_or_none(yy, bb, ww) or np.nan),
                                               stays, y, p1, p0, w, n=a.boot, strata=dsx))}
            # the swap's shift toward the other dataset's practice
            to = sw_an.reindex(common)["prompt_dataset"].to_numpy()
            dp = pp["swap"].reindex(common).to_numpy(float) - pp["known"].reindex(common).to_numpy(float)
            dr = np.array([prev[prog].get(t_, np.nan) - prev[prog].get(f_, np.nan) for t_, f_ in zip(to, dsx, strict=True)])
            ok = np.isfinite(dp) & np.isfinite(dr)
            if ok.sum() > 10 and np.std(dr[ok]) > 0:
                slope = float(np.polyfit(dr[ok], dp[ok], 1)[0])
                row["swap_shift"] = {"slope": slope, "corr": float(np.corrcoef(dr[ok], dp[ok])[0, 1]),
                                     "anchors": int(ok.sum())}
            res[prog] = row
        rep["site_prompts"] = res

    # (b) familiarity: novelty quintiles and eICU hospitals
    te_an, x_te = load_states(Path(a.states) / f"test_{a.design}")
    tr_an, x_tr = load_states(Path(a.states) / f"train_{a.design}")
    zt = x_te / np.maximum(np.linalg.norm(x_te, axis=1, keepdims=True), 1e-9)
    zr = x_tr / np.maximum(np.linalg.norm(x_tr, axis=1, keepdims=True), 1e-9)
    nov = np.empty(len(zt))
    for i in range(0, len(zt), 512):
        nov[i:i + 512] = 1 - (zt[i:i + 512] @ zr.T).max(1)
    nov_of = pd.Series(nov, index=te_an["anchor"])
    fam = {}
    q = np.quantile(nov, [0.2, 0.4, 0.6, 0.8])
    for prog in prim:
        if prog not in lab:
            continue
        rowp = {}
        for b in range(5):
            lo, hi = (q[b - 1] if b else -np.inf), (q[b] if b < 4 else np.inf)
            x = nov_of.index[(nov_of > lo) & (nov_of <= hi)]
            t_, n_, k_ = auc(x, prog, tw32[prog])
            l_, _n, _k = auc(x, prog, lgb[prog]) if prog in lgb else (None, 0, 0)
            rowp[f"q{b + 1}"] = {"twin": t_, "lgbm": l_, "anchors": n_, "positives": k_}
        fam[prog] = rowp
    rep["novelty"] = {"quintile_edges": q.tolist(), "programs": fam}
    from collections import Counter

    import pyarrow.parquet as pq

    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(Path(a.tokens) / "vocab.json")
    site_tok = {}
    for split in ("train", "held_out"):
        t = pq.read_table(Path(a.tokens) / "eicu" / f"{split}.parquet", columns=["stay_id", "ids"]).to_pylist()
        for r in t:
            for tok in r["ids"][:16]:
                m = v.meta[int(tok)]
                if m.get("cls") == "prefix" and m.get("field") == "site":
                    site_tok[(split, str(r["stay_id"]))] = str(m.get("value"))
    st_te = fst.Store(Path(a.test)).stays.set_index("stay_id")
    hosp_te = st_te["site"].astype(str)
    st_tr = fst.Store(Path(a.train)).stays
    vol = Counter(st_tr.loc[st_tr["dataset"] == "eicu", "site"].astype(str))
    e_an = an[an["dataset"] == "eicu"]
    h = e_an["twin"].astype(str).map(lambda s: hosp_te.get(s, ""))
    has_tok = e_an["twin"].astype(str).map(lambda s: site_tok.get(("held_out", s), "unknown") not in ("unknown", "", "None"))
    vols = h.map(lambda s: vol.get(s, 0)).to_numpy()
    tq = np.quantile(vols, [1 / 3, 2 / 3])
    eicu = {}
    for prog in prim:
        if prog not in lab:
            continue
        rowp = {}
        for name, mask in (("volume_low", vols <= tq[0]), ("volume_mid", (vols > tq[0]) & (vols <= tq[1])),
                           ("volume_high", vols > tq[1]), ("site_token", has_tok.to_numpy()),
                           ("no_site_token", ~has_tok.to_numpy())):
            x = e_an["anchor"][mask]
            t_, n_, k_ = auc(x, prog, tw32[prog])
            l_, _n, _k = auc(x, prog, lgb[prog]) if prog in lgb else (None, 0, 0)
            rowp[name] = {"twin": t_, "lgbm": l_, "anchors": n_, "positives": k_}
        eicu[prog] = rowp
    rep["eicu_familiarity"] = {"volume_tertile_edges": tq.tolist(), "programs": eicu}

    # (c) supervised transfer: leave one dataset out
    tr, va, te = fst.Store(Path(a.train)), fst.Store(Path(a.val)), fst.Store(Path(a.test))
    fz = bl.Featurizer(bl.feature_channels(tr, 64)).fit_levels(tr)
    qidx = {str(s): i for i, s in enumerate(te.stays["stay_id"].tolist())}
    an_t = an[an["twin"].astype(str).isin(qidx)].reset_index(drop=True)
    rows_te = fl.Rows(np.array([qidx[str(s)] for s in an_t["twin"]], dtype=np.int64), an_t["t_min"].to_numpy(np.int64))
    rows_tr, rows_va = bl.training_rows(tr, a.design, 4, seed=0), bl.training_rows(va, a.design, 2, seed=1)
    ev_tr, ev_va, ev_te = fpr.Evaluator(tr), fpr.Evaluator(va), fpr.Evaluator(te)
    x_tr2, x_va2, x_te2 = fz(tr, rows_tr, ev_tr), fz(va, rows_va, ev_va), fz(te, rows_te, ev_te)
    ds_tr = tr.stays["dataset"].to_numpy()[rows_tr.stay]
    ds_va = va.stays["dataset"].to_numpy()[rows_va.stay]
    ds_te = an_t["dataset"].to_numpy()
    w_te = an_t["weight"].to_numpy(float)
    cat = {f"{p.name}@{p.horizon}": p for p in fpr.catalog(tr.channels)}
    # the probe's states: train-split states of the design, labelled on the train store
    tidx = {str(s): i for i, s in enumerate(tr.stays["stay_id"].tolist())}
    okp = tr_an["twin"].astype(str).isin(tidx).to_numpy()
    rows_ptr = fl.Rows(np.array([tidx[str(s)] for s in tr_an["twin"][okp]], dtype=np.int64),
                       tr_an["t_min"].to_numpy(np.int64)[okp])
    xp_tr, ds_ptr = x_tr[okp], tr_an["dataset"].to_numpy()[okp]
    pos_te = {x: i for i, x in enumerate(te_an["anchor"])}
    xp_te = x_te[[pos_te[x] for x in an_t["anchor"]]]
    lodo = {}
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    for prog in prim:
        p_ = cat.get(prog)
        if p_ is None:
            continue
        y_tr = bl.labels(tr, p_, rows_tr, ev_tr)
        y_va = bl.labels(va, p_, rows_va, ev_va)
        y_te = bl.labels(te, p_, rows_te, ev_te)
        y_ptr = bl.labels(tr, p_, rows_ptr, ev_tr)
        rowp = {}
        for d in sorted(set(ds_te)):
            mt = np.isfinite(y_te) & (ds_te == d)
            if np.nansum(y_te[mt]) < 5:
                continue
            res_d = {}
            m_tr = np.isfinite(y_tr) & (ds_tr != d)    # without the dataset (A1's pooled and per-dataset models are stored)
            if len(np.unique(y_tr[m_tr])) >= 2:
                m_va = np.isfinite(y_va) & (ds_va != d)
                mdl = bl.fit_lgbm(x_tr2[m_tr], y_tr[m_tr], x_va2[m_va], y_va[m_va])
                res_d["lgbm_lodo"] = bench.auroc_or_none(y_te[mt], mdl.predict_proba(x_te2[mt])[:, 1], w_te[mt])
            for name, model in (("pooled", "lgbm_pooled"), ("within", "lgbm_per_dataset")):
                g = base[(base["program"] == prog) & (base["model"] == model)].drop_duplicates("anchor")                     .set_index("anchor")["p"]
                p_ = g.reindex(an_t["anchor"][mt]).to_numpy(float)
                okp_ = np.isfinite(p_)
                res_d[f"lgbm_{name}"] = bench.auroc_or_none(y_te[mt][okp_], p_[okp_], w_te[mt][okp_])
            for name, m_tr in (("lodo", np.isfinite(y_ptr) & (ds_ptr != d)), ("pooled", np.isfinite(y_ptr))):
                if len(np.unique(y_ptr[m_tr])) < 2:
                    continue
                pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
                pipe.fit(xp_tr[m_tr], y_ptr[m_tr])
                res_d[f"probe_{name}"] = bench.auroc_or_none(y_te[mt], pipe.predict_proba(xp_te[mt])[:, 1], w_te[mt])
            t_, _n, k_ = auc(an_t["anchor"][mt], prog, tw32[prog])
            res_d["twin_zero_shot"] = t_
            res_d["positives"] = int(np.nansum(y_te[mt]))
            rowp[d] = res_d
        lodo[prog] = rowp
        print(json.dumps({prog: {d: (r.get("lgbm_lodo"), r.get("lgbm_within"), r.get("twin_zero_shot"))
                                 for d, r in rowp.items()}}), flush=True)
    rep["leave_one_dataset_out"] = lodo
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"transfer_{a.design}.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
