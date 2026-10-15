"""Learnability with existing tools (plan Phase F, "Learnability"; no training): on tabular snapshots of trajectory
episodes (context rows: each context patient at its label times; query rows: each query at its prediction times;
features: the last value in the day before t of the program's own channels and core vitals and labs, and hours since
ICU admission), zero-shot TabICLv2 (public checkpoint) and an ExtraTrees probe, twins against real dev episodes of the
same programs; and the switchable-context test, real queries answered with a twin context against a real context of
the same program, site and size. AUROC for binary labels (ECE for TabICL), R^2 for values. Aggregates only.

    python tools/factory_learnability.py --twins data/factory/stores/bank_dev --real data/factory/stores/real_dev \\
        --pool data/factory/pools/bank_dev --n 300 --out runs/factory/learnability.json --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CKPT = Path(".cache/tabicl")


def snapshot_xy(s, ep: dict, part: str) -> tuple:
    from icutwin.factory import labels as fl
    from icutwin.factory.episodes import _program

    prog = _program(ep["recipe"]["program"])
    d = ep[part]
    ids = ep["recipe"]["context"] if part == "context" else [q["stay"] for q in ep["recipe"]["queries"]]
    st, t, y = [], [], []
    for k, sid in enumerate(ids):
        a, b = d["lab_off"][k], d["lab_off"][k + 1]
        ok = d["lab_valid"][a:b]
        st += [s.index[sid]] * int(ok.sum())
        t += d["lab_t"][a:b][ok].tolist()
        y += d["lab_y"][a:b][ok].tolist()
    if not st:
        return np.zeros((0, 1)), np.zeros(0)
    rows = fl.Rows(np.asarray(st, dtype=np.int64), np.asarray(t, dtype=np.int64))
    return s.snapshot(prog, rows), np.asarray(y, dtype=np.float64)


def ece(y, p, bins=10) -> float:
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def score(xc, yc, xq, yq, binary: bool, clf=None) -> dict:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.metrics import r2_score, roc_auc_score

    out = {}
    if binary:
        if len(np.unique(yc)) < 2 or len(np.unique(yq)) < 2:
            return out
        et = ExtraTreesClassifier(100, min_samples_leaf=2, random_state=0, n_jobs=4).fit(xc, yc)
        out["et_auroc"] = float(roc_auc_score(yq, et.predict_proba(xq)[:, 1]))
        if clf is not None and len(yc) >= 10:
            clf.fit(xc, yc.astype(int))
            p = clf.predict_proba(xq)[:, 1]
            out["tabicl_auroc"], out["tabicl_ece"] = float(roc_auc_score(yq, p)), ece(yq, p)
            out["prevalence"] = float(yq.mean())
    else:
        if len(yc) < 10 or np.std(yq) == 0:
            return out
        et = ExtraTreesRegressor(100, min_samples_leaf=2, random_state=0, n_jobs=4).fit(xc, yc)
        out["et_r2"] = float(r2_score(yq, et.predict(xq)))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("twins", "real", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--vocab", default="data/tokens/full_r4/vocab.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import pandas as pd
    from tabicl import TabICLClassifier

    from icutwin.factory import episodes, pool, programs as fpr
    from icutwin.factory import store as fst
    from icutwin.factory.episodes import NoEpisode, _program
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(Path(a.vocab))
    tw, rl = fst.Store(Path(a.twins)), fst.Store(Path(a.real))
    tw_stats = tw.channels.stats
    rl.channels.stats = tw_stats                       # one standardization: the training twins'
    progs, weights, support = pool.load(Path(a.pool)) if a.pool else (None, None, None)
    st = episodes.Sampler(tw, v, seed=a.seed, programs=progs, weights=weights, targets=pool.load_targets(),
                          support=support)
    sr = episodes.Sampler(rl, v, seed=a.seed, programs=fpr.catalog(rl.channels), split="dev")
    clf = TabICLClassifier(model_path=str(CKPT / "tabicl-classifier-v2-20260212.ckpt"), device=a.device,
                           n_estimators=4)
    rows, t0 = [], time.time()
    for i in range(a.n):
        try:
            ep = st.episode(i)
        except NoEpisode:
            continue
        prog = _program(ep["recipe"]["program"])
        if prog.label in ("forecast",) or not ep["recipe"]["context"]:
            continue
        binary = prog.label in ("binary", "hosp_death", "tte", "multihorizon")
        xc, yc = snapshot_xy(st, ep, "context")
        xq, yq = snapshot_xy(st, ep, "query")
        r = {"index": i, "family": prog.family, "label": prog.label, "name": prog.name, "source": "twins",
             **score(xc, yc, xq, yq, binary, clf if binary else None)}
        rows.append(r)
    # real dev episodes of the catalog programs, and the switchable-context test on them
    for i in range(a.n):
        try:
            ep = sr.episode(i)
        except NoEpisode:
            continue
        prog = _program(ep["recipe"]["program"])
        if not ep["recipe"]["context"] or prog.label == "forecast":
            continue
        binary = prog.label in ("binary", "hosp_death", "tte", "multihorizon")
        xc, yc = snapshot_xy(sr, ep, "context")
        xq, yq = snapshot_xy(sr, ep, "query")
        r = {"index": i, "family": prog.family, "label": prog.label, "name": prog.name, "source": "real",
             **score(xc, yc, xq, yq, binary, clf if binary else None)}
        # a twin context: the same program, cadence and site, as many patients, from the twin store
        site, cad = ep["recipe"]["site"], tuple(ep["recipe"]["cadence"])
        try:
            rows_t, b_t = st.site_labels(prog, cad, site)
        except Exception:      # noqa: BLE001 (a site the twins lack)
            rows.append(r)
            continue
        cand = np.unique(rows_t.stay[b_t.valid])
        if len(cand) >= len(ep["recipe"]["context"]):
            rng = np.random.default_rng([a.seed, i])
            ctx = rng.choice(cand, size=len(ep["recipe"]["context"]), replace=False)
            m = np.isin(rows_t.stay, ctx) & b_t.valid
            xt = st.snapshot(prog, rows_t.take(np.flatnonzero(m)))
            yt = b_t.y[m].astype(np.float64)
            sw = score(xt, yt, xq, yq, binary, clf if binary else None)
            r.update({f"twin_ctx_{k}": x for k, x in sw.items()})
        rows.append(r)
    df = pd.DataFrame(rows)
    rep = {"episodes": len(df), "seconds": round(time.time() - t0, 1)}
    for src in ("twins", "real"):
        sub = df[df["source"] == src]
        rep[src] = {c: float(sub[c].median()) for c in ("et_auroc", "tabicl_auroc", "tabicl_ece", "et_r2")
                    if c in sub and sub[c].notna().any()}
        rep[src]["by_family_et_auroc"] = sub.groupby("family")["et_auroc"].median().round(3).to_dict() \
            if "et_auroc" in sub else {}
    real = df[df["source"] == "real"]
    if "twin_ctx_tabicl_auroc" in real:
        both = real.dropna(subset=["tabicl_auroc", "twin_ctx_tabicl_auroc"])
        rep["switchable_context"] = {"pairs": len(both), "real_ctx_auroc": float(both["tabicl_auroc"].median()),
                                     "twin_ctx_auroc": float(both["twin_ctx_tabicl_auroc"].median()),
                                     "median_paired_gap": float((both["tabicl_auroc"]
                                                                 - both["twin_ctx_tabicl_auroc"]).median())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    df.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
