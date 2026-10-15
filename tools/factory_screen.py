"""The general-task data screen (plan Phase F, "General tasks", S6; no training): episodes of one task prior where it
applies (threshold crossings, value targets and functional heads over a store's own physiological channels) from
three pools, the real tuning stays, the twins and the generic trajectories, each episode described by task
meta-features of its tabular snapshot (``Sampler.snapshot``: context rows to fit, query rows to score):
* size (log context rows), label type, prevalence and label entropy;
* difficulty: an ExtraTrees probe's AUROC (binary) or R^2 (values) from context to queries; a linear probe's, and
  their difference (nonlinearity);
* signal: the best single feature's |Spearman| with the label, and its share of the features' total
  (concentration of the feature-label association).
Then the coverage of the real pool (density and coverage, Naeem 2020, on standardized meta-features) by twin episodes,
by generic episodes and by both, and the real episodes neither covers, by label type and difficulty. The screen finds
regions of the real task space to fill with programs or views; it does not set the generic share (the training
sweep does). Aggregates only.

    python tools/factory_screen.py --real data/factory/stores/real_dev --twins data/factory/stores/bank_dev \\
        --generic data/factory/stores/generic_dev --n 600 --workers 12 --out runs/factory/screen_s6.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FEATURES = ("log_n", "binary", "prevalence", "entropy", "et", "linear", "nonlinearity", "signal", "concentration")
_S: dict = {}


def _sampler(store: str, vocab: str, split: str | None, seed: int):
    key = (store, split, seed)
    if _S.get("key") != key:
        from icutwin.factory import episodes, pool
        from icutwin.factory import store as fst
        from icutwin.tokenization.vocab import Vocab

        st = fst.Store(Path(store))
        ch = st.channels
        targets = pool.load_targets()
        progs = ([p for p in pool.f6_candidates(ch, tte=False)] + pool.f7_value_candidates(ch)
                 + pool.fh_candidates(ch, 600, seed, targets))
        progs = [p for p in progs if p.label in ("binary", "regression")]
        cfg = episodes.load_config()
        cfg = {**cfg, "families": {g: x for g, x in cfg["families"].items() if g != "clinical"}}
        _S.update(key=key, s=episodes.Sampler(st, Vocab.load(Path(vocab)), seed=seed, programs=progs, split=split,
                                              config=cfg, targets=None))
    return _S["s"]


def meta(xc, yc, xq, yq, binary: bool) -> dict | None:
    from scipy.stats import spearmanr
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score, roc_auc_score

    if len(yc) < 10 or len(yq) < 5:
        return None
    if binary and (len(np.unique(yc)) < 2 or len(np.unique(yq)) < 2):
        return None
    if not binary and (np.std(yc) == 0 or np.std(yq) == 0):
        return None
    out = {"log_n": float(np.log(len(yc))), "binary": float(binary)}
    if binary:
        p = float(yc.mean())
        out["prevalence"] = p
        out["entropy"] = float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))
        et = ExtraTreesClassifier(100, min_samples_leaf=2, random_state=0, n_jobs=1).fit(xc, yc)
        lr = LogisticRegression(max_iter=500).fit(xc, yc)
        out["et"] = float(roc_auc_score(yq, et.predict_proba(xq)[:, 1]))
        out["linear"] = float(roc_auc_score(yq, lr.predict_proba(xq)[:, 1]))
    else:
        out["prevalence"] = 0.0
        out["entropy"] = float(np.log(np.std(yc) + 1e-9) - np.log(np.mean(np.abs(yc)) + 1e-9))   # relative spread
        et = ExtraTreesRegressor(100, min_samples_leaf=2, random_state=0, n_jobs=1).fit(xc, yc)
        lr = Ridge().fit(xc, yc)
        out["et"] = float(np.clip(r2_score(yq, et.predict(xq)), -1, 1))
        out["linear"] = float(np.clip(r2_score(yq, lr.predict(xq)), -1, 1))
    out["nonlinearity"] = out["et"] - out["linear"]
    rho = []
    for j in range(xc.shape[1]):
        if np.std(xc[:, j]) > 0:
            r = spearmanr(xc[:, j], yc).statistic
            rho.append(abs(r) if np.isfinite(r) else 0.0)
    out["signal"] = float(max(rho)) if rho else 0.0
    out["concentration"] = float(max(rho) / (sum(rho) + 1e-9)) if rho else 0.0
    return out


def _job(args) -> list[dict]:
    store, vocab, split, seed, idx, source = args
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from factory_learnability import snapshot_xy

    from icutwin.factory.episodes import NoEpisode, _program

    s = _sampler(store, vocab, split, seed)
    out = []
    for i in idx:
        try:
            ep = s.episode(i)
        except NoEpisode:
            continue
        prog = _program(ep["recipe"]["program"])
        if not ep["recipe"]["context"]:
            continue
        binary = prog.label == "binary"
        xc, yc = snapshot_xy(s, ep, "context")
        xq, yq = snapshot_xy(s, ep, "query")
        m = meta(xc, yc, xq, yq, binary)
        if m is not None:
            out.append({"source": source, "family": prog.family, "label": prog.label, **m})
    return out


def coverage(real: np.ndarray, other: np.ndarray, k: int = 5) -> dict:
    from icutwin.eval.fidelity import prdc

    if len(other) <= k or len(real) <= k:
        return {}
    return {x: round(v, 4) for x, v in prdc(real, other, k=k).items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("real", "twins", "generic", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--vocab", default="data/tokens/full_r4/vocab.json")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    jobs = []
    for source, store, split in (("real", a.real, None), ("twins", a.twins, "train"), ("generic", a.generic, "train")):
        for k in range(0, a.n, 25):
            jobs.append((store, a.vocab, split, a.seed, list(range(k, min(k + 25, a.n))), source))
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for part in ex.map(_job, jobs) for r in part]
    by = {s: [r for r in rows if r["source"] == s] for s in ("real", "twins", "generic")}
    vec = {s: np.array([[r[f] for f in FEATURES] for r in rs], dtype=np.float64) for s, rs in by.items()}
    mu, sd = vec["real"].mean(0), vec["real"].std(0) + 1e-9
    z = {s: (x - mu) / sd for s, x in vec.items() if len(x)}
    both = np.vstack([z["twins"], z["generic"]]) if len(vec["twins"]) and len(vec["generic"]) else np.zeros((0, 9))
    rep = {"seconds": round(time.time() - t0, 1), "episodes": {s: len(r) for s, r in by.items()},
           "labels": {s: {lab: sum(r["label"] == lab for r in rs) for lab in ("binary", "regression")}
                      for s, rs in by.items()},
           "meta_quantiles": {s: {f: np.quantile(vec[s][:, j], [0.1, 0.5, 0.9]).round(3).tolist()
                                  for j, f in enumerate(FEATURES)} for s in by if len(vec[s])},
           "coverage_of_real": {"twins": coverage(z["real"], z["twins"]), "generic": coverage(z["real"], z["generic"]),
                                "both": coverage(z["real"], both)},
           "real_vs_real": {}}
    half = len(z["real"]) // 2                                  # the control: one half of the real pool covering the other
    if half > 10:
        rep["real_vs_real"] = coverage(z["real"][:half], z["real"][half:])
    from icutwin.eval.diversity import vendi

    rep["task_vendi"] = {}                                      # the effective number of distinct tasks
    for s_, x in z.items():
        if len(x) > 1:
            u = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
            rep["task_vendi"][s_] = {"episodes": len(x), "vendi": round(vendi(u), 2)}
    if len(both):                                               # the real episodes neither pool covers
        from sklearn.neighbors import NearestNeighbors

        r_real = NearestNeighbors(n_neighbors=6).fit(z["real"]).kneighbors(z["real"])[0][:, -1]
        d = NearestNeighbors(n_neighbors=1).fit(both).kneighbors(z["real"])[0][:, 0]
        gap = d > r_real
        rr = by["real"]
        rep["uncovered"] = {"share": round(float(gap.mean()), 4),
                            "by_label": {lab: round(float(gap[[r["label"] == lab for r in rr]].mean()), 4)
                                         for lab in ("binary", "regression")
                                         if any(r["label"] == lab for r in rr)},
                            "et_median_uncovered": float(np.median([rr[i]["et"] for i in np.flatnonzero(gap)]))
                            if gap.any() else None,
                            "et_median_covered": float(np.median([rr[i]["et"] for i in np.flatnonzero(~gap)]))
                            if (~gap).any() else None}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
