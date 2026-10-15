"""Task realism (plan *Prediction-task generation* 6, S8 *Tasks*): the same tasks on a real cache (held-out stays) and
on a candidate's cache (its decoded twins, ``factory.stream``; the same grid concepts), compared per task: label
prevalence (classification: the class shares' total variation; regression: the mean in SDs of the real labels),
difficulty (an ExtraTrees model's score on a context / query split, AUROC or R^2), the rank correlation of the
features' mutual information with the label, and transfer (the candidate-trained model scored on the real query:
TSTR against the real-trained TRTR). Tasks are drawn by the factory's prior on the real cache, realism transforms
off; a task too small on either side is skipped.
"""

from __future__ import annotations

import numpy as np

from icutwin.factory import tasks as ft


def _score(family: str, xa, ya, xb, yb, seed: int) -> float | None:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.metrics import r2_score, roc_auc_score

    if family == "regression":
        m = ExtraTreesRegressor(n_estimators=50, min_samples_leaf=3, random_state=seed, n_jobs=1).fit(xa, ya)
        return float(r2_score(yb, m.predict(xb)))
    if len(np.unique(ya)) < 2 or len(np.unique(yb)) < 2:
        return None
    m = ExtraTreesClassifier(n_estimators=50, min_samples_leaf=3, random_state=seed, n_jobs=1).fit(xa, ya)
    p = m.predict_proba(xb)
    if p.shape[1] == 2:
        return float(roc_auc_score(yb, p[:, 1]))
    try:
        return float(roc_auc_score(yb, p, multi_class="ovr", labels=m.classes_))
    except ValueError:
        return None


def _mi(family: str, x, y, seed: int) -> np.ndarray:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    f = mutual_info_regression if family == "regression" else mutual_info_classif
    return f(x, y, random_state=seed)


def _task(c, spec, rows: int, rng):
    idx, y = ft.labels(c, spec)
    if len(idx) < rows:
        return None
    take = rng.choice(len(idx), size=rows, replace=False)
    idx, y = idx[take], y[take]
    half = np.arange(rows // 2)
    x = ft.design(c, spec, idx, half)
    return x, y


def compare(real, cand, n_tasks: int = 40, rows: int = 400, seed: int = 0) -> dict:
    """Per task and summarized (medians): prevalence gap, difficulty gap, MI rank correlation, TSTR / TRTR."""
    from scipy.stats import spearmanr

    prior = ft.ICUPrior(real, seed=seed, filter_margin=None, dedup=True, realism_share=0.0, min_rows=rows,
                        max_rows=rows)
    rng = np.random.default_rng(seed)
    out = []
    tries = 0
    while len(out) < n_tasks and tries < 20 * n_tasks:
        tries += 1
        spec = prior.sample_spec()
        spec.realism = {}
        r = _task(real, spec, rows, rng)
        s = _task(cand, spec, rows, rng)
        if r is None or s is None:
            continue
        (xr, yr), (xs, ys) = r, s
        if xr.shape[1] != xs.shape[1] or xr.shape[1] == 0:      # a column constant on one side only: not comparable
            continue
        if spec.family == "regression":
            sd = np.std(yr) or 1.0
            prev = abs(np.mean(ys) - np.mean(yr)) / sd
        else:
            cls = np.union1d(np.unique(yr), np.unique(ys))
            prev = 0.5 * sum(abs(np.mean(yr == k) - np.mean(ys == k)) for k in cls)
        h = rows // 2
        d_r = _score(spec.family, xr[:h], yr[:h], xr[h:], yr[h:], seed)
        d_s = _score(spec.family, xs[:h], ys[:h], xs[h:], ys[h:], seed)
        tstr = _score(spec.family, xs, ys, xr[h:], yr[h:], seed)
        mi_r, mi_s = _mi(spec.family, xr, yr, seed), _mi(spec.family, xs, ys, seed)
        mi_corr = float(spearmanr(mi_r, mi_s).statistic) if len(mi_r) > 2 and np.std(mi_r) > 0 and np.std(mi_s) > 0 else None
        base = 0.0 if spec.family == "regression" else 0.5
        ratio = ((tstr - base) / (d_r - base) if tstr is not None and d_r is not None and d_r - base > 0.02 else None)
        out.append({"hash": ft.spec_hash(spec)[:12], "family": spec.family, "target": spec.target,
                    "prevalence_gap": float(prev), "difficulty_real": d_r, "difficulty_candidate": d_s,
                    "mi_rank_corr": mi_corr, "tstr_over_trtr": ratio})
    med = lambda k: float(np.median([t[k] for t in out if t[k] is not None])) if any(t[k] is not None for t in out) else None  # noqa: E731
    gaps = [abs(t["difficulty_real"] - t["difficulty_candidate"]) for t in out
            if t["difficulty_real"] is not None and t["difficulty_candidate"] is not None]
    return {"tasks": len(out), "per_task": out,
            "summary": {"prevalence_gap": med("prevalence_gap"), "difficulty_gap": float(np.median(gaps)) if gaps else None,
                        "mi_rank_corr": med("mi_rank_corr"), "tstr_over_trtr": med("tstr_over_trtr")}}
