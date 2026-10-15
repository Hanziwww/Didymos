"""Paper P's prediction benchmark (``docs/p1_paper_protocol.md`` A1, A2, B2): the twin model predicts real held-out
stays' futures by sampling, every label read by the factory's label engine.

* anchors: real stays cut at a prediction time. ``landmark_anchors`` (L24, L48): every stay still in the ICU at the
  landmark, a natural sample of ``per_dataset`` per dataset; the rolling design (R) is ``factory.oracle.choose`` on
  real stays (enriched, with inclusion weights).
* ``prompt``: the stay as known at the anchor, on its own clock (the real stores decode a stay's tokens on the clock
  keyed by its stay id, so the labels the store gives at a time and the prompt cut there agree);
  ``unknown_site`` sets its dataset and site tokens to unknown (B2).
* ``anchor_time``: the prediction time, the prompt's last event in minutes after ICU admission on that clock.
* metrics on (label, probability) pairs with inclusion weights: AUROC, AUPRC, Brier, ECE (equal-mass bins), observed
  over expected, the logistic recalibration's slope and intercept; ``bootstrap`` intervals over anchors, paired for
  differences.
"""

from __future__ import annotations

import numpy as np

from icutwin.factory import oracle


# -- anchors and prompts ---------------------------------------------------------------------------------------------
DATASETS = ("mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire")


def store_stay_ids(store_dir) -> set:
    """A store's stay ids, read with pyarrow (vLLM's environment has no pandas)."""
    from pathlib import Path

    import pyarrow.parquet as pq

    keep = set()
    for f in Path(store_dir).glob("seg_*/stays.parquet"):
        keep |= {str(x) for x in pq.read_table(f, columns=["stay_id"]).column(0).to_pylist()}
    return keep


def real_rows(tokens, split: str, keep: set, datasets=DATASETS) -> list[dict]:
    """The token sequences of a split's stays among ``keep`` (stay_id, dataset, ids); ``split`` may list several,
    comma-separated (an inference-only database's stays are all test stays)."""
    from pathlib import Path

    import pyarrow.parquet as pq

    rows = []
    for ds in datasets:
        for sp in split.split(","):
            t = pq.read_table(Path(tokens) / ds / f"{sp}.parquet", columns=["stay_id", "ids"]).to_pylist()
            rows += [{"stay_id": str(r["stay_id"]), "dataset": ds, "ids": r["ids"]} for r in t
                     if str(r["stay_id"]) in keep]
    return rows


def design_anchors(stays: list, v, design: str, per_dataset: int, seed: int = 0) -> list:
    """A1's anchors of a design: ``L24``/``L48`` landmarks, ``R`` the rolling design (``factory.oracle.choose``)."""
    if design in ("L24", "L48"):
        return landmark_anchors(stays, v, float(design[1:]), per_dataset, seed)
    if design == "R":
        return oracle.choose(stays, v, per_dataset, seed, horizon=48.0)
    raise ValueError(f"unknown design {design}")


def landmark_anchors(stays: list, v, hours: float, per_dataset: int, seed: int = 0) -> list:
    """Anchors at ``hours`` after ICU admission (``stays``: (stay id, dataset, stay of the labels' decode)): stays
    whose ICU stay lasts past the landmark (so alive and in the ICU there), at most ``per_dataset`` per dataset drawn
    at random; weight 1 (a natural sample)."""
    rng = np.random.default_rng(seed)
    codes = oracle.event_codes(v)
    by_ds: dict = {}
    for sid, ds, s in stays:
        los = (s.icu_out - s.icu_in).total_seconds() / 3600
        if los > hours:
            by_ds.setdefault(ds, []).append((sid, los, s))
    out = []
    for ds in sorted(by_ds):
        items = by_ds[ds]
        pick = rng.permutation(len(items))[:per_dataset]
        for k in sorted(pick):
            sid, los, s = items[k]
            flag = oracle.flagged(oracle.onsets(s, codes), hours, 48.0)
            out.append(oracle.Anchor(f"{sid}@{hours:g}", sid, ds, float(hours), los, oracle.los_band(los), False,
                                     bool(flag), 1.0))
    return out


def prompt(ids, v, stay_id: str, hours: float) -> list[int]:
    """The real stay as known ``hours`` after its ICU admission, on its own clock."""
    return oracle.prompt(ids, v, stay_id, hours)


def unknown_site(p: list[int], v) -> list[int]:
    """The prompt with its dataset and site tokens set to unknown (the training-time augmentation's tokens)."""
    from icutwin.model.data import layout

    lay = layout(v)
    out = list(p)
    for k, tok in enumerate(out[:16]):
        m = v.meta[int(tok)]
        if m.get("cls") == "prefix" and m.get("field") == "dataset":
            out[k] = lay.dataset_unknown
        elif m.get("cls") == "prefix" and m.get("field") == "site":
            out[k] = lay.site_unknown
    return out


def anchor_time(p: list[int], v, stay_id: str) -> int:
    """The prediction time of a prompt: its last event, minutes after ICU admission on the stay's clock."""
    from icutwin.factory import twins

    end = [v.id("[END]"), v.id("[EOS]")]
    s = twins.decode_labels(list(p) + end, v, stay_id, clock_id=stay_id)
    return int(np.floor(max((e[0] - s.icu_in).total_seconds() for e in s.timeline) / 60))


# -- metrics ---------------------------------------------------------------------------------------------------------
def ece(y, p, w=None, bins: int = 10) -> float:
    """Expected calibration error over equal-mass bins of p (weighted)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    order = np.argsort(p, kind="stable")
    cw = np.cumsum(w[order]) / w.sum()
    idx = np.minimum((cw * bins - 1e-12).astype(int), bins - 1)
    err = 0.0
    for b in range(bins):
        k = order[idx == b]
        if len(k):
            err += w[k].sum() / w.sum() * abs(np.average(y[k], weights=w[k]) - np.average(p[k], weights=w[k]))
    return float(err)


def recalibration(y, p, w=None) -> tuple[float, float]:
    """(slope, intercept) of the logistic regression of y on logit(p) (1, 0 when calibrated)."""
    from scipy.optimize import minimize

    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    x = np.log(p / (1 - p))

    def nll(b):
        z = b[0] + b[1] * x
        return float(np.sum(w * (np.logaddexp(0, z) - y * z)))

    r = minimize(nll, np.array([0.0, 1.0]), method="BFGS")
    return float(r.x[1]), float(r.x[0])


def binary_metrics(y, p, w=None) -> dict:
    """Discrimination and calibration of probabilities ``p`` for 0/1 labels ``y`` (weights ``w``)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    out = {"n": int(len(y)), "positives": int(y.sum()), "prevalence": float(np.average(y, weights=w)),
           "mean_predicted": float(np.average(p, weights=w)),
           "brier": float(np.average((y - p) ** 2, weights=w))}
    if 0 < y.sum() < len(y):
        slope, icpt = recalibration(y, p, w)
        out.update(auroc=float(roc_auc_score(y, p, sample_weight=w)),
                   auprc=float(average_precision_score(y, p, sample_weight=w)), ece=ece(y, p, w),
                   observed_over_expected=float(np.average(y, weights=w) / max(np.average(p, weights=w), 1e-9)),
                   calibration_slope=slope, calibration_intercept=icpt)
    return out


def bootstrap(fn, groups, *arrays, n: int = 1000, seed: int = 0, strata=None) -> tuple[float, float]:
    """The 95% percentile interval of ``fn(*arrays)`` resampled by ``groups`` (stays; all rows of a drawn group
    together), within ``strata`` (e.g. datasets) when given. ``fn`` returns a float or None (skipped)."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    strata = np.zeros(len(groups), dtype=int) if strata is None else np.asarray(strata)
    by = {}
    for s in np.unique(strata):
        g = np.unique(groups[strata == s])
        by[s] = g
    rows = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    vals = []
    for _ in range(n):
        idx = np.concatenate([rows[g] for s, gs in by.items() for g in rng.choice(gs, size=len(gs))])
        v = fn(*(np.asarray(a)[idx] for a in arrays))
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < n // 2:
        return (None, None)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_draws(fn, groups, *arrays, n: int = 1000, seed: int = 0, strata=None) -> np.ndarray:
    """The ``n`` resampled values of ``fn(*arrays)`` that ``bootstrap`` takes its interval from: the same draws (groups
    within ``strata``, ``seed``), NaN where ``fn`` returns None or a non-finite value. ``bootstrap``'s interval is the
    2.5 and 97.5 percentiles of the finite draws (when at least half are finite)."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    strata = np.zeros(len(groups), dtype=int) if strata is None else np.asarray(strata)
    by = {}
    for s in np.unique(strata):
        g = np.unique(groups[strata == s])
        by[s] = g
    rows = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    out = np.full(n, np.nan)
    for i in range(n):
        idx = np.concatenate([rows[g] for s, gs in by.items() for g in rng.choice(gs, size=len(gs))])
        v = fn(*(np.asarray(a)[idx] for a in arrays))
        if v is not None and np.isfinite(v):
            out[i] = v
    return out


def auroc_or_none(y, p, w=None):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p, sample_weight=w))


def auroc_fast(y, s, w=None) -> float:
    """The weighted AUROC of scores ``s`` for 0/1 labels ``y`` (ties count one half, as ``roc_auc_score``), for
    bootstrap loops; NaN without both classes."""
    y, s = np.asarray(y, dtype=np.float64), np.asarray(s, dtype=np.float64)
    w = np.ones_like(s) if w is None else np.asarray(w, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    start = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    wp, wn = np.add.reduceat(w * y, start), np.add.reduceat(w * (1 - y), start)
    pos, neg = wp.sum(), wn.sum()
    if pos <= 0 or neg <= 0:
        return float("nan")
    return float((wp * (np.cumsum(wn) - wn + 0.5 * wn)).sum() / (pos * neg))


def resamples(groups, n: int = 1000, seed: int = 0, strata=None) -> list[np.ndarray]:
    """``n`` bootstrap resamples (row indices) by ``groups`` (all rows of a drawn group together) within ``strata``,
    as ``bootstrap`` draws them; one set shared by several statistics makes their differences paired."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    strata = np.zeros(len(groups), dtype=int) if strata is None else np.asarray(strata)
    rows = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    by = {s: np.unique(groups[strata == s]) for s in np.unique(strata)}
    return [np.concatenate([rows[g] for gs in by.values() for g in rng.choice(gs, size=len(gs))]) for _ in range(n)]


def crps_ensemble(x, y: float) -> float:
    from icutwin.eval.forecast import crps_ensemble as _c

    return _c(np.asarray(x, float), float(y))


def interval_coverage(ens: np.ndarray, y: np.ndarray, levels=(0.5, 0.8, 0.95)) -> dict:
    """Share of truths inside the ensembles' central intervals (``ens`` [n, k] with NaN for missing members)."""
    out = {}
    for lv in levels:
        lo = np.nanquantile(ens, (1 - lv) / 2, axis=1)
        hi = np.nanquantile(ens, 1 - (1 - lv) / 2, axis=1)
        out[f"{lv:g}"] = float(np.mean((y >= lo) & (y <= hi)))
    return out


def c_index(time, event, score) -> float | None:
    """Harrell's C: among comparable pairs (the earlier time an event), the share where the earlier one has the
    higher risk ``score`` (ties 1/2)."""
    t, e, s = (np.asarray(x, float) for x in (time, event, score))
    num = den = 0.0
    for i in np.flatnonzero(e > 0):
        later = t > t[i]
        den += later.sum()
        num += (s[i] > s[later]).sum() + 0.5 * (s[i] == s[later]).sum()
    return float(num / den) if den else None


def km_survival(times: np.ndarray, events: np.ndarray, t: float) -> float:
    """Kaplan-Meier survival at ``t`` from one ensemble's durations (``events`` 1 where the end was observed, 0 where
    censored); NaN when no member is still at risk at an event time before ``t`` and none ended (no information)."""
    times, events = np.asarray(times, float), np.asarray(events, bool)
    ok = np.isfinite(times)
    times, events = times[ok], events[ok]
    if not len(times):
        return float("nan")
    s = 1.0
    for u in np.unique(times[events & (times <= t)]):
        at_risk = (times >= u).sum()
        if at_risk:
            s *= 1.0 - ((times == u) & events).sum() / at_risk
    if (times > t).sum() == 0 and (events & (times <= t)).sum() == 0:
        return float("nan")
    return float(s)
