"""The task prior and the ICL episode contract (plan S9; *Prediction-task generation*).

A task (``TaskSpec``): a cohort (datasets), a trigger (hours after ICU admission: the observation window is
[0, trigger)), a target of one family, features (concepts of the cache's grid x an aggregator over the window, and
statics), and the episode's size and context share. Targets:
* binary: an event (IMV, vasopressor, RRT, AKI stage >= 2) within ``horizon`` hours after the trigger, among stays
  in the ICU at the trigger without the event before it; in-hospital death; still in the ICU at trigger + horizon;
* multiclass (<= 10 classes): the band of the remaining ICU stay; the 24-hour SOFA band (trigger >= 24);
* regression: a concept's value in the hour trigger + horizon (stays with one);
* mined (MOTOR-style, from the grid's concepts): a concept above its population 90th percentile, or below its 10th,
  within the horizon (binary, stays with a value there); a concept's change from its last value before the trigger
  to the hour trigger + horizon (regression).
Realism (O'Prior): each task may carry missingness (MCAR, MAR on another column, MNAR on the value itself), noise,
unit changes of a few columns, and a context / query shift (the context mostly from one dataset, the query from the
others), applied to the raw features before the context-mean imputation, seeded by the task's hash.
Features read the window only (a leakage test checks it); missing values become the context mean with an indicator
when the feature is missing for some rows; constant columns are dropped.

``ICUPrior.get_batch(B)`` returns TabICL's ``PriorDataset.get_batch`` layout: ``X[B, T, H]``, ``y[B, T]``,
``d[B]`` (features per episode), ``seq_lens[B]`` (rows), ``train_sizes[B]`` (context rows, 0.1-0.9 of T), zero
padded; and ``get_ts_batch`` the time-series contract ``X_ts[B, T, L, C]``, ``M_ts`` (observed), ``X_static``, ``y``,
``train_sizes``.

Task-diversity controls (plan *Diversity assurance* 5, TabICLv2): an ExtraTrees model fitted on an episode's context
must beat chance on its query (AUROC > 0.5 + margin; R^2 > margin) or the task is drawn again (``informative``); the
rows follow TabICLv2's curriculum stages (``CURRICULUM``, log-uniform within a stage); a task (its spec without the
episode's size and split, ``spec_hash``) is drawn at most once per prior; ``stats`` counts every draw's fate, by
family, and a family below its floor (default: half of an equal share) is drawn until it is back above it. Every episode has a task card (``task_card``).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np

from icutwin.factory.cache import EVENTS, STATIC_CATEGORICAL, STATIC_NUMERIC, Cache
from icutwin.factory.contracts import check_batch

AGGREGATORS = ("last", "mean", "min", "max", "count")
BINARY = tuple(f"{e}_within" for e in EVENTS) + ("death", "in_icu_at")
MINED_BINARY = ("concept_above", "concept_below")
MINED_REGRESSION = ("concept_change",)
MULTICLASS = ("remaining_los_band", "sofa_band")
LOS_BANDS = (0, 24, 48, 96, 168, 336)            # hours (the goal block's)
SOFA_BANDS = (0, 2, 4, 6, 8, 10, 12)
CURRICULUM = ((1024, 1024), (400, 10240), (400, 60000))      # TabICLv2's stages: rows, log-uniform within


@dataclass
class TaskSpec:
    family: str                       # binary | multiclass | regression
    target: str
    trigger: int                      # hours
    horizon: int = 0                  # hours
    concept: int = -1                 # regression target's concept
    datasets: tuple = ()
    features: list = field(default_factory=list)   # (concept index, aggregator)
    statics: tuple = ()
    rows: int = 512
    train_ratio: float = 0.5
    threshold: float | None = None    # a mined binary target's threshold (the concept's population quantile)
    realism: dict = field(default_factory=dict)   # O'Prior transforms: mcar, mar, mnar, noise, units, shift


def window_features(grid: np.ndarray, trigger: int, feats: list) -> np.ndarray:
    """[n, len(feats)] aggregates over hours [0, trigger) (NaN where a stay has no value)."""
    w = grid[:, :trigger, :]
    out = np.full((grid.shape[0], len(feats)), np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slices: NaN by design
        for k, (c, agg) in enumerate(feats):
            x = w[:, :, c]
            if agg == "count":
                out[:, k] = np.sum(~np.isnan(x), axis=1)
            elif agg == "mean":
                out[:, k] = np.nanmean(x, axis=1)
            elif agg == "min":
                out[:, k] = np.nanmin(x, axis=1)
            elif agg == "max":
                out[:, k] = np.nanmax(x, axis=1)
            else:   # last
                has = ~np.isnan(x)
                idx = np.where(has.any(1), x.shape[1] - 1 - np.argmax(has[:, ::-1], axis=1), 0)
                out[:, k] = np.where(has.any(1), x[np.arange(len(x)), idx], np.nan)
    return out


def labels_tte(c: Cache, event: str, trigger: int, datasets: tuple = ()) -> tuple:
    """Time-to-event labels (plan *Prediction-task generation* 5): among stays in the ICU at the trigger without the
    event before it, hours from the trigger to the event's first onset (``y_event`` 1) or to the ICU discharge, where
    the stay is censored (``y_event`` 0). Returns (stays, y_time, y_event)."""
    cohort = np.isin(c.dataset, datasets) if datasets else np.ones(len(c.los_hours), dtype=bool)
    on = c.onset[event]
    ok = cohort & (c.los_hours > trigger) & ~(on <= trigger)
    observed = ok & (on > trigger) & (on <= c.los_hours)
    y_time = np.where(observed, on, c.los_hours) - trigger
    idx = np.flatnonzero(ok)
    return idx, y_time[idx], observed[idx].astype(float)


def labels_multilabel(c: Cache, events: tuple, trigger: int, horizon: int, datasets: tuple = ()) -> tuple:
    """Multi-label targets (plan *Prediction-task generation* 5): per stay in the ICU at the trigger, each event's
    onset within (trigger, trigger + horizon] as one column of ``Y`` [n, K], NaN where that event began before the
    trigger (the column does not apply to the stay). Returns (stays, Y)."""
    cohort = np.isin(c.dataset, datasets) if datasets else np.ones(len(c.los_hours), dtype=bool)
    idx = np.flatnonzero(cohort & (c.los_hours > trigger))
    cols = []
    for e in events:
        on = c.onset[e][idx]
        y = ((on > trigger) & (on <= trigger + horizon)).astype(float)
        y[on <= trigger] = np.nan
        cols.append(y)
    return idx, np.stack(cols, axis=1) if cols else np.zeros((len(idx), 0))


def multilabel_to_binary(y: np.ndarray, k: int) -> tuple:
    """The adapter of a multi-label episode to its k-th binary task: (rows where the column applies, labels)."""
    keep = np.flatnonzero(~np.isnan(y[:, k]))
    return keep, y[keep, k]


def binary_at_horizon(y_time: np.ndarray, y_event: np.ndarray, horizon: float) -> tuple:
    """The censoring-aware reduction of time-to-event labels to a binary task at ``horizon`` hours: the event by
    then is 1, no event and followed past the horizon is 0, censored before it without the event is left out.
    Returns (rows kept, labels)."""
    pos = (y_event == 1) & (y_time <= horizon)
    keep = pos | (y_time > horizon)
    return np.flatnonzero(keep), pos[keep].astype(float)


def labels(c: Cache, spec: TaskSpec) -> tuple[np.ndarray, np.ndarray]:
    """(eligible stays, label per eligible stay) of a task."""
    t, h = spec.trigger, spec.horizon
    in_icu = c.los_hours > t
    cohort = np.isin(c.dataset, spec.datasets) if spec.datasets else np.ones(len(c.los_hours), dtype=bool)
    ok = in_icu & cohort
    if spec.target.endswith("_within"):
        on = c.onset[spec.target[:-len("_within")]]
        ok &= ~(on <= t)                                   # no onset before the trigger
        y = ((on > t) & (on <= t + h)).astype(float)
    elif spec.target == "death":
        ok &= ~np.isnan(c.death)
        y = c.death.copy()
    elif spec.target == "in_icu_at":
        y = (c.los_hours > t + h).astype(float)
    elif spec.target == "remaining_los_band":
        y = np.digitize(c.los_hours - t, LOS_BANDS[1:]).astype(float)
    elif spec.target == "sofa_band":
        ok &= ~np.isnan(c.sofa_24h) & (t >= 24)
        y = np.digitize(np.nan_to_num(c.sofa_24h), SOFA_BANDS[1:]).astype(float)
    elif spec.target in MINED_BINARY:
        if t >= c.hours:                                   # the window lies past the cache's grid
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        w = c.grid[:, t:min(t + h, c.hours), spec.concept]
        has = ~np.isnan(w).all(1)
        ok &= has
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ext = np.nanmax(w, 1) if spec.target == "concept_above" else np.nanmin(w, 1)
        y = (ext > spec.threshold) if spec.target == "concept_above" else (ext < spec.threshold)
        y = y.astype(float)
    elif spec.target == "concept_change":
        hour = t + h
        if hour >= c.hours:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        before = window_features(c.grid, t, [(spec.concept, "last")])[:, 0]
        y = c.grid[:, hour, spec.concept].astype(float) - before
        ok &= ~np.isnan(y)
    elif spec.family == "regression":
        hour = t + h
        if hour >= c.hours:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        y = c.grid[:, hour, spec.concept].astype(float)
        ok &= ~np.isnan(y)
    else:
        raise ValueError(spec.target)
    idx = np.flatnonzero(ok)
    return idx, y[idx]


def design(c: Cache, spec: TaskSpec, idx: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Features of the stays ``idx`` (rows of the cache): window aggregates, then statics; missing values are the
    mean of the ``context`` rows (positions in idx) with indicators; constant columns dropped."""
    from icutwin.factory.trajectories import availability

    x = window_features(c.grid[idx], spec.trigger, spec.features)
    cols = [x]
    avail = availability()
    for s in spec.statics:
        v = c.static[s][idx].astype(np.float64)
        if s in STATIC_CATEGORICAL:
            v = np.where(v < 0, np.nan, v)
        late = avail.get(s, {})                 # a header field documented after the trigger is unknown at it
        if late:                                # (design-review correction 3; spec.yaml available_within_hours)
            v = np.where([late.get(str(d), 0) > spec.trigger * 60 for d in c.dataset[idx]], np.nan, v)
        cols.append(v[:, None])
    x = np.concatenate(cols, axis=1) if cols else np.zeros((len(idx), 0))
    if spec.realism:
        x = realism(x, spec.realism, np.random.default_rng(int(spec_hash(spec)[:12], 16)))
    miss = np.isnan(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)        # a column missing in every context row: 0 below
        mu = np.nanmean(np.where(miss[context], np.nan, x[context]), axis=0) if len(context) else np.zeros(x.shape[1])
    mu = np.where(np.isnan(mu), 0.0, mu)
    x = np.where(miss, mu[None, :], x)
    some = miss.any(0) & ~miss.all(0)
    x = np.concatenate([x, miss[:, some].astype(np.float64)], axis=1)
    keep = np.ptp(x[context], axis=0) > 0 if len(context) else np.ones(x.shape[1], dtype=bool)
    return x[:, keep]


def realism(x: np.ndarray, r: dict, rng: np.random.Generator) -> np.ndarray:
    """O'Prior's realism transforms of raw features (NaN: missing), before imputation: ``mcar`` (each value missing
    with this probability), ``mar`` (a column's values missing more where another column is high), ``mnar`` (a
    column's values missing more where they are high themselves), ``noise`` (multiplicative Gaussian noise of this
    scale), ``units`` (this many columns rescaled by a unit factor)."""
    x = x.copy()
    n, d = x.shape
    if d == 0 or n == 0:
        return x
    if r.get("mcar"):
        x[rng.random((n, d)) < r["mcar"]] = np.nan
    for key in ("mar", "mnar"):
        p = r.get(key)
        if not p:
            continue
        j = int(rng.integers(d))
        driver = x[:, int(rng.integers(d))] if key == "mar" else x[:, j]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # an all-missing column: no driver
            z = np.where(np.isnan(driver), np.nan, (driver - np.nanmean(driver)) / (np.nanstd(driver) + 1e-9))
        prob = np.where(np.isnan(z), p, 2 * p / (1 + np.exp(-2 * np.nan_to_num(z))))
        x[rng.random(n) < prob, j] = np.nan
    if r.get("noise"):
        x = x * (1 + rng.normal(0, r["noise"], size=x.shape))
    for j in rng.choice(d, size=min(int(r.get("units", 0)), d), replace=False):
        x[:, j] = x[:, j] * float(rng.choice([0.001, 0.01, 0.1, 10.0, 100.0, 1000.0]))
    return x


def spec_hash(spec: TaskSpec) -> str:
    """The task's identity: its spec without the episode's size and context share (canonical JSON, SHA-1)."""
    d = asdict(spec)
    d.pop("rows")
    d.pop("train_ratio")
    d["datasets"] = sorted(d["datasets"])
    d["features"] = sorted(map(list, d["features"]))
    d["statics"] = sorted(d["statics"])
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()


def informative(x: np.ndarray, y: np.ndarray, n_train: int, family: str, margin: float = 0.05,
                seed: int = 0, z: float = 2.33) -> tuple[bool, float]:
    """TabICLv2's filter: an ExtraTrees model fitted on the context rows, scored on the query rows (AUROC, one
    against the rest for several classes; R^2 for regression); informative when above 0.5 + the larger of
    ``margin`` and ``z`` chance standard deviations of the AUROC (Mann-Whitney under no signal: the positive class
    against the rest, or the classes' one-against-rest average; about 0.04 at 200 query rows), R^2 above ``margin``."""
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.metrics import roc_auc_score

    xa, ya, xb, yb = x[:n_train], y[:n_train], x[n_train:], y[n_train:]
    if family == "regression":
        m = ExtraTreesRegressor(n_estimators=25, min_samples_leaf=2, random_state=seed, n_jobs=1).fit(xa, ya)
        var = np.mean((yb - ya.mean()) ** 2)
        score = 1 - np.mean((m.predict(xb) - yb) ** 2) / var if var > 0 else 0.0
        return bool(score > margin), float(score)
    m = ExtraTreesClassifier(n_estimators=25, min_samples_leaf=2, random_state=seed, n_jobs=1).fit(xa, ya)
    p = m.predict_proba(xb)
    if p.shape[1] == 2:
        score = roc_auc_score(yb, p[:, 1])
    else:
        score = roc_auc_score(yb, p, multi_class="ovr", labels=m.classes_)
    counts = np.unique(yb, return_counts=True)[1]
    var = [(len(yb) + 1) / (12 * k * (len(yb) - k)) for k in counts]       # each class against the rest
    chance_sd = np.sqrt(var[-1] if len(counts) == 2 else np.sum(var) / len(counts) ** 2)
    return bool(score > 0.5 + max(margin, z * chance_sd)), float(score)


def _target_text(c: Cache, spec: TaskSpec) -> str:
    t, h = spec.trigger, spec.horizon
    if spec.target.endswith("_within"):
        ev = spec.target[:-len("_within")]
        return f"{ev} starting within {h} h after hour {t} of the ICU stay, among stays without {ev} before hour {t}"
    if spec.target in MINED_BINARY:
        side = "above" if spec.target == "concept_above" else "below"
        return (f"{c.concepts[spec.concept]} {side} {spec.threshold:.4g} within {h} h after hour {t}, among stays with "
                "a value there")
    if spec.target == "concept_change":
        return f"change of {c.concepts[spec.concept]} from its last value before hour {t} to hour {t + h}"
    named = {"death": "in-hospital death",
             "in_icu_at": f"still in the ICU at hour {t + h}",
             "remaining_los_band": f"band of the remaining ICU stay after hour {t} (band edges {LOS_BANDS[1:]} h)",
             "sofa_band": f"band of the 24-hour SOFA score (band edges {SOFA_BANDS[1:]})"}
    if spec.target in named:
        return named[spec.target]
    return f"value of {c.concepts[spec.concept]} in hour {t + h}"


def task_card(c: Cache, spec: TaskSpec, y: np.ndarray, n_train: int, score: float | None = None) -> dict:
    """The episode's card (plan *Prediction-task generation* 5): the task in words, its windows, cohort, features,
    label summary and the filter's score."""
    out = {"hash": spec_hash(spec), "family": spec.family, "target": spec.target,
           "description": f"{spec.family}: {_target_text(c, spec)}; stays in the ICU at hour {spec.trigger}",
           "windows": {"trigger": f"ICU admission + {spec.trigger} h", "input": f"[0, {spec.trigger}) h",
                       "target": (f"({spec.trigger}, {spec.trigger + spec.horizon}] h" if spec.horizon
                                  else "the stay's outcome")},
           "cohort": list(spec.datasets) or "all",
           "features": [f"{c.concepts[k]}:{agg}" for k, agg in spec.features] + list(spec.statics),
           "rows": int(len(y)), "context_rows": int(n_train), "filter_score": score}
    if spec.family == "regression":
        out["label"] = {"mean": float(np.mean(y)), "sd": float(np.std(y))}
    else:
        k, n = np.unique(y, return_counts=True)
        out["label"] = {"classes": int(len(k)), "shares": (n / n.sum()).round(4).tolist()}
        if len(k) == 2:
            out["label"]["prevalence"] = float(n[1] / n.sum())
    return out


class ICUPrior:
    """Episodes from a cache (module docstring); every draw is seeded, so a batch is reproducible. ``stage``: a
    TabICLv2 curriculum stage (its row range replaces ``min_rows`` / ``max_rows``); ``filter_margin`` None turns
    the ExtraTrees filter off."""

    def __init__(self, cache: Cache, min_rows: int = 64, max_rows: int = 1024, max_features: int = 100,
                 max_classes: int = 10, seed: int = 0, families: tuple = ("binary", "multiclass", "regression"),
                 stage: int | None = None, filter_margin: float | None = 0.05, dedup: bool = True,
                 floors: dict | None = None, mined_share: float = 0.3, realism_share: float = 0.5):
        if stage is not None:
            min_rows, max_rows = CURRICULUM[stage]
        self.c, self.min_rows, self.max_rows = cache, min_rows, max_rows
        self.max_features, self.max_classes, self.families = max_features, max_classes, families
        self.filter_margin, self.dedup, self.seed = filter_margin, dedup, seed
        self.floors = floors if floors is not None else {f: 0.5 / len(families) for f in families}
        self.mined_share, self.realism_share, self.thresholds = mined_share, realism_share, {}
        self.rng = np.random.default_rng(seed)
        self.datasets = sorted(set(cache.dataset.tolist()))
        self.seen: set = set()
        self.stats: Counter = Counter()
        self.last_score: float | None = None

    def family(self) -> str:
        """A family, uniformly; the one furthest below its floor (a minimum share of the accepted episodes, plan
        *Diversity assurance* 5) instead when the filter has pushed one under it."""
        n = sum(self.stats[f"accepted:{f}"] for f in self.families)
        if n >= 10:
            gap = {f: self.floors.get(f, 0.0) - self.stats[f"accepted:{f}"] / n for f in self.families}
            low = max(gap, key=gap.get)
            if gap[low] > 0:
                return low
        return str(self.rng.choice(self.families))

    def sample_spec(self) -> TaskSpec:
        r, c = self.rng, self.c
        fam = self.family()
        trig = int(r.choice([6, 12, 24, 48]))
        horizon = int(r.choice([6, 12, 24, 48]))
        mined = r.random() < self.mined_share
        concept = int(r.integers(len(c.concepts)))
        threshold = None
        if fam == "binary":
            target = str(r.choice(MINED_BINARY if mined else BINARY))
            if mined:
                threshold = self.threshold(concept, 0.9 if target == "concept_above" else 0.1)
        elif fam == "multiclass":
            target = str(r.choice(MULTICLASS))
        else:
            target = "concept_change" if mined else "concept_value"
            horizon = int(r.choice([1, 6, 12, 24]))
        ds = tuple(r.choice(self.datasets, size=int(r.integers(1, len(self.datasets) + 1)), replace=False).tolist())
        n_feat = int(r.integers(2, min(self.max_features // 2, len(c.concepts) * len(AGGREGATORS)) + 1))
        conc = r.choice(len(c.concepts), size=min(n_feat, len(c.concepts)), replace=False)
        feats = [(int(k), str(r.choice(AGGREGATORS))) for k in conc]
        statics = tuple(r.choice(STATIC_NUMERIC + STATIC_CATEGORICAL, size=int(r.integers(0, 5)), replace=False).tolist())
        rows = int(np.exp(r.uniform(np.log(self.min_rows), np.log(self.max_rows))))
        real = {}
        if r.random() < self.realism_share:
            real = {"mcar": float(r.choice([0.0, 0.05, 0.2])), "mar": float(r.choice([0.0, 0.2])),
                    "mnar": float(r.choice([0.0, 0.2])), "noise": float(r.choice([0.0, 0.02, 0.1])),
                    "units": int(r.choice([0, 1, 2])), "shift": bool(len(ds) > 1 and r.random() < 0.3)}
        uses_concept = fam == "regression" or target in MINED_BINARY
        return TaskSpec(fam, target, trig, horizon, concept if uses_concept else -1, ds, feats, statics, rows,
                        float(r.uniform(0.1, 0.9)), threshold, real)

    def threshold(self, k: int, q: float) -> float:
        """A concept's population quantile over the whole grid (no row's label: a fixed clinical-style cut)."""
        if (k, q) not in self.thresholds:
            v = self.c.grid[:, :, k]
            self.thresholds[(k, q)] = float(np.nanquantile(v[~np.isnan(v)], q)) if (~np.isnan(v)).any() else 0.0
        return self.thresholds[(k, q)]

    def episode(self, max_tries: int = 1000):
        """(spec, X [T, H], y [T], train_size, stays) of one valid task (``stays``: the rows' cache indices):
        enough rows, every class in context and query, at most ``max_classes`` classes, at least one feature."""
        st = self.stats
        for _ in range(max_tries):
            spec = self.sample_spec()
            st["drawn"] += 1
            h = spec_hash(spec)
            if self.dedup and h in self.seen:
                st["duplicate"] += 1
                continue
            idx, y = labels(self.c, spec)
            if len(idx) < self.min_rows:
                st["too_few_rows"] += 1
                continue
            take = self.rng.choice(len(idx), size=min(spec.rows, len(idx)), replace=False)
            idx, y = idx[take], y[take]
            if spec.realism.get("shift"):          # shift stress: the context mostly from one dataset
                home = self.rng.choice(sorted(set(self.c.dataset[idx].tolist())))
                order = np.argsort(self.c.dataset[idx] != home, kind="stable")
                idx, y = idx[order], y[order]
            n_train = int(np.clip(round(spec.train_ratio * len(idx)), 0.1 * len(idx), 0.9 * len(idx)))
            context = np.arange(n_train)
            if spec.family != "regression":
                cls = np.unique(y)
                if not 2 <= len(cls) <= self.max_classes or set(np.unique(y[:n_train])) != set(cls) \
                        or set(np.unique(y[n_train:])) != set(cls):
                    st["classes"] += 1
                    continue
                y = np.searchsorted(cls, y).astype(np.float64)       # classes 0..k-1
            x = design(self.c, spec, idx, context)
            if x.shape[1] == 0:
                st["no_features"] += 1
                continue
            if x.shape[1] > self.max_features:
                x = x[:, :self.max_features]
            score = None
            if self.filter_margin is not None:
                ok, score = informative(x, y, n_train, spec.family, self.filter_margin, self.seed)
                if not ok:
                    st["uninformative"] += 1
                    st[f"uninformative:{spec.family}"] += 1
                    continue
            self.seen.add(h)
            st["accepted"] += 1
            st[f"accepted:{spec.family}"] += 1
            self.last_score = score
            return spec, x.astype(np.float32), y.astype(np.float32), n_train, idx
        raise RuntimeError("no valid task in max_tries draws")

    def get_batch(self, batch_size: int):
        import torch

        eps, cards = [], []
        for _ in range(batch_size):
            e = self.episode()
            eps.append(e)
            cards.append(task_card(self.c, e[0], e[2], e[3], self.last_score))
        self.last_cards = cards
        t = max(e[1].shape[0] for e in eps)
        h = max(e[1].shape[1] for e in eps)
        x = np.zeros((batch_size, t, h), dtype=np.float32)
        y = np.zeros((batch_size, t), dtype=np.float32)
        for b, (_s, xb, yb, _n, _i) in enumerate(eps):
            x[b, :xb.shape[0], :xb.shape[1]], y[b, :len(yb)] = xb, yb
        d = torch.tensor([e[1].shape[1] for e in eps])
        seq = torch.tensor([e[1].shape[0] for e in eps])
        train = torch.tensor([e[3] for e in eps])
        self.last_specs = [e[0] for e in eps]
        check_batch(x, y, d.numpy(), seq.numpy(), train.numpy(), families=[e[0].family for e in eps],
                    max_features=self.max_features, max_classes=self.max_classes)     # layer (d): the gate
        return torch.from_numpy(x), torch.from_numpy(y), d, seq, train

    def tte_episode(self, max_tries: int = 1000):
        """A time-to-event episode: (spec, X [T, H], y_time [T], y_event [T], context rows, stays), the features as
        ``design`` builds them; events and censoring in both the context and the query."""
        r = self.rng
        for _ in range(max_tries):
            spec = self.sample_spec()
            event = str(r.choice(EVENTS))
            spec = TaskSpec("time_to_event", f"{event}_time", spec.trigger, 0, -1, spec.datasets, spec.features,
                            spec.statics, spec.rows, spec.train_ratio, None, spec.realism)
            idx, y_time, y_event = labels_tte(self.c, event, spec.trigger, spec.datasets)
            if len(idx) < self.min_rows:
                continue
            take = r.choice(len(idx), size=min(spec.rows, len(idx)), replace=False)
            idx, y_time, y_event = idx[take], y_time[take], y_event[take]
            n_train = int(np.clip(round(spec.train_ratio * len(idx)), 0.1 * len(idx), 0.9 * len(idx)))
            if not (0 < y_event[:n_train].sum() < n_train and 0 < y_event[n_train:].sum() < len(idx) - n_train):
                continue
            x = design(self.c, spec, idx, np.arange(n_train))
            if x.shape[1] == 0:
                continue
            return spec, x[:, :self.max_features].astype(np.float32), y_time.astype(np.float32), \
                y_event.astype(np.float32), n_train, idx
        raise RuntimeError("no valid time-to-event task in max_tries draws")

    def get_ts_batch(self, batch_size: int, channels: int = 16):
        """Time-series episodes: ``X_ts[B, T, L, C]`` hourly windows of ``channels`` concepts (L = the batch's
        longest trigger), ``M_ts`` observed, ``X_static[B, T, S]``, ``y[B, T]``, ``train_sizes[B]``."""
        import torch

        eps = []
        for _ in range(batch_size):
            spec, _x, y, n_train, idx = self.episode()
            eps.append((spec, y, n_train, idx))
        t = max(len(e[1]) for e in eps)
        L = max(e[0].trigger for e in eps)
        xs = np.zeros((batch_size, t, L, channels), dtype=np.float32)
        ms = np.zeros((batch_size, t, L, channels), dtype=bool)
        st = np.zeros((batch_size, t, len(STATIC_NUMERIC)), dtype=np.float32)
        ys = np.zeros((batch_size, t), dtype=np.float32)
        for b, (spec, y, _n, idx) in enumerate(eps):
            ch = self.rng.choice(len(self.c.concepts), size=min(channels, len(self.c.concepts)), replace=False)
            w = self.c.grid[idx][:, :spec.trigger][:, :, ch]
            xs[b, :len(idx), :spec.trigger, :len(ch)] = np.nan_to_num(w)
            ms[b, :len(idx), :spec.trigger, :len(ch)] = ~np.isnan(w)
            st[b, :len(idx)] = np.nan_to_num(np.stack([self.c.static[k][idx] for k in STATIC_NUMERIC], 1))
            ys[b, :len(y)] = y
        return (torch.from_numpy(xs), torch.from_numpy(ms), torch.from_numpy(st), torch.from_numpy(ys),
                torch.tensor([e[2] for e in eps]))

    def get_forecast_batch(self, batch_size: int, channels: int = 16, targets: int = 4, horizon: int = 12):
        """Forecasting episodes (plan *Prediction-task generation* 5): the time-series contract's inputs over hours
        [0, trigger) and ``Y_future[B, T, horizon, targets]``, each stay's hourly values of ``targets`` concepts in
        hours [trigger, trigger + horizon) with their mask ``M_future`` (observed); stays in the ICU at the trigger.
        Returns (X_ts, M_ts, X_static, Y_future, M_future, train_sizes)."""
        import torch

        r, c = self.rng, self.c
        eps = []
        while len(eps) < batch_size:
            trig = int(r.choice([t for t in (6, 12, 24, 48) if t + horizon <= c.hours] or [c.hours - horizon]))
            idx = np.flatnonzero(c.los_hours > trig)
            if len(idx) < self.min_rows:
                continue
            idx = r.choice(idx, size=min(len(idx), int(np.exp(r.uniform(np.log(self.min_rows),
                                                                          np.log(self.max_rows))))), replace=False)
            tg = r.choice(len(c.concepts), size=min(targets, len(c.concepts)), replace=False)
            future = c.grid[idx][:, trig:trig + horizon][:, :, tg]
            if (~np.isnan(future)).sum() < len(idx):                 # too little to forecast
                continue
            n_train = int(np.clip(round(r.uniform(0.1, 0.9) * len(idx)), 0.1 * len(idx), 0.9 * len(idx)))
            eps.append((trig, idx, tg, future, n_train))
        t = max(len(e[1]) for e in eps)
        L = max(e[0] for e in eps)
        xs = np.zeros((batch_size, t, L, channels), dtype=np.float32)
        ms = np.zeros_like(xs, dtype=bool)
        st = np.zeros((batch_size, t, len(STATIC_NUMERIC)), dtype=np.float32)
        yf = np.zeros((batch_size, t, horizon, targets), dtype=np.float32)
        mf = np.zeros_like(yf, dtype=bool)
        for b, (trig, idx, tg, future, _n) in enumerate(eps):
            ch = r.choice(len(c.concepts), size=min(channels, len(c.concepts)), replace=False)
            w = c.grid[idx][:, :trig][:, :, ch]
            xs[b, :len(idx), :trig, :len(ch)] = np.nan_to_num(w)
            ms[b, :len(idx), :trig, :len(ch)] = ~np.isnan(w)
            st[b, :len(idx)] = np.nan_to_num(np.stack([c.static[k][idx] for k in STATIC_NUMERIC], 1))
            yf[b, :len(idx), :, :len(tg)] = np.nan_to_num(future)
            mf[b, :len(idx), :, :len(tg)] = ~np.isnan(future)
        self.last_forecast = [(e[0], e[2]) for e in eps]
        return (torch.from_numpy(xs), torch.from_numpy(ms), torch.from_numpy(st), torch.from_numpy(yf),
                torch.from_numpy(mf), torch.tensor([e[4] for e in eps]))


def counterfactual_labels(do: list, dont: list, outcome: str) -> tuple:
    """Counterfactual targets (plan *Prediction-task generation* 4, experimental): per prefix, the outcome's share
    among its rollouts under do(treatment) (``y1``) and under do(no treatment) (``y0``), each a list of
    ``eval.forecast.Reader.outcomes`` dicts, and the twin-world effect ``y1 - y0`` (NaN where either side has no
    rollout that resolves the outcome). Interventional within the twin world only: the effects inherit the
    observational data's confounding by indication. Returns (y0, y1, effect)."""
    def share(rs):
        known = [r[outcome] for r in rs if r.get(outcome) is not None]
        return float(np.mean(known)) if known else np.nan

    y0 = np.array([share(rs) for rs in dont])
    y1 = np.array([share(rs) for rs in do])
    return y0, y1, y1 - y0
