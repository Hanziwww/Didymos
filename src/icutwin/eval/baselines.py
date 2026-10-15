"""Paper P's supervised baselines (``docs/p1_paper_protocol.md`` A1, A4): tabular features of a stay as known at a
prediction time, read from a factory stay store, and the models trained on them.

* ``feature_channels``: the input numeric channels charted in the most stays of a store (``k`` of them).
* ``Featurizer``: per row (stay, t): for each channel the last value, mean, minimum, maximum and count over the 6 and
  24 hours before t (``labels.window_values``) and the hours since its last value; the treatment states active at t
  (IMV, NIPPV, vasoactives, RRT; the derived episodes, which only use data charted by then); statics documented at
  ICU admission (age, sex, unit type, admission type and source; the diagnosis group and surgical status are left out:
  several datasets document them up to 24 hours later); the dataset; hours since ICU admission.
* ``training_rows``: a store's rows for a design: the landmark (24 or 48 h) of every stay still in the ICU, or up to
  ``per_stay`` random ICU hours per stay (R).
* ``fit_lgbm`` / ``fit_logistic``: LightGBM (1,000 trees at most, learning rate 0.05, 31 leaves, early stopping on a
  validation set) and L2 logistic regression (median imputation, missing indicators, standardization).
* ``sofa24``: the first-day SOFA total of each row's stay (a score for death at the landmarks).
"""

from __future__ import annotations

import numpy as np

from icutwin.factory import labels as fl
from icutwin.factory import programs as fpr

WINDOWS = (360, 1440)
AGGS = ("last", "mean", "min", "max", "count")
STATES = ("imv", "nippv", "vasopressor", "rrt")
STATICS = ("sex", "unit_type", "admission_type", "admission_source")


def feature_channels(store, k: int = 64) -> list[str]:
    """The ``k`` input numeric channels charted in the most stays."""
    ch = store.channels
    cov = []
    for cid, name in enumerate(ch.names):
        if ch.kind[cid] != "numeric" or not ch.is_input(cid):
            continue
        ev = store.channel_events(cid)
        n = len(np.unique(ev["stay"])) if len(ev["stay"]) else 0
        if n:
            cov.append((n, name))
    return [n for _c, n in sorted(cov, key=lambda x: (-x[0], x[1]))[:k]]


class Featurizer:
    """Features of rows (stay, t) of one store (module docstring); ``levels`` fixes the statics' categories (those of
    the training store) so train and test columns agree."""

    def __init__(self, channels: list[str], levels: dict | None = None):
        self.channels = list(channels)
        self.levels = levels

    def fit_levels(self, store) -> Featurizer:
        st = store.stays
        self.levels = {f: sorted(str(x) for x in st[f].dropna().unique()) if f in st else [] for f in STATICS}
        self.levels["dataset"] = sorted(str(x) for x in st["dataset"].unique())
        return self

    def names(self) -> list[str]:
        out = [f"{c}|{agg}|{w}" for c in self.channels for w in WINDOWS for agg in AGGS]
        out += [f"{c}|hours_since" for c in self.channels]
        out += [f"state|{s}" for s in STATES] + ["age", "hours_in_icu"]
        out += [f"{f}={x}" for f in (*STATICS, "dataset") for x in self.levels[f]]
        return out

    def __call__(self, store, rows: fl.Rows, ev: fpr.Evaluator | None = None) -> np.ndarray:
        ev = ev or fpr.Evaluator(store)
        cols = []
        for c in self.channels:
            e = fl.finite(ev.events(c)) if c in store.channels.index else None
            for w in WINDOWS:
                for agg in AGGS:
                    if e is None or not len(e["stay"]):
                        cols.append(np.full(len(rows), np.nan) if agg != "count" else np.zeros(len(rows)))
                        continue
                    y, _n = fl.window_values(e, rows, -w, 0, agg)
                    cols.append(y.astype(np.float64))
            cols.append(self._since(e, rows))
        for s in STATES:
            cols.append(self._in_state(ev, s, rows))
        st = store.stays
        age = st["age"].to_numpy(dtype=np.float64) if "age" in st else np.full(len(st), np.nan)
        cols.append(age[rows.stay])
        cols.append(rows.t / 60.0)
        for f in (*STATICS, "dataset"):
            vals = st[f].astype(str).to_numpy() if f in st else np.full(len(st), "", dtype=object)
            for x in self.levels[f]:
                cols.append((vals[rows.stay] == x).astype(np.float64))
        return np.stack(cols, 1)

    @staticmethod
    def _since(e, rows: fl.Rows) -> np.ndarray:
        if e is None or not len(e["stay"]):
            return np.full(len(rows), np.nan)
        keys = fl.keyed(e)
        j = np.searchsorted(keys, fl.comp(rows.stay, rows.t), "right") - 1
        jc = np.maximum(j, 0)
        same = (j >= 0) & (np.asarray(e["stay"])[jc] == rows.stay)
        return np.where(same, (rows.t - np.asarray(e["t"])[jc]) / 60.0, np.nan)

    @staticmethod
    def _in_state(ev: fpr.Evaluator, kind: str, rows: fl.Rows) -> np.ndarray:
        try:
            p = ev.process(("event", kind))
        except (KeyError, ValueError):
            return np.zeros(len(rows))
        if not len(p.stay):
            return np.zeros(len(rows))
        keys = fl.comp(p.stay, p.s)
        k = np.searchsorted(keys, fl.comp(rows.stay, rows.t), "right") - 1
        kc = np.maximum(k, 0)
        return ((k >= 0) & (p.stay[kc] == rows.stay) & (p.e[kc] > rows.t)).astype(np.float64)


def training_rows(store, design: str, per_stay: int = 4, seed: int = 0, stays=None) -> fl.Rows:
    """A store's training rows for a design (module docstring)."""
    if design in ("L24", "L48"):
        return fl.landmark(store, int(design[1:]) * 60, stays)
    rng = np.random.default_rng(seed)
    r = fl.rolling(store, 60, stays)                 # ordered by stay
    _u, first, count = np.unique(r.stay, return_index=True, return_counts=True)
    keep = [f + rng.choice(c, size=min(per_stay, c), replace=False) for f, c in zip(first, count, strict=True)]
    k = np.sort(np.concatenate(keep)) if keep else np.zeros(0, dtype=np.int64)
    return r.take(k)


def labels(store, prog, rows: fl.Rows, ev: fpr.Evaluator | None = None) -> np.ndarray:
    """The program's labels at the rows (NaN where it does not apply)."""
    b = (ev or fpr.Evaluator(store)).evaluate(prog, rows)
    return np.where(b.valid, b.y, np.nan).astype(np.float64)


def fit_lgbm(x, y, x_val=None, y_val=None, seed: int = 0, n_jobs: int = 4):
    import lightgbm as lgb

    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=n_jobs)
    if x_val is not None and len(np.unique(y_val)) == 2:
        m.fit(x, y, eval_set=[(x_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        m.set_params(n_estimators=300)
        m.fit(x, y)
    return m


def fit_logistic(x, y, seed: int = 0):
    from sklearn.impute import MissingIndicator, SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(make_union(SimpleImputer(strategy="median", keep_empty_features=True),
                                    MissingIndicator(features="all")),
                         StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
    return pipe.fit(x, y)


def sofa24(store, rows: fl.Rows) -> np.ndarray:
    """The first-day SOFA total of each row's stay (NaN when not derived)."""
    d = store.derived_events("sofa24:sofa_total")
    out = np.full(len(store), np.nan)
    if len(d["stay"]):
        out[np.asarray(d["stay"], dtype=np.int64)] = np.asarray(d["v"], dtype=np.float64)
    return out[rows.stay]
