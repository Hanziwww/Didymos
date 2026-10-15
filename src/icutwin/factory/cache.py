"""The factory's cache (plan S9): per ICU stay, the features any task reads and the label timelines any task asks
for, so a task is assembled by indexing (no re-extraction). Built identically for real splits and for decoded
synthetic stays.

Per stay: statics (age, sex, the header's categorical fields, pre-ICU length of stay, admission hour); an hourly
grid over the first ``hours`` hours of the ICU stay of the most covered numeric concepts (vitals, labs, ... : the
last value charted in each hour, NaN without one); and label timelines in hours since ICU admission: ICU
discharge (length of stay), in-hospital death (``meds.outcomes``: the expired discharge or an ICU death), the first
IMV, vasopressor and RRT episode starts and the first AKI stage >= 2 (derived concepts), the 24-hour SOFA and the
maximum AKI stage.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from icutwin.meds.outcomes import in_hospital_death
from icutwin.tokenization.stays import Stay

STATIC_NUMERIC = ("age", "pre_icu_los", "admission_hour", "prior_icu_stays")
STATIC_CATEGORICAL = ("sex", "unit_type", "admission_type", "admission_source", "adm_dx_group", "surgical_status")
EVENTS = ("imv", "vasopressor", "rrt", "aki2")


@dataclass
class Cache:
    stay_id: list
    dataset: np.ndarray                 # [n] str
    site: np.ndarray                    # [n] str
    static: dict                        # name -> [n] float (numeric) or [n] int (category codes, -1 unknown)
    categories: dict                    # categorical name -> its labels (code = index)
    concepts: list                      # the grid's numeric codes
    grid: np.ndarray                    # [n, hours, C] float32
    los_hours: np.ndarray               # [n]
    death: np.ndarray                   # [n] float: 1 in-hospital death, 0 alive, NaN unknown
    onset: dict = field(default_factory=dict)   # event -> [n] hours of the first onset (NaN: none in the stay)
    sofa_24h: np.ndarray | None = None
    kdigo_max: np.ndarray | None = None

    @property
    def hours(self) -> int:
        return self.grid.shape[1]

    def subset(self, idx: np.ndarray) -> Cache:
        return Cache([self.stay_id[i] for i in idx], self.dataset[idx], self.site[idx],
                     {k: v[idx] for k, v in self.static.items()}, self.categories, self.concepts, self.grid[idx],
                     self.los_hours[idx], self.death[idx], {k: v[idx] for k, v in self.onset.items()},
                     None if self.sofa_24h is None else self.sofa_24h[idx],
                     None if self.kdigo_max is None else self.kdigo_max[idx])

    def save(self, path: Path) -> None:
        np.savez_compressed(path, stay_id=np.array(self.stay_id), dataset=self.dataset, site=self.site,
                            concepts=np.array(self.concepts), grid=self.grid, los_hours=self.los_hours, death=self.death,
                            sofa_24h=self.sofa_24h, kdigo_max=self.kdigo_max,
                            **{f"static__{k}": v for k, v in self.static.items()},
                            **{f"cats__{k}": np.array(v) for k, v in self.categories.items()},
                            **{f"onset__{k}": v for k, v in self.onset.items()})

    @classmethod
    def load(cls, path: Path) -> Cache:
        z = np.load(path, allow_pickle=False)
        pick = lambda p: {k[len(p):]: z[k] for k in z.files if k.startswith(p)}  # noqa: E731
        return cls(z["stay_id"].tolist(), z["dataset"], z["site"], pick("static__"),
                   {k: v.tolist() for k, v in pick("cats__").items()}, z["concepts"].tolist(), z["grid"],
                   z["los_hours"], z["death"], pick("onset__"), z["sofa_24h"], z["kdigo_max"])


def _hours(t, t0) -> float:
    return (t - t0).total_seconds() / 3600.0


def numeric_coverage(stays: list[Stay], hours: int) -> Counter:
    """Per numeric code, the stays with a value in the first ``hours`` ICU hours."""
    c = Counter()
    for s in stays:
        seen = {code for t, _r, code, num, _x in s.timeline if num is not None and 0 <= _hours(t, s.icu_in) < hours}
        c.update(seen)
    return c


def build(stays: list[Stay], derived: dict | None = None, hours: int = 72, n_concepts: int = 64,
          concepts: list | None = None) -> Cache:
    """The cache of ``stays`` (``derived``: ``tokenization.stays.load_derived`` of their datasets, keyed by stay
    id); ``concepts``: the grid's codes (default: the ``n_concepts`` best covered numeric codes of these stays)."""
    derived = derived or {}
    if concepts is None:
        concepts = [c for c, _n in numeric_coverage(stays, hours).most_common(n_concepts)]
    col = {c: i for i, c in enumerate(concepts)}
    n = len(stays)
    grid = np.full((n, hours, len(concepts)), np.nan, dtype=np.float32)
    cats = {k: sorted({str(s.header.get(k) if k != "sex" else s.prefix.get(k)) for s in stays} - {"None"})
            for k in STATIC_CATEGORICAL}
    static = {k: np.full(n, np.nan) for k in STATIC_NUMERIC}
    static.update({k: np.full(n, -1, dtype=np.int64) for k in STATIC_CATEGORICAL})
    los, death = np.zeros(n), np.full(n, np.nan)
    onset = {e: np.full(n, np.nan) for e in EVENTS}
    sofa, kmax = np.full(n, np.nan), np.full(n, np.nan)
    for i, s in enumerate(stays):
        for t, _r, code, num, _x in s.timeline:          # in time order: the hour's last value wins
            j = col.get(code)
            if j is None or num is None:
                continue
            h = _hours(t, s.icu_in)
            if 0 <= h < hours:
                grid[i, int(h), j] = num
        static["age"][i] = s.prefix.get("age") if s.prefix.get("age") is not None else np.nan
        for k in ("pre_icu_los", "admission_hour", "prior_icu_stays"):
            x = s.header.get(k)
            static[k][i] = float(x) if isinstance(x, (int, float)) else np.nan
        for k in STATIC_CATEGORICAL:
            x = s.prefix.get(k) if k == "sex" else s.header.get(k)
            static[k][i] = cats[k].index(str(x)) if x is not None and str(x) in cats[k] else -1
        los[i] = _hours(s.icu_out, s.icu_in)
        death[i] = in_hospital_death([*s.end, *(c for _t, _r, c, _n, _x in s.timeline if c.startswith("ICU//discharge//"))])
        for ep, a, _b in (derived.get("episodes") or {}).get(s.stay_id, []):
            key = {"imv": "imv", "vasopressor": "vasopressor", "rrt": "rrt"}.get(ep)
            if key is not None:
                h = _hours(a, s.icu_in)
                if h >= 0 and not (onset[key][i] <= h):
                    onset[key][i] = h
        for t, stage in (derived.get("kdigo") or {}).get(s.stay_id, []):
            if stage is not None:
                kmax[i] = np.nanmax([kmax[i], stage])
                h = _hours(t, s.icu_in)
                if stage >= 2 and h >= 0 and not (onset["aki2"][i] <= h):
                    onset["aki2"][i] = h
        x = (derived.get("sofa_24h") or {}).get(s.stay_id)
        if x is not None:
            sofa[i] = x
    return Cache([s.stay_id for s in stays], np.array([s.dataset for s in stays]),
                 np.array([str(s.prefix.get("site")) for s in stays]), static, cats, concepts, grid, los, death,
                 onset, sofa, kmax)
