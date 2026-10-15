"""Label programs (plan Phase F, Factory design 4; "Label sampling" 2): canonical specs holding only the parameters that
change the label (design-review correction 14: the identity of a task is its label function, not its sampling
details), evaluated on a store at any prediction times (``evaluate``).

A target is a nested tuple:
* ``("event", name)``: a derived episode's onset (``ep:<name>`` of the store: imv, vasopressor, rrt, nippv, ...);
* ``("stage", kind, level)``: a derived point series at or above a level (``kdigo``, ``kdigo_creat``, ``kdigo_uo``,
  ``nee``), holding until its next row;
* ``("point", kind)``: a derived point event (``ase_onset``, ``ase_onset_wo_lactate``, ``presumed_infection``);
* ``("cross", channel, direction, edge, sustain, hold)``: a channel below (bins <= edge) or at or above (bins >=
  edge + 1) one of its decile edges, sustained ``sustain`` minutes, each measurement holding ``hold`` minutes. Exact
  under both decodes, the default for thresholds;
* ``("cross_value", channel, direction, value, sustain, hold)``: the same against a clinical cut point inside a bin
  (a flagged variant: the decoder decides it within the bin);
* ``("and", a, b)`` / ``("or", a, b)``: composites;
* ``("death",)`` / ``("discharge",)``: the ICU death, the ICU discharge alive.

A program: ``family`` (F1-F7, FC forecasting, FH functional heads), its ``target``, ``label`` (``binary`` within
``horizon`` minutes, ``horizon`` -1 meaning before the ICU discharge; ``tte``; ``regression``; ``ordinal`` with
``bands``; ``hosp_death``) and ``eligibility``. Regression targets: ``("value", channels, agg, a, b, baseline)`` (the
values of one or several channels in (t + a, t + b], optionally as a change from the last value in the ``baseline``
minutes before t), ``("los",)``, ``("stage_max", kind, a, b)``.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from icutwin.factory import labels as fl
from icutwin.factory import processes as fp

CATALOG_YAML = Path(__file__).resolve().parents[1] / "configs" / "factory" / "catalog.yaml"
END_OF_STAY = -1


@dataclass(frozen=True)
class Program:
    family: str
    target: tuple
    label: str = "binary"
    horizon: int = 0
    eligibility: tuple = ("in_icu", "not_in_event")
    bands: tuple = ()
    name: str = ""                      # the catalog's name (not part of the identity)
    flags: tuple = ()                   # e.g. ("clinical_cut",), ("nowcast",) (not part of the identity)

    def spec(self) -> dict:
        """The canonical spec: only label-affecting fields; a horizon only where the label has one."""
        d = {"target": _canon(self.target), "label": self.label, "eligibility": sorted(self.eligibility)}
        if self.label == "binary":
            d["horizon"] = self.horizon
        if self.label in ("ordinal", "multihorizon"):
            d["bands"] = list(self.bands)          # ordinal: the band edges; multi-horizon: the horizons (minutes)
        return d

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(self.spec(), sort_keys=True).encode()).hexdigest()


def _canon(t):
    return [_canon(x) for x in t] if isinstance(t, (tuple, list)) else t


def _nbytes(x, depth: int = 0) -> int:
    if isinstance(x, np.ndarray):
        return int(x.nbytes)
    if isinstance(x, dict) and depth < 2:
        return sum(_nbytes(v, depth + 1) for v in x.values() if v is not x)
    if isinstance(x, fp.Process):
        return sum(int(getattr(x, f).nbytes) for f in ("stay", "s", "e", "on_stay", "on_t"))
    return 0


class Cache:
    """A least-recently-used dict bounded by its values' array bytes (a worker's memory stays bounded whatever the
    pool's size; the arrays an entry adds after insertion, such as ``labels.keyed``'s, count from its next use)."""

    def __init__(self, max_mb: float):
        self.max = int(max_mb * 2**20)
        self.d: OrderedDict = OrderedDict()
        self.bytes = 0

    def __contains__(self, k) -> bool:
        return k in self.d

    def __getitem__(self, k):
        v, n = self.d[k]
        self.d.move_to_end(k)
        m = _nbytes(v)
        if m != n:
            self.d[k] = (v, m)
            self.bytes += m - n
            self._evict()
        return v

    def __setitem__(self, k, v) -> None:
        if k in self.d:
            self.bytes -= self.d.pop(k)[1]
        n = _nbytes(v)
        self.d[k] = (v, n)
        self.bytes += n
        self._evict()

    def _evict(self) -> None:
        while self.bytes > self.max and len(self.d) > 1:
            _k, (_v, n) = self.d.popitem(last=False)
            self.bytes -= n

    def __len__(self) -> int:
        return len(self.d)


@dataclass
class Evaluator:
    """Evaluates programs on one store, caching each target's process and each channel set's events (least recently
    used first out, within ``cache_mb`` each)."""
    store: object
    cache_mb: float = 2048.0
    _proc: Cache = None
    _ev: Cache = None

    def __post_init__(self):
        self._proc = Cache(self.cache_mb)
        self._ev = Cache(self.cache_mb)

    def events(self, channels) -> dict:
        """The events of one or several channels (names), ordered by stay and time."""
        key = tuple(channels) if isinstance(channels, (tuple, list)) else (channels,)
        if key not in self._ev:
            index = self.store.channels.index
            parts = [self.store.channel_events(index[c]) for c in key if c in index]
            if not parts:
                self._ev[key] = {f: np.zeros(0) for f in ("stay", "t", "v", "bin", "f")}
            elif len(parts) == 1:
                self._ev[key] = parts[0]
            else:
                cat = {f: np.concatenate([p[f] for p in parts]) for f in parts[0]}
                order = np.lexsort((cat["t"], cat["stay"]))
                self._ev[key] = {f: x[order] for f, x in cat.items()}
        return self._ev[key]

    def process(self, target: tuple) -> fp.Process:
        if target in self._proc:
            return self._proc[target]
        op = target[0]
        s = self.store
        if op == "event":
            p = fp.episodes(s.derived_events(f"ep:{target[1]}"))
        elif op == "stage":
            p = fp.staged(s.derived_events(target[1]), target[2], hold=7 * 24 * 60)
        elif op == "point":
            dv = s.derived_events(target[1])
            p = fp.points(dv["stay"], dv["s"], np.ones(len(dv["s"]), dtype=bool), hold=1)
        elif op == "cross":
            _, ch, direction, edge, sustain, hold = target
            p = fp.threshold(self.events(ch), edge, direction, hold, sustain)
        elif op == "cross_value":
            _, ch, direction, value, sustain, hold = target
            ev = self.events(ch)
            x = np.asarray(ev["v"], dtype=np.float64)
            ok = ~np.isnan(x)
            on = (x < value) if direction == "below" else (x >= value)
            p = fp.points(ev["stay"][ok], ev["t"][ok], on[ok], hold, sustain)
        elif op in ("and", "or"):
            a, b = self.process(target[1]), self.process(target[2])
            p = fp.intersect(a, b) if op == "and" else fp.union(a, b)
        elif op in ("death", "discharge"):
            p = fp.ends(s, op)
        else:
            raise ValueError(f"unknown target {target!r}")
        self._proc[target] = p
        return p

    def reference_rows(self, n: int = 50_000, seed: int = 0) -> fl.Rows:
        """A fixed sample of the store's hourly rows: the population a functional head's cut is a quantile of."""
        if getattr(self, "_ref", None) is None:
            r = fl.rolling(self.store, 60)
            if len(r) > n:
                r = r.take(np.sort(np.random.default_rng(seed).choice(len(r), n, replace=False)))
            self._ref = r
        return self._ref

    def _fn(self, prog: Program, rows: fl.Rows) -> fl.LabelBlock:
        """A functional head (``factory.functional``): its score on the window's features; binary labels cut at the
        store-level quantile giving its prevalence, ordinal ones at equal-mass score bands."""
        from icutwin.factory import functional

        _, kind, seed, chans, aggs, a, b, q = prog.target

        def run(r):
            x, seen = functional.features(self, r, chans, aggs, a, b)
            return functional.score(kind, seed, x), seen

        s, seen = run(rows)
        los = self.store.los[rows.stay]
        inside = (rows.t >= 0) & (rows.t < los)
        late = inside & (rows.t + b > los)
        ok = inside & ~late & (seen > 0)
        reason = (np.where(~inside, fl.REASONS["not_in_icu"], 0) | np.where(late, fl.REASONS["censored"], 0)
                  | np.where(inside & ~late & (seen == 0), fl.REASONS["no_value"], 0)).astype(np.uint8)
        if prog.label in ("binary", "ordinal"):
            key = ("fn_cut", prog.target, prog.label, prog.bands)
            if key not in self._proc:
                ref = self.reference_rows()
                rs, rseen = run(ref)
                rlos = self.store.los[ref.stay]
                rok = (ref.t + b <= rlos) & (rseen > 0)
                pool = rs[rok] if rok.any() else rs
                if prog.label == "binary":         # a piecewise-constant score (trees) ties at its cut: the side
                    c = float(np.quantile(pool, 1 - q))   # whose share is nearer the drawn prevalence
                    strict = bool(abs(np.mean(pool > c) - q) < abs(np.mean(pool >= c) - q))
                    self._proc[key] = (np.array([c]), strict)
                else:
                    self._proc[key] = (np.quantile(pool, [k / (len(prog.bands) + 1)
                                                          for k in range(1, len(prog.bands) + 1)]), False)
            cuts, strict = self._proc[key]
            if prog.label == "binary":
                y = ((s > cuts[0]) if strict else (s >= cuts[0])).astype(np.float32)
            else:
                y = np.digitize(s, cuts).astype(np.float32)
        else:
            y = s.astype(np.float32)
        n = len(rows)
        return fl.LabelBlock(np.where(ok, y, np.nan).astype(np.float32), ok, reason, np.full(n, np.nan, np.float32),
                             np.full(n, -1, np.int8), ((los - rows.t) / 60.0).astype(np.float32),
                             (rows.t + max(b, 1)).astype(np.int64))   # a nowcast is known just after t

    def multi(self, prog: Program, rows: fl.Rows) -> tuple:
        """A multi-horizon label: (time-to-event block, y [n, H], valid [n, H]) at the horizons ``prog.bands``."""
        b = fl.tte(self.store, self.process(prog.target), rows, prog.eligibility)
        y, ok = fl.multi_horizon(b, rows, prog.bands)
        return b, y.astype(np.float32), ok & b.valid[:, None]

    def forecast(self, prog: Program, rows: fl.Rows) -> tuple:
        """A forecast of the patient's own values (target ``("forecast", channels, grid, mode)``): ``grid`` mode,
        each channel's last value in the step ending at t + h for every h of the grid (an hourly grid for vitals);
        ``next`` mode, each channel's first value after t within the grid's last horizon and its delay in minutes
        (labs). Returns (Y [n, H, C], M [n, H, C] observed, valid [n]); nothing after the ICU discharge counts."""
        _, chans, grid, mode = prog.target
        grid = np.asarray(grid, dtype=np.int64)
        los = self.store.los[rows.stay]
        n = len(rows)
        if mode == "grid":
            step = int(grid[0]) if len(grid) == 1 else int(np.min(np.diff(np.r_[0, grid])))
            y = np.full((n, len(grid), len(chans)), np.nan, dtype=np.float32)
            for c, ch in enumerate(chans):
                ev = fl.finite(self.events(ch))
                for j, h in enumerate(grid):
                    v, _cnt = fl.window_values(ev, rows, int(h) - step, int(h), "last")
                    y[:, j, c] = np.where(rows.t + h <= los, v, np.nan)
        else:
            y = np.full((n, 2, len(chans)), np.nan, dtype=np.float32)
            for c, ch in enumerate(chans):
                ev = fl.finite(self.events(ch))
                keys = fl.keyed(ev)
                lo = np.searchsorted(keys, fp.comp(rows.stay, rows.t), "right")
                hi = np.searchsorted(keys, fp.comp(rows.stay, rows.t + int(grid[-1])), "right")
                has = (hi > lo)
                first = np.minimum(lo, max(len(keys) - 1, 0))
                tt = np.asarray(ev["t"])[first] if len(keys) else np.zeros(n)
                has &= (tt <= los) if len(keys) else has
                y[:, 0, c] = np.where(has, np.asarray(ev["v"])[first] if len(keys) else np.nan, np.nan)
                y[:, 1, c] = np.where(has, tt - rows.t, np.nan)
        m = np.isfinite(y)
        inside = (rows.t >= 0) & (rows.t < los)
        return y, m, inside & m.any(axis=(1, 2))

    def evaluate(self, prog: Program, rows: fl.Rows) -> fl.LabelBlock:
        """A one-dimensional block for every label type (a multi-horizon label: its time-to-event block; a forecast:
        valid where some value is observed, y missing)."""
        s = self.store
        if prog.target[0] == "fn":
            return self._fn(prog, rows)
        if prog.label == "multihorizon":
            return self.multi(prog, rows)[0]
        if prog.label == "forecast":
            _y, _m, ok = self.forecast(prog, rows)
            n = len(rows)
            los = s.los[rows.stay]
            reason = np.where(ok, 0, fl.REASONS["no_value"]).astype(np.uint8)
            return fl.LabelBlock(np.full(n, np.nan, np.float32), ok, reason, np.full(n, np.nan, np.float32),
                                 np.full(n, -1, np.int8), ((los - rows.t) / 60.0).astype(np.float32),
                                 (rows.t + int(prog.target[2][-1])).astype(np.int64))
        if prog.label in ("binary", "tte"):
            b = fl.tte(s, self.process(prog.target), rows, prog.eligibility)
            if prog.label == "tte":
                return b
            h = int(1e9) if prog.horizon == END_OF_STAY else prog.horizon
            return fl.at_horizon(b, rows, h)
        if prog.label == "hosp_death":
            return fl.hospital_death(s, rows)
        if prog.label in ("regression", "ordinal"):
            b = self._value(prog.target, rows)
            return fl.band(b, prog.bands) if prog.label == "ordinal" else b
        raise ValueError(f"unknown label {prog.label!r}")

    def _value(self, target: tuple, rows: fl.Rows) -> fl.LabelBlock:
        op = target[0]
        if op == "value":
            _, chans, agg, a, b, baseline = target
            return fl.value_target(self.store, self.events(chans), rows, a, b, agg, baseline)
        if op == "los":
            return fl.remaining_los(self.store, rows)
        if op == "stage_max":
            _, kind, a, b = target
            return fl.stage_max(self.store, self.store.derived_events(kind), rows, a, b)
        raise ValueError(f"unknown value target {target!r}")


def names(t) -> list[str]:
    """Every string of a target (its operator, channels, concepts), depth first."""
    out = []
    for x in t:
        if isinstance(x, tuple):
            out += names(x)
        elif isinstance(x, str):
            out.append(x)
    return out


def nearest_edge(channels, name: str, value: float) -> int:
    """The index of a channel's decile edge nearest a clinical value."""
    edges = np.asarray(channels.edges[channels.index[name]], dtype=np.float64)
    return int(np.argmin(np.abs(edges - value)))


def _target(spec, channels) -> tuple:
    """A catalog target (YAML lists) as a canonical tuple; ``cross`` with a clinical ``value`` resolves to the
    nearest edge."""
    op = spec[0]
    if op == "cross":                       # [cross, channel, direction, value, sustain, hold]
        _, ch, direction, value, sustain, hold = spec
        return ("cross", ch, direction, nearest_edge(channels, ch, value), int(sustain), int(hold))
    if op in ("and", "or"):
        return (op, _target(spec[1], channels), _target(spec[2], channels))
    if op == "value":                       # [value, [channels], agg, a, b, baseline]
        _, chans, agg, a, b, base = spec
        return ("value", tuple(chans) if isinstance(chans, list) else (chans,), agg, int(a), int(b),
                None if base is None else int(base))
    return tuple(x if not isinstance(x, list) else tuple(x) for x in spec)


def catalog(channels, path: Path = CATALOG_YAML) -> list[Program]:
    """The declarative clinical programs (``configs/factory/catalog.yaml``), one per (entry, horizon), resolved
    against a store's channels; entries whose channels the store lacks are left out."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = []
    for e in doc["programs"]:
        try:
            tgt = _target(e["target"], channels)
        except KeyError:
            continue
        for h in e.get("horizons", [0]):
            out.append(Program(e["family"], tgt, e.get("label", "binary"), int(h),
                               tuple(e.get("eligibility", ("in_icu", "not_in_event"))), tuple(e.get("bands", ())),
                               e["name"], tuple(e.get("flags", ()))))
    return out


def descriptor(prog: Program, channels) -> dict:
    """The structured task descriptor the model may see (study lead, Q&A 11): family, label type, horizon, eligibility,
    and the target's channels, directions and thresholds (in units, and as z from the store's statistics)."""
    d = {"family": prog.family, "label": prog.label, "horizon_min": prog.horizon, "eligibility": list(prog.eligibility),
         "terms": []}

    def walk(t):
        if t[0] in ("cross", "cross_value"):
            _, ch, direction, x, sustain, hold = t
            value = float(channels.edges[channels.index[ch]][x]) if t[0] == "cross" else float(x)
            st = channels.stats.get(ch) or {}
            spread = (st.get("p95", np.nan) - st.get("p05", np.nan)) / 3.29 if st else np.nan
            z = (value - st["median"]) / spread if st and spread > 0 else None
            d["terms"].append({"channel": ch, "direction": direction, "threshold": value, "z": z,
                               "sustain_min": sustain})
        elif t[0] in ("and", "or"):
            d["terms"].append({"op": t[0]})
            walk(t[1])
            walk(t[2])
        else:
            d["terms"].append({"op": t[0], "args": [x for x in t[1:] if not isinstance(x, tuple)]})

    walk(prog.target)
    return d
