"""The calibration benchmark's rollouts (plan Phase F: S0 runs them, S8 analyses them). The twin world's own
probability p* of a label at a prediction time, estimated by continuing a twin from that time many times with the
generator, gives the learnability ceiling (the oracle) and a reference for calibration.

* ``choose``: anchors, twins cut at a prediction time on an hourly grid. About ``per_dataset`` per dataset, spread
  evenly over ICU length-of-stay bands; within a twin an ICU day is drawn uniformly, then an hour in it, so long stays
  do not dominate. Half of each band is enriched: its anchors have a new IMV, vasopressor or RRT start, or an ICU
  death, in the twin's own next ``horizon`` hours. Every anchor records its inclusion weight (its kind's natural share
  in the band over the design's share), so the natural mix can be restored.
* ``prompt``: the twin as known at the anchor (``tokenization.encode.prospective`` of its labels' decode on its own
  clock, ``factory.twins``): header fields documented later unknown, no goal and no end block.
* ``token_caps``: per dataset, the ``q`` quantile of the twins' tokens in the ``hours`` after an ICU time. A rollout's
  budget is capped there, and the grammar's budget layer ends a capped rollout on the censored path.
* ``run``: ``n`` rollouts per anchor. Rollout 0 is the anchor's label and rollouts 1..n-1 estimate p*. (The bank was
  raked to whole-stay margins while rollouts come from the model as is, so the twin's own future against p* measures
  the raking and is reported apart.) Batches are written atomically as Parquet, a written batch is skipped on
  resume, the datasets are taken in turn, and no batch starts after the deadline.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

LOS_BANDS = (0.0, 24.0, 48.0, 96.0, 168.0, float("inf"))    # hours
KINDS = ("imv", "vasopressor", "rrt")
DEATH = "ICU//discharge//dead"


@dataclass
class Anchor:
    anchor: str          # "<twin>@<hours>"
    twin: str
    dataset: str
    hours: float         # the prediction time, hours since ICU admission
    los_hours: float
    band: int
    enriched: bool       # drawn by the enriched half of its band
    flag: bool           # an event in the twin's own next ``horizon`` hours
    weight: float        # inclusion weight: natural share of its kind in the band over the design's


def event_codes(v) -> dict:
    """The codes of each treatment (``eval.forecast.Reader``'s token sets, which the engine's obligations use)."""
    from icutwin.eval.forecast import Reader

    rd = Reader(v)
    return {k: frozenset(v.meta[i]["code"] for i in rd.sets[k]) for k in KINDS}


def onsets(stay, codes: dict) -> dict:
    """Hours after ICU admission of each treatment's first event in the ICU, and of an ICU death (None: none)."""
    out = {k: None for k in (*KINDS, "death")}
    for t, _r, code, _n, _x in stay.timeline:
        h = (t - stay.icu_in).total_seconds() / 3600
        if h < 0:
            continue
        for k in KINDS:
            if out[k] is None and code in codes[k]:
                out[k] = h
        if out["death"] is None and code == DEATH:
            out["death"] = h
    return out


def flagged(on: dict, hours: float, horizon: float) -> bool:
    """A new treatment start or an ICU death in (hours, hours + horizon]: a treatment already started by then does
    not count."""
    return any(x is not None and hours < x <= hours + horizon for x in on.values())


def los_band(los_hours: float) -> int:
    return int(np.searchsorted(LOS_BANDS, los_hours, side="right") - 1)


def draw_hour(rng: np.random.Generator, los_hours: float, min_hours: float) -> float:
    """An ICU day uniformly among those with an admissible hour, then an admissible whole hour in it
    (``min_hours`` <= t <= LOS - 1 h)."""
    lo, hi = int(np.ceil(min_hours)), int(np.floor(los_hours - 1))
    days = sorted({h // 24 for h in range(lo, hi + 1)})
    d = days[rng.integers(len(days))]
    hours = [h for h in range(max(lo, 24 * d), min(hi, 24 * d + 23) + 1)]
    return float(hours[rng.integers(len(hours))])


def choose(stays: list, v, per_dataset: int = 200, seed: int = 0, enrich: float = 0.5, horizon: float = 48.0,
           min_hours: float = 4.0, probe: int = 2000, max_tries: int = 200) -> list[Anchor]:
    """Anchors (module docstring) from ``stays``: (twin id, dataset, stay of the labels' decode) triples."""
    rng = np.random.default_rng(seed)
    codes = event_codes(v)
    by_ds: dict = {}
    for twin, ds, s in stays:
        los = (s.icu_out - s.icu_in).total_seconds() / 3600
        if los >= min_hours + 1:
            by_ds.setdefault(ds, []).append((twin, los, onsets(s, codes)))
    out, seen = [], set()
    for ds in sorted(by_ds):
        bands: dict = {}
        for item in by_ds[ds]:
            bands.setdefault(los_band(item[1]), []).append(item)
        keys = sorted(bands)
        quota = {b: per_dataset // len(keys) + (1 if i < per_dataset % len(keys) else 0) for i, b in enumerate(keys)}
        for k, b in enumerate(keys):
            members = bands[b]

            def draw(members=members):
                twin, los, on = members[rng.integers(len(members))]
                h = draw_hour(rng, los, min_hours)
                return twin, los, h, flagged(on, h, horizon)

            def fresh(d) -> bool:            # an anchor (twin, hour) at most once: its rollouts are keyed by it
                if (d[0], d[2]) in seen:
                    return False
                seen.add((d[0], d[2]))
                return True

            pi = float(np.mean([draw()[3] for _ in range(min(probe, 20 * len(members)))]))
            n_enr = round(enrich * quota[b]) if pi > 0 else 0
            natural, tries = [], 0
            while len(natural) < quota[b] - n_enr and tries < max_tries * max(quota[b], 1):
                d = draw()
                tries += 1
                if fresh(d):
                    natural.append(d)
            enriched, tries = [], 0
            while len(enriched) < n_enr and tries < max_tries * max(n_enr, 1):
                d = draw()
                tries += 1
                if d[3] and fresh(d):
                    enriched.append(d)
            n = len(natural) + len(enriched)
            rest = keys[k + 1:]
            for j, b2 in enumerate(rest):      # a band without enough distinct anchors passes its shortfall on
                quota[b2] += (quota[b] - n) // len(rest) + (1 if j < (quota[b] - n) % len(rest) else 0)
            design = (len(natural) * pi + len(enriched)) / n if n else 0.0
            w1 = pi / design if design > 0 else 0.0
            w0 = (1 - pi) / (1 - design) if design < 1 else 0.0
            for (twin, los, h, f), enr in [*((d, False) for d in natural), *((d, True) for d in enriched)]:
                out.append(Anchor(f"{twin}@{h:g}", twin, ds, h, los, b, enr, bool(f), w1 if f else w0))
    return out


def prompt(ids, v, twin_id: str, hours: float, seed: int = 0) -> list[int]:
    """The twin as known ``hours`` after its ICU admission, on its own clock (``factory.twins``)."""
    from icutwin.factory import twins
    from icutwin.tokenization.encode import prospective

    return prospective(twins.decode_labels(ids, v, twin_id, seed), v, hours)


def token_walk(ids, rd) -> np.ndarray:
    """Minutes since the ICU admission marker at every position (interval tokens at their pooled medians; NaN before
    the marker)."""
    out = np.full(len(ids), np.nan)
    t = None
    for k, tok in enumerate(ids):
        tok = int(tok)
        if t is None and tok in rd.adm:
            t = 0.0
        if t is not None:
            t += rd.minutes.get(tok, 0.0)
            out[k] = t
    return out


def token_caps(seqs: dict, v, hours: float = 72.0, q: float = 0.95, seed: int = 0, per_dataset: int = 500,
               starts: int = 5) -> dict:
    """Per dataset (``seqs``: dataset -> token sequences), the ``q`` quantile of the tokens in the ``hours`` after an
    ICU time (start times drawn uniformly over each twin's ICU stay)."""
    from icutwin.eval.forecast import Reader

    rd = Reader(v)
    rng = np.random.default_rng(seed)
    caps = {}
    for ds, rows in sorted(seqs.items()):
        counts = []
        for k in rng.permutation(len(rows))[:per_dataset]:
            m = token_walk(rows[k], rd)
            ok = ~np.isnan(m)
            if not ok.any():
                continue
            mm = m[ok]
            for t0 in rng.uniform(0, mm[-1], size=starts):
                counts.append(int(np.sum((mm > t0) & (mm <= t0 + hours * 60))))
        caps[ds] = int(np.quantile(counts, q)) if counts else None
    return caps


def _atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def batches(anchors: list[Anchor], size: int) -> list[tuple[str, int, list[int]]]:
    """(dataset, batch number, anchor positions) with the datasets taken in turn."""
    by_ds: dict = {}
    for i, a in enumerate(anchors):
        by_ds.setdefault(a.dataset, []).append(i)
    chunks = {ds: [idx[k:k + size] for k in range(0, len(idx), size)] for ds, idx in sorted(by_ds.items())}
    out, k = [], 0
    while any(k < len(c) for c in chunks.values()):
        out += [(ds, k, c[k]) for ds, c in chunks.items() if k < len(c)]
        k += 1
    return out


def run(engine, anchors: list[Anchor], prompts: list[list[int]], n: int, out_dir: Path, caps: dict, seed: int = 0,
        size: int = 32, deadline: float | None = None) -> dict:
    """``n`` rollouts per anchor (``engine.simulate``), batch by batch (module docstring); the manifest lists every
    written batch with its checksum. Returns the manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {"batches": {}}
    man.update({"anchors": len(anchors), "n": n, "caps": caps, "seed": seed, "size": size})
    man.pop("stopped", None)
    for g, (ds, k, idx) in enumerate(batches(anchors, size)):
        name = f"{ds}_{k:04d}.parquet"
        if (out_dir / name).exists():          # written before (a manifest lost with a crash is rebuilt)
            if name not in man["batches"]:
                man["batches"][name] = {"sha256": hashlib.sha256((out_dir / name).read_bytes()).hexdigest()}
            continue
        if deadline is not None and time.time() > deadline:
            man["stopped"] = "deadline"
            break
        t0 = time.time()
        conts = engine.simulate([prompts[i] for i in idx], n, seed=seed * 10_000 + g, max_new=caps.get(ds))
        rows = [(i, j, c) for i, cs in zip(idx, conts, strict=True) for j, c in enumerate(cs)]
        meta = [asdict(anchors[i]) for i, _j, _c in rows]
        cols = {f: [m[f] for m in meta] for f in Anchor.__dataclass_fields__}
        cols["rollout"] = [j for _i, j, _c in rows]
        cols["prompt"] = pa.array([list(map(int, prompts[i])) for i, _j, _c in rows], type=pa.list_(pa.int32()))
        cols["continuation"] = pa.array([list(map(int, c)) for _i, _j, c in rows], type=pa.list_(pa.int32()))
        sink = pa.BufferOutputStream()
        pq.write_table(pa.table(cols), sink)
        data = sink.getvalue().to_pybytes()
        _atomic(out_dir / name, data)
        man["batches"][name] = {"anchors": len(idx), "rollouts": len(rows), "seconds": round(time.time() - t0, 1),
                                "sha256": hashlib.sha256(data).hexdigest()}
        _atomic(man_path, json.dumps(man, indent=1).encode())
        print(json.dumps({"batch": name, **man["batches"][name]}), flush=True)
    man["complete"] = all(f"{ds}_{k:04d}.parquet" in man["batches"] for ds, k, _i in batches(anchors, size))
    _atomic(man_path, json.dumps(man, indent=1).encode())
    return man
