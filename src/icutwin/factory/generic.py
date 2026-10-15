"""Generic trajectories (plan Phase F, Factory design 10; the study lead's general-task policy): stays of anonymous
channels from open time-series generators, passed through an ICU-like observation process and written as a stay store,
so the label programs, episodes and formats apply unchanged. A switchable source: training from scratch mixes it at
10-30% until a training sweep sets the share; the twin stream's own generic share stays 0.

* ``kernel_synth``: a latent series drawn from a Gaussian-process prior whose kernel is a random sum or product of
  1-5 kernels of a bank (periodic, linear, RBF, rational quadratic, white noise, constant). Adapted from KernelSynth
  (Ansari et al. 2024, Chronos; Apache-2.0, github.com/amazon-science/chronos-forecasting, scripts/kernel-synth.py):
  the bank's periods are set to ICU timescales on an hourly axis (8-hour shifts, half-days, days, weeks).
* ``ts_mixup``: a convex mix (Dirichlet weights) of 1-3 standardized latent series, adapted from TSMixup (ibid.).
* ``observe``: a channel's charting: vital-like channels hourly with dropout and jitter; lab-like channels every 4-24
  hours, more often while the series is extreme (informative sampling); values an affine map of the latent series.
* ``build_store``: generic stays (lengths log-normal like ICU stays, a death share) with their channel table (decile
  edges and statistics from the generated values), in ``factory.store``'s layout.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

CITATION = "KernelSynth and TSMixup adapted from Chronos (Ansari et al. 2024; Apache-2.0)"


def kernel_bank(length: int) -> list:
    """Kernels on [0, 1] for a series of ``length`` hours (periods in hours, as fractions of the length)."""
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        DotProduct,
        ExpSineSquared,
        RationalQuadratic,
        WhiteKernel,
    )

    periods = [8, 12, 24, 48, 24 * 7]
    bank = [ExpSineSquared(periodicity=min(p / length, 1.0)) for p in periods]
    bank += [DotProduct(sigma_0=s) for s in (0.0, 1.0, 10.0)]
    bank += [RBF(length_scale=s) for s in (0.05, 0.2, 1.0)]
    bank += [RationalQuadratic(alpha=a) for a in (0.1, 1.0, 10.0)]
    bank += [WhiteKernel(noise_level=n) for n in (0.05, 0.3)]
    bank += [ConstantKernel()]
    return bank


GP_POINTS = 256       # a latent series is drawn on at most this many points, then interpolated to hours


def kernel_synth(rng: np.random.Generator, length: int, max_kernels: int = 5) -> np.ndarray:
    """One latent series of ``length`` hourly points (module docstring), standardized: drawn on at most
    ``GP_POINTS`` points of the stay (Cholesky, an SVD where the covariance is not positive definite) and interpolated
    to hours for longer stays."""
    bank = kernel_bank(length)
    n = min(length, GP_POINTS)
    for _ in range(20):
        ks = [bank[i] for i in rng.integers(len(bank), size=int(rng.integers(1, max_kernels + 1)))]
        kernel = functools.reduce(lambda a, b: a + b if rng.random() < 0.5 else a * b, ks)
        x = np.linspace(0, 1, n)[:, None]
        try:
            cov = kernel(x) + 1e-6 * np.eye(n)
            try:
                ts = np.linalg.cholesky(cov) @ rng.standard_normal(n)
            except np.linalg.LinAlgError:
                ts = rng.multivariate_normal(np.zeros(n), cov, method="svd")
        except (np.linalg.LinAlgError, ValueError):
            continue
        sd = ts.std()
        if np.isfinite(ts).all() and sd > 1e-6:
            if n < length:
                ts = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, n), ts)
                sd = ts.std()
            return (ts - ts.mean()) / sd
    return rng.normal(size=length)


def ts_mixup(rng: np.random.Generator, length: int, k_max: int = 3, alpha: float = 1.5) -> np.ndarray:
    k = int(rng.integers(1, k_max + 1))
    parts = np.stack([kernel_synth(rng, length) for _ in range(k)])
    w = rng.dirichlet(np.full(k, alpha))
    z = w @ parts
    return (z - z.mean()) / (z.std() or 1.0)


def observe(rng: np.random.Generator, z: np.ndarray, kind: str, offset: float, scale: float) -> tuple:
    """(minutes, values) of one channel's charting over an hourly latent series ``z`` (module docstring)."""
    n = len(z)
    if kind == "vital":
        keep = rng.random(n) < rng.uniform(0.7, 0.98)
        h = np.flatnonzero(keep)
        t = h * 60 + rng.integers(0, 60, size=len(h))
    else:
        t, cur = [], int(rng.integers(0, 6 * 60))
        base = rng.uniform(4, 24) * 60
        while cur < n * 60:
            t.append(cur)
            zz = abs(z[min(cur // 60, n - 1)])
            cur += int(max(60, base / (1 + max(zz - 1, 0) * 2) * rng.uniform(0.5, 1.5)))
        t = np.asarray(t, dtype=np.int64)
        h = np.minimum(t // 60, n - 1)
    v = offset + scale * (z[np.minimum(t // 60, n - 1)] + rng.normal(0, 0.1, size=len(t)))
    return np.asarray(t, dtype=np.int64), v.astype(np.float32)


def build_store(out: Path, n_stays: int = 2000, n_channels: int = 16, seed: int = 0, val_pct: float = 10.0,
                los_median_h: float = 37.0, death_share: float = 0.08) -> dict:
    """A generic store of ``n_stays`` stays over ``n_channels`` anonymous channels (``GEN//c<k>``)."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):          # small matrices: one BLAS thread, not every core of the host
        return _build_store(out, n_stays, n_channels, seed, val_pct, los_median_h, death_share)


def _build_store(out: Path, n_stays: int, n_channels: int, seed: int, val_pct: float, los_median_h: float,
                 death_share: float) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from icutwin.factory.channels import Channels
    from icutwin.factory.store import (
        ARRAYS,
        DERIVED,
        HEADER,
        PREFIX,
        _index_arrays,
        _sha,
        twin_split,
    )

    rng = np.random.default_rng(seed)
    kinds = ["vital" if rng.random() < 0.6 else "lab" for _ in range(n_channels)]
    offs, scs = rng.normal(50, 30, n_channels), np.exp(rng.normal(1.5, 1.0, n_channels))
    names = [f"GEN//c{k:03d}" for k in range(n_channels)]
    stays, ev = [], {k: [] for k, _ in ARRAYS["ev"]}
    for i in range(n_stays):
        los_h = float(np.clip(np.exp(rng.normal(np.log(los_median_h), 0.9)), 6, 30 * 24))
        length = int(np.ceil(los_h)) + 1
        t_all, c_all, v_all = [], [], []
        for c in range(n_channels):
            if rng.random() < 0.1:                       # a channel this stay never charts
                continue
            z = ts_mixup(rng, length) if rng.random() < 0.3 else kernel_synth(rng, length)
            t, v = observe(rng, z, kinds[c], offs[c], scs[c])
            ok = t < los_h * 60
            t_all.append(t[ok])
            c_all.append(np.full(ok.sum(), c))
            v_all.append(v[ok])
        t, c, v = np.concatenate(t_all), np.concatenate(c_all), np.concatenate(v_all)
        order = np.lexsort((c, t))
        sid = f"generic:{seed}:{i:06d}"
        dead = rng.random() < death_share
        stays.append({"stay_id": sid, "dataset": "generic", "site": "generic", "split": twin_split(sid, val_pct),
                      "weight": 1.0, "cell": "generic", "los_min": int(los_h * 60),
                      "end_kind": "dead" if dead else "alive", "hosp_death": bool(dead), "end_codes": "[]",
                      **{k: None for k in (*PREFIX, *HEADER)}, "n_ev": int(len(t)), "n_iv": 0, "n_dv": 0})
        ev["t"].append(t[order].astype(np.int32))
        ev["c"].append(c[order].astype(np.int16))
        ev["v"].append(v[order])
        ev["f"].append(np.ones(len(t), dtype=np.uint8))
    allv = {c: np.concatenate([v[cc == c] for v, cc in zip(ev["v"], ev["c"])]) for c in range(n_channels)}
    edges = [np.unique(np.quantile(allv[c], np.linspace(0.1, 0.9, 9))).tolist() if len(allv[c]) else []
             for c in range(n_channels)]
    ev["bin"] = []
    for vv, ccs in zip(ev["v"], ev["c"]):
        b = np.zeros(len(vv), dtype=np.uint8)
        for c in np.unique(ccs):
            m = ccs == c
            b[m] = np.searchsorted(np.asarray(edges[c]), vv[m], side="right")
        ev["bin"].append(b)
    out = Path(out)
    seg = out / "seg_0000"
    tmp = out / ".seg_0000.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    counts = [len(x) for x in ev["t"]]
    for g, fields in ARRAYS.items():
        n = counts if g == "ev" else [0] * len(stays)
        np.save(tmp / f"{g}_ptr.npy", np.concatenate([[0], np.cumsum(n)]).astype(np.int64))
        for k, dt in fields:
            data = np.concatenate(ev[k]).astype(dt) if g == "ev" else np.zeros(0, dtype=dt)
            np.save(tmp / f"{g}_{k}.npy", data)
    _index_arrays(tmp, {"ev": n_channels, "dv": len(DERIVED)})
    pq.write_table(pa.Table.from_pylist(stays), tmp / "stays.parquet")
    shutil.rmtree(seg, ignore_errors=True)
    os.replace(tmp, seg)
    stats = {names[c]: {"p05": float(np.quantile(allv[c], 0.05)), "median": float(np.median(allv[c])),
                        "p95": float(np.quantile(allv[c], 0.95)), "n": int(len(allv[c]))}
             for c in range(n_channels) if len(allv[c])}
    ch = Channels(names, ["GEN"] * n_channels, ["vitals" if k == "vital" else "labs" for k in kinds],
                  ["numeric"] * n_channels, edges, [True] * n_channels, stats)
    ch.save(out / "channels.json")
    entry = {"name": "seg_0000", "source": {"kind": "generic", "citation": CITATION, "seed": seed},
             "stays": len(stays), "events": int(sum(counts)),
             "files": {p.name: _sha(p) for p in sorted(seg.iterdir())}}
    man = {"version": 1, "segments": [entry], "channels_sha256": _sha(out / "channels.json"),
           "generator": {"n_channels": n_channels, "kinds": kinds, "los_median_h": los_median_h,
                         "death_share": death_share, "digest": hashlib.sha1(json.dumps(edges).encode()).hexdigest()}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return entry
