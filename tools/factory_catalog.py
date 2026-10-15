"""The clinical catalog on stores (plan Phase F, S2/S4): every catalog program evaluated on each store's hourly rows
(at most ``--per-stay`` rows per stay, drawn uniformly), with aggregates only: valid rows, prevalence per row and
per stay (binary), class shares (ordinal), mean and SD (regression), and seconds. Twins against real stays and the
real-vs-real control show the label gaps; the timing sizes the label engine.

    python tools/factory_catalog.py --stores data/factory/stores/bank_dev data/factory/stores/real_dev \\
        data/factory/stores/real_control --out runs/factory/catalog.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def rows_for(st, per_stay: int, seed: int):
    from icutwin.factory import labels as fl

    r = fl.rolling(st, 60)
    if per_stay:
        rng = np.random.default_rng(seed)
        keep = np.zeros(len(r), dtype=bool)
        starts = np.r_[0, np.flatnonzero(np.diff(r.stay)) + 1]
        ends = np.r_[starts[1:], len(r)]
        for a, b in zip(starts, ends):
            k = rng.choice(np.arange(a, b), size=min(per_stay, b - a), replace=False)
            keep[k] = True
        r = r.take(np.flatnonzero(keep))
    return r


def summary(p, b, rows) -> dict:
    ok = b.valid
    out = {"valid": int(ok.sum())}
    if not ok.any():
        return out
    y = b.y[ok]
    if p.label in ("binary", "hosp_death"):
        out["prevalence"] = float(y.mean())
        stays = rows.stay[ok]
        pos = np.zeros(stays.max() + 1, dtype=bool)
        pos[stays[y > 0]] = True
        u = np.unique(stays)
        out["per_stay"] = float(pos[u].mean())
    elif p.label == "ordinal":
        k, n = np.unique(y, return_counts=True)
        out["shares"] = {str(int(a)): round(float(c / n.sum()), 4) for a, c in zip(k, n)}
    elif p.label == "tte":
        out["events"] = float(y.mean())
        out["median_t"] = float(np.nanmedian(b.t_event[ok])) if np.isfinite(b.t_event[ok]).any() else None
    else:
        out["mean"], out["sd"] = float(np.mean(y)), float(np.std(y))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-stay", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst

    report = {}
    for path in a.stores:
        st = fst.Store(Path(path))
        rows = rows_for(st, a.per_stay, a.seed)
        ev = fpr.Evaluator(st)
        res = {"stays": len(st), "rows": len(rows), "programs": {}}
        t_all = time.time()
        for p in fpr.catalog(st.channels):
            t0 = time.time()
            b = ev.evaluate(p, rows)
            res["programs"][f"{p.name}@{p.horizon}"] = {**summary(p, b, rows), "family": p.family,
                                                       "seconds": round(time.time() - t0, 3)}
        res["seconds"] = round(time.time() - t_all, 1)
        report[Path(path).name] = res
        print(json.dumps({"store": Path(path).name, "stays": len(st), "rows": len(rows),
                          "seconds": res["seconds"]}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    names = list(report)
    keys = list(report[names[0]]["programs"])
    print("program".ljust(34) + "".join(n[:14].rjust(16) for n in names))
    for k in keys:
        cells = []
        for n in names:
            r = report[n]["programs"].get(k, {})
            x = r.get("prevalence", r.get("mean", r.get("events")))
            cells.append(("-" if x is None else f"{x:.3f}") + f" ({r.get('valid', 0)})")
        print(k[:34].ljust(34) + "".join(c.rjust(16) for c in cells))


if __name__ == "__main__":
    main()
