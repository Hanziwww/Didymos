"""TrajPFN against the tabular baselines on identical rows (plan Phase G; reported, not gated): the run's per-row
predictions (``tools/icl_eval.py``: ``preds_<set>.npz``) joined with the baselines' (``tools/icl_baselines.py``:
``<set>.npz``) by episode and row order. Binary labels: per-episode log loss, Brier score and AUROC (episodes with both
classes among their queries for AUROC) for TrajPFN, the context's base rate, ExtraTrees, TabICLv2 and TabPFN-3.5;
values: per-episode MAE and R^2. Medians over episodes every method scored, and the share of episodes where TrajPFN
beats each. Writes ``<run>/compare_<set>.json``.

    python tools/icl_compare.py --run-dir runs/icl/M_pilot --set twin_val --ema
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _auroc(p, y):
    from scipy.stats import rankdata

    pos = y > 0.5
    if pos.all() or not pos.any() or not np.isfinite(p).all():
        return np.nan
    r = rankdata(p)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def _logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if np.isfinite(p).all() else np.nan


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--sets", default="data/icl/sets")
    ap.add_argument("--baselines", default="data/icl/baselines")
    ap.add_argument("--ema", action="store_true")
    a = ap.parse_args(argv)
    from icutwin.icl.train import load_set

    run = Path(a.run_dir)
    mp = np.load(run / f"preds_{a.set}{'_ema' if a.ema else ''}.npz", allow_pickle=False)
    bl = np.load(Path(a.baselines) / f"{a.set}.npz", allow_pickle=False)
    base_rate = {}
    for ep in load_set(str(Path(a.sets) / a.set)):            # the context's prevalence, as the baseline head uses it
        c = ep["context"]
        ok = np.asarray(c["lab_valid"], dtype=bool)
        y = np.nan_to_num(np.asarray(c["lab_y"], dtype=np.float64))[ok]
        base_rate[int(ep["recipe"]["index"])] = (y.sum() + 0.5) / (len(y) + 1.0)
    methods = ("trajpfn", "base_rate", "et", "tabicl", "tabpfn")
    per = {"binary": [], "regression": []}
    for e in np.unique(mp["episode"]):
        mm, bm = mp["episode"] == e, bl["episode"] == e
        if not bm.any() or mm.sum() != bm.sum():
            continue
        kind = str(mp["label"][mm][0])
        y = mp["y"][mm].astype(np.float64)
        preds = {"trajpfn": mp["pred"][mm].astype(np.float64), "et": bl["et"][bm], "tabicl": bl["tabicl"][bm],
                 "tabpfn": bl["tabpfn"][bm]}
        if kind in ("binary", "hosp_death"):
            preds["base_rate"] = np.full(len(y), base_rate.get(int(e), np.nan))
            per["binary"].append({m: {"logloss": _logloss(preds[m], y), "brier": float(np.mean((preds[m] - y) ** 2)),
                                      "auroc": _auroc(preds[m], y)} for m in methods if m in preds})
        elif kind == "regression":
            preds["base_rate"] = np.full(len(y), np.nan)
            var = np.var(y)
            per["regression"].append({m: {"mae": float(np.mean(np.abs(preds[m] - y))),
                                          "r2": float(1 - np.mean((preds[m] - y) ** 2) / var) if var > 0 else np.nan}
                                      for m in methods if m in preds})
    out: dict = {"set": a.set, "run": str(run), "ema": a.ema}
    for kind, rows in per.items():
        if not rows:
            continue
        metrics = ("logloss", "brier", "auroc") if kind == "binary" else ("mae", "r2")
        rep = {"episodes": len(rows)}
        for met in metrics:
            have = [m for m in methods if all(np.isfinite(r.get(m, {}).get(met, np.nan)) for r in rows)]
            usable = [r for r in rows if all(np.isfinite(r[m][met]) for m in methods if m in r and
                                             np.isfinite(r[m][met]))]
            rep[met] = {}
            for m in methods:
                vals = [r[m][met] for r in rows if m in r and np.isfinite(r[m][met])]
                if not vals:
                    continue
                rep[met][m] = {"median": float(np.median(vals)), "episodes": len(vals)}
                if m != "trajpfn":
                    pair = [(r["trajpfn"][met], r[m][met]) for r in rows
                            if m in r and np.isfinite(r[m][met]) and np.isfinite(r["trajpfn"][met])]
                    if pair:
                        a_, b_ = np.asarray(pair).T
                        better = (a_ > b_) if met in ("auroc", "r2") else (a_ < b_)
                        rep[met][m]["trajpfn_better_share"] = float(better.mean())
            rep[met]["_complete_methods"] = have
            rep[met]["_episodes_all_scored"] = len(usable)
        out[kind] = rep
    path = run / f"compare_{a.set}{'_ema' if a.ema else ''}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v.get("episodes") if isinstance(v, dict) else v for k, v in out.items()}))


if __name__ == "__main__":
    main()
