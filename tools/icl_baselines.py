"""Tabular baselines for TrajPFN's learning confirmation (plan Phase G; reported, not gated): on tabular snapshots of
the same evaluation episodes (``tools/icl_train.py make-set``), the same query rows (valid targets, in order) and every
valid context row as the training set:
* ExtraTrees fitted per episode (the factory's probe);
* TabICLv2 zero-shot (public checkpoints in ``.cache/tabicl``);
* TabPFN-3.5 zero-shot when its weights are present (``--tabpfn``): local inference only, never Prior Labs' API, with
  the package's telemetry switched off and the hub offline (its license allows research, evaluation and internal
  benchmarking only; the user accepts it and supplies the weights).
Features: the factory's snapshot (``Sampler.snapshot``: the last value in the day before t of the program's own
channels and core vitals and labs, and hours since admission). Binary and ordinal labels are classified, values
regressed. Writes ``<out>/<set>.npz`` (per row: episode, label, target, each baseline's probability or value) and a
summary JSON. Aggregates only.

    python tools/icl_baselines.py --set twin_val --store data/factory/stores/bank_dev --out data/icl/baselines
    python tools/icl_baselines.py --set real_dev --store data/factory/stores/real_dev \\
        --stats-from data/factory/stores/bank_dev --out data/icl/baselines --tabpfn .cache/tabpfn/<checkpoint>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

CLASSES = ("binary", "hosp_death", "ordinal")


def _rows(s, ep: dict, part: str, prefix: str = "lab_"):
    from icutwin.factory import labels as fl

    d = ep[part]
    ids = ep["recipe"]["context"] if part == "context" else [q["stay"] for q in ep["recipe"]["queries"]]
    st, t, y = [], [], []
    for k, sid in enumerate(ids):
        a, b = int(d[f"{prefix}off"][k]), int(d[f"{prefix}off"][k + 1])
        ok = np.asarray(d[f"{prefix}valid"][a:b], dtype=bool)
        st += [s.index[sid]] * int(ok.sum())
        t += np.asarray(d[f"{prefix}t"][a:b])[ok].tolist()
        y += np.asarray(d[f"{prefix}y"][a:b])[ok].tolist()
    return fl.Rows(np.asarray(st, dtype=np.int64), np.asarray(t, dtype=np.int64)), np.asarray(y, dtype=np.float64)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="a set under --sets (twin_val, real_dev, ...)")
    ap.add_argument("--sets", default="data/icl/sets")
    ap.add_argument("--store", required=True)
    ap.add_argument("--stats-from", default=None)
    ap.add_argument("--vocab", default="data/tokens/full_r4/vocab.json")
    ap.add_argument("--out", default="data/icl/baselines")
    ap.add_argument("--tabicl", default=".cache/tabicl")
    ap.add_argument("--tabpfn", default=None, help="a local TabPFN-3.5 checkpoint (.safetensors)")
    ap.add_argument("--tabpfn-reg", default=None, help="its regression checkpoint, if separate")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")         # the package's usage telemetry off
    os.environ.setdefault("HF_HUB_OFFLINE", "1")                   # weights are local; nothing is fetched
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    from icutwin.factory import episodes, store as fst
    from icutwin.factory.episodes import _program
    from icutwin.icl.train import load_set
    from icutwin.tokenization.vocab import Vocab

    st = fst.Store(Path(a.store))
    if a.stats_from:
        st.channels.stats = fst.Store(Path(a.stats_from)).channels.stats
    s = episodes.Sampler(st, Vocab.load(Path(a.vocab)), split=None)
    eps = load_set(str(Path(a.sets) / a.set))[:a.limit] if a.limit else load_set(str(Path(a.sets) / a.set))
    tabicl_c = tabicl_r = None
    ck = Path(a.tabicl)
    if (ck / "tabicl-classifier-v2-20260212.ckpt").exists():
        from tabicl import TabICLClassifier, TabICLRegressor

        tabicl_c = TabICLClassifier(model_path=str(ck / "tabicl-classifier-v2-20260212.ckpt"), device=a.device,
                                    allow_auto_download=False)
        tabicl_r = TabICLRegressor(model_path=str(ck / "tabicl-regressor-v2-20260212.ckpt"), device=a.device,
                                   allow_auto_download=False)
    tabpfn_c = tabpfn_r = None
    if a.tabpfn:
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        tabpfn_c = TabPFNClassifier(model_path=a.tabpfn, device=a.device)
        tabpfn_r = TabPFNRegressor(model_path=a.tabpfn_reg or a.tabpfn, device=a.device)
    cols = {k: [] for k in ("episode", "label", "y", "et", "tabicl", "tabpfn")}
    t0 = time.time()
    skipped = 0
    for ep in eps:
        prog = _program(ep["recipe"]["program"])
        kind = prog.label
        if kind not in CLASSES + ("regression",):
            continue
        rc, yc = _rows(s, ep, "context")
        rq, yq = _rows(s, ep, "query")
        if len(yc) < 5 or len(yq) == 0 or (kind in CLASSES and len(np.unique(yc)) < 2):
            skipped += 1
            continue
        xc, xq = s.snapshot(prog, rc), s.snapshot(prog, rq)
        out = {"et": np.full(len(yq), np.nan), "tabicl": np.full(len(yq), np.nan), "tabpfn": np.full(len(yq), np.nan)}
        if kind in ("binary", "hosp_death"):
            et = ExtraTreesClassifier(200, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(xc, yc)
            out["et"] = et.predict_proba(xq)[:, list(et.classes_).index(1.0)] if 1.0 in et.classes_ else 0.0
            for name, clf in (("tabicl", tabicl_c), ("tabpfn", tabpfn_c)):
                if clf is not None:
                    clf.fit(xc, yc)
                    p = clf.predict_proba(xq)
                    out[name] = p[:, list(clf.classes_).index(1.0)]
        elif kind == "ordinal":                                    # the expected class (scored by rank only)
            et = ExtraTreesClassifier(200, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(xc, yc)
            out["et"] = et.predict_proba(xq) @ et.classes_
        else:
            out["et"] = ExtraTreesRegressor(200, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(xc, yc).predict(xq)
            for name, reg in (("tabicl", tabicl_r), ("tabpfn", tabpfn_r)):
                if reg is not None:
                    out[name] = reg.fit(xc, yc).predict(xq)
        n = len(yq)
        cols["episode"].append(np.full(n, int(ep["recipe"]["index"])))
        cols["label"].append(np.full(n, kind))
        cols["y"].append(yq)
        for k in ("et", "tabicl", "tabpfn"):
            cols[k].append(np.broadcast_to(np.asarray(out[k], dtype=np.float64), (n,)).copy())
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    arr = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    np.savez_compressed(od / f"{a.set}.npz", **arr)
    summ = {"set": a.set, "episodes": int(len(np.unique(arr["episode"]))) if len(arr["episode"]) else 0,
            "rows": int(len(arr["y"])), "skipped": skipped, "seconds": round(time.time() - t0, 1),
            "tabicl": tabicl_c is not None, "tabpfn": tabpfn_c is not None}
    (od / f"{a.set}.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
