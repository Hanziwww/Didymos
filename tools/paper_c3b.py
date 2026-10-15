"""Paper P amendment 7, F6 and D4 (``docs/p1_paper_protocol.md``): C3's virtual cohorts with hospitals, intervals
and the death gap (no new twins; C3's stores and profiles, ``tools/paper_c3.py``).

* eICU hospitals with at least 30 held-out stays and 30 twins: per hospital, the ICU death, IMV, vasoactive and median
  ICU stay of twins and of real stays; the Spearman correlation across hospitals, twins against real, beside the same
  for real control (train) stays against real held-out stays (the reference), with bootstrap intervals over hospitals;
* the inter-dataset structure (C3's correlation of profile distances) with a bootstrap interval (stays resampled
  within datasets);
* the death gap (D4): ICU death by length-of-stay band, and the share of stays by end kind (dead, alive, censored),
  twins against real per dataset; ICU death among twins whose stay ended (not censored).

Writes ``<out>/c3b.json`` (aggregates).

    python tools/paper_c3b.py --twins data/factory/stores/bank_dev --real data/factory/stores/real_test \\
        --control data/factory/stores/real_control --out runs/paper/c3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import _sha  # noqa: E402
from paper_c3 import QUANT, RATES, profile  # noqa: E402

LOS_BANDS = (0, 24, 72, 168, np.inf)
HOSP_RATES = ("icu_death", "ever_imv", "ever_vasopressor")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("twins", "real", "control", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--min-stays", type=int, default=30)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args(argv)
    from scipy.stats import spearmanr

    from icutwin.factory import store as fst

    tw, re_, co = (fst.Store(Path(x)) for x in (a.twins, a.real, a.control))
    pt, pr, pc = profile(tw), profile(re_), profile(co)
    rng = np.random.default_rng(7)
    rep = {"git": _sha()}

    # eICU hospitals
    def hosp(store, p):
        m = store.stays["dataset"].to_numpy() == "eicu"
        site = store.stays["site"].astype(str).to_numpy()
        out = {}
        for s in np.unique(site[m]):
            k = m & (site == s)
            if k.sum() >= a.min_stays:
                out[s] = {**{r: float(np.nanmean(p[r][k])) for r in HOSP_RATES},
                          "los_median": float(np.median(p["los_h"][k])), "stays": int(k.sum())}
        return out

    ht, hr, hc = hosp(tw, pt), hosp(re_, pr), hosp(co, pc)

    def corr(ha, hb, key):
        common = sorted(set(ha) & set(hb))
        if len(common) < 5:
            return {"hospitals": len(common), "spearman": None, "ci": [None, None]}
        x = np.array([ha[s][key] for s in common])
        y = np.array([hb[s][key] for s in common])
        r = spearmanr(x, y).statistic
        bs = []
        for _ in range(a.boot):
            i = rng.integers(0, len(common), len(common))
            if np.std(x[i]) > 0 and np.std(y[i]) > 0:
                bs.append(spearmanr(x[i], y[i]).statistic)
        return {"hospitals": len(common), "spearman": float(r),
                "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else [None, None]}

    rep["eicu_hospitals"] = {k: {"twins_vs_real": corr(ht, hr, k), "control_vs_real": corr(hc, hr, k)}
                             for k in (*HOSP_RATES, "los_median")}

    # the inter-dataset structure with an interval
    dsets = sorted(set(pr["dataset"]))

    def vec(p, idx_of):
        return np.array([[(np.nanmedian if k in RATES else np.nanmean)(p[k][idx_of[d]]) for k in QUANT]
                         + [np.median(p["los_h"][idx_of[d]])] for d in dsets])

    def dist(v):
        z = (v - v.mean(0)) / np.where(v.std(0) > 0, v.std(0), 1)
        dm = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
        return dm[np.triu_indices(len(dsets), 1)]

    idx = {name: {d: np.flatnonzero(p["dataset"] == d) for d in dsets} for name, p in (("t", pt), ("r", pr), ("c", pc))}
    point = {"twins": float(np.corrcoef(dist(vec(pt, idx["t"])), dist(vec(pr, idx["r"])))[0, 1]),
             "control": float(np.corrcoef(dist(vec(pc, idx["c"])), dist(vec(pr, idx["r"])))[0, 1])}
    bs = {"twins": [], "control": []}
    for _ in range(200):
        draw = {n: {d: rng.choice(ix, size=len(ix)) for d, ix in idx[n].items()} for n in idx}
        dr = dist(vec(pr, draw["r"]))
        bs["twins"].append(np.corrcoef(dist(vec(pt, draw["t"])), dr)[0, 1])
        bs["control"].append(np.corrcoef(dist(vec(pc, draw["c"])), dr)[0, 1])
    rep["inter_dataset"] = {k: {"r": point[k], "ci": [float(np.percentile(bs[k], 2.5)),
                                                      float(np.percentile(bs[k], 97.5))]} for k in point}

    # D4: the death gap
    def gap(store, p):
        out = {}
        ds = p["dataset"]
        ek = np.asarray(store.end_kind)
        for d in dsets:
            m = ds == d
            row = {"stays": int(m.sum()),
                   "end_kind": {k: float(np.mean(ek[m] == k)) for k in ("dead", "alive", "censored", "unknown")},
                   "icu_death_ended": float(np.mean(ek[m & np.isin(ek, ("dead", "alive"))] == "dead"))
                   if (m & np.isin(ek, ("dead", "alive"))).any() else None, "by_los": {}}
            for lo, hi in zip(LOS_BANDS[:-1], LOS_BANDS[1:], strict=True):
                k = m & (p["los_h"] >= lo) & (p["los_h"] < hi)
                row["by_los"][f"{lo:g}-{hi:g}h"] = {"share": float(k.sum() / max(m.sum(), 1)),
                                                     "icu_death": float(np.mean(ek[k] == "dead")) if k.any() else None}
            out[d] = row
        return out

    rep["death_gap"] = {"twins": gap(tw, pt), "real": gap(re_, pr), "control": gap(co, pc)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "c3b.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"hospitals": {k: (v["twins_vs_real"]["spearman"], v["control_vs_real"]["spearman"])
                                    for k, v in rep["eicu_hospitals"].items()}, "inter": rep["inter_dataset"]}))


if __name__ == "__main__":
    main()
