"""Paper P, C3: virtual cohorts (``docs/p1_paper_protocol.md`` C3). The internal bank's dataset-conditioned twins (a
factory store) against real held-out stays, per dataset, with the real-against-real control (real train stays against
the same held-out stays):

* profiles: age, female share, ICU length of stay (median, quartiles), ICU and in-hospital death, IMV, vasoactive and
  RRT at any time in the ICU, stays under 1 hour, and (medians over stays of at least 4 hours) events per ICU hour
  and lab results per ICU day; each with a bootstrap interval;
* the inter-dataset structure: the correlation of the datasets' pairwise profile distances (standardized profiles),
  twins against real (and control against real);
* eICU hospitals: total variation of the hospital distribution (as ``eval.diversity``'s hospital_tv);
* rare header strata (dataset x age band x sex x unit type x admission type with at least 20 stays and under 0.5% of
  their dataset in the real train store): their share among twins and real stays, and their ICU death rates.

Writes ``<out>/c3.json`` (aggregates).

    python tools/paper_c3.py --twins data/factory/stores/bank_dev --real data/factory/stores/real_test \\
        --control data/factory/stores/real_control --train data/factory/stores/real_train --out runs/paper/c3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import _sha  # noqa: E402

AGE_BANDS = (18, 45, 65, 80, 200)


def profile(store) -> dict:
    """Per-stay quantities of a store (arrays aligned with its stays)."""
    from icutwin.factory import programs as fpr

    st = store.stays
    ev = fpr.Evaluator(store)
    hours = np.maximum(store.los / 60.0, 1e-3)
    out = {"dataset": st["dataset"].to_numpy(), "age": st["age"].to_numpy(dtype=float),
           "female": (st["sex"].astype(str) == "female").to_numpy(dtype=float),
           "los_h": hours, "icu_death": (store.end_kind == "dead").astype(float),
           "hosp_death": st["hosp_death"].astype(float).to_numpy(),
           "events_per_hour": st["n_ev"].to_numpy(dtype=float) / hours}
    for k in ("imv", "vasopressor", "rrt"):
        first = ev.process(("event", k)).first_onset(len(store))
        out[f"ever_{k}"] = (first < store.los).astype(float)
    labs = np.zeros(len(store))
    ch = store.channels
    for cid, g in enumerate(ch.group):
        if g == "labs":
            e = store.channel_events(cid)
            if len(e["stay"]):
                labs += np.bincount(np.asarray(e["stay"], dtype=np.int64), minlength=len(store))
    out["labs_per_day"] = labs / (hours / 24.0)
    short = hours < 4                                # rates of very short stays are noise: not scored
    out["events_per_hour"] = np.where(short, np.nan, out["events_per_hour"])
    out["labs_per_day"] = np.where(short, np.nan, out["labs_per_day"])
    out["stay_lt_1h"] = (hours < 1).astype(float)
    return out


QUANT = ("age", "female", "icu_death", "hosp_death", "ever_imv", "ever_vasopressor", "ever_rrt", "stay_lt_1h",
         "events_per_hour", "labs_per_day")
RATES = ("events_per_hour", "labs_per_day")         # medians (stays of at least 4 hours); the others are means


def summary(p: dict, m: np.ndarray, rng, boot: int = 300) -> dict:
    out = {"stays": int(m.sum())}
    for k in QUANT:
        x = p[k][m]
        x = x[np.isfinite(x)]
        if not len(x):
            continue
        f = np.median if k in RATES else np.mean
        bs = [f(rng.choice(x, size=len(x))) for _ in range(boot)]
        out[k] = {"median" if k in RATES else "mean": float(f(x)),
                  "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    los = p["los_h"][m]
    out["los_h"] = {"median": float(np.median(los)), "q25": float(np.quantile(los, 0.25)),
                    "q75": float(np.quantile(los, 0.75))}
    return out


def strata(store) -> np.ndarray:
    st = store.stays
    band = np.digitize(st["age"].to_numpy(dtype=float), AGE_BANDS[1:-1])
    return np.array([f"{d}|{b}|{s}|{u}|{t}" for d, b, s, u, t in
                     zip(st["dataset"], band, st["sex"].astype(str), st["unit_type"].astype(str),
                         st["admission_type"].astype(str), strict=True)])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("twins", "real", "control", "train", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args(argv)
    from collections import Counter

    from icutwin.eval.diversity import _tv
    from icutwin.factory import store as fst

    tw, re_, co, tr = (fst.Store(Path(x)) for x in (a.twins, a.real, a.control, a.train))
    rng = np.random.default_rng(0)
    pt, pr, pc = profile(tw), profile(re_), profile(co)
    rep = {"git": _sha(), "datasets": {}}
    dsets = sorted(set(pr["dataset"]))
    for d in dsets:
        rep["datasets"][d] = {"twins": summary(pt, pt["dataset"] == d, rng), "real": summary(pr, pr["dataset"] == d, rng),
                              "control": summary(pc, pc["dataset"] == d, rng)}

    def dist(p):
        vec = np.array([[(np.nanmedian if k in RATES else np.nanmean)(p[k][p["dataset"] == d]) for k in QUANT]
                        + [np.median(p["los_h"][p["dataset"] == d])] for d in dsets])
        z = (vec - vec.mean(0)) / np.where(vec.std(0) > 0, vec.std(0), 1)
        dm = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
        return dm[np.triu_indices(len(dsets), 1)]

    dr = dist(pr)
    rep["inter_dataset"] = {"twins_vs_real": float(np.corrcoef(dist(pt), dr)[0, 1]),
                            "control_vs_real": float(np.corrcoef(dist(pc), dr)[0, 1])}
    site_t = tw.stays.loc[tw.stays["dataset"] == "eicu", "site"].astype(str).tolist()
    site_r = re_.stays.loc[re_.stays["dataset"] == "eicu", "site"].astype(str).tolist()
    site_c = co.stays.loc[co.stays["dataset"] == "eicu", "site"].astype(str).tolist()
    rep["eicu_hospitals"] = {"twins_vs_real_tv": _tv(Counter(site_r), Counter(site_t)) if site_t else None,
                             "control_vs_real_tv": _tv(Counter(site_r), Counter(site_c)) if site_c else None,
                             "hospitals_twins": len(set(site_t)), "hospitals_real": len(set(site_r))}
    s_tr, s_tw, s_re = strata(tr), strata(tw), strata(re_)
    ds_tr = tr.stays["dataset"].to_numpy()
    rare = []
    for k, n in zip(*np.unique(s_tr, return_counts=True), strict=True):
        d = k.split("|")[0]
        if n >= 20 and n / (ds_tr == d).sum() < 0.005:
            rare.append(k)
    in_tw, in_re = np.isin(s_tw, rare), np.isin(s_re, rare)
    rep["rare_strata"] = {"strata": len(rare),
                          "share_twins": float(in_tw.mean()), "share_real": float(in_re.mean()),
                          "share_train": float(np.isin(s_tr, rare).mean()),
                          "strata_reached_by_twins": int(len(set(s_tw[in_tw]))),
                          "icu_death_twins": float(pt["icu_death"][in_tw].mean()) if in_tw.any() else None,
                          "icu_death_real": float(pr["icu_death"][in_re].mean()) if in_re.any() else None}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "c3.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({"inter_dataset": rep["inter_dataset"], "eicu": rep["eicu_hospitals"],
                      "rare": rep["rare_strata"]}, indent=1))


if __name__ == "__main__":
    main()
