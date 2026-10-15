"""Paper P amendment 7, F5's analysis (``docs/p1_paper_protocol.md``): the what-if rollouts of
``tools/server/paper_whatif2.py`` decoded on their stays' clocks into a store; C1's pre-specified responses
(``tools/paper_c1.py``: channels, windows, aggregates, expected signs, the urine-output cap) per rollout, averaged per
anchor and arm; the real responses from the real store at the same times.

* (a) A/A: per prefix, withhold (seed 7) minus withhold (seed 8): the share of C1's "expected sign" (null 0.5) and the
  mean with its bootstrap interval (null 0).
* (b) the factual design, per treatment and response:
  - the Twin's contrast, give minus withhold, on starters and on non-starters (mean, interval, share of the expected
    sign);
  - the observational contrast: the starters' real responses minus their matched non-starters' (pairs);
  - the Twin's version of it: the starters' give arm minus the non-starters' withhold arm (the factual arms);
  - the factual check: each factual arm's ensemble against the real response (80% interval coverage, the ensemble's
    CRPS against persistence, i.e. the same aggregate over the preceding window, and the PIT's KS distance).

Writes ``<out>/c1b.json`` (aggregates).

    python tools/paper_c1b.py --whatif runs/paper/f5 --work data/paper/stores/f5 --out runs/paper/f5 --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import _sha  # noqa: E402
from paper_c1 import CAP, RESPONSES  # noqa: E402


def boot_mean(x: np.ndarray, n: int = 1000, seed: int = 0) -> list:
    if len(x) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    b = [np.mean(rng.choice(x, size=len(x))) for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("whatif", "work", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--tokens", default="data/tokens/full_r4")
    ap.add_argument("--store", default="data/factory/stores/real_test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--segment", type=int, default=15000)
    a = ap.parse_args(argv)
    import pyarrow.parquet as pq

    from icutwin.eval import bench
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst

    wd, tokens, work = Path(a.whatif), Path(a.tokens), Path(a.work)
    an = pq.read_table(wd / "anchors.parquet").to_pandas().set_index("anchor")
    rows = []
    for f in sorted((wd / "rollouts").rglob("*.parquet")):
        t = pq.read_table(f, columns=["anchor", "arm", "rollout", "prompt", "continuation"])
        for x, arm, k, p, c in zip(*(t.column(n).to_pylist() for n in ("anchor", "arm", "rollout")),
                                   t.column("prompt").to_pylist(), t.column("continuation").to_pylist(), strict=True):
            rows.append({"stay_id": f"{x}|{arm}#{k}", "ids": np.asarray(list(p) + list(c), np.int32),
                         "clock": str(an.loc[x, "twin"]), "split": "whatif", "weight": 1.0, "cell": arm})
        del t
    done = set()
    if (work / "manifest.json").exists():
        done = {s["name"] for s in json.loads((work / "manifest.json").read_text())["segments"]}
    for k in range(-(-len(rows) // a.segment)):
        name = f"seg_{k:04d}"
        if name not in done:
            fst.build(rows[k * a.segment:(k + 1) * a.segment], tokens / "vocab.json", work, name,
                      {"kind": "paper_c1b", "whatif": str(wd), "part": k},
                      persistence=tokens / "rank_persistence.json", workers=a.workers, chunk=100)
    del rows
    st = fst.Store(work)
    ev = fpr.Evaluator(st)
    sid = st.stays["stay_id"].tolist()
    anc = np.array([s.split("|")[0] for s in sid])
    arm = np.array([s.split("|")[1].split("#")[0] for s in sid])
    t_roll = an.loc[anc, "t_min"].to_numpy(dtype=np.int64)
    rows_r = fl.Rows(np.arange(len(sid), dtype=np.int64), t_roll)
    real = fst.Store(Path(a.store))
    ev_q = fpr.Evaluator(real)
    qidx = {str(s): i for i, s in enumerate(real.stays["stay_id"].tolist())}
    an_ok = an[an["twin"].astype(str).isin(qidx)]
    rows_q = fl.Rows(np.array([qidx[str(s)] for s in an_ok["twin"]], dtype=np.int64),
                     an_ok["t_min"].to_numpy(dtype=np.int64))

    def values(e, store_ev, r, resp, chans, win, agg):
        evs = fl.finite(store_ev.events(list(chans)))
        if resp in CAP:
            keep = np.asarray(evs["v"], dtype=float) <= CAP[resp]
            evs = {k: np.asarray(evs[k])[keep] for k in ("stay", "t", "v", "bin", "f") if k in evs}
        val, _n = fl.window_values(evs, r, 0, win, agg)
        prev, _n2 = fl.window_values(evs, fl.Rows(r.stay, r.t - win), 0, win, agg)   # persistence
        return val, prev

    treat_of = an["treatment"].to_dict()
    group_of = an["group"].to_dict()
    rep = {"git": _sha(), "rollouts": len(sid), "aa": {}, "factual": {}}
    for name, resps in RESPONSES.items():
        for resp, chans, win, agg, sign in resps:
            v_r, _p = values(None, ev, rows_r, resp, chans, win, agg)
            v_q, prev_q = values(None, ev_q, rows_q, resp, chans, win, agg)
            real_of = dict(zip(an_ok.index, v_q, strict=True))
            prev_of = dict(zip(an_ok.index, prev_q, strict=True))
            by: dict = {}
            for x, ar, y in zip(anc, arm, v_r, strict=True):
                if np.isfinite(y) and treat_of[x] == name:
                    by.setdefault((x, ar), []).append(y)
            means = {k: float(np.mean(vv)) for k, vv in by.items()}
            # (a) A/A on C1's prefixes
            aa = [means[(x, "withhold_a")] - means[(x, "withhold_b")] for x in an.index
                  if group_of[x] == "aa" and treat_of[x] == name
                  and (x, "withhold_a") in means and (x, "withhold_b") in means]
            aa = np.array(aa)
            rep["aa"].setdefault(name, {})[resp] = {
                "prefixes": int(len(aa)), "share_expected_sign": float(np.mean(np.sign(aa) == sign)) if len(aa) else None,
                "mean": float(aa.mean()) if len(aa) else None, "mean_ci": boot_mean(aa),
                "sd": float(aa.std()) if len(aa) else None}
            # (b) the factual design
            res = {}
            for grp in ("starter", "nonstarter"):
                xs = [x for x in an.index if group_of[x] == grp and treat_of[x] == name]
                eff = np.array([means[(x, "give")] - means[(x, "withhold")] for x in xs
                                if (x, "give") in means and (x, "withhold") in means])
                res[f"twin_contrast_{grp}"] = {"anchors": int(len(eff)), "mean": float(eff.mean()) if len(eff) else None,
                                               "ci": boot_mean(eff), "share_expected_sign":
                                               float(np.mean(np.sign(eff) == sign)) if len(eff) else None}
                fact = "give" if grp == "starter" else "withhold"
                cov, crps, crps_p, pit = [], [], [], []
                for x in xs:
                    ens = np.array(by.get((x, fact), []))
                    y = real_of.get(x, np.nan)
                    if len(ens) < 4 or not np.isfinite(y):
                        continue
                    lo, hi = np.quantile(ens, [0.1, 0.9])
                    cov.append(lo <= y <= hi)
                    crps.append(bench.crps_ensemble(ens, y))
                    pv = prev_of.get(x, np.nan)
                    if np.isfinite(pv):
                        crps_p.append(abs(pv - y))
                    pit.append((np.sum(ens < y) + 0.5 * np.sum(ens == y)) / len(ens))
                pit = np.array(pit)
                ks = float(np.max(np.abs(np.sort(pit) - (np.arange(1, len(pit) + 1) - 0.5) / len(pit)))) if len(pit) else None
                res[f"factual_{grp}"] = {"anchors": len(cov), "coverage_80": float(np.mean(cov)) if cov else None,
                                         "crps": float(np.mean(crps)) if crps else None,
                                         "crps_persistence": float(np.mean(crps_p)) if crps_p else None,
                                         "pit_ks": ks}
            pairs = an[(an["treatment"] == name) & (an["group"] != "aa")].groupby("pair")
            obs, tw = [], []
            for _p, g in pairs:
                s_ = g.index[g["group"] == "starter"]
                n_ = g.index[g["group"] == "nonstarter"]
                if len(s_) != 1 or len(n_) != 1:
                    continue
                ys, yn = real_of.get(s_[0], np.nan), real_of.get(n_[0], np.nan)
                if np.isfinite(ys) and np.isfinite(yn):
                    obs.append(ys - yn)
                if (s_[0], "give") in means and (n_[0], "withhold") in means:
                    tw.append(means[(s_[0], "give")] - means[(n_[0], "withhold")])
            obs, tw = np.array(obs), np.array(tw)
            res["observational_contrast"] = {"pairs": int(len(obs)), "mean": float(obs.mean()) if len(obs) else None,
                                             "ci": boot_mean(obs)}
            res["twin_factual_contrast"] = {"pairs": int(len(tw)), "mean": float(tw.mean()) if len(tw) else None,
                                            "ci": boot_mean(tw)}
            res["expected_sign"] = sign
            rep["factual"].setdefault(name, {})[resp] = res
            print(json.dumps({name: {resp: {"aa_share": rep["aa"][name][resp]["share_expected_sign"],
                                            "obs": res["observational_contrast"]["mean"],
                                            "twin_starters": res["twin_contrast_starter"]["mean"]}}}), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "c1b.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
