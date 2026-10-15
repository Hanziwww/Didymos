"""The calibration benchmark's analysis (plan Phase F, S8; ``factory.oracle``): every rollout (the anchor's prompt and
one continuation) decoded on its twin's clock into an oracle store with derived concepts; each anchor's prediction
time is its prompt's last event. Per binary catalog program, over anchors (inclusion-weighted):

* the label (rollout 0) and p* (the share of positives among rollouts 1..n-1, at least ``--min-valid`` valid);
* the oracle: AUROC of p* for the label, and the ceiling it estimates (labels drawn again from Bernoulli(p*)); the
  noise-corrected Brier score (Brier minus p*(1 - p*)/(n - 1));
* the raking gap: the twin's own future (the bank, raked to whole-stay margins) against p* (the model as is);
* a probe's calibration against p*: ExtraTrees on the bank's snapshot rows of other twins, scored at the anchors
  (AUROC for the label, correlation with p*, the calibration slope of p* on the probe).

Programs with at least ``--min-pos`` positive labels are scored (the S8 gate: >= 30). Aggregates only.

    python tools/oracle_analysis.py --oracle runs/oracle/MUC32_main --bank data/bank/MUC32_main_bank \\
        --store data/factory/stores/bank_dev --out runs/factory/oracle_s8.json --workers 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def auroc(y, p, w=None) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p, sample_weight=w))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("oracle", "store", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--tokens", default="data/tokens/full_r4")
    ap.add_argument("--work", default="data/factory/stores/oracle_MUC32_main")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-valid", type=int, default=10)
    ap.add_argument("--min-pos", type=int, default=30)
    a = ap.parse_args(argv)
    import pyarrow.parquet as pq

    from icutwin.factory import episodes, twins
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst
    from icutwin.tokenization.vocab import Vocab

    tokens = Path(a.tokens)
    v = Vocab.load(tokens / "vocab.json")
    anchors = pq.read_table(Path(a.oracle) / "anchors.parquet").to_pandas()
    n_anchor_rows = len(anchors)
    anchors = anchors.drop_duplicates("anchor").set_index("anchor")   # an anchor drawn twice (choose before the
    rows, seen = [], set()                                            # uniqueness rule): its first rollouts only
    for f in sorted((Path(a.oracle) / "rollouts").glob("*.parquet")):
        for r in pq.read_table(f, columns=["anchor", "twin", "rollout", "prompt", "continuation"]).to_pylist():
            if (r["anchor"], r["rollout"]) not in seen:
                seen.add((r["anchor"], r["rollout"]))
                rows.append(r)
    work = Path(a.work)
    if not (work / "manifest.json").exists():
        seqs = [{"stay_id": f"{r['anchor']}#{r['rollout']}", "ids": list(r["prompt"]) + list(r["continuation"]),
                 "clock": r["twin"], "split": "oracle", "weight": float(anchors.loc[r["anchor"], "weight"]),
                 "cell": "oracle"} for r in rows]
        fst.build(seqs, tokens / "vocab.json", work, "seg_0000", {"kind": "oracle", "rollouts": str(a.oracle)},
                  persistence=tokens / "rank_persistence.json", workers=a.workers, chunk=250)
    orc = fst.Store(work)
    idx = {s: i for i, s in enumerate(orc.stays["stay_id"].tolist())}
    bank = fst.Store(Path(a.store))
    bidx = {s: i for i, s in enumerate(bank.stays["stay_id"].tolist())}
    prompts = {r["anchor"]: (r["twin"], list(r["prompt"])) for r in rows if r["rollout"] == 0}
    t_end = {}
    end = [v.id("[END]"), v.id("[EOS]")]
    for anc, (twin, p) in prompts.items():          # the prediction time: the prompt's last event, on the twin's clock
        s = twins.decode_labels(p + end, v, anc, clock_id=twin)
        t_end[anc] = int(np.floor(max((e[0] - s.icu_in).total_seconds() for e in s.timeline) / 60))
    n_roll = 1 + max(r["rollout"] for r in rows)
    progs = [p for p in fpr.catalog(orc.channels) if p.label == "binary"]
    ev_o, ev_b = fpr.Evaluator(orc), fpr.Evaluator(bank)
    names = [a_ for a_ in prompts if a_ in anchors.index]
    stays_o = np.array([[idx.get(f"{a_}#{k}", -1) for k in range(n_roll)] for a_ in names])
    ok_o = (stays_o >= 0).all(1)
    names = [n for n, ok in zip(names, ok_o) if ok]
    stays_o = stays_o[ok_o]
    tt = np.array([t_end[a_] for a_ in names], dtype=np.int64)
    w = anchors.loc[names, "weight"].to_numpy(dtype=np.float64)
    own_idx = np.array([bidx.get(anchors.loc[a_, "twin"], -1) for a_ in names])
    s_bank = episodes.Sampler(bank, v, seed=0, filter_on=False)
    ctx_rows = fl.rolling(bank, 360)                    # the probes' rows: other twins every 6 hours
    rep = {"anchors": len(names), "duplicate_anchors_dropped": n_anchor_rows - len(anchors), "rollouts": int(n_roll),
           "programs": {}}
    for p in progs:
        r_all = fl.Rows(stays_o.reshape(-1).astype(np.int64), np.repeat(tt, n_roll))
        b = ev_o.evaluate(p, r_all)
        y = np.where(b.valid, b.y, np.nan).reshape(len(names), n_roll)
        y0, rest = y[:, 0], y[:, 1:]
        n_valid = np.isfinite(rest).sum(1)
        keep = np.isfinite(y0) & (n_valid >= a.min_valid)
        if keep.sum() < 20:
            continue
        p_star = np.nanmean(rest[keep], axis=1)
        lab, ww = y0[keep], w[keep]
        npos = int(lab.sum())
        res = {"anchors": int(keep.sum()), "positives": npos, "mean_p_star": float(np.average(p_star, weights=ww)),
               "label_rate": float(np.average(lab, weights=ww))}
        if npos >= a.min_pos:
            res["oracle_auroc"] = auroc(lab, p_star, ww)
            rng = np.random.default_rng(0)
            sims = [auroc(rng.random(len(p_star)) < p_star, p_star, ww) for _ in range(200)]
            res["oracle_ceiling_auroc"] = float(np.nanmean([s for s in sims if s is not None]))
            nv = n_valid[keep]
            res["brier_corrected"] = float(np.average((lab - p_star) ** 2 - p_star * (1 - p_star) / (nv - 1),
                                                      weights=ww))
            # the twin's own future at the same time (the raked bank) against p*
            oi = own_idx[keep]
            has = oi >= 0
            if has.any():
                bo = ev_b.evaluate(p, fl.Rows(oi[has].astype(np.int64), tt[keep][has]))
                own = np.where(bo.valid, bo.y, np.nan)
                m = np.isfinite(own)
                res["own_rate"] = float(np.average(own[m], weights=ww[has][m])) if m.any() else None
                res["own_auroc_by_p_star"] = auroc(own[m], p_star[has][m], ww[has][m]) if m.any() else None
            # a probe on other twins' snapshot rows, scored at the anchors
            cm = ~np.isin(ctx_rows.stay, own_idx)
            cb = ev_b.evaluate(p, ctx_rows.take(np.flatnonzero(cm)))
            sel = np.flatnonzero(cb.valid)
            if len(sel) > 50 and len(np.unique(cb.y[sel])) == 2:
                from sklearn.ensemble import ExtraTreesClassifier

                cr = ctx_rows.take(np.flatnonzero(cm)).take(sel[:20000])
                xc = s_bank.snapshot(p, cr)
                et = ExtraTreesClassifier(100, min_samples_leaf=5, random_state=0, n_jobs=4)
                et.fit(xc, cb.y[sel[:20000]])
                qa = fl.Rows(oi[has].astype(np.int64), tt[keep][has])
                pr = et.predict_proba(s_bank.snapshot(p, qa))[:, 1]
                res["probe_auroc"] = auroc(lab[has], pr, ww[has])
                res["probe_corr_with_p_star"] = float(np.corrcoef(pr, p_star[has])[0, 1])
                slope = np.polyfit(pr, p_star[has], 1, w=ww[has])[0] if np.std(pr) > 0 else None
                res["calibration_slope"] = None if slope is None else float(slope)
        rep["programs"][f"{p.name}@{p.horizon}"] = res
    scored = {k: r for k, r in rep["programs"].items() if "oracle_auroc" in r}
    rep["scored_programs"] = len(scored)
    rep["median_oracle_auroc"] = float(np.median([r["oracle_auroc"] for r in scored.values()])) if scored else None
    rep["median_probe_auroc"] = float(np.nanmedian([r.get("probe_auroc") or np.nan for r in scored.values()])) \
        if scored else None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("anchors", "rollouts", "scored_programs", "median_oracle_auroc",
                                          "median_probe_auroc")}, indent=1))
    for k, r in sorted(scored.items()):
        print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in r.items()})


if __name__ == "__main__":
    main()
