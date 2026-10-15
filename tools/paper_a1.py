"""Paper P, A1 and A2's analysis (``docs/p1_paper_protocol.md``): the twin model's rollouts on real anchors
(``tools/server/paper_rollouts.py``) scored against the real stays' own futures.

Every rollout (its prompt and continuation) is decoded on its stay's clock into a rollout store with derived
concepts (as the S8 oracle's); the real labels come from the real store (``real_test``: the same decode). At each
anchor's prediction time (the prompt's last event, ``anchors.parquet`` ``t_min``), per catalog program:

* binary programs: the label, and p-hat = the share of positive rollouts among the valid ones (at least
  ``--min-valid``); metrics per dataset and pooled (``eval.bench.binary_metrics``), AUROC intervals by a stratified
  bootstrap over anchors, the rollout-count curve for the primary endpoints;
* remaining ICU stay over 7 days (``remaining_los_gt_7d``): from the stores' ICU discharge (a capped rollout past 7 days
  counts, one capped before is unknown);
* regressions: the rollouts' ensemble against the value (MAE of the median, CRPS, interval coverage);
* times to event: Harrell's C of the rollouts' 48-hour event share against the real (time, event).

Writes ``<out>/predictions.parquet`` (per anchor and program; host only) and ``<out>/a1.json`` (aggregates).

    python tools/paper_a1.py --design-dir runs/paper/a1/held_out/L24 --store data/factory/stores/real_test \\
        --work data/paper/stores/a1_L24 --out runs/paper/a1/held_out/L24 --workers 12
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

PRIMARY = {"L24": ("icu_death@-1", "hospital_death@0", "remaining_los_gt_7d"),
           "L48": ("icu_death@-1", "hospital_death@0", "remaining_los_gt_7d"),
           "R": ("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880", "aki_2@2880",
                 "sepsis_ase@1440", "hypotension@240", "hyperkalemia_top@1440", "thrombocytopenia@2880")}


def applies(p, design: str) -> bool:
    """The programs a design scores (protocol A1): landmarks every program but event-triggered ones; R binary
    programs within 48 h, times to event, regressions within 48 h."""
    if p.name.startswith("extubation_failure"):
        return False
    if design in ("L24", "L48"):
        return True
    if p.label == "binary":
        return 0 < p.horizon <= 2880
    if p.label == "tte":
        return True
    if p.label == "regression":
        return p.name not in ("remaining_los",)
    return False


def key(p) -> str:
    return f"{p.name}@{p.horizon}"


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("design-dir", "store", "work", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--tokens", default="data/tokens/full_r4")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--segment", type=int, default=15000, help="stays per store segment (bounds the parent's memory)")
    ap.add_argument("--chunk", type=int, default=100, help="stays per decode job (a worker's memory grows with it)")
    ap.add_argument("--min-valid", type=int, default=16)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--max-rollouts", type=int, default=0,
                    help="amendment 7 (F9a): only rollouts numbered below this (e.g. the first 32 of 128)")
    ap.add_argument("--curve-draws", type=int, default=0,
                    help="amendment 7: the curve as the mean over this many random k-subsets (0: the first k, A1)")
    a = ap.parse_args(argv)
    import pandas as pd
    import pyarrow.parquet as pq

    from icutwin.eval import bench
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst

    dd, tokens = Path(a.design_dir), Path(a.tokens)
    design = json.loads((dd / "design.json").read_text())["design"]
    anchors = pq.read_table(dd / "anchors.parquet").to_pandas().drop_duplicates("anchor").reset_index(drop=True)
    rows, seen = [], set()                         # token lists as packed int32 (Python lists of ints cost 7x)
    for f in sorted((dd / "rollouts").rglob("*.parquet")):     # amendment 7: lanes write rollouts/shard<i>/
        t = pq.read_table(f, columns=["anchor", "twin", "rollout", "prompt", "continuation"])
        for anc, twin, k, pr, co in zip(t.column("anchor").to_pylist(), t.column("twin").to_pylist(),
                                        t.column("rollout").to_pylist(), t.column("prompt").to_pylist(),
                                        t.column("continuation").to_pylist(), strict=True):
            if (anc, k) not in seen and (not a.max_rollouts or k < a.max_rollouts):
                seen.add((anc, k))
                rows.append({"anchor": anc, "twin": twin, "rollout": k,
                             "ids": np.concatenate([np.asarray(pr, np.int32), np.asarray(co, np.int32)])})
        del t
    work = Path(a.work)
    spec = json.loads((dd / "design.json").read_text())
    real_ids: dict = {}                             # the alignment control: each anchor's real stay as a member
    for d in sorted(set(anchors["dataset"])):
        for sp in spec["split"].split(","):        # several splits, comma-separated (amendment 16a)
            t = pq.read_table(tokens / d / f"{sp}.parquet", columns=["stay_id", "ids"]).to_pylist()
            want = set(anchors["twin"].astype(str))
            real_ids.update({str(r["stay_id"]): r["ids"] for r in t if str(r["stay_id"]) in want})
    done = set()                                   # segments already built (a rerun resumes)
    if (work / "manifest.json").exists():
        done = {s["name"] for s in json.loads((work / "manifest.json").read_text())["segments"]}
    n_seg = -(-(len(rows) + len(anchors)) // a.segment)
    if len(done) < n_seg:
        w_of = dict(zip(anchors["anchor"], anchors["weight"], strict=True))
        if spec.get("unknown_site") or spec.get("prompt_transform", "none") != "none":
            from icutwin.model.data import layout    # B2, F9a: the real dataset and site tokens back (the decoder
            from icutwin.tokenization.vocab import Vocab   # writes the dataset into CLIF and draws its gaps; the

            lay = layout(Vocab.load(tokens / "vocab.json"))   # conditioning was the point, not the content)
            swap = spec.get("prompt_transform") == "swap"
            for r in rows:
                real = real_ids.get(str(r["twin"]))
                if real is None:
                    continue
                head = r["ids"][:16]                 # a view: assignments change the rollout's tokens
                if swap:
                    head[lay.dataset_pos] = int(real[lay.dataset_pos])
                    head[lay.site_pos] = int(real[lay.site_pos])
                else:
                    head[head == lay.dataset_unknown] = int(real[lay.dataset_pos])
                    head[head == lay.site_unknown] = int(real[lay.site_pos])
        seqs = [{"stay_id": f"{r['anchor']}#{r['rollout']}", "ids": r["ids"],
                 "clock": r["twin"], "split": "rollout", "weight": float(w_of.get(r["anchor"], 1.0)),
                 "cell": "rollout"} for r in rows]
        seqs += [{"stay_id": f"{x}#real", "ids": np.asarray(real_ids[str(s)], np.int32), "clock": str(s), "split": "control",
                  "weight": 1.0, "cell": "control"} for x, s in zip(anchors["anchor"], anchors["twin"], strict=True)
                 if str(s) in real_ids]
        for k in range(n_seg):                     # segments of a.segment stays: the parent's memory stays bounded
            name = f"seg_{k:04d}"
            if name not in done:
                fst.build(seqs[k * a.segment:(k + 1) * a.segment], tokens / "vocab.json", work, name,
                          {"kind": "paper_a1", "rollouts": str(dd), "part": k},
                          persistence=tokens / "rank_persistence.json", workers=a.workers, chunk=a.chunk)
        del seqs
    ro, real = fst.Store(work), fst.Store(Path(a.store))
    ridx = {s: i for i, s in enumerate(ro.stays["stay_id"].tolist())}
    qidx = {str(s): i for i, s in enumerate(real.stays["stay_id"].tolist())}
    n_roll = 1 + max(r["rollout"] for r in rows)
    have = anchors["anchor"].map(lambda x: all(f"{x}#{k}" in ridx for k in range(n_roll))) & \
        anchors["twin"].astype(str).isin(qidx)
    anchors = anchors[have].reset_index(drop=True)
    st_r = np.array([[ridx[f"{x}#{k}"] for k in range(n_roll)] for x in anchors["anchor"]], dtype=np.int64)
    st_q = np.array([qidx[str(x)] for x in anchors["twin"]], dtype=np.int64)
    tt = anchors["t_min"].to_numpy(dtype=np.int64)
    w = anchors["weight"].to_numpy(dtype=np.float64)
    ds = anchors["dataset"].to_numpy()
    ev_r, ev_q = fpr.Evaluator(ro), fpr.Evaluator(real)
    rows_r = fl.Rows(st_r.reshape(-1), np.repeat(tt, n_roll))
    rows_q = fl.Rows(st_q, tt)
    preds, rep = [], {"design": design, "anchors": int(len(anchors)), "rollouts": int(n_roll),
                      "git": _sha(), "programs": {}}
    rep["censored_rollouts"] = float(np.mean(np.isin(ro.end_kind[st_r.reshape(-1)], ("censored", "unknown"))))

    def add_pred(name, y, p, extra=None):
        for i in np.flatnonzero(np.isfinite(y) & np.isfinite(p)):
            preds.append({"anchor": anchors["anchor"][i], "stay": str(anchors["twin"][i]), "dataset": ds[i],
                          "weight": w[i], "program": name, "label": float(y[i]), "p": float(p[i]),
                          **({k: float(v[i]) for k, v in extra.items()} if extra else {})})

    def score_binary(name, y, p, nv=None):
        ok = np.isfinite(y) & np.isfinite(p)
        res = {"pooled": bench.binary_metrics(y[ok], p[ok], w[ok])}
        if "auroc" in res["pooled"] and a.boot:
            res["pooled"]["auroc_ci"] = bench.bootstrap(lambda yy, pp, ww: bench.auroc_or_none(yy, pp, ww),
                                                        anchors["twin"].to_numpy()[ok], y[ok], p[ok], w[ok],
                                                        n=a.boot, strata=ds[ok])
        res["datasets"] = {}
        for d in sorted(set(ds[ok])):
            m = ok & (ds == d)
            r = bench.binary_metrics(y[m], p[m], w[m])
            if "auroc" in r and a.boot and name in PRIMARY.get(design, ()):
                r["auroc_ci"] = bench.bootstrap(lambda yy, pp, ww: bench.auroc_or_none(yy, pp, ww),
                                                anchors["twin"].to_numpy()[m], y[m], p[m], w[m], n=a.boot)
            res["datasets"][d] = r
        if nv is not None:
            res["valid_rollouts_median"] = float(np.median(nv[ok])) if ok.any() else None
        return res

    def curve(y, yr):
        """AUROC and Brier with the first k rollouts (k = 1, 2, 4, ..., n); with ``--curve-draws``, their means over
        that many random k-subsets of the rollouts (amendment 7)."""
        out = {}
        rng_c = np.random.default_rng(7)
        for k in (1, 2, 4, 8, 16, 32, 64, 128):
            if k > n_roll:
                break
            draws = [np.arange(k)] if not a.curve_draws or k == n_roll else \
                [rng_c.choice(n_roll, size=k, replace=False) for _ in range(a.curve_draws)]
            au, br = [], []
            for cols in draws:
                sub = yr[:, cols]
                nv = np.isfinite(sub).sum(1)
                p = np.where(nv > 0, np.nansum(sub, 1) / np.maximum(nv, 1), np.nan)
                ok = np.isfinite(y) & np.isfinite(p)
                au.append(bench.auroc_or_none(y[ok], p[ok], w[ok]))
                br.append(float(np.average((y[ok] - p[ok]) ** 2, weights=w[ok])) if ok.any() else None)
            au, br = [x for x in au if x is not None], [x for x in br if x is not None]
            out[str(k)] = {"auroc": float(np.mean(au)) if au else None, "brier": float(np.mean(br)) if br else None}
        return out

    progs = [p for p in fpr.catalog(ro.channels) if applies(p, design)]
    # the alignment control: the real stay decoded as a rollout member on its clock must give the real store's label
    ctl = np.array([ridx.get(f"{x}#real", -1) for x in anchors["anchor"]], dtype=np.int64)
    has_ctl = ctl >= 0
    rows_c = fl.Rows(ctl[has_ctl], tt[has_ctl])
    rep["alignment_control"] = {}
    for p in progs:
        if p.label not in ("binary", "hosp_death") or not has_ctl.any():
            continue
        bc, bq0 = ev_r.evaluate(p, rows_c), ev_q.evaluate(p, fl.Rows(st_q[has_ctl], tt[has_ctl]))
        both = bc.valid & bq0.valid
        rep["alignment_control"][key(p)] = {
            "valid_agreement": float(np.mean(bc.valid == bq0.valid)),
            "label_agreement": float(np.mean(bc.y[both] == bq0.y[both])) if both.any() else None,
            "clinical_cut": "clinical_cut" in p.flags}
    agree = [min(r["label_agreement"], r["valid_agreement"]) for r in rep["alignment_control"].values()
             if r["label_agreement"] is not None and not r["clinical_cut"]]
    rep["alignment_control_min"] = float(min(agree)) if agree else None     # bin-edge programs: must be 1
    for p in progs:
        bq = ev_q.evaluate(p, rows_q)
        br = ev_r.evaluate(p, rows_r)
        y = np.where(bq.valid, bq.y, np.nan).astype(np.float64)
        yr = np.where(br.valid, br.y, np.nan).astype(np.float64).reshape(len(anchors), n_roll)
        nv = np.isfinite(yr).sum(1)
        if p.label in ("binary", "hosp_death"):
            ph = np.where(nv >= a.min_valid, np.nansum(yr, 1) / np.maximum(nv, 1), np.nan)
            name = key(p)
            rep["programs"][name] = score_binary(name, y, ph, nv)
            if name in PRIMARY.get(design, ()):
                rep["programs"][name]["rollout_curve"] = curve(y, yr)
            add_pred(name, y, ph, {"n_valid": nv})
        elif p.label == "regression":
            ok = np.isfinite(y) & (nv >= a.min_valid)
            if ok.sum() < 30:
                continue
            med = np.nanmedian(yr[ok], 1)
            crps = [bench.crps_ensemble(r[np.isfinite(r)], t) for r, t in zip(yr[ok], y[ok], strict=True)]
            rep["programs"][key(p)] = {"n": int(ok.sum()), "mae_median": float(np.mean(np.abs(med - y[ok]))),
                                       "mae_baseline_mean": float(np.mean(np.abs(y[ok] - np.mean(y[ok])))),
                                       "crps": float(np.mean(crps)), "coverage": bench.interval_coverage(yr[ok], y[ok])}
            add_pred(key(p), y, np.where(ok, np.nanmedian(yr, 1), np.nan))
        elif p.label == "tte":
            te_r = np.where(br.valid & (br.cause == 0), br.t_event, np.inf).reshape(len(anchors), n_roll)
            vr = br.valid.reshape(len(anchors), n_roll)
            risk = np.where(vr.sum(1) >= a.min_valid, ((te_r <= 48) & vr).sum(1) / np.maximum(vr.sum(1), 1), np.nan)
            ok = bq.valid & np.isfinite(risk)
            t_obs = np.where(bq.cause == 0, bq.t_event, bq.t_censor)[ok]
            rep["programs"][key(p)] = {"n": int(ok.sum()), "events": int((bq.cause[ok] == 0).sum()),
                                       "c_index_48h_share": bench.c_index(t_obs, bq.cause[ok] == 0, risk[ok])}
        elif p.label == "ordinal" and p.name == "remaining_los_band":
            pass                                             # replaced by remaining_los_gt_7d below
    # remaining ICU stay over 7 days, from the stores' discharge times (capped rollouts past 7 days count)
    rem_r = (ro.los[st_r] - tt[:, None]) / 60.0
    cens = np.isin(ro.end_kind[st_r], ("censored", "unknown"))
    yr = np.where(rem_r > 168, 1.0, np.where(cens, np.nan, 0.0))
    nv = np.isfinite(yr).sum(1)
    # amendment 3: Kaplan-Meier over the rollouts (a capped rollout is censored at its cap, not dropped)
    ph = np.array([bench.km_survival(rem_r[i], ~cens[i], 168.0) for i in range(len(anchors))])
    rep["remaining_los_estimator"] = "kaplan_meier"
    ph_drop = np.where(nv >= a.min_valid, np.nansum(yr, 1) / np.maximum(nv, 1), np.nan)
    rem_q = (real.los[st_q] - tt) / 60.0
    y = np.where(rem_q > 168, 1.0, np.where(np.isin(real.end_kind[st_q], ("censored", "unknown")) & (rem_q <= 168),
                                            np.nan, 0.0))
    if design in ("L24", "L48"):
        mcc = np.isfinite(ph_drop) & np.isfinite(y)
        rep["remaining_los_gt_7d_complete_case"] = bench.binary_metrics(y[mcc], ph_drop[mcc], w[mcc])
        rep["programs"]["remaining_los_gt_7d"] = score_binary("remaining_los_gt_7d", y, ph, nv)
        rep["programs"]["remaining_los_gt_7d"]["rollout_curve"] = curve(y, yr)
        add_pred("remaining_los_gt_7d", y, ph, {"n_valid": nv})
        ens = np.where(cens, np.nan, rem_r)
        okr = np.isfinite(rem_q) & (np.isfinite(ens).sum(1) >= a.min_valid)
        rep["remaining_los_dispersion"] = __import__("icutwin.eval.forecast", fromlist=["dispersion"]).dispersion(
            [float(x) for x in rem_q[okr]], [list(r[np.isfinite(r)]) for r in ens[okr]])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_parquet(out / "predictions.parquet", index=False)
    (out / "a1.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    prim = {k: {x: rep["programs"][k]["pooled"].get(x) for x in ("auroc", "auroc_ci", "observed_over_expected")}
            for k in PRIMARY.get(design, ()) if k in rep["programs"]}
    print(json.dumps({"anchors": rep["anchors"], "rollouts": n_roll, "censored": rep["censored_rollouts"],
                      "alignment_control_min": rep["alignment_control_min"], "primary": prim}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
