"""Paper P amendment 7, F1-F3's analysis (``docs/p1_paper_protocol.md``): one subset's 128 rollouts per anchor
(``tools/server/paper_rollouts.py``), their hazards and the real futures' (``tools/server/paper_hazards.py``), and
A1's stored predictions on the same anchors.

* F1: A1's estimator from the label engine (``tools/paper_a1.py`` on the subset: ``predictions.parquet``, the curve
  over random k-subsets) against A1's 32-rollout Twin, LightGBM pooled, TabPFN-3.5 and the probe on the same anchors:
  AUROC, AUPRC, Brier and O/E (inclusion weights), paired bootstrap differences.
* F2: E0, E1 and E2 (``eval.hazards.estimate``) for death within 24 h, new IMV and vasopressor within 24 h and new RRT
  within 48 h (and, with ``--long``, ICU death until discharge, also cut at A1's cap), at k = 1 ... 128 rollouts (the
  mean over 20 random k-subsets, the same subsets for all three), against the real labels; the self-check (sampled
  events over summed hazards within the rollouts).
* F3: D1 (the real futures' compensator against their observed events: O/E by dataset and ICU day), D2 (the drift
  ratio: the rollouts' mean compensator over the real futures', the hourly hazards since the anchor, minutes per
  token), D3 (censoring: capped rollouts by the real outcome; ICU death at A1's cap against four times it).

Writes ``<out>/f2.json`` (aggregates) and, since amendment 16, ``<out>/estimates.parquet``: every anchor's E0, E1 and
E2 at all of its rollouts (and at A1's cap, ``*_a1cap``), so that paired P values can be exact for the hazard estimate.

    python tools/paper_f2.py --subset runs/paper/f1/R128 --a1 runs/paper/a1/test_R --out runs/paper/f1/R128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import _sha  # noqa: E402

ENDPOINTS = {   # program: (cause, horizon hours)
    "decompensation@1440": ("death", 24.0),
    "imv_onset@1440": ("imv", 24.0),
    "vasopressor_onset@1440": ("vasopressor", 24.0),
    "rrt_onset@2880": ("rrt", 48.0),
}
LONG = {"icu_death@-1": ("death", np.inf)}
KS = (1, 2, 4, 8, 16, 32, 64, 128)


def load_hazards(d: Path) -> dict:
    parts = [np.load(f) for f in sorted(d.glob("part_*.npz"))]
    if not parts:
        return {}
    keys = [k for k in parts[0].files if k != "edges"]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["edges"] = parts[0]["edges"]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="the subset's design dir (anchors, hazards, predictions)")
    ap.add_argument("--a1", required=True, help="A1's design dir of the same anchors (stored predictions)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--long", action="store_true")
    a = ap.parse_args(argv)
    import pandas as pd

    from icutwin.eval import bench
    from icutwin.eval import hazards as hz

    sd, a1 = Path(a.subset), Path(a.a1)
    an = pd.read_parquet(sd / "anchors.parquet").drop_duplicates("anchor").reset_index(drop=True)
    w = an["weight"].to_numpy(dtype=float)
    ds = an["dataset"].to_numpy()
    stay = an["twin"].astype(str).to_numpy()
    H = load_hazards(sd / "hazards")
    edges = H["edges"]
    roll = H["rollout"] >= 0
    rep = {"git": _sha(), "anchors": int(len(an)), "subset": str(sd)}

    # labels: the real stores' (A1's predictions of the same anchors carry them)
    pa1 = pd.read_parquet(a1 / "predictions.parquet")
    lab = {p: g.set_index("anchor")["label"].reindex(an["anchor"]).to_numpy(dtype=float)
           for p, g in pa1.groupby("program")}

    # per anchor: its rollouts' rows in H (and its real future's)
    order = np.lexsort((H["rollout"], H["anchor_idx"]))
    ai_sorted = H["anchor_idx"][order]
    rows_of = {i: order[ai_sorted == i] for i in np.unique(ai_sorted)}
    roll_of = {i: r[H["rollout"][r] >= 0] for i, r in rows_of.items()}
    real_of = {i: r[H["rollout"][r] < 0][0] for i, r in rows_of.items() if (H["rollout"][r] < 0).any()}
    n_roll = int(max(len(r) for r in roll_of.values())) if roll_of else 0
    rep["rollouts_per_anchor"] = n_roll

    def arrays(rr, cause, cut=False):
        first, eh, ek = H[f"first_{cause}"][rr], H["end_hour"][rr], H["end_kind"][rr].astype(int)
        if cause == "death":
            ev, ot = (H["inccut_death"][rr], H["inccut_alive"][rr]) if cut else (H["inc_death"][rr], H["inc_alive"][rr])
        else:
            ev, ot = H[f"inc_{cause}"][rr], H[f"comp_{cause}"][rr]
        e_ = edges[:ev.shape[1] + 1]
        if cut:
            ch = H["cut_hour"][rr]
            hit = ch < eh
            eh, ek = np.where(hit, ch, eh), np.where(hit, hz.END_CENSORED, ek)
        return e_, first, eh, ek, ev, ot

    def metrics(y, p, ww):
        ok = np.isfinite(y) & np.isfinite(p)
        return bench.binary_metrics(y[ok], p[ok], ww[ok]) if ok.sum() and 0 < y[ok].sum() < ok.sum() else {"n": 0}

    def paired(y, p1, p2, ww):
        ok = np.isfinite(y) & np.isfinite(p1) & np.isfinite(p2)
        if not ok.sum() or not 0 < y[ok].sum() < ok.sum():
            return None
        d = bench.auroc_or_none(y[ok], p1[ok], ww[ok]) - bench.auroc_or_none(y[ok], p2[ok], ww[ok])
        ci = bench.bootstrap(lambda yy, aa, bb, w_: (bench.auroc_or_none(yy, aa, w_) or np.nan)
                             - (bench.auroc_or_none(yy, bb, w_) or np.nan),
                             stay[ok], y[ok], p1[ok], p2[ok], ww[ok], n=a.boot, strata=ds[ok])
        return {"diff": float(d), "ci": list(ci), "n": int(ok.sum()), "positives": int(y[ok].sum())}

    rng = np.random.default_rng(7)
    subsets = {k: [np.sort(rng.choice(n_roll, size=k, replace=False)) for _ in range(a.draws)] if k < n_roll
               else [np.arange(n_roll)] for k in KS if k <= n_roll}

    # A1's stored predictions on these anchors: the 32-rollout Twin and the baselines
    others = {"twin_a1_32": pa1}
    for f, name in (("baseline_predictions.parquet", None), ("tabpfn_v3_5_predictions.parquet", "tabpfn_v3_5"),
                    ("probe_predictions.parquet", None)):
        if (a1 / f).exists():
            others[f] = pd.read_parquet(a1 / f)

    def other_preds(prog):
        out = {}
        g = pa1[pa1["program"] == prog].set_index("anchor")["p"]
        out["twin_a1_32"] = g.reindex(an["anchor"]).to_numpy(dtype=float)
        for f, d_ in others.items():
            if f == "twin_a1_32":
                continue
            d_ = d_[d_["program"] == prog]
            for m_, g2 in d_.groupby("model"):
                if m_ in ("lgbm_pooled", "tabpfn_v3_5", "twin_probe"):
                    out[m_] = g2.drop_duplicates("anchor").set_index("anchor")["p"].reindex(an["anchor"]).to_numpy(float)
        return out

    # F1: A1's estimator with all the subset's rollouts (tools/paper_a1.py on the subset) against the comparators,
    # every primary endpoint of the design
    rep["f1"] = {}
    if (sd / "predictions.parquet").exists() and n_roll:
        from paper_a1 import PRIMARY

        ps = pd.read_parquet(sd / "predictions.parquet")
        design = "L24" if a.long else "R"
        if (sd / "design.json").exists():               # the subset's own design (L48x since amendment 16)
            design = json.loads((sd / "design.json").read_text(encoding="utf-8")).get("design", design)
        for prog in PRIMARY[design]:
            if prog not in lab:
                continue
            y = lab[prog]
            p128 = ps[ps["program"] == prog].set_index("anchor")["p"].reindex(an["anchor"]).to_numpy(float)
            op = other_preds(prog)
            rep["f1"][prog] = {"twin_128": metrics(y, p128, w),
                               "comparators": {m_: metrics(y, p_, w) for m_, p_ in op.items()},
                               "paired": {f"twin_128_minus_{m_}": paired(y, p128, p_, w) for m_, p_ in op.items()}}

    # F2 and F1's comparisons per endpoint
    eps = dict(ENDPOINTS)
    if a.long:
        eps.update(LONG)
    rep["f2"] = {}
    per_anchor = []                                   # every anchor's estimates at all rollouts (amendment 16)
    for prog, (cause, hor) in eps.items():
        if prog not in lab or n_roll == 0:            # D1 alone (real futures, no rollouts)
            continue
        y = lab[prog]
        res = {"curve": {}, "cause": cause, "horizon": hor if np.isfinite(hor) else "discharge"}
        full = {}
        for k, draws in subsets.items():
            vals = {e: [] for e in ("e0", "e1", "e2")}
            for cols in draws:
                est = {e: np.full(len(an), np.nan) for e in vals}
                for i, rr in roll_of.items():
                    if len(rr) < n_roll:
                        continue
                    e_, first, eh, ek, ev, ot = arrays(rr[cols], cause)
                    r_ = hz.estimate(cause, hor, e_, first, eh, ek, ev, ot)
                    for e in vals:
                        est[e][i] = r_[e]
                for e in vals:
                    vals[e].append(bench.auroc_or_none(y[np.isfinite(y) & np.isfinite(est[e])],
                                                       est[e][np.isfinite(y) & np.isfinite(est[e])],
                                                       w[np.isfinite(y) & np.isfinite(est[e])]))
                if k == n_roll:
                    full = est
            res["curve"][str(k)] = {e: (float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v)
                                        else None) for e, v in vals.items()}
        res["estimators"] = {e: metrics(y, full[e], w) for e in ("e0", "e1", "e2")}
        for e in ("e0", "e1", "e2"):
            per_anchor.append(pd.DataFrame({"anchor": an["anchor"], "program": prog, "estimator": e, "p": full[e]}))
        op = other_preds(prog)
        res["comparators"] = {m_: metrics(y, p_, w) for m_, p_ in op.items()}
        res["paired"] = {f"e2_minus_{m_}": paired(y, full["e2"], p_, w) for m_, p_ in op.items()}
        res["paired"]["e2_minus_e0"] = paired(y, full["e2"], full["e0"], w)
        res["paired"]["e0_128_minus_twin_a1_32"] = paired(y, full["e0"], op.get("twin_a1_32", full["e0"] * np.nan), w)
        if not np.isfinite(hor):                          # D3: ICU death under A1's cap
            cut = {e: np.full(len(an), np.nan) for e in ("e0", "e1", "e2")}
            for i, rr in roll_of.items():
                e_, first, eh, ek, ev, ot = arrays(rr, cause, cut=True)
                r_ = hz.estimate(cause, hor, e_, first, eh, ek, ev, ot)
                for e in cut:
                    cut[e][i] = r_[e]
            res["at_a1_cap"] = {e: metrics(y, cut[e], w) for e in cut}
            for e in cut:
                per_anchor.append(pd.DataFrame({"anchor": an["anchor"], "program": prog, "estimator": f"{e}_a1cap",
                                                "p": cut[e]}))
        # the self-check: sampled events over summed hazards within the rollouts (whole follow-up in the bins)
        rr_all = np.flatnonzero(roll)
        e_, first, eh, ek, ev, ot = arrays(rr_all, cause)
        first = np.where(first <= eh, first, np.inf)       # events inside the bins the hazards were summed over
        n_ev = float(np.sum((first > 0) & (first <= e_[-1])))
        comp = float(ev.sum())
        res["self_check"] = {"events": n_ev, "compensator": comp, "o_e": n_ev / comp if comp else None}
        rep["f2"][prog] = res
        print(json.dumps({prog: {e: res["estimators"][e].get("auroc") for e in ("e0", "e1", "e2")},
                          "self_check": res["self_check"]["o_e"]}), flush=True)

    # F3, D1 and D2: the real futures' compensator against their events, and the rollouts' against the real futures'
    rep["f3"] = {}
    have_real = np.array(sorted(real_of))
    for prog, (cause, hor) in eps.items():
        if prog not in lab or not len(have_real):
            continue
        h = hor if np.isfinite(hor) else float(edges[-1])
        nb = int(np.searchsorted(edges, h, side="right") - 1)
        ri = np.array([real_of[i] for i in have_real])
        ev_real = (H["inc_death"] if cause == "death" else H[f"inc_{cause}"])[ri][:, :nb].sum(1)
        y = lab[prog][have_real]
        ww = w[have_real]
        ok = np.isfinite(y)
        d1 = {"pooled": {"observed": float(np.sum(ww[ok] * y[ok])), "expected": float(np.sum(ww[ok] * ev_real[ok]))}}
        for d in sorted(set(ds[have_real])):
            m = ok & (ds[have_real] == d)
            d1[d] = {"observed": float(np.sum(ww[m] * y[m])), "expected": float(np.sum(ww[m] * ev_real[m])),
                     "anchors": int(m.sum())}
        for x in d1.values():
            x["o_e"] = x["observed"] / x["expected"] if x["expected"] else None
        dsr = ds[have_real]
        brng = np.random.default_rng(0)                # a bootstrap over anchors, within datasets
        bs = []
        for _ in range(500):
            idx = np.concatenate([brng.choice(np.flatnonzero(ok & (dsr == d)), size=int((ok & (dsr == d)).sum()))
                                  for d in sorted(set(dsr[ok]))])
            e_ = np.sum(ww[idx] * ev_real[idx])
            if e_ > 0:
                bs.append(np.sum(ww[idx] * y[idx]) / e_)
        d1["pooled"]["o_e_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None
        roll_mean = np.array([(H["inc_death"] if cause == "death" else H[f"inc_{cause}"])[roll_of[i]][:, :nb]
                              .sum(1).mean() if len(roll_of.get(i, [])) else np.nan for i in have_real])
        okr = ok & np.isfinite(roll_mean)
        d2 = {"rollout_compensator": float(np.sum(ww[okr] * roll_mean[okr])),
              "real_compensator": float(np.sum(ww[okr] * ev_real[okr])), "observed": float(np.sum(ww[okr] * y[okr]))}
        d2["drift_ratio"] = d2["rollout_compensator"] / d2["real_compensator"] if d2["real_compensator"] else None
        rep["f3"][prog] = {"d1": d1, "d2": d2}
    # D2: hourly death and alive-discharge hazards since the anchor among those at risk, rollouts against real
    if len(have_real):
        hb = min(48, len(edges) - 1)
        lo = edges[:hb]
        curves = {}
        for c in ("death", "alive"):
            inc = H[f"inc_{c}"][:, :hb]
            stop = H["end_hour"]
            at = (stop[:, None] > lo[None, :]).astype(float)
            rr = np.flatnonzero(roll)
            ri = np.flatnonzero(~roll)
            curves[c] = {"rollouts": (inc[rr].sum(0) / np.maximum(at[rr].sum(0), 1)).tolist(),
                         "real": (inc[ri].sum(0) / np.maximum(at[ri].sum(0), 1)).tolist()}
        mpt = H["cont_minutes"] / np.maximum(H["cont_tokens"], 1)
        rep["f3"]["hourly_hazards"] = curves
        rep["f3"]["minutes_per_token"] = {"rollouts_median": float(np.median(mpt[roll])),
                                          "real_median": float(np.median(mpt[~roll])) if (~roll).any() else None}
    # D3: capped rollouts by the real outcome
    capped = np.array([np.mean(H["end_kind"][roll_of[i]] == hz.END_CENSORED) if len(roll_of.get(i, [])) else np.nan
                       for i in range(len(an))])
    d3 = {"capped_share_mean": float(np.nanmean(capped))}
    for prog in ("hospital_death@0", "icu_death@-1", "decompensation@1440"):
        if prog in lab:
            y = lab[prog]
            d3[prog] = {str(int(v_)): float(np.nanmean(capped[y == v_])) for v_ in (0, 1) if (y == v_).any()}
    rep["f3"]["d3"] = d3
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "f2.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    if per_anchor:
        pd.concat(per_anchor, ignore_index=True).to_parquet(out / "estimates.parquet", index=False)


if __name__ == "__main__":
    main()
