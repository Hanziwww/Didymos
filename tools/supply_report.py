"""The twin-supply smoke's report (plan Phase F, S1b gate; ``generate.supply``), aggregates only:
* plan against realized twins per mechanism, and each mechanism's filter drops (duplicates, novelty, raking);
* event goals kept, on the tokens (the obligation registers) and on the derived concepts (in the ICU);
* value goals' acceptance (the twin's derived concept in the requested band);
* strata and sites: the realized prefix, header and site fields against the requested ones;
* the weights: within each dataset, outcome and header shares of all twins weighted, of the base twins (the model's
  natural distribution under the accepted sampler) and of the training stays; pooled over datasets at the natural
  mix; the unweighted shares show the enrichment the weights undo;
* the segment append's twins per core-hour and its byte-reproducibility (two builds of the same shards).

    python tools/supply_report.py --store data/factory/stores/s1b --store2 data/factory/stores/s1b_again \\
        --plan data/supply/s1b/plan.json --marginals data/tokens/full_r4/supply_marginals.json \\
        --shards data/supply/s1b --build-log logs/s1b_store.log --out runs/supply/s1b_smoke.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GOAL_RATES = (("icu_outcome", ("dead",)), ("rrt", ("yes",)), ("mcs", ("yes",)), ("imv", ("yes",)),
              ("vasopressor", ("yes",)), ("kdigo_max", ("3+",)), ("sofa_1d", ("10-12", "12+")))
HEAD_FIELDS = ("sex", "age_band")


def _rate(rows, pred, w=None) -> float | None:
    if not len(rows):
        return None
    y = np.array([bool(pred(x)) for x in rows], dtype=float)
    return float(np.average(y, weights=w)) if w is None or np.sum(w) > 0 else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("store", "plan", "marginals", "shards", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--store2", default=None)
    ap.add_argument("--build-log", default=None)
    a = ap.parse_args(argv)
    import pyarrow.parquet as pq

    from icutwin.factory import store as fst
    from icutwin.generate import supply

    plan = supply.SupplyPlan.load(Path(a.plan))
    marg = json.loads(Path(a.marginals).read_text(encoding="utf-8"))
    st = fst.Store(Path(a.store))
    attrs = supply.store_attributes(st)
    w = st.stays["weight"].to_numpy(dtype=np.float64)
    split = st.stays["split"].to_numpy()
    mech = np.array([str(x["cell"]).split("/")[0] for x in attrs])
    rep: dict = {"plan": plan.name, "twins": len(attrs), "weighted": bool(st.weighted)}

    # plan against realized, and the filters
    parts = sorted(p for p in Path(a.shards).iterdir() if p.is_dir() and p.name.startswith("part"))
    mans = [json.loads((p / "manifest.json").read_text(encoding="utf-8")) for p in parts]
    drops: dict = {}
    drawn: dict = {}
    unmet = 0
    for m in mans:
        for s in m["shards"]:
            k = "base" if s["dataset"].startswith("base_") else "conditioned"
            drawn[k] = drawn.get(k, 0) + s["drawn"]
            unmet += s.get("event_goals_unmet", 0)
            for r, n in s["dropped"].items():
                drops.setdefault(k, {})[r] = drops.get(k, {}).get(r, 0) + n
    rep["mechanisms"] = {m: {"planned": q, "kept": int((mech == m).sum())} for m, q in plan.mechanisms().items()}
    rep["drawn"] = drawn
    rep["dropped"] = drops
    rep["drop_rates"] = {k: {r: round(n / drawn[k], 4) for r, n in d.items()} for k, d in drops.items()}
    rep["event_goals_unmet_on_tokens"] = unmet

    # goals
    tok = {}
    for p in parts:
        for f in sorted(p.glob("*.parquet")):
            for r in pq.read_table(f, columns=["twin", "conditions", "goal_met"]).to_pylist():
                tok[r["twin"]] = r["goal_met"]
    goals: dict = {}
    for x in attrs:
        c = json.loads(x["conditions"]) if x.get("conditions") else {}
        req = {**(c.get("goals") or {}), **(c.get("filter") or {})}
        for f, lab in req.items():
            g = goals.setdefault(f"{f}={lab}" if f not in ("sofa_1d",) else "sofa_1d>=10", {"twins": 0,
                                                                                         "derived_met": 0,
                                                                                         "tokens_met": 0})
            g["twins"] += 1
            g["derived_met"] += int(x["goals"].get(f) == lab)
            g["tokens_met"] += int(tok.get(x["stay_id"]) is True)
    for g in goals.values():
        g["derived_share"] = round(g["derived_met"] / g["twins"], 4)
        g["tokens_share"] = round(g["tokens_met"] / g["twins"], 4)
    rep["goals"] = goals
    rep["rejected_goal_missed"] = int((split == "rejected").sum())
    ev = [g for k, g in goals.items() if k.split("=")[0] in supply.EVENT_GOALS]
    rep["event_goals_kept_on_tokens"] = (round(sum(g["tokens_met"] for g in ev) / sum(g["twins"] for g in ev), 4)
                                         if ev else None)

    # strata and sites as requested
    ok_s, n_s, ok_h, n_h = 0, 0, 0, 0
    for x in attrs:
        c = json.loads(x["conditions"]) if x.get("conditions") else {}
        if c.get("header"):
            n_s += 1
            sex, band, unit, adm, dx = x["stratum"].split("|")
            ok_s += int(sex == c["prefix"]["sex"] and band == supply.coarse_age(c["prefix"]["age"])
                        and unit == c["header"]["unit_type"] and adm == c["header"]["admission_type"]
                        and dx == c["header"]["adm_dx_group"])
        if c.get("site"):
            n_h += 1
            ok_h += int(x["site"] == c["site"])
    rep["strata_as_requested"] = round(ok_s / n_s, 4) if n_s else None
    rep["sites_as_requested"] = round(ok_h / n_h, 4) if n_h else None

    # weights: within datasets, and pooled at the natural mix
    keep = split != "rejected"
    p_nat = {d: n / sum(marg["datasets"].values()) for d, n in marg["datasets"].items()}
    preds = {f"{f}:{'|'.join(labs)}": (lambda x, f=f, labs=labs: x["goals"].get(f) in labs) for f, labs in GOAL_RATES}
    preds["sex:female"] = lambda x: x["sex"] == "female"
    preds["age:80+"] = lambda x: x["age_band"] == "80+"
    preds["age:18-45"] = lambda x: x["age_band"] == "18-45"

    def train_rate(d, name):
        f, labs = name.split(":")
        if f in ("sex", "age"):
            strata = marg["strata"][d]
            n = sum(s["n"] for s in strata.values())
            pos = sum(s["n"] for k, s in strata.items()
                      if (k.split("|")[0] == "female" if f == "sex" else k.split("|")[1] == labs))
            return pos / n
        g = marg["goals"][d][f]
        return sum(g.get(lab, 0) for lab in labs.split("|")) / sum(g.values())     # labels such as '3+'

    per_ds: dict = {}
    pooled: dict = {k: {"weighted": 0.0, "base": 0.0, "unweighted": 0.0, "training": 0.0} for k in preds}
    for d in sorted(set(x["dataset"] for x in attrs)):
        m_all = np.array([x["dataset"] == d for x in attrs]) & keep
        m_base = m_all & (mech == "base")
        rows_all = [attrs[i] for i in np.flatnonzero(m_all)]
        rows_base = [attrs[i] for i in np.flatnonzero(m_base)]
        r = {"twins": int(m_all.sum()), "base_twins": int(m_base.sum())}
        for k, f in preds.items():
            vals = {"weighted": _rate(rows_all, f, w[m_all]), "base": _rate(rows_base, f),
                    "unweighted": _rate(rows_all, f), "training": train_rate(d, k)}
            r[k] = {x: (round(y, 4) if y is not None else None) for x, y in vals.items()}
            for x, y in vals.items():
                if y is not None:
                    pooled[k][x] += p_nat.get(d, 0) * y
        per_ds[d] = r
    tot = sum(p_nat[d] for d in per_ds)
    rep["pooled_at_the_natural_mix"] = {k: {x: round(y / tot, 4) for x, y in v.items()} for k, v in pooled.items()}
    gaps_train = [abs(v["weighted"] - v["training"]) for v in rep["pooled_at_the_natural_mix"].values()]
    gaps_base = [abs(v["weighted"] - v["base"]) for v in rep["pooled_at_the_natural_mix"].values()]
    rep["max_gap_weighted_to_training"] = round(max(gaps_train), 4)
    rep["max_gap_weighted_to_base"] = round(max(gaps_base), 4)
    rep["per_dataset"] = per_ds
    ww = w[keep]
    rep["weights_p01_p50_p99_max"] = np.quantile(ww, [0.01, 0.5, 0.99, 1.0]).round(4).tolist()
    rep["weights_ess_share"] = round(float(ww.sum() ** 2 / (ww ** 2).sum() / len(ww)), 4)

    # the store append
    if a.build_log and Path(a.build_log).exists():
        for line in Path(a.build_log).read_text(encoding="utf-8").splitlines():
            if line.startswith("{") and "twins_per_core_hour" in line:
                rep["store_append"] = json.loads(line)
    if a.store2:
        m1 = json.loads((Path(a.store) / "manifest.json").read_text(encoding="utf-8"))
        m2 = json.loads((Path(a.store2) / "manifest.json").read_text(encoding="utf-8"))
        rep["byte_reproducible"] = [s["files"] for s in m1["segments"]] == [s["files"] for s in m2["segments"]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in rep if k not in ("per_dataset",)}, indent=1))


if __name__ == "__main__":
    main()
