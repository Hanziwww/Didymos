"""Paper P, C1's analysis (``docs/p1_paper_protocol.md`` C1, amendment 2): the what-if rollouts
(``tools/server/paper_whatif.py``) decoded on their stays' clocks into a store, and each pre-specified response read
over its window after the prediction time t (the prompt's last event): the mean of a channel's values (``mean``),
their sum (``sum``) or how often it is charted (``count``), per rollout, averaged over an arm's rollouts with a value.
The effect of a prefix is give (median bin) minus withhold; per treatment and response: the prefixes scored, the
share with the expected sign, the mean and median effect with a bootstrap interval over prefixes, and for the dose
arms the mean response per given bin. Writes ``<out>/c1.json`` (aggregates).

    python tools/paper_c1.py --whatif runs/paper/c1 --work data/paper/stores/c1 --out runs/paper/c1 --workers 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import _sha  # noqa: E402

MAP = ("VITAL//map//arterial", "VITAL//map//cuff")
UO = ("OUTPUT//indwelling_urinary_catheter", "OUTPUT//external_urinary_catheter")
CAP = {"urine_output": 2000.0}   # mL per record: the real store's 99th percentile is 1.5 million (cumulative totals)
RESPONSES = {   # treatment: [(response, channels, window minutes, aggregate, expected sign)]
    "norepinephrine": [("map", MAP, 360, "mean", 1)],
    "insulin": [("glucose", ("LAB//glucose//mg/dl//poc", "LAB//glucose_serum//mg/dl//bmp"), 360, "mean", -1)],
    "crrt": [("potassium", ("LAB//potassium//mmol/l//bmp",), 360, "mean", -1),
             ("creatinine_24h", ("LAB//creatinine//mg/dl//bmp",), 1440, "mean", -1)],
    "furosemide": [("urine_output", UO, 360, "sum", 1)],
    "imv": [("fio2_charted", ("RESP//fio2_set",), 360, "count", 1), ("spo2", ("VITAL//spo2",), 360, "mean", 1)],
    "fluid_bolus": [("map", MAP, 360, "mean", 1), ("urine_output", UO, 360, "sum", 1)],
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("whatif", "work", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--tokens", default="data/tokens/full_r4")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(argv)
    import pyarrow.parquet as pq

    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst

    wd, tokens = Path(a.whatif), Path(a.tokens)
    anchors = pq.read_table(wd / "anchors.parquet").to_pandas().set_index("anchor")
    rows = []
    for f in sorted((wd / "rollouts").glob("*.parquet")):
        rows += pq.read_table(f).to_pylist()
    work = Path(a.work)
    if not (work / "manifest.json").exists():
        seqs = [{"stay_id": f"{r['anchor']}|{r['arm']}#{r['rollout']}", "ids": list(r["prompt"]) + list(r["continuation"]),
                 "clock": anchors.loc[r["anchor"], "twin"], "split": "whatif", "weight": 1.0, "cell": r["arm"]}
                for r in rows]
        fst.build(seqs, tokens / "vocab.json", work, "seg_0000", {"kind": "paper_c1", "whatif": str(wd)},
                  persistence=tokens / "rank_persistence.json", workers=a.workers, chunk=250)
    st = fst.Store(work)
    ev = fpr.Evaluator(st)
    sid = st.stays["stay_id"].tolist()
    anc = np.array([s.split("|")[0] for s in sid])
    arm = np.array([s.split("|")[1].split("#")[0] for s in sid])
    t = anchors.loc[anc, "t_min"].to_numpy(dtype=np.int64)
    rows_all = fl.Rows(np.arange(len(sid), dtype=np.int64), t)
    rep = {"git": _sha(), "rollouts": len(sid), "treatments": {}}
    rng = np.random.default_rng(0)
    for name, resps in RESPONSES.items():
        mine = anchors.index[anchors["treatment"] == name]
        if not len(mine):
            continue
        rep["treatments"][name] = {"prefixes": int(len(mine)), "responses": {}}
        for resp, chans, win, agg, sign in resps:
            evs = fl.finite(ev.events(list(chans)))
            if resp in CAP:                        # amendment 6: implausible single records (source artefacts) out
                v_ = np.asarray(evs["v"], dtype=float)
                keep = v_ <= CAP[resp]
                evs = {k: np.asarray(evs[k])[keep] for k in ("stay", "t", "v", "bin", "f") if k in evs}
            if agg == "count":
                val, _n = fl.window_values(evs, rows_all, 0, win, "count")
            else:
                val, _n = fl.window_values(evs, rows_all, 0, win, agg)
            by: dict = {}
            for x, ar, y in zip(anc, arm, val, strict=True):
                if x in mine and np.isfinite(y):
                    by.setdefault((x, ar), []).append(y)
            means = {k: float(np.mean(v)) for k, v in by.items()}
            eff = np.array([means[(x, "give_median")] - means[(x, "withhold")] for x in mine
                            if (x, "give_median") in means and (x, "withhold") in means])
            res = {"scored": int(len(eff)), "expected_sign": sign}
            if len(eff):
                boots = [np.mean(rng.choice(eff, size=len(eff))) for _ in range(a.boot)]
                res.update(share_expected_sign=float(np.mean(np.sign(eff) == sign)),
                           mean_effect=float(np.mean(eff)), median_effect=float(np.median(eff)),
                           mean_effect_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                           withhold_mean=float(np.mean([means[(x, "withhold")] for x in mine
                                                        if (x, "withhold") in means])))
            dose = {}
            for ar in sorted({k[1] for k in means if k[1].startswith("give_")}):
                vals = [means[(x, ar)] for x in mine if (x, ar) in means]
                if vals:
                    dose[ar] = {"prefixes": len(vals), "mean_response": float(np.mean(vals))}
            if len(dose) > 1:
                res["dose_response"] = dose
            rep["treatments"][name]["responses"][resp] = res
        print(json.dumps({name: {r: (x.get("share_expected_sign"), x.get("mean_effect"))
                                 for r, x in rep["treatments"][name]["responses"].items()}}), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "c1.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
