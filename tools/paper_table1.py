"""Paper P's materials: the cohort table (Table 1) from every stay's tokens (all splits of the r4 tokens, each stay
decoded as the model sees it: ``factory.twins.decode_labels``), per dataset: stays per split, age (5-year bands at their
midpoints), female share, admission and unit types, ICU length of stay, ICU death (the ICU discharge as dead) and in-hospital death (discharge
category expired), race where recorded, IMV, vasoactive and
RRT in the ICU, timeline events and tokens per stay. Aggregates only, written to ``<out>/table1.json``.

    python tools/paper_table1.py --tokens data/tokens/full_r4 --out runs/paper/materials --workers 12
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATASETS = ("mimic", "eicu", "hirid", "aumc", "sicdb", "nwicu", "inspire")
SPLITS = ("train", "tuning", "held_out")
_V: dict = {}


def _job(args) -> list:
    tokens, rows = args
    from icutwin.factory import oracle, twins
    from icutwin.tokenization.vocab import Vocab

    if "v" not in _V:
        _V["v"] = Vocab.load(Path(tokens) / "vocab.json")
        _V["codes"] = oracle.event_codes(_V["v"])
    v, codes = _V["v"], _V["codes"]
    out = []
    for sid, ids in rows:
        s = twins.decode_labels(ids, v, str(sid))
        los = (s.icu_out - s.icu_in).total_seconds() / 3600
        on = oracle.onsets(s, codes)
        end = " ".join(s.end)
        age = s.prefix.get("age")
        try:
            age = float(age)
        except (TypeError, ValueError):
            age = np.nan
        out.append({"age": age, "sex": str(s.prefix.get("sex")), "adm": str(s.header.get("admission_type")),
                    "unit": str(s.header.get("unit_type")), "race": str(s.prefix.get("race")),
                    "era": str(s.prefix.get("era")), "los": los,
                    "icu_death": on["death"] is not None,                 # the ICU discharge as dead
                    "hosp_death": "HOSP//discharge_category//expired" in s.end,
                    "imv": on["imv"] is not None, "vaso": on["vasopressor"] is not None, "rrt": on["rrt"] is not None,
                    "events": len(s.timeline), "tokens": len(ids), "end": end})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=500)
    a = ap.parse_args(argv)
    import pyarrow.parquet as pq

    tokens = Path(a.tokens)
    rep = {"datasets": {}}
    ends = Counter()
    for ds in DATASETS:
        counts, recs = {}, []
        for sp in SPLITS:
            t = pq.read_table(tokens / ds / f"{sp}.parquet", columns=["stay_id", "ids"])
            counts[sp] = t.num_rows
            rows = list(zip(t.column("stay_id").to_pylist(), t.column("ids").to_pylist(), strict=True))
            jobs = [(str(tokens), rows[k:k + a.chunk]) for k in range(0, len(rows), a.chunk)]
            with ProcessPoolExecutor(a.workers) as ex:
                for part in ex.map(_job, jobs):
                    recs += part
        for r in recs:
            for code in r["end"].split():
                if code.startswith(("ICU//", "HOSP//discharge")):
                    ends[code] += 1

        def pct(k):
            return float(np.mean([r[k] for r in recs]) * 100)

        los = np.array([r["los"] for r in recs])
        age = np.array([r["age"] for r in recs])
        rep["datasets"][ds] = {
            "stays": int(sum(counts.values())), "splits": counts,
            "age_mean": float(np.nanmean(age)), "age_sd": float(np.nanstd(age)),
            "female_pct": float(np.mean([r["sex"] == "female" for r in recs]) * 100),
            "admission_type": {k: round(v / len(recs) * 100, 1) for k, v in
                               Counter(r["adm"] for r in recs).most_common(5)},
            "unit_type": {k: round(v / len(recs) * 100, 1) for k, v in Counter(r["unit"] for r in recs).most_common(5)},
            "race": {k: round(v / len(recs) * 100, 1) for k, v in Counter(r["race"] for r in recs).most_common(6)},
            "era": {k: round(v / len(recs) * 100, 1) for k, v in sorted(Counter(r["era"] for r in recs).items())},
            "icu_los_h": {"median": float(np.median(los)), "q25": float(np.quantile(los, 0.25)),
                          "q75": float(np.quantile(los, 0.75))},
            "icu_death_pct": pct("icu_death"), "hosp_death_pct": pct("hosp_death"), "imv_pct": pct("imv"),
            "vasoactive_pct": pct("vaso"), "rrt_pct": pct("rrt"),
            "events_per_stay": {"median": float(np.median([r["events"] for r in recs])),
                                "q25": float(np.quantile([r["events"] for r in recs], 0.25)),
                                "q75": float(np.quantile([r["events"] for r in recs], 0.75))},
            "tokens_per_stay_median": float(np.median([r["tokens"] for r in recs]))}
        print(json.dumps({ds: {k: rep["datasets"][ds][k] for k in ("stays", "icu_death_pct", "hosp_death_pct")}}),
              flush=True)
    rep["end_codes"] = dict(ends.most_common(30))
    rep["stays"] = int(sum(x["stays"] for x in rep["datasets"].values()))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "table1.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
