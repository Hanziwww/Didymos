"""The derived concepts of real stays three ways (aggregates only), for ``generate.clif_out``:
* ``canonical``: ``meds.derived.derive`` on the dataset's canonical CLIF (the reference);
* ``meds``: its MEDS stays -> ``clif_out`` -> ``derive`` (what MEDS carries; ``--canonical`` needed for the first);
* ``tokens``: its MEDS stays -> tokens (``--vocab``) -> decoded (values sampled within their bins, as twins are) ->
  ``clif_out`` -> ``derive`` (what the tokens carry: the path of every twin).
Compared per stay as the goal block reads them (``tokenization.stays._goal_from_meta``): an IMV, vasopressor or RRT
episode overlapping the ICU stay, the highest KDIGO stage in the ICU, the 24-hour SOFA; each pair: agreement, and the
first ICU-overlapping episode's start difference in hours (median, 90th percentile).
    python tools/clif_roundtrip.py --meds data/meds/subset/mimic --canonical data/clif_canonical/subset/mimic \
        --vocab data/tokens/full_r4/vocab.json --out .tmp/clif_rt/mimic [--splits held_out] [--max-stays 1000]"""
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from icutwin.generate import clif_out
from icutwin.meds.derived import derive
from icutwin.tokenization.decode import decode, load_persistence
from icutwin.tokenization.encode import encode
from icutwin.tokenization.stays import iter_stays
from icutwin.tokenization.vocab import Vocab

EPISODES = ("imv", "vasopressor", "rrt")


def _naive(t):
    return t.replace(tzinfo=None) if t is not None and t.tzinfo else t


def icu_windows(clif: Path, keys: set) -> dict:
    """stay -> (ICU admission, ICU discharge) of a CLIF directory's ext_icu_stay (each path reads its own)."""
    df = pl.read_parquet(clif / "clif_ext_icu_stay.parquet", columns=["stay_id", "icu_in_dttm", "icu_out_dttm"])
    return {str(s): (_naive(a), _naive(b)) for s, a, b in df.iter_rows() if str(s) in keys}


def concepts(derived: Path, windows: dict) -> dict:
    """stay -> {imv, vasopressor, rrt: first ICU-overlapping start or None, kdigo_max, sofa_total}."""
    out = {k: {"kdigo_max": None, "sofa_total": None, **dict.fromkeys(EPISODES)} for k in windows}
    p = derived / "episodes.parquet"
    if p.exists():
        for sid, ep, a, b in pl.read_parquet(p).select("stay_id", "episode", "start_dttm", "end_dttm").iter_rows():
            w = windows.get(str(sid))
            if w is None or ep not in EPISODES:
                continue
            a, b = _naive(a), _naive(b)
            if a < w[1] and b > w[0]:
                cur = out[str(sid)][ep]
                out[str(sid)][ep] = a if cur is None or a < cur else cur
    p = derived / "kdigo.parquet"
    if p.exists():
        for sid, t, stage in pl.read_parquet(p).select("stay_id", "t", "aki_stage").iter_rows():
            w, t = windows.get(str(sid)), _naive(t)
            if w is not None and stage is not None and (t is None or w[0] <= t <= w[1]):
                cur = out[str(sid)]["kdigo_max"]
                out[str(sid)]["kdigo_max"] = stage if cur is None else max(cur, stage)
    p = derived / "sofa_24h.parquet"
    if p.exists():
        for sid, v in pl.read_parquet(p).select("stay_id", "sofa_total").iter_rows():
            if str(sid) in out:
                out[str(sid)]["sofa_total"] = v
    return out


def compare(a: dict, b: dict) -> dict:
    keys = sorted(set(a) & set(b))
    rep = {"stays": len(keys)}
    for ep in EPISODES:
        x = np.array([a[k][ep] is not None for k in keys])
        y = np.array([b[k][ep] is not None for k in keys])
        d = [abs((a[k][ep] - b[k][ep]).total_seconds()) / 3600 for k in keys if a[k][ep] and b[k][ep]]
        rep[ep] = {"both": int((x & y).sum()), "only_first": int((x & ~y).sum()), "only_second": int((~x & y).sum()),
                   "agreement": round(float((x == y).mean()), 4) if keys else None,
                   "start_diff_h_median": round(float(np.median(d)), 2) if d else None,
                   "start_diff_h_p90": round(float(np.quantile(d, 0.9)), 2) if d else None}
    for c in ("kdigo_max", "sofa_total"):
        pairs = [(a[k][c] or 0, b[k][c] or 0) for k in keys if a[k][c] is not None or b[k][c] is not None]
        x, y = np.array([p[0] for p in pairs], float), np.array([p[1] for p in pairs], float)
        rep[c] = {"stays_with_value": len(pairs), "exact": round(float((x == y).mean()), 4) if pairs else None,
                  "mean_abs_diff": round(float(np.abs(x - y).mean()), 3) if pairs else None,
                  "mean_first": round(float(x.mean()), 3) if pairs else None,
                  "mean_second": round(float(y.mean()), 3) if pairs else None}
    return rep


def run_derive(clif: Path, out: Path) -> Path:
    derived = out / "derived"
    derive(clif, derived, out / "work", min_age_years=None)
    shutil.rmtree(out / "work", ignore_errors=True)
    return derived


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meds", required=True)
    ap.add_argument("--canonical", default=None)
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "tuning", "held_out"])
    ap.add_argument("--max-stays", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode", choices=("sample", "median", "times"), default="sample",
                    help="decoded values and gaps drawn within their bins, at their medians, or gaps drawn and "
                         "values at their medians (the factory's labels)")
    ap.add_argument("--persistence", default=None,
                    help="tools/rank_persistence.py's JSON: drawn ranks continue within a bin (tokenization.decode)")
    a = ap.parse_args()
    out = Path(a.out)
    stays = [s for split in a.splits for s in iter_stays(Path(a.meds), split)]
    if a.max_stays and len(stays) > a.max_stays:
        stays = random.Random(a.seed).sample(stays, a.max_stays)
    keys = [s.stay_id for s in stays]
    res, written = {}, {}
    if a.canonical:
        res["canonical"] = concepts(run_derive(Path(a.canonical), out / "canonical"),
                                    icu_windows(Path(a.canonical), set(keys)))
    written["meds"] = clif_out.write(stays, keys, out / "meds" / "clif")
    res["meds"] = concepts(run_derive(out / "meds" / "clif", out / "meds"),
                           icu_windows(out / "meds" / "clif", set(keys)))
    if a.vocab:
        v = Vocab.load(Path(a.vocab))
        rng = np.random.default_rng(a.seed) if a.decode != "median" else None
        rho = load_persistence(Path(a.persistence), v) if a.persistence else None
        decoded = [decode(encode(s, v).ids, v, rng, rho, **({"value_rng": None} if a.decode == "times" else {}))
                   for s in stays]
        for s, d in zip(stays, decoded, strict=True):     # the decoded clock starts at T0: back on the real one
            shift = s.icu_in - d.icu_in
            d.timeline = [(t + shift, *rest) for t, *rest in d.timeline]
            d.icu_in, d.icu_out = d.icu_in + shift, d.icu_out + shift
        written["tokens"] = clif_out.write(decoded, keys, out / "tokens" / "clif")
        res["tokens"] = concepts(run_derive(out / "tokens" / "clif", out / "tokens"),
                                 icu_windows(out / "tokens" / "clif", set(keys)))
    report = {"written": written}
    if "canonical" in res:
        report["canonical_vs_meds"] = compare(res["canonical"], res["meds"])
    if "tokens" in res:
        report["meds_vs_tokens"] = compare(res["meds"], res["tokens"])
        if "canonical" in res:
            report["canonical_vs_tokens"] = compare(res["canonical"], res["tokens"])
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, default=str))


if __name__ == "__main__":
    main()
