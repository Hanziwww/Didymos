"""Within-bin rank persistence of real stays (aggregates only), for the decoder's optional AR(1) draw (plan S5): per
numeric code, consecutive values of one stay that fall in the same bin; each value's rank within its bin (the inverse
of the decoder's piecewise-linear sketch of its dataset's bin), as a normal score; the correlation of consecutive
scores. Codes whose bins hold one value (exact bins) are skipped.
    python tools/rank_persistence.py --meds data/meds/subset --vocab data/tokens/full_r4/vocab.json --out <json>"""
import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from icutwin.tokenization.stays import iter_stays
from icutwin.tokenization.vocab import Vocab, bin_of

DATASETS = ("mimic", "eicu", "hirid", "aumc", "sicdb", "nwicu", "inspire")


def rank_in(x: float, sketch: list) -> float | None:
    """The decoder's ``read_sketch`` inverted: u in [0, 1] with ``read_sketch(sketch, u) == x`` (None: a flat bin)."""
    if sketch[0] == sketch[-1]:
        return None
    k = len(sketch) - 1
    i = int(np.searchsorted(sketch, x, side="right")) - 1
    if i < 0:
        return 0.0
    if i >= k:
        return 1.0
    lo, hi = sketch[i], sketch[i + 1]
    return (i + (0.5 if hi == lo else (x - lo) / (hi - lo))) / k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meds", required=True, help="a root of per-dataset MEDS directories")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pairs", type=int, default=200)
    a = ap.parse_args()
    v, nd = Vocab.load(Path(a.vocab)), NormalDist()
    pairs: dict = {}
    for ds in DATASETS:
        for split in a.splits:
            for s in iter_stays(Path(a.meds) / ds, split):
                last: dict = {}
                for _t, _r, code, num, _x in s.timeline:
                    info = v.codes.get(code)
                    if num is None or not info or not info.get("edges"):
                        continue
                    b = bin_of(float(num), info["edges"])
                    own = (v.value_sketches_ds.get(code) or {}).get(ds)
                    sk = own[b] if own and own[b] is not None else (v.value_sketches.get(code) or [None] * (b + 1))[b]
                    u = rank_in(float(num), sk) if sk else None
                    if u is None:
                        last.pop(code, None)
                        continue
                    z = nd.inv_cdf(min(max(u, 0.005), 0.995))
                    prev = last.get(code)
                    if prev is not None and prev[0] == b:
                        pairs.setdefault(code, []).append((prev[1], z))
                    last[code] = (b, z)
    per_code = {}
    for code, p in pairs.items():
        if len(p) >= a.min_pairs:
            x = np.asarray(p)
            per_code[code] = {"pairs": len(p), "rho": round(float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]), 4)}
    by_domain: dict = {}
    for code, r in per_code.items():
        by_domain.setdefault(code.split("//", 1)[0], []).append(r["rho"])
    out = {"codes": len(per_code), "pairs": sum(r["pairs"] for r in per_code.values()),
           "rho_median": round(float(np.median([r["rho"] for r in per_code.values()])), 4) if per_code else None,
           "by_domain": {d: {"codes": len(x), "median": round(float(np.median(x)), 4)}
                         for d, x in sorted(by_domain.items())},
           "per_code": per_code}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("codes", "pairs", "rho_median", "by_domain")}))


if __name__ == "__main__":
    main()
