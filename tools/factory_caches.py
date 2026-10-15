"""The factory's caches and task realism (plan S8 *Tasks*): real held-out stays sampled to the bank's per-dataset
counts, a disjoint second real sample (the control), and the bank's twins, all built the same way
(``factory.stream.cache_from_sequences``: decoded from tokens, derived concepts through canonical CLIF from values at
their bins' medians on the drawn clock, features drawn with the real rank persistence), the second and the bank on
the first one's grid concepts; then ``eval.realism.compare`` of each against the first. Caches hold stays' features:
they stay on approved machines (``.tmp`` here); the printed reports are aggregates.
    python tools/factory_caches.py real --tokens data/tokens/full_r4 --counts '{"mimic": 2763}' --out .tmp/factory
    python tools/factory_caches.py bank --bank data/bank/MUC32_main_bank --tokens data/tokens/full_r4 --out .tmp/factory
    python tools/factory_caches.py realism --out .tmp/factory [--tasks 60 --rows 400]"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from icutwin.factory import stream
from icutwin.factory.cache import Cache
from icutwin.tokenization.decode import load_persistence
from icutwin.tokenization.vocab import Vocab


def _vocab(tokens: Path):
    v = Vocab.load(tokens / "vocab.json")
    p = tokens / "rank_persistence.json"
    return v, (load_persistence(p, v) if p.exists() else None)


def _labels(c: Cache) -> dict:
    """Per dataset: stays, the share with each onset, SOFA known and its mean, the median ICU stay (hours)."""
    out = {}
    for ds in sorted(set(c.dataset)):
        m = c.dataset == ds
        sofa = c.sofa_24h[m][~np.isnan(c.sofa_24h[m])]
        out[ds] = {"stays": int(m.sum()), **{e: round(float(np.mean(~np.isnan(c.onset[e][m]))), 3) for e in c.onset},
                   "sofa_24h_mean": round(float(sofa.mean()), 2) if len(sofa) else None,
                   "death": round(float(np.nanmean(c.death[m])), 3),
                   "los_h_median": round(float(np.median(c.los_hours[m])), 1)}
    return out


def real(a) -> None:
    tokens, out = Path(a.tokens), Path(a.out)
    v, rho = _vocab(tokens)
    counts = json.loads(a.counts)
    rng = np.random.default_rng(a.seed)
    samples: dict = {"real": ([], []), "real2": ([], [])}
    for ds, n in sorted(counts.items()):
        t = pq.read_table(tokens / ds / "held_out.parquet", columns=["stay_id", "ids"])
        order = rng.permutation(t.num_rows)
        take = min(int(n), t.num_rows // 2)
        for name, rows in (("real", order[:take]), ("real2", order[take:2 * take])):
            sub = t.take(rows)
            samples[name][0].extend(sub.column("ids").to_pylist())
            samples[name][1].extend(str(s) for s in sub.column("stay_id").to_pylist())
    concepts = None
    for name in ("real", "real2"):
        t0 = time.time()
        seqs, ids = samples[name]
        c, kept = stream.cache_from_sequences(seqs, v, ids, hours=a.hours, concepts=concepts, seed=a.seed,
                                              derived_dir=out / f"{name}_derived", persistence=rho)
        concepts = concepts or c.concepts
        (out / name).mkdir(parents=True, exist_ok=True)
        c.save(out / name / "cache.npz")
        print(json.dumps({name: {"stays": len(c.stay_id), "decodable": len(kept), "seconds": round(time.time() - t0),
                                 "labels": _labels(c)}}), flush=True)


def bank(a) -> None:
    tokens, out = Path(a.tokens), Path(a.out)
    v, rho = _vocab(tokens)
    concepts = Cache.load(out / "real" / "cache.npz").concepts
    t0 = time.time()
    c, w = stream.cache_from_bank(Path(a.bank), v, hours=a.hours, concepts=concepts, seed=a.seed,
                                  derived_dir=out / "bank_derived", persistence=rho)
    (out / "bank").mkdir(parents=True, exist_ok=True)
    c.save(out / "bank" / "cache.npz")
    np.save(out / "bank" / "weights.npy", w)
    print(json.dumps({"bank": {"stays": len(c.stay_id), "seconds": round(time.time() - t0), "labels": _labels(c)}}),
          flush=True)


def realism(a) -> None:
    from icutwin.eval import realism as er

    out = Path(a.out)
    ref = Cache.load(out / "real" / "cache.npz")
    rep = {}
    for name in ("real2", "bank"):
        if (out / name / "cache.npz").exists():
            r = er.compare(ref, Cache.load(out / name / "cache.npz"), n_tasks=a.tasks, rows=a.rows, seed=a.seed)
            med = lambda x: round(float(np.median(x)), 3) if x else None  # noqa: E731
            for by in ("family", "target"):
                groups: dict = {}
                for t in r["per_task"]:
                    key = t[by] if by == "family" or not t["target"].startswith("concept") else "concept tasks"
                    g = groups.setdefault(key, {"tasks": 0, "prevalence_gap": [], "tstr_over_trtr": []})
                    g["tasks"] += 1
                    g["prevalence_gap"].append(t["prevalence_gap"])
                    if t["tstr_over_trtr"] is not None:
                        g["tstr_over_trtr"].append(t["tstr_over_trtr"])
                r[f"by_{by}"] = {k: {"tasks": g["tasks"], "prevalence_gap": med(g["prevalence_gap"]),
                                     "tstr_over_trtr": med(g["tstr_over_trtr"])} for k, g in sorted(groups.items())}
            rep[name] = r
            print(json.dumps({name: {k: r[k] for k in ("tasks", "summary", "by_family", "by_target")}}), flush=True)
    (out / "realism.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("real", "bank", "realism"))
    ap.add_argument("--tokens", default="data/tokens/full_r4")
    ap.add_argument("--bank", default=None)
    ap.add_argument("--counts", default=None, help="dataset -> stays (JSON), the bank's mix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=int, default=72)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=60)
    ap.add_argument("--rows", type=int, default=400)
    a = ap.parse_args()
    {"real": real, "bank": bank, "realism": realism}[a.step](a)


if __name__ == "__main__":
    main()
