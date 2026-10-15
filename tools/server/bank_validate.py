"""icutwin on the AutoDL host: checks of a twin bank (aggregates only): the manifests (kept and dropped per dataset),
grammar validity of every twin, a sample decoded and written as MEDS (layer b: schema, order, codes), and the ICL
factory's smoke on that sample (``factory.stream``: the cache with derived concepts through canonical CLIF, label
coverage per dataset, task prior, TabICL batch contract).
    python bank_validate.py <bank dir> <tokens root> [--sample 300]"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bank")
    ap.add_argument("tokens")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--meds", default=None, help="the real MEDS root (its codes complete the reference)")
    ap.add_argument("--no-factory", action="store_true", help="skip the factory smoke")
    a = ap.parse_args()

    from icutwin.factory import contracts as fct
    from icutwin.factory import tasks as ft
    from icutwin.generate import meds_out as mo
    from icutwin.tokenization.decode import decode
    from icutwin.tokenization.fsm import accepts, compile_tables
    from icutwin.tokenization.vocab import Vocab

    bank, tokens = Path(a.bank), Path(a.tokens)
    v = Vocab.load(tokens / "vocab.json")
    t = compile_tables(v)
    kept, dropped, rows = Counter(), Counter(), []
    for man_path in sorted(bank.glob("part*/manifest.json")):
        man = json.loads(man_path.read_text(encoding="utf-8"))
        for s in man["shards"]:
            kept[s["dataset"]] += s["kept"]
            for k, n in s["dropped"].items():
                dropped[f"{s['dataset']}:{k}"] += n
        for f in sorted(man_path.parent.glob("*.parquet")):
            rows += pq.read_table(f, columns=["twin", "dataset", "ids"]).to_pylist()
    print(json.dumps({"twins": len(rows), "kept": dict(kept), "dropped": dict(dropped)}), flush=True)
    valid = Counter()
    for r in rows:
        valid[(r["dataset"], bool(accepts(t, r["ids"])[0]))] += 1
    print(json.dumps({"grammar_valid": {ds: valid[(ds, True)] / max(valid[(ds, True)] + valid[(ds, False)], 1)
                                        for ds in sorted(kept)}}), flush=True)
    rng = np.random.default_rng(0)
    pick = [rows[i] for i in sorted(rng.choice(len(rows), size=min(a.sample * len(kept), len(rows)), replace=False))]
    stays = [decode(r["ids"], v, np.random.default_rng(k)) for k, r in enumerate(pick)]
    meds = Path(a.meds) if a.meds else None       # the real MEDS codes (static and header codes live there)
    known = mo.known_codes(meds_root=meds, v=v) | {"ICU//admission//UNK"}
    rep = mo.validate(mo.table(stays, list(range(len(stays)))), known)
    print(json.dumps({"meds_layer_b": {k: rep[k] for k in ("passed", "subjects") if k in rep},
                      "failures": rep.get("failures", [])[:5]}), flush=True)
    if a.no_factory:
        return
    import tempfile

    from icutwin.factory import stream as fs
    from icutwin.tokenization.decode import load_persistence

    rho = tokens / "rank_persistence.json"
    with tempfile.TemporaryDirectory(dir=bank) as work:      # derived concepts of the sample (factory.stream)
        cache, _kept = fs.cache_from_sequences([r["ids"] for r in pick], v, [r["twin"] for r in pick], hours=72,
                                               seed=0, derived_dir=Path(work),
                                               persistence=load_persistence(rho, v) if rho.exists() else None)
    labels = {ds: {"stays": int((cache.dataset == ds).sum()),
                   **{e: round(float(np.mean(~np.isnan(cache.onset[e][cache.dataset == ds]))), 3) for e in cache.onset},
                   "sofa_24h_known": round(float(np.mean(~np.isnan(cache.sofa_24h[cache.dataset == ds]))), 3),
                   "sofa_24h_mean": round(float(np.nanmean(cache.sofa_24h[cache.dataset == ds])), 2)}
              for ds in sorted(set(cache.dataset))}
    print(json.dumps({"factory_labels": labels}), flush=True)
    prior = ft.ICUPrior(cache, min_rows=48, max_rows=300, seed=0)
    x, y, d, seq, train = (b.numpy() for b in prior.get_batch(8))
    fam = [s.family for s in prior.last_specs]
    print(json.dumps({"cache_stays": len(cache.stay_id), "grid": list(cache.grid.shape),
                      "batch": [list(x.shape), list(y.shape)], "families": fam,
                      "contract_failures": fct.batch_failures(x, y, d, seq, train, fam),
                      "prior_stats": dict(prior.stats)}, default=str), flush=True)


if __name__ == "__main__":
    main()
