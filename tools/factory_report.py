"""The factory report (plan Phase F, "Acceptance"; S4 measures, S9 accepts) on the host: episodes of the twin store and
its pool in ``--workers`` processes, then every gate's measure, each with its real-vs-real control where one applies.
Aggregates only.

    python tools/factory_report.py --store data/factory/stores/bank_dev --pool data/factory/pools/bank_dev \\
        --real data/factory/stores/real_dev --control data/factory/stores/real_control --n 20000 --workers 12 \\
        --out runs/factory/report_s4.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_S: dict = {}


def _sampler(store: str, pool_dir: str | None, vocab: str, seed: int, split: str):
    key = (store, pool_dir, seed, split)
    if _S.get("key") != key:
        from icutwin.factory import episodes, pool
        from icutwin.factory import store as fst
        from icutwin.tokenization.vocab import Vocab

        st = fst.Store(Path(store))
        progs, weights, support = pool.load(Path(pool_dir)) if pool_dir else (None, None, None)
        _S.update(key=key, st=st, s=episodes.Sampler(st, Vocab.load(Path(vocab)), seed=seed, programs=progs,
                                                       weights=weights, targets=pool.load_targets(), split=split,
                                                       support=support))
    return _S["st"], _S["s"]


def _real(real: str, vocab: str, seed: int):
    if _S.get("real_key") != (real, seed):
        from icutwin.factory import episodes
        from icutwin.factory import store as fst
        from icutwin.tokenization.vocab import Vocab

        _S.update(real_key=(real, seed), real=episodes.Sampler(fst.Store(Path(real)), Vocab.load(Path(vocab)),
                                                               seed=seed, split=None))
    return _S["real"]


def _job(args) -> list[dict]:
    store, pool_dir, vocab, seed, split, idx, every, real = args
    from icutwin.factory import contracts, report
    from icutwin.factory.episodes import NoEpisode

    st, s = _sampler(store, pool_dir, vocab, seed, split)
    out = []
    for i in idx:
        t0 = time.time()
        try:
            ep = s.episode(i)
        except NoEpisode:
            out.append({"index": i, "no_episode": True})
            continue
        sec = time.time() - t0
        row = report.summarize(ep, st)
        row["seconds"] = sec
        row["fails"] = contracts.episode_failures(ep)
        if (ep["recipe"].get("site_variant") or {}).get("kind") == "cross_site" and i % max(every // 2, 1) == 0:
            row["shift"] = report.shift_loss(s, ep)
        if i % every == 0:
            row["probe"] = report.difficulty(s, ep)
            row["context_vendi"] = report.context_vendi(st, [s.index[x] for x in ep["recipe"]["context"]]) \
                if len(ep["recipe"]["context"]) > 1 else None
            if real:
                row["context_vendi_ratio"] = report.context_vendi_ratio(s, _real(real, vocab, seed), ep, seed)
        out.append(row)
    return out


def run(store, pool_dir, vocab, seed, split, n, workers, every, real=None) -> tuple[list, float]:
    chunk = 50
    jobs = [(store, pool_dir, vocab, seed, split, list(range(k, min(k + chunk, n))), every, real)
            for k in range(0, n, chunk)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(_job, jobs) for r in part]
    return rows, time.time() - t0


def catalog_alignment(stores: dict, per_stay: int = 12, seed: int = 0) -> dict:
    """Per catalog program: prevalence (or mean) on twins, real dev and the real control; the share of clinical
    programs whose twin-to-dev gap is within 1.5 times the control-to-dev gap (or within 0.005 absolute)."""
    from icutwin.factory import programs as fpr

    sys_path = Path(__file__).resolve().parent
    import sys

    sys.path.insert(0, str(sys_path))
    from factory_catalog import rows_for, summary

    out = {}
    for name, st in stores.items():
        rows = rows_for(st, per_stay, seed)
        ev = fpr.Evaluator(st)
        for p in fpr.catalog(st.channels):
            b = ev.evaluate(p, rows)
            s = summary(p, b, rows)
            x = s.get("prevalence", s.get("mean", s.get("events")))
            out.setdefault(f"{p.name}@{p.horizon}", {"family": p.family})[name] = x
    ok, n = 0, 0
    for k, r in out.items():
        have = all(r.get(x) is not None for x in ("twins", "dev", "control"))
        if r["family"] in ("F1", "F2", "F3", "F4", "F5") and have:
            n += 1
            gap, ctrl = abs(r["twins"] - r["dev"]), abs(r["control"] - r["dev"])
            r["within"] = bool(gap <= max(1.5 * ctrl, 0.005 if r["dev"] < 1 else 0.05 * abs(r["dev"])))
            ok += r["within"]
    return {"programs": out, "clinical_within_band": ok / max(n, 1), "clinical_programs": n}


def base_rate_coverage(rows: list[dict], alignment: dict, catalog: list, real_store=None) -> dict:
    """Whether episodes vary each clinical program's base rate beyond the twin-to-real gap (the condition under which a
    prevalence mismatch between twins and real stays does not limit what an in-context learner learns: context and
    queries share their world in every episode, and the model must read the base rate from its context). Per binary
    catalog program, over its rolling-cadence episodes: the context's prevalence at the 10th, 50th and 90th percentile,
    the real and the twin rate (``catalog_alignment``), whether the real rate lies inside the episodes' p10-p90, and
    the spread p90 / p10 against the gap max(real, twins) / min(real, twins). With ``real_store`` the real rate is
    the one a random real context shows (every hour of every stay, as a context's label timeline counts them; the
    alignment's rows sample each stay equally). Also per program: the share of contexts without a case, and the
    prevalence quantiles of contexts taken as drawn (descriptor shown, natural selection: the bedside regime)."""
    key = {p.hash(): f"{p.name}@{p.horizon}" for p in catalog if p.label in ("binary", "hosp_death")}
    hourly = {}
    if real_store is not None:
        from icutwin.factory import labels as fl
        from icutwin.factory import programs as fpr

        ev, hrs = fpr.Evaluator(real_store), fl.rolling(real_store, 60)
        for p in catalog:
            k = key.get(p.hash())
            if k:
                blk = ev.evaluate(p, hrs)
                hourly[k] = float((blk.y[blk.valid] > 0).mean()) if blk.valid.any() else None
    by: dict = {}
    drawn: dict = {}
    for r in rows:
        k = key.get(r["hash"])
        if k and r["cadence"] == "rolling" and r.get("prevalence") is not None:
            by.setdefault(k, []).append(r["prevalence"])
            if r.get("as_drawn"):
                drawn.setdefault(k, []).append(r["prevalence"])
    out = {}
    for k, v in sorted(by.items()):
        a = alignment["programs"].get(k, {})
        real, twins = (hourly.get(k) if real_store is not None else a.get("dev")), a.get("twins")
        if real_store is not None and a.get("dev") and real is not None:
            twins = twins * real / a["dev"] if twins is not None else None      # the same gap, hour-weighted
        if real is None or twins is None or len(v) < 10 or min(real, twins) <= 0:
            continue
        lo, med, hi = (float(x) for x in np.quantile(v, [0.1, 0.5, 0.9]))
        out[k] = {"episodes": len(v), "context_p10_p50_p90": [round(lo, 4), round(med, 4), round(hi, 4)],
                  "real": round(real, 4), "twins": round(twins, 4), "real_within": bool(lo <= real <= hi),
                  "spread": round(hi / lo, 2) if lo > 0 else None,
                  "gap": round(max(real, twins) / min(real, twins), 2),
                  "no_case": round(float(np.mean(np.asarray(v) == 0)), 4)}
        d = drawn.get(k, [])
        if len(d) >= 10:
            dl, dm, dh = (float(x) for x in np.quantile(d, [0.1, 0.5, 0.9]))
            out[k]["as_drawn_p10_p50_p90"] = [round(dl, 4), round(dm, 4), round(dh, 4)]
            out[k]["real_within_as_drawn"] = bool(dl <= real <= dh)
    ok = [r["real_within"] for r in out.values()]
    wide = [r["spread"] is None or r["spread"] >= r["gap"] for r in out.values()]
    ok_d = [r["real_within_as_drawn"] for r in out.values() if "real_within_as_drawn" in r]
    return {"programs": out, "share_real_within": float(np.mean(ok)) if ok else None,
            "share_spread_beyond_gap": float(np.mean(wide)) if wide else None, "scored": len(out),
            "share_real_within_as_drawn": float(np.mean(ok_d)) if ok_d else None, "scored_as_drawn": len(ok_d)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("store", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--real", default=None)
    ap.add_argument("--control", default=None)
    ap.add_argument("--vocab", default="data/tokens/full_r4/vocab.json")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--every", type=int, default=10, help="probe and context Vendi every k-th episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train")
    a = ap.parse_args(argv)
    from icutwin.factory import episodes, pool, report
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst

    cfg, targets = episodes.load_config(), pool.load_targets()
    rows, wall = run(a.store, a.pool, a.vocab, a.seed, a.split, a.n, a.workers, a.every, a.real)
    missing = sum(bool(r.get("no_episode")) for r in rows)
    rows = [r for r in rows if not r.get("no_episode")]
    st = fst.Store(Path(a.store))
    prev_by_hash = {}
    if a.pool:
        import pandas as pd

        pdf = pd.read_parquet(Path(a.pool) / "pool.parquet", columns=["hash", "prevalence_bin"])
        prev_by_hash = dict(zip(pdf["hash"], pdf["prevalence_bin"], strict=True))
    rep = {"episodes": len(rows), "no_episode": missing, "wall_s": round(wall, 1),
           "episodes_per_s": round(len(rows) / wall, 2),
           "contract_failures": sum(bool(r["fails"]) for r in rows),
           "mix": report.mix_report(rows, cfg),
           "distributions": report.distribution_report(rows, targets, prev_by_hash),
           "programs": report.program_report(rows, {p.hash() for p in fpr.catalog(st.channels)}),
           "reuse": report.reuse_report(rows, st, a.split),
           "probe_deciles": {kind: np.quantile(v, np.linspace(0.1, 0.9, 9)).round(3).tolist()
                             for kind in ("auroc", "r2")
                             if (v := [r["probe"] for r in rows if r.get("probe") is not None
                                       and r["probe_kind"] == kind])},
           "probe_by_family": {f"{f}:{kind}": round(float(np.median(v)), 3)
                               for f in sorted({r["family"] for r in rows}) for kind in ("auroc", "r2")
                               if (v := [r["probe"] for r in rows if r.get("probe") is not None
                                         and r["family"] == f and r["probe_kind"] == kind])},
           "shift_loss_p10_p50_p90": (np.quantile([r["shift"]["loss"] for r in rows if r.get("shift")],
                                                  [0.1, 0.5, 0.9]).round(3).tolist()
                                      if any(r.get("shift") for r in rows) else None),
           "shift_episodes_scored": sum(bool(r.get("shift")) for r in rows),
           "censoring_deciles": np.quantile([r["censored"] for r in rows if r.get("censored") is not None],
                                            np.linspace(0.1, 0.9, 9)).round(3).tolist(),
           "context_size_p5_50_95": np.quantile([r["context"] for r in rows], [0.05, 0.5, 0.95]).tolist(),
           "events_p50_p95": np.quantile([r["events"] for r in rows], [0.5, 0.95]).tolist(),
           "thinned_share": float(np.mean([r["thinned"] > 0 for r in rows])),      # episodes with any
           "thinned_trajectories": float(sum(r["thinned_trajectories"] for r in rows)
                                         / max(sum(r["context"] for r in rows), 1)),    # context trajectories
           "thinned_queries": float(sum(r.get("thinned_queries", 0) for r in rows)
                                    / max(sum(r["queries"] for r in rows), 1)),         # query trajectories (QA #129)
           "difficulty_loss_p50_p90": (np.quantile([r["difficulty_loss"] for r in rows
                                                    if r.get("difficulty_loss") is not None], [0.5, 0.9]).tolist()
                                       if any(r.get("difficulty_loss") is not None for r in rows) else None),
           "icu_day_of_predictions": report.shares(min(int(t // 1440), 7) for r in rows for t in r["times"])}
    hashes = [r["hash"] for r in rows if r["family"] in ("F6", "FH", "F7", "FC")]
    progs, _w = pool.load_programs(__import__("pandas").read_parquet(Path(a.pool) / "pool.parquet")) if a.pool else \
        (fpr.catalog(st.channels), None)
    by_hash = {p.hash(): p for p in progs}
    rng = np.random.default_rng(a.seed)
    sample = [by_hash[h] for h in dict.fromkeys(hashes) if h in by_hash and by_hash[h].label == "binary"]
    sample = [sample[i] for i in rng.permutation(len(sample))[:400]]
    rep["label_functions"] = report.label_functions(st, sample + [p for p in fpr.catalog(st.channels)
                                                                 if p.label == "binary"])
    vend = [r["context_vendi"] for r in rows if r.get("context_vendi") is not None]
    rep["context_vendi_mean"] = float(np.mean(vend)) if vend else None
    ratio = [r["context_vendi_ratio"] for r in rows if r.get("context_vendi_ratio") is not None]
    rep["context_vendi_ratio_p10_p50_p90"] = np.quantile(ratio, [0.1, 0.5, 0.9]).round(3).tolist() if ratio else None
    if a.real:
        real = fst.Store(Path(a.real))
        rs = np.random.default_rng(1)
        rr = rs.choice(len(real), size=min(3000, len(real)), replace=False)
        tt = rs.choice(len(st), size=min(3000, len(st)), replace=False)
        rep["coverage_of_real"] = report.coverage(real, rr, st, tt)
        if a.control:
            ctrl = fst.Store(Path(a.control))
            cc = rs.choice(len(ctrl), size=min(3000, len(ctrl)), replace=False)
            rep["coverage_control"] = report.coverage(real, rr, ctrl, cc)
            rep["alignment"] = catalog_alignment({"twins": st, "dev": real, "control": ctrl})
            rep["base_rate_coverage"] = base_rate_coverage(rows, rep["alignment"], fpr.catalog(st.channels), real)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    brief = {k: rep[k] for k in ("episodes", "no_episode", "episodes_per_s", "contract_failures", "programs",
                                 "probe_deciles",
                                 "label_functions", "context_vendi_mean", "context_vendi_ratio_p10_p50_p90",
                                 "shift_loss_p10_p50_p90") if k in rep}
    brief["groups"] = rep["mix"]["groups"]
    brief["reuse"] = {k: rep["reuse"][k] for k in ("gini", "max_over_mean", "strata_reached")}
    for k in ("coverage_of_real", "coverage_control"):
        if k in rep:
            brief[k] = rep[k]
    if "alignment" in rep:
        brief["alignment"] = {k: rep["alignment"][k] for k in ("clinical_within_band", "clinical_programs")}
    print(json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    main()
