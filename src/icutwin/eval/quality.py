"""Small-scale quality pilot of a candidate that passed the M7 privacy gate (plan S8 *Fidelity*): grammar-constrained
samples conditioned on each dataset (``generate.engine``), scored against held-out stays with the fidelity harness,
the diversity metrics and their controls (``eval.fidelity``, ``eval.diversity``; the copying generator and
``copy_rate`` against a pool of training stays disjoint from the positive control), per dataset and pooled; and
utility (``eval.utility``: TSTR against TRTR on 24-hour landmark tasks, pooled).

Two steps, two environments:
    vllm-env/bin/python -m icutwin.eval.quality generate --model <hf dir> --vocab <vocab.json> --out <dir> --per-dataset 1000
    .venv/bin/python -m icutwin.eval.quality evaluate --tokens <tokens root> --samples <dir> --out <report.json>
Samples are evaluation artefacts, not deliverables: they stay on the approved host next to the report, are never
shared, and are deleted with ``purge`` once the report is written.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

DATASETS = ("mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire")
README = ("EVALUATION-ONLY samples of a candidate model (quality pilot, icutwin.eval.quality). Not synthetic data "
          "deliverables: never share or train on them; deleted once the pilot report is written.\n")


def sample_seeds(seed: int, dataset: str, n: int) -> list[int]:
    """A sample's seed: the run's seed, the dataset's fixed place in ``DATASETS`` (not its place in the list a
    process was given, so processes sharing the datasets never share seeds) and its index."""
    if n > 100_000:
        raise ValueError("at most 100,000 samples per dataset")
    return [seed * 1_000_000 + DATASETS.index(dataset) * 100_000 + i for i in range(n)]


def generate(a) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from icutwin.generate.engine import Request, TwinEngine

    eng = TwinEngine(a.model, a.vocab, gpu_memory_utilization=a.gpu_memory_utilization)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.txt").write_text(README, encoding="utf-8")
    shares = json.loads(Path(a.los_bands).read_text(encoding="utf-8")) if a.los_bands else None
    calib = json.loads(Path(a.calibration).read_text(encoding="utf-8")) if a.calibration else None
    for ds in a.datasets:
        t0 = time.time()
        n = a.per_dataset if shares is None and calib is None else int(np.ceil(a.oversample * a.per_dataset))
        seeds = sample_seeds(a.seed, ds, n)
        seqs = eng.sample([Request(dataset=ds)] * n, seeds, draw_goals=a.draw_goals)
        info = {}
        if a.draw_goals:                # goal-first sampling: the goal blocks out; how the stays kept them
            from icutwin.eval.forecast import Reader
            from icutwin.generate.engine import strip_goal

            seqs, drawn = (list(x) for x in zip(*(strip_goal(s, eng.v) for s in seqs)))
            info["goals"] = goal_agreement(seqs, drawn, Reader(eng.v))
        if calib is not None:           # round 3: raked to the training stays' length and treatment margins
            from icutwin.eval.forecast import Reader
            from icutwin.generate import calibrate

            keep, info = calibrate.resample(seqs, Reader(eng.v), calib[ds], a.per_dataset,
                                            np.random.default_rng(seeds[0]))
            seqs = [seqs[i] for i in sorted(keep)]
        elif shares is not None:        # the length marginal corrected at the twin level (generate.los)
            from icutwin.eval.forecast import Reader
            from icutwin.generate import los

            rd = Reader(eng.v)
            hours = [rd.outcomes(x)["icu_los_hours"] for x in seqs]
            keep, info = los.resample(seqs, hours, shares[ds], a.per_dataset, np.random.default_rng(seeds[0]))
            seqs = [seqs[i] for i in sorted(keep)]
        pq.write_table(pa.table({"ids": pa.array(seqs, type=pa.list_(pa.int32()))}), out / f"{ds}.parquet")
        print(json.dumps({"dataset": ds, "samples": len(seqs), "ended": sum(s[-1] == eng.v.id("[EOS]") for s in seqs),
                          "tokens": sum(map(len, seqs)), "seconds": round(time.time() - t0, 1), **info}), flush=True)


def goal_agreement(seqs: list, drawn: list, rd) -> dict:
    """Goal-first sampling: how the stays kept the goals drawn for them (aggregates): per treatment and label the
    share of stays with one of its events (1 for 'yes' unless the budget forced the end, 0 for 'no'); the share
    whose ICU outcome and ICU length-of-stay band (the decoder's reading) are the drawn ones."""
    import numpy as np

    out = {"with_goals": int(sum(bool(d) for d in drawn))}
    flags = [rd.outcomes(s) for s in seqs]
    for k in ("imv", "vasopressor", "rrt"):
        for lab in ("yes", "no"):
            hits = [f[k] for f, d in zip(flags, drawn) if d.get(k) == lab]
            out[f"{k}_{lab}"] = {"n": len(hits), "realized": float(np.mean(hits)) if hits else None}
    same_outcome, in_band = [], []
    for s, f, d in zip(seqs, flags, drawn):
        if d.get("icu_outcome") not in (None, "unknown"):
            dis = next((rd.v.tokens[t] for t in s if t in rd.dis), None)
            same_outcome.append(dis == f"ICU//discharge//{d['icu_outcome']}")
        if d.get("icu_los") not in (None, "unknown"):
            lo, _, hi = d["icu_los"].partition("-")
            h = f["icu_los_hours"]
            in_band.append(h is not None and float(lo.rstrip("+")) <= h < (float(hi) if hi else np.inf))
    out["icu_outcome_kept"] = float(np.mean(same_outcome)) if same_outcome else None
    out["icu_los_in_band"] = float(np.mean(in_band)) if in_band else None
    return out


def draws(tokens: Path, samples: Path, datasets, cap: int, seed: int, pool_factor: int = 5, bare: bool = True):
    """Per dataset, in order, the sets the pilot scores (one generator of ``seed`` over the datasets in turn):
    yields (dataset, held, train, pool, synthetic) with ``n = min(samples, held-out stays, cap)`` held-out stays,
    as many training stays (the positive control), ``pool_factor`` x n further training stays (for the copy
    rate) and n samples. The authenticity audit draws the same sets (``docs/m7_authenticity.md``). With ``bare``,
    clock blocks (round 3) are removed from every set (pre-registered: the measures read the same tokens as for
    M32_main); the model checks keep them (the model and the automaton read them)."""
    import numpy as np
    import pyarrow.parquet as pq

    from icutwin.eval import privacy as pv

    from icutwin.tokenization.vocab import Vocab

    rng = np.random.default_rng(seed)
    meta = Vocab.load(Path(tokens) / "vocab.json").meta
    clock = np.array([m.get("cls") == "clock" for m in meta], dtype=bool)
    # round 3: clock blocks (spec clock) left out, so the measures read the stays' own tokens, as before the blocks
    strip = (lambda s: s[~clock[s]]) if bare and clock.any() else (lambda s: s)  # noqa: E731
    for ds in datasets:
        path = Path(samples) / f"{ds}.parquet"
        if not path.exists():
            continue
        syn = [strip(np.asarray(x, dtype=np.int32)) for x in pq.read_table(path).column("ids").to_pylist()]
        held = [strip(np.asarray(x)) for x in pv.first_windows(tokens, ds, "held_out")]
        train_all = [strip(np.asarray(x)) for x in pv.first_windows(tokens, ds, "train")]
        n = min(len(syn), len(held), cap)
        held = [held[i] for i in rng.choice(len(held), size=n, replace=False)]
        k = rng.permutation(len(train_all))
        train = [train_all[i] for i in k[:n]]                                   # the positive control
        pool = [train_all[i] for i in k[n:n + pool_factor * n]]                  # training stays for copy_rate
        syn = [syn[i] for i in rng.choice(len(syn), size=n, replace=False)]
        yield ds, held, train, pool, syn


def grammar_validity(path: Path, t) -> float:
    """The share of the twins in ``path`` the grammar accepts, each as the sampler wrote it: the measures read them
    without clock blocks (``draws``, ``bare``), but a clock grammar requires the block after a day-crossing interval."""
    import numpy as np
    import pyarrow.parquet as pq

    from icutwin.tokenization.fsm import accepts

    raw = pq.read_table(path).column("ids").to_pylist()
    return float(np.mean([accepts(t, list(s))[0] for s in raw])) if raw else float("nan")


def prevalence(twins: list, real: list, rd) -> dict:
    """Per utility task (``eval.utility``: new-onset IMV, vasopressor and RRT among stays without one before the
    24-hour landmark, death, ICU stay over 7 days), the stays it applies to, the positives and their share, for the
    twins and the real stays; and the share of stays without an ICU discharge (runaway twins). Round 3's S1."""
    import numpy as np

    from icutwin.eval import utility as ut

    out = {}
    for name, seqs in (("twins", twins), ("real", real)):
        _w, labels = ut.examples(seqs, rd)
        out[name] = {k: {"n": int(np.sum(~np.isnan(y))), "positives": int(np.nansum(y)),
                         "rate": float(np.nanmean(y)) if np.any(~np.isnan(y)) else None} for k, y in labels.items()}
        out[name]["no_icu_discharge"] = float(np.mean([rd.outcomes(s)["icu_los_hours"] is None for s in seqs]))
    return out


def evaluate(a) -> None:

    from icutwin.eval import fidelity as fd
    from icutwin.eval.forecast import Reader
    from icutwin.tokenization.fsm import compile_tables
    from icutwin.tokenization.vocab import Vocab

    tokens = Path(a.tokens)
    v = Vocab.load(tokens / "vocab.json")
    voc = fd.vocabulary(v)
    t = compile_tables(v)
    rd = Reader(v)
    report = {"datasets": {}}
    pooled_held, pooled_train, pooled_pool, pooled_syn = [], [], [], []
    for ds, held, train, pool, syn in draws(tokens, Path(a.samples), a.datasets, a.cap, a.seed, a.pool_factor):
        rep = fd.harness(held, train, voc, {"model": syn}, seed=a.seed, train=pool)
        rep["grammar_valid"] = grammar_validity(Path(a.samples) / f"{ds}.parquet", t)
        rep["prevalence"] = prevalence(syn, held, rd)
        report["datasets"][ds] = rep
        pooled_held += held
        pooled_train += train
        pooled_pool += pool
        pooled_syn += syn
        m = rep["candidates"]["model"]["metrics"]
        print(ds, json.dumps({**{k: m[k] for k in ("discriminator_auc", "code_jsd", "value_jsd", "coverage", "recall")},
                              **{k: m["diversity"].get(k) for k in ("vendi_ratio", "copy_rate", "duplicate_rate")}},
                             default=float), "flags", rep["candidates"]["model"]["flags"], flush=True)
    report["pooled"] = fd.harness(pooled_held, pooled_train, voc, {"model": pooled_syn}, seed=a.seed, train=pooled_pool)
    report["pooled_prevalence"] = prevalence(pooled_syn, pooled_held, rd)
    from icutwin.eval import utility as ut

    report["utility"] = ut.tstr(pooled_train, pooled_held, {"model": pooled_syn}, v, seed=a.seed)
    print("utility", json.dumps({t: [report["utility"]["trtr"][t]["auroc"], x["auroc"], x["utility_ratio"]]
                                 for t, x in report["utility"]["sets"]["model"].items()}, default=float), flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")


def purge(a) -> None:
    shutil.rmtree(a.samples, ignore_errors=True)
    print("deleted", a.samples)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--vocab", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--per-dataset", type=int, default=1000)
    g.add_argument("--datasets", nargs="+", default=list(DATASETS))
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    g.add_argument("--los-bands", default=None, help="los_bands.json: resample the twins to these length shares")
    g.add_argument("--calibration", default=None,
                   help="calibration.json: rake the twins to these length and treatment margins (over --los-bands)")
    g.add_argument("--oversample", type=float, default=2.0)
    g.add_argument("--draw-goals", action="store_true",
                   help="goal-first sampling: the model draws the goal block, which binds the stay (stripped after)")
    e = sub.add_parser("evaluate")
    e.add_argument("--tokens", required=True)
    e.add_argument("--samples", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--datasets", nargs="+", default=list(DATASETS))
    e.add_argument("--cap", type=int, default=1000)
    e.add_argument("--pool-factor", type=int, default=5)
    e.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("purge")
    p.add_argument("--samples", required=True)
    a = ap.parse_args(argv)
    {"generate": generate, "evaluate": evaluate, "purge": purge}[a.cmd](a)


if __name__ == "__main__":
    main()
