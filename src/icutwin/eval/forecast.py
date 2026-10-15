"""Twin forecasting (plan S8 *Forecasting*, the CoMET protocol): held-out stays as known at a cutoff (prospective
prefixes: ``tokenization.encode.prospective``) are continued ``n`` times by the twin engine; an outcome's predicted
probability is its share among the rollouts, scored against what happened in the real stay (AUROC, AUPRC, expected
calibration error, observed / expected).

Outcomes, read the same way off a rollout and off the real stay's whole sequence:
* ``death``: in-hospital death (``meds.outcomes``: the 'expired' discharge in the end block or an ICU discharge
  'dead'; None when the stay has no end block or records neither outcome);
* ``imv`` / ``vasopressor`` / ``rrt``: such an event (``generate.engine.obligations``' sets) after the cutoff;
* ``icu_los_gt_7d``: ICU discharge more than 7 days after the ICU admission marker (interval tokens' median gaps).

By horizon (round 3, ``horizons``): the treatment events, the ICU discharge and the ICU discharge 'dead' within 6,
24 and 72 hours of the cutoff (``Reader.within``), so an error that grows as a rollout runs on shows.

Dispersion (plan *Diversity assurance* 6, forecast row) of the one continuous outcome, the ICU length of stay in
hours (``icu_los_hours``; a rollout that does not reach an ICU discharge is censored and left out, its share
reported): the randomized probability integral transform of the real value within each stay's rollouts (uniform for
a calibrated forecaster; ``pit_ks``: its Kolmogorov-Smirnov distance from uniform; ``pit_deciles``), the ensemble
CRPS, and spread against skill (mean ensemble standard deviation over the RMSE of the ensemble mean: about 1 when
calibrated, below 1 for an under-dispersed forecaster, above 1 for an over-dispersed one).
"""

from __future__ import annotations

import numpy as np

from icutwin.meds.outcomes import TokenOutcome
from icutwin.tokenization.vocab import Vocab

OUTCOMES = ("death", "imv", "vasopressor", "rrt", "icu_los_gt_7d")
DATASETS = ("mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire")


class Reader:
    """Outcome flags of token sequences (whole stays, or a prefix and its continuation)."""

    def __init__(self, v: Vocab):
        from icutwin.generate.engine import obligations

        self.v = v
        req, _end, _forbid = obligations(v, {"imv": "yes", "vasopressor": "yes", "rrt": "yes"})
        self.sets = {"imv": req[0], "vasopressor": req[1], "rrt": req[2]}
        self.end = v.id("[END]")
        self.outcome = TokenOutcome(v)
        self.adm = {i for i, m in enumerate(v.meta) if (m.get("code") or "") == "ICU//admission"}
        self.dis = {i for i, m in enumerate(v.meta) if (m.get("code") or "").startswith("ICU//discharge//")}
        sk = v.interval_sketches["*"]
        self.minutes = {i: (sk[m["interval"]][len(sk[m["interval"]]) // 2] if sk[m["interval"]] else 0.0)
                        for i, m in enumerate(v.meta) if m.get("cls") == "interval"}

    def within(self, ids: list[int], after: int, hours: float) -> dict:
        """Flags of the ``hours`` after position ``after`` (round 3's horizons: does a rollout's error grow with
        time?): an IMV, vasopressor or RRT event, an ICU discharge and an ICU discharge 'dead' among the tokens
        less than ``hours`` after it (interval tokens at their median gaps)."""
        ids, t, stop = list(ids), 0.0, len(ids)
        for k in range(after, len(ids)):
            if t >= hours * 60:
                stop = k
                break
            t += self.minutes.get(ids[k], 0.0)
        tail = set(ids[after:stop])
        out = {k: bool(s & tail) for k, s in self.sets.items()}
        out["icu_discharge"] = bool(self.dis & tail)
        out["icu_death"] = any((self.v.meta[x].get("code") or "").endswith("//dead") for x in self.dis & tail)
        return out

    def outcomes(self, ids: list[int], after: int = 0) -> dict:
        """Flags of a sequence; events count from position ``after`` (the prefix's length)."""
        ids = list(ids)
        tail = set(ids[after:])
        at = ids.index(self.end) if self.end in ids else None
        out = {"death": self.outcome.death(ids)}
        for k, s in self.sets.items():
            out[k] = bool(s & tail)
        t, start, los = 0.0, None, None
        for tok in ids[:at] if at is not None else ids:
            t += self.minutes.get(tok, 0.0)
            if start is None and tok in self.adm:
                start = t
            elif start is not None and tok in self.dis:
                los = t - start
                break
        out["icu_los_gt_7d"] = None if los is None else bool(los > 7 * 24 * 60)
        out["icu_los_hours"] = None if los is None else los / 60
        return out


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        k = idx == b
        if k.any():
            err += k.mean() * abs(y[k].mean() - p[k].mean())
    return float(err)


def score(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    ok = ~np.isnan(y)
    y, p = y[ok], p[ok]
    out = {"stays": int(len(y)), "prevalence": float(y.mean()) if len(y) else None,
           "mean_predicted": float(p.mean()) if len(p) else None}
    if len(np.unique(y)) == 2:
        out.update(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)), ece=ece(y, p),
                   observed_over_expected=float(y.mean() / max(p.mean(), 1e-9)))
    return out


def crps_ensemble(x: np.ndarray, y: float) -> float:
    """CRPS of an ensemble ``x`` against ``y``: E|X - y| - E|X - X'| / 2 (the ensemble's own distribution)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    pairs = np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2        # E|X - X'| / 2 over ordered pairs
    return float(np.mean(np.abs(x - y)) - pairs)


def dispersion(truth: list[float | None], ensembles: list[list[float]], seed: int = 0) -> dict:
    """PIT, CRPS and spread-skill of a continuous outcome (module docstring); stays with a real value and at least
    two rollouts that resolve it."""
    from scipy.stats import kstest

    rng = np.random.default_rng(seed)
    pit, crps, sd, err = [], [], [], []
    for y, xs in zip(truth, ensembles):
        if y is None or len(xs) < 2:
            continue
        x = np.asarray(xs, dtype=np.float64)
        below, ties = np.sum(x < y), np.sum(x == y)
        pit.append((below + rng.uniform() * (ties + 1)) / (len(x) + 1))
        crps.append(crps_ensemble(x, y))
        sd.append(x.std(ddof=1))
        err.append(x.mean() - y)
    if not pit:
        return {"stays": 0}
    pit = np.array(pit)
    rmse = float(np.sqrt(np.mean(np.square(err))))
    return {"stays": len(pit), "pit_ks": float(kstest(pit, "uniform").statistic),
            "pit_deciles": np.histogram(pit, bins=10, range=(0, 1))[0].tolist(), "crps": float(np.mean(crps)),
            "spread_skill": float(np.mean(sd) / rmse) if rmse > 0 else None}


HORIZONS = (6, 24, 72)
HORIZON_OUTCOMES = ("imv", "vasopressor", "rrt", "icu_discharge", "icu_death")


def evaluate(truth: list[dict], rollouts: list[list[dict]], outcomes: tuple = OUTCOMES) -> dict:
    """Per outcome, the rollout share against the real stay's flag (stays with a known flag and rollouts that
    resolve it); with the whole-stay outcomes, the ICU length of stay's dispersion, with the share of rollouts
    censored before an ICU discharge."""
    out = {}
    for k in outcomes:
        y, p = [], []
        for t, rs in zip(truth, rollouts):
            known = [r[k] for r in rs if r[k] is not None]
            if t[k] is None or not known:
                continue
            y.append(float(t[k]))
            p.append(float(np.mean(known)))
        out[k] = score(np.array(y), np.array(p))
    if "icu_los_gt_7d" not in outcomes:
        return out
    ens = [[r["icu_los_hours"] for r in rs if r.get("icu_los_hours") is not None] for rs in rollouts]
    out["icu_los_hours"] = dispersion([t.get("icu_los_hours") for t in truth], ens)
    total = sum(len(rs) for rs in rollouts)
    out["icu_los_hours"]["rollouts_censored"] = float(1 - sum(map(len, ens)) / total) if total else None
    return out


# -- runner (the server): rollouts in vLLM's environment, scores in either ----------------------------------------
def rollouts(a) -> None:
    """Prospective prefixes of held-out stays still in the ICU at the cutoff, ``n`` rollouts each; writes
    ``<out>/<dataset>.parquet`` (stay_id, prefix length, rollout index, continuation)."""
    import json
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    from icutwin.generate.engine import TwinEngine
    from icutwin.tokenization.encode import prospective
    from icutwin.tokenization.stays import iter_stays

    eng = TwinEngine(a.model, a.vocab, gpu_memory_utilization=a.gpu_memory_utilization)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for ds in a.datasets:
        stays = [s for s in iter_stays(Path(a.meds) / ds, "held_out", Path(a.derived) / ds)
                 if (s.icu_out - s.icu_in).total_seconds() > a.cutoff * 3600]
        stays = [stays[i] for i in np.sort(rng.choice(len(stays), size=min(a.stays, len(stays)), replace=False))]
        prefixes = [prospective(s, eng.v, a.cutoff) for s in stays]
        # each dataset its own seeds, by its fixed place in DATASETS (processes may share the datasets)
        conts = eng.simulate(prefixes, a.n, seed=100 * a.seed + DATASETS.index(ds))
        rows = [(s.stay_id, len(p), j, c) for s, p, cs in zip(stays, prefixes, conts) for j, c in enumerate(cs)]
        pq.write_table(pa.table({"stay_id": [r[0] for r in rows], "prefix_len": [r[1] for r in rows],
                                 "rollout": [r[2] for r in rows], "prefix": [list(map(int, prefixes[i // a.n])) for i in range(len(rows))],
                                 "continuation": pa.array([r[3] for r in rows], type=pa.list_(pa.int32()))}),
                       out / f"{ds}.parquet")
        print(json.dumps({"dataset": ds, "stays": len(stays), "rollouts": len(rows)}), flush=True)


def score_dir(a) -> None:
    import json
    from pathlib import Path

    import pyarrow.parquet as pq

    from icutwin.tokenization.vocab import Vocab

    tokens = Path(a.tokens)
    v = Vocab.load(tokens / "vocab.json")
    rd = Reader(v)
    report = {}
    for ds in a.datasets:
        p = Path(a.rollouts) / f"{ds}.parquet"
        if not p.exists():
            continue
        real = {r["stay_id"]: r["ids"] for r in
                pq.read_table(tokens / ds / "held_out.parquet", columns=["stay_id", "ids"]).to_pylist()}
        by_stay: dict = {}
        by_h: dict = {h: {} for h in HORIZONS}
        for r in pq.read_table(p).to_pylist():
            seq = r["prefix"] + r["continuation"]
            by_stay.setdefault(r["stay_id"], []).append(rd.outcomes(seq, after=r["prefix_len"]))
            for h in HORIZONS:
                by_h[h].setdefault(r["stay_id"], []).append(rd.within(seq, r["prefix_len"], h))
        sids = [s for s in by_stay if s in real]
        # truth: the real stay's flags, its events counted after the prefix as the rollouts' are (the prefix is the
        # stay's first tokens up to the cutoff, so the same position)
        plen = {r["stay_id"]: r["prefix_len"] for r in pq.read_table(p, columns=["stay_id", "prefix_len"]).to_pylist()}
        truth = [rd.outcomes(real[s], after=plen[s]) for s in sids]
        report[ds] = evaluate(truth, [by_stay[s] for s in sids])
        report[ds]["horizons"] = {str(h): evaluate([rd.within(real[s], plen[s], h) for s in sids],
                                                   [by_h[h][s] for s in sids], HORIZON_OUTCOMES) for h in HORIZONS}
        print(ds, json.dumps({k: {m: round(x, 3) for m, x in d.items() if isinstance(x, float)} for k, d in report[ds].items()}),
              flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("rollouts")
    for k in ("model", "vocab", "meds", "derived", "out"):
        g.add_argument(f"--{k}", required=True)
    g.add_argument("--datasets", nargs="+", default=list(DATASETS))
    g.add_argument("--stays", type=int, default=300)
    g.add_argument("--n", type=int, default=32)
    g.add_argument("--cutoff", type=float, default=24.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    s = sub.add_parser("score")
    for k in ("tokens", "rollouts", "out"):
        s.add_argument(f"--{k}", required=True)
    s.add_argument("--datasets", nargs="+", default=list(DATASETS))
    a = ap.parse_args(argv)
    {"rollouts": rollouts, "score": score_dir}[a.cmd](a)


if __name__ == "__main__":
    main()
