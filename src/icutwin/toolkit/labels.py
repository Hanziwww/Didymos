"""Labels of rollouts with the factory's label engine, as Paper P's A1 read them (``tools/paper_a1.py``).

Each rollout (prompt and continuation) is decoded on its anchor's clock (the stay's id: ``clock_id``) into a rollout
store with derived concepts (``factory.store.build``: CLIF, then ``meds.derived``); with the patient's whole sequence
at hand, an alignment row (``<anchor>#real``) is decoded the same way and carries the patient's own label (the
twin's own future, or a held-out stay's). For a prompt with its dataset and site set to unknown, the real tokens go
back before decoding (the decoder writes the dataset; the conditioning was the point). At each anchor's prediction
time ``t_min``, per catalog program (``factory.programs``):

* binary: the share of positive rollouts among the valid ones (at least ``min_valid``, half the rollouts by default);
* regression: the valid rollouts' ensemble (10th, 50th and 90th percentiles, mean);
* time to event: the share with the event within 48 h;
* the remaining ICU stay: Kaplan-Meier over the rollouts (a capped rollout censored at its cap): P(over 7 days) and the
  median.

``applies``: the programs a design scores (A1's rule: landmarks every program but event-triggered ones; otherwise
binary programs within 48 h, times to event and regressions within 48 h). ``token_shares`` is the cheap reading
without the label engine (``eval.forecast.Reader``: events in the hours after the prompt, interval tokens at their
median gaps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def applies(p, design: str) -> bool:
    """The programs a design scores (``tools/paper_a1.py``)."""
    if p.name.startswith("extubation_failure"):
        return False
    if design in ("L24", "L48"):
        return True
    if p.label == "binary":
        return 0 < p.horizon <= 2880
    if p.label == "tte":
        return True
    if p.label == "regression":
        return p.name not in ("remaining_los",)
    return False


def key(p) -> str:
    return f"{p.name}@{p.horizon}"


@dataclass
class RolloutSet:
    """One anchor's rollouts for the label engine."""
    anchor: str
    clock_id: str
    t_min: int
    prompt: np.ndarray
    conts: list
    weight: float = 1.0
    real_ids: list | None = None          # the patient's whole sequence: the alignment row
    restore: dict = field(default_factory=dict)   # position -> real token (a prompt's unknown dataset and site)


def build_store(sets: list, v_path: Path, work: Path, persistence: Path | None = None, workers: int = 2,
                chunk: int = 100, segment: int = 15000):
    """The rollout store of ``sets`` (resumable by segment)."""
    import json

    from icutwin.factory import store as fst

    rows = []
    for s in sets:
        for k, c in enumerate(s.conts):
            ids = np.concatenate([np.asarray(s.prompt, np.int32), np.asarray(c, np.int32)])
            for pos, tok in s.restore.items():
                ids[int(pos)] = int(tok)
            rows.append({"stay_id": f"{s.anchor}#{k}", "ids": ids, "clock": s.clock_id, "split": "rollout",
                         "weight": float(s.weight), "cell": "rollout"})
        if s.real_ids is not None:
            rows.append({"stay_id": f"{s.anchor}#real", "ids": np.asarray(s.real_ids, np.int32), "clock": s.clock_id,
                         "split": "control", "weight": 1.0, "cell": "control"})
    work = Path(work)
    done = set()
    if (work / "manifest.json").exists():
        done = {x["name"] for x in json.loads((work / "manifest.json").read_text())["segments"]}
    for k in range(-(-len(rows) // segment)):
        name = f"seg_{k:04d}"
        if name not in done:
            fst.build(rows[k * segment:(k + 1) * segment], v_path, work, name, {"kind": "toolkit", "part": k},
                      persistence=persistence, workers=workers, chunk=chunk)
    return fst.Store(work)


def km_quantile(times: np.ndarray, events: np.ndarray, q: float = 0.5) -> float:
    """The Kaplan-Meier ``q`` quantile of the durations (inf when the survival never falls to 1 - q)."""
    times, events = np.asarray(times, float), np.asarray(events, bool)
    ok = np.isfinite(times)
    times, events = times[ok], events[ok]
    s = 1.0
    for u in np.unique(times[events]):
        at = (times >= u).sum()
        if at:
            s *= 1.0 - ((times == u) & events).sum() / at
        if s <= 1 - q:
            return float(u)
    return float("inf")


def readout(store, sets: list, design: str = "R", min_valid: int | None = None, programs=None) -> dict:
    """anchor -> {"binary", "regression", "tte": {program: ...}, "remaining_los": ...} (module docstring); each with
    ``observed`` (the alignment row's label) when the set has one."""
    from icutwin.eval import bench
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr

    idx = {s: i for i, s in enumerate(store.stays["stay_id"].tolist())}
    n_roll = max(len(s.conts) for s in sets)
    st = np.array([[idx.get(f"{s.anchor}#{k}", -1) for k in range(n_roll)] for s in sets], dtype=np.int64)
    if (st < 0).any():
        raise ValueError("rollouts missing from the store")
    tt = np.array([s.t_min for s in sets], dtype=np.int64)
    ctl = np.array([idx.get(f"{s.anchor}#real", -1) for s in sets], dtype=np.int64)
    has = ctl >= 0
    ev = fpr.Evaluator(store)
    rows = fl.Rows(st.reshape(-1), np.repeat(tt, n_roll))
    rows_c = fl.Rows(ctl[has], tt[has])
    mv = min_valid if min_valid is not None else max(1, n_roll // 2)
    progs = [p for p in fpr.catalog(store.channels) if (programs is None and applies(p, design))
             or (programs is not None and key(p) in programs)]
    out = {s.anchor: {"binary": {}, "regression": {}, "tte": {}} for s in sets}

    def observed(p, kind):
        if not has.any():
            return np.full(len(sets), np.nan)
        b = ev.evaluate(p, rows_c)
        y = np.full(len(sets), np.nan)
        if kind == "tte":
            y[has] = np.where(b.valid & (b.cause == 0), b.t_event, np.where(b.valid, np.inf, np.nan))
        else:
            y[has] = np.where(b.valid, b.y, np.nan)
        return y

    for p in progs:
        name = key(p)
        b = ev.evaluate(p, rows)
        yr = np.where(b.valid, b.y, np.nan).astype(np.float64).reshape(len(sets), n_roll)
        nv = np.isfinite(yr).sum(1)
        if p.label in ("binary", "hosp_death"):
            ph = np.where(nv >= mv, np.nansum(yr, 1) / np.maximum(nv, 1), np.nan)
            obs = observed(p, "binary")
            for i, s in enumerate(sets):
                out[s.anchor]["binary"][name] = {"p": ph[i], "n_valid": int(nv[i]), "observed": obs[i]}
        elif p.label == "regression":
            obs = observed(p, "regression")
            for i, s in enumerate(sets):
                r = yr[i][np.isfinite(yr[i])]
                ok = len(r) >= mv
                out[s.anchor]["regression"][name] = {
                    "n_valid": len(r), "observed": obs[i],
                    **({"q10": float(np.quantile(r, 0.1)), "q50": float(np.median(r)), "q90": float(np.quantile(r, 0.9)),
                        "mean": float(r.mean())} if ok else {"q10": None, "q50": None, "q90": None, "mean": None})}
        elif p.label == "tte":
            te = np.where(b.valid & (b.cause == 0), b.t_event, np.inf).reshape(len(sets), n_roll)
            vr = b.valid.reshape(len(sets), n_roll)
            risk = np.where(vr.sum(1) >= mv, ((te <= 48) & vr).sum(1) / np.maximum(vr.sum(1), 1), np.nan)
            obs = observed(p, "tte")
            for i, s in enumerate(sets):
                out[s.anchor]["tte"][name] = {"p_48h": risk[i], "n_valid": int(vr[i].sum()), "observed_hours": obs[i]}
    rem = (store.los[st] - tt[:, None]) / 60.0
    cens = np.isin(store.end_kind[st], ("censored", "unknown"))
    for i, s in enumerate(sets):
        o = None
        if has[i]:
            o = float((store.los[ctl[i]] - tt[i]) / 60.0) if store.end_kind[ctl[i]] not in ("censored", "unknown") \
                else None
        out[s.anchor]["remaining_los"] = {"p_gt_7d": bench.km_survival(rem[i], ~cens[i], 168.0),
                                          "median_hours": km_quantile(rem[i], ~cens[i], 0.5),
                                          "censored_share": float(cens[i].mean()), "observed_hours": o}
    return out


def token_shares(v, prompt, conts: list, hours=(6.0, 24.0, 72.0)) -> dict:
    """The cheap reading (module docstring): per horizon, the share of rollouts with each flag."""
    from icutwin.eval.forecast import Reader

    rd = Reader(v)
    out = {}
    for h in hours:
        flags = [rd.within(list(map(int, prompt)) + list(map(int, c)), len(prompt), h) for c in conts]
        out[f"{h:g}h"] = {k: float(np.mean([f[k] for f in flags])) for k in flags[0]} if flags else {}
    return out
