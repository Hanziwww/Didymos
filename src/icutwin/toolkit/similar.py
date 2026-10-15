"""Similar patients (toolkit tool 4, ``docs/toolkit.md``): outcomes among the held-out stays nearest to a patient in
the twin's representation (Paper P C4, F7), beside the twin's own forecast.

* The index (``build_index``): held-out anchors of a design, each with the model's mean state over its prompt
  (``tools/server/paper_states.py``'s ``states.npz`` where it exists, else ``Teacher.states``) and its labels (A1's
  stored labels, ``predictions.parquet``); cosine similarity, as F7. Kept under ``runs/toolkit/index`` (git-ignored;
  per-anchor labels of real stays never leave the approved host).
* A query (``run``): the patient's mean state at the hour, its ``k`` nearest anchors (anchors of the patient's own
  stay left out), and per program the inclusion-weighted rate among those with a label, a Wilson interval, and the
  counts; a count of positives or negatives under ``min_cell`` is suppressed (the rate shown as a bound). No
  neighbour is shown.
* Exemplars (``build_bank_index``, ``exemplars``): the nearest bank twins (synthetic, internal) at matched hours, for
  cards and teaching.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def index_path(session, name: str) -> Path:
    return session.paths.out / "index" / f"{name}.npz"


def build_index(session, name: str, anchors_dir, labels_parquet, split: str = "held_out") -> Path:
    """The similarity index of a design's anchors (``anchors_dir``: ``anchors.parquet`` and, where it exists,
    ``states.npz``; ``labels_parquet``: per anchor and program its label)."""
    import pandas as pd

    from icutwin.eval import bench

    d = Path(anchors_dir)
    an = pd.read_parquet(d / "anchors.parquet").drop_duplicates("anchor").reset_index(drop=True)
    if (d / "states.npz").exists():
        z = np.load(d / "states.npz")
        pos = {a: i for i, a in enumerate(z["anchor"].tolist())}
        an = an[an["anchor"].isin(pos)].reset_index(drop=True)
        mean = z["mean"][[pos[a] for a in an["anchor"]]].astype(np.float32)
    else:
        v = session.vocab
        rows = {r["stay_id"]: r for r in bench.real_rows(session.paths.tokens, split, set(an["twin"].astype(str)))}
        an = an[an["twin"].astype(str).isin(rows)].reset_index(drop=True)
        prompts = [bench.prompt(rows[str(s)]["ids"], v, str(s), float(h)) for s, h in zip(an["twin"], an["hours"],
                                                                                           strict=True)]
        mean = session.teacher.states(prompts)[1]
    lab = pd.read_parquet(labels_parquet)
    progs = sorted(lab["program"].unique())
    y = np.full((len(an), len(progs)), np.nan, dtype=np.float32)
    for j, (_p, g) in enumerate(lab.groupby("program", sort=True)):
        y[:, j] = g.drop_duplicates("anchor").set_index("anchor")["label"].reindex(an["anchor"]).to_numpy(np.float32)
    out = index_path(session, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, state=_unit(mean).astype(np.float16), anchor=an["anchor"].to_numpy(str),
                        stay=an["twin"].astype(str).to_numpy(str), dataset=an["dataset"].to_numpy(str),
                        hours=an["hours"].to_numpy(np.float32), weight=an["weight"].to_numpy(np.float32),
                        labels=y, programs=np.array(progs, dtype=str),
                        meta=np.asarray(json.dumps({"anchors_dir": str(d), "labels": str(labels_parquet),
                                                    "states": "states.npz" if (d / "states.npz").exists()
                                                    else "teacher"})))
    return out


def wilson(pos: float, n: float, z: float = 1.96) -> list:
    if n <= 0:
        return [None, None]
    p = pos / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def neighbour_rates(sim: np.ndarray, labels: np.ndarray, weights: np.ndarray, programs, k: int,
                    min_cell: int) -> dict:
    """Per program over the ``k`` most similar anchors: the weighted rate, a Wilson interval, the counts (suppressed
    under ``min_cell``)."""
    nb = np.argsort(-sim, kind="stable")[:k]
    out = {}
    for j, prog in enumerate(programs):
        y, w = labels[nb, j], weights[nb]
        ok = np.isfinite(y)
        n, pos = int(ok.sum()), int(np.nansum(y[ok]))
        r = {"neighbours_with_label": n}
        if n == 0:
            r.update(rate=None, ci=[None, None])
        elif pos < min_cell or n - pos < min_cell:
            r.update(rate=None, ci=[None, None], suppressed=True,
                     bound=f"<{min_cell}/{n}" if pos < min_cell else f">{n - min_cell}/{n}")
        else:
            r.update(rate=float(np.sum(w[ok] * y[ok]) / np.sum(w[ok])), ci=wilson(pos, n), positives=pos)
        out[prog] = r
    return out


def run(session, patient, hours: float, index: str | None = None, k: int = 50, min_cell: int = 10,
        forecast: dict | None = None, exemplar_index: str | None = None, exemplars_k: int = 5) -> dict:
    """``forecast``: a Forecast result of the same anchor, shown beside the neighbours' rates."""
    from icutwin.toolkit import schema
    from icutwin.toolkit.forecast import design_of

    if isinstance(patient, str):
        patient = session.patient(patient)
    index = index or f"test_{design_of(hours)}"
    z = np.load(index_path(session, index))
    a = patient.at(session.vocab, float(hours))
    q = _unit(session.teacher.states([a.prompt])[1])[0]
    sim = z["state"].astype(np.float32) @ q
    same = z["stay"] == str(patient.stay_id)
    sim = np.where(same, -np.inf, sim)
    rates = neighbour_rates(sim, z["labels"], z["weight"], z["programs"].tolist(), k, min_cell)
    if forecast is not None:
        for prog, r in rates.items():
            fr = (forecast.get("outputs", {}).get("risks") or {}).get(prog)
            if fr is not None:
                r["twin_risk"] = fr.get("risk")
    nb = np.argsort(-sim, kind="stable")[:k]
    out = {"index": index, "k": k, "min_cell": min_cell, "excluded_own_stay": int(same.sum()),
           "similarity": {"median": float(np.median(sim[nb])), "min": float(sim[nb].min())},
           "neighbour_datasets": {d: int(c) for d, c in zip(*np.unique(z["dataset"][nb], return_counts=True),
                                                           strict=True)},
           "rates": rates}
    if exemplar_index:
        out["exemplars"] = exemplars(session, q, exemplar_index, exemplars_k, float(hours))
    return schema.result("similar", session, patient=patient, anchor=a,
                         settings={"index": index, "k": k, "min_cell": min_cell}, outputs=out,
                         evidence_keys=("refs", "claim"))


def build_bank_index(session, name: str = "bank", hours=(6.0, 12.0, 24.0, 48.0), max_twins: int | None = None,
                     seed: int = 0) -> Path:
    """The states of bank twins at ``hours`` (twins whose ICU stay lasts past the hour), for exemplars."""
    import pyarrow.parquet as pq

    from icutwin.toolkit.inputs import from_ids

    v = session.vocab
    rows = []
    for f in sorted(session.paths.bank.glob("part*/*.parquet")):
        rows += pq.read_table(f, columns=["twin", "dataset", "ids"]).to_pylist()
    if max_twins and len(rows) > max_twins:
        pick = np.sort(np.random.default_rng(seed).permutation(len(rows))[:max_twins])
        rows = [rows[i] for i in pick]
    prompts, meta = [], []
    for r in rows:
        p = from_ids(r["ids"], r["twin"], r["dataset"], "twin")
        los = p.los_hours(v)
        for h in hours:
            if h < los:
                prompts.append(p.at(v, h).prompt)
                meta.append((r["twin"], r["dataset"], h))
    mean = session.teacher.states(prompts)[1]
    out = index_path(session, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, state=_unit(mean).astype(np.float16), twin=np.array([m[0] for m in meta], dtype=str),
                        dataset=np.array([m[1] for m in meta], dtype=str),
                        hours=np.array([m[2] for m in meta], dtype=np.float32))
    return out


def exemplars(session, q: np.ndarray, name: str, k: int, hours: float | None = None) -> list:
    """The ``k`` nearest bank twins to a state (one anchor per twin; matched hours when given)."""
    z = np.load(index_path(session, name))
    sim = z["state"].astype(np.float32) @ q
    if hours is not None:
        sim = np.where(np.abs(z["hours"] - hours) <= 6, sim, -np.inf)
    out, seen = [], set()
    for i in np.argsort(-sim, kind="stable"):
        if not np.isfinite(sim[i]):
            break
        if z["twin"][i] in seen:
            continue
        seen.add(z["twin"][i])
        out.append({"twin": str(z["twin"][i]), "dataset": str(z["dataset"][i]), "hours": float(z["hours"][i]),
                    "similarity": float(sim[i])})
        if len(out) == k:
            break
    return out
