"""Cohort synthesizer (toolkit tool 5, ``docs/toolkit.md``): synthetic ICU cohorts on request, exported as CLIF 3.0
and MEDS, with conformance, fidelity and novelty reports (Paper P A3, A4, C3, F6, F8).

A spec (a dict or YAML): ``n``; ``mode`` (``select``: twins of the bank, no GPU; ``sample``: new twins from the
sampler under the gate); ``datasets`` (dataset -> share, or a list); ``site``; ``filters`` on the twins' attributes:
dataset, site, sex, age (range) or age_band (18-44, 45-64, 65-79, 80+), unit_type, admission_type,
admission_source, adm_dx_group, surgical_status, los_hours (range), icu_outcome (alive / dead), hosp_death,
imv / vasopressor / rrt (yes / no: any episode in the ICU); ``seed``; ``export`` (clif, meds).

* ``select``: the bank's store (``factory.store``; attributes from its stays and derived episodes), filtered, then
  drawn per dataset share without replacement (a shortfall is reported).
* ``sample``: one ``generate.engine.Request`` per draw: the dataset (by share), the site, single-valued header
  fields (multi-valued ones in turn) and the sex as conditions; the other filters by rejection, with the bank's
  acceptance rate setting the oversampling (at most ``max_oversample``). Goal blocks are not used (the model barely
  conditions on them). Exact duplicates are dropped.
* Exports (``generate.clif_out.write``, ``generate.meds_out``): the twins decoded with drawn values (the features'
  decode), each its own patient; a manifest and a README mark the cohort internal (DUA-derived).
* Reports: the share meeting each filter; C3's profile (``tools/paper_c3.py``: age, female, ICU and hospital death,
  IMV, vasoactive and RRT at any time in the ICU, stays under an hour, events per hour, labs per day, stay length)
  against the real held-out stays meeting the same filters (aggregates with bootstrap intervals); novelty against the
  training stays when a saved reference exists (``generate.bank.Novelty``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

AGE_BANDS = ((18, 45, "18-44"), (45, 65, "45-64"), (65, 80, "65-79"), (80, 200, "80+"))
HEADER = ("unit_type", "admission_type", "admission_source", "adm_dx_group", "surgical_status")
EVENTS = ("imv", "vasopressor", "rrt")
QUANT = ("age", "female", "icu_death", "hosp_death", "ever_imv", "ever_vasopressor", "ever_rrt", "stay_lt_1h",
         "events_per_hour", "labs_per_day")
RATES = ("events_per_hour", "labs_per_day")


def age_band(age) -> str | None:
    if age is None or not np.isfinite(age):
        return None
    return next((name for lo, hi, name in AGE_BANDS if lo <= age < hi), None)


def attributes(store):
    """One row per stay of a store: the attributes the filters read, and C3's profile quantities."""
    import pandas as pd

    from icutwin.factory import programs as fpr

    st = store.stays
    ev = fpr.Evaluator(store)
    hours = np.maximum(store.los / 60.0, 1e-3)
    a = pd.DataFrame({"stay_id": st["stay_id"].astype(str), "dataset": st["dataset"].astype(str),
                      "site": st["site"].astype(str) if "site" in st else None,
                      "sex": st["sex"].astype(str), "age": st["age"].to_numpy(dtype=float),
                      **{h: st[h].astype(str) if h in st else None for h in HEADER},
                      "los_hours": hours, "icu_outcome": np.where(store.end_kind == "dead", "dead",
                                                                  np.where(store.end_kind == "alive", "alive", "unknown")),
                      "hosp_death": st["hosp_death"].astype(float).to_numpy() if "hosp_death" in st else np.nan})
    a["age_band"] = [age_band(x) for x in a["age"]]
    for k in EVENTS:
        first = ev.process(("event", k)).first_onset(len(store))
        a[k] = np.where(first < store.los, "yes", "no")
        a[f"ever_{k}"] = (first < store.los).astype(float)
    labs = np.zeros(len(store))
    for cid, g in enumerate(store.channels.group):
        if g == "labs":
            e = store.channel_events(cid)
            if len(e["stay"]):
                labs += np.bincount(np.asarray(e["stay"], dtype=np.int64), minlength=len(store))
    short = hours < 4
    a["female"] = (a["sex"] == "female").astype(float)
    a["icu_death"] = (store.end_kind == "dead").astype(float)
    a["events_per_hour"] = np.where(short, np.nan, st["n_ev"].to_numpy(dtype=float) / hours)
    a["labs_per_day"] = np.where(short, np.nan, labs / (hours / 24.0))
    a["stay_lt_1h"] = (hours < 1).astype(float)
    return a


def match(a, filters: dict) -> np.ndarray:
    """Rows meeting every filter (lists: membership; [lo, hi] for age and los_hours; else equality)."""
    m = np.ones(len(a), dtype=bool)
    for k, want in (filters or {}).items():
        if k in ("age", "los_hours"):
            lo, hi = want
            x = a[k].to_numpy(dtype=float)
            m &= (x >= (lo if lo is not None else -np.inf)) & (x <= (hi if hi is not None else np.inf))
        elif k == "hosp_death":
            m &= a[k].to_numpy(dtype=float) == float(bool(want))
        else:
            vals = {str(x) for x in (want if isinstance(want, (list, tuple, set)) else [want])}
            m &= a[k].astype(str).isin(vals).to_numpy()
    return m


def profile(a, m: np.ndarray, rng, boot: int = 300) -> dict:
    """C3's summary of the rows ``m`` (means, medians for the rates; bootstrap intervals)."""
    out = {"stays": int(m.sum())}
    for k in QUANT:
        x = a[k].to_numpy(dtype=float)[m]
        x = x[np.isfinite(x)]
        if not len(x):
            continue
        f = np.median if k in RATES else np.mean
        bs = [f(rng.choice(x, size=len(x))) for _ in range(boot)]
        out[k] = {"median" if k in RATES else "mean": float(f(x)),
                  "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    los = a["los_hours"].to_numpy(dtype=float)[m]
    if len(los):
        out["los_h"] = {"median": float(np.median(los)), "q25": float(np.quantile(los, 0.25)),
                        "q75": float(np.quantile(los, 0.75))}
    return out


def shares_of(spec: dict) -> dict | None:
    d = spec.get("datasets")
    if d is None:
        return None
    if isinstance(d, (list, tuple)):
        return {x: 1.0 / len(d) for x in d}
    tot = float(sum(d.values()))
    return {k: float(x) / tot for k, x in d.items()}


def allocate(n: int, shares: dict) -> dict:
    """Largest-remainder counts per dataset."""
    keys = sorted(shares)
    raw = np.array([shares[k] * n for k in keys])
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base))[: n - base.sum()]:
        base[i] += 1
    return dict(zip(keys, base.tolist(), strict=True))


def select(a, spec: dict, rng) -> tuple[list, dict]:
    """Twins of the bank meeting the filters, drawn per dataset share; (stay ids, shortfall per dataset)."""
    filters = dict(spec.get("filters") or {})
    if spec.get("site"):
        filters["site"] = spec["site"]
    m = match(a, filters)
    shares = shares_of(spec)
    n = int(spec["n"])
    if shares is None:
        idx = np.flatnonzero(m)
        pick = rng.permutation(idx)[:n]
        return a["stay_id"].to_numpy()[np.sort(pick)].tolist(), ({} if len(idx) >= n else {"all": n - len(idx)})
    out, short = [], {}
    for ds, k in allocate(n, shares).items():
        idx = np.flatnonzero(m & (a["dataset"].to_numpy() == ds))
        pick = np.sort(rng.permutation(idx)[:k])
        out += a["stay_id"].to_numpy()[pick].tolist()
        if len(idx) < k:
            short[ds] = k - len(idx)
    return out, short


def bank_ids(session) -> dict:
    import pyarrow.parquet as pq

    out = {}
    for f in sorted(session.paths.bank.glob("part*/*.parquet")):
        t = pq.read_table(f, columns=["twin", "ids"])
        out.update(zip(t.column("twin").to_pylist(), t.column("ids").to_pylist(), strict=True))
    return out


def sample_requests(session, spec: dict, acceptance: dict, max_oversample: int, seed: int) -> list:
    from icutwin.toolkit.jobs import SampleRequest

    filters = spec.get("filters") or {}
    n = int(spec["n"])
    shares = shares_of(spec) or {d: 1.0 / len(filters.get("dataset", ["mimic"])) for d in
                                 (filters.get("dataset") if isinstance(filters.get("dataset"), list)
                                  else [filters.get("dataset", "mimic")])}
    header = {h: (filters[h] if isinstance(filters[h], (list, tuple)) else [filters[h]]) for h in HEADER if h in filters}
    sex = filters.get("sex")
    max_new = spec.get("max_new")                    # a token budget per twin (the context by default)
    reqs, k = [], 0
    for ds, cnt in allocate(n, shares).items():
        over = int(np.clip(np.ceil(1.0 / max(acceptance.get(ds, 1.0), 1.0 / max_oversample)), 1, max_oversample))
        for j in range(cnt * over):
            reqs.append(SampleRequest(f"{ds}_{j:06d}", int(seed) * 1_000_000 + k, ds, spec.get("site"),
                                      {h: vals[j % len(vals)] for h, vals in header.items()},
                                      {"sex": sex} if isinstance(sex, str) else {},
                                      max_new=None if max_new is None else int(max_new)))
            k += 1
    return reqs


def export(session, ids: dict, out: Path, formats=("clif", "meds")) -> dict:
    """The twins in CLIF 3.0 and MEDS (drawn values), with a manifest; each twin its own patient."""
    import pyarrow.parquet as pq

    from icutwin.factory import twins
    from icutwin.tokenization.decode import load_persistence

    v = session.vocab
    rho_path = session.paths.tokens / "rank_persistence.json"
    rho = load_persistence(rho_path, v) if rho_path.exists() else None
    keys = sorted(ids)
    stays = [twins.decode_features(ids[k], v, k, 0, rho) for k in keys]
    out.mkdir(parents=True, exist_ok=True)
    rep = {}
    if "clif" in formats:
        from icutwin.generate import clif_out

        rep["clif"] = clif_out.write(stays, keys, out / "clif")
    if "meds" in formats:
        from icutwin.generate import meds_out

        t = meds_out.table(stays, list(range(len(stays))))
        (out / "meds").mkdir(exist_ok=True)
        pq.write_table(t, out / "meds" / "data.parquet")
        (out / "meds" / "subjects.json").write_text(json.dumps(dict(zip(range(len(keys)), keys, strict=True))),
                                                   encoding="utf-8", newline="\n")
        rep["meds"] = {"rows": int(t.num_rows), "subjects": len(keys)}
    (out / "README.txt").write_text(
        "INTERNAL synthetic ICU stays from the ICU digital twin (toolkit cohort synthesizer), a model trained on\n"
        "DUA-covered data. DUA-derived: keep on the approved hosts; never share unless cleared.\n",
        encoding="utf-8", newline="\n")
    return rep


def novelty_of(session, seqs: list, datasets: list) -> dict | None:
    """Distances to the nearest training stay against the held-out quantile (``generate.bank.Novelty``), when a
    saved reference exists (``runs/toolkit/novelty/<run>.npz``)."""
    f = session.paths.out / "novelty" / f"{session.run}.npz"
    if not f.exists():
        return None
    from icutwin.generate.bank import Novelty

    nov = Novelty.load(f, session.device if session.device != "cuda" else "cuda")
    d = nov.distances(seqs)
    ok = np.array([x >= nov.thr.get(ds, nov.pooled) for x, ds in zip(d, datasets, strict=True)])
    return {"novel_share": float(ok.mean()), "distance_median": float(np.median(d)),
            "distance_q05": float(np.quantile(d, 0.05)), "reference": str(f)}


def run(session, spec, job_dir=None, max_oversample: int = 20) -> dict:
    import yaml

    from icutwin.factory import store as fst
    from icutwin.generate import bank
    from icutwin.toolkit import schema

    if isinstance(spec, (str, Path)):
        spec = yaml.safe_load(Path(spec).read_text(encoding="utf-8"))
    spec = dict(spec)
    seed = int(spec.get("seed", session.seed))
    rng = np.random.default_rng(seed)
    mode = spec.get("mode", "select")
    job = session.job("cohort", {"spec": spec}, job_dir)
    bstore = fst.Store(session.paths.bank_store)
    ba = attributes(bstore)
    filters = dict(spec.get("filters") or {})
    if spec.get("site"):
        filters["site"] = spec["site"]
    caveats = []
    if mode == "select":
        chosen, short = select(ba, spec, rng)
        allids = bank_ids(session)
        ids = {k: allids[k] for k in chosen}
        coh = ba[ba["stay_id"].isin(set(chosen))].reset_index(drop=True)
        info = {"bank_matching": int(match(ba, filters).sum()), "shortfall": short}
    elif mode == "sample":
        rej = {k: x for k, x in filters.items() if k not in HEADER + ("dataset", "site", "sex")}
        acc = {}
        for ds in (shares_of(spec) or {}):
            m_ds = ba["dataset"].to_numpy() == ds
            hdr = match(ba, {k: x for k, x in filters.items() if k in HEADER + ("site", "sex")})
            base = (m_ds & hdr).sum()
            acc[ds] = float((m_ds & hdr & match(ba, rej)).sum() / base) if base else 1.0
        if not (job.root / "sample_requests.parquet").exists():
            job.write_requests(sample_requests(session, spec, acc, max_oversample, seed))
        session.sample(job)
        drawn = job.samples()
        rows = [{"stay_id": k, "ids": np.asarray(x, np.int32), "split": "cohort", "weight": 1.0, "cell": "cohort"}
                for k, x in sorted(drawn.items())]
        dup = bank.exact_duplicates([r["ids"].tolist() for r in rows])
        rows = [r for r, d in zip(rows, dup, strict=True) if not d]
        work = job.root / "store"
        if not (work / "manifest.json").exists():
            fst.build(rows, session.paths.tokens / "vocab.json", work, "seg_0000", {"kind": "toolkit_cohort"},
                      persistence=session.paths.tokens / "rank_persistence.json", workers=session.workers)
        ca = attributes(fst.Store(work))
        ok = match(ca, rej)
        keep, short = [], {}
        for ds, k in allocate(int(spec["n"]), shares_of(spec) or {"all": 1.0}).items():
            m = ok & ((ca["dataset"].to_numpy() == ds) if ds != "all" else True)
            pick = ca["stay_id"].to_numpy()[m][:k].tolist()
            keep += pick
            if len(pick) < k:
                short[ds] = k - len(pick)
        ids = {k: drawn[k] for k in keep}
        coh = ca[ca["stay_id"].isin(set(keep))].reset_index(drop=True)
        info = {"drawn": len(drawn), "duplicates": int(np.sum(dup)), "acceptance_bank": acc,
                "accepted": int(ok.sum()), "shortfall": short}
        caveats.append("Sampled twins: conditional sampling from [BOS] under the gate; filters beyond the header by "
                       "rejection.")
    else:
        raise ValueError(f"unknown mode {mode}")
    conform = {k: float(match(coh, {k: x}).mean()) if len(coh) else None for k, x in filters.items()}
    real = session.paths.stores.get("held_out")
    if real is None or not (Path(real) / "manifest.json").exists():
        real = session.paths.stores.get("tuning")
        caveats.append("Fidelity reference: the tuning split's real stays (no held-out store on this host).")
    fidelity = None
    if real is not None and (Path(real) / "manifest.json").exists():
        ra = attributes(fst.Store(Path(real)))
        fidelity = {"cohort": profile(coh, np.ones(len(coh), bool), rng),
                    "real": profile(ra, match(ra, filters), rng), "reference": str(real)}
    exported = export(session, ids, job.root / "export", tuple(spec.get("export", ("clif", "meds")))) \
        if spec.get("export", ("clif", "meds")) else {}
    nov = novelty_of(session, [list(map(int, ids[k])) for k in sorted(ids)],
                     coh.set_index("stay_id").loc[sorted(ids), "dataset"].tolist()) if ids else None
    if nov is None:
        caveats.append("Novelty not computed: no saved training reference on this host.")
    res = schema.result("cohort", session, settings={"spec": spec, "job": str(job.root)},
                        outputs={"n": len(ids), "mode": mode, **info, "conformance": conform,
                                 "fidelity": fidelity, "novelty": nov, "export": exported,
                                 "export_dir": str(job.root / "export") if exported else None,
                                 "datasets": {d: int(c) for d, c in coh["dataset"].value_counts().items()}},
                        caveats=caveats, evidence_keys=("refs", "claim"))
    (job.root / "cohort_ids.json").write_text(json.dumps(sorted(ids)), encoding="utf-8", newline="\n")
    job.write_result(res)
    return res
