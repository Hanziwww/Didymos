"""Forecast (toolkit tool 1, ``docs/toolkit.md``): what happens next to a patient, by the twin's rollouts from the
patient as known at an hour (A1, F1, F2; amendment 10's comparators).

Per anchor, ``n`` rollouts (fast: 16; standard: 128) capped at the 72-hour token quantile by default
(``Session.caps``; ``cap_mult`` 4 with 168 h for ICU death until discharge, F1), then:

* E2 (``toolkit.teacher``: the step hazards, ``eval.hazards``): cumulative incidence of death, alive discharge and
  the first IMV, vasopressor and RRT, hourly to 48 h with E0 (the plain share) beside it; ICU death until discharge;
  the E2 risk of the programs F2 scored (death within 24 h, new IMV and vasopressor within 24 h, new RRT within
  48 h);
* the label engine (``toolkit.labels``): the risk of every binary program of the design (the share of valid
  rollouts), value forecasts (10th/50th/90th percentiles), times to event, the remaining ICU stay (Kaplan-Meier);
* recalibration (``toolkit.recalibration``) where amendment 11's maps exist, and each program's paper O/E.

``run`` returns one result per anchor (a list for several patients or hours: a rolling trajectory); ``batch``
writes per-anchor predictions of a design (``predictions.parquet``, ``e2.parquet``) for the conformance test.
"""

from __future__ import annotations

import numpy as np

MODES = {"fast": 16, "standard": 128}
E2_PROGRAMS = {"decompensation@1440": ("death", 24.0), "imv_onset@1440": ("imv", 24.0),
               "vasopressor_onset@1440": ("vasopressor", 24.0), "rrt_onset@2880": ("rrt", 48.0),
               "icu_death@-1": ("death", float("inf"))}
CAUSES = ("death", "alive", "imv", "vasopressor", "rrt")
CURVE_HOURS = tuple(range(1, 49))


def design_of(hours: float) -> str:
    """The paper design whose program set and evidence apply at an hour (landmarks at 24 and 48 h, else rolling)."""
    return {24.0: "L24", 48.0: "L48"}.get(float(hours), "R")


def requests(session, anchors: list, n: int, caps: dict, seed: int) -> list:
    from icutwin.toolkit.jobs import SimRequest

    return [SimRequest(a.id, tuple(a.prompt), int(n), int(seed) * 100_000 + i, session.cap_for(caps, a.dataset))
            for i, a in enumerate(anchors)]


def restore_of(session, patient) -> dict:
    """The real dataset and site tokens of an unknown-site prompt (put back before decoding)."""
    if not patient.unknown_site or patient.dataset is None:
        return {}
    from icutwin.model.data import layout

    lay = layout(session.vocab)
    return {lay.dataset_pos: int(patient.ids[lay.dataset_pos]), lay.site_pos: int(patient.ids[lay.site_pos])}


def e2_summary(paths: list, edges) -> dict:
    """E2 and E0 incidence curves, horizons and the E2 programs of one anchor's hazard paths."""
    from icutwin.eval import hazards as hz
    from icutwin.toolkit.teacher import estimates

    out = {"curves": {"hours": list(CURVE_HOURS)}, "programs": {}}
    for c in CAUSES:
        est = [estimates(paths, c, float(h), edges) for h in CURVE_HOURS]
        out["curves"][c] = {"e2": [e["e2"] for e in est], "e0": [e["e0"] for e in est]}
    for prog, (c, h) in E2_PROGRAMS.items():
        e = estimates(paths, c, h, edges)
        out["programs"][prog] = {"e2": e["e2"], "e0": e["e0"], "e1": e["e1"], "censored_before": e["censored_before"]}
    out["censored_share"] = float(np.mean([p.end_kind == hz.END_CENSORED for p in paths]))
    return out


def save_paths(path, ids: list, paths: list, edges: np.ndarray) -> None:
    """Every anchor's hazard paths (``eval.hazards.Path``) as arrays, so E2 can be recomputed on any rollout subset."""
    from icutwin.eval import hazards as hz

    flat = [(i, k, p) for i, ps in zip(ids, paths, strict=True) for k, p in enumerate(ps)]
    arr = {"anchor": np.array([x[0] for x in flat], dtype=str), "rollout": np.array([x[1] for x in flat], np.int32),
           "end_hour": np.array([x[2].end_hour for x in flat], np.float32),
           "end_kind": np.array([x[2].end_kind for x in flat], np.int8), "edges": edges}
    for c in hz.CAUSES:
        arr[f"first_{c}"] = np.array([x[2].first[c] for x in flat], np.float32)
        arr[f"inc_{c}"] = np.stack([x[2].inc[c] for x in flat]).astype(np.float32)
        if c in hz.TREATMENTS:
            arr[f"comp_{c}"] = np.stack([x[2].comp[c] for x in flat]).astype(np.float32)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arr)
    tmp.replace(path)


def load_paths(path) -> tuple[dict, np.ndarray]:
    """anchor id -> its paths in rollout order, and the bins' edges (``save_paths``)."""
    from icutwin.eval import hazards as hz

    with np.load(path) as z:                         # each array decompressed once (an NpzFile reads on every access)
        z = {k: z[k] for k in z.files}
    out: dict = {}
    for j in np.lexsort((z["rollout"], z["anchor"])):
        p = hz.Path({c: float(z[f"first_{c}"][j]) for c in hz.CAUSES}, float(z["end_hour"][j]), int(z["end_kind"][j]),
                    {c: z[f"inc_{c}"][j] for c in hz.CAUSES}, {c: z[f"comp_{c}"][j] for c in hz.TREATMENTS})
        out.setdefault(str(z["anchor"][j]), []).append(p)
    return out, z["edges"]


def pipeline(session, anchors: list, job, n: int, caps: dict, seed: int, e2: bool = True, labels: bool = True,
             design: str | None = None, observed: bool = False) -> dict:
    """anchor id -> {"e2": ..., "labels": ...} (module docstring); the job's stages are resumable (the hazard paths
    are kept in ``hazards.npz``)."""
    from icutwin.toolkit import labels as tl
    from icutwin.toolkit.teacher import LONG_EDGES, HazardInput

    if not (job.root / "requests.parquet").exists():
        job.write_requests(requests(session, anchors, n, caps, seed))
    session.sample(job)
    conts = job.rollouts()
    out = {a.id: {} for a in anchors}
    if e2:
        hp = job.root / "hazards.npz"
        if hp.exists():
            by, edges = load_paths(hp)
            paths = [by[a.id] for a in anchors]
        else:
            items = [HazardInput(np.asarray(a.prompt, np.int64), conts[a.id], a.dataset, a.patient.stay_id,
                                 session.cap_for(caps, a.dataset)) for a in anchors]
            paths, edges = session.teacher.hazards(items, LONG_EDGES), LONG_EDGES
            save_paths(hp, [a.id for a in anchors], paths, edges)
        for a, ps in zip(anchors, paths, strict=True):
            out[a.id]["e2"] = e2_summary(ps, edges)
    if labels:
        sets = [tl.RolloutSet(a.id, a.patient.stay_id, a.t_min, np.asarray(a.prompt, np.int32), conts[a.id],
                              real_ids=a.patient.ids if observed else None, restore=restore_of(session, a.patient))
                for a in anchors]
        store = tl.build_store(sets, session.paths.tokens / "vocab.json", job.root / "store",
                               persistence=session.paths.tokens / "rank_persistence.json", workers=session.workers)
        by_design: dict = {}
        for a, s in zip(anchors, sets, strict=True):
            by_design.setdefault(design or design_of(a.hours), []).append(s)
        for d, ss in by_design.items():
            for k, r in tl.readout(store, ss, design=d).items():
                out[k]["labels"] = {**r, "design": d}
    return out


def outputs_of(raw: dict, maps: dict) -> dict:
    """One anchor's outputs: risks (share, E2, recalibrated, the paper's O/E), curves, values, the remaining stay."""
    from icutwin.toolkit import recalibration as rc
    from icutwin.toolkit.schema import evidence

    cal = evidence()["forecast"]["calibration"]
    lab, e2 = raw.get("labels") or {}, raw.get("e2") or {}
    risks = {}
    for prog, r in (lab.get("binary") or {}).items():
        risks[prog] = {"share": r["p"], "n_valid": r["n_valid"]}
        if r.get("observed") is not None and np.isfinite(r["observed"]):
            risks[prog]["observed"] = r["observed"]
    for prog, r in (e2.get("programs") or {}).items():
        risks.setdefault(prog, {}).update(e2=r["e2"], e0=r["e0"], censored_before=r["censored_before"])
    for prog, r in risks.items():
        raw_p = r.get("e2") if r.get("e2") is not None and np.isfinite(r.get("e2")) else r.get("share")
        p_cal, ok = rc.apply(maps, prog, raw_p)
        r["risk"] = p_cal if ok else raw_p
        r["calibrated"] = bool(ok)
        r["estimator"] = "e2" if raw_p is r.get("e2") else "share"
        if prog in cal:
            r["paper_oe"] = cal[prog]["oe"]
    return {"risks": risks, "incidence": e2.get("curves"), "censored_share": e2.get("censored_share"),
            "values": lab.get("regression"), "time_to_event": lab.get("tte"),
            "remaining_los": lab.get("remaining_los"), "design": lab.get("design")}


def run(session, patient, hours, mode: str = "fast", n: int | None = None, cap_hours: float = 72.0,
        cap_mult: float = 1.0, design: str | None = None, e2: bool = True, labels: bool = True,
        observed: bool = False, job_dir=None, seed: int | None = None, caps: dict | None = None):
    """Forecast one patient (a ``Patient`` or a reference) at one hour or a list of hours, or several patients.
    ``observed``: add the patient's own labels (a twin's or a held-out stay's), for evaluation and teaching;
    ``caps``: token caps per dataset in place of the session's (``Session.caps``)."""
    from icutwin.toolkit import recalibration as rc
    from icutwin.toolkit import schema

    pats = patient if isinstance(patient, list) else [patient]
    pats = [session.patient(p) if isinstance(p, str) else p for p in pats]
    hrs = [float(h) for h in (hours if isinstance(hours, (list, tuple)) else [hours])]
    n = int(n or MODES[mode])
    seed = session.seed if seed is None else int(seed)
    v = session.vocab
    anchors = [p.at(v, h) for p in pats for h in hrs]
    caps = caps or session.caps(cap_hours, cap_mult)
    spec = {"patients": [p.ref for p in pats], "hours": hrs, "n": n, "mode": mode, "cap_hours": cap_hours,
            "cap_mult": cap_mult, "caps": caps, "seed": seed, "e2": e2, "labels": labels, "observed": observed}
    job = session.job("forecast", spec, job_dir)
    raw = pipeline(session, anchors, job, n, caps, seed, e2, labels, design, observed)
    maps = rc.load(rc.path_of(session))
    res = []
    for a in anchors:
        extra = [] if maps else ["No recalibration maps: risks are the raw Twin's (see paper_oe per program)."]
        if cap_mult < 4 and e2:
            extra.append("ICU death until discharge is censored by the token cap (cap_mult < 4).")
        res.append(schema.result("forecast", session, patient=a.patient, anchor=a, outputs=outputs_of(raw[a.id], maps),
                                 settings={"n": n, "mode": mode, "cap_hours": cap_hours, "cap_mult": cap_mult,
                                           "cap_tokens": session.cap_for(caps, a.dataset), "seed": seed,
                                           "job": str(job.root)},
                                 caveats=extra, evidence_keys=("refs", "claim", "e2")))
    job.write_result(res)
    return res[0] if len(res) == 1 else res


def batch(session, anchors: list, job_dir, n: int, caps: dict, seed: int = 0, e2: bool = True,
          labels: bool = True, design: str | None = None, weights: dict | None = None, observed: bool = False) -> dict:
    """Per-anchor predictions of a design for evaluation (``predictions.parquet``: anchor, stay, dataset, weight,
    program, p, n_valid, and with ``observed`` the alignment row's label; ``e2.parquet``: anchor, program, e0, e1,
    e2); returns the raw outputs."""
    from icutwin.toolkit.jobs import Job, write_parquet

    job = Job(job_dir)
    if not job.spec_path.exists():
        job.write_spec({"tool": "forecast_batch", "anchors": len(anchors), "n": n, "caps": caps, "seed": seed,
                        "observed": observed, "provenance": session.provenance()})
    raw = pipeline(session, anchors, job, n, caps, seed, e2, labels, design, observed)
    w = weights or {}
    if labels:
        rec = [(a.id, a.patient.stay_id, a.dataset, float(w.get(a.id, 1.0)), prog, r["p"], r["n_valid"],
                r.get("observed")) for a in anchors for prog, r in raw[a.id]["labels"]["binary"].items()]
        cols = {"anchor": [x[0] for x in rec], "stay": [x[1] for x in rec], "dataset": [x[2] for x in rec],
                "weight": [x[3] for x in rec], "program": [x[4] for x in rec],
                "p": [float("nan") if x[5] is None else float(x[5]) for x in rec], "n_valid": [x[6] for x in rec]}
        if observed:
            cols["label"] = [float("nan") if x[7] is None else float(x[7]) for x in rec]
        write_parquet(job.root / "predictions.parquet", cols)
    if e2:
        rec = [(a.id, prog, r["e0"], r["e1"], r["e2"]) for a in anchors
               for prog, r in raw[a.id]["e2"]["programs"].items()]
        write_parquet(job.root / "e2.parquet", {"anchor": [x[0] for x in rec], "program": [x[1] for x in rec],
                                                 "e0": [float(x[2]) for x in rec], "e1": [float(x[3]) for x in rec],
                                                 "e2": [float(x[4]) for x in rec]})
    return raw
