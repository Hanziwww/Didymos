"""Teaching cases (toolkit tool 7, ``docs/toolkit.md``; the study lead's addition): synthetic ICU cases for clinical
teaching, built from bank twins (synthetic, internal) with the twin's forecast, what-if branches and attributions
precomputed at a decision point.

* **Case builder**: the learning objective is a cohort filter on the bank (``toolkit.cohort``: e.g. a vasopressor
  episode in the ICU and AKI); the decision point is, by default, one minute before the twin's own first start of the
  objective's treatment (F5's faithful case: a real start's prompt cut a minute earlier), at least 2.5 h after ICU
  admission, the treatment's first record being a start (C1's eligibility); or a fixed hour. Cases are the most
  typical candidates (the highest mean cosine similarity of the model's state to the other candidates' at their
  decision points) or drawn at random, with optionally the least typical one.
* **Per case**: a fast forecast (16 rollouts, E2, the label engine) with the twin's own outcomes (its continuation is
  the case's truth), What-if for the learner's options (the objective's treatment and any others; each response
  carries its F5 flag, insulin with its warning), the top attributions of a program for the teacher (Explain), the
  twin's own course after the decision point (events per family over the next 24 h).
* **Outputs**: ``case_<k>.json`` and the cards (``toolkit.card``): the learner's (history, risk sliders, a choice of
  action, a reveal with the forecast against the learner's estimates, the options' simulated futures with the A/A band
  and the twin's own course; answers downloaded as JSON), the teacher's (all of it, with attributions and discussion
  prompts) and the prognosis drill (every case's risk questions, a calibration plot of the learner at the end).

Every case is a synthetic patient, internal until cleared; branches are observational simulation within the twin
world. The two-step chain of the plan (a second decision on a branch's medoid rollout) is not built in this version.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

DRILL = ("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "aki_2@2880")


def first_start(stay, treatment: str) -> float | None:
    """Hours after ICU admission of the treatment's first record when it is a start (C1's rule), else None."""
    from icutwin.toolkit import treatments as tr

    start_re, fam, _dose = tr.TREATMENTS[treatment]
    start_c = re.compile(start_re)
    fam_c = re.compile(fam) if fam not in ("rrt", "imv") else None
    for t, _r, code, _n, _x in stay.timeline:
        hit = start_c.match(code) or (fam_c.match(code) if fam_c else False)
        if hit:
            h = (t - stay.icu_in).total_seconds() / 3600
            return h if start_c.match(code) and h >= 0 else None
    return None


def candidates(session, objective: dict, treatment: str | None, decision, limit: int, seed: int) -> list:
    """(patient, decision hour) of bank twins meeting the objective (at most ``limit``, drawn at random)."""
    from icutwin.factory import store as fst
    from icutwin.toolkit import cohort

    a = cohort.attributes(fst.Store(session.paths.bank_store))
    m = cohort.match(a, objective.get("filters") or {})
    if objective.get("datasets"):
        m &= a["dataset"].isin(list(objective["datasets"])).to_numpy()
    ids = a["stay_id"].to_numpy()[m]
    rng = np.random.default_rng(seed)
    out = []
    v = session.vocab
    for tw in rng.permutation(ids):
        p = session.patient(f"bank:{tw}")
        los = p.los_hours(v)
        if decision == "before_treatment":
            h = first_start(p.stay(v), treatment) if treatment else None
            if h is None or h < 2.5:
                continue
            h = h - 1 / 60
        else:
            h = float(decision)
        if h < los:
            out.append((p, h))
        if len(out) >= limit:
            break
    return out


def typical(session, cands: list, k: int, atypical: bool = False) -> list:
    """The ``k`` candidates with the highest mean cosine similarity of their states to the others' (and the least
    typical one)."""
    if len(cands) <= k:
        return list(range(len(cands)))
    v = session.vocab
    mean = session.teacher.states([p.at(v, h).prompt for p, h in cands])[1]
    z = mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-9)
    s = z @ z.T
    np.fill_diagonal(s, np.nan)
    score = np.nanmean(s, axis=1)
    order = list(np.argsort(-score))
    pick = order[:k]
    if atypical and order[-1] not in pick:
        pick.append(order[-1])
    return [int(i) for i in pick]


def own_course(session, patient, hours: float, window: float = 24.0) -> dict:
    """The twin's own events after the decision point, per family and block of hours (the case's truth)."""
    from icutwin.eval.attribution import family

    s = patient.stay(session.vocab)
    t0 = s.icu_in.timestamp() + hours * 3600
    out: dict = {}
    for t, _r, code, num, _x in s.timeline:
        dt = (t.timestamp() - t0) / 3600
        if 0 < dt <= window:
            f = family(code)
            out.setdefault(f, []).append({"hours": round(dt, 2), "code": code,
                                          "value": None if num is None or not np.isfinite(num) else float(num)})
    end = (s.icu_out - s.icu_in).total_seconds() / 3600
    return {"events": out, "icu_discharge_hours": end - hours,
            "end": [c for c in s.end] if hasattr(s, "end") else None}


def run(session, objective: dict, treatment: str | None = "norepinephrine", options=None, cases: int = 3,
        decision="before_treatment", program: str = "decompensation@1440", n: int = 16, pool: int = 60,
        pick: str = "typical", atypical: bool = False, seed: int | None = None, out_dir=None) -> dict:
    """``options``: treatments offered to the learner (default: the objective's treatment)."""
    from icutwin.toolkit import card, schema

    seed = session.seed if seed is None else int(seed)
    options = list(options or ([treatment] if treatment else []))
    cands = candidates(session, objective, treatment, decision, pool, seed)
    if not cands:
        raise ValueError("no bank twin meets the objective at its decision point")
    sel = typical(session, cands, cases, atypical) if pick == "typical" else list(range(min(cases, len(cands))))
    out = Path(out_dir) if out_dir else session.paths.out / "teach" / f"{seed}_{len(cands)}"
    out.mkdir(parents=True, exist_ok=True)
    built = []
    for k, i in enumerate(sel):
        p, h = cands[i]
        fc = session.forecast(p, h, n=n, observed=True)
        wi = {}
        for opt in options:
            try:
                wi[opt] = session.whatif(p, h, opt, n=n)
            except KeyError as e:                      # no C1 record for this dataset
                wi[opt] = {"error": str(e)}
        try:
            ex = session.explain(p, h, program)
        except (KeyError, FileNotFoundError) as e:
            ex = {"error": str(e)}
        case = {"case": k, "patient": p.ref, "dataset": p.dataset, "hours": h, "objective": objective,
                "typicality": "atypical" if atypical and k == len(sel) - 1 and len(sel) > cases else pick,
                "forecast": fc, "whatif": wi, "explain": ex, "course": own_course(session, p, h),
                "drill": [x for x in DRILL if x in fc["outputs"]["risks"]]}
        (out / f"case_{k}.json").write_text(json.dumps(schema.clean(case), indent=1), encoding="utf-8",
                                            newline="\n")
        card.write(card.teaching(case, session, mode="learner"), out / f"case_{k}_learner.html")
        card.write(card.teaching(case, session, mode="teacher"), out / f"case_{k}_teacher.html")
        built.append(case)
    card.write(card.drill(built, session), out / "drill.html")
    res = schema.result("teach", session,
                        settings={"objective": objective, "treatment": treatment, "options": options, "cases": cases,
                                  "decision": decision, "program": program, "n": n, "pool": pool, "pick": pick,
                                  "seed": seed},
                        outputs={"cases": [{"case": c["case"], "patient": c["patient"], "hours": c["hours"],
                                            "files": [f"case_{c['case']}.json", f"case_{c['case']}_learner.html",
                                                      f"case_{c['case']}_teacher.html"]} for c in built],
                                 "drill": "drill.html", "dir": str(out), "candidates": len(cands)},
                        evidence_keys=("refs",))
    (out / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    return res
