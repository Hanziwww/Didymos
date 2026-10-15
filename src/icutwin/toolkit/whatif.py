"""What-if (toolkit tool 2, ``docs/toolkit.md``): the twin's futures for one patient when a treatment is given at the
prompt's end or withheld throughout, against a same-treatment null (Paper P C1, F5).

Arms (``toolkit.treatments``): *give* at the low, median or high rate bin of the dataset's C1 record (norepinephrine
and insulin: three doses; otherwise the median), *withhold* (every token of the treatment's family forbidden in the
continuation), and *withhold_b*, the same arm with another seed: their difference is the A/A null of this patient.
An intervention the grammar rejects (e.g. starting an infusion that already runs) drops that arm, with the reason.
``n`` rollouts per arm (16 by default), capped at the 24-hour token quantile.

Responses, each rollout's value over its window after the prediction time t (``toolkit.treatments``: C1's
channels, windows, aggregates, the urine-output cap), through the label engine (the rollouts decoded on the
patient's clock):

* per arm: mean, standard deviation, quantiles and the rollouts with a value;
* contrasts, give minus withhold: the mean difference with a Welch 95% interval;
* the A/A band: +-1.96 standard errors of a difference between two withhold arms (their pooled spread); a contrast
  outside it is distinguishable from rollout noise, which is not an effect estimate;
* the evidence flag of each response for this treatment (F5: reproduced, not reproduced, unresolved; exploratory
  for responses not pre-specified for it).

Per arm also the share of rollouts with a death, IMV, vasopressor or RRT event within 6 and 24 h (token reading).
Observationally faithful simulation within the twin world, not causal effect estimation; prompted and intervened
sampling is outside the privacy gate's version 1, so outputs stay on the approved host.
"""

from __future__ import annotations

import numpy as np


def arm_requests(session, anchor, treatment: str, n: int, cap: int, seed: int, natural: bool = False):
    """The arms' ``SimRequest``s and the arms the grammar dropped (arm -> reason)."""
    from icutwin.generate.engine import Intervention, intervene
    from icutwin.toolkit import treatments as tr
    from icutwin.toolkit.jobs import SimRequest

    v, t = session.vocab, session.tables
    give = tr.give_arms(v, treatment, anchor.dataset)
    hold = tr.family_tokens(v, treatment)
    reqs, dropped, tokens = [], {}, {}
    k = 0

    def add(arm, **kw):
        nonlocal k
        reqs.append(SimRequest(arm, tuple(anchor.prompt), n, int(seed) * 100_000 + k, cap, **kw))
        k += 1

    for arm, tok in give.items():
        try:
            intervene(anchor.prompt, Intervention(give=(tok,)), v, t)
        except ValueError as e:
            dropped[arm] = str(e)
            continue
        tokens[arm] = {"token": int(tok), "code": v.meta[tok].get("code"), "bin": v.meta[tok].get("bin")}
        add(arm, give=(int(tok),))
    add("withhold", withhold=hold)
    add("withhold_b", withhold=hold)
    if natural:
        add("natural")
    return reqs, dropped, tokens


def response_values(store, sets: list, t_min: int | None = None, key=None) -> dict:
    """response -> set key (default: the label after the anchor's ``|``, the arm) -> each rollout's value (NaN
    without one), every pre-specified response, read from each set's prediction time (``t_min`` overrides it)."""
    key = key or (lambda s: s.anchor.split("|", 1)[1])
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.toolkit import treatments as tr

    ev = fpr.Evaluator(store)
    idx = {s: i for i, s in enumerate(store.stays["stay_id"].tolist())}
    out: dict = {}
    for resp, (chans, win, agg, _sign, _t) in tr.all_responses().items():
        evs = fl.finite(ev.events(list(chans)))
        if resp in tr.CAP:
            keep = np.asarray(evs["v"], dtype=float) <= tr.CAP[resp]
            evs = {k: np.asarray(evs[k])[keep] for k in ("stay", "t", "v", "bin", "f") if k in evs}
        for s in sets:
            st = np.array([idx[f"{s.anchor}#{k}"] for k in range(len(s.conts))], dtype=np.int64)
            t = s.t_min if t_min is None else t_min
            val, _n = fl.window_values(evs, fl.Rows(st, np.full(len(st), t, dtype=np.int64)), 0, win, agg)
            out.setdefault(resp, {})[key(s)] = np.asarray(val, dtype=float)
    return out


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"difference": None, "ci": [None, None], "n": [len(a), len(b)]}
    d = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    return {"difference": d, "ci": [d - 1.96 * se, d + 1.96 * se], "se": se, "n": [len(a), len(b)]}


def summarize(values: dict, treatment: str) -> dict:
    """Per response: arm statistics, contrasts against withhold, the A/A band and the evidence flag."""
    from icutwin.toolkit import treatments as tr
    from icutwin.toolkit.schema import evidence

    ev = evidence()["whatif"]["responses"]
    mine = {r[0] for r in tr.responses(treatment)}
    out = {}
    for resp, by_arm in values.items():
        arms = {}
        for arm, x in by_arm.items():
            f = x[np.isfinite(x)]
            arms[arm] = {"n_valid": len(f), "mean": float(f.mean()) if len(f) else None,
                         "sd": float(f.std(ddof=1)) if len(f) > 1 else None,
                         "q10": float(np.quantile(f, 0.1)) if len(f) else None,
                         "q50": float(np.median(f)) if len(f) else None,
                         "q90": float(np.quantile(f, 0.9)) if len(f) else None}
        w, wb = by_arm.get("withhold"), by_arm.get("withhold_b")
        aa = welch(w, wb) if w is not None and wb is not None else None
        band = None
        if aa and aa.get("se") is not None:
            band = 1.96 * aa["se"]
        contrasts = {}
        for arm, x in by_arm.items():
            if arm.startswith("give_") and w is not None:
                c = welch(x, w)
                c["beyond_aa_band"] = None if band is None or c["difference"] is None else bool(abs(c["difference"]) > band)
                contrasts[arm] = c
        flag = (ev.get(treatment, {}).get(resp) or {}).get("status") if resp in mine else "exploratory"
        out[resp] = {"pre_specified": resp in mine, "evidence": flag or "exploratory",
                     "paper": ev.get(treatment, {}).get(resp) if resp in mine else None,
                     "expected_sign": tr.all_responses()[resp][3] if resp in mine else None,
                     "arms": arms, "contrasts": contrasts, "aa": aa, "aa_band": band}
    return out


def run(session, patient, hours: float, treatment: str, n: int = 16, cap_hours: float = 24.0,
        caps: dict | None = None, seed: int | None = None, natural: bool = False, job_dir=None) -> dict:
    from icutwin.toolkit import labels as tl
    from icutwin.toolkit import schema
    from icutwin.toolkit.forecast import restore_of

    if isinstance(patient, str):
        patient = session.patient(patient)
    v = session.vocab
    seed = session.seed if seed is None else int(seed)
    a = patient.at(v, float(hours))
    caps = caps or session.caps(cap_hours)
    cap = session.cap_for(caps, a.dataset)
    reqs, dropped, tokens = arm_requests(session, a, treatment, n, cap, seed, natural)
    job = session.job("whatif", {"patient": patient.ref, "hours": float(hours), "treatment": treatment, "n": n,
                                 "cap": cap, "seed": seed, "natural": natural}, job_dir)
    if not (job.root / "requests.parquet").exists():
        job.write_requests(reqs)
    session.sample(job)
    conts = job.rollouts()
    sets = [tl.RolloutSet(f"{a.id}|{r.id}", patient.stay_id, a.t_min, np.asarray(a.prompt, np.int32), conts[r.id],
                          restore=restore_of(session, patient)) for r in reqs]
    store = tl.build_store(sets, session.paths.tokens / "vocab.json", job.root / "store",
                           persistence=session.paths.tokens / "rank_persistence.json", workers=session.workers)
    values = response_values(store, sets, a.t_min)
    shares = {r.id: tl.token_shares(v, a.prompt, conts[r.id], hours=(6.0, 24.0)) for r in reqs}
    caveats = []
    if dropped:
        caveats.append(f"Arms the grammar rejected at this prompt: {', '.join(sorted(dropped))}.")
    if treatment == "insulin":
        caveats.append("Insulin: the glucose response is flagged not reproduced (F5).")
    res = schema.result("whatif", session, patient=patient, anchor=a,
                        settings={"treatment": treatment, "n": n, "cap_tokens": cap, "seed": seed, "job": str(job.root)},
                        outputs={"arms": {r.id: {"given": tokens.get(r.id), "withheld_tokens": len(r.withhold)}
                                          for r in reqs}, "dropped": dropped,
                                 "responses": summarize(values, treatment), "event_shares": shares,
                                 "timing": schema.evidence()["whatif"]["timing"]},
                        caveats=caveats, evidence_keys=("refs", "framing", "null"))
    job.write_result(res)
    return res
