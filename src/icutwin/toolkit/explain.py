"""Explain (toolkit tool 3, ``docs/toolkit.md``): which parts of a patient's history move the twin's estimate (Paper P
C2, F4, amendment 9).

The history as known at the hour is split into event groups (``eval.attribution.stay_groups``: a channel family x a
block before the cutoff, 0-6 h, 6-24 h, over 24 h, plus the statics). The estimate explained is the Twin probe of a
program (``runs/paper/a1/test_<design>/probe_models.npz``: a logistic regression on the model's last and mean
states at train-split anchors, ``tools/paper_probe.py``), the differentiable surrogate C2 attributed:

* ``edit`` (primary): each group removed from the decoded stay and the prompt encoded again (``eval.attribution.
  edited_prompt``; infusion records left orphaned dropped too), the probe logit's change (positive: the group raised
  the risk);
* ``knockout`` (amendment 9): attention from every other position to the group's tokens blocked in every layer, the
  probe logit's fall (prompts up to ``max_knockout_tokens``);
* ``generator`` (optional, slow): rollouts of the original and of the ``generator_top`` groups' edits, the program's
  risk (E2 for the hazard programs, the label engine's share otherwise) and its change.

Attention rollout is not offered: it fails the randomization check (amendment 9).
"""

from __future__ import annotations

import numpy as np


def probe_of(session, program: str, design: str) -> np.ndarray:
    f = session.paths.paper / "a1" / f"test_{design}" / "probe_models.npz"
    if not f.exists():
        raise FileNotFoundError(f"no Twin probes for design {design} ({f})")
    z = np.load(f)
    k = program.replace("@", "_at_")
    if k not in z.files:
        raise KeyError(f"no probe for {program} in {design}; available: "
                       f"{sorted(x.replace('_at_', '@') for x in z.files)}")
    return np.asarray(z[k], dtype=np.float64)


def groups_of(session, patient, hours: float):
    """(prospective stay, groups, base prompt, group -> edited prompt) with the patient's prompt transform."""
    from icutwin.eval import attribution as at
    from icutwin.eval import bench
    from icutwin.tokenization.encode import prospective_stay

    v = session.vocab
    s = prospective_stay(patient.stay(v), float(hours))
    groups = at.stay_groups(s, float(hours))
    names = [g for g, ix in groups.items() if ix or g == "statics"]
    tf = (lambda p: [int(x) for x in bench.unknown_site(p, v)]) if patient.unknown_site else \
        (lambda p: [int(x) for x in p])
    p0 = patient.at(v, float(hours)).prompt
    edits = {g: tf(at.edited_prompt(s, v, float(hours), g, groups)) for g in names}
    return s, groups, p0, edits


def run(session, patient, hours: float, program: str, design: str | None = None, knockout: bool = True,
        generator_top: int = 0, n: int = 16, max_knockout_tokens: int = 8192, caps: dict | None = None,
        seed: int | None = None, job_dir=None) -> dict:
    from icutwin.eval import attribution as at
    from icutwin.toolkit import schema
    from icutwin.toolkit.forecast import design_of

    if isinstance(patient, str):
        patient = session.patient(patient)
    v = session.vocab
    design = design or design_of(hours)
    coef = probe_of(session, program, design)
    a = patient.at(v, float(hours))
    _s, groups, p0, edits = groups_of(session, patient, hours)
    names = list(edits)
    logits = session.teacher.probe_logits([p0] + [edits[g] for g in names], coef)
    f0 = float(logits[0])
    edit = {g: f0 - float(x) for g, x in zip(names, logits[1:], strict=True)}
    ko = {}
    if knockout and len(p0) <= max_knockout_tokens:
        positions: dict = {}
        for k, gname in enumerate(at.token_groups(p0, v)):
            if gname is not None:
                positions.setdefault(gname, []).append(k)
        ko = session.teacher.knockout(p0, coef, positions)
    gen = {}
    caveats = []
    if knockout and len(p0) > max_knockout_tokens:
        caveats.append(f"Knockout skipped: the prompt has {len(p0)} tokens (limit {max_knockout_tokens}).")
    if generator_top:
        gen = generator_effects(session, patient, a, program, edits, sorted(edit, key=lambda g: -abs(edit[g]))
                                [:generator_top], n, caps, session.seed if seed is None else int(seed), job_dir)
    rows = []
    for g in sorted(names, key=lambda g: -abs(edit[g])):
        fam, _, blk = g.partition("|")
        rows.append({"group": g, "family": fam, "block": blk or None, "events": len(groups.get(g, ())),
                     "edit": edit[g], "knockout": ko.get(g), "generator": gen.get(g)})
    agree = at.rank_agreement(edit, ko) if ko else None
    res = schema.result("explain", session, patient=patient, anchor=a,
                        settings={"program": program, "design": design, "knockout": bool(ko),
                                  "generator_top": generator_top, "n": n if generator_top else None},
                        outputs={"probe": {"logit": f0, "risk": float(1 / (1 + np.exp(-f0)))},
                                 "groups": rows, "agreement_edit_knockout": agree,
                                 "generator_original": gen.get("_original")},
                        caveats=caveats, evidence_keys=("refs", "methods"))
    return res


def generator_effects(session, patient, a, program: str, edits: dict, top: list, n: int, caps: dict | None,
                      seed: int, job_dir=None) -> dict:
    """The program's risk from rollouts of the original prompt and of each top group's edit, and its change
    (original minus edit: positive when the group raised the risk)."""
    from icutwin.eval import bench
    from icutwin.toolkit import forecast
    from icutwin.toolkit.inputs import Anchor

    v = session.vocab
    anchors = [a] + [Anchor(f"{a.id}|edit:{g}", patient, a.hours, edits[g],
                            int(bench.anchor_time(edits[g], v, patient.stay_id))) for g in top]
    caps = caps or session.caps(48.0)
    job = session.job("explain_generator", {"patient": patient.ref, "hours": a.hours, "program": program,
                                            "groups": top, "n": n, "seed": seed}, job_dir)
    e2 = program in forecast.E2_PROGRAMS
    raw = forecast.pipeline(session, anchors, job, n, caps, seed, e2=e2, labels=not e2)

    def risk(x):
        if e2:
            return x["e2"]["programs"][program]["e2"]
        r = x["labels"]["binary"].get(program)
        return None if r is None else r["p"]

    r0 = risk(raw[a.id])
    out = {"_original": r0}
    for g, x in zip(top, anchors[1:], strict=True):
        r = risk(raw[x.id])
        out[g] = None if r0 is None or r is None else float(r0 - r)
    return out
