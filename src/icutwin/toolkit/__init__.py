"""The ICU digital twin's application toolkit (Paper P, ``docs/toolkit.md``): the validated capabilities of the twin
model as tools a researcher, a trial designer or a teacher runs on a patient or a cohort.

* ``Session``: one model, its tokens and its gate (``generate.producer.require_gate``, checked once), the token caps,
  the treatment registry and the backends; ``Session.patient`` reads a held-out stay, a bank twin or a MEDS stay.
* Patient-level tools: ``forecast`` (risks of the label engine's programs, E2 incidence curves, value forecasts,
  remaining stay), ``whatif`` (give or withhold a treatment against an A/A band), ``explain`` (probe editing and
  attention knockout over the history's event groups), ``similar`` (outcomes of the nearest held-out stays by the
  model's state).
* Cohort-level tools: ``cohort`` (synthetic cohorts from the bank or the sampler, exported as CLIF 3.0 and MEDS, with
  fidelity and novelty reports) and ``trial`` (eligibility, event rates, prognostic enrichment and prognostic
  covariate adjustment from kept per-anchor predictions; CPU only).
* ``teach``: teaching cases from bank twins (decision points with precomputed what-if branches, learner and teacher
  cards, a prognosis drill).
* ``card`` / ``figures``: self-contained HTML cards and PNG/PDF figures of any result.

Every result is JSON with its provenance (weights, gate status and scope, seeds, sizes, caps) and, per output, the
paper evidence it rests on and its caveats (``evidence``). What-if outputs are observational simulation within the
twin world, not causal effects. Outputs computed from real stays stay on the approved host (the data-use
agreements); figures and teaching cases use twins.

Sampling runs in vLLM's environment (a subprocess, ``toolkit.stage``) or, as the reference, with transformers in
process (``backends.HfSampler``); the teacher-forced passes (E2 hazards, states, knockout) and the label engine run in
the main environment. Modules import their heavy dependencies inside functions, so ``toolkit.stage`` imports in
vLLM's environment, which has no pandas.
"""

from __future__ import annotations

__all__ = ["Session"]


def __getattr__(name: str):
    if name == "Session":
        from icutwin.toolkit.session import Session

        return Session
    raise AttributeError(name)
