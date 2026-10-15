"""The twin engine (plan S7, ``TwinEngine``): conditioned, grammar-constrained sampling of ICU stays with vLLM, decoded
to stays (events with times and values, ``tokenization.decode``).

``sample(requests)``: each request fixes any of the dataset, the site, header fields and requested outcomes (a goal
block); the grammar mask (``generate.constrain``) makes every sample a valid first window, and the engine's budget
rule ends it within the model's context. Runs in vLLM's environment (``vllm-env``), on one GPU per engine.

Event-type goals are guaranteed by obligation registers (``obligations``): a requested IMV, vasopressor, RRT or MCS
must occur before the end block (or never, for 'no'), and a requested ICU or hospital outcome is the one drawn. Not
yet: the state-summary continuation of stays longer than the context. An ICU length-of-stay goal is a register too
(``los_band``: the ICU discharge within the band, by the interval tokens' median gaps). ``simulate`` rolls real prefixes forward, optionally under an ``Intervention``
(do(treatment): events written at the cutoff; do(no treatment): tokens withheld from the continuation), which is
interventional within the twin world only (plan *Prediction-task generation* 4). The bank is ``generate.producer``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from icutwin.generate.constrain import Conditions
from icutwin.tokenization.fsm import compile_tables
from icutwin.tokenization.spec import load_spec
from icutwin.tokenization.vocab import Vocab


@dataclass
class Request:
    """What a sample is conditioned on, by label: ``dataset`` / ``site`` (prefix values), ``prefix`` (other prefix
    fields -> label, e.g. sex and age: drawn in the spec's order, so fixing a field conditions every later one),
    ``header`` (field -> label), ``goals`` (goal field -> label)."""
    dataset: str | None = None
    site: str | None = None
    header: dict = field(default_factory=dict)
    goals: dict = field(default_factory=dict)
    prefix: dict = field(default_factory=dict)


# CLIF 3.0's med_group 'vasoactives' (medication_admin_continuous), the derived vasopressor episode's drugs
VASOACTIVES = ("angiotensin", "dobutamine", "dopamine", "epinephrine", "isoproterenol", "milrinone", "norepinephrine",
               "phenylephrine", "vasopressin")


def _events(v: Vocab, pred) -> frozenset:
    return frozenset(i for i, m in enumerate(v.meta) if m.get("cls") == "event" and pred(m.get("code") or "", m))


def obligations(v: Vocab, goals: dict) -> tuple[tuple, tuple, frozenset]:
    """Obligation registers of event-type goals (plan S7): (token sets required in the timeline, in the end block,
    tokens forbidden). imv / vasopressor / rrt / mcs 'yes' requires one of their events (an IMV device record, a
    vasoactive infusion's start or presence, a CRRT or IHD record, an MCS record) and 'no' forbids them; an ICU or
    hospital outcome requires its own token and forbids the others. The LOS band and the value-type goals (KDIGO,
    SOFA) are conditioning only (model-dependent; rejection sampling where a guarantee is needed)."""
    sets = {"imv": _events(v, lambda c, m: c == "RESP//device_category//imv"),
            "vasopressor": _events(v, lambda c, m: c.startswith("MED_CON//") and c.split("//")[1] in VASOACTIVES
                                   and m.get("action") in ("start", "going")),
            "rrt": _events(v, lambda c, m: c.startswith(("CRRT//", "IHD//"))),
            "mcs": _events(v, lambda c, m: c.startswith("MCS//"))}
    require, require_end, forbid = [], [], set()
    for f, label in goals.items():
        if f in sets and label in ("yes", "no"):
            (require.append(sets[f]) if label == "yes" else forbid.update(sets[f]))
        elif f == "icu_outcome" and label != "unknown":
            require.append(_events(v, lambda c, m: c == f"ICU//discharge//{label}"))
            forbid.update(_events(v, lambda c, m: c.startswith("ICU//discharge//") and c != f"ICU//discharge//{label}"))
        elif f == "hosp_outcome" and label != "unknown":
            end = [i for i, m in enumerate(v.meta) if m.get("cls") == "end"
                   and (m.get("code") or "").startswith("HOSP//discharge_category//")]
            require_end.append(frozenset(i for i in end if v.meta[i]["code"] == f"HOSP//discharge_category//{label}"))
            forbid.update(i for i in end if v.meta[i]["code"] != f"HOSP//discharge_category//{label}")
    return tuple(require), tuple(require_end), frozenset(forbid)


@dataclass(frozen=True)
class Intervention:
    """do(...) for rollouts (plan S7 ``simulate``): ``give`` event tokens written into the prefix at the cutoff as one
    time group after the ``gap`` interval (the prefix ends at its last event before the cutoff), and ``withhold``
    tokens never drawn in the continuation. Interventional within the twin world only: effects inherit the
    observational data's confounding by indication."""
    give: tuple = ()
    withhold: frozenset = frozenset()
    gap: str = "INT//1-5m"


def treatment_tokens(v: Vocab, name: str) -> frozenset:
    """The tokens of a treatment, as the obligation registers read them: 'vasopressor' (a vasoactive's start or a
    running one), 'imv' (an IMV device record), 'rrt' (a CRRT or IHD record), 'mcs'."""
    req, _end, _forbid = obligations(v, {name: "yes"})
    return req[0]


def intervene(prefix: list[int], iv: Intervention, v: Vocab, t) -> list[int]:
    """The rollout prompt of a prefix under ``iv``: the given events (canonical order) after the gap; checked by the
    grammar automaton, which must accept it as an unfinished timeline."""
    from icutwin.tokenization.fsm import Automaton

    out = [int(x) for x in prefix]
    if iv.give:
        out += [v.id(iv.gap), *sorted(int(x) for x in iv.give)]
    a = Automaton(t)
    for tok in out:
        if not a.step(tok):
            raise ValueError(f"the intervention does not fit the prefix: {a.violation}")
    if a.phase != "tl" or a.last != "event":
        raise ValueError("a rollout prompt ends on a timeline event")
    return out


def rollout_requests(prefixes: list[list[int]], n: int, seed: int, context: int, v: Vocab, t,
                     interventions=None, max_new: int | None = None) -> tuple[list, list]:
    """Prompts and sampling arguments of ``simulate`` (one intervention for every prefix, one per prefix, or
    none): ``n`` rollouts per prefix, each with its own seed, the grammar budget to the context's end (or to
    ``max_new`` tokens when that is shorter: the budget layer then ends a rollout on the censored path) and the
    withheld tokens as forbidden."""
    if len(prefixes) * n > 1_000_000:           # the seeds below stay distinct within a call
        raise ValueError("at most 1,000,000 rollouts per call")
    if interventions is None or isinstance(interventions, Intervention):
        interventions = [interventions] * len(prefixes)
    prompts, args = [], []
    for k, (pre, iv) in enumerate(zip(prefixes, interventions, strict=True)):
        prompt = intervene(pre, iv, v, t) if iv is not None else [int(x) for x in pre]
        budget = context - len(prompt) - 1 if max_new is None else min(max_new, context - len(prompt) - 1)
        extra = {"max_new": budget}
        if iv is not None and iv.withhold:
            extra["conditions"] = Conditions(forbid=frozenset(int(x) for x in iv.withhold)).to_args()
        for j in range(n):
            prompts.append(prompt)
            args.append({"max_tokens": budget, "seed": seed * 1_000_000 + k * n + j, "icutwin": extra})
    return prompts, args


def los_band(v: Vocab, label: str):
    """The register of an ICU length-of-stay goal label ('24-48', '336+'; hours) for this vocabulary: its bounds in
    minutes, the ICU admission marker, the ICU discharge tokens and each interval token's median minutes (pooled
    over datasets, as the decoder's and the forecasting reader's deterministic reading)."""
    from icutwin.generate.constrain import LosBand

    lo, _, hi = label.partition("-")
    lo_h = float(lo.rstrip("+"))
    hi_h = float(hi) if hi else float("inf")
    sk = v.interval_sketches["*"]
    minutes = tuple((i, float(sk[m["interval"]][len(sk[m["interval"]]) // 2]) if sk[m["interval"]] else 0.0)
                    for i, m in enumerate(v.meta) if m.get("cls") == "interval")
    marker = next(i for i, m in enumerate(v.meta) if (m.get("code") or "") == "ICU//admission")
    discharge = frozenset(i for i, m in enumerate(v.meta) if m.get("cls") == "event"
                          and (m.get("code") or "").startswith("ICU//discharge//"))
    return LosBand(lo_h * 60, hi_h * 60, marker, discharge, minutes)


# the derived vasopressor episode's drugs (meds.derived.VASOPRESSORS: norepinephrine-equivalent drugs and
# angiotensin, not the inotropes), which the goal block's 'vasopressor' label reads
GOAL_VASOPRESSORS = ("norepinephrine", "epinephrine", "phenylephrine", "dopamine", "vasopressin", "angiotensin")


def goal_plan(v: Vocab, adopted: dict | None = None):
    """Goal-first sampling (round 3, ``constrain.Conditions.draw_goals``): what each drawn goal binds
    (``constrain.GoalPlan``) by the rules of ``generate.goal_bindings`` (as the goal block's labels are derived,
    ``tokenization.stays._goal_from_meta``: a treatment drawn 'no' has none of its tokens after the ICU admission
    marker, the vasopressor set being the derived episode's drugs; an ICU outcome forbids the other ICU discharges, a
    first window possibly ending before any; a hospital outcome requires its end token and forbids the others), in
    the datasets that adopt each binding (``adopted``: ``goal_bindings.adopted``; None: all). A treatment drawn 'yes',
    the length-of-stay band and the KDIGO and SOFA bands are conditioning only. A goal is drawn only if the dataset's
    support allows one token of each of its sets (a treatment's events for 'yes', the outcome token)."""
    from icutwin.generate.constrain import GoalPlan
    from icutwin.generate.goal_bindings import rules

    rl = rules(v)
    marker = next((i for i, m in enumerate(v.meta) if (m.get("code") or "") == "ICU//admission"), -1)
    datasets = frozenset(i for i, m in enumerate(v.meta) if m.get("cls") == "prefix" and m.get("field") == "dataset")
    plan = GoalPlan(marker, datasets, adopted=adopted)
    for g, (kind, toks, need) in rl.items():
        if kind == "forbid_icu":
            plan.forbid_icu[g] = toks
        elif kind == "forbid":
            plan.forbid[g] = toks
        else:
            plan.require_end[g] = (need,)
            plan.forbid[g] = toks
            plan.feasible[g] = (need,)
    for i, m in enumerate(v.meta):
        if m.get("cls") == "goal":
            _g, field_, label = v.tokens[i].split("//", 2)
            if field_ in ("imv", "vasopressor", "rrt", "mcs") and label == "yes":
                no = v.index.get(f"GOAL//{field_}//no")
                if no is not None and no in rl:
                    plan.feasible[i] = (rl[no][1],)
    return plan


def strip_goal(ids: list[int], v: Vocab) -> tuple[list[int], dict]:
    """A goal-first sample without its goal block (the twin carries a stay's own tokens, as real stays) and the drawn
    goals (field -> label)."""
    g, end = v.id("[GOAL]"), v.id("[/GOAL]")
    if g not in ids or end not in ids:
        return list(ids), {}
    a, b = ids.index(g), ids.index(end)
    drawn = dict(v.tokens[t].split("//", 2)[1:] for t in ids[a + 1:b])
    return list(ids[:a]) + list(ids[b + 1:]), drawn


def conditions(v: Vocab, r: Request) -> Conditions:
    """The token conditions of a request (labels as the vocabulary writes them; an unknown label raises), with the
    obligation registers of its event-type goals and of an ICU length-of-stay goal."""
    spec = load_spec()
    prefix = {}
    for i, f in enumerate(spec["prefix"]):
        label = {**r.prefix, "dataset": r.dataset, "site": r.site}.get(f["field"])
        if label is not None:
            prefix[i] = v.id(f"PREFIX//{f['field']}//{label}")
    header = frozenset(v.id(f"HEADER//{k}//{lab}") for k, lab in r.header.items())
    goal = frozenset(v.id(f"GOAL//{k}//{lab}") for k, lab in r.goals.items())
    require, require_end, forbid = obligations(v, r.goals)
    los = r.goals.get("icu_los")
    return Conditions(prefix, header, goal, require, require_end, forbid,
                      los_band(v, los) if los not in (None, "unknown") else None)


class HeaderOrder:
    """Canonical header order for generated stays (round 2, ``docs/m7_authenticity.md``): the model draws the header
    fields in any order (training shuffles them so that any subset can condition the rest), while the encoder
    writes real stays' headers in the spec's field order. The header is a set, so sorting a sample's header tokens
    into the spec's order loses nothing and gives twins the representation of real stays; unsorted, the order alone
    told twins from real stays (real against real with one side's headers shuffled: AUC 0.58 -> 0.94 on the first
    6 hours of the 2% subset)."""

    def __init__(self, v: Vocab):
        from icutwin.model.data import layout

        self.lay = layout(v)
        self.rank = {f["field"]: k for k, f in enumerate(load_spec()["header"])}

    def __call__(self, ids: list[int]) -> list[int]:
        ids = [int(x) for x in ids]
        a = 1 + self.lay.n_prefix
        b = a + self.lay.n_header
        head = ids[a:b]
        if len(head) != self.lay.n_header or not all(t in self.lay.header_field for t in head):
            return ids                     # not a whole header there (a malformed or cut sample): as it is
        return ids[:a] + sorted(head, key=lambda t: self.rank[self.lay.header_field[t]]) + ids[b:]


def header_unknown_rate(model_dir: Path) -> float:
    """The header augmentation rate the model was trained with (``train_config.transforms.header_unknown`` of its
    run's ``FROZEN.json`` or ``config.json``; ``model.data.TransformConfig``'s default when neither records it)."""
    from icutwin.model.data import TransformConfig

    run = Path(model_dir).parent.parent
    for name in ("FROZEN.json", "config.json"):
        f = run / name
        if f.exists():
            cfg = json.loads(f.read_text(encoding="utf-8"))
            cfg = cfg.get("train_config", cfg)
            rate = (cfg.get("transforms") or {}).get("header_unknown")
            if rate is not None:
                return float(rate)
    return TransformConfig().header_unknown


class TwinEngine:
    def __init__(self, model_dir: str | Path, vocab_path: str | Path, gpu_memory_utilization: float = 0.85,
                 dtype: str = "bfloat16", seed: int = 0, max_model_len: int | None = None):
        from vllm import LLM

        from icutwin.generate.vllm_grammar import GrammarLogitsProcessor

        os.environ["ICUTWIN_VOCAB"] = str(vocab_path)
        os.environ["ICUTWIN_HEADER_UNKNOWN"] = str(header_unknown_rate(Path(model_dir)))
        support = Path(vocab_path).parent / "support.json"      # generate.support, beside the vocabulary
        os.environ["ICUTWIN_SUPPORT"] = str(support) if support.exists() else ""
        self.v = Vocab.load(Path(vocab_path))
        self.t = compile_tables(self.v)
        self.header_order = HeaderOrder(self.v)
        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        self.context = int(max_model_len or cfg["max_position_embeddings"])
        self.llm = LLM(model=str(model_dir), skip_tokenizer_init=True, dtype=dtype, max_model_len=self.context,
                       gpu_memory_utilization=gpu_memory_utilization, logits_processors=[GrammarLogitsProcessor],
                       seed=seed, enable_prefix_caching=False)

    def sample(self, requests: list[Request], seeds: list[int], max_new: int | None = None,
               draw_goals: bool = False) -> list[list[int]]:
        """One stay per request (token ids from ``[BOS]`` to ``[EOS]``), ancestral at temperature 1 without
        truncation (plan S7: the bank keeps the tails), its header in the spec's order (``HeaderOrder``). With
        ``draw_goals`` (goal-first sampling) the model draws a whole goal block after the header, which binds the
        stay; the samples keep it (``strip_goal`` removes it)."""
        from dataclasses import replace

        from vllm import SamplingParams

        bos, eos = self.v.id("[BOS]"), self.v.id("[EOS]")
        max_new = max_new or self.context - 1
        params = [SamplingParams(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, max_tokens=max_new, seed=s,
                                 stop_token_ids=[eos], detokenize=False,
                                 extra_args={"icutwin": {"max_new": max_new, "conditions": replace(
                                     conditions(self.v, r), draw_goals=draw_goals).to_args()}})
                  for r, s in zip(requests, seeds, strict=True)]
        outs = self.llm.generate([{"prompt_token_ids": [bos]} for _ in requests], params, use_tqdm=False)
        return [self.header_order([bos, *o.outputs[0].token_ids]) for o in outs]

    def simulate(self, prefixes: list[list[int]], n: int, seed: int = 0, interventions=None,
                 max_new: int | None = None) -> list[list[list[int]]]:
        """Monte Carlo rollouts (plan S7 ``simulate``): ``n`` continuations of each prefix (a stay as known at a
        cutoff, ``tokenization.encode.prospective``), grammar-constrained, each ended within the context (or within
        ``max_new`` tokens, on the censored path), under an optional ``Intervention`` (one for all prefixes or one
        per prefix); the prompt's tokens past the prefix (the given events) and the continuation."""
        from vllm import SamplingParams

        eos = self.v.id("[EOS]")
        prompts, args = rollout_requests(prefixes, n, seed, self.context, self.v, self.t, interventions, max_new)
        params = [SamplingParams(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, max_tokens=a["max_tokens"],
                                 seed=a["seed"], stop_token_ids=[eos], detokenize=False,
                                 extra_args={"icutwin": a["icutwin"]}) for a in args]
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        cont = [prompts[i][len(prefixes[i // n]):] + list(o.outputs[0].token_ids) for i, o in enumerate(outs)]
        return [cont[k * n:(k + 1) * n] for k in range(len(prefixes))]

    def decode(self, ids: list[int], rng=None):
        from icutwin.tokenization.decode import decode

        return decode(ids, self.v, rng)
