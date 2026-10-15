"""Constrained decoding with the grammar automaton (plan S7, *Structured-output guarantee*): at every step the
logits of the tokens the automaton does not accept are set to -inf, so sampling renormalizes over the rest.

On top of the grammar (``tokenization.fsm.allowed``), generation rules of the sampler:
* a new stay is a first window: no ``[CONT]`` or ``[STATE]`` (continuation windows are built by the engine from its
  registers);
* conditions (:class:`Conditions`): fixed prefix fields, header fields drawn first, and a goal block after the
  header (requested outcomes are conditioning, never generated: without a request there is no goal block);
* goal-first sampling (``draw_goals``, round 3): the model draws a whole goal block after the header, every goal field
  once; the stay's outcomes come from one prediction each, and the timeline keeps those its dataset's real stays
  always keep (``GoalPlan``, ``generate.goal_bindings``); the others are conditioning;
* budget: when the tokens left can only hold the end, the timeline takes ``[END]`` (once the grammar allows it),
  the end block ``[EOS]``; with no room left at all the sequence is cut (a censored stay, reported).
The Hugging Face wrapper (:class:`GrammarProcessor`) is the reference implementation; the vLLM processor and the
engine's own sampler apply the same masks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from icutwin.tokenization.fsm import Automaton, MaskTables, Tables, allowed

END_RESERVE = 32   # tokens kept for the end block (outcome, death, diagnoses and [EOS]) when the budget runs out


@dataclass(frozen=True)
class LosBand:
    """An ICU length-of-stay goal as a register (plan S7): minutes since the ICU admission marker, counted with each
    interval token's median gap (the decoder's deterministic reading); the ICU discharge waits until the count
    reaches ``lo`` and is due before the count passes ``hi`` (no interval that would overshoot is drawn)."""
    lo: float
    hi: float
    marker: int
    discharge: frozenset
    minutes: tuple                   # (interval token, median minutes)

    def to_args(self) -> dict:
        return {"lo": self.lo, "hi": None if self.hi == float("inf") else self.hi, "marker": int(self.marker),
                "discharge": sorted(map(int, self.discharge)), "minutes": [[int(k), float(m)] for k, m in self.minutes]}

    @classmethod
    def from_args(cls, a: dict | None):
        if not a:
            return None
        return cls(float(a["lo"]), float("inf") if a.get("hi") is None else float(a["hi"]), int(a["marker"]),
                   frozenset(a["discharge"]), tuple((int(k), float(m)) for k, m in a["minutes"]))


@dataclass(frozen=True)
class Conditions:
    """What the engine fixes in a sample (plan S7 ``sample(header=..., goals=...)``), enforced by the mask:
    ``prefix`` maps prefix field positions to their token (the dataset, the site, ...); ``header`` holds header
    tokens drawn before any other header field (the header is trained in any order, so the model completes the
    rest given these); ``goal`` holds the tokens of a goal block written after the header ([GOAL] ... [/GOAL]:
    requested outcomes, conditioning only)."""
    prefix: dict = field(default_factory=dict)
    header: frozenset = frozenset()
    goal: frozenset = frozenset()
    require: tuple = ()              # obligation registers (event-type goals): token sets, each drawn in the timeline
    require_end: tuple = ()          # ... or in the end block, before it closes
    forbid: frozenset = frozenset()  # tokens never drawn
    los: LosBand | None = None       # the ICU length-of-stay band, as a register
    draw_goals: bool = False         # round 3: the model draws a whole goal block, which then binds the stay

    def to_args(self) -> dict:
        out = {"prefix": {str(k): int(v) for k, v in self.prefix.items()}, "header": sorted(map(int, self.header)),
               "goal": sorted(map(int, self.goal)), "require": [sorted(map(int, r)) for r in self.require],
               "require_end": [sorted(map(int, r)) for r in self.require_end], "forbid": sorted(map(int, self.forbid))}
        if self.los is not None:
            out["los"] = self.los.to_args()
        if self.draw_goals:
            out["draw_goals"] = True
        return out

    @classmethod
    def from_args(cls, a: dict | None) -> Conditions:
        a = a or {}
        return cls({int(k): int(v) for k, v in (a.get("prefix") or {}).items()}, frozenset(a.get("header") or ()),
                   frozenset(a.get("goal") or ()), tuple(frozenset(r) for r in a.get("require") or ()),
                   tuple(frozenset(r) for r in a.get("require_end") or ()), frozenset(a.get("forbid") or ()),
                   LosBand.from_args(a.get("los")), bool(a.get("draw_goals")))


@dataclass(frozen=True)
class GoalPlan:
    """What a drawn goal binds (goal-first sampling, ``engine.goal_plan``; ``generate.goal_bindings``): per goal
    token, tokens never drawn after the ICU admission ``marker`` (a treatment drawn 'no'), tokens never drawn (the
    other ICU or hospital outcomes) and token sets of the end block (the drawn hospital outcome); ``feasible``: token
    sets of which the dataset's support must allow one for the goal to be drawn at all. A goal binds only in the
    datasets that adopt it (``adopted``: prefix dataset token -> goal tokens; None: every dataset adopts every
    binding); elsewhere, and for a treatment drawn 'yes', the goal is conditioning only."""
    marker: int
    datasets: frozenset
    forbid_icu: dict = field(default_factory=dict)
    forbid: dict = field(default_factory=dict)
    require_end: dict = field(default_factory=dict)
    feasible: dict = field(default_factory=dict)
    adopted: dict | None = None

    def binds(self, dataset: int | None, tok: int) -> bool:
        return self.adopted is None or tok in self.adopted.get(dataset, frozenset())


class Obligations:
    """A sample's obligation registers: which required token sets it has drawn so far (fed every token). With
    ``draw_goals`` the registers grow as the goal block is drawn (``plan``: a :class:`GoalPlan`)."""

    def __init__(self, cond: Conditions, plan: GoalPlan | None = None):
        self.cond = cond
        self.plan = plan if cond.draw_goals else None
        self.require, self.require_end = list(cond.require), list(cond.require_end)
        self.forbid, self.los = frozenset(cond.forbid), cond.los
        self.forbid_icu, self.in_icu, self.dataset = frozenset(), False, None
        self.done = [False] * len(self.require)
        self.done_end = [False] * len(self.require_end)
        self.forced = False              # the budget closed the stay before an obligation was met
        self.elapsed = None              # LOS register: minutes since the ICU admission marker (None before it)
        self.discharged = False
        self.minutes = dict(cond.los.minutes) if cond.los is not None else {}

    def feed(self, tok: int) -> None:
        p = self.plan
        if p is not None:
            if self.dataset is None and tok in p.datasets:
                self.dataset = tok                               # the prefix's dataset: which bindings it adopts
            elif tok == p.marker:
                self.in_icu = True
            elif (tok in p.forbid_icu or tok in p.forbid or tok in p.require_end) and p.binds(self.dataset, tok):
                req_end = p.require_end.get(tok, ())             # a drawn goal: what its label binds, from here on
                self.require_end += list(req_end)
                self.done_end += [False] * len(req_end)
                self.forbid_icu = self.forbid_icu | p.forbid_icu.get(tok, frozenset())
                self.forbid = self.forbid | p.forbid.get(tok, frozenset())
        for i, r in enumerate(self.require):
            self.done[i] = self.done[i] or tok in r
        for i, r in enumerate(self.require_end):
            self.done_end[i] = self.done_end[i] or tok in r
        los = self.los
        if los is not None:
            if tok == los.marker and self.elapsed is None:
                self.elapsed = 0.0
            elif self.elapsed is not None and tok in self.minutes:
                self.elapsed += self.minutes[tok]
            if tok in los.discharge:
                self.discharged = True

    def open_timeline(self) -> bool:
        return not all(self.done) or (self.los is not None and not self.discharged)

    def open_end(self) -> bool:
        return not all(self.done_end)


NONE = Conditions()


def _only(mask: np.ndarray, *toks) -> np.ndarray:
    out = np.zeros_like(mask)
    for tok in toks:
        out[tok] = True
    return out


def generation_mask(a: Automaton, m: MaskTables, left: int | None = None, cond: Conditions = NONE,
                    obl: Obligations | None = None, support: np.ndarray | None = None) -> np.ndarray:
    """The tokens the sampler may draw: the grammar's, without the blocks only the engine writes, under the
    sample's conditions and obligations, and with the budget rule when ``left`` (tokens still allowed after this one)
    is given (the budget wins over an obligation: such a sample is marked forced)."""
    sp, t = a.t.special, a.t
    mask = allowed(a, m)
    if support is not None:                 # the dataset's support (generate.support); conditions below still win
        mask &= support[:len(mask)]
    forbid = cond.forbid if obl is None else obl.forbid | (obl.forbid_icu if obl.in_icu else frozenset())
    if forbid:                                                   # with drawn goals, theirs too (in the ICU)
        mask[np.fromiter(forbid, dtype=np.int64)] = False
    if a.phase == "prefix" and a.i in cond.prefix:
        return _only(mask, cond.prefix[a.i]) if mask[cond.prefix[a.i]] else mask & False
    if a.phase == "header":
        pending = [tok for tok in cond.header if t.header[tok] not in a.seen]
        if pending:
            return _only(mask, *pending)
        mask[sp["[CONT]"]] = mask[sp["[STATE]"]] = False
        if len(a.seen) >= t.n_header:          # the header is complete: the goal block when asked, else the timeline
            return _only(mask, sp["[GOAL]"] if cond.goal or cond.draw_goals else sp["[TL]"])
        mask[sp["[GOAL]"]] = mask[sp["[TL]"]] = False
        return mask
    if a.phase == "goal" and cond.draw_goals:     # the model's own goal block: every field once, then [/GOAL]
        if len(a.seen) < len(m.goal_fields):
            mask[sp["[/GOAL]"]] = False
        if obl is not None and obl.plan is not None:
            for g in np.flatnonzero(mask):        # no goal the stay could not keep: a set without a token the
                if any(not r or (support is not None and not support[np.fromiter(r, dtype=np.int64)].any())
                       for r in obl.plan.feasible.get(int(g), ())):  # dataset's support allows
                    mask[g] = False
        return mask
    if a.phase == "goal":
        pending = [tok for tok in cond.goal if t.goal[tok] not in a.seen]
        return _only(mask, *pending) if pending else _only(mask, sp["[/GOAL]"])
    if a.phase == "after_goal":
        return _only(mask, sp["[TL]"])
    if left is not None:
        if a.phase == "tl" and left <= END_RESERVE:
            if mask[sp["[END]"]]:
                if obl is not None and obl.open_timeline():
                    obl.forced = True
                return _only(mask, sp["[END]"])
            # right after an interval [END] is not allowed: no further interval (a repeatable week could go on to
            # the end of the budget), so the next event makes [END] allowed
            ints = mask.copy()
            mask[np.fromiter(t.interval, dtype=np.int64)] = False
            if not mask.any():
                mask = ints
        if a.phase == "end" and left <= 1:
            if obl is not None and obl.open_end():
                obl.forced = True
            return _only(mask, sp["[EOS]"])
    if obl is not None:
        los = obl.los
        if los is not None and a.phase == "tl" and not obl.discharged:
            if obl.elapsed is None or obl.elapsed < los.lo:      # no discharge before the admission and the band
                mask[np.fromiter(los.discharge, dtype=np.int64)] = False
            if obl.elapsed is not None:
                over = [k for k, m in los.minutes if obl.elapsed + m >= los.hi]
                if over:
                    mask[np.array(over, dtype=np.int64)] = False
        if a.phase == "tl" and obl.open_timeline():
            mask[sp["[END]"]] = False
        if a.phase == "end" and obl.open_end():
            mask[sp["[EOS]"]] = False
    return mask


class HeaderAugmentation:
    """The inverse of training's header augmentation (``model.data.transform``: each header field set to unknown with
    probability ``rate``, independently): the model learned P(unknown | context) = rate + (1 - rate) P_real(unknown |
    context), so free sampling would emit unknown header values at about ``rate`` more than the data has (the
    pilot's discriminator found them). Within each field not yet drawn, ``rate`` x the field's probability is taken
    from its unknown token and the field keeps its total, so which field comes next is unchanged and each field's
    value follows P_real. Applies in the header phase only."""

    def __init__(self, v, rate: float):
        self.rate = float(rate)
        self.unknown = {m["field"]: i for i, m in enumerate(v.meta) if m.get("cls") == "header"
                        and m.get("value") == "unknown"}
        fields: dict = {}
        for i, m in enumerate(v.meta):
            if m.get("cls") == "header":
                fields.setdefault(m["field"], []).append(i)
        self.fields = {f: np.array(toks, dtype=np.int64) for f, toks in fields.items()}

    def __call__(self, logits: np.ndarray, a: Automaton) -> np.ndarray:
        if self.rate <= 0 or a.phase != "header":
            return logits
        ok = np.isfinite(logits)
        if not ok.any():
            return logits
        p = np.zeros(len(logits), dtype=np.float64)
        p[ok] = np.exp(logits[ok] - logits[ok].max())
        p /= p.sum()
        for f, u in self.unknown.items():
            if f in a.seen or p[u] <= 0:
                continue
            toks = self.fields[f]
            q = p[toks].sum()
            new_u = max(p[u] - self.rate * q, 0.0)
            rest = q - p[u]
            if rest + new_u <= 0:
                continue
            p[toks] *= q / (rest + new_u)
            p[u] = new_u * q / (rest + new_u)
        out = np.full(len(logits), -np.inf, dtype=logits.dtype)
        pos = p > 0
        out[pos] = np.log(p[pos])
        return out


class GrammarProcessor:
    """A transformers ``LogitsProcessor``: one automaton per row, fed the prompt and every generated token; rows
    that have ended (``[EOS]``) may only take the pad token, as ``generate`` pads them."""

    def __init__(self, t: Tables, m: MaskTables, pad: int, max_length: int | None = None,
                 conditions: list[Conditions] | Conditions = NONE, header_augmentation: HeaderAugmentation | None = None,
                 support=None, goal_plan: dict | None = None):
        self.t, self.m, self.pad, self.max_length = t, m, pad, max_length
        self.conditions = conditions
        self.undo = header_augmentation
        self.support = support               # generate.support.Support: each row's dataset support
        self.goal_plan = goal_plan           # engine.goal_plan: what a drawn goal binds (draw_goals)
        self.dataset_token: list | None = None
        self.obligations: list[Obligations] | None = None
        self.automata: list[Automaton] | None = None
        self.consumed = 0
        self.violations = 0

    def __call__(self, input_ids, scores):
        import torch

        ids = input_ids.detach().cpu().numpy()
        if self.automata is None:
            self.automata = [Automaton(self.t) for _ in range(len(ids))]
            self.dataset_token = [None] * len(ids)
            self.obligations = [Obligations(self.conditions[r] if isinstance(self.conditions, list) else self.conditions,
                                            self.goal_plan) for r in range(len(ids))]
        masks = np.zeros((len(ids), scores.shape[-1]), dtype=bool)
        for r, a in enumerate(self.automata):
            if a.phase != "done" and a.violation is None:
                for tok in ids[r, self.consumed:]:
                    if a.phase == "prefix" and a.i == 0:
                        self.dataset_token[r] = int(tok)
                    if not a.step(int(tok)):
                        self.violations += 1        # a prompt the grammar does not accept: the row is stopped
                        break
                    self.obligations[r].feed(int(tok))
            if a.phase == "done" or a.violation is not None:
                masks[r, self.pad] = True
            else:
                left = None if self.max_length is None else self.max_length - ids.shape[1] - 1
                cond = self.conditions[r] if isinstance(self.conditions, list) else self.conditions
                sup = self.support.mask(self.dataset_token[r]) if self.support is not None else None
                masks[r, :self.m.n] = generation_mask(a, self.m, left, cond, self.obligations[r], sup)
                if not masks[r].any():
                    masks[r, self.pad] = True
        self.consumed = ids.shape[1]
        out = scores.masked_fill(~torch.from_numpy(masks).to(scores.device), float("-inf"))
        if self.undo is not None:
            for r, a in enumerate(self.automata):
                if a.phase == "header" and a.violation is None:
                    row = out[r, :self.m.n].float().cpu().numpy()
                    out[r, :self.m.n] = torch.from_numpy(self.undo(row, a)).to(out.device, out.dtype)
        return out
