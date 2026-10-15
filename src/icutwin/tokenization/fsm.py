"""The grammar automaton (D4): token-by-token acceptance with registers, compiled from the spec and vocabulary.

Rules (``grammar.enforce`` of the spec):
* block_order: ``[BOS]`` prefix fields in spec order, header fields, optional ``[GOAL] ... [/GOAL]``, optional
  ``[CONT]``, optional ``[STATE] ... [/STATE]``, ``[TL]``, the timeline, then ``[END]`` end block ``[EOS]`` (a
  window that is not the stay's last ends inside the timeline, at an event boundary);
* field_once: each header and goal field at most once; every header field before the timeline; in the end
  block the hospital outcome and death at most once each, before the diagnoses (tokens of the spec's
  diagnosis domains: a hierarchical system token with its sub-tokens, or a whole code of another system);
* interval_between_groups: no interval first in a timeline (except in a continuation window) or last; after
  an interval other than the 7-day token comes an event;
* group_order: within a group (and among diagnoses) events in canonical order: their first token ids do not
  decrease;
* infusion_state: an action the spec lists as 'continues' or 'ends' (dose change, stop) of an infusion key
  only while the key runs; a 'begins' action (start, presence) or a state token makes it run, an 'ends' action
  stops it;
* clock (round 3, spec ``clock``; only with a vocabulary that has its tokens): after an interval token that takes
  the ICU clock into a new ICU day comes exactly the clock block the registers give (the day, each treatment's
  state); nowhere else, and not in a continuation window. The encoder writes it with :func:`insert_clock`, the
  sampler draws it as the only allowed token.
Hierarchical events are a system token followed by sub-tokens of increasing level. The independent parser
(:mod:`icutwin.tokenization.parser`) checks the same rules with separate code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from icutwin.tokenization.spec import load_spec
from icutwin.tokenization.vocab import Vocab


@dataclass(frozen=True)
class Tables:
    """Token classes compiled from the vocabulary metadata."""
    special: dict
    prefix: tuple                  # per prefix field: frozenset of its tokens
    header: dict                   # header token -> field
    goal: dict                     # goal token -> field
    n_header: int
    state: frozenset
    state_key: dict                # infusion state token -> key
    interval: frozenset
    week: int                      # the repeating >= 7 days interval token
    event_first: frozenset         # tokens that start a timeline event
    system: frozenset              # hierarchical system tokens (timeline and end block)
    chunk_level: dict              # sub-token -> level (1, 2, 3)
    end_stage: dict                # end token -> stage (0 hospital outcome, 1 death, 2 diagnoses); -1 OOV
    dx_stage: int                  # the diagnoses' stage
    key: dict                      # infusion event token -> (key, action)
    begins: frozenset              # infusion actions of the spec
    needs_running: frozenset
    ends: frozenset
    clock: Clock | None = None     # the clock block; None with a vocabulary without its tokens


@dataclass(frozen=True)
class Clock:
    """The clock block's tables (spec ``clock``)."""
    marker: int
    days: dict                     # day -> token (the last day standing for it and later)
    status: dict                   # (treatment, state) -> token
    treatments: tuple
    tokens: frozenset              # every clock token
    minutes: dict                  # interval token -> minutes (the median gap, as the evaluation reads time)
    adm: frozenset                 # the ICU admission marker
    device: dict                   # respiratory device token -> whether it is IMV
    ever: dict                     # treatment -> its event tokens (the engine's obligation sets)
    vaso_keys: frozenset           # infusion keys of vasoactive drugs
    day_minutes: float
    rrt_minutes: float
    max_day: int


def compile_clock(v: Vocab, spec: dict) -> Clock | None:
    c = spec.get("clock")
    if not c or c["token"] not in v.index:
        return None
    meta = v.meta
    code = [(m.get("code") or "") if m.get("cls") == "event" else "" for m in meta]
    sk = v.interval_sketches["*"]
    minutes = {i: (sk[m["interval"]][len(sk[m["interval"]]) // 2] if sk[m["interval"]] else 0.0)
               for i, m in enumerate(meta) if m.get("cls") == "interval"}
    vaso = set(c["vasoactives"])
    ever = {"imv": frozenset(i for i, x in enumerate(code) if x == c["device_prefix"] + c["imv_device"]),
            "vasopressor": frozenset(i for i, (x, m) in enumerate(zip(code, meta)) if x.startswith("MED_CON//")
                                     and x.split("//")[1] in vaso and m.get("action") in c["vasopressor_actions"]),
            "rrt": frozenset(i for i, x in enumerate(code) if x.split("//", 1)[0] in c["rrt_domains"])}
    keys = {m["key"] for m in meta if m.get("key")}
    return Clock(
        v.id(c["token"]), {m["day"]: i for i, m in enumerate(meta) if m.get("role") == "day"},
        {(m["treatment"], m["state"]): i for i, m in enumerate(meta) if m.get("role") == "status"},
        tuple(c["treatments"]), frozenset(i for i, m in enumerate(meta) if m.get("cls") == "clock"), minutes,
        frozenset(i for i, x in enumerate(code) if x == spec["timeline"]["marker"]["token"]),
        {i: x == c["device_prefix"] + c["imv_device"] for i, x in enumerate(code) if x.startswith(c["device_prefix"])},
        ever, frozenset(k for k in keys if k.startswith("MED_CON//") and k.split("//")[1] in vaso),
        float(c["hours"]) * 60, float(c["rrt_active_hours"]) * 60, int(c["max_day"]))


def compile_tables(v: Vocab) -> Tables:
    spec = load_spec()
    special = {t: v.id(t) for t in spec["special"]}
    meta = v.meta
    prefix = tuple(frozenset(i for i, m in enumerate(meta) if m.get("cls") == "prefix" and m["field"] == f["field"])
                   for f in spec["prefix"])
    header = {i: m["field"] for i, m in enumerate(meta) if m.get("cls") == "header"}
    goal = {i: m["field"] for i, m in enumerate(meta) if m.get("cls") == "goal"}
    state = frozenset(i for i, m in enumerate(meta) if m.get("cls") == "state")
    inf = spec["windows"]["state"]["infusions"]["domain"] + "//"
    state_key = {i: m["key"] for i, m in enumerate(meta) if m.get("cls") == "state" and m["key"].startswith(inf)}
    interval = frozenset(i for i, m in enumerate(meta) if m.get("cls") == "interval")
    end_codes = [e for e in spec["end"] if "code" in e]
    dx_domains = {e["domain"] for e in spec["end"] if "domain" in e}
    end_stage = {}
    for i, m in enumerate(meta):
        if m.get("cls") != "end":
            continue
        if m.get("kind") == "oov":
            end_stage[i] = -1
        elif m.get("kind") == "system" or m["code"].split("//", 1)[0] in dx_domains:
            end_stage[i] = len(end_codes)
        else:
            end_stage[i] = next(k for k, e in enumerate(end_codes) if m["code"].startswith(e["code"].split("{")[0]))
    actions = spec["windows"]["state"]["infusions"]
    return Tables(
        special, prefix, header, goal, len(spec["header"]), state, state_key, interval, max(interval),
        frozenset(i for i, m in enumerate(meta) if m.get("cls") == "event"),
        frozenset(i for i, m in enumerate(meta) if m.get("kind") == "system"),
        {i: m["level"] for i, m in enumerate(meta) if m.get("cls") == "chunk"},
        end_stage, len(end_codes),
        {i: (m["key"], m["action"]) for i, m in enumerate(meta) if m.get("cls") == "event" and "key" in m},
        frozenset(actions["begins"]), frozenset(actions["continues"] + actions["ends"]), frozenset(actions["ends"]),
        compile_clock(v, spec))


@dataclass
class Automaton:
    t: Tables
    phase: str = "start"
    i: int = 0                          # prefix field index
    seen: set = field(default_factory=set)
    cont: bool = False
    running: set = field(default_factory=set)
    last: str = "none"                  # timeline: none | event | week | interval
    group_first: int = -1               # first token id of the last event of the group (or of the diagnoses)
    hier: int = -1                      # level of the last hierarchical token; -1 outside one
    end_stage: int = 0
    end_seen: set = field(default_factory=set)
    violation: str | None = None
    # the clock (Tables.clock): minutes since the sequence start, at the ICU admission marker, the ICU day, the
    # block still to write, and the treatment registers
    clock_t: float = 0.0
    adm_t: float | None = None
    day: int = 1
    pending: list = field(default_factory=list)
    imv_now: bool = False
    ever: set = field(default_factory=set)
    rrt_t: float | None = None

    def _fail(self, why: str) -> bool:
        self.violation = why
        return False

    def step(self, tok: int) -> bool:
        """Consume ``tok``; False (and ``violation``) when the grammar does not allow it here."""
        if self.violation is not None:
            return False
        handler = getattr(self, "_" + self.phase)
        return handler(tok)

    def _start(self, tok: int) -> bool:
        if tok != self.t.special["[BOS]"]:
            return self._fail("a sequence starts with [BOS]")
        self.phase, self.i = "prefix", 0
        return True

    def _prefix(self, tok: int) -> bool:
        if tok not in self.t.prefix[self.i]:
            return self._fail(f"prefix field {self.i} expected")
        self.i += 1
        if self.i == len(self.t.prefix):
            self.phase, self.seen = "header", set()
        return True

    def _header(self, tok: int) -> bool:
        if tok in self.t.header:
            if self.t.header[tok] in self.seen:
                return self._fail("header field repeated")
            self.seen.add(self.t.header[tok])
            return True
        if len(self.seen) < self.t.n_header:
            return self._fail("header incomplete before the timeline")
        return self._blocks(tok, goal=True, cont=True)

    def _goal(self, tok: int) -> bool:
        if tok in self.t.goal:
            if self.t.goal[tok] in self.seen:
                return self._fail("goal field repeated")
            self.seen.add(self.t.goal[tok])
            return True
        if tok == self.t.special["[/GOAL]"] and self.seen:
            self.phase = "after_goal"
            return True
        return self._fail("goal block: goal token or [/GOAL] expected")

    def _after_goal(self, tok: int) -> bool:
        return self._blocks(tok, goal=False, cont=True)

    def _cont(self, tok: int) -> bool:
        return self._blocks(tok, goal=False, cont=False)

    def _blocks(self, tok: int, goal: bool, cont: bool) -> bool:
        sp = self.t.special
        if goal and tok == sp["[GOAL]"]:
            self.phase, self.seen = "goal", set()
        elif cont and tok == sp["[CONT]"]:
            self.phase, self.cont = "cont", True
        elif tok == sp["[STATE]"]:
            self.phase = "state"
        elif tok == sp["[TL]"]:
            self.phase, self.last, self.group_first, self.hier = "tl", "none", -1, -1
        else:
            return self._fail("[GOAL], [CONT], [STATE] or [TL] expected")
        return True

    def _state(self, tok: int) -> bool:
        if tok in self.t.state:
            if tok in self.t.state_key:
                self.running.add(self.t.state_key[tok])
            return True
        if tok == self.t.special["[/STATE]"]:
            self.phase = "pre_tl"
            return True
        return self._fail("state block: state token or [/STATE] expected")

    def _pre_tl(self, tok: int) -> bool:
        if tok != self.t.special["[TL]"]:
            return self._fail("[TL] expected after the state block")
        self.phase, self.last, self.group_first, self.hier = "tl", "none", -1, -1
        return True

    def _track(self, tok: int) -> None:
        """The clock's registers after a timeline event."""
        c = self.t.clock
        if tok in c.adm and self.adm_t is None:
            self.adm_t = self.clock_t
        if tok in c.device:
            self.imv_now = c.device[tok]
        for k, s in c.ever.items():
            if tok in s:
                self.ever.add(k)
        if tok in c.ever["rrt"]:
            self.rrt_t = self.clock_t

    def clock_block(self) -> list[int]:
        """The block for the current ICU day: marker, day, each treatment's state."""
        c = self.t.clock
        out = [c.marker, c.days[min(self.day, c.max_day)]]
        for k in c.treatments:
            if k == "imv":
                active = self.imv_now
            elif k == "vasopressor":
                active = bool(self.running & c.vaso_keys)
            else:
                active = self.rrt_t is not None and self.clock_t - self.rrt_t <= c.rrt_minutes
            out.append(c.status[(k, "active" if active else "ended" if k in self.ever else "never")])
        return out

    def _tick(self, tok: int) -> None:
        """After an interval token: the clock advances; a new ICU day makes its block due."""
        c = self.t.clock
        self.clock_t += c.minutes.get(tok, 0.0)
        if self.adm_t is None or self.cont:
            return
        day = int((self.clock_t - self.adm_t) // c.day_minutes) + 1
        if day > self.day:
            self.day = day
            self.pending = self.clock_block()

    def _tl(self, tok: int) -> bool:
        t = self.t
        if self.pending:
            if tok != self.pending[0]:
                return self._fail("the clock block expected")
            self.pending.pop(0)
            return True
        if tok in t.chunk_level:
            if self.hier < 0 or t.chunk_level[tok] != self.hier + 1:
                return self._fail("sub-token out of place")
            self.hier = t.chunk_level[tok]
            return True
        if tok in t.event_first:
            if tok < self.group_first:
                return self._fail("group order: an event before the previous one")
            if tok in t.key:
                key, action = t.key[tok]
                if action in t.begins:
                    self.running.add(key)
                elif action in t.needs_running:
                    if key not in self.running:
                        return self._fail(f"infusion {action} while not running")
                    if action in t.ends:
                        self.running.discard(key)
            self.group_first, self.last = tok, "event"
            self.hier = 0 if tok in t.system else -1
            if t.clock is not None:
                self._track(tok)
            return True
        if tok in t.interval:
            if self.last == "interval" or (self.last == "none" and not self.cont):
                return self._fail("interval out of place")
            self.last, self.group_first, self.hier = ("week" if tok == t.week else "interval"), -1, -1
            if t.clock is not None:
                self._tick(tok)
            return True
        if tok == t.special["[END]"]:
            if self.last in ("interval", "week"):
                return self._fail("interval before [END]")
            self.phase, self.end_stage, self.group_first, self.hier = "end", 0, -1, -1
            return True
        return self._fail("timeline: event, interval or [END] expected")

    def _end(self, tok: int) -> bool:
        t = self.t
        if tok == t.special["[EOS]"]:
            self.phase = "done"
            return True
        if tok in t.chunk_level:
            if self.hier < 0 or t.chunk_level[tok] != self.hier + 1:
                return self._fail("end block: sub-token out of place")
            self.hier = t.chunk_level[tok]
            return True
        stage = t.end_stage.get(tok)
        if stage is None:
            return self._fail("end block: end token or [EOS] expected")
        self.hier = 0 if tok in t.system else -1
        if stage < 0:
            return True
        if stage < self.end_stage:
            return self._fail("end block order")
        if stage == t.dx_stage:   # diagnoses, in canonical order of their first tokens
            if tok < self.group_first:
                return self._fail("diagnoses out of order")
            self.group_first = tok
        elif stage in self.end_seen:
            return self._fail("end field repeated")
        self.end_seen.add(stage)
        self.end_stage = stage
        return True

    def _done(self, tok: int) -> bool:
        return self._fail("a token after [EOS]")

    def complete(self) -> bool:
        """A whole stay: ended with [EOS]."""
        return self.violation is None and self.phase == "done"

    def window_ok(self) -> bool:
        """A window that is not the stay's last: no violation, and it stops at an event boundary."""
        return self.violation is None and (self.phase == "done" or (self.phase == "tl" and self.last == "event"))


@dataclass(frozen=True)
class MaskTables:
    """The token classes of :class:`Tables` as boolean vectors over the vocabulary, for :func:`allowed`."""
    n: int
    prefix: tuple                  # per prefix field
    header_field: np.ndarray       # field index per header token, -1 elsewhere
    goal_field: np.ndarray         # field index per goal token, -1 elsewhere
    state: np.ndarray
    event: np.ndarray
    chunk_level: np.ndarray        # 1-3 per sub-token, 0 elsewhere
    interval: np.ndarray
    end_stage: np.ndarray          # stage per end token (-1 OOV), -2 elsewhere
    key: np.ndarray                # infusion key index per infusion event token, -1 elsewhere
    needs_running: np.ndarray      # infusion events allowed only while their key runs
    ids: np.ndarray                # token ids (group order)
    fields: tuple                  # header field names, in index order
    goal_fields: tuple
    keys: tuple                    # infusion keys, in index order


def mask_tables(t: Tables, n: int) -> MaskTables:
    vec = lambda ids: np.isin(np.arange(n), np.fromiter(ids, dtype=np.int64, count=len(ids)))  # noqa: E731
    fields = tuple(sorted(set(t.header.values())))
    goal_fields = tuple(sorted(set(t.goal.values())))
    keys = tuple(sorted({k for k, _a in t.key.values()} | set(t.state_key.values())))
    hf, gf, cl, es, key = (np.full(n, -1, dtype=np.int64) for _ in range(5))
    cl[:] = 0
    es[:] = -2
    for i, f in t.header.items():
        hf[i] = fields.index(f)
    for i, f in t.goal.items():
        gf[i] = goal_fields.index(f)
    for i, lv in t.chunk_level.items():
        cl[i] = lv
    for i, s in t.end_stage.items():
        es[i] = s
    nr = np.zeros(n, dtype=bool)
    for i, (k, action) in t.key.items():
        key[i] = keys.index(k)
        nr[i] = action in t.needs_running and action not in t.begins
    return MaskTables(n, tuple(vec(p) for p in t.prefix), hf, gf, vec(t.state), vec(t.event_first), cl,
                      vec(t.interval), es, key, nr, np.arange(n), fields, goal_fields, keys)


def allowed(a: Automaton, m: MaskTables) -> np.ndarray:
    """Every token :meth:`Automaton.step` accepts in ``a``'s state, as a boolean vector (the mask of constrained
    decoding; tested against ``step`` token by token)."""
    t, sp = a.t, a.t.special
    out = np.zeros(m.n, dtype=bool)
    if a.violation is not None or a.phase == "done":
        return out

    def specials(*names):
        for nm in names:
            out[sp[nm]] = True

    if a.phase == "start":
        specials("[BOS]")
    elif a.phase == "prefix":
        out |= m.prefix[a.i]
    elif a.phase in ("header", "goal"):
        fld, names = (m.header_field, m.fields) if a.phase == "header" else (m.goal_field, m.goal_fields)
        seen = np.array([f in a.seen for f in names], dtype=bool)
        out |= (fld >= 0) & ~seen[np.maximum(fld, 0)]
        if a.phase == "goal":
            if a.seen:
                specials("[/GOAL]")
        elif len(a.seen) >= t.n_header:
            specials("[GOAL]", "[CONT]", "[STATE]", "[TL]")
    elif a.phase == "after_goal":
        specials("[CONT]", "[STATE]", "[TL]")
    elif a.phase == "cont":
        specials("[STATE]", "[TL]")
    elif a.phase == "state":
        out |= m.state
        specials("[/STATE]")
    elif a.phase == "pre_tl":
        specials("[TL]")
    elif a.phase == "tl" and a.pending:      # the clock block: written, not chosen
        out[a.pending[0]] = True
    elif a.phase == "tl":
        if a.hier >= 0:
            out |= m.chunk_level == a.hier + 1
        running = np.array([k in a.running for k in m.keys], dtype=bool)
        ok_inf = ~m.needs_running | ((m.key >= 0) & running[np.maximum(m.key, 0)])
        out |= m.event & (m.ids >= a.group_first) & ok_inf
        if not (a.last == "interval" or (a.last == "none" and not a.cont)):
            out |= m.interval
        if a.last not in ("interval", "week"):
            specials("[END]")
    elif a.phase == "end":
        specials("[EOS]")
        if a.hier >= 0:
            out |= m.chunk_level == a.hier + 1
        stage = m.end_stage
        ok = stage == -1                                            # OOV end tokens: anywhere in the end block
        later = (stage >= a.end_stage) & (stage >= 0)
        dx = later & (stage == t.dx_stage) & (m.ids >= a.group_first)
        other = later & (stage != t.dx_stage) & ~np.isin(stage, list(a.end_seen))
        out |= ok | dx | other
    return out


def insert_clock(ids: list[int], tables: Tables) -> list[int]:
    """``ids`` with the clock blocks the automaton requires written in (spec ``clock``; the encoder's and the
    stored token shards' path to the same blocks the sampler writes). A sequence that has them, a continuation
    window, or a vocabulary without clock tokens comes back unchanged; a sequence the grammar rejects otherwise
    raises ValueError."""
    if tables.clock is None:
        return list(ids)
    a, out = Automaton(tables), []
    for pos, tok in enumerate(ids):
        tok = int(tok)
        while a.pending and tok != a.pending[0]:
            out.append(a.pending[0])
            a.step(a.pending[0])
        out.append(tok)
        if not a.step(tok):
            raise ValueError(f"{a.violation} at token {pos}")
    return out


def accepts(tables: Tables, ids: list[int], whole: bool = True) -> tuple[bool, str | None]:
    a = Automaton(tables)
    for pos, tok in enumerate(ids):
        if not a.step(tok):
            return False, f"{a.violation} at token {pos}"
    ok = a.complete() if whole else a.window_ok()
    return ok, None if ok else "incomplete sequence"
