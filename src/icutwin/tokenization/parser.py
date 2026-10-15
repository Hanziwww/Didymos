"""Independent recursive-descent parser of token sequences (D4, A5).

Checks the grammar with code separate from the automaton (:mod:`icutwin.tokenization.fsm`), reading token
classes straight from the vocabulary metadata, and returns the sequence's structure, which the decoder uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from icutwin.tokenization.spec import load_spec
from icutwin.tokenization.vocab import Vocab


class ParseError(ValueError):
    pass


@dataclass
class Parsed:
    prefix: dict = field(default_factory=dict)
    header: dict = field(default_factory=dict)
    goal: dict | None = None
    cont: bool = False
    state: list = field(default_factory=list)
    groups: list = field(default_factory=list)      # [(interval ids before the group, [event token tuples])]
    end: list | None = None                         # [event token tuples]; None: a window without the end block
    clock: list = field(default_factory=list)       # clock blocks [(day, {treatment: state})]: conditioning only


class _Reader:
    def __init__(self, ids: list[int], v: Vocab):
        self.ids, self.pos, self.v = ids, 0, v

    def peek(self) -> int | None:
        return self.ids[self.pos] if self.pos < len(self.ids) else None

    def meta(self, tok: int | None) -> dict:
        return self.v.meta[tok] if tok is not None else {}

    def take(self) -> int:
        tok = self.peek()
        if tok is None:
            raise ParseError("unexpected end of sequence")
        self.pos += 1
        return tok

    def expect(self, name: str) -> None:
        if self.take() != self.v.id(name):
            raise ParseError(f"{name} expected at {self.pos - 1}")

    def is_(self, name: str) -> bool:
        return self.peek() == self.v.id(name)


def parse(ids: list[int], v: Vocab, whole: bool = True) -> Parsed:
    spec = load_spec()
    r, p = _Reader(ids, v), Parsed()
    r.expect("[BOS]")
    for f in spec["prefix"]:
        tok = r.take()
        m = r.meta(tok)
        if m.get("cls") != "prefix" or m["field"] != f["field"]:
            raise ParseError(f"prefix field {f['field']} expected at {r.pos - 1}")
        p.prefix[f["field"]] = m["value"]
    while r.meta(r.peek()).get("cls") == "header":
        m = r.meta(r.take())
        if m["field"] in p.header:
            raise ParseError(f"header field {m['field']} twice")
        p.header[m["field"]] = m["value"]
    if set(p.header) != {f["field"] for f in spec["header"]}:
        raise ParseError("header fields missing")
    if r.is_("[GOAL]"):
        r.take()
        p.goal = {}
        while r.meta(r.peek()).get("cls") == "goal":
            m = r.meta(r.take())
            if m["field"] in p.goal:
                raise ParseError(f"goal field {m['field']} twice")
            p.goal[m["field"]] = m["value"]
        if not p.goal:
            raise ParseError("empty goal block")
        r.expect("[/GOAL]")
    if r.is_("[CONT]"):
        r.take()
        p.cont = True
    running: set[str] = set()
    if r.is_("[STATE]"):
        r.take()
        while r.meta(r.peek()).get("cls") == "state":
            tok = r.take()
            p.state.append(tok)
            key = r.meta(tok)["key"]
            if key.startswith(spec["windows"]["state"]["infusions"]["domain"] + "//"):
                running.add(key)
        r.expect("[/STATE]")
    r.expect("[TL]")
    _timeline(r, p, running)
    if r.is_("[END]"):
        r.take()
        p.end = _end(r, spec)
        r.expect("[EOS]")
    elif whole:
        raise ParseError("[END] expected")
    if r.peek() is not None:
        raise ParseError(f"tokens after the sequence at {r.pos}")
    return p


def _event(r: _Reader, end: bool = False) -> tuple[int, ...]:
    """One event: a token, or a system token and sub-tokens of levels 1, 2, ... in order."""
    tok = r.take()
    m = r.meta(tok)
    out = [tok]
    if m.get("kind") == "system":
        level = 0
        while r.meta(r.peek()).get("cls") == "chunk" and r.meta(r.peek())["level"] == level + 1:
            out.append(r.take())
            level += 1
        if r.meta(r.peek()).get("cls") == "chunk":
            raise ParseError(f"sub-token of the wrong level at {r.pos}")
    elif r.meta(r.peek()).get("cls") == "chunk":
        raise ParseError(f"sub-token without a system token at {r.pos}")
    return tuple(out)


def _timeline(r: _Reader, p: Parsed, running: set) -> None:
    week = max(i for i, m in enumerate(r.v.meta) if m.get("cls") == "interval")
    first = True
    while True:
        ints = []
        while True:     # intervals, each that opens a new ICU day followed by its clock block (spec clock)
            cls = r.meta(r.peek()).get("cls")
            if cls == "interval":
                if ints and ints[-1] != week:
                    raise ParseError(f"interval after a non-repeating interval at {r.pos}")
                ints.append(r.take())
            elif cls == "clock" and ints:
                p.clock.append(_clock(r))
            else:
                break
        if ints and first and not p.cont:
            raise ParseError("timeline starts with an interval")
        if not ints and not first:
            return   # [END] or the end of a window follows the group
        events = []
        while r.meta(r.peek()).get("cls") == "event":
            ev = _event(r)
            if events and ev[0] < events[-1][0]:
                raise ParseError(f"group events out of canonical order at {r.pos}")
            _infusion(r.meta(ev[0]), running, r.pos, load_spec()["windows"]["state"]["infusions"])
            events.append(ev)
        if not events:
            if ints:
                raise ParseError(f"interval not followed by an event at {r.pos}")
            return   # an empty timeline
        p.groups.append((ints, events))
        first = False


def _clock(r: _Reader) -> tuple:
    """A clock block after intervals (spec ``clock``): the marker, a day, then one state per treatment in the
    spec's order; returns (day, {treatment: state}). Where blocks belong is the automaton's to check."""
    spec = load_spec()["clock"]
    if r.meta(r.take()).get("role") != "marker":
        raise ParseError(f"clock block without its marker at {r.pos - 1}")
    m = r.meta(r.take())
    if m.get("role") != "day":
        raise ParseError(f"clock block: a day expected at {r.pos - 1}")
    states = {}
    for t in spec["treatments"]:
        s = r.meta(r.take())
        if s.get("role") != "status" or s["treatment"] != t:
            raise ParseError(f"clock block: the state of {t} expected at {r.pos - 1}")
        states[t] = s["state"]
    return m["day"], states


def _infusion(m: dict, running: set, pos: int, actions: dict) -> None:
    if "key" not in m:
        return
    if m["action"] in actions["begins"]:
        running.add(m["key"])
    elif m["action"] in actions["continues"] or m["action"] in actions["ends"]:
        if m["key"] not in running:
            raise ParseError(f"infusion {m['action']} of a key that does not run at {pos}")
        if m["action"] in actions["ends"]:
            running.discard(m["key"])


def _end(r: _Reader, spec: dict) -> list:
    codes = [e["code"].split("{")[0] for e in spec["end"] if "code" in e]
    dx_domains = {e["domain"] for e in spec["end"] if "domain" in e}
    out, stage, seen, last_dx = [], 0, set(), -1
    while r.meta(r.peek()).get("cls") == "end":
        ev = _event(r, end=True)
        m = r.meta(ev[0])
        out.append(ev)
        if m.get("kind") == "oov":
            continue
        if m.get("kind") == "system" or m["code"].split("//", 1)[0] in dx_domains:
            this = len(codes)   # a diagnosis: a hierarchical system token or a whole code of another system
        else:
            this = next(k for k, c in enumerate(codes) if m["code"].startswith(c))
        if this < stage or (this < len(codes) and this in seen):
            raise ParseError("end block out of order or repeated")
        if this == len(codes):
            if ev[0] < last_dx:
                raise ParseError("diagnoses out of order")
            last_dx = ev[0]
        stage = this
        seen.add(this)
    return out
