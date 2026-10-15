"""Validation of the tokenizer on encoded stays (A5, A6, A9, A10) and the measured, unenforced rules.

* A5 no over-constraint: the automaton and the independent parser accept every encoded sequence (whole stays
  and windows; with and without the goal block).
* A6 round trip: the canonical view of a stay (per time group its interval labels and its events as code and
  value bin; prefix, header and goal labels; the end block) equals that of its decoded stay, and encoding the
  decoded stay gives the same tokens, with values and gaps at their bins' medians and sampled from the bins'
  empirical distributions.
* A9 goal consistency: the goal tokens equal the outcomes and derived concepts recomputed from MEDS and the
  derived files.
* Window continuity: the windows' timelines, put together, are the whole sequence's (no token lost or repeated
  at a boundary), and each continuation window's state block names exactly the infusions running, and the
  respiratory device and mode last charted, before its first group (replayed on the whole sequence).
* Separation (prospective inputs, ``encode.prospective``): at 0, 1, 6 and 24 h after ICU admission the input has
  no goal or end-block token, header fields documented later are unknown and the others as in the whole stay,
  the automaton accepts it as a window, and it does not change when everything after the cutoff (events,
  outcomes, goal values, so every derived concept) is changed.
* A10 report: tokens per stay, windows per stay, oversized windows, events without a token by reason.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from icutwin.tokenization import timeline as tl
from icutwin.tokenization.decode import decode, event_of
from icutwin.tokenization.encode import _field_token, available_by, encode, event_tokens, head_tokens, prospective
from icutwin.tokenization.fsm import Automaton, Tables, accepts
from icutwin.tokenization.parser import ParseError, parse
from icutwin.tokenization.spec import load_spec
from icutwin.tokenization.stays import Stay
from icutwin.tokenization.vocab import Vocab


def canonical_view(s: Stay, v: Vocab, goal: bool = False) -> list:
    """The canonicalized MEDS of a stay: labels of its prefix, header (and goal) fields; per time group the
    interval labels before it and its events as (code after OOV and truncation, value bin | 'na' | 'text' |
    None) in canonical order; the end block's codes."""
    spec = load_spec()
    unknown = spec["missing"]
    label = lambda block, f, val: v.tokens[_field_token(v, block, f, val, unknown)]  # noqa: E731
    head = [[label("PREFIX", f, s.prefix.get(f["field"])) for f in spec["prefix"]],
            [label("HEADER", f, s.header.get(f["field"])) for f in spec["header"]],
            [label("GOAL", f, s.goal.get(f["field"])) for f in spec["goal"]] if goal else None]
    groups, prev, dummy = [], None, Counter()
    for anchor, events in tl.group(s.timeline):
        ints = [] if prev is None else [tl.labels()[k] for k, _ in tl.split_gap((anchor - prev).total_seconds() / 60)]
        evs = sorted((event_tokens(code, num, text, v, dummy), code) for _t, _r, code, num, text in events)
        groups.append((ints, [_view(toks, v) for toks, _code in evs]))
        prev = anchor
    end = [_view(event_tokens(code, None, None, v, dummy, end=True), v) for code in s.end]
    return [*head, groups, end]


def _view(toks: tuple[int, ...], v: Vocab) -> tuple:
    m = v.meta[toks[0]]
    code = event_of(toks, v, "")[0]
    return code, m.get("bin") if m.get("kind") == "bin" else m.get("kind")


@dataclass
class Report:
    stays: int = 0
    tokens: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    oversized_windows: int = 0
    fsm_rejected: Counter = field(default_factory=Counter)
    parser_rejected: Counter = field(default_factory=Counter)
    round_trip_failed: int = 0
    reencode_failed: int = 0
    sampled_reencode_failed: int = 0
    continuity_failed: int = 0
    separation_failed: int = 0
    goal_mismatch: Counter = field(default_factory=Counter)
    unmapped: Counter = field(default_factory=Counter)
    measured: Counter = field(default_factory=Counter)
    examples: dict = field(default_factory=lambda: defaultdict(list))

    def merge(self, other: Report) -> Report:
        """Add another part's counts (parallel validation); lists keep the parts' order."""
        for name in ("stays", "oversized_windows", "round_trip_failed", "reencode_failed", "sampled_reencode_failed",
                     "continuity_failed", "separation_failed"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.tokens += other.tokens
        self.windows += other.windows
        for name in ("fsm_rejected", "parser_rejected", "goal_mismatch", "unmapped", "measured"):
            getattr(self, name).update(getattr(other, name))
        for k, v in other.examples.items():
            self.examples[k].extend(v)
        return self

    def summary(self) -> dict:
        t, w = np.asarray(self.tokens or [0]), np.asarray(self.windows or [0])
        return {"stays": self.stays,
                "tokens_per_stay": {"median": float(np.median(t)), "p90": float(np.quantile(t, 0.9)),
                                    "p99": float(np.quantile(t, 0.99)), "max": int(t.max()), "total": int(t.sum())},
                "windows_per_stay": {"1": int((w == 1).sum()), ">1": int((w > 1).sum()), "max": int(w.max())},
                "oversized_windows": self.oversized_windows,
                "fsm_rejected": dict(self.fsm_rejected), "parser_rejected": dict(self.parser_rejected),
                "round_trip_failed": self.round_trip_failed, "reencode_failed": self.reencode_failed,
                "sampled_reencode_failed": self.sampled_reencode_failed,
                "continuity_failed": self.continuity_failed, "separation_failed": self.separation_failed,
                "goal_mismatch": dict(self.goal_mismatch), "events_without_own_token": dict(self.unmapped),
                "measured": dict(self.measured), "examples": {k: v[:3] for k, v in self.examples.items()}}


def _check(ids: list[int], tables: Tables, v: Vocab, whole: bool, where: str, rep: Report) -> None:
    ok, why = accepts(tables, ids, whole=whole)
    if not ok:
        rep.fsm_rejected[where] += 1
        rep.examples["fsm:" + where].append(why)
    try:
        parse(ids, v, whole=whole)
    except ParseError as e:
        rep.parser_rejected[where] += 1
        rep.examples["parser:" + where].append(str(e))


def validate_stay(s: Stay, v: Vocab, tables: Tables, rep: Report, derived: dict | None = None) -> None:
    max_tokens = load_spec()["windows"]["max_tokens"]
    rep.stays += 1
    for goal in (False, True):
        enc = encode(s, v, goal=goal)
        where = "goal" if goal else "plain"
        _check(enc.ids, tables, v, True, f"whole/{where}", rep)
        for k, w in enumerate(enc.windows):
            _check(w, tables, v, k == len(enc.windows) - 1, f"window/{where}", rep)
            rep.oversized_windows += int(len(w) > max_tokens)
        if not goal:
            rep.tokens.append(len(enc.ids))
            rep.windows.append(len(enc.windows))
            rep.unmapped.update(enc.unmapped)
            why = _continuity(enc.ids, enc.windows, v, tables)
            if why:
                rep.continuity_failed += 1
                rep.examples["continuity"].append(f"{s.stay_id}: {why}")
        # A6: decode, compare canonical views, re-encode (values and gaps at the bins' medians, and sampled)
        try:
            dec = decode(enc.ids, v)
            if canonical_view(dec, v, goal) != canonical_view(s, v, goal):
                rep.round_trip_failed += 1
                rep.examples["round_trip"].append(s.stay_id)
            if encode(dec, v, goal=goal).ids != enc.ids:
                rep.reencode_failed += 1
                rep.examples["reencode"].append(s.stay_id)
            if not goal and encode(decode(enc.ids, v, random.Random(s.subject_id)), v).ids != enc.ids:
                rep.sampled_reencode_failed += 1
                rep.examples["sampled_reencode"].append(s.stay_id)
        except ParseError as e:
            rep.round_trip_failed += 1
            rep.examples["round_trip"].append(f"{s.stay_id}: {e}")
        if goal and derived is not None:
            try:
                _goal_consistency(s, enc.ids, v, rep, derived)
            except ParseError:   # a rejected sequence: counted above, its goals are not read
                rep.examples["goal: not read"].append(s.stay_id)
    for cutoff in SEPARATION_CUTOFFS_H:
        why = _separation(s, v, tables, cutoff)
        if why:
            rep.separation_failed += 1
            rep.examples["separation"].append(f"{s.stay_id} at {cutoff} h: {why}")
    _measure(s, rep)


SEPARATION_CUTOFFS_H = (0, 1, 6, 24)   # at admission; INSPIRE's 1-hour documentation window; 6 h; a day


def _timeline_part(ids: list[int], v: Vocab) -> list[int]:
    return ids[ids.index(v.id("[TL]")) + 1:]


def _continuity(whole: list[int], windows: list[list[int]], v: Vocab, tables: Tables) -> str | None:
    """None when the windows tile the whole sequence's timeline and every continuation window's state block names
    the state replayed on the whole sequence up to its first group (infusion keys, respiratory device and mode).
    Clock blocks (written in the whole stay and the first window only) are left out of the tiling; the replay runs
    over the whole sequence as written."""
    clock = {i for i, m in enumerate(v.meta) if m.get("cls") == "clock"}
    bare = [t for t in whole if t not in clock]
    parts = [_timeline_part([t for t in w if t not in clock], v) for w in windows]
    if [t for p in parts for t in p] != _timeline_part(bare, v):
        return "windows do not tile the timeline"
    start = bare.index(v.id("[TL]")) + 1
    resp = load_spec()["windows"]["state"]["respiratory"]
    for k, (w, part) in enumerate(zip(windows, parts)):
        if k:
            a, n = Automaton(tables), 0
            for tok in whole:                  # up to the window's first group, counted without the clock tokens
                if n == start:
                    break
                a.step(tok)
                n += tok not in clock
            last_resp = {}
            for tok in _timeline_part(bare[:start], v):
                code = v.meta[tok].get("code") or ""
                for p in resp:
                    if code.startswith(p) and f"STATE//{code}" in v.index:
                        last_resp[p] = f"STATE//{code}"
            state = w[w.index(v.id("[STATE]")) + 1:w.index(v.id("[/STATE]"))] if v.id("[STATE]") in w else []
            keys = {tables.state_key[t] for t in state if t in tables.state_key}
            if keys != set(a.running):
                return f"window {k}: state infusions {sorted(keys)} != running {sorted(a.running)}"
            named = {v.tokens[t] for t in state if v.tokens[t].startswith("STATE//RESP//")}
            if named != set(last_resp.values()):
                return f"window {k}: state respiratory {sorted(named)} != last charted {sorted(last_resp.values())}"
        start += len(part)
    return None


def _separation(s: Stay, v: Vocab, tables: Tables, cutoff: float) -> str | None:
    spec = load_spec()
    ids = prospective(s, v, cutoff)
    if any(v.meta[t].get("cls") in ("goal", "end") or v.tokens[t] in ("[GOAL]", "[END]", "[EOS]") for t in ids):
        return "a goal or end-block token in the input"
    header = lambda toks: {v.meta[t]["field"]: v.meta[t]["value"] for t in toks  # noqa: E731
                           if v.meta[t].get("cls") == "header"}
    whole, got = header(head_tokens(s, v, False)), header(ids)
    for f in spec["header"]:
        late = available_by(f, s.dataset) > cutoff
        if got.get(f["field"]) != (spec["missing"] if late else whole[f["field"]]):
            return (f"header field {f['field']} " + ("documented after" if late else "documented by")
                    + f" {cutoff} h: {got.get(f['field'])} in the input, {whole[f['field']]} in the stay")
    ok, why = accepts(tables, ids, whole=False)
    if not ok:
        return f"automaton: {why}"
    cut = s.icu_in + timedelta(hours=cutoff)
    other = Stay(s.subject_id, s.stay_id, s.dataset, s.split, s.icu_in, s.icu_out, s.clock_valid, dict(s.prefix),
                 dict(s.header), {k: None for k in s.goal},
                 [e for e in s.timeline if e[0] <= cut]
                 + [(e[0], e[1], e[2], None if e[3] is None else e[3] * 3 + 1, e[4]) for e in s.timeline if e[0] > cut][::-1],
                 list(reversed(s.end)) + ["MEDS_DEATH"])
    other.timeline.sort(key=lambda e: e[0])
    if prospective(other, v, cutoff) != ids:
        return "the input changes when data after the cutoff change"
    return None


def _goal_consistency(s: Stay, ids: list[int], v: Vocab, rep: Report, derived: dict) -> None:
    """A9: every goal token against the value recomputed here from the stay's MEDS events and times and from the
    derived concept files (a separate code path from the encoder's)."""
    p = parse(ids, v)
    spec = load_spec()
    unknown = spec["missing"]
    within = lambda a, b: a < s.icu_out and b > s.icu_in  # noqa: E731
    episodes = (derived.get("episodes") or {}).get(s.stay_id, [])
    kdigo = [x for t, x in (derived.get("kdigo") or {}).get(s.stay_id, [])
             if x is not None and (t is None or s.icu_in <= t <= s.icu_out)]
    expect = {
        "icu_outcome": next((c.rsplit("//", 1)[1] for _t, _r, c, *_ in s.timeline if c.startswith("ICU//discharge//")),
                            None),
        "icu_los": (s.icu_out - s.icu_in).total_seconds() / 3600,
        "hosp_outcome": next((c.rsplit("//", 1)[1] for c in s.end if c.startswith("HOSP//discharge_category//")), None),
        "mcs": "yes" if any(c.startswith("MCS//") and s.icu_in <= t <= s.icu_out for t, _r, c, *_ in s.timeline)
        else "no",
        "kdigo_max": max(kdigo) if kdigo else None,
        "sofa_1d": (derived.get("sofa_24h") or {}).get(s.stay_id),
    }
    for ep in ("imv", "vasopressor", "rrt"):
        expect[ep] = "yes" if any(e == ep and within(a, b) for e, a, b in episodes) else "no"
    for f in spec["goal"]:
        got = p.goal.get(f["field"])
        want = v.tokens[_field_token(v, "GOAL", f, expect.get(f["field"]), unknown)].rsplit("//", 1)[1]
        if got != want:
            rep.goal_mismatch[f["field"]] += 1
            rep.examples["goal:" + f["field"]].append(f"{s.stay_id}: {got} != {want}")


def _measure(s: Stay, rep: Report) -> None:
    """Rules real data are known to break, measured rather than enforced: respiratory settings charted before
    any respiratory device of the stay."""
    device = False
    for _t, _r, code, *_ in s.timeline:
        if code.startswith("RESP//device_category//"):
            device = True
        elif code.startswith("RESP//") and code.endswith("_set") and not device:
            rep.measured["respiratory_settings_without_device"] += 1
