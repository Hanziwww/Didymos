"""Encoder: a stay -> token ids (whole stay and windows), with the canonical form the round trip compares.

Sequence (grammar spec): ``[BOS]`` prefix (fixed order), header (spec order here; training may shuffle),
optional ``[GOAL] ... [/GOAL]``, ``[TL]`` timeline (event groups, interval tokens between them), ``[END]``
hospital outcome, death and diagnoses, ``[EOS]``. An infusion running at the window start is a 'going' event of
the timeline's first group (S4 writes it from the events before the window), never inferred from a later event. Long stays are cut into windows of at most ``windows.max_tokens``: each repeats the prefix,
header (and goal), then ``[CONT]``, the state summary at its first group and the timeline from that group (with
the interval from the previous group); only the last window holds the end block. With a vocabulary that has the
clock tokens (spec ``clock``), the clock blocks are written after the cut (``fsm.insert_clock``: the whole stay and
every window but the continuation windows), so a first window can exceed ``max_tokens`` by its blocks.
"""

from __future__ import annotations

import math
import weakref
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta

from icutwin.tokenization import timeline as tl
from icutwin.tokenization.fsm import compile_tables, insert_clock
from icutwin.tokenization.spec import band_labels, band_of, load_spec
from icutwin.tokenization.stays import Stay
from icutwin.tokenization.vocab import Vocab, bin_of, hierarchical, infusion_key


@dataclass
class Encoded:
    ids: list[int]                          # the whole stay
    windows: list[list[int]]
    unmapped: Counter = field(default_factory=Counter)   # events without a token of their own, by reason
    groups: int = 0


def _field_token(v: Vocab, block: str, f: dict, value, unknown: str) -> int:
    name = f["field"]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        label = unknown
    elif "bands" in f:
        label = band_labels(f["bands"])[band_of(float(value), f["bands"])]
    else:
        label = str(value)
    tok = f"{block}//{name}//{label}"
    if tok not in v.index and "rare" in f:
        tok = f"{block}//{name}//{f['rare']}"
    return v.index.get(tok, v.index[f"{block}//{name}//{unknown}"])


def head_tokens(s: Stay, v: Vocab, goal: bool) -> list[int]:
    spec = load_spec()
    unknown = spec["missing"]
    ids = [v.id("[BOS]")]
    ids += [_field_token(v, "PREFIX", f, s.prefix.get(f["field"]), unknown) for f in spec["prefix"]]
    ids += [_field_token(v, "HEADER", f, s.header.get(f["field"]), unknown) for f in spec["header"]]
    if goal:
        ids += [v.id("[GOAL]")]
        ids += [_field_token(v, "GOAL", f, s.goal.get(f["field"]), unknown) for f in spec["goal"]]
        ids += [v.id("[/GOAL]")]
    return ids


def event_tokens(code: str, num, text, v: Vocab, unmapped: Counter, end: bool = False) -> tuple[int, ...]:
    """The tokens of one event (one, or a hierarchical code's system token and sub-tokens)."""
    info = v.codes.get(code)
    if info is not None and info["kind"] != "oov":
        if info["kind"] == "flag":
            return (info["first"],)
        if num is not None and not math.isnan(num):
            return (info["first"] + bin_of(float(num), info["edges"]),)
        return (info["text"],) if text is not None else (info["na"],)
    h = hierarchical(code)
    if h is not None and h[0] in v.systems:
        out = [v.systems[h[0]]]
        for sub in h[1]:
            if sub not in v.chunks:
                unmapped["diagnosis or procedure detail below the subject threshold"] += 1
                break
            out.append(v.chunks[sub])
        return tuple(out)
    unmapped["code below the subject threshold (OOV token)"] += 1
    if end:
        return (v.codes["END//OOV"]["first"],)
    oov = v.codes.get(code.split("//", 1)[0] + "//OOV")
    return (oov["first"],) if oov else (v.codes["END//OOV"]["first"],)


def _state_after(groups: list, upto: int, v: Vocab) -> list[int]:
    """The state summary at the start of group ``upto``: running infusions with their last rate bin, the last
    respiratory device and mode, CRRT and MCS charted within ``active_hours``."""
    st = load_spec()["windows"]["state"]
    inf = st["infusions"]
    sfx = load_spec()["timeline"]["value_suffix"]
    running: dict[str, float | None] = {}
    resp: dict[str, str] = {}
    t0 = groups[upto][0] if upto < len(groups) else groups[-1][0]   # the end-only window: after the last group
    recent = timedelta(hours=st["active_hours"])
    active: set[str] = set()
    for anchor, events in groups[:upto]:
        for _t, _rank, code, num, _text in events:
            key = infusion_key(code)
            if key is not None and key[0] in v.state_keys:
                if key[1] in inf["ends"]:
                    running.pop(key[0], None)
                elif key[1] in inf["begins"] or key[1] in inf["continues"]:
                    running[key[0]] = num if num is not None and not math.isnan(num) else running.get(key[0])
            for p in st["respiratory"]:
                if code.startswith(p) and f"STATE//{code}" in v.index:
                    resp[p] = code
            if t0 - anchor <= recent and code.split("//", 1)[0] in st["active_domains"]:
                active.add(code.split("//", 1)[0])
    ids = []
    for key in sorted(running):
        info, rate = v.state_keys[key], running[key]
        tok = None if rate is None else f"STATE//{key}" + sfx["bin"].format(k=bin_of(rate, info["edges"]))
        ids.append(v.index.get(tok, info["first"]) if tok else info["first"])   # no rate (bin): '|na'
    ids += [v.id(f"STATE//{resp[p]}") for p in st["respiratory"] if p in resp]
    ids += [v.id(f"STATE//{dom}//active") for dom in st["active_domains"] if dom in active]
    return ids


def available_by(f: dict, dataset: str) -> float:
    """Hours after ICU admission by which a header field is documented (spec ``available_within_hours``)."""
    return float((f.get("available_within_hours") or {}).get(dataset, 0))


def prospective_stay(s: Stay, cutoff_hours: float) -> Stay:
    """The stay as known ``cutoff_hours`` after ICU admission, the input of a prospective task: timeline events at
    or before the cutoff, header fields documented later unknown, no goal and no end block (outcomes)."""
    spec = load_spec()
    cutoff = s.icu_in + timedelta(hours=cutoff_hours)
    header = {k: v for k, v in s.header.items()}
    for f in spec["header"]:
        if available_by(f, s.dataset) > cutoff_hours:
            header[f["field"]] = None
    return Stay(s.subject_id, s.stay_id, s.dataset, s.split, s.icu_in, s.icu_out, s.clock_valid, dict(s.prefix),
                header, {}, [e for e in s.timeline if e[0] <= cutoff], [], Counter())


def prospective(s: Stay, v: Vocab, cutoff_hours: float, max_tokens: int | None = None) -> list[int]:
    """Model input for a prediction at ``cutoff_hours`` after ICU admission: the last window of the prospective
    stay (prefix, header, [CONT] and state when it continues, the timeline up to the cutoff), without [END]."""
    enc = encode(prospective_stay(s, cutoff_hours), v, goal=False, max_tokens=max_tokens)
    last = enc.windows[-1]
    assert last[-2:] == [v.id("[END]"), v.id("[EOS]")]
    return last[:-2]


def encode(s: Stay, v: Vocab, goal: bool = False, max_tokens: int | None = None) -> Encoded:
    spec = load_spec()
    max_tokens = max_tokens or spec["windows"]["max_tokens"]
    unmapped = Counter(s.dropped)
    head = head_tokens(s, v, goal)
    groups = tl.group(s.timeline)
    group_tokens = []          # per group: (interval tokens before it, sorted event token tuples)
    prev = None
    for anchor, events in groups:
        evs = sorted(event_tokens(code, num, text, v, unmapped) for _t, _r, code, num, text in events)
        ints = [] if prev is None else [v.id(f"INT//{tl.labels()[k]}")
                                        for k, _m in tl.split_gap((anchor - prev).total_seconds() / 60)]
        group_tokens.append((ints, evs))
        prev = anchor
    end = [v.id("[END]")] + [i for code in s.end for i in event_tokens(code, None, None, v, unmapped, end=True)] \
        + [v.id("[EOS]")]
    body = [i for ints, evs in group_tokens for i in [*ints, *(t for e in evs for t in e)]]
    whole = head + [v.id("[TL]")] + body + end
    windows = _windows(head, group_tokens, groups, end, v, max_tokens)
    t = _tables(v)
    if t.clock is not None:   # the clock blocks (spec clock): after the cut, as the stored shards get them (task=clock)
        whole, windows = insert_clock(whole, t), [insert_clock(w, t) for w in windows]
    return Encoded(whole, windows, unmapped, len(groups))


_TABLES: dict = {}


def _tables(v: Vocab):
    """The grammar tables of ``v`` (compiled once per vocabulary object)."""
    hit = _TABLES.get(id(v))
    if hit is None or hit[0]() is not v:
        hit = (weakref.ref(v), compile_tables(v))
        _TABLES[id(v)] = hit
    return hit[1]


def _windows(head, group_tokens, groups, end, v: Vocab, max_tokens: int) -> list[list[int]]:
    """Cut at group boundaries; a continuation window starts with the interval from the previous group. A group
    larger than a window stays whole (reported as oversized)."""
    windows, cur, has_groups = [], [*head, v.id("[TL]")], False
    for g, (ints, evs) in enumerate(group_tokens):
        toks = [*ints, *(t for e in evs for t in e)]
        if has_groups and len(cur) + len(toks) > max_tokens:
            windows.append(cur)
            st = _state_after(groups, g, v)
            cur = [*head, v.id("[CONT]"), *([v.id("[STATE]"), *st, v.id("[/STATE]")] if st else []), v.id("[TL]")]
            has_groups = False
        cur += toks
        has_groups = True
    if has_groups and len(cur) + len(end) > max_tokens:   # the end block opens a window of its own, with the state
        windows.append(cur)
        st = _state_after(groups, len(groups), v)
        cur = [*head, v.id("[CONT]"), *([v.id("[STATE]"), *st, v.id("[/STATE]")] if st else []), v.id("[TL]")]
    windows.append(cur + end)
    return windows
