"""Time groups and interval tokens of a timeline (independent of the vocabulary).

A group starts at its first event (its anchor); an event less than ``group_minutes`` after the anchor joins the
group, so a group spans less than a minute and no interval token is emitted inside it. Between consecutive
anchors the gap is written as interval tokens: one ``>= 7 days`` token per whole 7 days, then the bin of the rest
when it is at least the first edge (1 minute). Decoding adds the gaps back, so times round-trip to their bins.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from icutwin.tokenization.spec import interval_labels, load_spec


def _edges() -> list[int]:
    return list(load_spec()["timeline"]["interval_edges"])


@lru_cache(maxsize=1)
def _labels() -> tuple[str, ...]:
    return tuple(interval_labels(_edges()))


def labels() -> list[str]:
    return list(_labels())


def group(events: list[tuple]) -> list[tuple[datetime, list[tuple]]]:
    """``events`` sorted by time (first element) -> [(anchor time, events of the group)]."""
    width = float(load_spec()["timeline"]["group_minutes"]) * 60
    groups: list[tuple[datetime, list[tuple]]] = []
    for e in events:
        if groups and (e[0] - groups[-1][0]).total_seconds() < width:
            groups[-1][1].append(e)
        else:
            groups.append((e[0], [e]))
    return groups


def split_gap(minutes: float) -> list[tuple[int, float]]:
    """A gap between anchors -> [(interval index, minutes it stands for)]: whole 7-day tokens, then the rest."""
    edges = _edges()
    week = edges[-1]
    out = [(len(edges) - 1, float(week))] * int(minutes // week)
    rest = minutes - week * int(minutes // week)
    if rest >= edges[0]:
        k = max(i for i, e in enumerate(edges[:-1]) if rest >= e)
        out.append((k, rest))
    return out
