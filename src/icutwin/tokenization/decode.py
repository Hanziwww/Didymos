"""Decoder: token ids -> a stay (D5 dequantization), and the canonical event view the round trip compares (A6).

Times: groups are placed at the sum of their interval tokens' gaps (a 7-day token is 7 days) from ``T0``; the stay's
ICU window runs from its ICU admission marker to its first ICU discharge event after it (a stay cut before one: to its
last event; without a marker both stay at ``T0``). Values: a bin
decodes from the empirical distribution of its train values for the stay's dataset (pooled over datasets
where the dataset has none): the sketch's median by default (deterministic), or a draw from it when a random
generator is given (inverse of the sketch's piecewise-linear distribution function; plan S5), optionally with the
plan's within-bin rank persistence: a code's value in the same bin as its previous one draws its rank as an AR(1)
normal score, with the code's correlation of consecutive same-bin ranks in real stays (``load_persistence``;
independent draws inflate change-based labels such as KDIGO's 0.3 mg/dL creatinine rise). Intervals
decode the same way from the gaps of their dataset. Every sketch lies inside its bin, so encoding a decoded
stay, sampled or not, gives its tokens back. Prefix, header and goal bands decode to their midpoint (an
integer band of width one, e.g. a KDIGO stage or the admission hour, to its value; an open last band to its
lower edge).
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist

from icutwin.tokenization.parser import Parsed, parse
from icutwin.tokenization.spec import load_spec
from icutwin.tokenization.stays import Stay, domain_rank
from icutwin.tokenization.vocab import Vocab

T0 = datetime(2000, 1, 1)
_ND = NormalDist()


def load_persistence(path: Path, v: Vocab) -> dict:
    """code -> rho of the within-bin rank persistence (``tools/rank_persistence.py``'s JSON, beside the vocabulary):
    a code's own estimate, else its domain's median, else 0; clipped to [0, 0.99]."""
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    own = {c: x["rho"] for c, x in r["per_code"].items() if x["rho"] == x["rho"]}
    dom = {d: x["median"] for d, x in r["by_domain"].items() if x["median"] == x["median"]}
    out = {}
    for code, info in v.codes.items():
        if info.get("edges"):
            rho = own.get(code, dom.get(code.split("//", 1)[0], 0.0))
            out[code] = min(max(float(rho), 0.0), 0.99)
    return out


def _band_value(label: str):
    m = re.fullmatch(r"(-?[\d.]+)-(-?[\d.]+)", label)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return a if b - a == 1 and a.is_integer() else (a + b) / 2
    m = re.fullmatch(r"(-?[\d.]+)\+", label)
    return float(m.group(1)) if m else None


def _field_value(f: dict, label: str | None, unknown: str):
    if label is None or label == unknown:
        return None
    return _band_value(label) if "bands" in f else label


def read_sketch(sketch: list[float], u: float | None) -> float:
    """A value of a within-bin sketch (quantiles at equal steps): its median when ``u`` is None, else the
    inverse of the piecewise-linear distribution function through the sketch at ``u`` in [0, 1)."""
    if u is None:
        return sketch[len(sketch) // 2]
    pos = u * (len(sketch) - 1)
    i = min(int(pos), len(sketch) - 2)
    return sketch[i] + (pos - i) * (sketch[i + 1] - sketch[i])


def event_of(ev: tuple[int, ...], v: Vocab, dataset: str, rng=None, u: float | None = None
             ) -> tuple[str, float | None, str | None]:
    """(code, numeric value, text) of an event's tokens; ``rng`` (``random.Random``) samples the value, at the rank
    ``u`` within its bin when given."""
    m = v.meta[ev[0]]
    kind = m.get("kind")
    if kind == "system":
        return m["code"] + "//" + "".join(v.meta[t]["code"].split("//", 2)[2] for t in ev[1:]), None, None
    code = m["code"]
    if kind == "bin":
        own = (v.value_sketches_ds.get(code) or {}).get(dataset)
        sketch = own[m["bin"]] if own and own[m["bin"]] is not None else v.value_sketches[code][m["bin"]]
        if sketch is None:   # a bin without train values (quantile edges can leave one empty; infusion closure codes
            edges = v.codes[code]["edges"]   # have none): its lower edge; below the first edge, a value inside it
            if not edges:
                return code, 0.0, None
            if m["bin"] > 0:
                return code, float(edges[m["bin"] - 1]), None
            return code, float(edges[0] / 2 if edges[0] > 0 else edges[0] - 1.0), None
        return code, float(read_sketch(sketch, u if u is not None else rng.random() if rng else None)), None
    if kind == "text":
        return code, None, "text"
    return code, None, None


def interval_minutes(tok: int, v: Vocab, dataset: str, rng=None) -> float:
    k = v.meta[tok]["interval"]
    own = v.interval_sketches.get(dataset)
    sketch = own[k] if own and own[k] is not None else v.interval_sketches["*"][k]
    if sketch is None:   # an interval no train gap fell into: the bin's lower edge
        return float(load_spec()["timeline"]["interval_edges"][k])
    return float(read_sketch(sketch, rng.random() if rng else None))


def _persistent_rank(ev: tuple[int, ...], v: Vocab, rng, rho: dict, last: dict) -> float | None:
    """A bin value's rank: an AR(1) normal score continuing its code's previous value in the same bin (``rho``),
    else a fresh one; None for other events."""
    m = v.meta[ev[0]]
    if m.get("kind") != "bin":
        return None
    code, b = m["code"], m["bin"]
    eps = _ND.inv_cdf(min(max(rng.random(), 1e-12), 1 - 1e-12))
    r, prev = rho.get(code, 0.0), last.get(code)
    z = r * prev[1] + math.sqrt(1 - r * r) * eps if prev is not None and prev[0] == b else eps
    last[code] = (b, z)
    return _ND.cdf(z)


SAME = object()          # value draws share the gaps' generator


def to_stay(p: Parsed, v: Vocab, rng=None, persistence: dict | None = None, value_rng=SAME) -> Stay:
    """``rng`` draws the gaps and, unless ``value_rng`` is given (a generator of its own, or None for the bins'
    medians), the values: two decodes with one ``rng`` seed share their clock."""
    if value_rng is SAME:
        value_rng = rng
    spec = load_spec()
    unknown = spec["missing"]
    dataset = p.prefix.get("dataset") or ""
    s = Stay(0, "decoded", dataset, "", T0, T0, p.header.get("admission_hour") not in (None, unknown))
    s.prefix = {f["field"]: _field_value(f, p.prefix.get(f["field"]), unknown) for f in spec["prefix"]}
    s.header = {f["field"]: _field_value(f, p.header.get(f["field"]), unknown) for f in spec["header"]}
    if p.goal is not None:
        s.goal = {f["field"]: _field_value(f, p.goal.get(f["field"]), unknown) for f in spec["goal"] if f["field"] in p.goal}
    t = T0
    last: dict = {}                  # code -> (bin, normal score) of its last drawn value
    for ints, events in p.groups:
        t += timedelta(minutes=sum(interval_minutes(i, v, dataset, rng) for i in ints))
        for ev in events:
            u = _persistent_rank(ev, v, value_rng, persistence, last) if value_rng is not None and persistence else None
            code, num, text = event_of(ev, v, dataset, value_rng, u)
            s.timeline.append((t, domain_rank(code), code, num, text))
    if p.end is not None:
        s.end = [event_of(ev, v, dataset)[0] for ev in p.end]
    marker = next((t for t, _r, c, *_ in s.timeline if c == spec["timeline"]["marker"]["token"]), None)
    if marker is not None:   # the ICU window: the marker to the first ICU discharge after it, else the last event
        s.icu_in = marker
        s.icu_out = next((t for t, _r, c, *_ in s.timeline if c.startswith("ICU//discharge//") and t >= marker),
                         s.timeline[-1][0])
    return s


def decode(ids: list[int], v: Vocab, rng=None, persistence: dict | None = None, value_rng=SAME) -> Stay:
    """The stay of a whole token sequence; values and gaps at their bin's median, or sampled with ``rng``, with
    ``persistence`` (``load_persistence``) the values' ranks within their bins continuing as real stays' do;
    ``value_rng``: the values' own generator, or None for medians (``to_stay``)."""
    return to_stay(parse(ids, v, whole=True), v, rng, persistence, value_rng)


def canonical_events(ids: list[int], v: Vocab) -> list:
    """The canonical view of an encoded stay: per group its interval labels and its events as (code, value bin or
    'na'/'text'/None), in order; then the end block's codes."""
    p = parse(ids, v, whole=True)
    groups = []
    for ints, events in p.groups:
        evs = []
        for ev in events:
            m = v.meta[ev[0]]
            code = event_of(ev, v, p.prefix.get("dataset") or "")[0] if m.get("kind") == "system" else m["code"]
            evs.append((code, m.get("bin") if m.get("kind") == "bin" else m.get("kind")))
        groups.append(([v.tokens[i] for i in ints], evs))
    end = [event_of(ev, v, "")[0] for ev in (p.end or [])]
    return [p.prefix, p.header, p.goal, groups, end]
