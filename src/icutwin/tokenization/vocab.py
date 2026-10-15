"""The token vocabulary, fitted on the train split of every dataset (D3, D5).

Compiled from the grammar spec and the train stays only:
* special, prefix, header and goal tokens (``PREFIX|HEADER|GOAL//<field>//<value>``; bands labelled by the spec);
* interval tokens (``INT//<label>``) and state tokens (``STATE//...``);
* timeline event tokens: per code seen in at least ``min_subjects`` train subjects, a value-less token
  ``<code>``, or for numeric codes ``<code>|q<k>`` per value bin plus ``<code>|na`` and ``<code>|text``;
  hierarchical codes as a system token ``<DOMAIN>//<SYSTEM>`` and sub-tokens ``<SYSTEM>//<level>//<chars>``;
  rarer codes as ``<DOMAIN>//OOV``.
Timeline event tokens are numbered in the canonical event order (domain rank, code, bin), so events of a time
group are in canonical order exactly when their first token ids do not decrease (what the FSM checks).

Dequantization statistics come from the same train stays: per (code, dataset, bin) and per (interval, dataset)
a sketch of the empirical distribution inside the bin (``QUANTILES``: minimum, quartiles, maximum), which the
decoder reads at its median or samples (plan S5: values and times dequantized from each bin's empirical
distribution, per dataset). Every token carries metadata: class, field, code, and whether its CLIF category is a
standard CLIF 3.0 term or one of our extensions (consortium practice, local, ``ext_`` concept).
"""

from __future__ import annotations

import bisect
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from icutwin.tokenization import timeline as tl
from icutwin.tokenization.spec import band_labels, load_spec, spec_hash
from icutwin.tokenization.stays import domain_rank

VERSION = 2
QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)   # the within-bin sketch: minimum, quartiles, maximum


@dataclass
class Vocab:
    tokens: list[str]
    meta: list[dict]
    codes: dict[str, dict]               # timeline/end code -> {"kind", "edges", "first", ...}
    chunks: dict[str, int]               # hierarchical sub-token -> id
    systems: dict[str, int]              # hierarchical system token -> id
    state_keys: dict[str, dict]          # infusion key -> {"edges", "first"}
    value_sketches: dict[str, list]      # code -> per bin the pooled sketch (QUANTILES; None: no train value)
    value_sketches_ds: dict[str, dict]   # code -> dataset -> per bin sketch
    interval_sketches: dict[str, list]   # dataset or "*" -> per interval token the sketch of its minutes
    spec_hash: str
    fitted_on: dict = field(default_factory=dict)

    def __post_init__(self):
        self.index = {t: i for i, t in enumerate(self.tokens)}

    def id(self, token: str) -> int:
        return self.index[token]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {k: getattr(self, k) for k in ("tokens", "meta", "codes", "chunks", "systems", "state_keys",
                                              "value_sketches", "value_sketches_ds", "interval_sketches",
                                              "spec_hash", "fitted_on")}
        path.write_text(json.dumps({"version": VERSION, **body}, sort_keys=True, separators=(",", ":")),
                        encoding="utf-8")

    @staticmethod
    def load(path: Path) -> Vocab:
        body = json.loads(path.read_text(encoding="utf-8"))
        if body.pop("version") != VERSION:
            raise ValueError(f"{path}: vocabulary format of another version; run task=fit")
        return Vocab(**body)


# -- category kinds (A3) ------------------------------------------------------------------------------------
def _category_kind(code: str) -> str:
    """'standard', or the registry/extension kind of a code's CLIF category ('' for codes without one)."""
    from icutwin.clif.conform import extensions
    from icutwin.meds.build import EXTENSION_COLUMNS

    parts = code.split("//")
    if parts[0] not in EXTENSION_COLUMNS or len(parts) < 2:
        return ""
    table, column = EXTENSION_COLUMNS[parts[0]]
    spec = ((extensions().get(table) or {}).get(column) or {}).get(parts[1])
    if spec is None:
        return "standard"
    return (spec or {}).get("kind") or ("concept" if parts[1].startswith("ext_") else "consortium")


# -- fitting --------------------------------------------------------------------------------------------------
class _Counts:
    def __init__(self):
        self.subjects = defaultdict(set)           # code -> subject ids
        self.values = defaultdict(lambda: defaultdict(list))   # code -> dataset -> values
        self.numeric = set()
        self.prefix = defaultdict(lambda: defaultdict(set))    # field -> value -> subjects
        self.header = defaultdict(set)
        self.goal = defaultdict(set)
        self.gaps = defaultdict(lambda: defaultdict(list))     # dataset -> interval index -> minutes
        self.state = defaultdict(lambda: defaultdict(list))    # infusion key -> dataset -> rates
        self.state_subjects = defaultdict(set)
        self.resp = defaultdict(set)

    def plain(self) -> dict:
        """A picklable copy for parallel fitting, values as float64 arrays."""
        arrays = lambda d2: {a: {b: np.asarray(v, dtype=np.float64) for b, v in d.items()} for a, d in d2.items()}  # noqa: E731
        return {"subjects": dict(self.subjects), "values": arrays(self.values), "numeric": set(self.numeric),
                "prefix": {f: dict(d) for f, d in self.prefix.items()}, "header": dict(self.header),
                "goal": dict(self.goal), "gaps": arrays(self.gaps), "state": arrays(self.state),
                "state_subjects": dict(self.state_subjects), "resp": dict(self.resp)}

    def merge(self, p: dict) -> None:
        """Add a part's statistics (:meth:`plain`): set unions; value arrays kept per part (:func:`_flat`)."""
        for name in ("subjects", "header", "goal", "state_subjects", "resp"):
            mine = getattr(self, name)
            for k, s in p[name].items():
                mine[k] |= s
        for f, d in p["prefix"].items():
            for v, s in d.items():
                self.prefix[f][v] |= s
        self.numeric |= p["numeric"]
        for name in ("values", "gaps", "state"):
            mine = getattr(self, name)
            for a, d in p[name].items():
                for b, arr in d.items():
                    mine[a][b].append(arr)


def _flat(values) -> np.ndarray:
    """Values collected as floats (one process) or as arrays (merged parts), as one float64 array."""
    if len(values) and isinstance(values[0], np.ndarray):
        return np.concatenate(values)
    return np.asarray(values, dtype=np.float64)


def _pool(per_dataset: dict) -> np.ndarray:
    return np.concatenate([_flat(v) for v in per_dataset.values()]) if per_dataset else np.empty(0)


def hierarchical(code: str) -> tuple[str, list[str]] | None:
    """('<DOMAIN>//<SYSTEM>', [sub-tokens]) for a code of a character-hierarchical system, else None."""
    h = load_spec()["timeline"]["hierarchical"]
    parts = code.split("//")
    if len(parts) != 3 or parts[0] not in h["domains"] or parts[1] not in h["systems"]:
        return None
    rest, subs = parts[2], []
    for level, n in enumerate(h["chunks"], start=1):
        if not rest:
            break
        piece, rest = (rest, "") if n is None else (rest[:n], rest[n:])
        subs.append(f"{parts[1]}//{level}//{piece}")
    return f"{parts[0]}//{parts[1]}", subs


def infusion_key(code: str) -> tuple[str, str] | None:
    """(key, action) of a continuous-medication code (``MED_CON//drug//unit//action//route``)."""
    st = load_spec()["windows"]["state"]["infusions"]
    parts = code.split("//")
    if parts[0] != st["domain"] or len(parts) < max(st["key_parts"]) + 1:
        return None
    return "//".join([parts[0], *(parts[i] for i in st["key_parts"])]), parts[3]


def infusion_code(key: str, action: str) -> str:
    """The code of an infusion key's action (inverse of :func:`infusion_key`)."""
    st = load_spec()["windows"]["state"]["infusions"]
    parts = key.split("//")
    out: list = [None] * (max(st["key_parts"]) + 1)
    out[0] = parts[0]
    for idx, val in zip(st["key_parts"], parts[1:]):
        out[idx] = val
    out[next(i for i in range(1, len(out)) if out[i] is None)] = action
    return "//".join(out)


def sort_key(code: str) -> str:
    """Canonical order of codes within a domain: the code, except that one infusion key's events follow the
    spec's action order (start before dose change before stop) instead of the alphabet."""
    key = infusion_key(code)
    if key is None:
        return code
    order = load_spec()["windows"]["state"]["infusions"]["action_order"]
    rank = order.index(key[1]) if key[1] in order else len(order)
    return f"{key[0]}//{rank:02d}//{key[1]}"


def _band_fields(fields: list[dict]) -> dict[str, list[str]]:
    return {f["field"]: band_labels(f["bands"]) for f in fields if "bands" in f}


def fit(stays, min_subjects: int | None = None) -> Vocab:
    """Fit the vocabulary on an iterable of train :class:`Stay` objects (all datasets)."""
    spec = load_spec()
    min_subjects = spec["vocabulary"]["min_subjects"] if min_subjects is None else min_subjects
    c = _Counts()
    n_stays, datasets = accumulate(c, stays)
    return _compile(c, spec, min_subjects, n_stays, sorted(datasets))


def fit_merged(parts: list[tuple[dict, int, set]], min_subjects: int | None = None) -> Vocab:
    """The vocabulary from statistics gathered in parts (:func:`accumulate` then ``_Counts.plain``, e.g. by
    parallel processes): the same as :func:`fit` on all their stays, since every statistic is a set or a
    multiset of values."""
    spec = load_spec()
    min_subjects = spec["vocabulary"]["min_subjects"] if min_subjects is None else min_subjects
    c, n_stays, datasets = _Counts(), 0, set()
    for plain, n, ds in parts:
        c.merge(plain)
        n_stays += n
        datasets |= ds
    return _compile(c, spec, min_subjects, n_stays, sorted(datasets))


def accumulate(c: _Counts, stays) -> tuple[int, set]:
    """Add the stays' statistics to ``c``; returns (stays, datasets)."""
    spec = load_spec()
    n_stays, datasets = 0, set()
    resp_prefixes = spec["windows"]["state"]["respiratory"]
    for s in stays:
        n_stays += 1
        datasets.add(s.dataset)
        for f, v in s.prefix.items():
            c.prefix[f][str(v)].add(s.subject_id)
        for f, v in s.header.items():
            c.header[f].add(v)
        for f, v in s.goal.items():
            c.goal[f].add(v)
        for t, _rank, code, num, text in s.timeline:
            c.subjects[code].add(s.subject_id)
            if num is not None and not math.isnan(num):
                c.numeric.add(code)
                c.values[code][s.dataset].append(float(num))
            key = infusion_key(code)
            if key is not None:
                c.state_subjects[key[0]].add(s.subject_id)
                if num is not None and not math.isnan(num) and key[1] not in spec["windows"]["state"]["infusions"]["ends"]:
                    c.state[key[0]][s.dataset].append(float(num))
            if any(code.startswith(p) for p in resp_prefixes):
                c.resp[code].add(s.subject_id)
        for code in s.end:
            c.subjects[code].add(s.subject_id)
        anchors = [a for a, _ in tl.group(s.timeline)]
        for a, b in zip(anchors, anchors[1:]):
            for k, minutes in tl.split_gap((b - a).total_seconds() / 60):
                c.gaps[s.dataset][k].append(minutes)
    return n_stays, datasets


def _edges(values: list[float], spec: dict) -> list[float]:
    v = np.asarray(values, dtype=np.float64)
    uniq = np.unique(v)
    if len(uniq) <= spec["timeline"]["exact_max_distinct"]:
        return [float(x) for x in (uniq[:-1] + uniq[1:]) / 2]
    k = spec["timeline"]["value_bins"]
    qs = np.quantile(v, [i / k for i in range(1, k)], method="linear")
    return [float(x) for x in np.unique(qs)]


def bin_of(value: float, edges: list[float]) -> int:
    """The bin of a value: the number of edges at or below it (numpy ``searchsorted(side='right')``)."""
    return bisect.bisect_right(edges, value)


def _sketches(values: list[float], edges: list[float]) -> list:
    """Per bin of ``edges`` the sketch (``QUANTILES``) of the values in it; None for a bin without values.
    Every sketch lies inside its bin, so a value read from it bins back to the same token."""
    v = _flat(values)
    if not len(v):
        return [None] * (len(edges) + 1)
    b = np.searchsorted(np.asarray(edges, dtype=np.float64), v, side="right")
    return [[float(x) for x in np.quantile(v[b == k], QUANTILES)] if (b == k).any() else None
            for k in range(len(edges) + 1)]


def _compile(c: _Counts, spec: dict, min_subjects: int, n_stays: int, datasets: list[str]) -> Vocab:
    suffix = spec["timeline"]["value_suffix"]
    unknown = spec["missing"]
    tokens: list[str] = []
    meta: list[dict] = []

    def add(token: str, **m) -> int:
        tokens.append(token)
        meta.append(m)
        return len(tokens) - 1

    for t in spec["special"]:
        add(t, cls="special")
    # prefix: values seen in the train split (sites below the subject threshold share 'rare'), bands, unknown
    bands = {**_band_fields(spec["prefix"]), **_band_fields(spec["header"]), **_band_fields(spec["goal"])}
    for f in spec["prefix"]:
        name = f["field"]
        if name in bands:
            values = bands[name]
        else:
            values = sorted(v for v, subj in c.prefix[name].items() if len(subj) >= min_subjects or "rare" not in f)
            if "rare" in f:
                values.append(f["rare"])
        for v in [*values, unknown]:
            add(f"PREFIX//{name}//{v}", cls="prefix", field=name, value=v)
    for block, fields, seen in (("HEADER", spec["header"], c.header), ("GOAL", spec["goal"], c.goal)):
        for f in fields:
            name = f["field"]
            if name in bands:
                values = bands[name]
            elif "episode" in f or "domain_events" in f:
                values = ["no", "yes"]
            else:
                values = sorted(str(v) for v in seen[name] if v is not None)
            for v in [*values, unknown]:
                add(f"{block}//{name}//{v}", cls=block.lower(), field=name, value=v)
    for k, label in enumerate(tl.labels()):
        add(f"INT//{label}", cls="interval", interval=k)
    # state tokens: infusion keys (bins of their rates) and respiratory device / mode codes
    state_keys = {}
    for key in sorted(k for k, subj in c.state_subjects.items() if len(subj) >= min_subjects):
        vals = _pool(c.state[key])
        edges = _edges(vals, spec) if len(vals) else []
        first = add(f"STATE//{key}{suffix['missing']}", cls="state", key=key, bin=None)
        for k in range(len(edges) + 1 if len(vals) else 0):
            add(f"STATE//{key}" + suffix["bin"].format(k=k), cls="state", key=key, bin=k)
        state_keys[key] = {"edges": edges, "first": first}
    for code in sorted(k for k, subj in c.resp.items() if len(subj) >= min_subjects):
        add(f"STATE//{code}", cls="state", key=code, bin=None)
    for dom in spec["windows"]["state"]["active_domains"]:
        add(f"STATE//{dom}//active", cls="state", key=f"STATE//{dom}//active", bin=None)
    # timeline and end codes, numbered in canonical order (domain rank, code, bin)
    end_domains = {e["domain"] for e in spec["end"] if "domain" in e}
    # an infusion's codes (start, dose change, stop, ...) stand or fall with its key, so the state rule can hold;
    # and a kept key has every action the rule reads, seen in train or not (a held-out stay's infusion running at its
    # window start is a 'going' event its key may never have had in train): the closure codes take the key's rate
    # bins and have no train values
    kept = {code for code, subj in c.subjects.items()
            if (len(c.state_subjects[infusion_key(code)[0]]) if infusion_key(code) else len(subj)) >= min_subjects}
    inf = spec["windows"]["state"]["infusions"]
    closure: dict[str, list] = {}
    for key in sorted({infusion_key(code)[0] for code in kept if infusion_key(code)}):
        for action in (*inf["begins"], *inf["continues"], *inf["ends"]):
            code = infusion_code(key, action)
            if code not in kept:
                kept.add(code)
                closure[code] = [] if action in inf["ends"] or key not in state_keys else state_keys[key]["edges"]
    hier_subjects: dict[str, set] = defaultdict(set)
    for code, subj in c.subjects.items():
        h = hierarchical(code)
        if h:
            hier_subjects[h[0]] |= subj
            for sub in h[1]:
                hier_subjects[sub] |= subj
    entries = []   # (rank, sort key, code, kind)
    for code in sorted(kept):
        if hierarchical(code) is None:
            entries.append((domain_rank(code), sort_key(code), code, "code"))
    from icutwin.meds.domains import DOMAIN_ORDER
    domains = sorted({*DOMAIN_ORDER, *(code.split("//", 1)[0] for code in c.subjects)} - end_domains)
    for d in domains:
        entries.append((domain_rank(d + "//"), d + "//OOV", d + "//OOV", "oov"))
    systems_seen = sorted(s for s in hier_subjects if s.count("//") == 1 and len(hier_subjects[s]) >= min_subjects)
    for sysname in systems_seen:
        entries.append((domain_rank(sysname), sysname, sysname, "system"))
    entries.sort()
    codes, systems = {}, {}
    value_sketches, value_sketches_ds = {}, {}
    for rank, _sort, code, kind in entries:
        if kind == "system":
            systems[code] = add(code, cls="end" if code.split("//")[0] in end_domains else "event",
                                kind="system", code=code, rank=rank, category_kind="")
            continue
        if kind == "oov":
            codes[code] = {"kind": "oov", "first": add(code, cls="event", kind="oov", code=code, rank=rank,
                                                       category_kind="")}
            continue
        cls = "end" if code.split("//")[0] in end_domains or code in {e for e in _end_codes(spec, code)} else "event"
        info = {"kind": "flag", "category_kind": _category_kind(code)}
        key = infusion_key(code)
        extra = {"key": key[0], "action": key[1]} if key else {}
        extra["category_kind"] = info["category_kind"]
        if code in c.numeric or code in closure:
            vals = c.values[code] if code in c.numeric else {}
            pooled = _pool(vals)
            edges = _edges(pooled, spec) if code in c.numeric else list(closure[code])
            info.update(kind="numeric", edges=edges)
            info["first"] = add(code + suffix["bin"].format(k=0), cls=cls, kind="bin", code=code, bin=0, rank=rank,
                                **extra)
            for k in range(1, len(edges) + 1):
                add(code + suffix["bin"].format(k=k), cls=cls, kind="bin", code=code, bin=k, rank=rank, **extra)
            info["na"] = add(code + suffix["missing"], cls=cls, kind="na", code=code, rank=rank, **extra)
            info["text"] = add(code + suffix["text"], cls=cls, kind="text", code=code, rank=rank, **extra)
            value_sketches[code] = _sketches(pooled, edges)
            value_sketches_ds[code] = {ds: _sketches(v, edges) for ds, v in sorted(vals.items())}
        else:
            info["first"] = add(code, cls=cls, kind="flag", code=code, rank=rank, **extra)
        codes[code] = info
    codes["END//OOV"] = {"kind": "oov", "first": add("END//OOV", cls="end", kind="oov", code="END//OOV",
                                                     rank=len(DOMAIN_ORDER), category_kind="")}
    chunks = {}
    for sub in sorted(s for s in hier_subjects if s.count("//") == 2 and len(hier_subjects[s]) >= min_subjects):
        chunks[sub] = add(sub, cls="chunk", level=int(sub.split("//")[1]), code=sub)
    for tok, m in clock_tokens(spec):           # last, so that extend_clock gives an older vocabulary the same tokens
        add(tok, **m)
    interval_sketches = {}
    for ds in [*datasets, "*"]:
        per = defaultdict(list)
        for d, ks in c.gaps.items():
            if ds in ("*", d):
                for k, mins in ks.items():
                    per[k].append(_flat(mins))
        pooled_gaps = {k: np.concatenate(v) for k, v in per.items()}
        interval_sketches[ds] = [[float(x) for x in np.quantile(pooled_gaps[k], QUANTILES)]
                                 if k in pooled_gaps and len(pooled_gaps[k]) else None for k in range(len(tl.labels()))]
    for m in meta:
        m.setdefault("category_kind", "")
    return Vocab(tokens, meta, codes, chunks, systems, state_keys, value_sketches, value_sketches_ds,
                 interval_sketches, spec_hash(), {"stays": n_stays, "datasets": datasets, "min_subjects": min_subjects,
                                                  "infusion_closure_codes": len(closure)})


def clock_tokens(spec: dict) -> list[tuple[str, dict]]:
    """The clock block's tokens (spec ``clock``) in their fixed order: the marker, the days 2 to ``max_day`` (the
    first block opens day 2), then per treatment its states."""
    c = spec.get("clock")
    if not c:
        return []
    out = [(c["token"], {"cls": "clock", "role": "marker"})]
    out += [(f"DAY//{d}", {"cls": "clock", "role": "day", "day": d}) for d in range(2, int(c["max_day"]) + 1)]
    out += [(f"STATUS//{t}//{s}", {"cls": "clock", "role": "status", "treatment": t, "state": s})
            for t in c["treatments"] for s in c["states"]]
    return out


def extend_clock(v: Vocab) -> Vocab:
    """``v`` with the clock tokens appended (a vocabulary fitted before the spec had them: every id unchanged, the
    new tokens last, as a fit now places them); ``v`` itself when it has them."""
    new = [(t, {**m, "category_kind": ""}) for t, m in clock_tokens(load_spec()) if t not in v.index]
    if not new:
        return v
    return Vocab([*v.tokens, *(t for t, _m in new)], [*v.meta, *(m for _t, m in new)], v.codes, v.chunks, v.systems,
                 v.state_keys, v.value_sketches, v.value_sketches_ds, v.interval_sketches, spec_hash(),
                 {**v.fitted_on, "clock_tokens_appended": len(new)})


def _end_codes(spec: dict, code: str) -> list[str]:
    from icutwin.tokenization.spec import code_pattern
    return [code for e in spec["end"] if "code" in e and code_pattern(e["code"]).match(code)]
