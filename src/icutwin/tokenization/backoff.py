"""The opt-in code back-off for databases outside the vocabulary's training data (Paper P amendment 17, arm Z1).

Before encoding, each timeline event whose code the frozen vocabulary lacks is rewritten to the nearest code it keeps
(``backoff.yaml``: whole-code substitutions; for medications the same drug's kept unit and route, converting the value
where the units convert and dropping it where they do not). Events with a kept code are never touched. The rewrite
runs on the stay, so the state summary, validation and the round trip see the rewritten codes consistently.

Off unless the environment names a rule file (``ICUTWIN_BACKOFF``, set by ``tokenization.cli`` from ``backoff:``):
the training databases encode exactly as before.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from importlib import resources
from pathlib import Path

import yaml

ENV = "ICUTWIN_BACKOFF"
MED = ("MED_CON", "MED_INT")


@lru_cache(maxsize=4)
def load(path: str) -> dict:
    text = (resources.files("icutwin.tokenization").joinpath("backoff.yaml").read_text(encoding="utf-8")
            if path in ("", "default") else Path(path).read_text(encoding="utf-8"))
    rules = yaml.safe_load(text)
    rules["_family"] = {u: (fam, f) for fam, units in rules["unit_families"].items() for u, f in units.items()}
    return rules


def active() -> dict | None:
    """The rule set the environment names, or None (back-off off)."""
    p = os.environ.get(ENV)
    return load(p) if p is not None else None


def _kept(v, code: str) -> bool:
    info = v.codes.get(code)
    return info is not None and info["kind"] != "oov"


_INDEX: dict = {}


def _units(v, dom: str, drug: str, action: str, route: str) -> list[str]:
    """The kept units of (domain, drug, action, route), from the vocabulary's codes (cached per vocabulary)."""
    idx = _INDEX.get(id(v))
    if idx is None:
        idx = {}
        for code, info in v.codes.items():
            p = code.split("//")
            if p[0] in MED and len(p) == 5 and info["kind"] != "oov":
                idx.setdefault((p[0], p[1], p[3], p[4]), []).append(p[2])
        _INDEX[id(v)] = idx
    return idx.get((dom, drug, action, route), [])


def factor(a: str, b: str, drug: str, rules: dict) -> float | None:
    """1 <a> = factor <b> for this drug (the same family, or a drug-specific conversion), else None."""
    if a == b:
        return 1.0
    fam = rules["_family"]
    fa, fb = fam.get(a), fam.get(b)
    if fa and fb and fa[0] == fb[0]:
        return fa[1] / fb[1]
    for a2, tos in (rules.get("drug_conversions") or {}).get(drug, {}).items():
        ka = 1.0 if a2 == a else (fa[1] / fam[a2][1] if fa and a2 in fam and fam[a2][0] == fa[0] else None)
        if ka is None:
            continue
        for b2, f in tos.items():
            kb = 1.0 if b2 == b else (fam[b2][1] / fb[1] if fb and b2 in fam and fam[b2][0] == fb[0] else None)
            if kb is not None:
                return ka * f * kb
    return None


def backoff_code(code: str, num, v, rules: dict) -> tuple[str, object, str]:
    """(code, value, rule) for one event: the code itself when kept, else the first rule's kept code."""
    if _kept(v, code):
        return code, num, "kept"
    sub = (rules.get("codes") or {}).get(code)
    if sub and _kept(v, sub):
        return sub, num, "substitution"
    p = code.split("//")
    if p[0] not in MED or len(p) != 5:
        return code, num, "none"
    dom, drug, unit, action, route = p
    has = num is not None and num == num
    for rt in [route] + [r for r in rules.get("route_fallback", []) if r != route]:
        kept = _units(v, dom, drug, action, rt)
        if not kept:
            continue
        if dom == "MED_INT":
            if unit != "UNK" and has:                  # a convertible unit keeps the dose
                for u in sorted(kept, key=lambda x: (rules["dose_units"] + [x]).index(x)):
                    f = factor(unit, u, drug, rules)
                    if f is not None:
                        return f"{dom}//{drug}//{u}//{action}//{rt}", float(num) * f, "converted"
            for u in rules["dose_units"]:
                if u in kept:
                    return f"{dom}//{drug}//{u}//{action}//{rt}", None, "unit dropped"
            return f"{dom}//{drug}//{sorted(kept)[0]}//{action}//{rt}", None, "unit dropped"
        for u in rules["infusion_units"]:               # MED_CON
            if u in kept:
                val = num if action == "stop" else None
                return f"{dom}//{drug}//{u}//{action}//{rt}", val, "rate not recorded"
    return code, num, "none"


def rewrite(s, v, counts: Counter | None = None):
    """The stay with every lacking timeline code backed off (``s`` itself when the back-off is off)."""
    rules = active()
    if rules is None:
        return s
    out, changed = [], False
    for t, r, code, num, text in s.timeline:
        c2, n2, why = backoff_code(code, num, v, rules)
        if counts is not None and why not in ("kept",):
            counts[(why, code, c2)] += 1
        changed |= c2 != code
        out.append((t, r, c2, n2, text))
    return replace(s, timeline=out) if changed else s
