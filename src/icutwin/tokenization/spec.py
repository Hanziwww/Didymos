"""The grammar spec (``configs/grammar/spec.yaml``) and the names every compiled artifact derives from it."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from importlib import resources

import yaml


@lru_cache(maxsize=1)
def spec_text() -> str:
    return resources.files("icutwin.configs").joinpath("grammar", "spec.yaml").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def load_spec() -> dict:
    return yaml.safe_load(spec_text())


def spec_hash() -> str:
    return hashlib.sha256(spec_text().encode("utf-8")).hexdigest()


def code_pattern(template: str) -> re.Pattern:
    """``"SITE//{dataset}//{value}"`` -> a regex capturing ``value`` (other placeholders match one code part)."""
    out, pos = "", 0
    for m in re.finditer(r"\{(\w+)\}", template):
        out += re.escape(template[pos:m.start()])
        out += "(?P<value>[^/]+)" if m.group(1) == "value" else "[^/]+"
        pos = m.end()
    return re.compile("^" + out + re.escape(template[pos:]) + "$")


def band_labels(bands: list[float]) -> list[str]:
    """Labels of [a, b) bands with the last band open: [0, 18, 30] -> ['0-18', '18-30', '30+']."""
    fmt = lambda x: str(int(x)) if float(x).is_integer() else str(x)  # noqa: E731
    return [f"{fmt(a)}-{fmt(b)}" for a, b in zip(bands, bands[1:])] + [f"{fmt(bands[-1])}+"]


def band_of(value: float, bands: list[float]) -> int:
    """Index of the band holding ``value``; values below the first edge count in the first band."""
    k = 0
    for i, edge in enumerate(bands):
        if value >= edge:
            k = i
    return k


def interval_labels(edges: list[int]) -> list[str]:
    """Minute edges -> interval token labels ('1-5m', ..., '3-7d', '7d+')."""
    def unit(m: int) -> str:
        if m % 1440 == 0:
            return f"{m // 1440}d"
        if m % 60 == 0:
            return f"{m // 60}h"
        return f"{m}m"

    labels = []
    for a, b in zip(edges, edges[1:]):
        ua, ub = unit(a), unit(b)
        labels.append(f"{ua[:-1]}-{ub}" if ua[-1] == ub[-1] else f"{ua}-{ub}")
    return labels + [f"{unit(edges[-1])}+"]
