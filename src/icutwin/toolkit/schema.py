"""Toolkit results (``docs/toolkit.md``): plain JSON dictionaries, one per tool call.

Every result has ``tool``, ``version``, ``patient`` (reference, kind, dataset; no data), ``anchor`` (hours, the
prediction time ``t_min``), ``settings`` (rollouts, caps, seeds, mode), ``outputs`` (the tool's), ``evidence`` (the
paper's references and claims for these outputs, ``evidence.yaml``), ``caveats`` and ``provenance`` (the session's:
weights, gate status and scope, backend, git). ``clean`` makes any value JSON-safe (NaN and infinities as null).
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

VERSION = 1
REQUIRED = ("tool", "version", "settings", "outputs", "evidence", "caveats", "provenance")


@lru_cache(maxsize=1)
def evidence() -> dict:
    import yaml

    return yaml.safe_load((Path(__file__).with_name("evidence.yaml")).read_text(encoding="utf-8"))


def clean(x):
    """JSON-safe: numpy scalars and arrays as Python values, NaN and infinities as None, paths as strings."""
    import numpy as np

    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def patient_info(p) -> dict:
    return {"ref": p.ref, "kind": p.kind, "dataset": p.dataset, "unknown_site": p.unknown_site}


def result(tool: str, session, *, patient=None, anchor=None, settings=None, outputs=None, caveats=None,
           evidence_keys=None, provenance=None) -> dict:
    ev = evidence().get(tool, {})
    sel = {k: ev[k] for k in (evidence_keys or ("refs", "claim", "framing")) if k in ev}
    prov = session.provenance(**(provenance or {}))
    scope = (prov.get("gate_not_in_v1") or {}).get("conditional_and_counterfactual_sampling")
    gate_note = [f"Prompted and intervened sampling under the privacy gate: {scope}."] \
        if scope and tool in ("forecast", "whatif", "teach") else []
    out = {"tool": tool, "version": VERSION,
           "patient": patient_info(patient) if patient is not None else None,
           "anchor": {"hours": anchor.hours, "t_min": anchor.t_min, "prompt_tokens": len(anchor.prompt)}
           if anchor is not None else None,
           "settings": settings or {}, "outputs": outputs or {}, "evidence": sel,
           "caveats": list(ev.get("caveats", [])) + gate_note + list(caveats or []), "provenance": prov}
    return clean(out)


def validate(r: dict) -> None:
    missing = [k for k in REQUIRED if k not in r]
    if missing:
        raise ValueError(f"a toolkit result without {missing}")
    if r["version"] != VERSION:
        raise ValueError(f"result version {r['version']}, toolkit {VERSION}")
