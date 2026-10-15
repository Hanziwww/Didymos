"""Goal-first sampling's bindings (round 3, amendment 3 as revised; ``docs/m7_refinement.md``): which drawn goals the
grammar enforces, per dataset. The goal block's labels are derived (``tokenization.stays``: treatment episodes that
overlap the ICU stay, ``meds.derived``), while the sampler can only read tokens; the two disagree on some sources (a
single IMV charting forms no episode, so 22% of INSPIRE's 'imv no' stays chart IMV; mode-inferred ventilation has no
device token). A drawn goal therefore binds only where its token reading holds for the dataset's real stays, the
plan's rule for enforced constraints (value coupling: a rule qualifies when it holds for at least 99.9% of the real
cases; the others are conditioning, measured not enforced):

* a treatment drawn 'no' (IMV, vasopressor, RRT, MCS): none of its tokens after the ICU admission marker;
* an ICU outcome: no other ICU discharge token;
* a hospital outcome: its token in the end block and no other.

A treatment drawn 'yes' binds nothing (conditioning only), so no stay is held open waiting for an event its source
may chart otherwise. ``compute`` counts, per dataset and goal label, the training stays with the label and those
breaking its binding (the whole stay, ``ids``); ``save`` writes ``goal_bindings.json`` beside the vocabulary
(counts only, no stay).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

THRESHOLD = 0.001      # a binding is adopted for a dataset when at most 0.1% of its labelled training stays break it
MIN_STAYS = 100        # ... and at least this many training stays carry the label


def rules(v) -> dict:
    """Per goal token: (kind, token ids, required end token ids or None); kinds 'forbid_icu', 'forbid', 'end'."""
    from icutwin.generate.engine import GOAL_VASOPRESSORS, _events

    sets = {"imv": _events(v, lambda c, m: c == "RESP//device_category//imv"),
            "vasopressor": _events(v, lambda c, m: c.startswith("MED_CON//") and c.split("//")[1] in GOAL_VASOPRESSORS
                                   and m.get("action") in ("start", "going")),
            "rrt": _events(v, lambda c, m: c.startswith(("CRRT//", "IHD//"))),
            "mcs": _events(v, lambda c, m: c.startswith("MCS//"))}
    dis = {i: (m.get("code") or "") for i, m in enumerate(v.meta)
           if m.get("cls") == "event" and (m.get("code") or "").startswith("ICU//discharge//")}
    hosp = {i: (m.get("code") or "") for i, m in enumerate(v.meta)
            if m.get("cls") == "end" and (m.get("code") or "").startswith("HOSP//discharge_category//")}
    out = {}
    for i, m in enumerate(v.meta):
        if m.get("cls") != "goal":
            continue
        _g, f, label = v.tokens[i].split("//", 2)
        if f in sets and label == "no":
            out[i] = ("forbid_icu", frozenset(sets[f]), None)
        elif f == "icu_outcome" and label != "unknown":
            out[i] = ("forbid", frozenset(k for k, c in dis.items() if c != f"ICU//discharge//{label}"), None)
        elif f == "hosp_outcome" and label != "unknown":
            out[i] = ("end", frozenset(k for k, c in hosp.items() if c != f"HOSP//discharge_category//{label}"),
                      frozenset(k for k, c in hosp.items() if c == f"HOSP//discharge_category//{label}"))
    return out


def breaks(ids: np.ndarray, kind: str, toks: frozenset, need, marker: int, end: int) -> bool:
    """Whether a real stay (its whole token sequence) breaks a binding of its own goal label."""
    if kind == "forbid_icu":
        at = np.flatnonzero(ids == marker) if marker is not None else []
        tail = ids[at[0] + 1:] if len(at) else ids[:0]
        return bool(np.isin(tail, list(toks)).any()) if toks else False
    if kind == "forbid":
        return bool(np.isin(ids, list(toks)).any()) if toks else False
    at = np.flatnonzero(ids == end)
    block = ids[at[0] + 1:] if len(at) else ids[:0]
    return not len(at) or not np.isin(block, list(need)).any() or (bool(np.isin(block, list(toks)).any()) if toks else False)


def compute(tokens_root: Path, datasets: list, v=None, max_stays: int | None = None) -> dict:
    """Per dataset and goal token (by name): [training stays with the label, stays breaking its binding]."""
    import pyarrow.parquet as pq

    from icutwin.tokenization.vocab import Vocab

    v = v or Vocab.load(Path(tokens_root) / "vocab.json")
    rl = rules(v)
    g0, g1 = v.id("[GOAL]"), v.id("[/GOAL]")
    marker = next((i for i, m in enumerate(v.meta) if (m.get("code") or "") == "ICU//admission"), None)
    end = v.id("[END]")
    out = {}
    for ds in datasets:
        tab = pq.read_table(Path(tokens_root) / ds / "train.parquet", columns=["ids", "goal_ids"])
        if max_stays:
            tab = tab.slice(0, max_stays)
        counts = {v.tokens[g]: [0, 0] for g in rl}
        for ids, goal in zip(tab.column("ids").to_pylist(), tab.column("goal_ids").to_pylist()):
            if not goal or g0 not in goal:
                continue
            ids = np.asarray(ids, dtype=np.int64)
            for g in goal[goal.index(g0) + 1:goal.index(g1)]:
                if g in rl:
                    kind, toks, need = rl[g]
                    c = counts[v.tokens[g]]
                    c[0] += 1
                    c[1] += breaks(ids, kind, toks, need, marker, end)
        out[ds] = counts
    return out


def save(tokens_root: Path, datasets: list) -> Path:
    path = Path(tokens_root) / "goal_bindings.json"
    data = {"threshold": THRESHOLD, "min_stays": MIN_STAYS, "datasets": compute(tokens_root, datasets)}
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path


def adopted(v, data: dict) -> dict:
    """Prefix dataset token -> the goal tokens whose binding that dataset adopts."""
    thr, low = float(data.get("threshold", THRESHOLD)), int(data.get("min_stays", MIN_STAYS))
    tok = {m.get("value"): i for i, m in enumerate(v.meta) if m.get("cls") == "prefix" and m.get("field") == "dataset"}
    out = {}
    for ds, counts in data["datasets"].items():
        if ds in tok:
            out[tok[ds]] = frozenset(v.id(g) for g, (n, bad) in counts.items() if n >= low and bad <= thr * n)
    return out


def main(argv: list[str] | None = None) -> None:
    """``python -m icutwin.generate.goal_bindings <tokens root> [datasets...]``: write ``goal_bindings.json``."""
    import sys

    args = argv if argv is not None else sys.argv[1:]
    root = Path(args[0])
    datasets = args[1:] or ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
    path = save(root, datasets)
    data = json.loads(path.read_text(encoding="utf-8"))
    for ds, counts in data["datasets"].items():
        print(ds, {g.split("//", 1)[1]: f"{bad}/{n}" for g, (n, bad) in counts.items() if n})
    print("wrote", path)


if __name__ == "__main__":
    main()
