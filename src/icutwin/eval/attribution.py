"""Paper P, C2: attribution on individual trajectories (``docs/p1_paper_protocol.md`` C2).

A stay as known at a prediction time is split into **event groups**: a channel family (``family``) x a time block
before the cutoff (0-6 h, 6-24 h, over 24 h, the pre-ICU window counting as over 24 h), plus the statics (prefix and
header fields other than the dataset and site). Three attributions of a target risk to the groups:

* **counterfactual editing** (the reference): the group's events removed from the decoded stay (statics set to
  unknown) and the prompt encoded again (``edited_prompt``; valid by construction), the risk's change;
* **integrated gradients** on the token embeddings, summed over each group's tokens (``token_groups``: a token's
  family from its code, its block from the interval tokens' pooled medians walked back from the prompt's end);
* **attention rollout** (Abnar & Zuidema 2020) and attention x gradient, from the last token, summed the same way.

``deletion_curve``: the risk after removing the top-k groups of a ranking, cumulatively; its area against a random
ranking measures a method's faithfulness.
"""

from __future__ import annotations

import copy
import re

import numpy as np

BLOCKS = ((0, 360, "0-6h"), (360, 1440, "6-24h"), (1440, 10 ** 9, ">24h"))
VASOACTIVE = re.compile(r"^MED_CON//(norepinephrine|epinephrine|vasopressin|dopamine|phenylephrine|dobutamine|"
                        r"milrinone|angiotensin|levosimendan)//")
SEDATION = re.compile(r"^MED_CON//(propofol|midazolam|dexmedetomidine|fentanyl|morphine|ketamine|lorazepam|"
                      r"remifentanil|sufentanil|hydromorphone|clonidine|ketamine)//")
FLUIDS = re.compile(r"^(OUTPUT//|INPUT//|MED_CON//(sodium_chloride|lactated_ringers_solution|dextrose|albumin|"
                    r"balanced|plasma)|MED_INT//(sodium_chloride|lactated_ringers_solution|albumin)//)")
FAMILIES = ("vitals", "labs", "vasoactives", "sedation", "fluids_output", "respiratory", "rrt_mcs", "other_meds",
            "assessments", "other")


def family(code: str) -> str:
    """The channel family of an event code."""
    if code.startswith("VITAL//"):
        return "vitals"
    if code.startswith("LAB//"):
        return "labs"
    if VASOACTIVE.match(code):
        return "vasoactives"
    if SEDATION.match(code):
        return "sedation"
    if FLUIDS.match(code):
        return "fluids_output"
    if code.startswith("RESP//"):
        return "respiratory"
    if code.startswith(("CRRT//", "IHD//", "MCS//", "ECMO//", "RRT//")):
        return "rrt_mcs"
    if code.startswith(("MED_CON//", "MED_INT//", "MED_ORD//")):
        return "other_meds"
    if code.startswith("ASSESS//"):
        return "assessments"
    return "other"


def block(minutes_before: float) -> str:
    for lo, hi, name in BLOCKS:
        if lo <= minutes_before < hi:
            return name
    return ">24h"


def stay_groups(s, cutoff_hours: float) -> dict:
    """{group: [timeline indices]} of a prospective stay (events at or before the cutoff), plus ``statics``."""
    cut = s.icu_in.timestamp() + cutoff_hours * 3600
    out: dict = {}
    for k, (t, _r, code, _n, _x) in enumerate(s.timeline):
        before = (cut - t.timestamp()) / 60.0
        if before < 0:
            continue
        out.setdefault(f"{family(code)}|{block(before)}", []).append(k)
    out["statics"] = []
    return out


def consistent(timeline: list) -> list:
    """The timeline without infusion records that no longer follow a running start after an edit (the grammar's
    infusion_state rule: a dose change or a stop only while the infusion, keyed by drug, unit and route, runs; a
    start or a running record makes it run, a stop ends it)."""
    from icutwin.tokenization.spec import load_spec

    acts = load_spec()["windows"]["state"]["infusions"]
    begins = set(acts.get("begins", ("start", "going")))
    continues = set(acts.get("continues", ("dose_change",)))
    ends = set(acts.get("ends", ("stop",)))
    running: set = set()
    out = []
    for ev in timeline:
        code = ev[2]
        parts = code.split("//")
        if parts[0] == "MED_CON" and len(parts) == 5:
            key, action = (parts[1], parts[2], parts[4]), parts[3]
            if action in begins:
                running.add(key)
            elif action in continues and key not in running:
                continue
            elif action in ends:
                if key not in running:
                    continue
                running.discard(key)
        out.append(ev)
    return out


def edited_prompt(s, v, cutoff_hours: float, group, groups: dict) -> list[int]:
    """The prompt of the stay with one group (or a set of groups) removed (``tokenization.encode.prospective``),
    infusion records left orphaned by the removal dropped too (``consistent``)."""
    from icutwin.tokenization.encode import prospective
    from icutwin.tokenization.spec import load_spec

    drop_groups = {group} if isinstance(group, str) else set(group)
    e = copy.copy(s)
    if "statics" in drop_groups:
        e.prefix = {k: (x if k in ("dataset", "site") else None) for k, x in s.prefix.items()}
        e.header = {f["field"]: None for f in load_spec()["header"]}
    drop = {k for g in drop_groups if g != "statics" for k in groups.get(g, ())}
    if drop:
        e.timeline = consistent([x for k, x in enumerate(s.timeline) if k not in drop])
    return prospective(e, v, cutoff_hours)


def token_groups(prompt: list[int], v) -> list[str | None]:
    """Each prompt token's group (None for special, interval and state tokens): the code's family, and the block from
    the minutes walked back from the prompt's end over the interval tokens (their pooled medians)."""
    from icutwin.eval.forecast import Reader

    rd = Reader(v)
    meta = v.meta
    after = np.zeros(len(prompt))                    # minutes from each token to the prompt's end
    acc = 0.0
    for k in range(len(prompt) - 1, -1, -1):
        after[k] = acc
        acc += rd.minutes.get(int(prompt[k]), 0.0)
    out: list = []
    for k, tok in enumerate(prompt):
        m = meta[int(tok)]
        cls = m.get("cls")
        if cls in ("prefix", "header"):
            out.append(None if m.get("field") in ("dataset", "site") else "statics")
        elif cls == "event" and m.get("code"):
            out.append(f"{family(m['code'])}|{block(after[k])}")
        else:
            out.append(None)
    return out


def group_sums(values: np.ndarray, groups: list) -> dict:
    out: dict = {}
    for x, g in zip(values, groups, strict=True):
        if g is not None:
            out[g] = out.get(g, 0.0) + float(x)
    return out


def deletion_curve(risk_of, order: list[str], k: int = 5) -> list[float]:
    """The risk after removing the first 1..k groups of ``order`` cumulatively (``risk_of(set of groups)``)."""
    return [risk_of(set(order[:j])) for j in range(1, min(k, len(order)) + 1)]


def rank_agreement(a: dict, b: dict, top: int = 3) -> dict:
    """Spearman correlation and top-k overlap of two group scores (on their common groups, by absolute value of
    a signed effect where given)."""
    from scipy.stats import spearmanr

    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return {"spearman": None, "top_overlap": None, "groups": len(common)}
    x, y = np.array([a[g] for g in common]), np.array([b[g] for g in common])
    rho = spearmanr(x, y).statistic
    ta = set(np.array(common)[np.argsort(-x)[:top]])
    tb = set(np.array(common)[np.argsort(-y)[:top]])
    return {"spearman": None if not np.isfinite(rho) else float(rho), "top_overlap": len(ta & tb) / top,
            "groups": len(common)}
