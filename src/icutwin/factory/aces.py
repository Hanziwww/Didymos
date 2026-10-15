"""ACES task configs for the factory's binary targets (plan *Prediction-task generation* 1: tasks are ACES configs,
predicates over the harmonized codes defined once for all seven datasets), and label extraction with the ACES
reference implementation (es-aces), so a task runs identically on real and synthetic MEDS.

The trigger is the ICU admission (``ICU//admission//<unit>``; one per MEDS subject, an ICU stay). With trigger time
t and horizon h (hours after the ICU admission):
* ``death``: in-hospital death (``meds.outcomes``: the 'expired' hospital discharge or an ICU discharge 'dead')
  after t;
* ``icu_discharge_within``: an ICU discharge in (t, t + h] (the factory's ``in_icu_at`` is its complement);
* ``<event>_within``: the event in (t, t + h], among stays without it in [0, t]. Events are raw-code predicates,
  as the obligation registers read them: an IMV device record; a vasoactive infusion's start or a running one; a
  CRRT or IHD record (``code_onsets`` gives the cache the same onsets; its default onsets are the derived episodes).
Every task keeps the stays still in the ICU at t (no ICU discharge in [0, t]); the input window is [0, t].
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from icutwin.generate.engine import VASOACTIVES
from icutwin.meds.outcomes import EXPIRED, ICU_DEAD

PREDICATES = {
    "icu_admission": {"code": {"regex": "^ICU//admission//"}},
    "icu_discharge": {"code": {"regex": "^ICU//discharge//"}},
    "expired": {"code": EXPIRED},
    "icu_dead": {"code": ICU_DEAD},
    "death": {"expr": "or(expired, icu_dead)"},
    "imv": {"code": "RESP//device_category//imv"},
    "vasopressor": {"code": {"regex": "^MED_CON//(?:" + "|".join(VASOACTIVES) + ")//.+//(?:start|going)//[^/]+$"}},
    "rrt": {"code": {"regex": "^(?:CRRT|IHD)//"}},
}
EVENTS = ("imv", "vasopressor", "rrt")


def config(target: str, trigger_h: int, horizon_h: int = 0) -> dict:
    """The ACES config of a task (module docstring); ``target``: death, icu_discharge_within or <event>_within."""
    t, h = int(trigger_h), int(horizon_h)
    windows = {
        "input": {"start": "trigger", "end": f"start + {t}h", "start_inclusive": True, "end_inclusive": True,
                  "has": {"icu_discharge": "(None, 0)"}, "index_timestamp": "end"},
    }
    if target == "death":
        windows["target"] = {"start": "input.end", "end": None, "start_inclusive": False, "end_inclusive": True,
                             "label": "death"}
    elif target == "icu_discharge_within" or target.endswith("_within"):
        pred = "icu_discharge" if target == "icu_discharge_within" else target[:-len("_within")]
        if pred != "icu_discharge":
            windows["input"]["has"][pred] = "(None, 0)"
        windows["target"] = {"start": "input.end", "end": f"start + {h}h", "start_inclusive": False,
                             "end_inclusive": True, "label": pred}
    else:
        raise ValueError(target)
    preds = dict(PREDICATES)
    used = {"icu_admission", "icu_discharge", "death", "expired", "icu_dead", *EVENTS}
    return {"predicates": {k: v for k, v in preds.items() if k in used}, "trigger": "icu_admission",
            "windows": windows}


def extract(cfg: dict, meds_files: list) -> dict:
    """ACES labels of a task over MEDS data shards: subject id -> label (bool)."""
    import polars as pl
    import ruamel.yaml
    from aces import config as aces_config
    from aces import predicates, query
    from omegaconf import DictConfig

    out = {}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "task.yaml"
        with path.open("w", encoding="utf-8") as f:
            ruamel.yaml.YAML(typ="safe", pure=True).dump(cfg, f)
        for f in meds_files:
            task = aces_config.TaskExtractorConfig.load(path)   # per shard: a query changes the config it is given
            preds = predicates.get_predicates_df(task, DictConfig({"path": str(f), "standard": "meds"}))
            res = query.query(task, preds)
            if res is None or res.height == 0:
                continue
            for sid, lab in res.select(pl.col("subject_id"), pl.col("label")).iter_rows():
                out[int(sid)] = bool(lab)
    return out


def code_onsets(stays: list) -> dict:
    """Per event, the hours since the ICU admission of each stay's first matching code at or after it (NaN: none):
    the onsets ACES's predicates see, for a cache that should match them."""
    import re

    pats = {e: re.compile(PREDICATES[e]["code"]["regex"]) if isinstance(PREDICATES[e]["code"], dict)
            else re.compile("^" + re.escape(PREDICATES[e]["code"]) + "$") for e in EVENTS}
    out = {e: np.full(len(stays), np.nan) for e in EVENTS}
    for i, s in enumerate(stays):
        for t, _r, code, _n, _x in s.timeline:
            h = (t - s.icu_in).total_seconds() / 3600
            if h < 0:
                continue
            for e, p in pats.items():
                if np.isnan(out[e][i]) and p.match(code):
                    out[e][i] = h
    return out
