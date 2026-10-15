"""Medication dose units: clifpy's cleaning plus the two cases it gets wrong."""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

MILLION_UNITS = re.compile(r"(?i)^\s*million\s*units?\s*$")   # clifpy would clean it to 'monu'
# counting units (a count of product units, no amount of drug), after clifpy's cleaning: one spelling each, the
# one MIMIC's eMAR uses where it has the unit ('tab', 'cap', 'puff', 'neb', 'inh')
COUNTING_UNITS = {"tablet": "tab", "tablets": "tab", "tabs": "tab", "splittab": "tab", "capsule": "cap",
                  "capsules": "cap", "caps": "cap", "puffs": "puff", "inhalation": "inh", "inhalations": "inh",
                  "nebs": "neb", "nebule": "neb", "nebules": "neb", "drops": "drop", "gtt": "drop",
                  "sachets": "sachet", "packets": "packet", "sprays": "spray", "films": "film", "vl": "vial",
                  "vials": "vial", "ea": "each", "application": "appl", "applications": "appl"}


@lru_cache(maxsize=4096)
def clean_dose_unit(unit: str | None) -> tuple[str | None, float]:
    """(unit, dose factor): clifpy's dose-unit cleaning (``unit_converter``: formats, then name variants:
    'mL/hour' -> 'ml/hr', 'units' -> 'u', 'grams' -> 'g', 'µg' -> 'mcg'), plus 'gm' -> 'g' (MIMIC eMAR),
    'meq.' -> 'meq' (MIMIC's 'mEq.' of CRRT potassium) and million units -> u x 1e6 (penicillin G), which
    clifpy does not know, and one spelling per counting unit (``COUNTING_UNITS``: 'tablet' -> 'tab'). Other
    unknown units (mEq, mmol, tab) keep their cleaned spelling."""
    if unit is None:
        return None, 1.0
    if MILLION_UNITS.match(unit):
        return "u", 1e6
    from clifpy.utils.unit_converter import _clean_dose_unit_formats, _clean_dose_unit_names
    out = _clean_dose_unit_names(_clean_dose_unit_formats(pd.Series([unit], dtype="object"))).iloc[0]
    if out is None or (isinstance(out, float) and out != out):
        return None, 1.0
    out = re.sub(r"^meq\.(?=$|/)", "meq", re.sub(r"^gm(?=$|/)", "g", str(out)))
    return COUNTING_UNITS.get(out, out), 1.0
