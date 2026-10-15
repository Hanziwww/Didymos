"""In-hospital death: one definition for every label read off MEDS events or tokens (the factory's cache, the ACES
predicates, forecasting, utility, fidelity): the hospital discharge category 'expired' or an ICU discharge 'dead'.

``MEDS_DEATH`` is not an in-hospital outcome: it is the patient's static death event at any recorded time
(``docs/meds.md``). MIMIC-IV, AmsterdamUMCdb, SICdb, NWICU and INSPIRE record deaths after discharge (MIMIC-IV up to a
year, from state records), and eICU's patient-level death carries a later hospitalization's: on the 2% subset 788
MIMIC stays have ``MEDS_DEATH`` against 220 in-hospital deaths.
"""

from __future__ import annotations

EXPIRED = "HOSP//discharge_category//expired"
ICU_DEAD = "ICU//discharge//dead"
HOSP_OUTCOME = "HOSP//discharge_category//"


def in_hospital_death(codes) -> float:
    """1.0 for an in-hospital death, 0.0 for a hospital discharge alive, NaN when neither is recorded."""
    codes = set(codes)
    if EXPIRED in codes or ICU_DEAD in codes:
        return 1.0
    if any(c.startswith(HOSP_OUTCOME) for c in codes):
        return 0.0
    return float("nan")


class TokenOutcome:
    """The same definition over token ids (a vocabulary's tokens of those codes)."""

    def __init__(self, v):
        self.end = v.id("[END]")
        self.dead = {i for i, m in enumerate(v.meta) if m.get("code") in (EXPIRED, ICU_DEAD)}
        self.hosp = {i for i, m in enumerate(v.meta) if m.get("cls") == "end"
                     and (m.get("code") or "").startswith(HOSP_OUTCOME)}

    def death(self, ids) -> bool | None:
        """True / False as ``in_hospital_death``; None without an end block, or with one but neither outcome."""
        ids = [int(x) for x in ids]
        if self.end not in ids:
            return None
        if self.dead & set(ids):
            return True
        return False if self.hosp & set(ids[ids.index(self.end):]) else None
