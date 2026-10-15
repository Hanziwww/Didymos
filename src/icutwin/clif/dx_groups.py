"""Harmonized admission diagnosis group (``ext_icu_stay.adm_dx_group``): operative status x organ system.

The groups are the APACHE II / IV admission diagnostic groups, as HiRID records them and the HiRID-ICU-Benchmark
maps them (Yeche et al. 2021, appendix table 7), with eICU's APACHE IV organ systems (musculoskeletal/skin,
transplant) and AmsterdamUMCdb's APACHE II / IV / NICE groups:

* ``medical_<system>`` (APACHE "non-operative") and ``surgical_<system>`` (APACHE "operative"/"post-operative");
* sepsis and intoxication are medical groups (APACHE II's sepsis group, APACHE IV's intoxication group; eICU's
  APACHE IV files sepsis under cardiovascular and overdoses under neurology, the diagnosis names separate them);
* gynecologic is a surgical group (HiRID's APACHE II 'Gynecology surgical'); a medical admission of that system
  is genitourinary;
* ``<status>_other``: the source's 'other' group; ``<status>_unspecified``: the operative status is known but
  the source records no system (MIMIC's general medicine and general surgery services);
* NULL: neither is known (NWICU; HiRID and SICdb stays without a group or surgery record).

Every source takes the group from what is known at ICU admission: APACHE admission diagnoses (eICU, HiRID,
AmsterdamUMCdb), the surgery site and admission form (SICdb), the service (MIMIC), the operation (INSPIRE).
Discharge diagnoses (MIMIC, NWICU and SICdb billing codes) are never used: they are not known at admission.
"""

from __future__ import annotations

SYSTEMS = ("cardiovascular", "respiratory", "gastrointestinal", "neurologic", "sepsis", "trauma",
           "metabolic_endocrine", "hematologic", "genitourinary", "gynecologic", "musculoskeletal_skin",
           "intoxication", "transplant", "other")
MEDICAL_ONLY = ("sepsis", "intoxication")
SURGICAL_AS_MEDICAL = {"gynecologic": "genitourinary"}   # a medical admission of a surgical-only system
MEDICAL = tuple(f"medical_{s}" for s in SYSTEMS if s not in SURGICAL_AS_MEDICAL) + ("medical_unspecified",)
SURGICAL = tuple(f"surgical_{s}" for s in SYSTEMS if s not in MEDICAL_ONLY) + ("surgical_unspecified",)
ADM_DX_GROUPS = MEDICAL + SURGICAL


def group_sql(surgical: str, system: str) -> str:
    """SQL for ``adm_dx_group`` from a boolean ``surgical`` (NULL: unknown) and a ``system`` (NULL: none).

    A medical-only system (sepsis, intoxication) of a surgical admission does not name its group: the
    operation does (``surgical_unspecified`` unless the source gives the operated system)."""
    as_medical = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in SURGICAL_AS_MEDICAL.items())
    medical_only = ", ".join(f"'{s}'" for s in MEDICAL_ONLY)
    return f"""(CASE WHEN ({surgical}) IS NULL THEN NULL
                     WHEN ({surgical}) THEN 'surgical_' || CASE WHEN ({system}) IS NULL OR ({system}) IN ({medical_only})
                                                                THEN 'unspecified' ELSE ({system}) END
                     ELSE 'medical_' || coalesce(CASE ({system}) {as_medical} ELSE ({system}) END, 'unspecified') END)"""
