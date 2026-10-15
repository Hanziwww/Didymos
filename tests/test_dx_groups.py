"""Admission diagnosis groups: taxonomy, group rule and the shared system lexicon."""

from __future__ import annotations

import duckdb
import pytest

from icutwin.clif.conform import allowed_values
from icutwin.clif.dx_groups import ADM_DX_GROUPS, group_sql
from icutwin.clif.lexicon import resolve


def test_schema_declares_the_taxonomy() -> None:
    assert allowed_values("ext_icu_stay", "adm_dx_group") == set(ADM_DX_GROUPS)
    assert "surgical_sepsis" not in ADM_DX_GROUPS and "medical_gynecologic" not in ADM_DX_GROUPS


@pytest.mark.parametrize(("surgical", "system", "group"), [
    ("true", "'cardiovascular'", "surgical_cardiovascular"),
    ("true", "'sepsis'", "surgical_unspecified"),          # sepsis is a medical group: the operation names it
    ("true", "NULL", "surgical_unspecified"),
    ("false", "'gynecologic'", "medical_genitourinary"),   # gynecologic is a surgical group
    ("false", "NULL", "medical_unspecified"),
    ("false", "'intoxication'", "medical_intoxication"),
    ("NULL", "'trauma'", None),                            # operative status unknown: no group
])
def test_group_rule(surgical: str, system: str, group: str | None) -> None:
    assert duckdb.sql(f"SELECT {group_sql(surgical, system)}").fetchone()[0] == group


SYSTEM_NAMES = {   # APACHE organ systems as eICU, AmsterdamUMCdb and NICE name them
    "Cardiovascular": "cardiovascular", "Neurology": "neurologic", "Neurologic": "neurologic",
    "Respiratory": "respiratory", "Gastrointestinal": "gastrointestinal", "Metabolic/Endocrine": "metabolic_endocrine",
    "Trauma": "trauma", "Genitourinary": "genitourinary", "Hematology": "hematologic",
    "Musculoskeletal/Skin": "musculoskeletal_skin", "Transplant": "transplant",
    "Post-operatief cardiovasculair": "cardiovascular", "Non-operative gastro-intestinal": "gastrointestinal",
    "APACHEIV Post-operative musculoskeletal /skin": "musculoskeletal_skin",
    "APACHEIV  Non-operative haematological": "hematologic", "APACHE IV Non-operative Gastro-intestinal": "gastrointestinal",
    "APACHEIV Non-operative transplant": "transplant", "APACHEIV Post-operative trauma": "trauma",
    "Apache II Operatief  Gastr-intenstinaal": "gastrointestinal", "Apache II Non-Operatief Metabolisme": "metabolic_endocrine",
    "Apache II Operatief Renaal": "genitourinary", "Apache II Non-Operatief Respiratoir": "respiratory",
    "D_Thoraxchirurgie_CABG en Klepchirurgie": "cardiovascular", "D_Thoraxchirurgie_Cardio anders": "cardiovascular",
    "D_Thoraxchirurgie_Aorta chirurgie": "cardiovascular", "D_Thoraxchirurgie_Pulmonale chirurgie": "respiratory",
    "D_Algemene chirurgie_Buikchirurgie": "gastrointestinal", "D_Algemene chirurgie_Vaatchirurgie": "cardiovascular",
    "D_Algemene chirurgie_Endocrinologische chirurgie": "metabolic_endocrine",
    "D_Algemene chirurgie_KNO/Overige": "other", "D_Algemene chirurgie_Orthopedische chirurgie": "musculoskeletal_skin",
    "D_Algemene chirurgie_Transplantatie chirurgie": "transplant", "D_Algemene chirurgie_Trauma": "trauma",
    "D_Algemene chirurgie_Urogenitaal": "genitourinary",
    "D_Neurochirurgie _Vasculair chirurgisch": "neurologic", "D_Neurologie_Vasculair neurologisch": "neurologic",
    "D_Interne Geneeskunde_Cardiovasculair": "cardiovascular", "D_Interne Geneeskunde_Pulmonaal": "respiratory",
    "D_Interne Geneeskunde_Abdominaal": "gastrointestinal", "D_Interne Geneeskunde_Infectieziekten": "sepsis",
    "D_Interne Geneeskunde_Renaal": "genitourinary", "D_Interne Geneeskunde_Hematologisch": "hematologic",
    "D_Interne Geneeskunde_Overige": "other", "D_KNO/Overige": "other",
    "Sepsis, pulmonary": "sepsis", "Overdose, street drugs (opiates, cocaine, amphetamine)": "intoxication",
    "Toxicity, drug (i.e., beta blockers, calcium channel blockers, etc.)": "intoxication",
}


def test_lexicon_resolves_apache_and_aumc_system_names() -> None:
    got = {n: (r or {}).get("system") for n, r in resolve(list(SYSTEM_NAMES), name="dx_systems").items()}
    assert got == SYSTEM_NAMES
    assert resolve(["D_Algemene chirurgie_Algemeen"], name="dx_systems")["D_Algemene chirurgie_Algemeen"] is None
