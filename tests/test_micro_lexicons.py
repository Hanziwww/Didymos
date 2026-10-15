"""Shared microbiology lexicons: CLIF conformance, reproducibility and known answers."""

from __future__ import annotations

import pytest

from icutwin.clif.conform import allowed_values
from icutwin.clif.lexicon import resolve, rows
from icutwin.clif.tools import micro_lexicons


def test_organism_rows_conform_to_clif() -> None:
    cats = set(allowed_values("microbiology_culture", "organism_category"))
    groups = set(allowed_values("microbiology_culture", "organism_group"))
    for r in rows("organisms"):
        if r["organism_category"] == "__skip__":
            continue
        assert r["organism_category"] in cats and r["organism_group"] in groups, r
    assert {r["antimicrobial_category"] for r in rows("antimicrobials")} - {"__skip__"} <= set(
        allowed_values("microbiology_susceptibility", "antimicrobial_category"))
    assert {r["fluid_category"] for r in rows("micro_specimens")} <= set(allowed_values("microbiology_culture", "fluid_category"))
    methods = {r["method_category"] for r in rows("micro_tests")} - {"__skip__", "__nonculture__"}
    assert methods <= set(allowed_values("microbiology_culture", "method_category"))


def test_generated_lexicons_match_their_generator() -> None:
    """organisms.csv / antimicrobials.csv are generated; hand edits would be lost on the next generation."""
    gen = {r["pattern"] for r in micro_lexicons.organisms()}
    assert gen == {r["pattern"] for r in rows("organisms")}
    assert {r["pattern"] for r in micro_lexicons.antimicrobials()} == {r["pattern"] for r in rows("antimicrobials")}


@pytest.mark.parametrize("name, category, group", [
    ("STAPH AUREUS COAG +", "staphylococcus_aureus", "staphylococcus_coag_pos"),
    ("STAPHYLOCOCCUS, COAGULASE NEGATIVE", "staphylococcus_coagneg", "staphylococcus_coag_neg"),
    ("ESCHERICHIA COLI", "escherichia_coli", "escherichia"), ("E. coli", "escherichia_coli", "escherichia"),
    ("Stenotrophomonas maltophila", "stenotrophomonas_maltophilia",          # eICU spelling
     "pseudomonas_stenotrophomonas_xanthomonas_maltophilia"),
    ("BETA STREPTOCOCCUS GROUP B", "streptococcus_agalactiae", "streptococcus"),
    ("BETA STREPTOCOCCI, NOT GROUP A", "streptococcus_sp", "streptococcus"),
    ("CLOSTRIDIUM DIFFICILE", "clostridioides_difficile", "clostridium_difficile"),
    ("CANDIDA GLABRATA", "candida_glabrata", "torulopsis_galbrata"),
    ("YEAST, PRESUMPTIVELY NOT C. ALBICANS", "yeast", "yeast"),
    ("gram positive cocci - in clusters", "gram_positive_cocci", "gram_positive_cocci"),
    ("no growth", "no_growth", "no_growth"), ("NO SALMONELLA OR SHIGELLA FOUND.", "no_growth", "no_growth"),
    ("no microorganisms seen.", "no_growth", "no_growth"), ("<10,000 organisms/ml.", "no_growth", "no_growth"),
    ("MIXED BACTERIAL FLORA", "bacteria_other", "bacteria_other"),
    ("CANCELLED", "__skip__", ""),
])
def test_organism_names(name: str, category: str, group: str) -> None:
    hit = resolve([name], "organisms")[name]
    assert (hit["organism_category"], hit["organism_group"]) == (category, group), name


@pytest.mark.parametrize("name, category", [
    ("TRIMETHOPRIM/SULFA", "trimethoprim_sulfamethoxazole"), ("PIPERACILLIN/TAZO", "piperacillin_tazobactam"),
    ("PENICILLIN G", "penicillin"), ("meropenam", "meropenem"), ("IMIPENEM/RELEBACTAM", "__skip__"),
    ("amoxicillin/clavulonic acid", "amoxicillin_clavulanate"),
])
def test_antimicrobial_names(name: str, category: str) -> None:
    assert resolve([name], "antimicrobials")[name]["antimicrobial_category"] == category


@pytest.mark.parametrize("name, method", [
    ("Blood Culture, Routine", "culture"), ("GRAM STAIN", "gram_stain"), ("ACID FAST SMEAR", "smear"),
    ("C. difficile PCR", "__nonculture__"), ("CMV IgG ANTIBODY", "__nonculture__"),
    ("Respiratory Viral Culture", "culture"), ("CHROMOSOME ANALYSIS-BONE MARROW", "__skip__"),
])
def test_test_methods(name: str, method: str) -> None:
    assert resolve([name], "micro_tests")[name]["method_category"] == method


@pytest.mark.parametrize("name, fluid", [
    ("BLOOD CULTURE", "blood_buffy"), ("Blood, Central Line", "blood_buffy"), ("URINE", "genito_urinary_tract"),
    ("BRONCHOALVEOLAR LAVAGE", "respiratory_tract_lower"), ("Sputum, Tracheal Specimen", "respiratory_tract_lower"),
    ("MRSA SCREEN", "nasopharynx_upperairway"), ("CATHETER TIP-IV", "catheter_tip"), ("CSF;SPINAL FLUID", "meninges_csf"),
    ("Rectal Swab", "gastrointestinal_tract"), ("FLUID RECEIVED IN BLOOD CULTURE BOTTLES", "other_unspecified"),
])
def test_specimen_fluids(name: str, fluid: str) -> None:
    assert resolve([name], "micro_specimens")[name]["fluid_category"] == fluid
