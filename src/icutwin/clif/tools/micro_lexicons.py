"""Generate the shared microbiology lexicons from the CLIF 3.0 vocabularies.

``lexicon/organisms.csv``: free-text organism name -> CLIF ``organism_category`` and ``organism_group``.
CLIF names organisms ``genus_species`` or ``genus_sp``; each species gets a pattern ``genus ... species``
(laboratory abbreviations of the genus allowed, species stem tolerant of common misspellings), each genus a
genus-only pattern matched after every species pattern. Curated rows (coagulase-negative staphylococci,
streptococcal groups, Gram-stain morphologies, no growth, flora, cancelled tests) come first. CLIF ships no
category -> group mapping for cultures, so the group is derived: an explicit override, else the group named
after the genus, else the kingdom-level 'other' group.

``lexicon/antimicrobials.csv``: susceptibility-panel antibiotic name -> CLIF ``antimicrobial_category``.

    python -m icutwin.clif.tools.micro_lexicons
"""

from __future__ import annotations

import csv
import io
import re
from importlib import resources
from pathlib import Path

from icutwin.clif.conform import allowed_values

OUT = Path(str(resources.files("icutwin.clif").joinpath("lexicon")))

GENUS_ALIAS = {   # laboratory abbreviations of genus names
    "staphylococcus": r"staph(ylococc(us|i))?\.?", "streptococcus": r"strep(tococc(us|i))?\.?",
    "escherichia": r"e(scherichia|\.)", "enterococcus": r"enterococc(us|i)", "clostridioides": r"(clostridi(oides|um)|c\.)",
    "haemophilus": r"(ha?emophilus|h\.)", "klebsiella": r"(klebsiella|k\.)", "pseudomonas": r"(pseudomonas|p\.)",
    "candida": r"(candida|c\.)", "mycobacterium": r"mycobacteri(um|a)",
}
CURATED = [  # (priority, category, pattern, provenance)
    (1, "__skip__", r"cancel|not done|test not performed|duplicate|contamination with|results are invalid"
                    r"|invalidates results|specimen rejected",
     "cancelled, not performed or invalid specimen: not a result"),
    (2, "no_growth", r"^\W*no\b|no growth|no significant growth|no (micro)?organisms?|<\s?10,?000|culture negative"
                     r"|^negative$|sterile",
     "no growth: result statements that open with 'no' ('no salmonella or shigella found', 'no anaerobes "
     "isolated', Gram stain 'no microorganisms seen'), and < 10,000 cfu/ml (CLIF example)"),
    # negations the species / group rows below would otherwise misread (RE2 has no look-around)
    (3, "streptococcus_sp", r"not group\W*a\b", "beta streptococci, not group A"),
    (3, "yeast", r"not c\.?\s?albicans", "yeast, presumptively not C. albicans"),
    (4, "mycobacterium_avium", r"intracellulare|chimaera|\bmai\b|avium", "M. avium / intracellulare / chimaera complex"),
    (4, "fungus_other", r"trichophyton|neoscytalidium|dermatophyte", "moulds without a CLIF category"),
    (5, "staphylococcus_coagneg", r"coag(ulase)?\W*neg|\bcons\b", "coagulase-negative staphylococci"),
    (5, "staphylococcus_aureus", r"aureus|\bmrsa\b|\bmssa\b|coag(ulase)?\W*pos|staph\w*\W+coag\W*\+",
     "Staphylococcus aureus (incl. MRSA / MSSA screens)"),
    (5, "streptococcus_agalactiae", r"group\W*b\b|\bgbs\b|agalactiae", "group B streptococci"),
    (5, "streptococcus_pyogenes", r"group\W*a\b|pyogenes", "group A streptococci"),
    (5, "streptococcus_viridans", r"viridans", "viridans streptococci"),
    (5, "streptococcus_alphahemolytic", r"alpha\W*h(a)?emoly", "alpha-hemolytic streptococci"),
    (5, "streptococcus_anginosus", r"anginosus|milleri", "S. anginosus (milleri) group"),
    (5, "clostridioides_difficile", r"difficile|c\.?\s?diff\b", "Clostridioides (Clostridium) difficile"),
    (5, "corynebacterium_sp", r"diphtheroid", "diphtheroids = Corynebacterium spp."),
    (6, "gram_negative_diplococci", r"gram\W*neg\w*\W+(\w+\W+){0,2}diplococc", "Gram stain morphology"),
    (6, "gram_positive_cocci", r"gram\W*pos\w*\W+(\w+\W+){0,2}cocc|\bgpc\b", "Gram stain morphology"),
    (6, "gram_negative_rod", r"gram\W*neg\w*\W+(\w+\W+){0,2}(rod|bacill)|\bgnr\b|non-?ferment", "Gram stain morphology"),
    (6, "gram_positive_rod", r"gram\W*pos\w*\W+(\w+\W+){0,2}(rod|bacill)|\bgpr\b", "Gram stain morphology"),
    (70, "yeast", r"\byeast", "yeast not identified further"),
    (80, "fungus_other", r"\bmou?ld\b|\bfung(us|i|al)\b|hyph", "mould / fungus not identified further"),
    (85, "virus_other", r"virus", "virus not identified further"),
    (90, "bacteria_other", r"flora|mixed|commensal|contaminant|bacteria|bacilli|\bcocci", "bacteria not identified further"),
    (95, "bacteria_other", r"^other$", "positive culture, organism not named (eICU 'Other'; mostly bacterial)"),
]
GROUP_OVERRIDE = {
    "staphylococcus_aureus": "staphylococcus_coag_pos", "staphylococcus_pseudintermedius": "staphylococcus_coag_pos",
    "staphylococcus_sp": "staphylococcus_nos", "clostridioides_difficile": "clostridium_difficile",
    "candida_glabrata": "torulopsis_galbrata", "burkholderia_cepacia": "pseudomonas_burkholderia_cepacia",
    "stenotrophomonas_maltophilia": "pseudomonas_stenotrophomonas_xanthomonas_maltophilia",
    "mycobacterium_tuberculosis": "tuberculosis", "mycobacterium_avium": "mycobacteria_avium_bovium_haemophilum_intercelluare",
    "moraxella_catarrhalis": "branhamelia_moraxella_catarrhalis", "serratia_marcescens": "serratia_marcescens",
    "helicobacter_pylori": "helicobacter_pylori", "fusobacterium_nucleatum": "fusobacterium_nucleatum",
    "streptococcus_pneumoniae": "streptococcus", "gram_negative_diplococci": "gram_negative_diplococci",
    "yeast": "yeast", "no_growth": "no_growth", "bacteria_other": "bacteria_other", "fungus_other": "fungus_other",
    "virus_other": "viral_other",
}
GENUS_GROUP = {   # genus -> group when the group is not named after the genus
    "staphylococcus": "staphylococcus_coag_neg", "clostridium": "clostridium_wo_difficile",
    "pseudomonas": "pseudomonas_wo_cepacia_maltophilia", "mycobacterium": "mycobacterium_other",
    "candida": "candida_nos", "aspergillus": "aspergillus_nos", "mucor": "mucormycosis_zygomycetes_rhizopus",
    "rhizopus": "mucormycosis_zygomycetes_rhizopus", "propionibacterium": "propionbacterium",
    "cutibacterium": "propionbacterium", "moraxella": "branhamelia_moraxella_catarrhalis",
    "peptostreptococcus": "anaerobes_wo_bacteroides_clostridium", "prevotella": "anaerobes_wo_bacteroides_clostridium",
    "veillonella": "anaerobes_wo_bacteroides_clostridium", "porphyromonas": "anaerobes_wo_bacteroides_clostridium",
    "fusobacterium": "anaerobes_wo_bacteroides_clostridium", "actinomyces": "anaerobes_wo_bacteroides_clostridium",
}
FUNGAL_GENERA = {"candida", "aspergillus", "cryptococcus", "fusarium", "mucor", "rhizopus", "penicillium",
                 "scedosporium", "trichosporon", "saccharomyces", "geotrichum", "alternaria", "cladosporium",
                 "paecilomyces", "acremonium", "rhodotorula", "pichia", "malassezia", "exophiala", "lichtheimia",
                 "curvularia", "trichophyton", "zygomycete", "pneumocystis", "histoplasma", "blastomyces",
                 "coccidioides", "sporothrix", "bipolaris", "scopulariopsis", "aureobasidium", "chrysosporium"}


def _group(cat: str, groups: set[str]) -> str:
    if cat in GROUP_OVERRIDE:
        return GROUP_OVERRIDE[cat]
    if cat in groups:
        return cat
    genus = cat.split("_")[0]
    if genus in GENUS_GROUP:
        return GENUS_GROUP[genus]
    if genus in groups:
        return genus
    return "fungus_other" if genus in FUNGAL_GENERA else "bacteria_other"


def _species_pattern(genus: str, species: str) -> str:
    g = GENUS_ALIAS.get(genus, re.escape(genus))
    stem = species if len(species) <= 6 else species[: len(species) - 3]   # 'maltophila' typos still match
    tail = r"\b" if stem == species else ""
    return rf"\b{g}\W*{re.escape(stem)}{tail}"


def organisms() -> list[dict]:
    cats = sorted(allowed_values("microbiology_culture", "organism_category"))
    groups = set(allowed_values("microbiology_culture", "organism_group"))
    rows = [{"priority": p, "organism_category": c, "pattern": pat,
             "organism_group": "" if c == "__skip__" else _group(c, groups), "provenance": prov}
            for p, c, pat, prov in CURATED]
    for c in cats:   # every category keeps its standard pattern; curated rows only add phrasings
        genus, _, species = c.partition("_")
        if not species:
            continue                                   # morphology / 'other' rows are curated above
        if species == "sp":
            pat, prio = rf"\b{GENUS_ALIAS.get(genus, re.escape(genus))}(\b|\s)", 60
        else:
            pat, prio = _species_pattern(genus, species), 20
        rows.append({"priority": prio, "organism_category": c, "pattern": pat, "organism_group": _group(c, groups),
                     "provenance": "CLIF 3.0 organism category name"})
    return rows


ANTI_ALIAS = {
    "trimethoprim_sulfamethoxazole": r"trimethoprim\W*sulfa|sulfa\w*\W*trimethoprim|co-?trimox|\bsxt\b|\btmp\W*smx",
    "piperacillin_tazobactam": r"piperacillin\W*tazo|pip\W*tazo|zosyn",
    "ampicillin_sulbactam": r"ampicillin\W*sulbactam|unasyn", "amoxicillin_clavulanate": r"amoxicillin\W*clavul|augmentin",
    "ceftazidime_avibactam": r"ceftazidime\W*avibactam", "ceftolozane_tazobactam": r"ceftolozane\W*tazo",
    "ticarcillin_clavulanate": r"ticarcillin\W*clavul", "meropenem_vaborbactam": r"meropenem\W*vaborbactam",
    "ceftazidime_clavulanate": r"ceftazidime\W*clavul", "quinupristin_dalfopristin": r"quinupristin|synercid",
    "penicillin": r"penicillin(\s*g|\s*v)?\b", "meropenem": r"merope?nem|meropenam", "imipenem": r"imipenem",
    "polymyxin_b": r"polymyxin\W*b", "fusidic_acid": r"fusidic", "rifampin": r"rifampi(n|cin)",
}


def antimicrobials() -> list[dict]:
    # imipenem/relebactam has no CLIF category; it must not be read as imipenem
    rows = [{"priority": 1, "antimicrobial_category": "__skip__", "pattern": r"relebactam",
             "provenance": "agent without a CLIF antimicrobial category"}]
    for c in sorted(allowed_values("microbiology_susceptibility", "antimicrobial_category")):
        combo = c in ANTI_ALIAS and "_" in c
        pat = ANTI_ALIAS.get(c, r"\b" + r"[ _-]?".join(re.escape(p) for p in c.split("_")) + r"\b")
        rows.append({"priority": 10 if combo else 50, "antimicrobial_category": c, "pattern": pat,
                     "provenance": "CLIF 3.0 antimicrobial category name" + (" + aliases" if c in ANTI_ALIAS else "")})
    return rows


def _write(name: str, rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (int(r["priority"]), list(r.values())[1])))
    (OUT / name).write_text(buf.getvalue(), encoding="utf-8")


def main() -> None:
    _write("organisms.csv", organisms())
    _write("antimicrobials.csv", antimicrobials())


if __name__ == "__main__":
    main()
