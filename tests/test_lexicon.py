"""Shared drug lexicon: vocabulary conformance, RE2 validity and known-answer resolution."""

from __future__ import annotations

import duckdb
import pytest

from icutwin.clif.conform import allowed_values
from icutwin.clif.lexicon import resolve, rows


def test_lexicon_categories_are_clif_or_declared() -> None:
    cont = allowed_values("medication_admin_continuous", "med_category")
    inter = allowed_values("medication_admin_intermittent", "med_category")
    for r in rows():
        assert not r["continuous"] or r["continuous"] in cont, r
        assert not r["intermittent"] or r["intermittent"] in inter, r


def test_categories_clif_mimic_emits_as_intermittent_are_allowed() -> None:
    """CLIF-MIMIC files some drugs CLIF 3.0 lists as continuous only (eptifibatide, argatroban, esomeprazole) as
    intermittent administrations; strict harmonization must accept every category its mapping can emit there."""
    import csv
    from importlib import resources

    text = resources.files("icutwin.clif").joinpath("maps", "mimic", "tool_intermittent_categories.csv").read_text(
        encoding="utf-8")
    inter = allowed_values("medication_admin_intermittent", "med_category")
    assert not [r["med_category"] for r in csv.DictReader(text.splitlines()) if r["med_category"] not in inter]


def test_local_vocabulary_extensions_are_documented() -> None:
    """Local extensions name their CLIF 3.0 status, source and reason; medication categories also their routes
    and units."""
    from icutwin.clif.conform import extensions

    local = {(table, column, cat): spec for table, cols in extensions().items() for column, cats in cols.items()
             for cat, spec in cats.items() if isinstance(spec, dict) and spec.get("origin") == "local"}
    assert {"argatroban", "eptifibatide", "esomeprazole"} <= {cat for _, _, cat in local}
    assert ("medication_admin_intermittent", "med_route_category", "inhaled") in local
    for (table, column, cat), spec in local.items():
        assert spec["clif_3_0"] and spec["source"] and spec["why"], cat
        if column == "med_category":
            assert spec["routes"] and spec["units"], cat


def test_every_mapping_csv_is_well_formed() -> None:
    """Regex patterns contain commas ({m,n}); an unquoted one silently shifts every later column."""
    import csv
    from importlib import resources

    root = resources.files("icutwin.clif")
    files = [*root.joinpath("lexicon").iterdir(), *(f for d in root.joinpath("maps").iterdir() if d.is_dir()
                                                      for f in d.iterdir())]
    for f in files:
        if not f.name.endswith(".csv"):
            continue
        for row in csv.DictReader(f.read_text(encoding="utf-8").splitlines()):
            assert None not in row and None not in row.values(), (f.name, row)


def test_lexicon_patterns_are_valid_re2() -> None:
    con = duckdb.connect()
    for r in rows():
        con.execute("SELECT regexp_matches('x', ?)", [r["pattern"]])   # raises on invalid RE2


@pytest.mark.parametrize("name, ingredient", [
    # HiRID (Swiss brands)
    ("Liquemin 5000 U/ml", "heparin"), ("Disoprivan 2% BOLUS", "propofol"), ("Sodium nitro", "nitroprusside"),
    ("Noradrenalin 20 µg/ml Perfusor", "norepinephrine"), ("Adrenalin 100 µg/ml Perfusor", "epinephrine"),
    ("Perlinganit mg/ml inf", "nitroglycerin"), ("Thyrotardin 100 ug", "liothyronine"),
    ("Kalium Phosphat für Perfusor", "potassium_phosphate"), ("Kalium Chlorid 15% 10 ml", "potassium_chloride"),
    ("Dalacin C 600 Phosphat Amp.", "clindamycin"), ("Tazobac 2 g inf", "piperacillin_tazobactam"),
    ("Augmentin 1.2 Inj Lsg", "amoxicillin_clavulanate"), ("Clamoxyl Inj Lsg", "amoxicillin"),
    ("Perfalgan 1g", "acetaminophen"), ("Dafalgan Brausetbl 1g", "acetaminophen"),
    ("Tracrium inj 10 mg/ml", "atracurium"), ("Glucose 20% /100ml", "dextrose_other"),
    ("Glucose 10%", "dextrose_10_water"), ("SmofKabiven NL 1500 ml +F;+E", "tpn"),
    ("Magnesium Sulfat 50% 10 ml", "magnesium_sulfate"), ("Magnesiocard 5 mmol sachet", "magnesium_sulfate"),
    ("Insulin NovoRapid", "insulin"), ("Catapresan 25µg /ml Perfusor", "clonidine"),
    ("Aspégic Inj Lsg 500 mg/5ml", "aspirin"), ("Vibravenös Inj Lsg 100 mg 5 ml", "doxycycline"),
    # eICU (US names)
    ("Nimbex", "cisatracurium"), ("piperacillin-tazobactam 3.375 g", "piperacillin_tazobactam"),
    ("ZOSYN", "piperacillin_tazobactam"), ("Levophed", "norepinephrine"), ("D5W", "dextrose_5_water"),
    ("LACTATED RINGERS", "lactated_ringers_solution"), ("SODIUM CHLORIDE 0.9 % IV SOLN", "sodium_chloride"),
    ("magnesium aspartate", "magnesium_sulfate"),
    # combination products (CLIF 3.0 combination categories; SICdb abbreviates HCTZ as 'hct')
    ("Coaprovel", "irbesartan_hctz"), ("irbesartan+hct", "irbesartan_hctz"),
    ("sacubitril-valsartan 49-51 mg oral tab", "sacubitril_valsartan"), ("valsartan+amlodipin", "valsartan_amlodipine"),
    ("lisinopril-hydrochlorothiazide 20-12.5 mg oral tab", "lisinopril_hctz"),
    # a two-component category wins over either component (the 'Tazobac' brand must not catch ceftolozane)
    ("ceftolozane + tazobactam", "ceftolozane_tazobactam"), ("piperacillin + tazobactam", "piperacillin_tazobactam"),
    ("imipenem-relebactam 1.25 g", "imipenem_relebactam"), ("meropenem-vaborbactam", "meropenem_vaborbactam"),
    ("sulbactam-durlobactam", "sulbactam_durlobactam"), ("ampicillin + sulbactam", "ampicillin_sulbactam"),
    # a vehicle never wins over an active ingredient, even one without a CLIF category
    ("calcium gluconate 2 g/50 ml ns", "calcium"), ("vancomycin 1 gm in ns 250 ml ivpb", "vancomycin"),
    ("NS 1000 ml", "sodium_chloride"), ("1000 ml  -  dextrose 5 % iv soln", "dextrose_5_water"),
    # line flushes and locks are not administrations of their drug
    ("saline flush", "line_flush"), ("sodium chl 0.9% flushsyrg(10ml)", "line_flush"),
    ("heparin lock flush 100 unit/ml", "line_flush"), ("Heparin Na 100 IE/ml Spülung", "line_flush"),
    # generic names of CLIF categories (INSPIRE uses generic names)
    ("cefotetan", "cefotetan"), ("cefoxitin in dextrose, iso-osm 2 gram/50 ml iv pgbk", "cefoxitin"),
    ("nafcillin", "nafcillin"), ("Noxafil susp", "posaconazole"), ("ropivacain 0.2%", "ropivacaine"),
    # Dutch / German spellings (AmsterdamUMCdb, SICdb): 'f' for 'ph', no final 'e'
    ("Morfine", "morphine"), ("Fenylefrine", "phenylephrine"), ("Noradrenaline (Norepinefrine)", "norepinephrine"),
    ("Natriumbikarbonat 8,4%", "sodium_bicarbonate"), ("Natriumbicarbonaat  8,4%", "sodium_bicarbonate"),
    ("natrium-kalium-fosfaat", "potassium_phosphate"), ("Theofylline", "aminophylline"), ("Efedrine", "ephedrine"),
    ("Cefotaxim", "cefotaxime"), ("Posaconazol", "posaconazole"), ("Tigecyclin", "tigecycline"),
    # a prodrug or a longer drug name is not its shorter namesake (found at full scale, 2026-10-05)
    ("ValACYclovir", "valacyclovir"), ("8457: Valaciclovir (Zelitrex)", "valacyclovir"), ("Aciclovir", "acyclovir"),
    ("VALGANCICLOVIR 450 MG ORAL TAB", "valganciclovir"), ("Cymevene i.v.", "ganciclovir"),
    ("DiCLOXacillin", "dicloxacillin"), ("Cloxacilline", "cloxacillin"), ("pivampicillin", "pivampicillin"),
    # a counter-ion is not the drug: calcium and magnesium salts of other actives, laxatives and antacids
    ("Leucovorin Calcium", "leucovorin"), ("Calciumfolinat 100 mg", "leucovorin"),
    ("LEVOMEFOLATE CALCIUM 15 MG ORAL TAB", "levomefolate"), ("PATIROMER CALCIUM SORBITEX 8.4 G ORAL PWPK", "patiromer"),
    ("calcium dobesilate", "calcium_dobesilate"), ("Calcium Gluconate", "calcium"), ("Calcium Acetate", "calcium"),
    ("Aluminum-Magnesium Hydrox.-Simethicone", "aluminium_magnesium_antacid"),
    ("Maalox/Diphenhydramine/Lidocaine", "aluminium_magnesium_antacid"),
    ("MAGNESIUM HYDROXIDE 400 MG/5 ML ORAL SUSP", "magnesium_hydroxide"), ("Magnesium Citrate", "magnesium_citrate"),
    ("omeprazole magnesium", "omeprazole"), ("Choline Magnesium Trisalicylate", "choline_magnesium_trisalicylate"),
    ("Magnesium Oxide", "magnesium_sulfate"), ("magnesium lactate", "magnesium_sulfate"),
    ("MAGNESIUM SULF 2 G/50 ML NS", "magnesium_sulfate"),
    # MIMIC's abbreviation of co-trimoxazole; buprenorphine-naloxone is buprenorphine, not a naloxone reversal dose
    ("Sulfameth/Trimethoprim DS", "trimethoprim_sulfamethoxazole"), ("Trimethoprim", "trimethoprim"),
    ("Buprenorphine-Naloxone Film (8mg-2mg)", "buprenorphine"), ("NARCAN", "naloxone"), ("Temgesic", "buprenorphine"),
    # AmsterdamUMCdb parenteral nutrition 'without potassium' is not potassium
    ("TPV A - zonder Kalium", "tpn"), ("TPV StructoKabiven", "tpn"),
    # non-systemic forms have no CLIF route category, whatever the ingredient
    ("acetaminophen 650 mg rect supp", "non_systemic_formulation"), ("Benuron Supp 1000 mg", "non_systemic_formulation"),
    ("tobramycin 0.3 % opht drop", "non_systemic_formulation"), ("NaCl 0,9% Augentropfen", "non_systemic_formulation"),
    ("fentanyl  25 mcg/hr transderm pt72", "non_systemic_formulation"), ("mupirocin 2 % top oint", "non_systemic_formulation"),
])
def test_known_names_resolve(name: str, ingredient: str) -> None:
    hit = resolve([name])[name]
    assert hit is not None and hit["ingredient"] == ingredient, (name, hit)


def test_ingredient_names_resolve_to_themselves() -> None:
    """An ingredient's own name (where its pattern matches it) resolves to that ingredient: a shorter namesake's
    pattern must not catch it (acyclovir caught valacyclovir, cloxacillin dicloxacillin)."""
    import re

    own = {}
    for r in rows():
        name = r["ingredient"].replace("_", " ")
        if re.search(r["pattern"], name):
            own[name] = r["ingredient"]
    assert len(own) > 400
    hits = resolve(sorted(own))
    wrong = {n: (h or {}).get("ingredient") for n, h in hits.items() if (h or {}).get("ingredient") != own[n]}
    assert wrong == {}


def test_unknown_names_stay_unresolved() -> None:
    assert resolve(["$  name session", "Survimed OPD"]) == {"$  name session": None, "Survimed OPD": None}


def test_known_ingredients_without_clif_category_are_not_gaps() -> None:
    """They resolve (not a lexicon gap); a category, where they get one, is a registry extension, never CLIF 3.0."""
    hits = resolve(["Trace elements", "Rivotril 0.5 mg/ml", "Catapresan 25µg /ml Perfusor"])
    for name, hit in hits.items():
        assert hit is not None, name
        for col in ("continuous", "intermittent"):
            assert not hit[col] or hit[f"{col}_kind"] == "concept", (name, col, hit[col])
    assert not hits["Trace elements"]["continuous"] and not hits["Trace elements"]["intermittent"]
    assert hits["Catapresan 25µg /ml Perfusor"]["continuous"] == "ext_clonidine"


def test_every_clif_medication_category_is_in_the_lexicon() -> None:
    """A CLIF 3.0 category no lexicon row can produce would be lost silently in every dataset."""
    cats = set(allowed_values("medication_admin_continuous", "med_category")) | set(
        allowed_values("medication_admin_intermittent", "med_category"))
    produced = {r["continuous"] for r in rows()} | {r["intermittent"] for r in rows()}
    not_products = {
        "micafungin_posaconazole_voriconazole",  # a grouping entry of the vocabulary, not a product
        "pitocin",  # brand duplicate of the continuous 'oxytocin' category; all oxytocin maps to 'oxytocin'
    }
    assert sorted(cats - produced - not_products) == []


def test_rows_without_category_never_name_a_clif_category() -> None:
    cats = set(allowed_values("medication_admin_continuous", "med_category")) | set(
        allowed_values("medication_admin_intermittent", "med_category"))
    wrong = [r["ingredient"] for r in rows() if not r["continuous"] and not r["intermittent"] and r["ingredient"] in cats]
    assert wrong == []


def test_pitocin_brand_maps_to_generic_oxytocin() -> None:
    hit = resolve(["PITOCIN 20 UNITS/1000 ML IV INFUSION"])["PITOCIN 20 UNITS/1000 ML IV INFUSION"]
    assert (hit["ingredient"], hit["continuous"]) == ("oxytocin", "oxytocin")


def test_mapping_files_hold_no_control_characters() -> None:
    """A regex written through an unescaped string turns a word boundary into a backspace and never matches."""
    from importlib import resources

    root = resources.files("icutwin.clif")
    files = [*root.joinpath("lexicon").iterdir(), *(f for d in root.joinpath("maps").iterdir() if d.is_dir()
                                                      for f in d.iterdir())]
    control = {chr(c) for c in range(32)} - {"\t", "\n", "\r"}
    for f in files:
        if f.name.endswith(".csv"):
            text = f.read_text(encoding="utf-8")
            assert not control & set(text), f.name


def test_european_spellings_of_clif_categories_resolve() -> None:
    """German and Dutch drop the final 'e' (cefotaxim, posaconazol) and Dutch writes 'f' for 'ph' (morfine)."""
    cats = set()
    for t in ("medication_admin_continuous", "medication_admin_intermittent", "medication_orders"):
        cats |= {c for c in allowed_values(t, "med_category") if not c.startswith("ext_")}
    variants = {}
    for c in cats:
        name = c.replace("_", " ")
        if " " in name:
            continue
        if name.endswith("e") and len(name) > 5:
            variants[name[:-1]] = c
        if "ph" in name:
            variants[name.replace("ph", "f")] = c
    not_spellings = {"acetaminofen", "alteplas"}   # Dutch / German use paracetamol; alteplase is spelled alike
    hits = resolve(sorted(set(variants) - not_spellings))
    wrong = [v for v, hit in hits.items() if hit is None or variants[v] not in (hit["continuous"], hit["intermittent"])]
    assert wrong == []
