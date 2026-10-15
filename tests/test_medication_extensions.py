"""Medication vocabulary registry (M4 D1): every category outside CLIF 3.0 is declared once, with its kind, and
standard CLIF terms stay distinct from consortium-practice, local and ``ext_`` categories."""

from __future__ import annotations

import pytest

from icutwin.clif.conform import (MEDICATION_TABLES, _permissible, allowed_values, extensions,
                                  medication_extensions, table_schema)
from icutwin.clif.lexicon import base_rows, rows

OTHER = {"medication_admin_continuous": "medication_admin_intermittent",
         "medication_admin_intermittent": "medication_admin_continuous"}
COLUMN = {"medication_admin_continuous": "continuous", "medication_admin_intermittent": "intermittent"}


def _standard(table: str) -> set[str]:
    return set(_permissible(table, "med_category"))


def _all_standard() -> set[str]:
    return set().union(*(_standard(t) for t in (*MEDICATION_TABLES, "medication_orders")))


REGISTRY = medication_extensions()


@pytest.mark.parametrize("r", REGISTRY, ids=lambda r: f"{r['table'][17:]}:{r['med_category']}")
def test_each_category_is_outside_clif_3_0_and_named_by_its_kind(r: dict) -> None:
    table, cat, kind = r["table"], r["med_category"], r["kind"]
    assert table in MEDICATION_TABLES
    assert cat not in _standard(table), "a CLIF 3.0 term of the table is standard, not an extension"
    assert kind in ("consortium", "local", "concept")
    if kind == "concept":   # a drug CLIF 3.0 has no category for in any medication table
        assert cat.startswith("ext_") and cat not in _all_standard()
        assert r["clif_3_0"] == "none"
    elif kind == "local":   # the CLIF 3.0 term of the other administration table
        assert not cat.startswith("ext_") and cat in _standard(OTHER[table])
        assert r["clif_3_0"] == f"{OTHER[table]} only"
    else:                   # emitted by an official CLIF ETL
        assert not cat.startswith("ext_") and "CLIF-" in r["source"]
    assert r["why"] and r["source"] and r["routes"] and r["units"] and r["ingredients"]


@pytest.mark.parametrize("r", REGISTRY, ids=lambda r: f"{r['table'][17:]}:{r['med_category']}")
def test_groups_and_routes_are_clif_vocabulary(r: dict) -> None:
    table = r["table"]
    groups = set(_permissible(table, "med_group"))
    if r["kind"] == "consortium":   # the official ETL emits these without a group of the table
        assert not r["med_group"] or r["med_group"] in groups
    else:
        assert r["med_group"] in groups, r["med_group"]
    routes = allowed_values(table, "med_route_category")
    assert set(r["routes"].split("|")) <= routes
    assert r["untyped_iv"] in ("", "continuous")
    if r["untyped_iv"]:   # only an intermittent term of a drug CLIF 3.0 gives as an infusion can defer to it
        assert table == "medication_admin_intermittent" and r["kind"] == "local"


def test_each_category_is_declared_once_per_table() -> None:
    keys = [(r["table"], r["med_category"]) for r in REGISTRY]
    assert len(keys) == len(set(keys))


def test_the_written_lexicon_holds_clif_3_0_categories_only() -> None:
    """drugs.csv names CLIF 3.0 terms of the table; everything else comes from the registry."""
    for r in base_rows("drugs"):
        for table in MEDICATION_TABLES:
            cat = r[COLUMN[table]]
            assert not cat or cat in _standard(table), (r["ingredient"], table, cat)


def test_effective_lexicon_categories_are_standard_or_registry_with_their_kind() -> None:
    declared = {(r["table"], r["med_category"]): r["kind"] for r in REGISTRY}
    for r in rows("drugs"):
        for table in MEDICATION_TABLES:
            col = COLUMN[table]
            if not r[col]:
                continue
            if r[f"{col}_kind"]:
                assert declared[(table, r[col])] == r[f"{col}_kind"], (r["ingredient"], r[col])
            else:
                assert r[col] in _standard(table), (r["ingredient"], r[col])


def test_every_registry_category_is_reachable_from_the_lexicon() -> None:
    produced = {(t, r[COLUMN[t]]) for r in rows("drugs") for t in MEDICATION_TABLES if r[COLUMN[t]]}
    assert [(r["table"], r["med_category"]) for r in REGISTRY
            if (r["table"], r["med_category"]) not in produced] == []


def test_orders_accept_the_union_of_administration_extensions() -> None:
    orders = (extensions().get("medication_orders") or {}).get("med_category") or {}
    standard = _standard("medication_orders")
    for r in REGISTRY:
        assert r["med_category"] in standard or r["med_category"] in orders, r["med_category"]
    assert not set(orders) & standard
    assert orders["adenosine"]["group"] == "cardiac"   # extensions.yaml takes precedence


def test_registry_groups_reach_the_group_mapping() -> None:
    from icutwin.clif.harmonize import group_mappings

    _, mapping = group_mappings()[("medication_admin_continuous", "med_group")]
    assert mapping["potassium_repletion"] == "fluids_electrolytes"
    assert mapping["norepinephrine"] == "vasoactives"   # CLIF 3.0 mapping, not overridden
    _, mapping = group_mappings()[("medication_admin_intermittent", "med_group")]
    assert mapping["ext_metoprolol"] == "anti_hypertensive" and mapping["potassium_repletion"] == "other"


def test_kinds_resolve_by_name() -> None:
    from icutwin.clif.lexicon import resolve

    hits = resolve(["Kalium Chlorid 15% 10 ml", "metoprolol tartrate", "Noradrenalin 10 µg/ml Bolus",
                    "insulin glargine", "Dexdor inf. Lsg"])
    k = hits["Kalium Chlorid 15% 10 ml"]
    assert (k["continuous"], k["continuous_kind"], k["intermittent"], k["intermittent_kind"]) == \
        ("potassium_repletion", "local", "potassium_repletion", "")
    m = hits["metoprolol tartrate"]
    assert (m["intermittent"], m["intermittent_kind"], m["continuous"]) == ("ext_metoprolol", "concept", "")
    n = hits["Noradrenalin 10 µg/ml Bolus"]
    assert (n["continuous_kind"], n["intermittent"], n["intermittent_kind"], n["untyped_iv"]) == \
        ("", "norepinephrine", "local", "continuous")
    assert hits["insulin glargine"]["intermittent_kind"] == "consortium"
    assert hits["Dexdor inf. Lsg"]["continuous"] == "dexmedetomidine"


def test_orders_declare_their_route_column() -> None:
    """CLIF 3.0 medication_orders has only med_route_name; the route category is a declared extension column whose
    vocabulary is the administration tables' routes."""
    from clifpy.utils.crosswalk import load_schema

    assert "med_route_category" not in {c["name"] for c in load_schema("medication_orders", "3.0")["columns"]}
    assert "med_route_category" in {c["name"] for c in table_schema("medication_orders")["columns"]}
    routes = allowed_values("medication_admin_intermittent", "med_route_category") | allowed_values(
        "medication_admin_continuous", "med_route_category")
    assert allowed_values("medication_orders", "med_route_category") == routes


def test_route_extension_is_declared_for_intermittent_only() -> None:
    assert "inhaled" in allowed_values("medication_admin_intermittent", "med_route_category")
    cols = {c["name"]: c for c in table_schema("medication_admin_intermittent")["columns"]}
    assert "inhaled" not in cols["med_route_category"]["permissible_values"]   # not CLIF 3.0 there
