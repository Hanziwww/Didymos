"""S2b canonical conventions shared by all datasets: lab units/order categories, row rules, BP site, density."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from icutwin.clif.conform import connect, table_schema
from icutwin.clif.harmonize import (
    _apply_fixes, _lab_reference_units, canonical_labs, canonical_vitals, harmonize_dir, hourly_respiratory,
    hourly_vitals, lab_conventions, value_fixes,
)

T = lambda h, m: datetime(2100, 1, 1, h, m, tzinfo=UTC)  # noqa: E731


def _labs(path: Path, rows: list[tuple]) -> Path:
    cols = ["hospitalization_id", "lab_result_dttm", "lab_name", "lab_category", "lab_order_category",
            "reference_unit", "lab_value_numeric"]
    pq.write_table(pa.table({c: [r[i] for r in rows] for i, c in enumerate(cols)}), path)
    return path


def test_lab_units_and_order_categories_are_canonicalised(tmp_path: Path) -> None:
    src = _labs(tmp_path / "l.parquet", [
        ("H", T(1, 0), "Sodium", "sodium", "bmp", "mEq/L", 140.0),            # mEq/L = mmol/L for Na
        ("H", T(1, 0), "PLT", "platelet_count", "cbc", "10*3/uL", 200.0),     # spelling variant
        ("H", T(1, 0), "INR", "inr", "coags", None, 1.1),                    # missing unit accepted
        ("H", T(1, 0), "pH", "ph_arterial", "blood_gas", "(no units)", 7.4),  # reference unit with a space
        ("H", T(1, 0), "Lac", "lactate", "blood_gas", "mmol/L", 1.5),         # site panel -> canonical misc
        ("H", T(1, 0), "HCO3", "bicarbonate", "blood_gas", "mmol/L", 24.0),   # kept: method difference
        ("H", T(1, 0), "TnT", "troponin_t", "cardiac", "ng/L", 14.0),
        ("H", T(1, 0), "ESR", "esr", "misc", "mm/Hr", 20.0),
        ("H", T(1, 0), "lymph", "lymphocytes_absolute", "cbc", "10^3/µL\t", 1.0),  # CLIF-MIMIC spelling + tab
    ])
    stats = canonical_labs(connect(), src, tmp_path / "o.parquet")
    assert stats["violations"] == {}
    out = pd.read_parquet(tmp_path / "o.parquet").set_index("lab_category")
    assert out.loc["sodium", "reference_unit"] == "mmol/l"
    assert out.loc["platelet_count", "reference_unit"] == "10^3/ul"
    assert out.loc["inr", "reference_unit"] == "(no units)"
    assert out.loc["lactate", "lab_order_category"] == "misc"
    assert out.loc["bicarbonate", "lab_order_category"] == "blood_gas"
    assert out.loc["troponin_t", "lab_order_category"] == "misc"
    assert out.loc["esr", "reference_unit"] == "mm/hour"
    assert out.loc["lymphocytes_absolute", "reference_unit"] == "10^3/ul"
    assert stats["rows_missing_unit"] == 1


def test_lab_unit_that_is_not_a_spelling_variant_is_a_violation(tmp_path: Path) -> None:
    src = _labs(tmp_path / "l.parquet", [
        ("H", T(1, 0), "Ca", "calcium_total", "bmp", "mmol/L", 2.2),   # must be converted to mg/dl by the ETL
        ("H", T(1, 0), "Mg", "magnesium", "misc", "mEq/L", 1.6),        # mEq/L != mmol/L for divalent ions
    ])
    stats = canonical_labs(connect(), src, tmp_path / "o.parquet")
    assert set(stats["violations"]["reference_unit"]) == {"calcium_total [mmol/L]", "magnesium [mEq/L]"}


def test_strict_harmonization_fails_on_lab_unit_errors(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [("H", T(1, 0), "Ca", "calcium_total", "bmp", "mmol/L", 2.2)]
    df = pd.read_parquet(_labs(tmp_path / "l.parquet", rows))
    for col in ("lab_order_dttm", "lab_collect_dttm"):
        df[col] = df["lab_result_dttm"]
    df["lab_value"] = "2.2"
    df["lab_specimen_category"] = "plasma_blood"
    df.to_parquet(raw / "clif_labs.parquet")
    with pytest.raises(ValueError, match="calcium_total"):
        harmonize_dir(raw, tmp_path / "out", temp_dir=tmp_path / "duck")


def test_table_rules_drop_duplicates_and_rescale_fractions(tmp_path: Path) -> None:
    src = _labs(tmp_path / "l.parquet", [
        ("H", T(1, 0), "9992: Act.HCO3 (bloed)", "bicarbonate", "blood_gas", "mmol/L", 24.0),
        ("H", T(1, 0), "9992: Act.HCO3 (bloed)", "carbon_dioxide", "bmp", "mmol/L", 24.0),   # duplicate copy
        ("H", T(1, 0), "12311: O2-Saturatie (bloed)", "so2_arterial", "blood_gas", "%", 0.98),
        ("H", T(2, 0), "SaO2", "so2_arterial", "blood_gas", "%", 97.0),
        ("H", T(1, 0), "6789: Protrombinetijd", "pt", "coags", "sec", 1.3),
    ])
    counts = _apply_fixes(connect(), src, tmp_path / "o.parquet", value_fixes()["labs"])
    out = pd.read_parquet(tmp_path / "o.parquet")
    assert "carbon_dioxide" not in set(out["lab_category"])
    assert sorted(out.loc[out.lab_category == "so2_arterial", "lab_value_numeric"]) == [97.0, 98.0]
    inr = out[out.lab_name.str.startswith("6789")]
    assert inr["lab_category"].tolist() == ["inr"] and inr["reference_unit"].tolist() == ["(no units)"]
    assert counts["_drop_rows[0]"] == 1 and counts["_scale[lab_value_numeric][0]"] == 1


def test_aumc_cotrimoxazole_combined_doses_become_the_trimethoprim_component(tmp_path: Path) -> None:
    """AmsterdamUMCdb records co-trimoxazole as the combined dose (IV item 8394, enteral forte item 9052)."""
    rows = [("8394: Co-Trimoxazol (Bactrimel)", 960.0), ("9052: Co-trimoxazol forte (Bactrimel)", 960.0),
            ("Bactrim forte Lacktbl", 160.0)]
    src = tmp_path / "m.parquet"
    pd.DataFrame({"hospitalization_id": "H", "admin_dttm": T(1, 0), "med_name": [n for n, _ in rows],
                  "med_category": "trimethoprim_sulfamethoxazole", "med_dose": [d for _, d in rows],
                  "med_dose_unit": "mg"}).to_parquet(src)
    _apply_fixes(connect(), src, tmp_path / "o.parquet", value_fixes()["medication_admin_intermittent"])
    out = pd.read_parquet(tmp_path / "o.parquet").set_index("med_name")["med_dose"]
    assert out.to_dict() == {n: pytest.approx(160.0) for n, _ in rows}


def test_every_canonical_lab_category_is_a_clif_or_declared_category() -> None:
    allowed = {c for col in table_schema("labs")["columns"] if col["name"] == "lab_category"
               for c in col["permissible_values"]} | set(_lab_reference_units())
    assert set(lab_conventions()["order_category"]) <= allowed
    orders = {c for col in table_schema("labs")["columns"] if col["name"] == "lab_order_category"
              for c in col["permissible_values"]}
    assert set(lab_conventions()["order_category"].values()) <= orders


@pytest.mark.parametrize("dataset", ["eicu", "inspire"])
def test_adapter_lab_maps_follow_the_conventions(dataset: str) -> None:
    """Our adapters emit canonical values directly (S2b would rewrite them anyway; this keeps maps honest)."""
    text = resources.files("icutwin.clif").joinpath("maps", dataset, "labs.csv").read_text(encoding="utf-8")
    ref, order = _lab_reference_units(), lab_conventions()["order_category"]
    keep = lab_conventions()["keep_order_categories"]
    for row in csv.DictReader(text.splitlines()):
        cat = row["lab_category"]
        assert row["reference_unit"] == ref[cat], row
        assert row["lab_order_category"] in [order[cat], *keep.get(cat, [])], row


def _vitals(path: Path, rows: list[tuple]) -> Path:
    cols = ["hospitalization_id", "recorded_dttm", "vital_name", "vital_category", "vital_value", "meas_site_name"]
    pq.write_table(pa.table({c: [r[i] for r in rows] for i, c in enumerate(cols)}), path)
    return path


def test_bp_site_is_resolved_across_etl_conventions_and_kept_apart(tmp_path: Path) -> None:
    src = _vitals(tmp_path / "v.parquet", [
        ("H", T(10, 5), "Arterial Blood Pressure systolic", "sbp", 130.0, None),     # CLIF-MIMIC
        ("H", T(10, 10), "Non Invasive Blood Pressure systolic", "sbp", 110.0, None),
        ("H", T(10, 20), "ART BP Systolic", "sbp", 134.0, None),
        ("H", T(10, 30), "6641: ABP systolisch", "sbp", 132.0, "arterial_line_2"),    # CLIF-AmsterdamUMCdb
        ("H", T(10, 40), "Temperature Fahrenheit", "temp_c", 37.0, "Oral"),
        ("H", T(10, 50), "Temperature Celsius", "temp_c", 38.0, "Blood"),           # other sites merge
    ])
    con = connect()
    stats = canonical_vitals(con, src, tmp_path / "c.parquet")
    assert stats["bp_sites"] == {"arterial": 3, "cuff": 1}
    hourly_vitals(con, tmp_path / "c.parquet", tmp_path / "o.parquet")
    out = pd.read_parquet(tmp_path / "o.parquet")
    sbp = out[out.vital_category == "sbp"].set_index("meas_site_name")["vital_value"].to_dict()
    assert sbp == {"arterial": 132.0, "cuff": 110.0}
    temp = out[out.vital_category == "temp_c"]
    assert len(temp) == 1 and temp["vital_value"].iloc[0] == pytest.approx(37.5)
    assert temp["meas_site_name"].iloc[0] == "Blood" and temp["recorded_dttm"].iloc[0].minute == 50


def test_respiratory_rows_without_device_do_not_split_buckets(tmp_path: Path) -> None:
    rows = [  # settings charted separately from the device (eICU, CLIF-MIMIC): one bucket per hour and device
        ("H", T(10, 0), "imv", 0.5), ("H", T(10, 20), None, 0.45), ("H", T(10, 40), "imv", None),
        ("H", T(10, 50), None, 0.4), ("H", T(11, 10), "hfnc", 0.6), ("H", T(11, 30), None, 0.5),
    ]
    src = tmp_path / "rs.parquet"
    pq.write_table(pa.table({
        "hospitalization_id": [r[0] for r in rows], "recorded_dttm": [r[1] for r in rows],
        "device_category": [r[2] for r in rows], "mode_category": [None] * len(rows),
        "fio2_set": [r[3] for r in rows],
    }), src)
    hourly_respiratory(connect(), src, tmp_path / "o.parquet")
    out = pd.read_parquet(tmp_path / "o.parquet").sort_values("recorded_dttm")
    assert out["device_category"].tolist() == ["imv", "hfnc"]
    assert out["fio2_set"].tolist() == [pytest.approx(0.4), pytest.approx(0.5)]
    assert [ts.minute for ts in out["recorded_dttm"]] == [50, 30]
