"""AmsterdamUMCdb supplement on a hand-built fixture (AmsterdamUMCdb 1.0.2 layout, no real patients)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from icutwin.clif.harmonize import harmonize_dir
from icutwin.clif.supplements.aumc import AumcSupplement

H = 3_600_000  # one hour in AmsterdamUMCdb milliseconds
E2010 = datetime(2010, 1, 1, tzinfo=UTC)


def _adm(pid: int, aid: int, loc: str, urgency: int, specialty: str, yeargroup: str = "2010-2016",
         dest: str | None = None) -> dict:
    return {"patientid": pid, "admissionid": aid, "admissioncount": 1, "location": loc, "urgency": urgency,
            "origin": "Verpleegafdeling zelfde ziekenhuis", "admittedat": 0, "admissionyeargroup": yeargroup,
            "dischargedat": 36 * H, "lengthofstay": 36, "destination": dest, "gender": "Man", "agegroup": "60-69",
            "weightgroup": "70-79", "heightgroup": "170-179", "specialty": specialty}


ITEMS = {13116: "D_Thoraxchirurgie_CABG en Klepchirurgie", 16671: "DMC_Thoraxchirurgie_CABG en Klepchirurgie",
         18671: "NICE APACHEIV diagnosen", 15808: "Opname Sepsis"}


def _list(aid: int, itemid: int, value: str, valueid: int = 1, updated: int = 0, at: int = 0) -> dict:
    return {"admissionid": aid, "itemid": itemid, "item": ITEMS.get(itemid, str(itemid)), "valueid": valueid,
            "value": value, "measuredat": at, "registeredat": at, "updatedat": updated, "islabresult": 0}


def _num(aid: int, itemid: int, item: str, value: float, unit: str, at: int, tag: str | None = None) -> dict:
    return {"admissionid": aid, "itemid": itemid, "item": item, "tag": tag, "value": value, "unit": unit,
            "measuredat": at, "registeredat": at, "islabresult": 1, "fluidout": None}


def _drug(aid: int, cat: str, item: str, start: int, stop: int | None, *, itemid: int = 1, dose: float | None = None,
          doseunit: str | None = None, per: str | None = None, perkg: int = 0, rate: float | None = None,
          action: str = "Nieuwe spuit", fluidin: float | None = None, cont: int = 1, ocid: int | None = None,
          administered: float | None = None) -> dict:
    return {"admissionid": aid, "orderid": 1, "ordercategoryid": ocid, "ordercategory": cat, "itemid": itemid,
            "item": item, "rate": rate,
            "rateunit": "ml/uur" if rate is not None else None, "doserateperkg": perkg, "dose": dose,
            "doseunit": doseunit, "doserateunit": per, "action": action, "start": start, "stop": stop,
            "fluidin": fluidin, "iscontinuous": cont, "administered": administered,
            "administeredunit": doseunit if administered is not None else None}


@pytest.fixture(scope="module")
def aumc(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("aumc_sup")
    src = d / "src"
    src.mkdir()
    pd.DataFrame([
        _adm(1, 1, "IC", 0, "Cardiochirurgie", dest="Overleden"),
        _adm(2, 2, "MC", 1, "Cardiologie"),                      # medium care only: not an ICU stay
        _adm(3, 3, "IC&MC", 1, "Cardiologie"),                   # medical specialty, surgical diagnosis
        _adm(4, 4, "IC", 1, "Cardiologie", yeargroup="2003-2009"),
    ]).to_parquet(src / "admissions.parquet")
    pd.DataFrame([
        # admission 1: the NICE APACHE IV entry 'group - diagnosis' (priority 6) outranks an older-source group
        _list(1, 18671, "Post-operatief cardiovasculair - CABG", valueid=300),
        _list(1, 13110, "Niet-operatief", updated=5 * H),
        _list(1, 13111, "Cardiochirurgie"),
        # admission 3: source priority wins over the later update; an entry made after the first 24 h does not
        # count, though its source (NICE APACHE IV) would outrank (the header is admission documentation)
        _list(3, 13116, "Klepchirurgie", updated=0), _list(3, 16671, "CABG", updated=9 * H),
        _list(3, 18671, "Niet-operatief cardiovasculair - Hartfalen", valueid=10, updated=30 * H, at=30 * H),
        # specimen of the blood gases at 1 h
        _list(1, 7432, "Arterieel", at=2 * H),
        # sepsis at admission: a medical admission is in the sepsis group (the last entry of the first 24 h
        # counts; a revision on day 2 does not)
        _list(4, 15808, "Nee", valueid=2, updated=0), _list(4, 15808, "Ja", valueid=1, updated=H),
        _list(4, 15808, "Nee", valueid=2, updated=30 * H),
    ]).to_parquet(src / "listitems.parquet")
    pd.DataFrame([{"admissionid": 1, "itemid": 11646, "item": "Afname (bloed)", "value": "Ven.", "comment": None,
                   "measuredat": H, "registeredat": H, "updatedat": H, "islabresult": 1}]
                 ).to_parquet(src / "freetextitems.parquet")
    pd.DataFrame([
        _num(1, 9990, "pCO2 (bloed)", 50.0, "mmHg", H),                # venous specimen -> pco2_venous
        _num(1, 9996, "PO2 (bloed)", 40.0, "mmHg", H),                 # venous pO2: no CLIF category, dropped
        _num(1, 9996, "PO2 (bloed)", 90.0, "mmHg", 3 * H),             # no specimen -> arterial
        _num(1, 9994, "B.E. (bloed)", 3.0, "mmol/l", H, tag="-"),      # sign in tag
        _num(1, 8835, "Intracraniele druk", 12.0, "mmHg", H),
        _num(1, 6655, "CVD", 8.0, "mmHg", H),
    ]).to_parquet(src / "numericitems.parquet")
    pump, crys = "2. Spuitpompen", "Infuus - Crystalloid"
    pd.DataFrame([
        # norepinephrine: overlapping syringes are summed
        _drug(1, pump, "Noradrenaline (Levophed)", 0, 2 * H, dose=0.1, doseunit="µg", per="min", perkg=1, rate=5.0),
        _drug(1, pump, "Noradrenaline (Levophed)", H, 3 * H, dose=0.05, doseunit="µg", per="min", perkg=1, rate=2.5),
        _drug(1, pump, "Noradrenaline (Levophed)", H, H + 60_000, dose=1.0, doseunit="µg", per="min", perkg=1,
              rate=60.0, action="Flush"),
        # insulin: back-to-back syringes at the same rate are one infusion
        _drug(1, pump, "Insuline", 0, H, dose=2.0, doseunit="IE", per="uur", rate=2.0),
        _drug(1, pump, "Insuline", H, 2 * H, dose=2.0, doseunit="IE", per="uur", rate=2.0),
        _drug(1, crys, "NaCl 0,9%", 0, H, rate=100.0, action="Nieuwe zak"),              # volume rate only
        _drug(1, crys, "Drukzak", 0, 10 * H, rate=3.0, action="Nieuwe zak"),             # arterial-line pressure bag
        _drug(1, "Infuus - Bloedproducten", "Erytrocytenconcentraat", 4 * H, 6 * H, itemid=7, rate=150.0,
              fluidin=300.0),
        _drug(1, "Voeding Enteraal ", "Nutrison", 5 * H, None, itemid=8, fluidin=500.0, cont=0),
        _drug(1, "Infuus - Colloid", "Albumine 20%", 5 * H, 6 * H, itemid=9, fluidin=0.0),
        # boluses: an item the tool does not map (added), one it maps (not re-mapped), an inhaled drug
        _drug(1, "Injecties Tractus Digestivus", "Pantoprazol (Pantozol)", 7 * H, 7 * H, itemid=12, dose=40.0,
              doseunit="mg", action="Nieuwe toediening", cont=0, ocid=26),
        _drug(1, "Injecties CZS/Sedatie/Analgetica", "Paracetamol", 7 * H, 7 * H, itemid=6891, dose=1000.0,
              doseunit="mg", action="Nieuwe toediening", cont=0, ocid=23),
        _drug(1, "Niet iv Tractus Respiratorius", "Salbutamol (Ventolin)", 7 * H, 7 * H, itemid=13, dose=2.0,
              doseunit="Puff", action="Nieuwe toediening", cont=0, ocid=70),
        # a dose given at a dose rate: 3.2 mmol per hour for 6.25 h, 20 mmol administered
        _drug(1, "Injecties Tractus Digestivus", "Natrium-kalium-fosfaat", 8 * H, 8 * H + 375 * 60_000, itemid=8999,
              dose=3.2, doseunit="mmol", per="uur", rate=8.0, action="Nieuwe toediening", cont=0, ocid=26,
              administered=20.0),
    ]).to_parquet(src / "drugitems.parquet")
    pd.DataFrame([{"admissionid": 1, "orderid": o, "ordercategoryid": 1, "ordercategoryname": "Kweken", "itemid": i,
                   "item": item, "registeredat": at, "registeredby": "x"}
                  for o, i, item, at in ((101, 9189, "Bloedkweken afnemen", 2 * H), (102, 8097, "Sputumkweek afnemen", 2 * H),
                                         (103, 19664, "Research Sputumkweek COUrSe", 2 * H),   # study sample: not care
                                         (104, 9224, "PT/CT afnemen", 2 * H))]                 # not a culture
                 ).to_parquet(src / "procedureorderitems.parquet")
    out = d / "out"
    out.mkdir()
    (out / "_provenance.json").write_text(json.dumps({"tool": "CLIF-AmsterdamUMCdb", "finished_at": "t1"}))
    AumcSupplement(src, out, d / "duck").run()
    return out


def _read(out: Path, table: str) -> pd.DataFrame:
    return pd.read_parquet(out / f"clif_{table}+icutwin.parquet")


def _hours(ts: pd.Series, epoch: datetime = E2010) -> list[float]:
    return [(t - epoch) / timedelta(hours=1) for t in ts]


def test_icu_stays_and_reason_for_admission(aumc: Path) -> None:
    s = _read(aumc, "ext_icu_stay").set_index("hospitalization_id")
    assert sorted(s.index) == ["1", "3", "4"]
    assert s.loc["1", "adm_dx_name"] == "Post-operatief cardiovasculair | Cardiochirurgie | CABG"
    assert s.loc["3", "adm_dx_name"] == "Klepchirurgie"
    assert pd.isna(s.loc["4", "adm_dx_name"])
    assert s["surgical_status"].to_dict() == {"1": "elective_surgery", "3": "emergency_surgery", "4": "nonsurgical"}
    # NICE group 'Post-operatief cardiovasculair'; the CABG/valve item; sepsis at admission without a diagnosis
    assert s["adm_dx_group"].to_dict() == {"1": "surgical_cardiovascular", "3": "surgical_cardiovascular",
                                           "4": "medical_sepsis"}
    assert json.loads(s.loc["1", "surgery_context"]) == {"specialty": "Cardiochirurgie"}
    assert s["icu_outcome"].to_dict() == {"1": "dead", "3": "alive", "4": "alive"}
    assert s.loc["1", "admission_source"] == "ward"
    assert s.loc["1", "icu_in_dttm"] == pd.Timestamp(E2010)
    assert s.loc["4", "icu_in_dttm"] == pd.Timestamp(2003, 1, 1, tz="UTC")   # epoch of the 2003-2009 group


def test_blood_gases_are_routed_by_specimen(aumc: Path) -> None:
    labs = _read(aumc, "labs")
    got = sorted(zip(labs["lab_category"], _hours(labs["lab_collect_dttm"]), labs["lab_value_numeric"]))
    assert got == [("ext_base_excess", 1.0, -3.0), ("pco2_venous", 1.0, 50.0), ("po2_arterial", 3.0, 90.0)]


def test_vitals_and_hemodynamics(aumc: Path) -> None:
    assert _read(aumc, "vitals")[["vital_category", "vital_value"]].values.tolist() == [["intracranial_pressure", 12.0]]
    hd = _read(aumc, "invasive_hemodynamics")
    assert hd[["measurement_category", "measurement_value"]].values.tolist() == [["cvp", 8.0]]


def test_continuous_rates_are_summed_into_start_change_stop(aumc: Path) -> None:
    mc = _read(aumc, "medication_admin_continuous").sort_values(["med_category", "admin_dttm"])
    got = {cat: [(h, a, round(dose, 6), u) for h, a, dose, u in
                 zip(_hours(g["admin_dttm"]), g["mar_action_category"], g["med_dose"], g["med_dose_unit"])]
           for cat, g in mc.groupby("med_category")}
    assert got == {
        "norepinephrine": [(0.0, "start", 0.1, "mcg/kg/min"), (1.0, "dose_change", 0.15, "mcg/kg/min"),
                           (2.0, "dose_change", 0.05, "mcg/kg/min"), (3.0, "stop", 0.0, "mcg/kg/min")],
        "insulin": [(0.0, "start", 2.0, "u/hr"), (2.0, "stop", 0.0, "u/hr")],
        "sodium_chloride": [(0.0, "start", 100.0, "ml/hr"), (1.0, "stop", 0.0, "ml/hr")],
    }
    assert set(mc["med_route_category"]) == {"iv"}


def test_input_volumes(aumc: Path) -> None:
    inp = _read(aumc, "input").sort_values("recorded_dttm")
    assert list(zip(_hours(inp["recorded_dttm"]), inp["input_category"], inp["input_volume"])) == [
        (5.0, "enteral_nutrition", 500.0), (6.0, "blood_products", 300.0)]
    assert inp["input_name"].tolist() == ["8: Nutrison", "7: Erytrocytenconcentraat"]


def test_harmonize_replaces_tool_continuous_table(aumc: Path, tmp_path: Path) -> None:
    raw = tmp_path / "aumc"   # the dataset's raw CLIF directory (its name selects the dataset's route rules)
    raw.mkdir()
    for f in aumc.iterdir():
        (raw / f.name).write_bytes(f.read_bytes())
    # the tool's continuous rows (dose without its time unit) are retracted in favour of the supplement's
    pd.DataFrame({"hospitalization_id": ["1"], "admin_dttm": [datetime(2010, 1, 1)], "med_name": ["Noradrenaline"],
                  "med_category": ["norepinephrine"], "med_route_category": ["iv"], "med_dose": [0.1],
                  "med_dose_unit": ["mcg"], "mar_action_category": ["start"]}
                 ).to_parquet(raw / "clif_medication_admin_continuous.parquet")
    report = harmonize_dir(raw, tmp_path / "out", tmp_path / "duck")
    mc = pd.read_parquet(tmp_path / "out" / "clif_medication_admin_continuous.parquet")
    assert len(mc) == 8 and "mcg" not in set(mc["med_dose_unit"])
    mi = pd.read_parquet(tmp_path / "out" / "clif_medication_admin_intermittent.parquet")
    assert set(mi["med_route_basis"].dropna()) == {"reported_class"}   # order categories, not explicit routes
    assert report["tables"]["medication_admin_continuous"]["retracted_from_tool"] == {"retract[0]": 1}
    # group columns derived from their categories (CLIF 3.0 mappings)
    ne = mc[mc.med_category == "norepinephrine"].sort_values("admin_dttm")
    assert set(ne["med_group"]) == {"vasoactives"}
    assert ne["mar_action_group"].tolist() == ["administered"] * 3 + ["not_administered"]


def test_harmonize_drops_the_tool_bicarbonate_copy_filed_as_total_co2(aumc: Path, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for f in aumc.iterdir():
        (raw / f.name).write_bytes(f.read_bytes())
    # CLIF-AmsterdamUMCdb maps 9992 Act.HCO3 (bloed) to bicarbonate and again to carbon_dioxide
    lab = {"hospitalization_id": "1", "lab_collect_dttm": datetime(2010, 1, 1, 2), "lab_name": "9992: Act.HCO3 (bloed)",
           "lab_value_numeric": 24.0, "reference_unit": "mmol/L"}
    pd.DataFrame([{**lab, "lab_category": "bicarbonate", "lab_order_category": "blood_gas"},
                  {**lab, "lab_category": "carbon_dioxide", "lab_order_category": "bmp"},
                  {**lab, "lab_name": "9945: Natrium (bloed)", "lab_category": "carbon_dioxide",
                   "lab_order_category": "bmp"}]).to_parquet(raw / "clif_labs.parquet")
    report = harmonize_dir(raw, tmp_path / "out", tmp_path / "duck", strict=False)
    labs = pd.read_parquet(tmp_path / "out" / "clif_labs.parquet")
    hco3 = labs[labs.lab_name.str.startswith("9992: ")]
    assert hco3["lab_category"].tolist() == ["bicarbonate"]
    assert report["tables"]["labs"]["retracted_from_tool"]["retract[1]"] == 1   # only the 9992 copy


def test_text_repairs_map_lossy_tokens_back(aumc: Path) -> None:
    report = json.loads((aumc / "_supplement_report.json").read_text(encoding="utf-8"))
    assert report["text_repairs"]["�g"] == "µg"   # Latin-1 'µ' read as lossy UTF-8


def test_lossy_text_is_repaired_token_by_token(tmp_path: Path) -> None:
    from icutwin.clif.conform import connect
    from icutwin.clif.harmonize import repair_lossy_text

    src = tmp_path / "pa.parquet"
    pd.DataFrame({"assessment_name": ["6735: Ge�ntubeerd", "6735: Wakker", "X�Y"], "v": [1, 2, 3]}).to_parquet(src)
    report = repair_lossy_text(connect(), src, tmp_path / "o.parquet", {"Ge�ntubeerd": "Geïntubeerd"})
    assert pd.read_parquet(tmp_path / "o.parquet")["assessment_name"].tolist() == [
        "6735: Geïntubeerd", "6735: Wakker", "X�Y"]
    assert report["unresolved"] == {"assessment_name": ["X�Y"]}


def test_intermittent_doses_the_tool_does_not_map(aumc: Path) -> None:
    i = _read(aumc, "medication_admin_intermittent").sort_values("med_category")
    # paracetamol is the tool's item; inhaled salbutamol takes the local route and category extensions (M4), its
    # dose in puffs (a counting unit) keeps its count, in the other sources' spelling
    assert i[["med_category", "med_route_category", "med_dose_unit"]].values.tolist() == [
        ["albuterol", "inhaled", "puff"], ["pantoprazole", "iv", "mg"], ["phosphate_repletion", "iv", "mmol"]]
    # a dose given at a dose rate is the amount administered, not the rate per hour
    assert i["med_dose"].tolist() == [2.0, 40.0, 20.0]


def test_tool_item_list_matches_the_pinned_clif_aumc_mapping() -> None:
    import csv
    from importlib import resources

    mapping = Path(__file__).resolve().parents[1] / "refs/CLIF-AmsterdamUMCdb/mappings/medication_admin_intermittent.yaml"
    if not mapping.exists():
        pytest.skip("CLIF-AmsterdamUMCdb is not cloned in refs/")
    from icutwin.clif.tools.aumc_tool_items import tool_items
    text = resources.files("icutwin.clif").joinpath("maps", "aumc", "tool_intermittent_items.csv").read_text(encoding="utf-8")
    assert {r["itemid"] for r in csv.DictReader(text.splitlines())} == {r["itemid"] for r in tool_items(mapping)}


def test_culture_orders_are_specimens_without_results(aumc: Path) -> None:
    c = _read(aumc, "microbiology_culture").sort_values("order_id")
    assert list(c["fluid_category"]) == ["blood_buffy", "respiratory_tract_lower"]
    assert list(c["patient_id"]) == ["1", "1"] and list(c["method_category"]) == ["culture", "culture"]
    assert _hours(c["collect_dttm"]) == [2.0, 2.0] and c["result_dttm"].isna().all() and c["organism_category"].isna().all()
