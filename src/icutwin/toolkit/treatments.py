"""The what-if registry (Paper P C1, amendment 2; F5): the treatments, their given records and withheld token sets, and
the pre-specified responses with their windows. The definitions are the paper scripts' (``tools/server/
paper_whatif.py``, ``tools/paper_c1.py``; a test checks they agree); the paper scripts stay as they ran.

* *give*: the dataset's most common start record among C1's held-out starters (``MODAL``, from C1's ``arms.json``) at
  the median rate bin (bin 4 of 0-9), and for norepinephrine and insulin also bins 1 and 8 (dose arms); written at the
  prompt's end after a 1-5 minute gap (``generate.engine.Intervention``);
* *withhold*: every token of the treatment's family, throughout the continuation;
* responses over (t, t + 6 h] (creatinine over 24 h): the mean of a channel's values, their sum (urine output, single
  records over 2,000 mL left out as source artefacts) or how often it is charted.
"""

from __future__ import annotations

import re

TREATMENTS = {   # name: (start-code regex, family-code regex or a treatment set of the engine, dose arms)
    "norepinephrine": (r"^MED_CON//norepinephrine//.+//start//iv$", r"^MED_CON//norepinephrine//", True),
    "insulin": (r"^MED_CON//insulin//.+//start//iv$", r"^MED_CON//insulin//", True),
    "furosemide": (r"^MED_(CON//furosemide//.+//start//iv|INT//furosemide//.+//given//iv)$",
                   r"^MED_(CON|INT)//furosemide//", False),
    "crrt": (r"^CRRT//crrt_mode_category//", "rrt", False),
    "imv": (r"^RESP//device_category//imv$", "imv", False),
    "fluid_bolus": (r"^MED_INT//(lactated_ringers_solution|sodium_chloride)//ml//given//iv$",
                    r"^MED_INT//(lactated_ringers_solution|sodium_chloride)//", False),
}
BINS = {"low": 1, "median": 4, "high": 8}

MAP = ("VITAL//map//arterial", "VITAL//map//cuff")
UO = ("OUTPUT//indwelling_urinary_catheter", "OUTPUT//external_urinary_catheter")
CAP = {"urine_output": 2000.0}   # mL per record: the real store's 99th percentile is 1.5 million (cumulative totals)
RESPONSES = {   # treatment: [(response, channels, window minutes, aggregate, expected sign)]
    "norepinephrine": [("map", MAP, 360, "mean", 1)],
    "insulin": [("glucose", ("LAB//glucose//mg/dl//poc", "LAB//glucose_serum//mg/dl//bmp"), 360, "mean", -1)],
    "crrt": [("potassium", ("LAB//potassium//mmol/l//bmp",), 360, "mean", -1),
             ("creatinine_24h", ("LAB//creatinine//mg/dl//bmp",), 1440, "mean", -1)],
    "furosemide": [("urine_output", UO, 360, "sum", 1)],
    "imv": [("fio2_charted", ("RESP//fio2_set",), 360, "count", 1), ("spo2", ("VITAL//spo2",), 360, "mean", 1)],
    "fluid_bolus": [("map", MAP, 360, "mean", 1), ("urine_output", UO, 360, "sum", 1)],
}

MODAL = {   # C1's given record per dataset (runs/paper/c1/arms.json "modal_code")
    "norepinephrine": {"mimic": "MED_CON//norepinephrine//mcg/kg/min//start//iv",
                       "aumc": "MED_CON//norepinephrine//mcg/kg/min//start//iv",
                       "eicu": "MED_CON//norepinephrine//ml/hr//start//iv",
                       "hirid": "MED_CON//norepinephrine//mcg/kg/min//start//iv",
                       "sicdb": "MED_CON//norepinephrine//mcg/kg/min//start//iv",
                       "nwicu": "MED_CON//norepinephrine//mcg/min//start//iv",
                       "inspire": "MED_CON//norepinephrine//UNK//start//iv"},
    "insulin": {d: "MED_CON//insulin//u/hr//start//iv" for d in ("mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu")},
    "furosemide": {"mimic": "MED_INT//furosemide//mg//given//iv", "aumc": "MED_INT//furosemide//mg//given//iv",
                   "eicu": "MED_CON//furosemide//ml/hr//start//iv", "hirid": "MED_INT//furosemide//mg//given//iv",
                   "sicdb": "MED_CON//furosemide//mg/hr//start//iv", "nwicu": "MED_INT//furosemide//mg//given//iv",
                   "inspire": "MED_INT//furosemide//UNK//given//iv"},
    "crrt": {"mimic": "CRRT//crrt_mode_category//cvvhdf", "aumc": "CRRT//crrt_mode_category//cvvh",
             "eicu": "CRRT//crrt_mode_category//cvvhd", "hirid": "CRRT//crrt_mode_category//UNK",
             "sicdb": "CRRT//crrt_mode_category//cvvhdf", "inspire": "CRRT//crrt_mode_category//UNK"},
    "imv": {d: "RESP//device_category//imv" for d in ("mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire")},
    "fluid_bolus": {d: "MED_INT//sodium_chloride//ml//given//iv" for d in ("mimic", "aumc", "nwicu")},
}


def family_tokens(v, name: str) -> tuple:
    """The withheld token set of a treatment (sorted ids)."""
    from icutwin.generate.engine import treatment_tokens

    fam = TREATMENTS[name][1]
    if fam in ("rrt", "imv"):
        return tuple(sorted(treatment_tokens(v, fam)))
    return tuple(i for i, m in enumerate(v.meta) if re.match(fam, m.get("code") or ""))


def modal_code(name: str, dataset: str | None) -> str:
    """The given record's code for a dataset (a prompt of no known dataset: the code most datasets give)."""
    by = MODAL[name]
    if dataset in by:
        return by[dataset]
    if dataset is not None:
        raise KeyError(f"{name}: no C1 starters in {dataset}, so no given record was defined there")
    codes = list(by.values())
    return max(sorted(set(codes)), key=codes.count)


def give_arms(v, name: str, dataset: str | None) -> dict:
    """arm -> the given token (``give_low`` / ``give_median`` / ``give_high``, as C1 built them)."""
    code = modal_code(name, dataset)
    cand = [i for i, m in enumerate(v.meta) if m.get("code") == code]
    if not cand:
        raise KeyError(f"{code} is not in the vocabulary")
    with_bin = {v.meta[i].get("bin"): i for i in cand if v.meta[i].get("kind") == "bin"}
    plain = [i for i in cand if v.meta[i].get("kind") == "na"] or cand
    if not with_bin:
        return {"give_median": int(plain[0])}
    distinct = len({with_bin.get(b) for b in BINS.values()} - {None}) == len(BINS)
    levels = BINS if TREATMENTS[name][2] and distinct else {"median": BINS["median"]}
    return {f"give_{k}": int(with_bin.get(b, next(iter(with_bin.values())))) for k, b in levels.items()}


def responses(name: str) -> list:
    return RESPONSES[name]


def all_responses() -> dict:
    """Every pre-specified response once: response -> (channels, window, aggregate, expected sign, treatment)."""
    out = {}
    for t, rs in RESPONSES.items():
        for resp, chans, win, agg, sign in rs:
            out.setdefault(resp, (chans, win, agg, sign, t))
    return out
