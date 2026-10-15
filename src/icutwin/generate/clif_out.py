"""Stays -> canonical CLIF tables (plan S7: detok -> MEDS -> CLIF), so the derived concepts (``meds.derived.derive``:
treatment episodes, KDIGO, norepinephrine equivalents, fluid balance, the 24-hour SOFA) of twins come from the same
definitions as those of real stays, read the same way.

A stay (``tokenization.stays.Stay``: a real stay read from MEDS, or a decoded twin) becomes one patient with one
hospitalization and one ICU stay, all keyed by its id. Its timeline events invert the ELF configs that made them
(CLIF_MEDS's domains, pinned in ``refs/``, our extension domains, the qualified codes of ``meds/qualifiers.yaml``): a
code's parts give the template's columns ('UNK' is NULL; a qualified code's extra last part is its qualifier column),
the value the template's numeric column, the time its time column. In a table of several concepts (the wide ones:
respiratory_support, crrt_therapy, intermittent_dialysis) one stay's events at one time make one row, unless an event
would overwrite a column; in the others every event is a row. The ICU window runs from the ICU admission marker to the
ICU discharge event (a twin cut at its budget: to its last event); the hospital admission precedes it by the header's
pre-ICU stay (else it is the first event), and the hospitalization's record ends with the ICU discharge (a stay holds
nothing later; clifpy's ASE counts antibiotic days up to the hospital discharge). The GCS total, left to the decoder
by the tokens (spec ``derived_not_tokenized``), is the sum of the three components charted together, where no total
was charted. A group column (``output_group``, ``med_group``, ...) follows from its category (``clif_groups.yaml``).
A continuous infusion without a stop in the record ends at its last event, as ``derive`` reads it: a stop past the
MEDS window is lost (INSPIRE subset: 7 stays' vasopressor episodes end before the ICU), but closing every open
infusion at the record's end would be worse, since sources leave infusions without a stop (36 INSPIRE stays would
gain one).

What MEDS does not carry, these tables cannot: a lab's one time is its result time (CLIF_MEDS's LAB config), which also
stands for its collection time (KDIGO reads creatinine at collection); free-text names are absent (twins have none,
so real stays drop theirs too); events before the MEDS window (48 h before ICU admission) are absent (KDIGO's 7-day
creatinine baseline, an episode under way at the window's start). ``tools/clif_roundtrip.py`` measures these
differences on real stays (canonical CLIF against MEDS -> CLIF, and against MEDS -> tokens -> CLIF).
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, timedelta
from functools import lru_cache
from importlib import resources
from pathlib import Path

import pyarrow as pa
import yaml

from icutwin.clif.conform import ClifWriter, connect
from icutwin.meds.build import SKIP_CLIF_MEDS, extension_configs, qualifiers
from icutwin.meds.elf import config_table, load_config
from icutwin.stage.spec import stay_dataset
from icutwin.tokenization.stays import Stay

MARKER = "ICU//admission"                 # the timeline's ICU admission marker (spec timeline.marker.token)
DISCHARGE = "ICU//discharge//"
HOSP_OUTCOME = "HOSP//discharge_category//"
GCS_PARTS = ("gcs_eye", "gcs_motor", "gcs_verbal")
STAY_TABLES = ("patient", "hospitalization", "ext_icu_stay", "adt")   # from the stay itself, not its events
# event tables written: what meds.derived reads (derive, and clifpy's SOFA and ASE); the others (positions, lines,
# susceptibilities keyed by their culture, ...) are counted, not written
EVENT_TABLES = ("vitals", "labs", "respiratory_support", "medication_admin_continuous",
                "medication_admin_intermittent", "crrt_therapy", "intermittent_dialysis", "input", "output",
                "patient_assessments", "microbiology_culture")
# columns CLIF_MEDS creates while converting (its medication unit conversion) -> the CLIF columns they stand for
RENAME = {"med_dose_converted": "med_dose", "med_dose_unit_converted": "med_dose_unit"}
ALSO = {"lab_result_dttm": "lab_collect_dttm"}    # a time column that also stands for another (module docstring)


def refs_dir() -> Path:
    return Path(os.environ.get("ICUTWIN_REFS") or Path(__file__).resolve().parents[3] / "refs")


@lru_cache(maxsize=1)
def groups() -> list[dict]:
    """``clif_groups.yaml``: per table, a group column as a function of its category column (codes carry the
    category only; KDIGO reads urine by ``output_group``, clifpy's ASE antibiotics by ``med_group``)."""
    text = resources.files("icutwin.generate").joinpath("clif_groups.yaml").read_text(encoding="utf-8")
    return list((yaml.safe_load(text) or {}).values())


@dataclass(frozen=True)
class Template:
    table: str
    parts: tuple              # per code part: a literal, or ("col", CLIF column)
    qualifier: str | None     # the column of a qualified code's extra last part
    time: str | None          # the time column (None: a static event)
    numeric: str | None       # the numeric value's column
    wide: bool                # the table has several concepts (rows merge per time)


def _parts(code) -> tuple:
    out = []
    for p in code if isinstance(code, list) else [code]:
        if not isinstance(p, str):
            raise ValueError(f"unsupported code part {p!r}")
        if p.startswith("col(") and p.endswith(")"):
            out.append(("col", RENAME.get(p[4:-1], p[4:-1])))
        else:
            out.extend(p.split("//"))      # a literal can hold several parts ('RESP//fio2_set')
    return tuple(out)


def _col(spec) -> str | None:
    return spec[4:-1] if isinstance(spec, str) and spec.startswith("col(") and spec.endswith(")") else None


@lru_cache(maxsize=1)
def templates() -> dict[str, list[Template]]:
    """A code's first part -> the templates that can make it: CLIF_MEDS's configs (pinned in ``refs/``; the build's
    overlay changes preferred units, not codes) and our extension domains, with the qualifier column of the
    qualified domains."""
    cfgs = {p.stem: load_config(p) for p in sorted((refs_dir() / "CLIF_MEDS" / "config").glob("*.yaml"))
            if p.stem not in SKIP_CLIF_MEDS}
    if not cfgs:
        raise FileNotFoundError(f"no CLIF_MEDS configs under {refs_dir()}")
    cfgs.update(extension_configs())
    out: dict[str, list[Template]] = {}
    for domain, cfg in cfgs.items():
        table, concepts = config_table(cfg)
        level = (qualifiers().get(domain) or {}).get("level")
        for m in concepts.values():
            parts = _parts(m["code"])
            num = m.get("numeric_value")
            out.setdefault(parts[0], []).append(Template(table, parts, level, _col(m.get("time")),
                                                         RENAME.get(num, num), len(concepts) > 1))
    return out


@lru_cache(maxsize=None)
def match(code: str) -> tuple[Template, tuple] | None:
    """The template a code instantiates (the one with the most literal parts) and its (column, value) pairs."""
    parts = code.split("//")
    best = None
    for tpl in templates().get(parts[0], ()):
        n = len(tpl.parts)
        if not (len(parts) == n or (len(parts) == n + 1 and tpl.qualifier)):
            continue
        vals = []
        for p, want in zip(parts, tpl.parts, strict=False):
            if isinstance(want, tuple):
                vals.append((want[1], None if p == "UNK" else p))
            elif p != want:
                break
        else:
            if len(parts) == n + 1:
                vals.append((tpl.qualifier, None if parts[-1] == "UNK" else parts[-1]))
            lit = sum(not isinstance(w, tuple) for w in tpl.parts)
            if best is None or lit > best[2]:
                best = (tpl, tuple(vals), lit)
    return None if best is None else best[:2]


@dataclass
class _Table:
    key: str = "hospitalization_id"
    pos: dict = field(default_factory=dict)       # column -> position in a row; the key first
    rows: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.pos.setdefault(self.key, 0)

    def ix(self, col: str) -> int:
        if col not in self.pos:
            self.pos[col] = len(self.pos)
            for r in self.rows:
                r.append(None)
        return self.pos[col]

    def new(self, key: str) -> list:
        row = [None] * len(self.pos)
        row[0] = key
        self.rows.append(row)
        return row


def _known(x):
    """A prefix, header or outcome value, None where it is ELF's NULL ('UNK') or the spec's unknown."""
    return None if x in ("UNK", "unknown") else x


def window(s: Stay):
    """(ICU admission, ICU discharge, ICU outcome or None, hospital admission, cut): the stay's ICU window (a decoded
    stay's runs from its marker to its first ICU discharge, ``tokenization.decode``); cut: no ICU discharge event (a
    twin cut at its budget: its window ends at its last event)."""
    dis = next((c[len(DISCHARGE):] for t, _r, c, *_ in s.timeline if c.startswith(DISCHARGE) and t >= s.icu_in), None)
    first = min([s.icu_in, *(t for t, *_ in s.timeline)])
    pre = _known(s.header.get("pre_icu_los"))
    hosp_in = s.icu_in - timedelta(hours=float(pre)) if isinstance(pre, int | float) else first
    return s.icu_in, s.icu_out, _known(dis), min(hosp_in, first), dis is None


def _utc(t):
    return None if t is None else t.replace(tzinfo=UTC)


def _events(s: Stay, key: str, tables: dict[str, _Table], unmatched: Counter) -> None:
    wide_rows: dict[tuple, list] = {}
    pa_start = len(tables["patient_assessments"].rows) if "patient_assessments" in tables else 0
    for t, _r, code, num, _text in s.timeline:
        if code == MARKER:
            continue
        if "OOV" in code.split("//")[1:]:      # a tokenizer coarsening (``<domain>//OOV``): no CLIF category (#120)
            unmatched["OOV"] += 1
            continue
        hit = match(code)
        if hit is None:
            unmatched[code.split("//", 1)[0]] += 1
            continue
        tpl, vals = hit
        if tpl.table not in EVENT_TABLES or tpl.time is None:
            if tpl.table not in STAY_TABLES:
                unmatched[f"not written: {tpl.table}"] += 1
            continue
        tab = tables.setdefault(tpl.table, _Table())
        sets = [(tab.ix(c), v) for c, v in vals]
        if tpl.numeric:
            sets.append((tab.ix(tpl.numeric), num))
        it = tab.ix(tpl.time)
        ia = tab.ix(ALSO[tpl.time]) if tpl.time in ALSO else None
        row = wide_rows.get((tpl.table, t)) if tpl.wide else None
        if row is None or any(row[i] is not None and row[i] != v for i, v in sets) \
                or (tpl.numeric and row[tab.ix(tpl.numeric)] is not None):
            row = tab.new(key)
            row[it] = _utc(t)
            if ia is not None:
                row[ia] = row[it]
            if tpl.wide:
                wide_rows[(tpl.table, t)] = row
        for i, v in sets:
            row[i] = v
    pas = tables.get("patient_assessments")
    if pas is None:
        return
    ic, iv, it = pas.ix("assessment_category"), pas.ix("numerical_value"), pas.ix("recorded_dttm")
    at: dict = {}
    for row in pas.rows[pa_start:]:
        at.setdefault(row[it], {})[row[ic]] = row[iv]
    for t, cats in at.items():
        if "gcs_total" not in cats and all(cats.get(c) is not None for c in GCS_PARTS):
            row = pas.new(key)
            row[it], row[ic], row[iv] = t, "gcs_total", float(sum(cats[c] for c in GCS_PARTS))


def _stay(s: Stay, key: str, tables: dict[str, _Table]) -> bool:
    icu_in, icu_out, outcome, hosp_in, cut = window(s)
    pre, head = {k: _known(v) for k, v in s.prefix.items()}, {k: _known(v) for k, v in s.header.items()}
    hosp = _known(next((c[len(HOSP_OUTCOME):] for c in s.end if c.startswith(HOSP_OUTCOME)), None))
    rows = {
        "patient": {"sex_category": pre.get("sex"), "race_category": pre.get("race"),
                    "ethnicity_category": pre.get("ethnicity")},
        "hospitalization": {"patient_id": key, "admission_dttm": _utc(hosp_in), "discharge_dttm": _utc(icu_out),
                            "age_at_admission": None if pre.get("age") is None else int(round(float(pre["age"]))),
                            "admission_type_category": head.get("admission_type"), "discharge_category": hosp},
        "ext_icu_stay": {"stay_id": key, "patient_id": key, "dataset": stay_dataset(s.dataset or pre.get("dataset"), key),
                         "site_id": pre.get("site"), "icu_in_dttm": _utc(icu_in), "icu_out_dttm": _utc(icu_out),
                         "unit_type": head.get("unit_type"), "clock_valid": s.clock_valid,
                         "admission_source": head.get("admission_source"),
                         "surgical_status": head.get("surgical_status"), "adm_dx_group": head.get("adm_dx_group"),
                         "icu_outcome": outcome, "era": pre.get("era")},
        "adt": {"in_dttm": _utc(icu_in), "out_dttm": _utc(icu_out), "location_category": "icu"},
    }
    for table, vals in rows.items():
        tab = tables.setdefault(table, _Table("patient_id" if table == "patient" else "hospitalization_id"))
        row = tab.new(key)
        for c, v in vals.items():
            row[tab.ix(c)] = v
    return cut


def write(stays: list[Stay], keys: list[str], out_dir: Path) -> dict:
    """``clif_<table>.parquet`` of ``stays`` (each its own patient, hospitalization and ICU stay, keyed by ``keys``)
    in ``out_dir``, through ``ClifWriter`` (CLIF 3.0 types, permissible values checked). Returns the rows per table,
    the stays cut before an ICU discharge, and the events not written (per domain without a template, per table
    outside ``EVENT_TABLES``)."""
    tables: dict[str, _Table] = {}
    unmatched: Counter = Counter()
    cut = 0
    for s, key in zip(stays, keys, strict=True):
        cut += _stay(s, key, tables)
        _events(s, key, tables, unmatched)
    for g in groups():
        tab = tables.get(g["table"])
        if tab is None or g["category"] not in tab.pos:
            continue
        ic, ig = tab.pos[g["category"]], tab.ix(g["group"])
        for r in tab.rows:
            r[ig] = g["map"].get(r[ic]) if r[ig] is None else r[ig]
    con = connect()
    w = ClifWriter(con, Path(out_dir))
    for name, tab in sorted(tables.items()):
        if not tab.rows:
            continue
        data = pa.table({c: [r[i] for r in tab.rows] for c, i in tab.pos.items()})
        con.register("rows_in", data)
        w.write(name, "SELECT * FROM rows_in")
        con.unregister("rows_in")
    return {"rows": {k: v["rows"] for k, v in w.written.items()}, "stays": len(stays), "cut": cut,
            "unmatched": dict(unmatched)}


def derived(stays: list[Stay], work: Path, chunk: int = 1000) -> dict:
    """``tokenization.stays.load_derived`` of ``stays`` (keyed by their ``stay_id``), derived as real stays are: written
    as canonical CLIF and run through ``meds.derived.derive`` (every age kept), ``chunk`` stays at a time (each concept
    is its stay's own). Under ``work``: per chunk its derived files and ``_write.json``; the CLIF is removed after."""
    import json
    import shutil

    from icutwin.meds.derived import derive
    from icutwin.tokenization.stays import load_derived

    out: dict = {"episodes": {}, "kdigo": {}, "sofa_24h": {}}
    for a in range(0, len(stays), chunk):
        part, d = stays[a:a + chunk], Path(work) / f"chunk_{a // chunk:04d}"
        shutil.rmtree(d, ignore_errors=True)
        rep = write(part, [s.stay_id for s in part], d / "clif")
        derive(d / "clif", d / "derived", d / "work", min_age_years=None)
        (d / "_write.json").write_text(json.dumps(rep), encoding="utf-8")
        got = load_derived(d / "derived")
        for k in out:
            out[k].update(got[k])
        for sub in ("clif", "work"):
            shutil.rmtree(d / sub, ignore_errors=True)
    return out
