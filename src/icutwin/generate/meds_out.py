"""Decoded stays -> MEDS v0.4 rows, and the MEDS verification layer (plan S7 *Structured-output guarantee* 4b).

``rows`` inverts ``tokenization.stays.build_stay`` with the grammar spec's code templates:
* prefix fields: ``DATASET``, ``SITE`` and ``PATIENT`` codes static (time null); the era at the ICU admission; the age
  as ``HOSP//age_charted`` (its band's midpoint) at the hospital admission;
* header fields at the ICU admission (the unit type as the ``ICU//admission//<unit>`` event, in the timeline marker's
  place), the admission type at the hospital admission (``pre_icu_los`` before the ICU admission, its band's
  midpoint; the ICU admission without it), ``ICU//stay_seq`` = prior ICU stays + 1; the admission hour sets the ICU
  admission's clock time (decoded times start at ``decode.T0``);
* timeline events as decoded; derivable fields derived (``PA//gcs_total`` where eye, motor and verbal share a time);
* end-block codes (hospital discharge, death, billing diagnoses) at the last timeline time (the hospital discharge
  time is not generated).
Tokenizer coarsenings have no exact MEDS code: an out-of-vocabulary event is written ``<domain>//OOV`` (in the end
block ``HOSP_DX//OOV``, a diagnosis), a diagnosis or procedure without sub-tokens ``<domain>//<system>//``, a rare
eICU hospital ``SITE//eicu//other``; ``validate`` counts them apart from unknown codes.

``validate`` (layer b): the MEDS DataSchema (``meds.DataSchema.validate``), rows sorted by subject and time (static
rows first), finite numeric values, and every code either in the reference ``codes.parquet`` files, a coarsening,
or (counted, not failed) a new combination of known sub-tokens of a hierarchical system (ICD, procedures).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from icutwin.tokenization.spec import load_spec
from icutwin.tokenization.stays import Stay

HIERARCHICAL = ("HOSP_DX//", "PT_DX//", "PROC//")
END_OOV = "HOSP_DX//OOV"
GCS = ("PA//gcs_eye", "PA//gcs_motor", "PA//gcs_verbal")


def _fmt(template: str, value, dataset: str) -> str:
    return template.replace("{value}", str(value)).replace("{dataset}", dataset)


def rows(s: Stay, subject_id: int) -> list[tuple]:
    """The MEDS rows (subject_id, time, code, numeric_value, text_value) of a decoded stay, sorted."""
    spec = load_spec()
    marker = spec["timeline"]["marker"]["token"]
    ds = s.prefix.get("dataset")
    site = s.prefix.get("site")
    if ds is None and site is not None:      # the unconditional mode (dataset unknown): the site names its dataset
        ds = "eicu" if site == "other" else str(site).split("_")[0]
    ds = ds or "unknown"
    icu_in = next((t for t, _r, c, _n, _x in s.timeline if c == marker), None)
    if icu_in is None:
        icu_in = s.timeline[0][0] if s.timeline else datetime(2000, 1, 1)
    shift = timedelta(0)
    hour = s.header.get("admission_hour")
    if hour is not None:
        midnight = icu_in.replace(hour=0, minute=0, second=0, microsecond=0)
        shift = midnight + timedelta(hours=float(hour)) - icu_in
    icu_in += shift
    pre = s.header.get("pre_icu_los")
    hosp_in = icu_in - timedelta(hours=float(pre)) if pre is not None else icu_in
    out = []
    for f in spec["prefix"]:
        x = s.prefix.get(f["field"])
        if x is None:
            continue
        if "numeric" in f:
            out.append((subject_id, hosp_in, f["numeric"], float(x), None))
        elif f["code"].startswith(("ICU//", "HOSP//")):
            out.append((subject_id, icu_in, _fmt(f["code"], x, ds), None, None))
        else:
            out.append((subject_id, None, _fmt(f["code"], x, ds), None, None))
    for f in spec["header"]:
        x = s.header.get(f["field"])
        if "code" in f:
            if x is None and f["code"] != spec["timeline"]["marker"]["from_code"]:
                continue
            code = _fmt(f["code"], "UNK" if x is None else x, ds)
            out.append((subject_id, hosp_in if code.startswith("HOSP//") else icu_in, code, None, None))
        elif "numeric" in f and x is not None:
            out.append((subject_id, icu_in, f["numeric"], float(x) - f.get("offset", 0), None))
    last = icu_in
    by_time: dict = {}
    for t, _r, code, num, text in s.timeline:
        t = t + shift
        last = max(last, t)
        if code == marker:
            continue
        out.append((subject_id, t, code, None if num is None else float(num), text))
        if code in GCS and num is not None:
            by_time.setdefault(t, {})[code] = float(num)
    for t, g in by_time.items():
        if len(g) == len(GCS):
            out.append((subject_id, t, "PA//gcs_total", sum(g.values()), None))
    for code in s.end:          # an end-block OOV is a diagnosis without a token of its own (the open domain)
        out.append((subject_id, last, END_OOV if code == "END//OOV" else code, None, None))
    out.sort(key=lambda r: (r[1] is not None, r[1] or datetime.min, r[2]))
    return out


def table(stays: list[Stay], subject_ids: list[int]):
    """One MEDS table of several decoded stays (DataSchema dtypes)."""
    import meds
    import pyarrow as pa

    rs = [r for s, i in zip(stays, subject_ids, strict=True) for r in rows(s, i)]
    t = pa.table({"subject_id": pa.array([r[0] for r in rs], pa.int64()),
                  "time": pa.array([r[1] for r in rs], pa.timestamp("us")),
                  "code": pa.array([r[2] for r in rs], pa.string()),
                  "numeric_value": pa.array([r[3] for r in rs], pa.float32()),
                  "text_value": pa.array([r[4] for r in rs], pa.large_string())})
    return meds.DataSchema.align(t)


def coarsening(code: str) -> bool:
    """A tokenizer coarsening: an out-of-vocabulary event, a rare site, or a coded diagnosis or procedure whose
    sub-tokens are all below the vocabulary's threshold (its system alone, ``<domain>//<system>//``)."""
    return (code.endswith("//OOV") or (code.startswith("SITE//") and code.endswith("//other"))
            or (code.startswith(HIERARCHICAL) and code.endswith("//")))


def validate(t, known_codes: set) -> dict:
    """Layer (b) of a MEDS table (``known_codes``: the reference codes.parquet codes); ``failures`` lists what
    fails the production gate."""
    import meds
    import numpy as np

    rep = {"rows": t.num_rows, "subjects": len(set(t.column("subject_id").to_pylist())), "failures": []}
    try:
        meds.DataSchema.validate(t)
    except Exception as e:  # noqa: BLE001 -- any schema error fails the gate
        rep["failures"].append(f"schema: {e}")
    sid = t.column("subject_id").to_numpy()
    tm = t.column("time").to_pylist()
    order_ok = True
    for i in range(1, len(sid)):
        if sid[i] < sid[i - 1]:
            order_ok = False
            break
        if sid[i] == sid[i - 1] and tm[i - 1] is not None and (tm[i] is None or tm[i] < tm[i - 1]):
            order_ok = False
            break
    if not order_ok:
        rep["failures"].append("rows not sorted by subject and time (static rows first)")
    num = t.column("numeric_value").to_numpy(zero_copy_only=False)
    bad = int(np.sum(~np.isfinite(num[~np.isnan(num)]))) if len(num) else 0
    if bad:
        rep["failures"].append(f"{bad} non-finite numeric values")
    codes = t.column("code").to_pylist()
    unknown = {c for c in set(codes) if c not in known_codes}
    coarse = {c for c in unknown if coarsening(c)}
    hier = {c for c in unknown - coarse if c.startswith(HIERARCHICAL)}
    other = unknown - coarse - hier
    rep["coarsened_codes"] = sorted(coarse)
    rep["new_hierarchical_codes"] = len(hier)
    rep["unknown_codes"] = sorted(other)
    if other:
        rep["failures"].append(f"{len(other)} codes outside the reference: {sorted(other)[:10]}")
    rep["passed"] = not rep["failures"]
    return rep


def known_codes(meds_root=None, v=None) -> set:
    """The reference codes: every dataset's ``metadata/codes.parquet`` under ``meds_root``, and the fitted
    vocabulary's timeline and end codes (``v.codes``: with the infusion closure codes, an infusion key's actions
    seen in train or not)."""
    from pathlib import Path

    import pyarrow.parquet as pq

    out = set(v.codes) if v is not None else set()
    for f in Path(meds_root).glob("*/metadata/codes.parquet") if meds_root is not None else ():
        out |= set(pq.read_table(f, columns=["code"]).column(0).to_pylist())
    return out
