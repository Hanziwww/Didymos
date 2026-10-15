"""ELF conversion of canonical CLIF tables: CLIF_MEDS's config language, executed as DuckDB SQL.

CLIF_MEDS (the CLIF consortium's CLIF -> MEDS ETL, pinned in ``refs/``) converts its 15 ELF domains from
per-domain YAML configs, one row at a time in Python, and only knows its own domains. This module executes
the same configs as set-based SQL with the semantics of CLIF_MEDS's generic processor
(``code/process.py::_process_domain``, ``code/resolve.py``, ``code/transforms.py``):

* ``code``: a literal string, or a list of parts joined by ``//``: literals, ``col(X)`` (NULL -> ``UNK``)
  and ``{col(X): {raw: mapped}}``;
* ``time``: ``col(X)``, rows whose time is NULL are dropped; ``null`` makes a static event;
* ``numeric_value`` / ``text_value``: column names (an event is emitted even when both are NULL);
* the subject is ``subject_id_col`` (default ``patient_id``); it and any other referenced column the table
  lacks come from ``hospitalization`` (by ``hospitalization_id``);
* ``*_category`` text columns are normalised (trimmed, lower case, spaces and hyphens -> ``_``); time zones
  are dropped keeping the wall-clock time;
* ``hospitalization_id`` is carried as an extra column, except in the ``PATIENT`` domain.

CLIF_MEDS's medication processors (``code/process_med_con.py``, ``process_med_int.py``) first convert doses with
clifpy's ``convert_dose_units_by_med_category`` (preferred units of the concept's ``unit_conversion`` block,
patient weights from vitals); :func:`convert_domain` does the same before the SQL.

Parity with CLIF_MEDS is checked on real data by ``icutwin.meds.parity``. Additions for our extension domains
(``meds/elf_ext/``) and qualified codes (``meds/qualifiers.yaml``), all absent from the official configs: per
concept, ``skip_empty: true`` drops events without a numeric or text value (as CLIF_MEDS's CRRT and ECMO
processors do), ``where`` restricts it to rows matching an SQL predicate, ``distinct: true`` drops repeated
events (long tables repeat a device state per setting row) and ``extra: {name: column}`` carries a column as an
extra event column (the medication route's basis); per domain, ``join: {table, key}`` takes missing
columns from another CLIF table before hospitalization (susceptibility rows from their culture) and
``normalize: [columns]`` normalises those columns as categories are (qualifier levels).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import yaml

META_KEYS = {"subject_id_col", "elf_version", "outlier_shaping",   # CLIF_MEDS code/config.py::_CONFIG_META_KEYS
             "join", "normalize"}                                     # icutwin extensions
EVENT_COLUMNS = ("subject_id", "time", "code", "numeric_value", "text_value")


def _p(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _lit(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _col(spec: object) -> str | None:
    if isinstance(spec, str) and spec.startswith("col(") and spec.endswith(")"):
        return spec[4:-1]
    return None


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def config_table(config: dict) -> tuple[str, dict]:
    """(CLIF table, concepts): the first non-meta key, as CLIF_MEDS reads it."""
    table = next(k for k in config if k not in META_KEYS)
    return table, config[table]


def _referenced(concepts: dict) -> set[str]:
    cols = set()
    for m in concepts.values():
        if (c := _col(m.get("time"))) is not None:
            cols.add(c)
        code = m.get("code")
        for part in code if isinstance(code, list) else ():
            c = _col(next(iter(part)) if isinstance(part, dict) else part)
            if c is not None:
                cols.add(c)
    return cols


def _str(col: str, types: dict[str, str]) -> str:
    """Python ``str(value)`` of a column as CLIF_MEDS writes it into a code (booleans as True/False)."""
    if types.get(col) == "BOOLEAN":
        return f'CASE WHEN "{col}" THEN \'True\' WHEN NOT "{col}" THEN \'False\' END'
    return f'CAST("{col}" AS VARCHAR)'


def _code_sql(spec: object, types: dict[str, str]) -> str:
    if isinstance(spec, str):
        return _lit(spec)
    parts = []
    for part in spec:
        if isinstance(part, dict):
            expr, mapping = next(iter(part.items()))
            raw = f"coalesce({_str(_col(expr) or expr, types)}, 'UNK')"
            whens = " ".join(f"WHEN {_lit(k)} THEN {_lit(v)}" for k, v in (mapping or {}).items() if isinstance(k, str))
            parts.append(f"CASE {raw} {whens} ELSE {raw} END" if whens else raw)
        elif (c := _col(part)) is not None:
            parts.append(f"coalesce({_str(c, types)}, 'UNK')")
        else:
            parts.append(_lit(part))
    return " || '//' || ".join(parts)


WHITESPACE = "' ' || chr(9) || chr(10) || chr(11) || chr(12) || chr(13)"


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _types(con: duckdb.DuckDBPyConnection, path: Path) -> dict[str, str]:
    return {n: t for n, t, *_ in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{_p(path)}')").fetchall()}


def _cast(expr: str, typ: str) -> str:
    return f"CAST({expr} AS TIMESTAMP)" if typ.startswith("TIMESTAMP WITH TIME ZONE") else expr  # UTC session


def _base(con: duckdb.DuckDBPyConnection, clif_dir: Path, table: str, needed: set[str], subject: str,
          join: dict | None = None, normalize: tuple[str, ...] = (), src: Path | None = None) -> dict[str, str]:
    """Create view ``elf_base``: the table (or ``src``, its unit-converted copy) with normalised categories and
    ``normalize`` columns, plus the referenced columns it lacks, taken from the ``join`` table (extension) and
    then from hospitalization."""
    src = src or clif_dir / f"clif_{table}.parquet"
    types = _types(con, src)
    select = []
    for n, t in types.items():
        if (n.endswith("_category") or n in normalize) and t == "VARCHAR":   # polars strip_chars(): whitespace
            select.append(f"replace(replace(lower(trim(t.{_q(n)}, {WHITESPACE})), ' ', '_'), '-', '_') AS {_q(n)}")
        else:
            select.append(f"{_cast('t.' + _q(n), t)} AS {_q(n)}")
    joins = []
    lookups = ([(join["table"], join["key"])] if join else []) + [("hospitalization", "hospitalization_id")]
    for i, (other, key) in enumerate(lookups):
        missing = sorted(c for c in needed | {subject, "hospitalization_id"} if c not in types)
        if not missing or key not in types:
            continue
        path = clif_dir / f"clif_{other}.parquet"
        otypes = _types(con, path)
        cols = [c for c in missing if c in otypes]
        if not cols:
            continue
        sel = ", ".join(f"{_cast(_q(c), otypes[c])} AS {_q(c)}" for c in cols)
        joins.append(f"LEFT JOIN (SELECT DISTINCT ON ({_q(key)}) {_q(key)}, {sel} "
                     f"FROM read_parquet('{_p(path)}')) j{i} USING ({_q(key)})")
        select += [f"j{i}.{_q(c)}" for c in cols]
        types.update({c: otypes[c] for c in cols})
    con.execute(f"CREATE OR REPLACE TEMP VIEW elf_base AS SELECT {', '.join(select)} "
                f"FROM read_parquet('{_p(src)}') t {' '.join(joins)}")
    return types


def domain_sql(con: duckdb.DuckDBPyConnection, clif_dir: Path, config: dict, domain: str,
               subject_id_col: str = "patient_id", src: Path | None = None) -> str | None:
    """SQL selecting the domain's MEDS events from ``elf_base`` (None when the CLIF table does not exist)."""
    table, concepts = config_table(config)
    if (config.get("outlier_shaping") or {}).get("enabled"):
        raise ValueError(f"{domain}: ELF outlier shaping clamps values; canonical CLIF nulls outliers instead "
                         "(disable it, as clif_meds_overlay.yaml does)")
    if not (clif_dir / f"clif_{table}.parquet").exists():
        return None
    subject = config.get("subject_id_col", subject_id_col)
    types = _base(con, clif_dir, table, _referenced(concepts), subject, config.get("join"),
                  tuple(config.get("normalize") or ()), src)
    hosp = domain != "PATIENT" and "hospitalization_id" in types
    extras = sorted({k for m in concepts.values() for k in (m.get("extra") or {})})
    selects = []
    for m in concepts.values():
        time_col = _col(m.get("time"))
        time = f'CAST("{time_col}" AS TIMESTAMP)' if time_col else "CAST(NULL AS TIMESTAMP)"
        num = f'CAST("{m["numeric_value"]}" AS FLOAT)' if "numeric_value" in m else "CAST(NULL AS FLOAT)"
        text = _str(m["text_value"], types) if "text_value" in m else "CAST(NULL AS VARCHAR)"
        where = []
        if time_col:
            where.append(f'"{time_col}" IS NOT NULL')
        if m.get("where"):
            where.append(f"({m['where']})")
        if m.get("skip_empty"):
            where.append(f"(({num}) IS NOT NULL OR ({text}) IS NOT NULL)")
        selects.append(
            f'SELECT {"DISTINCT " if m.get("distinct") else ""}CAST("{subject}" AS BIGINT) AS subject_id, {time} AS time, '
            f'{_code_sql(m["code"], types)} AS code, '
            f"{num} AS numeric_value, {text} AS text_value"
            + (", CAST(hospitalization_id AS BIGINT) AS hospitalization_id" if hosp else "")
            + "".join(f", {_extra(m, k, types)} AS {_q(k)}" for k in extras)
            + " FROM elf_base" + (f" WHERE {' AND '.join(where)}" if where else ""))
    return " UNION ALL ".join(selects)


def _extra(m: dict, name: str, types: dict[str, str]) -> str:
    col = (m.get("extra") or {}).get(name)
    return f"CAST({_q(col)} AS VARCHAR)" if col in types else "CAST(NULL AS VARCHAR)"


def unit_conversion(config: dict) -> dict | None:
    """The ``unit_conversion`` block CLIF_MEDS's medication processors read from the domain's first concept."""
    _, concepts = config_table(config)
    uc = next(iter(concepts.values()), {}).get("unit_conversion") if concepts else None
    return uc if isinstance(uc, dict) else None


WEIGHT_LOOKUPS = ("backward", "segment_start")
_ROW = "icutwin_row"


def _segment_start(med, conv, plain):
    """An infusion keeps the unit decided at its start: the rows of a segment (a canonical stream from a start to
    its stop) whose first row had no weight to convert with take the unweighted conversion (``plain``), so the
    segment does not change code when a weight is charted mid-way. Returns the rows and the number changed."""
    order = med[[_ROW, "hospitalization_id", "med_category", "med_dose_unit", "admin_dttm", "mar_action_category"]] \
        .sort_values(["hospitalization_id", "med_category", "med_dose_unit", "admin_dttm", _ROW], kind="stable")
    key = ["hospitalization_id", "med_category", "med_dose_unit"]
    stopped = order["mar_action_category"].eq("stop")
    seg = stopped.groupby([order[k].astype(str) for k in key], sort=False).shift(1, fill_value=False) \
        .groupby([order[k].astype(str) for k in key], sort=False).cumsum()
    order = order.assign(_seg=seg.to_numpy())
    c = conv.set_index(_ROW)
    p = plain.set_index(_ROW).reindex(c.index)   # clifpy returns rows in its own order
    weighted_row = (c["med_dose_unit_converted"].astype(str) != p["med_dose_unit_converted"].astype(str))
    first = order.groupby(key + ["_seg"], sort=False, dropna=False)[_ROW].transform("first")
    unweighted_segment = ~weighted_row.reindex(first.to_numpy()).to_numpy()
    take_plain = order[_ROW].to_numpy()[unweighted_segment & weighted_row.reindex(order[_ROW]).to_numpy()]
    out = c.copy()
    for col in ("med_dose_converted", "med_dose_unit_converted"):
        out.loc[take_plain, col] = p.loc[take_plain, col]
    return out.reset_index(), int(len(take_plain))


def _converted_medications(con: duckdb.DuckDBPyConnection, clif_dir: Path, table: str, uc: dict, dst: Path,
                           weight_lookup: str = "backward", stats: dict | None = None) -> Path:
    """The medication table with ``med_dose_converted`` / ``med_dose_unit_converted`` as CLIF_MEDS computes them:
    clifpy's ``convert_dose_units_by_med_category`` to the preferred units (weights from vitals), or the dose
    unchanged with unit 'UNK' when conversion is off.

    ``weight_lookup``: 'backward' is clifpy's, and so CLIF_MEDS's (the last weight at or before the dose; a dose
    charted before the hospitalization's first weight keeps its unweighted unit). 'segment_start' (the training
    MEDS) is causal as well and keeps an infusion in one code: a continuous segment uses the weight in effect at
    its start, and one that started before any weight stays unweighted to its stop. A weight charted after a dose
    never converts it (a prospective task must not see it; ``docs/meds.md``)."""
    if weight_lookup not in WEIGHT_LOOKUPS:
        raise ValueError(f"weight_lookup {weight_lookup!r}")
    med = con.execute(f"SELECT * FROM read_parquet('{_p(clif_dir / f'clif_{table}.parquet')}')").df()
    if len(med) and "med_dose_unit" in med and med["med_dose_unit"].isna().all():
        # no dose unit at all (zigongICU): clifpy's DuckDB reads an all-None object column as INTEGER; as a string
        # column every row converts as a NULL-unit row of a mixed table does (dose kept, unit NULL)
        med["med_dose_unit"] = med["med_dose_unit"].astype("string")
    prefs = uc.get("preferred_units") or {}
    if uc.get("enabled") and prefs and len(med):
        if not uc.get("allow_other_meds", True):
            med = med[med["med_category"].isin(set(prefs))]
        vitals = clif_dir / "clif_vitals.parquet"
        weights = con.execute(f"""SELECT hospitalization_id, recorded_dttm, vital_category, vital_value
                                  FROM read_parquet('{_p(vitals)}') WHERE vital_category = 'weight_kg'""").df() \
            if vitals.exists() else None
        from clifpy.utils.unit_converter import convert_dose_units_by_med_category
        convert = lambda w: convert_dose_units_by_med_category(  # noqa: E731
            med_df=med.copy(), vitals_df=w, preferred_units=dict(prefs), override=bool(uc.get("override", False)))[0]
        med[_ROW] = range(len(med))
        conv = convert(weights)
        if weight_lookup == "segment_start" and table == "medication_admin_continuous":
            conv, changed = _segment_start(med, conv, convert(None if weights is None else weights.iloc[0:0]))
            if stats is not None:
                stats["rows_kept_in_their_segment_unit"] = changed
        med = conv.sort_values(_ROW).drop(columns=[c for c in conv.columns if c.startswith("_") or c == _ROW])
    else:
        med["med_dose_converted"] = med["med_dose"]
        med["med_dose_unit_converted"] = "UNK"
    dst.parent.mkdir(parents=True, exist_ok=True)
    con.register("_med_converted", med)
    con.execute(f"COPY (SELECT * FROM _med_converted) TO '{_p(dst)}' (FORMAT parquet)")
    con.unregister("_med_converted")
    return dst


def convert_domain(con: duckdb.DuckDBPyConnection, clif_dir: Path, config: dict, domain: str, out: Path,
                   weight_lookup: str = "backward", stats: dict | None = None) -> int | None:
    """Write ``out`` (MEDS events of one domain); returns the number of events, None without the table.
    ``weight_lookup``: the dose conversion's weight in effect (``_converted_medications``), which fills ``stats``."""
    con.execute("SET TimeZone = 'UTC'")   # TIMESTAMPTZ -> TIMESTAMP keeps the stored (wall-clock) time
    table, _ = config_table(config)
    src = None
    if (uc := unit_conversion(config)) is not None and (clif_dir / f"clif_{table}.parquet").exists():
        src = _converted_medications(con, clif_dir, table, uc, out.parent / f"_{out.stem}_converted.parquet",
                                     weight_lookup, stats)
    try:
        sql = domain_sql(con, clif_dir, config, domain, src=src)
        if sql is None:
            return None
        out.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY ({sql}) TO '{_p(out)}' (FORMAT parquet, COMPRESSION zstd)")
        return pq.ParquetFile(out).metadata.num_rows
    finally:
        if src is not None:
            con.execute("DROP VIEW IF EXISTS elf_base")
            src.unlink(missing_ok=True)


def qualified_config(config: dict, level: str, where: str | None = None) -> dict:
    """The domain config with ``col(level)`` appended to every concept's code (``meds/qualifiers.yaml``); with
    ``where``, only matching rows are qualified and the others keep the ELF code (a second concept)."""
    table, concepts = config_table(config)
    out = {k: v for k, v in config.items() if k != table}
    out["normalize"] = sorted({*(config.get("normalize") or ()), level})
    qualified = {}
    for name, m in concepts.items():
        q = {**m, "code": [*(m["code"] if isinstance(m["code"], list) else [m["code"]]), f"col({level})"]}
        if where:
            own = f" AND ({m['where']})" if m.get("where") else ""
            q["where"] = f"coalesce(({where}), false){own}"
            qualified[f"{name}_unqualified"] = {**m, "where": f"NOT coalesce(({where}), false){own}"}
        qualified[name] = q
    out[table] = qualified
    return out
