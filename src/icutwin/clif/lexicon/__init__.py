"""Shared lexicons used by every adapter: free-text drug names -> ingredient -> CLIF medication category.

``drugs.csv`` rows: ``priority, ingredient, pattern, continuous, intermittent, provenance``. ``pattern`` is an
RE2 regular expression matched (partial match) against the lower-cased source name; the lowest matching
priority wins, so combination products (priority 10) precede their single ingredients. ``continuous`` /
``intermittent`` give the CLIF 3.0 category per administration type; both empty = a known ingredient without a
CLIF 3.0 category (counted as "known, not in CLIF" rather than as a lexicon gap).

Categories outside the CLIF 3.0 list of a table (consortium practice, local terms, ``ext_`` concepts) are not
written in ``drugs.csv``: :func:`rows` overlays them from the medication registry
(:func:`icutwin.clif.conform.medication_extensions`) by ingredient and adds ``continuous_kind`` /
``intermittent_kind`` ('' for CLIF 3.0, else the registry kind) and ``untyped_iv``. Adapters resolve names with
:meth:`ClifAdapter.regex_lookup` over this effective lexicon; :func:`resolve` applies the same rule.
"""

from __future__ import annotations

import csv
import io
from functools import lru_cache
from importlib import resources

import duckdb

KIND_COLUMNS = ("continuous_kind", "intermittent_kind", "untyped_iv")


@lru_cache(maxsize=None)
def base_rows(name: str = "drugs") -> tuple[dict, ...]:
    """The lexicon file as written (for ``drugs``: CLIF 3.0 categories only)."""
    text = resources.files("icutwin.clif").joinpath("lexicon", f"{name}.csv").read_text(encoding="utf-8")
    return tuple(csv.DictReader(text.splitlines()))


@lru_cache(maxsize=None)
def rows(name: str = "drugs") -> tuple[dict, ...]:
    """The effective lexicon: ``drugs`` with the registry's extension categories overlaid by ingredient."""
    base = base_rows(name)
    return _with_extensions(base) if name == "drugs" else base


def _with_extensions(base: tuple[dict, ...]) -> tuple[dict, ...]:
    from icutwin.clif.conform import medication_extensions

    ext: dict[tuple[str, str], dict] = {}
    for r in medication_extensions():
        col = "continuous" if r["table"] == "medication_admin_continuous" else "intermittent"
        for ing in (r["ingredients"] or "").split("|"):
            if not ing:
                continue
            if (col, ing) in ext:
                raise ValueError(f"medication_extensions.csv: ingredient {ing!r} has two {col} categories")
            ext[(col, ing)] = r
    unknown = sorted({ing for _, ing in ext} - {r["ingredient"] for r in base})
    if unknown:
        raise ValueError(f"medication_extensions.csv names ingredients the lexicon does not define: {unknown}")
    out = []
    for r in base:
        r = dict(r)
        for col in ("continuous", "intermittent"):
            e = ext.get((col, r["ingredient"]))
            r[f"{col}_kind"] = ""
            if e is not None:
                if r[col]:
                    raise ValueError(f"lexicon {r['ingredient']!r}: {col} category {r[col]!r} is CLIF 3.0; "
                                     f"medication_extensions.csv must not replace it with {e['med_category']!r}")
                r[col], r[f"{col}_kind"] = e["med_category"], e["kind"]
        r["untyped_iv"] = (ext.get(("intermittent", r["ingredient"])) or {}).get("untyped_iv") or ""
        out.append(r)
    return tuple(out)


def csv_text(name: str = "drugs") -> str:
    """The effective lexicon as CSV text (what adapters register)."""
    rs = rows(name)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rs[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rs)
    return buf.getvalue()


def resolve(names: list[str], name: str = "drugs", exclude: tuple[str, ...] = ()) -> dict[str, dict | None]:
    """Resolve names exactly as the adapters do (DuckDB RE2 semantics); returns name -> lexicon row or None.

    ``exclude`` drops lexicon rows by ingredient (e.g. ``non_systemic_formulation`` when only the ingredient
    matters, as for ingredient codes that do not distinguish formulations).
    """
    con = duckdb.connect()
    lex = rows(name)
    con.execute("CREATE TABLE lex (priority INTEGER, idx INTEGER, pattern VARCHAR)")
    con.executemany("INSERT INTO lex VALUES (?, ?, ?)", [(int(r["priority"]), i, r["pattern"])
                                                        for i, r in enumerate(lex) if r.get("ingredient") not in exclude])
    con.execute("CREATE TABLE t (name VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?)", [(n,) for n in names])
    hits = con.execute("""
        SELECT name, idx FROM (
            SELECT t.name, l.idx, row_number() OVER (PARTITION BY t.name ORDER BY l.priority, l.idx) AS k
            FROM t JOIN lex l ON regexp_matches(lower(t.name), l.pattern))
        WHERE k = 1""").fetchall()
    found = {n: lex[i] for n, i in hits}
    return {n: found.get(n) for n in names}
