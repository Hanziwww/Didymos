"""MEDS stays -> their tokenizable parts (independent of any fitted vocabulary).

A MEDS subject is an ICU stay (``icutwin.meds.build``): static events (time NULL), the hospitalization's header
and outcome events, and timeline events from ``window_start`` to ICU discharge. Per the grammar spec a stay
becomes
* ``prefix`` and ``header`` field values (static events, ICU and hospital header events, stay metadata),
* ``goal`` field values (outcome events and derived concepts),
* ``timeline``: (time, domain rank, code, numeric value, text) in MEDS order, the ICU admission as the marker,
  derivable fields left out,
* ``end``: hospital discharge category, death, billing diagnoses (canonical order).
Events no part takes are counted by reason (A10).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import polars as pl

from icutwin.meds.domains import DOMAIN_ORDER
from icutwin.tokenization.spec import code_pattern, load_spec

RANK = {d: i for i, d in enumerate(DOMAIN_ORDER)}


def domain_rank(code: str) -> int:
    return RANK.get(code.split("//", 1)[0], len(DOMAIN_ORDER))


@dataclass
class Stay:
    subject_id: int
    stay_id: str
    dataset: str
    split: str
    icu_in: datetime
    icu_out: datetime
    clock_valid: bool
    prefix: dict = field(default_factory=dict)
    header: dict = field(default_factory=dict)
    goal: dict = field(default_factory=dict)
    timeline: list = field(default_factory=list)   # (time, rank, code, numeric_value, text_value)
    end: list = field(default_factory=list)        # end-block codes in canonical order
    dropped: Counter = field(default_factory=Counter)   # events without a part, by reason


class _Rules:
    """The spec's code templates compiled once."""

    def __init__(self, spec: dict):
        self.prefix_codes = [(f["field"], code_pattern(f["code"])) for f in spec["prefix"] if "code" in f]
        self.prefix_numeric = {f["numeric"]: f["field"] for f in spec["prefix"] if "numeric" in f}
        self.header_codes = [(f["field"], code_pattern(f["code"])) for f in spec["header"] if "code" in f]
        self.header_numeric = {f["numeric"]: f for f in spec["header"] if "numeric" in f}
        self.goal_codes = [(f["field"], code_pattern(f["code"])) for f in spec["goal"] if "code" in f]
        marker = spec["timeline"]["marker"]
        self.marker = code_pattern(marker["from_code"])
        self.marker_token = marker["token"]
        self.end_codes = [code_pattern(e["code"]) for e in spec["end"] if "code" in e]
        self.end_domains = {e["domain"] for e in spec["end"] if "domain" in e}
        self.derived = [(d["code"], tuple(d["when_same_time"])) for d in spec["timeline"]["derived_not_tokenized"]]
        self.goal_only = [p for f, p in self.goal_codes]   # outcome codes also read by the goal block

    def match(self, rules: list, code: str) -> tuple[str, str] | None:
        for name, pat in rules:
            m = pat.match(code)
            if m:
                return name, m.group("value") if "value" in pat.groupindex else code
        return None


def _hospital_admission(rows: list[tuple]) -> datetime | None:
    for t, code, *_ in rows:
        if t is not None and code.startswith("HOSP//admission_type//"):
            return t
    for t, code, *_ in rows:
        if t is not None and code == "HOSP//age_charted":
            return t
    return None


def build_stay(meta: dict, rows: list[tuple], derived: dict | None = None) -> Stay:
    """``rows``: the subject's MEDS events (time, code, numeric_value, text_value) in MEDS order."""
    spec = load_spec()
    r = _rules()
    s = Stay(meta["subject_id"], meta["stay_id"], meta["dataset"], meta["split"], meta["icu_in"], meta["icu_out"],
             bool(meta.get("clock_valid")))
    hosp_adm = _hospital_admission(rows)
    same_time: dict[datetime, set[str]] = {}
    for t, code, *_ in rows:
        if t is not None:
            same_time.setdefault(t, set()).add(code)
    end_dx = []
    for t, code, num, text in rows:
        if code == "MEDS_BIRTH":
            s.dropped["birth (age band carries it)"] += 1
            continue
        if t is None:   # static: prefix
            hit = r.match(r.prefix_codes, code)
            if hit:
                s.prefix.setdefault(hit[0], hit[1])
            else:
                s.dropped["static code without a prefix field"] += 1
            continue
        if code in r.prefix_numeric:
            s.prefix.setdefault(r.prefix_numeric[code], num)
            continue
        hit = r.match(r.prefix_codes, code)   # a stay-level prefix field charted at ICU admission (era)
        if hit:
            s.prefix.setdefault(hit[0], hit[1])
            continue
        if code in r.header_numeric:
            f = r.header_numeric[code]
            if num is not None:
                s.header.setdefault(f["field"], num + f.get("offset", 0))
            continue
        hit = r.match(r.header_codes, code)
        if hit:
            s.header.setdefault(hit[0], hit[1])
            if r.marker.match(code):
                s.timeline.append((t, domain_rank(r.marker_token), r.marker_token, None, None))
            continue
        goal_hit = r.match(r.goal_codes, code)
        if goal_hit:
            s.goal.setdefault(goal_hit[0], goal_hit[1])
        if any(p.match(code) for p in r.end_codes):
            if code not in s.end:
                s.end.append(code)
            continue
        if code.split("//", 1)[0] in r.end_domains:
            end_dx.append(code)
            continue
        if code.startswith("HOSP//"):
            s.dropped["hospital code without a field"] += 1
            continue
        if any(code == c and set(parts) <= same_time.get(t, set()) for c, parts in r.derived):
            s.dropped["derivable (decoder derives it)"] += 1
            continue
        s.timeline.append((t, domain_rank(code), code, num, text))
    s.timeline.sort(key=lambda e: (e[0], e[1], e[2], e[3] if e[3] is not None else float("-inf"), e[4] or ""))
    # end block: the end codes in the spec's order (one each), then diagnoses in code order
    s.end = [next(c for c in s.end if p.match(c)) for p in r.end_codes if any(p.match(c) for c in s.end)] \
        + sorted(end_dx)
    _header_from_meta(s, spec, hosp_adm)
    _goal_from_meta(s, spec, rows, derived or {})
    return s


def _header_from_meta(s: Stay, spec: dict, hosp_adm: datetime | None) -> None:
    for f in spec["header"]:
        if "hours_between" in f:
            s.header[f["field"]] = None if hosp_adm is None else (s.icu_in - hosp_adm).total_seconds() / 3600
        if "clock_hour" in f:
            s.header[f["field"]] = s.icu_in.hour + s.icu_in.minute / 60 if s.clock_valid else None


def _goal_from_meta(s: Stay, spec: dict, rows: list[tuple], derived: dict) -> None:
    for f in spec["goal"]:
        name = f["field"]
        if "hours_between" in f:
            s.goal[name] = (s.icu_out - s.icu_in).total_seconds() / 3600
        elif "episode" in f:
            eps = derived.get("episodes") or {}
            s.goal[name] = "yes" if any(e == f["episode"] and a < s.icu_out and b > s.icu_in
                                        for e, a, b in eps.get(s.stay_id, ())) else "no"
        elif "domain_events" in f:
            dom = f["domain_events"] + "//"
            s.goal[name] = "yes" if any(t is not None and code.startswith(dom) and s.icu_in <= t <= s.icu_out
                                        for t, code, *_ in rows) else "no"
        elif "derived" in f:
            d = f["derived"]
            vals = (derived.get(d["file"]) or {}).get(s.stay_id)
            if d["how"] == "max_in_icu":
                vals = [v for t, v in (vals or ()) if v is not None and (t is None or s.icu_in <= t <= s.icu_out)]
                s.goal[name] = max(vals) if vals else None
            else:
                s.goal[name] = vals
        s.goal.setdefault(name, None)


@lru_cache(maxsize=1)
def _rules() -> _Rules:
    return _Rules(load_spec())


# -- loading ---------------------------------------------------------------------------------------------------
def load_stays(meds_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(meds_dir / "metadata" / "stays.parquet")


def load_events(meds_dir: Path, split: str) -> pl.DataFrame:
    files = sorted((meds_dir / "data" / split).glob("*.parquet"))
    cols = ["subject_id", "time", "code", "numeric_value", "text_value"]
    if not files:
        return pl.DataFrame(schema={"subject_id": pl.Int64, "time": pl.Datetime, "code": pl.Utf8,
                                    "numeric_value": pl.Float32, "text_value": pl.Utf8})
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files])


def load_derived(derived_dir: Path) -> dict:
    """The derived concepts the goal block reads, keyed by stay id."""
    out: dict = {"episodes": {}, "kdigo": {}, "sofa_24h": {}}
    p = derived_dir / "episodes.parquet"
    if p.exists():
        for sid, ep, a, b in pl.read_parquet(p).select("stay_id", "episode", "start_dttm", "end_dttm").iter_rows():
            out["episodes"].setdefault(sid, []).append((ep, _naive(a), _naive(b)))
    p = derived_dir / "kdigo.parquet"
    if p.exists():
        for sid, t, v in pl.read_parquet(p).select("stay_id", "t", "aki_stage").iter_rows():
            out["kdigo"].setdefault(sid, []).append((_naive(t), v))
    p = derived_dir / "sofa_24h.parquet"
    if p.exists():
        for sid, v in pl.read_parquet(p).select("stay_id", "sofa_total").iter_rows():
            out["sofa_24h"][sid] = v
    return out


def _naive(t):
    return t.replace(tzinfo=None) if t is not None and getattr(t, "tzinfo", None) is not None else t


def shard_files(meds_dir: Path, split: str) -> list[Path]:
    """A split's MEDS shards in the order the stays are read (each shard holds whole subjects)."""
    return sorted((meds_dir / "data" / split).glob("*.parquet"))


def iter_stays(meds_dir: Path, split: str, derived_dir: Path | None = None, *, file: Path | None = None,
               start: int = 0, stop: int | None = None, derived: dict | None = None):
    """Yield the :class:`Stay` of every subject of a split (MEDS order kept), or only of the subjects
    ``start:stop`` of one shard ``file`` (parallel encoding and validation); ``derived`` spares re-reading the
    derived concepts."""
    meta = {r["subject_id"]: r for r in load_stays(meds_dir).iter_rows(named=True)}
    if derived is None:
        derived = load_derived(derived_dir) if derived_dir is not None and derived_dir.exists() else {}
    if file is None:
        events = load_events(meds_dir, split)
    else:
        cols = ["subject_id", "time", "code", "numeric_value", "text_value"]
        events = pl.read_parquet(file, columns=cols)
        keep = events.select("subject_id").unique(maintain_order=True).to_series()[start:stop]
        events = events.filter(pl.col("subject_id").is_in(keep.implode()))
    for (sid,), part in events.group_by(["subject_id"], maintain_order=True):
        m = meta[sid]
        rows = part.select("time", "code", "numeric_value", "text_value").rows()
        yield build_stay({**m, "icu_in": _naive(m["icu_in"]), "icu_out": _naive(m["icu_out"])}, rows, derived)
