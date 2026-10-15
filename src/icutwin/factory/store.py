"""The stay store (plan Phase F, Factory design 1): every stay of a source as columnar arrays, built once,
memory-mapped, and versioned with its source's manifest. It is the factory's source of truth for trajectories and
labels.

A store is a directory of segments (``seg_<k>/``; the twin supply appends one per generated batch) plus
``channels.json`` (``factory.channels``, with the training twins' normalization statistics) and ``manifest.json``
(the sources, seeds, every file's SHA-256). Per segment:
* ``stays.parquet``: one row per stay: ``stay_id``, ``dataset``, ``site``, ``split`` (twins: ``train`` / ``val`` by a
  hash of the id; real stays: their source split), ``weight`` and ``cell`` (the generation plan's), ``los_min``,
  ``end_kind`` (``alive`` / ``dead`` / ``unknown`` / ``censored``: a stay without an ICU discharge), ``hosp_death``,
  the prefix and header fields, ``end_codes`` (the end block: hospital outcome, discharge diagnoses), and counts.
* events (``ev_*``, offsets ``ev_ptr``): ``ev_t`` int32 minutes since ICU admission (negative before it; times are
  when values became available: MEDS times a lab at its result), ``ev_c`` int16 channel, ``ev_v`` float32 value (NaN
  without one), ``ev_bin`` uint8 the value's decile bin (255: none), ``ev_f`` uint8 flags (bit 0 a value, bit 1 a
  text value, bits 2-4 an infusion's action, ``channels.ACTIONS``).
* state intervals (``iv_*``, offsets ``iv_ptr``): ``iv_c`` channel, ``iv_s`` / ``iv_e`` int32 minutes, ``iv_v`` float32
  (an infusion's rate): infusions from a start, going or dose change to the next one or a stop; the respiratory
  device and mode until the next one; CRRT and MCS active while charted within 24 hours (the tokenizer's state rules).
  An interval still open at the end of the stay ends there.
* derived timelines (``dv_*``, offsets ``dv_ptr``), from the labels' decode (``generate.clif_out.derived``): ``dv_k``
  uint8 (``DERIVED``), ``dv_s`` / ``dv_e`` int32 minutes, ``dv_v`` float32. Labels only: derived concepts are never
  input channels (study lead, Q&A 2).
* key-major copies (``evx_*`` by channel, ``dvx_*`` by derived kind, with offsets ``<g>x_ptr``): the label engine reads
  one channel or concept of a segment as a contiguous slice, so its cost does not grow with the pool.

A pool drawn by a supply plan (``generate.supply``) also keeps ``weights.parquet``: its twins' importance weights,
computed for a declared set of segments (``set_weights``; a store opened with other segments keeps each segment's own
``weight``).

Features come from the twin decoded with drawn values (rank persistence) and labels from the same twin with values at
their bins' medians on the same clock (``factory.twins``). Real stays' token sequences go through the same decode
(the real-through-tokens store), so twins and real stays compare like with like.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from icutwin.factory.channels import ACTIONS, Channels

EPISODES = ("imv", "vasopressor", "rrt", "nippv", "cpap", "nasal_cannula", "face_mask", "trach_collar", "room_air")
SOFA = ("sofa_total", "sofa_resp", "sofa_coag", "sofa_liver", "sofa_cv_97", "sofa_cns", "sofa_renal")
DERIVED = (*(f"ep:{e}" for e in EPISODES), "kdigo", "kdigo_creat", "kdigo_uo", "nee", *(f"sofa24:{s}" for s in SOFA),
           "fluid_net", "ase_onset", "ase_onset_wo_lactate", "presumed_infection")
DK = {k: i for i, k in enumerate(DERIVED)}
PREFIX = ("sex", "age", "era")
HEADER = ("unit_type", "admission_type", "admission_source", "adm_dx_group", "surgical_status", "prior_icu_stays",
          "pre_icu_los", "admission_hour")
ARRAYS = {"ev": (("t", np.int32), ("c", np.int16), ("v", np.float32), ("bin", np.uint8), ("f", np.uint8)),
          "iv": (("c", np.int16), ("s", np.int32), ("e", np.int32), ("v", np.float32)),
          "dv": (("k", np.uint8), ("s", np.int32), ("e", np.int32), ("v", np.float32))}
NO_BIN = 255
ACTIVE_MIN = 24 * 60
STATE_ACTIVE = {"CRRT": "STATE//CRRT//active", "MCS": "STATE//MCS//active"}


def twin_split(stay_id: str, val_pct: float = 10.0) -> str:
    """``val`` for about ``val_pct`` percent of twins (a hash of the id), else ``train`` (study lead, Q&A 16)."""
    h = int(hashlib.sha1(stay_id.encode()).hexdigest()[:8], 16)
    return "val" if h % 10_000 < val_pct * 100 else "train"


def _minutes(t, t0) -> int:
    """Whole minutes from ``t0`` to ``t``, rounded down (an event at 59.7 minutes lies in the first hour)."""
    return math.floor((t - t0).total_seconds() / 60)


# -- one stay -> arrays ------------------------------------------------------------------------------------------
def stay_arrays(s, ch: Channels, v, derived: dict | None = None, stay_key: str | None = None) -> tuple[dict, dict]:
    """(row, arrays) of one stay: ``s`` decoded for features, ``derived`` the derived concepts of its labels' decode
    (``derived_of``), keyed by ``stay_key`` (default ``s.stay_id``)."""
    from icutwin.meds.outcomes import in_hospital_death
    from icutwin.tokenization.vocab import bin_of, infusion_key

    t0 = s.icu_in
    los = _minutes(s.icu_out, t0)
    ev = {k: [] for k, _ in ARRAYS["ev"]}
    disc = None
    for t, _r, code, num, text in s.timeline:
        cid, act = ch.of(code)
        if cid < 0:
            continue
        m = _minutes(t, t0)
        has = num is not None and not (isinstance(num, float) and math.isnan(num))
        info = v.codes.get(code) or {}
        ev["t"].append(m)
        ev["c"].append(cid)
        ev["v"].append(float(num) if has else np.nan)
        numeric = has and info.get("kind") in ("numeric", "bin")
        ev["bin"].append(bin_of(float(num), info.get("edges") or []) if numeric else NO_BIN)
        ev["f"].append((1 if has else 0) | (2 if text is not None else 0) | (act << 2))
        if disc is None and code.startswith("ICU//discharge//") and m >= 0:
            disc = code.rsplit("//", 1)[1]
    iv = _intervals(s.timeline, t0, los, ch, infusion_key)
    dv = _derived_rows(derived or {}, stay_key or s.stay_id, t0)
    end_codes = list(s.end)
    hosp = in_hospital_death([*end_codes, *(c for _t, _r, c, _n, _x in s.timeline if c.startswith("ICU//discharge//"))])
    from icutwin.stage.spec import stay_dataset

    row = {"stay_id": s.stay_id, "dataset": stay_dataset(s.dataset, stay_key or s.stay_id), "site": str(s.prefix.get("site")),
           "los_min": los, "end_kind": disc if disc in ("alive", "dead", "unknown") else "censored",
           "hosp_death": None if hosp is None or (isinstance(hosp, float) and math.isnan(hosp)) else bool(hosp),
           "end_codes": json.dumps(end_codes),
           **{k: _plain(s.prefix.get(k)) for k in PREFIX}, **{k: _plain(s.header.get(k)) for k in HEADER},
           "n_ev": len(ev["t"]), "n_iv": len(iv["c"]), "n_dv": len(dv["k"])}
    return row, {"ev": ev, "iv": iv, "dv": dv}


def _plain(x):
    return None if x is None else (float(x) if isinstance(x, (int, float)) else str(x))


def _intervals(timeline, t0, los: int, ch: Channels, infusion_key) -> dict:
    from icutwin.tokenization.spec import load_spec

    st = load_spec()["windows"]["state"]
    inf = st["infusions"]
    resp = tuple(st["respiratory"])
    out = {k: [] for k, _ in ARRAYS["iv"]}
    running: dict = {}            # infusion key -> (start minute, rate)
    current: dict = {}            # respiratory prefix -> (channel, start)
    active: dict = {}             # domain -> (start, last charting)

    def emit(c, a, b, val=np.nan):
        if b > a:
            out["c"].append(c)
            out["s"].append(a)
            out["e"].append(b)
            out["v"].append(val)

    for t, _r, code, num, _x in timeline:
        m = _minutes(t, t0)
        k = infusion_key(code)
        if k is not None and k[0] in ch.index:
            cid = ch.index[k[0]]
            if k[1] in inf["ends"]:
                if k[0] in running:
                    a, rate = running.pop(k[0])
                    emit(cid, a, m, rate)
            elif k[1] in inf["begins"] or k[1] in inf["continues"]:
                rate = float(num) if num is not None and not math.isnan(num) else None
                if k[0] in running:
                    a, prev = running[k[0]]
                    emit(cid, a, m, prev)
                    rate = prev if rate is None else rate
                running[k[0]] = (m, np.nan if rate is None else rate)
        for p in resp:
            if code.startswith(p) and code in ch.index:
                if p in current and current[p][0] != ch.index[code]:
                    emit(current[p][0], current[p][1], m)
                if p not in current or current[p][0] != ch.index[code]:
                    current[p] = (ch.index[code], m)
        dom = code.split("//", 1)[0]
        if dom in STATE_ACTIVE and STATE_ACTIVE[dom] in ch.index:
            a, last = active.get(dom, (m, m))
            if m - last > ACTIVE_MIN:
                emit(ch.index[STATE_ACTIVE[dom]], a, last + ACTIVE_MIN)
                a = m
            active[dom] = (a, m)
    end = max(los, max((_minutes(t, t0) for t, *_ in timeline), default=los))
    for key, (a, rate) in running.items():
        emit(ch.index[key], a, end, rate)
    for c, a in current.values():
        emit(c, a, end)
    for dom, (a, last) in active.items():
        emit(ch.index[STATE_ACTIVE[dom]], a, min(last + ACTIVE_MIN, end))
    order = np.lexsort((np.asarray(out["c"]), np.asarray(out["s"]))) if out["c"] else []
    return {k: [out[k][i] for i in order] for k in out}


def _derived_rows(d: dict, key: str, t0) -> dict:
    out = {k: [] for k, _ in ARRAYS["dv"]}

    def add(kind, a, b, val):
        out["k"].append(DK[kind])
        out["s"].append(_minutes(a, t0))
        out["e"].append(_minutes(b, t0))
        out["v"].append(np.nan if val is None else float(val))

    for ep, a, b in d.get("episodes", {}).get(key, []):
        if f"ep:{ep}" in DK and a is not None:
            add(f"ep:{ep}", a, b if b is not None else a, None)
    for kind, rows in (("kdigo", d.get("kdigo", {})), ("kdigo_creat", d.get("kdigo_creat", {})),
                       ("kdigo_uo", d.get("kdigo_uo", {})), ("nee", d.get("nee", {}))):
        for t, val in rows.get(key, []):
            if t is not None and val is not None:
                add(kind, t, t, val)
    for comp, val in (d.get("sofa24", {}).get(key) or {}).items():
        if val is not None:
            out["k"].append(DK[f"sofa24:{comp}"])
            out["s"].append(0)
            out["e"].append(24 * 60)
            out["v"].append(float(val))
    for day, net in d.get("fluid_net", {}).get(key, []):
        out["k"].append(DK["fluid_net"])
        out["s"].append(int(day) * 1440)
        out["e"].append((int(day) + 1) * 1440)
        out["v"].append(float(net))
    for kind in ("ase_onset", "ase_onset_wo_lactate", "presumed_infection"):
        for t in d.get(kind, {}).get(key, []):
            add(kind, t, t, 1.0)
    order = np.lexsort((np.asarray(out["k"]), np.asarray(out["s"]))) if out["k"] else []
    return {k: [out[k][i] for i in order] for k in out}


def derived_of(derived_dir: Path) -> dict:
    """The derived concepts the store keeps, keyed by stay id (``meds.derived.derive``'s files in ``derived_dir``):
    episodes, KDIGO (full, creatinine-only, urine output), norepinephrine equivalents, the first-day SOFA's
    components, the daily fluid balance, ASE onsets."""
    import polars as pl

    from icutwin.tokenization.stays import _naive

    d = Path(derived_dir)
    out: dict = {k: {} for k in ("episodes", "kdigo", "kdigo_creat", "kdigo_uo", "nee", "sofa24", "fluid_net",
                                 "ase_onset", "ase_onset_wo_lactate", "presumed_infection")}

    def read(name, cols):
        p = d / f"{name}.parquet"
        return pl.read_parquet(p).select(cols).iter_rows() if p.exists() else ()

    for sid, ep, a, b in read("episodes", ["stay_id", "episode", "start_dttm", "end_dttm"]):
        out["episodes"].setdefault(sid, []).append((ep, _naive(a), _naive(b)))
    for sid, t, full, creat, uo in read("kdigo", ["stay_id", "t", "aki_stage", "aki_stage_creat", "aki_stage_uo"]):
        for k, val in (("kdigo", full), ("kdigo_creat", creat), ("kdigo_uo", uo)):
            if val is not None:
                out[k].setdefault(sid, []).append((_naive(t), val))
    for sid, t, nee in read("nee", ["hospitalization_id", "t", "nee_mcg_kg_min"]):
        out["nee"].setdefault(sid, []).append((_naive(t), nee))
    sofa = d / "sofa_24h.parquet"
    for r in (pl.read_parquet(sofa).iter_rows(named=True) if sofa.exists() else ()):
        out["sofa24"][r["stay_id"]] = {k: r.get(k) for k in SOFA}
    for sid, day, net in read("fluid_balance", ["stay_id", "icu_day", "net_ml"]):
        if day is not None and net is not None:
            out["fluid_net"].setdefault(sid, []).append((day, net))
    cols = ["stay_id", "ase_onset_w_lactate_dttm", "ase_onset_wo_lactate_dttm", "presumed_infection_onset_dttm"]
    for sid, w, wo, pi in read("ase", cols):
        for k, t in (("ase_onset", w), ("ase_onset_wo_lactate", wo), ("presumed_infection", pi)):
            if t is not None:
                out[k].setdefault(sid, []).append(_naive(t))
    return out


# -- building ----------------------------------------------------------------------------------------------------
_WORKER: dict = {}


def _worker_state(vocab_path: str, persistence_path: str | None):
    key = (vocab_path, persistence_path)
    if _WORKER.get("key") != key:
        from icutwin.tokenization.decode import load_persistence
        from icutwin.tokenization.vocab import Vocab

        v = Vocab.load(Path(vocab_path))
        _WORKER.update(key=key, v=v, ch=Channels.from_vocab(v),
                       rho=load_persistence(Path(persistence_path), v) if persistence_path else None)
    return _WORKER["v"], _WORKER["ch"], _WORKER["rho"]


def _chunk(job: tuple) -> tuple[list, dict]:
    """Rows and arrays of one chunk of stays (a worker process): decode, derive, convert."""
    rows, vocab_path, persistence_path, seed, work = job
    from icutwin.factory import twins

    v, ch, rho = _worker_state(vocab_path, persistence_path)
    feats = [twins.decode_features(r["ids"], v, r["stay_id"], seed, rho, r.get("clock")) for r in rows]
    der = {}
    if work is not None:
        from icutwin.generate import clif_out

        labs = [twins.decode_labels(r["ids"], v, r["stay_id"], seed, r.get("clock")) for r in rows]
        clif_out.derived(labs, Path(work))
        der = _merge([derived_of(p) for p in sorted(Path(work).glob("chunk_*/derived"))])
        shutil.rmtree(work, ignore_errors=True)
    out_rows, arrays = [], {g: {k: [] for k, _ in ARRAYS[g]} for g in ARRAYS}
    for r, s in zip(rows, feats, strict=True):
        row, arr = stay_arrays(s, ch, v, der)
        row.update({k: r[k] for k in ("split", "weight", "cell", "conditions", "goal_met") if k in r})
        out_rows.append(row)
        for g in ARRAYS:
            for k, _ in ARRAYS[g]:
                arrays[g][k] += arr[g][k]
    return out_rows, arrays


def _merge(parts: list[dict]) -> dict:
    out: dict = {}
    for p in parts:
        for k, d in p.items():
            out.setdefault(k, {}).update(d)
    return out


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _channels_of(out: Path, v_path: Path):
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(Path(v_path))
    ch = Channels.from_vocab(v)
    cpath = out / "channels.json"
    if cpath.exists():
        if Channels.load(cpath).names != ch.names:
            raise ValueError("the store's channels differ from this vocabulary's")
    else:
        ch.save(cpath)
    return v, ch


def build(rows: list[dict], v_path: Path, out: Path, segment: str, source: dict, *, seed: int = 0,
          persistence: Path | None = None, derived: bool = True, workers: int = 1, chunk: int = 500,
          work: Path | None = None) -> dict:
    """Append segment ``segment`` to the store at ``out``: ``rows`` are dicts with ``stay_id``, ``ids`` (tokens) and
    optionally ``split``, ``weight``, ``cell``; ``source`` is recorded in the manifest. Chunks run in ``workers``
    processes; the output does not depend on their number. Returns the segment's manifest entry."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _channels_of(out, v_path)
    work = Path(work or out / "_work") / segment
    jobs = [(rows[a:a + chunk], str(v_path), str(persistence) if persistence else None, seed,
             str(work / f"c{a // chunk:05d}") if derived else None) for a in range(0, len(rows), chunk)]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_chunk, jobs))
    else:
        results = [_chunk(j) for j in jobs]
    shutil.rmtree(work, ignore_errors=True)
    return _write(results, v_path, out, segment, {"source": source, "seed": seed, "derived": derived})


def build_from_stays(stays: list, derived: dict, v_path: Path, out: Path, segment: str, source: dict,
                     split: str | None = None) -> dict:
    """Append segment ``segment`` from stays given as they are (``tokenization.stays.Stay``: e.g. real stays read
    from MEDS by ``iter_stays``, the real-raw store that measures the tokens' quantization) with their derived
    concepts (``derived_of`` of their derived directory)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    v, ch = _channels_of(out, v_path)
    rows, arrays = [], {g: {k: [] for k, _ in ARRAYS[g]} for g in ARRAYS}
    for s in stays:
        row, arr = stay_arrays(s, ch, v, derived)
        if split is not None:
            row["split"] = split
        rows.append(row)
        for g in ARRAYS:
            for k, _ in ARRAYS[g]:
                arrays[g][k] += arr[g][k]
    return _write([(rows, arrays)], v_path, out, segment, {"source": source, "derived": bool(derived)})


INDEX = {"ev": ("c", ("t", "v", "bin", "f")), "dv": ("k", ("s", "e", "v"))}


def _index_arrays(d: Path, n_keys: dict, out: Path | None = None) -> None:
    """A segment's key-major copies (``<g>x_*``: events by channel, derived rows by kind), so the label engine reads one
    channel or concept of a segment as a contiguous slice: the rows stably sorted by key (each key's rows keep their
    stay and time order), each row's local stay, and each key's offsets (``<g>x_ptr``, ``n_keys + 1``)."""
    out = Path(out or d)
    for g, (col, fields) in INDEX.items():
        key = np.load(d / f"{g}_{col}.npy")
        ptr = np.load(d / f"{g}_ptr.npy")
        order = np.argsort(key, kind="stable")
        np.save(out / f"{g}x_ptr.npy", np.searchsorted(key[order], np.arange(n_keys[g] + 1), "left").astype(np.int64))
        np.save(out / f"{g}x_stay.npy", np.repeat(np.arange(len(ptr) - 1, dtype=np.int32), np.diff(ptr))[order])
        for f in fields:
            np.save(out / f"{g}x_{f}.npy", np.load(d / f"{g}_{f}.npy")[order])


def index_segments(root: Path) -> list[str]:
    """Add the key-major copies to a store's segments that lack them (stores built before them); returns the
    segments indexed. Files are written beside each segment, then moved in, and the manifest records them."""
    root = Path(root)
    man_path = root / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    n_keys = {"ev": len(Channels.load(root / "channels.json")), "dv": len(DERIVED)}
    done = []
    for s in man["segments"]:
        d = root / s["name"]
        if (d / "evx_ptr.npy").exists():
            continue
        tmp = root / f".{s['name']}.index.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        _index_arrays(d, n_keys, tmp)
        for p in sorted(tmp.iterdir()):
            s["files"][p.name] = _sha(p)
            os.replace(p, d / p.name)
        tmp.rmdir()
        done.append(s["name"])
    if done:
        tmpm = root / "manifest.json.partial"
        tmpm.write_text(json.dumps(man, indent=1), encoding="utf-8")
        os.replace(tmpm, man_path)
    return done


def _write(results: list, v_path: Path, out: Path, segment: str, info: dict) -> dict:
    """Write a segment's arrays (and their key-major copies) and stays atomically and record it in the manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tmp = out / f".{segment}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    stays = [r for rs, _a in results for r in rs]
    for g, fields in ARRAYS.items():
        counts = []
        for rs, _a in results:
            counts += [r[f"n_{g}"] for r in rs]
        np.save(tmp / f"{g}_ptr.npy", np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
        for k, dt in fields:
            np.save(tmp / f"{g}_{k}.npy", np.asarray([x for _r, a in results for x in a[g][k]], dtype=dt))
    _index_arrays(tmp, {"ev": len(Channels.load(out / "channels.json")), "dv": len(DERIVED)})
    pq.write_table(pa.Table.from_pylist(stays), tmp / "stays.parquet")
    files = {p.name: _sha(p) for p in sorted(tmp.iterdir())}
    final = out / segment
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    entry = {"name": segment, **info, "stays": len(stays), "events": int(sum(r["n_ev"] for r in stays)),
             "files": files}
    man_path = out / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {"version": 1, "segments": []}
    man["vocab_sha256"] = _sha(Path(v_path))
    man["channels_sha256"] = _sha(out / "channels.json")
    man["segments"] = [s for s in man["segments"] if s["name"] != segment] + [entry]
    tmpm = out / "manifest.json.partial"
    tmpm.write_text(json.dumps(man, indent=1), encoding="utf-8")
    os.replace(tmpm, man_path)
    return entry


# -- reading -----------------------------------------------------------------------------------------------------
@dataclass
class Segment:
    name: str
    stays: object                   # pandas DataFrame
    arr: dict                       # "ev_t" -> memory-mapped array, ...

    def rows(self, group: str, i: int) -> slice:
        p = self.arr[f"{group}_ptr"]
        return slice(int(p[i]), int(p[i + 1]))


class Store:
    """A store opened for reading: its segments' arrays memory-mapped, its stays concatenated (``stays``; the global
    stay index is the position there). ``events(i)``, ``intervals(i)``, ``derived(i)``: one stay's arrays (views);
    ``channel_events(c)`` / ``derived_events(kind)``: one channel's events or one derived concept's rows of every stay,
    ordered by stay and time (the label engine's input; the orders are computed once per segment)."""

    def __init__(self, root: Path, segments: list[str] | None = None):
        import pandas as pd

        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.channels = Channels.load(self.root / "channels.json")
        names = segments or [s["name"] for s in self.manifest["segments"]]
        self.segments = []
        for n in names:
            d = self.root / n
            arr = {p.stem: np.load(p, mmap_mode="r") for p in sorted(d.glob("*.npy"))}
            self.segments.append(Segment(n, pd.read_parquet(d / "stays.parquet"), arr))
        frames = [s.stays.assign(segment=s.name, local=np.arange(len(s.stays))) for s in self.segments]
        self.stays = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self.weighted = self._apply_weights(names)
        self._seg = {s.name: s for s in self.segments}
        sizes = [len(s.stays) for s in self.segments]
        self.offset = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self._seg_of = np.repeat(np.arange(len(sizes)), sizes)
        self._local = np.concatenate([np.arange(n) for n in sizes]) if sizes else np.zeros(0, dtype=np.int64)
        self.los = self.stays["los_min"].to_numpy(np.int64) if len(self.stays) else np.zeros(0, dtype=np.int64)
        self.end_kind = self.stays["end_kind"].to_numpy() if len(self.stays) else np.zeros(0, dtype=object)
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.stays)

    def _apply_weights(self, names: list[str]) -> bool:
        """The pool's importance weights (``weights.parquet``, ``set_weights``) when they were computed for exactly
        these segments: each stay's ``weight`` and ``goal_met``, and split ``rejected`` for a twin that missed its
        value goal (the episode sampler never draws it)."""
        info = self.manifest.get("weights")
        path = self.root / "weights.parquet"
        if not info or sorted(info["segments"]) != sorted(names) or not path.exists() or not len(self.stays):
            return False
        import pandas as pd

        w = pd.read_parquet(path).set_index("stay_id")
        ids = self.stays["stay_id"]
        self.stays["weight"] = ids.map(w["weight"]).to_numpy(dtype=np.float64)
        self.stays["goal_met"] = ids.map(w["goal_met"]).to_numpy(dtype=object)
        rej = ids.map(w["rejected"]).fillna(False).to_numpy(dtype=bool)
        if "split" in self.stays:
            self.stays.loc[rej, "split"] = "rejected"
        return True

    def _at(self, i: int) -> tuple[Segment, int]:
        return self.segments[self._seg_of[i]], int(self._local[i])

    def _group(self, group: str, i: int) -> dict:
        seg, j = self._at(i)
        sl = seg.rows(group, j)
        return {k: seg.arr[f"{group}_{k}"][sl] for k, _ in ARRAYS[group]}

    def events(self, i: int) -> dict:
        return self._group("ev", i)

    def intervals(self, i: int) -> dict:
        return self._group("iv", i)

    def derived(self, i: int) -> dict:
        return self._group("dv", i)

    def _order(self, k: int, group: str, col: str) -> tuple:
        """Per segment (cached): the rows of ``group`` stably sorted by ``col`` (so each value's rows keep their stay
        and time order), the sorted values, and each row's stay (global index)."""
        key = (k, group, col)
        if key not in self._cache:
            seg = self.segments[k]
            vals = np.asarray(seg.arr[f"{group}_{col}"])
            order = np.argsort(vals, kind="stable")
            ptr = np.asarray(seg.arr[f"{group}_ptr"])
            stay = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr)) + self.offset[k]
            self._cache[key] = (order, vals[order], stay)
        return self._cache[key]

    def _select(self, group: str, col: str, value: int, fields: tuple) -> dict:
        out = {f: [] for f in ("stay", *fields)}
        for k, seg in enumerate(self.segments):
            if f"{group}x_ptr" in seg.arr:               # the key-major copy: one contiguous slice
                p = seg.arr[f"{group}x_ptr"]
                a, b = (int(p[value]), int(p[value + 1])) if 0 <= value < len(p) - 1 else (0, 0)
                out["stay"].append(np.asarray(seg.arr[f"{group}x_stay"][a:b], dtype=np.int64) + self.offset[k])
                for f in fields:
                    out[f].append(np.asarray(seg.arr[f"{group}x_{f}"][a:b]))
                continue
            order, sv, stay = self._order(k, group, col)
            idx = order[np.searchsorted(sv, value, "left"):np.searchsorted(sv, value, "right")]
            out["stay"].append(stay[idx])
            for f in fields:
                out[f].append(np.asarray(seg.arr[f"{group}_{f}"])[idx])
        return {f: np.concatenate(x) if x else np.zeros(0) for f, x in out.items()}

    def channel_events(self, cid: int) -> dict:
        """Every event of channel ``cid``: stay, t, v, bin, f (ordered by stay, then time)."""
        return self._select("ev", "c", cid, ("t", "v", "bin", "f"))

    def derived_events(self, kind: str) -> dict:
        """Every row of a derived concept (``DERIVED``): stay, s, e, v (ordered by stay, then start)."""
        return self._select("dv", "k", DK[kind], ("s", "e", "v"))


def set_weights(root: Path, stay_ids: list, weight, goal_met, rejected, info: dict) -> dict:
    """Record a pool's importance weights (``generate.supply``) for the store's current segments: ``weights.parquet``
    (stay_id, weight, goal_met, rejected) and its manifest entry (the segments it covers, ``info``, its SHA-256),
    both written atomically. A store opened with another set of segments ignores them."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    man_path = root / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    tmp = root / "weights.parquet.partial"
    pq.write_table(pa.table({"stay_id": [str(x) for x in stay_ids],
                             "weight": pa.array(np.asarray(weight, dtype=np.float64)),
                             "goal_met": pa.array([None if g is None else bool(g) for g in goal_met], pa.bool_()),
                             "rejected": pa.array(np.asarray(rejected, dtype=bool))}), tmp)
    os.replace(tmp, root / "weights.parquet")
    man["weights"] = {"segments": [s["name"] for s in man["segments"]], **info,
                      "sha256": _sha(root / "weights.parquet")}
    tmpm = root / "manifest.json.partial"
    tmpm.write_text(json.dumps(man, indent=1), encoding="utf-8")
    os.replace(tmpm, man_path)
    return man["weights"]


def channel_stats(store: Store, split: str = "train", max_values: int = 200_000, seed: int = 0) -> dict:
    """Per numeric channel, the median, 5th and 95th percentiles of its values among ``split`` stays (a uniform
    sample of at most ``max_values`` values per channel)."""
    rng = np.random.default_rng(seed)
    vals: dict = {}
    keep = set(np.flatnonzero((store.stays["split"] == split).to_numpy())) if "split" in store.stays else None
    for seg in store.segments:
        ptr = seg.arr["ev_ptr"]
        offs = store.stays.index[store.stays["segment"] == seg.name].to_numpy()
        idx = [j for j, g in enumerate(offs) if keep is None or g in keep]
        sel = np.concatenate([np.arange(ptr[j], ptr[j + 1]) for j in idx]) if idx else np.zeros(0, dtype=np.int64)
        c, x = np.asarray(seg.arr["ev_c"])[sel], np.asarray(seg.arr["ev_v"])[sel]
        ok = ~np.isnan(x)
        for cid in np.unique(c[ok]):
            vals.setdefault(int(cid), []).append(x[ok & (c == cid)])
    out = {}
    for cid, parts in vals.items():
        a = np.concatenate(parts)
        if len(a) > max_values:
            a = rng.choice(a, max_values, replace=False)
        q = np.quantile(a, [0.05, 0.5, 0.95])
        out[store.channels.names[cid]] = {"p05": float(q[0]), "median": float(q[1]), "p95": float(q[2]),
                                          "n": int(len(a))}
    return out


def acts(f: np.ndarray) -> np.ndarray:
    """An infusion's action from event flags: 0 none, else 1 + its position in ``channels.ACTIONS``."""
    return (np.asarray(f) >> 2) & 7


__all__ = ["ACTIONS", "DERIVED", "Store", "build", "channel_stats", "stay_arrays", "twin_split"]
