"""The twin supply (plan Phase F, "Twin supply"; study lead's Q&A 7-10; S1b): the generative model in the factory loop.

The production pool (millions of twins, generated on the RTX PRO 6000 server in the training phase) is drawn by a
``SupplyPlan`` that splits the generation budget across four mechanisms (study lead, Q&A 10), each a set of cells:

* ``base`` (55%): the flattened dataset mix (training shares to the power 0.5), the model completing everything else,
  with the accepted sampler (round 3's length and treatment raking, ``generate.calibrate``);
* ``strata`` (15%): header strata filled, dataset x sex x age band x unit type x admission type x diagnosis group
  (strata with at least ``strata_min`` training stays, uniformly within a dataset): the cell's twins fix those prefix
  and header fields, with a 5-year age token drawn within the band by its natural share;
* ``site`` (10%): eICU hospitals with at least ``site_min`` training stays, uniformly, by their site token;
* ``goal`` (20%): rare outcomes, an equal share each: ICU death, MCS and RRT (event goals: ``filter`` cells, plain
  draws of the dataset kept when their tokens carry the event, about ``1 / rate`` draws per kept twin; conditioning on
  the goal block held MUC32_main's twins open to the token budget, S1b smoke and ``tools/server/goal_diag.py``), AKI
  stage 3 and a first-day SOFA of 10 or more (value goals: ``condition`` cells, the goal block as conditioning).
  Every goal is checked on the twin's derived concepts, as the weights read it (an RRT in the ICU, the maximal KDIGO
  stage there); a twin that missed one is rejected.

Conditioned cells sample the model directly: raking a stratum's or an outcome's twins to its dataset's margins would
undo the conditioning.

**Weights.** Every twin carries an importance weight w(x) = p(x) / q(x) for the natural distribution p (the training
stays'), whichever cell drew it: the balance heuristic over the mixture (Veach & Guibas 1995; the deterministic
mixture weights of Owen & Zhou 2000). Each component fixes attributes a (the dataset; the stratum; the site; one goal),
so q_k(x) / p(x) = q_k(a) / p(a), and w(x) = 1 / sum over the cells c that could have drawn x of n_c / (N p(a_c)),
with n_c the cell's twins of N: only the natural marginals of the fixed attributes are needed (``marginals``,
aggregates of the training stays' heads beside the vocabulary). A goal block follows the prefix and header, which the
model draws as usual, so a conditioned goal cell's ratio is 1[x in c] / p(c | dataset, stratum of x): the goal's
natural rate in the twin's stratum, shrunk to its dataset's rate; a filter cell keeps natural draws, so its ratio is
1[x in c] / p(c | dataset). The base component bounds the weights (defensive importance
sampling, Hesterberg 1995), and a base-only plan gives the bank's weights p(d) / p_flat(d). The weights assume that,
given what a cell fixes, the model draws the rest as it would unconditioned: exact for the fields drawn after the
fixed ones; a field drawn before them (an eICU twin's site, the era, race and ethnicity before a fixed header field)
keeps its natural conditional on what precedes it. The S1b smoke measures how well the weights restore the natural
marginals.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SHARES = {"base": 0.55, "strata": 0.15, "site": 0.10, "goal": 0.20}        # study lead, Q&A 10
STRATUM = ("sex", "age_band", "unit_type", "admission_type", "adm_dx_group")
AGE_BANDS = {"18-45": ("18-20", "20-25", "25-30", "30-35", "35-40", "40-45"),
             "45-65": ("45-50", "50-55", "55-60", "60-65"),
             "65-80": ("65-70", "70-75", "75-80"),
             "80+": ("80-85", "85-90", "90+")}
GOAL_CELLS = (("icu_outcome", ("dead",)), ("mcs", ("yes",)), ("rrt", ("yes",)), ("kdigo_max", ("3+",)),
              ("sofa_1d", ("10-12", "12+")))
EVENT_GOALS = ("icu_outcome", "hosp_outcome", "imv", "vasopressor", "rrt", "mcs")
GOAL_FIELDS = ("icu_outcome", "imv", "vasopressor", "rrt", "mcs", "kdigo_max", "sofa_1d")
UNKNOWN = "unknown"


def coarse_age(label: str | None) -> str:
    for band, fine in AGE_BANDS.items():
        if label in fine:
            return band
    return label or UNKNOWN


def stratum_key(a: dict) -> str:
    return "|".join(str(a.get(k) or UNKNOWN) for k in STRATUM)


def _largest_remainder(p: np.ndarray, total: int) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if total <= 0 or p.sum() <= 0:
        return np.zeros(len(p), dtype=int)
    x = p / p.sum() * total
    q = np.floor(x).astype(int)
    q[np.argsort(-(x - q), kind="stable")[: total - q.sum()]] += 1
    return q


# -- the natural marginals ---------------------------------------------------------------------------------------
def head_labels(ids, v) -> dict:
    """The labels of a sequence's head: prefix and header fields, and its goal block's when it has one."""
    out: dict = {"goals": {}}
    for tok in ids:
        m = v.meta[int(tok)]
        cls = m.get("cls")
        if cls in ("prefix", "header", "goal"):
            _blk, f, label = v.tokens[int(tok)].split("//", 2)
            (out["goals"] if cls == "goal" else out)[f] = label
        elif cls not in (None, "special"):
            break
    out["age_band"] = coarse_age(out.get("age"))
    return out


def marginals(tokens: Path, datasets: list[str], split: str = "train", v=None) -> dict:
    """Counts of the training stays (``split``) the plan and the weights need: per dataset; per site; per (dataset,
    stratum), with its 5-year age tokens and its goal labels; per (dataset, goal field, label). Aggregates only."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from icutwin.tokenization.spec import load_spec
    from icutwin.tokenization.vocab import Vocab

    tokens = Path(tokens)
    v = v or Vocab.load(tokens / "vocab.json")
    spec = load_spec()
    head = 1 + len(spec["prefix"]) + len(spec["header"]) + len(spec["goal"]) + 2
    out: dict = {"split": split, "datasets": {}, "sites": {}, "strata": {}, "goals": {},
                 "vocab_sha256": hashlib.sha256((tokens / "vocab.json").read_bytes()).hexdigest()}
    for ds in datasets:
        path = tokens / ds / f"{split}.parquet"
        if not path.exists():
            continue
        heads = (h for batch in pq.ParquetFile(path).iter_batches(columns=["goal_ids"], batch_size=8192)
                 for h in pc.list_slice(batch.column(0), 0, head).to_pylist())       # streamed: heads only
        strata: dict = {}
        goals: dict = {}
        n = 0
        for h in heads:
            n += 1
            a = head_labels(h, v)
            out["sites"][a.get("site", UNKNOWN)] = out["sites"].get(a.get("site", UNKNOWN), 0) + 1
            s = strata.setdefault(stratum_key(a), {"n": 0, "age": {}, "goals": {}})
            s["n"] += 1
            s["age"][a.get("age", UNKNOWN)] = s["age"].get(a.get("age", UNKNOWN), 0) + 1
            for f in GOAL_FIELDS:
                lab = a["goals"].get(f, UNKNOWN)
                goals.setdefault(f, {})[lab] = goals.setdefault(f, {}).get(lab, 0) + 1
                sg = s["goals"].setdefault(f, {})
                sg[lab] = sg.get(lab, 0) + 1
        out["datasets"][ds] = n
        out["strata"][ds] = strata
        out["goals"][ds] = goals
    return out


# -- the plan ----------------------------------------------------------------------------------------------------
@dataclass
class Cell:
    name: str
    mechanism: str                    # base | strata | site | goal
    dataset: str
    quota: int
    parts: list = field(default_factory=list)   # [{"site", "prefix", "header", "goals", "n"}]: the requests
    stratum: str | None = None        # strata: its key
    site: str | None = None           # site: its token label
    goal: tuple | None = None         # goal: (field, labels)
    mode: str = "condition"           # goal: "condition" (the goal block) or "filter" (plain draws kept on the event)


@dataclass
class SupplyPlan:
    name: str
    total: int
    shares: dict
    cells: list
    flat: dict                        # dataset -> its share in the flattened mix
    natural: dict                     # dataset -> its share among the training stays
    settings: dict = field(default_factory=dict)

    @classmethod
    def build(cls, marg: dict, total: int, name: str = "supply", shares: dict | None = None, power: float = 0.5,
              strata_min: int = 20, max_strata: int | None = None, site_min: int = 200, goal_min: int = 20,
              goal_cells: tuple = GOAL_CELLS, filter_events: bool = True, filter_margin: float = 1.25,
              max_filter_draws: int = 40) -> SupplyPlan:
        """The plan for ``total`` twins (module docstring). ``max_strata``: at most that many strata per dataset, the
        rarest eligible first (a smoke's budget); a mechanism with nothing to draw gives its budget to the base."""
        shares = dict(shares or SHARES)
        names = sorted(marg["datasets"])
        n_d = np.array([marg["datasets"][d] for d in names], dtype=np.float64)
        p_nat = n_d / n_d.sum()
        p_flat = p_nat ** power / (p_nat ** power).sum()
        budget = dict(zip(shares, _largest_remainder(np.array(list(shares.values())), total).tolist(), strict=True))
        cells: list[Cell] = []

        # header strata: per dataset by the flattened mix; uniformly over its eligible strata
        st_cells = []
        for d, q in zip(names, _largest_remainder(p_flat, budget.get("strata", 0)), strict=True):
            elig = sorted(((s["n"], k) for k, s in marg["strata"].get(d, {}).items()
                           if s["n"] >= strata_min and k.split("|")[1] in AGE_BANDS), key=lambda x: (x[0], x[1]))
            if max_strata is not None:
                elig = elig[:max_strata]
            if not elig or q == 0:
                continue
            for (_n, k), m in zip(elig, _largest_remainder(np.ones(len(elig)), int(q)), strict=True):
                if m == 0:
                    continue
                s = marg["strata"][d][k]
                sex, band, unit, adm, dx = k.split("|")
                fine = [a for a in AGE_BANDS[band] if s["age"].get(a, 0) > 0]
                parts = [{"site": None, "prefix": {"sex": sex, "age": a}, "goals": {},
                          "header": {"unit_type": unit, "admission_type": adm, "adm_dx_group": dx}, "n": int(c)}
                         for a, c in zip(fine, _largest_remainder(np.array([s["age"][a] for a in fine]), int(m)),
                                         strict=True) if c > 0]
                st_cells.append(Cell(f"strata/{d}/{k}", "strata", d, int(m), parts, stratum=k))
        # eICU sites, uniformly over those with enough training stays
        sites = sorted(s for s, n in marg["sites"].items() if s.startswith("eicu_") and n >= site_min)
        si_cells = [Cell(f"site/eicu/{s}", "site", "eicu", int(m), [{"site": s, "prefix": {}, "header": {},
                                                                     "goals": {}, "n": int(m)}], site=s)
                    for s, m in zip(sites, _largest_remainder(np.ones(len(sites)), budget.get("site", 0)), strict=True)
                    if m > 0] if "eicu" in names else []
        # goals: an equal share per goal; within one, the datasets with enough natural cases by the flattened mix
        go_cells = []
        per_goal = _largest_remainder(np.ones(len(goal_cells)), budget.get("goal", 0))
        for (f, labels), qg in zip(goal_cells, per_goal, strict=True):
            have = [d for d in names
                    if sum(marg["goals"].get(d, {}).get(f, {}).get(lab, 0) for lab in labels) >= goal_min]
            if not have or qg == 0:
                continue
            pf = np.array([p_flat[names.index(d)] for d in have])
            for d, m in zip(have, _largest_remainder(pf, int(qg)), strict=True):
                if m == 0:
                    continue
                g = marg["goals"][d][f]
                labs = [lab for lab in labels if g.get(lab, 0) > 0]
                if filter_events and f in EVENT_GOALS and len(labs) == 1:      # plain draws, kept on the event
                    rate = g[labs[0]] / sum(g.values())
                    draws = int(min(np.ceil(m / rate * filter_margin), max_filter_draws * m))
                    parts = [{"site": None, "prefix": {}, "header": {}, "goals": {}, "n": draws}]
                    mode = "filter"
                else:
                    parts = [{"site": None, "prefix": {}, "header": {}, "goals": {f: lab}, "n": int(c)}
                             for lab, c in zip(labs, _largest_remainder(np.array([g[lab] for lab in labs]), int(m)),
                                               strict=True) if c > 0]
                    mode = "condition"
                go_cells.append(Cell(f"goal/{d}/{f}={'+'.join(labels)}", "goal", d, int(m), parts,
                                     goal=(f, tuple(labels)), mode=mode))
        used = sum(c.quota for c in st_cells + si_cells + go_cells)
        for d, q in zip(names, _largest_remainder(p_flat, total - used), strict=True):   # the base, and what is left
            if q > 0:
                cells.append(Cell(f"base/{d}", "base", d, int(q),
                                  [{"site": None, "prefix": {}, "header": {}, "goals": {}, "n": int(q)}]))
        cells += st_cells + si_cells + go_cells
        settings = {"power": power, "strata_min": strata_min, "max_strata": max_strata, "site_min": site_min,
                    "goal_min": goal_min, "goal_cells": [[f, list(lab)] for f, lab in goal_cells],
                    "filter_events": filter_events, "filter_margin": filter_margin,
                    "max_filter_draws": max_filter_draws,
                    "marginals_vocab_sha256": marg.get("vocab_sha256")}
        return cls(name, int(total), shares, cells, dict(zip(names, p_flat.tolist(), strict=True)),
                   dict(zip(names, p_nat.tolist(), strict=True)), settings)

    # -- requests ------------------------------------------------------------------------------------------------
    def requests(self) -> list[tuple[int, dict]]:
        """One (cell index, request) per draw, in the plan's order (base cells first): the request as keyword
        arguments of ``engine.Request`` (a filter cell's draws are plain; its twins are those that keep the event)."""
        out = []
        for k, c in enumerate(self.cells):
            for p in c.parts:
                r = {"dataset": c.dataset, "site": p["site"], "prefix": dict(p["prefix"]),
                     "header": dict(p["header"]), "goals": dict(p["goals"])}
                out += [(k, r)] * int(p["n"])
        return out

    def counts(self) -> dict:
        return {c.name: c.quota for c in self.cells}

    def mechanisms(self) -> dict:
        out: dict = {}
        for c in self.cells:
            out[c.mechanism] = out.get(c.mechanism, 0) + c.quota
        return out

    # -- weights -------------------------------------------------------------------------------------------------
    def weights(self, attrs: list[dict], marg: dict, counts: dict | None = None, shrink: float = 100.0) -> np.ndarray:
        """Importance weights (module docstring), normalized to mean 1, of twins with realized attributes ``attrs``
        (``dataset``, ``site``, ``stratum``, ``goals``: field -> label); ``counts``: twins per cell (default: the
        quotas); ``shrink``: the prior strength pulling a stratum's goal rate to its dataset's."""
        counts = counts or self.counts()
        n_tot = float(sum(counts.values()))
        n_train = float(sum(marg["datasets"].values()))
        p_d = {d: n / n_train for d, n in marg["datasets"].items()}
        base = {c.dataset: counts.get(c.name, 0) for c in self.cells if c.mechanism == "base"}
        strata = {(c.dataset, c.stratum): counts.get(c.name, 0) for c in self.cells if c.mechanism == "strata"}
        sites = {c.site: counts.get(c.name, 0) for c in self.cells if c.mechanism == "site"}
        goals: dict = {}
        for c in self.cells:
            if c.mechanism == "goal":
                goals.setdefault(c.dataset, []).append((c.goal[0], set(c.goal[1]), counts.get(c.name, 0),
                                                        c.mode))
        w = np.zeros(len(attrs))
        for i, a in enumerate(attrs):
            d, s = a["dataset"], a["stratum"]
            pd_ = p_d.get(d, 0.0)
            if pd_ <= 0:
                continue
            nd = marg["datasets"][d]
            sm = marg["strata"].get(d, {}).get(s)
            den = base.get(d, 0) / n_tot / pd_
            if (d, s) in strata and sm:
                den += strata[(d, s)] / n_tot / (pd_ * sm["n"] / nd)
            if d == "eicu" and a.get("site") in sites:
                den += sites[a["site"]] / n_tot / (marg["sites"][a["site"]] / n_train)
            for f, labels, n_c, mode in goals.get(d, ()):
                if a["goals"].get(f) not in labels or n_c == 0:
                    continue
                rate_d = sum(marg["goals"][d].get(f, {}).get(lab, 0) for lab in labels) / nd
                if sm and mode != "filter":                # a conditioned cell keeps the model's own header
                    pos = sum(sm["goals"].get(f, {}).get(lab, 0) for lab in labels)
                    rate = (pos + shrink * rate_d) / (sm["n"] + shrink)
                else:
                    rate = rate_d
                den += n_c / n_tot / (pd_ * max(rate, 1e-9))
            w[i] = 1.0 / den if den > 0 else 0.0
        return w * (np.count_nonzero(w) / w.sum()) if w.sum() > 0 else w

    # -- storage -------------------------------------------------------------------------------------------------
    def to_json(self) -> dict:
        return {"name": self.name, "total": self.total, "shares": self.shares, "flat": self.flat,
                "natural": self.natural, "settings": self.settings, "mechanisms": self.mechanisms(),
                "cells": [asdict(c) for c in self.cells]}

    @classmethod
    def from_json(cls, d: dict) -> SupplyPlan:
        cells = [Cell(**{**c, "goal": (c["goal"][0], tuple(c["goal"][1])) if c.get("goal") else None})
                 for c in d["cells"]]
        return cls(d["name"], d["total"], d["shares"], cells, d["flat"], d["natural"], d.get("settings", {}))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SupplyPlan:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


# -- realized attributes -----------------------------------------------------------------------------------------
def store_attributes(store) -> list[dict]:
    """Each stay's realized attributes from a stay store (``factory.store``): dataset, site, stratum (prefix and
    header labels as decoded) and the goal fields' labels from its end and derived concepts, as the goal block reads
    them from real stays (``tokenization.stays``): ICU outcome; IMV, vasopressor and RRT episodes in the ICU; an MCS
    record in the ICU; the maximal KDIGO stage in the ICU; the first-day SOFA."""
    from icutwin.tokenization.spec import band_labels, band_of, load_spec

    spec = load_spec()
    gspec = {f["field"]: f for f in spec["goal"]}
    age_bands = next(f["bands"] for f in spec["prefix"] if f["field"] == "age")
    st = store.stays
    n = len(st)
    los = store.los

    def band(field_, x):
        b = gspec[field_]["bands"]
        return band_labels(b)[band_of(float(x), b)]

    def episode(kind):
        r = store.derived_events(f"ep:{kind}")
        hit = np.zeros(n, dtype=bool)
        ok = (r["s"] < los[r["stay"].astype(np.int64)]) & (r["e"] > 0)
        hit[r["stay"][ok].astype(np.int64)] = True
        return hit

    eps = {k: episode(k) for k in ("imv", "vasopressor", "rrt")}
    mcs = np.zeros(n, dtype=bool)
    for cid, name in enumerate(store.channels.names):
        if name.startswith("MCS//"):
            e = store.channel_events(cid)
            s = e["stay"].astype(np.int64)
            ok = (e["t"] >= 0) & (e["t"] <= los[s])
            mcs[s[ok]] = True
    kd = np.full(n, np.nan)
    r = store.derived_events("kdigo")
    s = r["stay"].astype(np.int64)
    ok = (r["s"] >= 0) & (r["s"] <= los[s]) & np.isfinite(r["v"])
    np.fmax.at(kd, s[ok], r["v"][ok])
    sofa = np.full(n, np.nan)
    r = store.derived_events("sofa24:sofa_total")
    first = np.unique(r["stay"].astype(np.int64), return_index=True)
    sofa[first[0]] = r["v"][first[1]]
    out = []
    cols = {c: st[c].tolist() if c in st else [None] * n for c in ("sex", "age", "unit_type", "admission_type",
                                                                   "adm_dx_group", "cell", "conditions")}
    for i in range(n):
        age = cols["age"][i]
        age5 = band_labels(age_bands)[band_of(float(age), age_bands)] if age is not None and np.isfinite(age) else None
        a = {"sex": cols["sex"][i], "age": age5, "age_band": coarse_age(age5), "unit_type": cols["unit_type"][i],
             "admission_type": cols["admission_type"][i], "adm_dx_group": cols["adm_dx_group"][i]}
        ek = store.end_kind[i]
        goals = {"icu_outcome": ek if ek in ("alive", "dead") else UNKNOWN,
                 **{k: "yes" if eps[k][i] else "no" for k in eps}, "mcs": "yes" if mcs[i] else "no",
                 "kdigo_max": band("kdigo_max", kd[i]) if np.isfinite(kd[i]) else UNKNOWN,
                 "sofa_1d": band("sofa_1d", sofa[i]) if np.isfinite(sofa[i]) else UNKNOWN}
        out.append({"stay_id": st["stay_id"].iat[i], "dataset": st["dataset"].iat[i], "site": st["site"].iat[i],
                    "stratum": stratum_key(a), **{k: a[k] for k in ("sex", "age", "age_band")}, "goals": goals,
                    "cell": cols["cell"][i], "conditions": cols["conditions"][i]})
    return out


def goal_check(attrs: list[dict]) -> np.ndarray:
    """Per stay: True when it met every goal its request fixed (``conditions``), False when it missed one, None
    without goals."""
    out = np.full(len(attrs), None, dtype=object)
    for i, a in enumerate(attrs):
        c = json.loads(a["conditions"]) if a.get("conditions") else {}
        req = {**(c.get("goals") or {}), **(c.get("filter") or {})}
        if req:
            out[i] = all(a["goals"].get(f) == lab for f, lab in req.items())
    return out


# -- production --------------------------------------------------------------------------------------------------
def shards(plan: SupplyPlan, shard_size: int) -> list[dict]:
    """The plan's shards: base twins per dataset (drawn with the accepted sampler's selection), then the conditioned
    cells' twins in plan order; each shard a block of the plan's twins with the index of its first."""
    reqs = plan.requests()
    out, i = [], 0
    streams: list[tuple[str, list]] = []
    for c in plan.cells:
        if c.mechanism == "base":
            streams.append((f"base_{c.dataset}", [r for r in reqs if plan.cells[r[0]].name == c.name]))
    streams.append(("cond", [r for r in reqs if plan.cells[r[0]].mechanism != "base"]))
    for stream, rs in streams:
        for k in range(0, len(rs), shard_size):
            block = rs[k:k + shard_size]
            out.append({"name": f"{stream}_{k // shard_size:05d}", "stream": stream, "first": i,
                        "requests": block, "select": stream.startswith("base_")})
            i += len(block)
    return out


_OBLIGATIONS: dict = {}


def event_goal_met(ids, v, goals: dict) -> bool | None:
    """Whether a twin's tokens keep its event goals as the obligation registers read them (``engine.obligations``:
    a treatment's token, an ICU outcome's discharge token, none of the forbidden ones); None without event goals.
    The derived concepts' reading (in the ICU) is ``goal_check``'s."""
    from icutwin.generate.engine import obligations

    ev = tuple(sorted((f, lab) for f, lab in goals.items() if f in EVENT_GOALS))
    if not ev:
        return None
    key = (id(v), ev)
    if key not in _OBLIGATIONS:
        req, req_end, forbid = obligations(v, dict(ev))
        _OBLIGATIONS[key] = ([set(r) for r in req], [set(r) for r in req_end], set(forbid))
    req, req_end, forbid = _OBLIGATIONS[key]
    have = {int(t) for t in ids}
    return all(have & r for r in req) and all(have & r for r in req_end) and not (have & forbid)


def produce(sample, plan: SupplyPlan, out_dir: Path, *, seed: int, shard_size: int, event: np.ndarray, gate: dict,
            model: dict, v, part: int = 0, parts: int = 1, novelty=None, oversample: int = 1, select=None,
            request=None) -> dict:
    """Generate this part's shards of ``plan`` into ``out_dir`` (shards ``k`` with ``k % parts == part``; one
    directory and manifest per part, as the bank): ``sample(requests, seeds)`` returns token sequences (a twin with a
    goal block keeps its own tokens: the block is removed, its goals recorded in ``conditions``); base shards draw
    ``oversample`` times their twins and keep ``select(seqs, dataset, n, rng)``'s (round 3's raking); then exact and
    near duplicates and the novelty filter (``bank.Novelty``) as in the bank. Each kept twin records its cell, its
    request (``conditions``, JSON) and, for event goals, whether its tokens keep them (``goal_met``). Shards are
    written atomically and the manifest after them, so a run resumes at the first shard it lacks."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from icutwin.generate import bank
    from icutwin.generate.engine import strip_goal
    from icutwin.generate.producer import (
        SEED_STRIDE,
        GateNotPassed,
        Shard,
        _atomic_write,
        _load_manifest,
    )

    if gate.get("gate") != "PASS":
        raise GateNotPassed("produce needs a passed M7 gate (producer.require_gate)")
    if request is None:
        from icutwin.generate.engine import Request

        def request(r):
            return Request(**r)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan_sha = hashlib.sha256(json.dumps(plan.to_json(), sort_keys=True).encode()).hexdigest()
    man = _load_manifest(out)
    if man and (man.get("seed") != seed or man.get("plan_sha256") != plan_sha or man.get("shard_size") != shard_size
                or man.get("oversample") != oversample or man.get("part") != [part, parts]):
        raise ValueError("an existing supply part with another plan, seed, shard size, oversampling or partition")
    man = man or {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "plan": plan.name, "plan_sha256": plan_sha,
                  "seed": seed, "shard_size": shard_size, "oversample": oversample, "part": [part, parts],
                  "model": model, "gate": gate, "shards": []}
    hashes_path = out / "hashes.txt"
    seen = set(hashes_path.read_text().split()) if hashes_path.exists() else set()
    done = {s["name"] for s in man["shards"]}
    todo = [s for k, s in enumerate(shards(plan, shard_size)) if k % parts == part]
    for sh in todo:
        if sh["name"] in done:
            continue
        k_over = oversample if sh["select"] else 1
        reqs = [r for _c, r in sh["requests"] for _ in range(k_over)]
        cells = [c for c, _r in sh["requests"] for _ in range(k_over)]
        seeds = [SEED_STRIDE * seed + sh["first"] * oversample + i for i in range(len(reqs))]
        raw = sample([request(r) for r in reqs], seeds)
        seqs, conds = [], []
        for x, r, c in zip(raw, reqs, cells, strict=True):
            ids, _drawn = strip_goal([int(t) for t in x], v)
            seqs.append(np.asarray(ids, dtype=np.int32))
            cell = plan.cells[c]
            conds.append({**r, "filter": {cell.goal[0]: cell.goal[1][0]}} if cell.mode == "filter" else r)
        drawn = len(seqs)
        dropped: dict = {}
        keep_f = [plan.cells[c].mode != "filter" or bool(event_goal_met(s, v, cd["filter"]))
                  for s, c, cd in zip(seqs, cells, conds, strict=True)]
        if not all(keep_f):                       # a filter cell keeps the draws whose tokens carry its event
            dropped["filtered_out"] = int(len(keep_f) - sum(keep_f))
            idx_f = [i for i, k in enumerate(keep_f) if k]
            seqs, seeds, conds, cells = ([seqs[i] for i in idx_f], [seeds[i] for i in idx_f],
                                         [conds[i] for i in idx_f], [cells[i] for i in idx_f])
        if sh["select"] and select is not None:
            n = len(sh["requests"])
            pick = sorted(int(i) for i in select(seqs, sh["requests"][0][1]["dataset"], n,
                                                 np.random.default_rng(seeds[0])))
            dropped["not_selected"] = drawn - len(pick)
            seqs, seeds, conds, cells = ([seqs[i] for i in pick], [seeds[i] for i in pick],
                                         [conds[i] for i in pick], [cells[i] for i in pick])
        keep = np.ones(len(seqs), dtype=bool)
        hs = [bank.seq_hash(s) for s in seqs]
        dup = np.array([h in seen for h in hs], dtype=bool) | bank.exact_duplicates(seqs)
        dropped["exact_duplicate"] = int(dup.sum())
        keep &= ~dup
        near = bank.near_duplicates(seqs, event) & keep
        dropped["near_duplicate"] = int(near.sum())
        keep &= ~near
        if novelty is not None and keep.any():
            idx = np.flatnonzero(keep)
            novel = novelty([seqs[i] for i in idx], [conds[i]["dataset"] for i in idx])
            dropped["not_novel"] = int((~novel).sum())
            keep[idx[~novel]] = False
        kept = np.flatnonzero(keep)
        met = [event_goal_met(seqs[i], v, {**conds[i]["goals"], **conds[i].get("filter", {})}) for i in kept]
        table = pa.table({"twin": [f"{plan.name}_{sh['name']}_{i:05d}" for i in kept],
                          "dataset": [conds[i]["dataset"] for i in kept],
                          "seed": pa.array([seeds[i] for i in kept], pa.int64()),
                          "cell": [plan.cells[cells[i]].name for i in kept],
                          "conditions": [json.dumps(conds[i], sort_keys=True) for i in kept],
                          "goal_met": pa.array(met, pa.bool_()),
                          "weight": pa.array([float("nan")] * len(kept), pa.float64()),
                          "ids": pa.array([seqs[i].tolist() for i in kept], type=pa.list_(pa.int32()))})
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        data = sink.getvalue().to_pybytes()
        _atomic_write(out / f"{sh['name']}.parquet", data)
        seen.update(hs[i] for i in kept)
        _atomic_write(hashes_path, "\n".join(sorted(seen)).encode())
        entry = Shard(sh["name"], sh["stream"], sh["first"], drawn, len(kept), dropped,
                      hashlib.sha256(data).hexdigest()).__dict__
        entry["event_goals_unmet"] = sum(m is False for m in met)       # kept: the weights step rejects them
        entry["cells"] = {}
        for i in kept:
            nm = plan.cells[cells[i]].name
            entry["cells"][nm] = entry["cells"].get(nm, 0) + 1
        man["shards"].append(entry)
        _atomic_write(out / "manifest.json", json.dumps(man, indent=1).encode())
    man["complete"] = all(s["name"] in {x["name"] for x in man["shards"]} for s in todo)
    _atomic_write(out / "manifest.json", json.dumps(man, indent=1).encode())
    return man
