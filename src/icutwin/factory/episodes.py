"""Trajectory episodes (plan Phase F, Factory design 5-6; "Task formats"): recipes sampled per global episode index,
materialized from the stay store.

A recipe (a small JSON-able dict) fixes everything an episode is: the program (canonical spec, name, family), the
site, the cadence, the context patients and the policy that chose them, the query patients and their prediction times,
whether the descriptor is shown, the anonymization seed, the stage and its budgets, and provenance.
``Sampler.recipe(i)`` draws it from ``np.random.default_rng([seed, i])``: the same index gives the same recipe whatever
the worker. The materializer (``Sampler.materialize``) builds the arrays of "Task formats":
* ``context``: N other patients, each a complete trajectory (``factory.trajectories``) with the program's label
  timeline at every cadence time of the stay;
* ``query``: Q patients, each its own trajectory up to its last prediction time t_K, its prediction times, the targets
  there, and its own label timeline at the cadence times up to t_K (``own``; each label visible from its
  ``t_resolved``);
* ``task``: the program, its descriptor (or None when hidden), the label type and encoding, the site, the stage.

Draws (``configs/factory/stages.yaml``; the study lead's Q&A decisions): the family group by the study lead's label mix,
a program of it, the site (a dataset of the store, by its share of eligible stays), the cadence (monitoring-weighted),
the context size (log-uniform over the stage's range; 5% none where the descriptor is shown), the context policy
(random, outcome-stratified, retrieved by similarity to the first query), 1-8 queries with up to the stage's times
each. An episode needs positive and negative context trajectories (binary), two classes (ordinal) or values
(regression), else the program is drawn again; but with the descriptor shown (the bedside regime: the clinician names
the outcome) a natural (random or retrieved) context of an event task is taken as drawn (``as_drawn``), a case or not,
so the base rate it shows is its site's, a rare outcome's often none. The task block declares how the context was
chosen (``selection``: the policy, the odds of a positive stay's selection against another's relative to the
queries', and the positive trajectories it guarantees): the model reads base rates from its context, so an
outcome-stratified context's enrichment must be visible rather than learned from the twins' own prevalences.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from icutwin.factory import labels as fl
from icutwin.factory import processes as fp
from icutwin.factory import programs as fpr
from icutwin.factory import sites as fsi
from icutwin.factory import trajectories as ftr

STAGES_YAML = Path(__file__).resolve().parents[1] / "configs" / "factory" / "stages.yaml"
VERSION = "phase_f.4"
LAB_TRIGGERS = ("LAB//creatinine//mg/dl//bmp", "LAB//lactate//mmol/l//misc", "LAB//potassium//mmol/l//bmp")
RETRIEVAL = ("VITAL//heart_rate", "VITAL//map//arterial", "VITAL//map//cuff", "VITAL//spo2",
             "VITAL//respiratory_rate", "VITAL//temp_c", "LAB//creatinine//mg/dl//bmp", "LAB//lactate//mmol/l//misc")


class NoEpisode(RuntimeError):
    """No episode for this index within the draws allowed (deterministic: the same index never yields one)."""


def load_config(path: Path = STAGES_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _pick(rng, shares: dict):
    keys = list(shares)
    p = np.asarray([shares[k] for k in keys], dtype=np.float64)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def case_odds(n: int, pos: int, size: int, k: int, cap: float = 1000.0) -> float:
    """The odds of a positive stay's selection against another's when ``k`` of ``size`` context stays are drawn from
    the ``pos`` positive stays among ``n`` candidates and the rest from the remaining stays (by counts; 1: as a
    natural context, capped at ``cap``): the case-control sampling ratio a model needs to read the base rate behind
    an outcome-stratified context."""
    if k <= 0 or pos <= 0 or size >= n:
        return 1.0
    other = (size - k) / (n - k)
    if other <= 0:
        return cap
    return float(min((k / pos + (1 - k / pos) * other) / other, cap))


class Sampler:
    """Episodes of one store. ``programs``: the label programs to draw from (by family; the catalog by default);
    ``split``: the stays' split episodes use (twins: ``train`` or ``val``); ``support``: per program hash, the sites
    that carry it and how often (``pool.load_support``: valid and positive stays on the pool's panel)."""

    def __init__(self, store, v, stage: int = 1, seed: int | None = None, programs: list | None = None,
                 split: str | None = "train", config: dict | None = None, weights=None, targets: dict | None = None,
                 filter_on: bool = True, support: dict | None = None):
        self.store, self.v = store, v
        self.cfg = config or load_config()
        self.stage, self.budget = stage, self.cfg["stages"][stage]
        self.seed = self.cfg["seed"] if seed is None else seed
        self.ev = fpr.Evaluator(store)
        self.reader = ftr.Reader(store, v)
        progs = programs if programs is not None else fpr.catalog(store.channels)
        allowed = set(self.budget["labels"])
        weights = np.ones(len(progs)) if weights is None else np.asarray(weights, dtype=np.float64)
        self.by_family: dict = {}
        self.weights: dict = {}
        fmax = self.budget.get("forecast_max_min", 10080)
        for p, w in zip(progs, weights, strict=True):
            if p.label == "forecast" and int(p.target[2][-1]) > fmax:
                continue
            if p.label in allowed and w > 0:
                self.by_family.setdefault(p.family, []).append(p)
                self.weights.setdefault(p.family, []).append(w)
        self.family_mass = {f: float(np.sum(w)) for f, w in self.weights.items()}   # a group's families by mass
        self.weights = {f: np.asarray(w) / np.sum(w) for f, w in self.weights.items()}
        self.targets = targets
        self.filter = filter_on
        if targets is not None:                   # fresh functional heads make the FH family available
            self.by_family.setdefault("FH", [])
        groups = {g: dict(x) for g, x in self.cfg["families"].items()
                  if any(f in self.by_family for f in x["families"])}
        self.groups = groups
        st = store.stays
        ok = np.ones(len(st), dtype=bool) if split is None or "split" not in st else (st["split"] == split).to_numpy()
        self.pool = np.flatnonzero(ok)
        if not len(self.pool):
            raise ValueError(f"no stays of split {split!r} in the store")
        self.datasets = st["dataset"].to_numpy()
        w = st["weight"].to_numpy(dtype=np.float64) if "weight" in st else np.ones(len(st))
        self.w = np.where(np.isfinite(w) & (w > 0), w, 1.0)     # importance weights (generate.supply)
        self.index = {sid: i for i, sid in enumerate(st["stay_id"].tolist())}
        self.site_names, self.site_counts = np.unique(self.datasets[self.pool], return_counts=True)
        self.by_site = {s: self.pool[self.datasets[self.pool] == s] for s in self.site_names}
        support = dict(support or {})
        self.site_factors = support.pop("_site_factors", None) or {}   # pool.load: episode use at the pool's shares
        self.support = support
        self.stats: dict = {}
        self.attempt_log: list | None = None    # [program hash, accepted] per attempt when a list (pool.acceptance)

    def _stat(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

    # -- prediction times --------------------------------------------------------------------------------------
    def cadence_rows(self, cadence: tuple, stays) -> fl.Rows:
        kind = cadence[0]
        if kind == "rolling":
            return fl.rolling(self.store, cadence[1], stays)
        if kind == "landmark":
            return fl.landmark(self.store, cadence[1], stays)
        if kind == "ends":                                   # extubation: the end of an IMV episode
            r = fl.triggers(self.store, self.ev.process(("event", cadence[1])), "ends")
        elif kind == "onsets":
            r = fl.triggers(self.store, self.ev.process(("event", cadence[1])), "onsets")
        elif kind == "lab":
            evs = self.ev.events(cadence[1])
            keep = (evs["t"] >= 0) & (evs["t"] < self.store.los[evs["stay"].astype(np.int64)])
            r = fl.Rows(evs["stay"][keep].astype(np.int64), evs["t"][keep].astype(np.int64))
        else:
            raise ValueError(cadence)
        m = np.isin(r.stay, np.asarray(stays))
        return r.take(np.flatnonzero(m))

    def _cadence(self, rng, prog) -> tuple:
        if prog.family == "F5" and "imv" in json.dumps(prog.target) and prog.name.startswith("extubation"):
            return ("ends", "imv")
        if prog.label == "hosp_death":
            return ("landmark", int(rng.choice(self.cfg["landmarks"])))
        c = _pick(rng, self.cfg["cadence"])
        if c == "hourly":
            return ("rolling", 60)
        if c == "six_hourly":
            return ("rolling", 360)
        if c == "daily":
            return ("rolling", 1440)
        if c == "landmark":
            return ("landmark", int(rng.choice(self.cfg["landmarks"])))
        trig = str(rng.choice(self.cfg["event_triggers"]))
        ch = self.store.channels
        labs = [x for x in LAB_TRIGGERS if x in ch.index] or [n for n, g in zip(ch.names, ch.group) if g == "labs"][:3]
        if trig == "vasopressor_start" and "MED_CON//norepinephrine//mcg/kg/min//iv" in ch.index:
            return ("onsets", "vasopressor")
        if labs:                                  # a store without clinical labs (generic) uses its own lab-like ones
            return ("lab", str(rng.choice(labs)))
        return ("rolling", 60)

    # -- sites and cohorts --------------------------------------------------------------------------------------
    def site_labels(self, prog, cadence: tuple, site: str, stays=None):
        """(rows, labels) of a program at the cadence's times of ``stays`` (default: every stay of the site)."""
        rows = self.cadence_rows(cadence, self.by_site[site] if stays is None else stays)
        return rows, self.ev.evaluate(prog, rows)

    def _binary(self, prog) -> bool:
        return prog.label in ("binary", "hosp_death", "tte", "multihorizon")

    def _carried(self, prog) -> list:
        """The sites that carry a program (on the pool's panel: a positive stay for an event, valid labels else);
        every site when the pool does not say."""
        sup = self.support.get(prog.hash())
        if not sup:
            return list(self.site_names)
        need = 1 if self._binary(prog) else 2
        out = [s for s in self.site_names if (sup.get(s) or (0, 0))[1 if self._binary(prog) else 0] >= need]
        return out or list(self.site_names)

    def _rate(self, prog, site: str) -> float | None:
        """A program's share of positive stays at a site (on the pool's panel), for events."""
        s = (self.support.get(prog.hash()) or {}).get(site)
        return s[1] / s[0] if s and s[0] > 0 and self._binary(prog) else None

    def _cohort(self, rng, site: str, prog) -> np.ndarray:
        """The stays an episode's labels are computed on: a cohort of the site's stays by weight, ``cohort.base`` of
        them (at least four times the stage's patients), raised so the program's expected positive stays reach
        ``cohort.positives``, at most ``cohort.max``; the whole site when smaller. Its cost per episode stays the
        same whatever the pool's size."""
        stays = self.by_site[site]
        c = self.cfg.get("cohort") or {}
        m = max(int(c.get("base", 512)), 4 * (int(self.budget["context"][1]) + int(self.budget["queries"][1])))
        rate = self._rate(prog, site)
        if rate:
            m = max(m, int(np.ceil(c.get("positives", 16) / rate)))
        m = min(m, int(c.get("max", 4096)))
        if len(stays) <= m:
            return stays
        return np.sort(rng.choice(stays, size=m, replace=False, p=self._p(stays)))

    # -- recipes -----------------------------------------------------------------------------------------------
    def recipe(self, index: int) -> dict:
        rng = np.random.default_rng([self.seed, index])
        cfg = self.cfg
        group = _pick(rng, {g: x["share"] for g, x in self.groups.items()})      # retries stay in the stratum,
        fams = [f for f in self.groups[group]["families"] if f in self.by_family]   # so the mix is exact
        mass = np.array([self.family_mass.get(f, 0.0) for f in fams])
        fam = str(rng.choice(fams, p=mass / mass.sum())) if mass.sum() > 0 else str(rng.choice(fams))
        shown = bool(rng.random() < cfg["descriptor_shown"])   # once per index, as the group: a hidden task needs
        anon = int(rng.integers(1 << 31)) if rng.random() < cfg["anonymize"] else None   # cases, so its retries
        for attempt in range(cfg["max_tries"]):                                      # must not tilt the share
            if attempt and attempt % (cfg["max_tries"] // 2) == 0:     # a stratum the store cannot carry: another
                self.stats["stratum_fallback"] = self.stats.get("stratum_fallback", 0) + 1   # (recorded)
                fam = str(rng.choice([f for f in self.by_family if f != fam] or [fam]))
            prog = self._program(rng, fam)
            if self.attempt_log is not None:
                self.attempt_log.append([prog.hash(), False])
            cadence = self._cadence(rng, prog)
            names = self._carried(prog)                    # a site that cannot carry the program is not drawn
            counts = np.array([self.site_counts[list(self.site_names).index(s)] * self.site_factors.get(s, 1.0)
                               for s in names], dtype=np.float64)
            site = str(rng.choice(names, p=counts / counts.sum()))
            ramp = 0.5 if index < cfg.get("ramp_episodes", 0) else 1.0
            scale = 0.5 ** min(attempt // 4, 3)            # repeated difficulty misses: weaker perturbations
            variant = fsi.draw_site(rng, cfg, list(names), ramp, scale)
            realism = fsi.draw_realism(rng, cfg, ramp, scale)
            q_site = variant.get("query_site", site)
            c_site = variant.get("context_site", site)
            rows, b = self.site_labels(prog, cadence, q_site, self._cohort(rng, q_site, prog))
            c_rows, c_b = (rows, b) if c_site == q_site else                 self.site_labels(prog, cadence, c_site, self._cohort(rng, c_site, prog))
            got = self._compose(rng, prog, rows, b, c_rows, c_b, variant, shown)   # shown: a natural context as drawn
            if got is None:
                self._stat("retry")
                continue
            ctx, sel, queries = got
            rows, b = c_rows, c_b                          # the context's labels: the filter reads the context
            if not shown and len(ctx) == 0:
                ctx, sel = self._context(rng, prog, rows, b, [q["stay"] for q in queries], "random",
                                         int(self.budget["context"][0]))
                sel.update(min_cases=self._min_cases(prog, sel["policy"], len(ctx), sel["min_cases"]), as_drawn=False)
            verdict = (self._filter(prog, rows, b, ctx, queries, one_class=sel["as_drawn"]) if self.filter
                       else {"stage": 0})
            if not verdict.get("ok", True):
                self.stats["uninformative"] = self.stats.get("uninformative", 0) + 1
                continue
            loss = self._difficulty(rng, prog, rows, b, ctx, variant, realism)
            if loss is not None and loss > cfg["difficulty"]["cap"]:
                self.stats["too_difficult"] = self.stats.get("too_difficult", 0) + 1
                continue
            self.stats["accepted"] = self.stats.get("accepted", 0) + 1
            if self.attempt_log is not None:
                self.attempt_log[-1][1] = True
            return {"version": VERSION, "index": int(index), "seed": int(self.seed), "stage": self.stage,
                    "program": {"spec": prog.spec(), "name": prog.name, "family": prog.family,
                                "flags": list(prog.flags), "hash": prog.hash()},
                    "site": c_site, "query_site": q_site, "site_variant": variant, "realism": realism,
                    "difficulty_loss": None if loss is None else round(float(loss), 4),
                    "cadence": list(cadence), "policy": sel["policy"], "case_odds": sel["case_odds"],
                    "min_cases": sel["min_cases"], "as_drawn": sel["as_drawn"],
                    "context": [self._id(i) for i in ctx],
                    "queries": [{"stay": self._id(q["stay"]), "times": q["times"]} for q in queries],
                    "descriptor": shown, "anonymize": anon, "attempts": attempt + 1,
                    "filter": {k: v for k, v in verdict.items() if k != "ok"}}
        raise NoEpisode(f"no episode in {cfg['max_tries']} draws (index {index})")

    def _program(self, rng, fam: str):
        """A program of family ``fam`` by the pool's weights; for functional heads, a fresh one with probability
        ``fresh_share`` (the label space stays unbounded: the pool calibrates, it does not limit)."""
        have = self.by_family[fam]
        if fam == "FH" and self.targets is not None and (not have or rng.random() < self.cfg.get("fresh_share", 0.5)):
            from icutwin.factory import pool

            return pool.fh_candidates(self.store.channels, 1, int(rng.integers(1 << 31)), self.targets)[0]
        return have[int(rng.choice(len(have), p=self.weights[fam]))]

    def _id(self, i: int) -> str:
        return str(self.store.stays["stay_id"].iat[int(i)])

    def _weights(self, stays, rows, b, variant) -> np.ndarray | None:
        """Selection probabilities over ``stays``: the pool's importance weights (a twin enriched by the supply plan
        counts at its natural share), times a case-mix or label-shift site's tilt; None when uniform."""
        w = self.w[np.asarray(stays, dtype=np.int64)]
        if variant and variant.get("transform") == "case_mix":
            w = w * fsi.tilt(self.store, stays, variant, None)
        elif variant and variant["kind"] == "label_shift":
            pos = np.isin(stays, np.unique(rows.stay[b.valid & (b.y > 0)]))
            w = w * np.where(pos, variant["enrich"], 1.0)
        if not len(w) or w.sum() <= 0 or np.all(w == w[0]):
            return None
        return w / w.sum()

    def _p(self, stays) -> np.ndarray | None:
        w = self.w[np.asarray(stays, dtype=np.int64)]
        return None if not len(w) or np.all(w == w[0]) else w / w.sum()

    def _compose(self, rng, prog, rows, b, c_rows=None, c_b=None, variant=None, shown: bool = False):
        """(context stays, selection, queries) for one program, queries from ``rows`` and the context from ``c_rows``
        (another site for a cross-site episode), or None when the site cannot carry it. With the descriptor
        ``shown``, a natural context of an event task is taken as drawn in the configured share (``as_drawn``)."""
        cfg = self.cfg
        variant = variant or {"kind": "plain"}
        c_rows, c_b = (rows, b) if c_rows is None else (c_rows, c_b)
        ok = b.valid
        if ok.sum() < 2 or c_b.valid.sum() < 2:
            self._stat("compose:no_valid_labels")
            return None
        stays_ok = np.unique(rows.stay[ok])
        nq = int(rng.integers(self.budget["queries"][0], self.budget["queries"][1] + 1))
        lo, hi = self.budget["context"]
        nc = 0 if rng.random() < cfg["zero_context"] else int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
        c_ok = np.unique(c_rows.stay[c_b.valid])
        same = c_rows is rows
        if same and len(stays_ok) < nq + max(nc, 2):
            nq = max(1, min(nq, len(stays_ok) // 3))
            nc = min(nc, len(stays_ok) - nq) if nc else 0
            if len(stays_ok) < nq + max(nc, 2):
                self._stat("compose:too_few_stays")
                return None
        if not same and (len(stays_ok) < nq or len(c_ok) < max(nc, 2)):
            self._stat("compose:too_few_stays")
            return None
        qst = rng.choice(stays_ok, size=nq, replace=False, p=self._weights(stays_ok, rows, b, variant))
        queries = []
        for s in qst:
            k = np.flatnonzero((rows.stay == s) & ok)
            take = np.sort(rng.choice(k, size=min(len(k), self.budget["times"]), replace=False))
            queries.append({"stay": int(s), "times": [int(t) for t in rows.t[take]]})
        if nc == 0:
            return np.zeros(0, dtype=np.int64), {"policy": "none", "case_odds": None, "min_cases": None,
                                                 "as_drawn": False}, queries
        policy = _pick(rng, cfg["context_policy"])
        as_drawn = bool(shown and policy in ("random", "retrieved") and self._binary(prog)
                        and rng.random() < cfg.get("as_drawn", 0.0))
        ctx, sel = self._context(rng, prog, c_rows, c_b, qst, policy, nc, variant, as_drawn)
        if not (len(ctx) if as_drawn else self._informative(prog, c_rows, c_b, ctx, sel["policy"])):
            self._stat(f"compose:uninformative_{sel['policy']}")
            return None
        sel["min_cases"] = 0 if as_drawn else self._min_cases(prog, sel["policy"], len(ctx), sel["min_cases"])
        return ctx, {**sel, "as_drawn": as_drawn}, queries

    def _context(self, rng, prog, rows, b, queries, policy: str, nc: int, variant: dict | None = None,
                 as_drawn: bool = False):
        """(context stays, selection): the policy that chose them (a case-mix or label-shift site's tilt, which the
        queries share, counts as random), the odds of a positive stay's selection against another's relative to the
        queries' (``case_odds``; 1 unless outcome-stratified) and the positive stays it selected (``min_cases``;
        None unless outcome-stratified). A rare outcome's natural context is outcome-stratified (``rare_policy``)
        unless taken ``as_drawn``."""
        ok = b.valid
        cand = np.setdiff1d(np.unique(rows.stay[ok]), np.asarray(queries))
        sel = {"policy": policy, "case_odds": 1.0, "min_cases": None}
        if len(cand) == 0:
            return np.zeros(0, dtype=np.int64), sel
        nc = min(nc, len(cand))
        rare = self.cfg.get("rare_policy")
        if policy in ("random", "retrieved") and prog.label in ("binary", "hosp_death") and rare and not as_drawn:
            rare = rare if isinstance(rare, dict) else {"to": rare, "below": 1.0}
            pos = np.intersect1d(np.unique(rows.stay[ok & (b.y > 0)]), cand)
            if 0 < len(pos) and nc * len(pos) / len(cand) < rare["below"]:   # a rare outcome: its natural context
                policy = rare["to"]                                         # would show no case (recorded)
                self._stat("policy:rare_to_stratified")
        tilted = bool(variant) and (variant.get("transform") == "case_mix" or variant["kind"] == "label_shift")
        if tilted and policy != "retrieved":                 # a case-mix or label-shift site selects its patients
            w = self._weights(cand, rows, b, variant)
            return np.sort(rng.choice(cand, size=nc, replace=False, p=w).astype(np.int64)), {**sel, "policy": "random"}
        if policy == "stratified" and self._binary(prog):      # event tasks, times to event included
            pos = np.unique(rows.stay[ok & (b.y > 0)])
            pos = np.intersect1d(pos, cand)
            k = min(len(pos), max(self._min_positive("stratified"), nc // 4), nc)
            a = rng.choice(pos, size=k, replace=False, p=self._p(pos)) if k else np.zeros(0, dtype=np.int64)
            rest = np.setdiff1d(cand, a)
            bpart = rng.choice(rest, size=min(nc - k, len(rest)), replace=False, p=self._p(rest))
            return (np.sort(np.concatenate([a, bpart]).astype(np.int64)),
                    {"policy": "stratified", "case_odds": round(case_odds(len(cand), len(pos), nc, k), 3),
                     "min_cases": int(k)})
        if policy == "retrieved":
            return np.sort(self._retrieve(int(queries[0]), cand, nc)), {**sel, "policy": "retrieved"}
        return (np.sort(rng.choice(cand, size=nc, replace=False, p=self._p(cand)).astype(np.int64)),
                {**sel, "policy": "random"})

    def _retrieve(self, q: int, cand: np.ndarray, nc: int) -> np.ndarray:
        """The ``nc`` candidates nearest the query by its first-24-h summary (the last value of a few vitals and labs
        in the first day, standardized, and age): a similarity a bedside context bank can compute for a new patient."""
        def emb(stays):
            rows = fl.Rows(np.asarray(stays, dtype=np.int64), np.zeros(len(stays), dtype=np.int64))
            cols = []
            for name in RETRIEVAL:
                if name not in self.store.channels.index:
                    continue
                y, _c = fl.window_values(self.ev.events(name), rows, -1440, 1440, "last")
                cid = self.store.channels.index[name]
                med, sc = self.reader.median[cid], self.reader.scale[cid]
                cols.append(np.nan_to_num((y - med) / sc if np.isfinite(sc) else y))
            age = self.store.stays["age"].to_numpy(dtype=np.float64)[np.asarray(stays)]
            cols.append(np.nan_to_num((age - 60) / 15))
            return np.stack(cols, 1)
        d = np.linalg.norm(emb(cand) - emb([q]), axis=1)
        return cand[np.argsort(d, kind="stable")[:nc]].astype(np.int64)

    def _min_positive(self, policy: str) -> int:
        m = self.cfg["min_positive_context"]
        return int(m.get(policy, 1)) if isinstance(m, dict) else int(m)

    def _min_cases(self, prog, policy: str, n: int, selected: int | None = None) -> int | None:
        """The positive trajectories a context of ``n`` stays guarantees for an event task (``_informative``; at
        least the ``selected`` ones of an outcome-stratified context); None for other labels."""
        if prog.label in ("binary", "hosp_death"):
            return max(min(self._min_positive(policy), n // 2), 1, selected or 0)
        return max(1, selected or 0) if prog.label in ("tte", "multihorizon") else None

    def _informative(self, prog, rows, b, ctx, policy: str = "stratified") -> bool:
        m = np.isin(rows.stay, ctx) & b.valid
        if prog.label in ("binary", "hosp_death"):
            per = {}
            for s, y in zip(rows.stay[m], b.y[m]):
                per[s] = per.get(s, 0) or y > 0
            pos = sum(bool(x) for x in per.values())
            return pos >= self._min_cases(prog, policy, len(ctx)) and len(per) - pos >= 1
        if prog.label == "ordinal":
            return len(np.unique(b.y[m])) >= 2
        if prog.label in ("tte", "multihorizon"):
            return (b.y[m] > 0).sum() >= 1 and (b.y[m] == 0).sum() >= 1
        if prog.label == "forecast":
            return True
        return m.sum() >= self.cfg["min_context_values"]

    # -- the cascade filter (design-review correction 10) -------------------------------------------------------
    def snapshot(self, prog, rows: fl.Rows) -> np.ndarray:
        """Tabular snapshot features of rows: the last value in the day before t of the program's own channels and
        a few core vitals and labs (standardized), and hours since ICU admission."""
        own = [x for x in fpr.names(prog.target) if x in self.store.channels.index]
        chans = list(dict.fromkeys([*own, *RETRIEVAL]))[:12]
        cols = [rows.t / 60.0]
        for name in chans:
            if name not in self.store.channels.index:
                continue
            y, _c = fl.window_values(fl.finite(self.ev.events(name)), rows, -1440, 0, "last")
            cid = self.store.channels.index[name]
            med, sc = self.reader.median[cid], self.reader.scale[cid]
            cols.append(np.nan_to_num((y - med) / sc if np.isfinite(sc) and sc > 0 else y, nan=0.0))
        return np.stack(cols, 1)

    def _filter(self, prog, rows, b, ctx, queries, one_class: bool = False) -> dict:
        """Accept an episode whose label is predictable from its inputs at all: stage 1, the best single feature's
        rank AUROC on the context's labels (|AUROC - 0.5| >= 0.1, Spearman |rho| >= 0.2 for values) accepts; else
        stage 2, ExtraTrees fitted on the context and scored on the queries (AUROC above 0.5 + max(0.05, 2.33
        chance SD), R^2 above 0.05), as M8's filter. A context taken as drawn (``one_class``) may show one class:
        it is accepted unscored."""
        from scipy.stats import rankdata

        m = np.isin(rows.stay, ctx) & b.valid
        if m.sum() < 10 or not len(ctx) or prog.label == "forecast":
            return {"ok": True, "stage": 0}
        sub = rows.take(np.flatnonzero(m))
        x, y = self.snapshot(prog, sub), b.y[m].astype(np.float64)
        binary = prog.label in ("binary", "hosp_death", "tte", "multihorizon")
        if binary and len(np.unique(y)) < 2:
            return {"ok": True, "stage": 0, "one_class": True} if one_class else {"ok": False, "stage": 1, "score": 0.5}
        best = 0.0
        ry = rankdata(y)
        for j in range(x.shape[1]):
            rx = rankdata(x[:, j])
            if binary:
                pos = y > 0
                auc = (rx[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
                best = max(best, abs(auc - 0.5))
            else:
                c = np.corrcoef(rx, ry)[0, 1]
                best = max(best, abs(c) if np.isfinite(c) else 0.0)
        if best >= (0.1 if binary else 0.2):
            return {"ok": True, "stage": 1, "score": round(float(best), 4)}
        qi = [int(q["stay"]) for q in queries]
        qrows = fl.Rows(np.concatenate([np.full(len(q["times"]), s) for q, s in zip(queries, qi)]).astype(np.int64),
                        np.concatenate([np.asarray(q["times"]) for q in queries]).astype(np.int64))
        qb = self.ev.evaluate(prog, qrows)
        qm = qb.valid
        if qm.sum() < 5:
            return {"ok": True, "stage": 2, "score": None}
        from icutwin.factory.tasks import informative

        xq, yq = self.snapshot(prog, qrows.take(np.flatnonzero(qm))), qb.y[qm].astype(np.float64)
        if binary and len(np.unique(yq)) < 2:
            return {"ok": True, "stage": 2, "score": None}
        family = "regression" if not binary else "binary"
        ok, score = informative(np.vstack([x, xq]), np.r_[y, yq], len(y), family)
        return {"ok": bool(ok), "stage": 2, "score": round(float(score), 4)}

    def _difficulty(self, rng, prog, rows, b, ctx, variant, realism) -> float | None:
        """The probe cost of a virtual site's or realism's measurement and label perturbations on the context's
        snapshot (``sites.snapshot_loss``); None when there are none or too few labels to tell."""
        if not realism and variant.get("transform") not in ("thinning", "label_noise"):
            return None
        m = np.isin(rows.stay, ctx) & b.valid
        if m.sum() < 20 or prog.label == "forecast":
            return None
        sub = rows.take(np.flatnonzero(m))
        binary = prog.label in ("binary", "hosp_death", "tte", "multihorizon")
        return fsi.snapshot_loss(self.snapshot(prog, sub), b.y[m].astype(np.float64), binary, variant, realism, rng)

    # -- materialization -----------------------------------------------------------------------------------------
    def materialize(self, r: dict) -> dict:
        prog = _program(r["program"])
        idx = self.index
        cadence = tuple(r["cadence"])
        bud = self.budget
        ctx = [idx[s] for s in r["context"]]
        per = min(bud["events_per_trajectory"], bud["context_events"] // max(len(ctx), 1))
        ctx_tr = [self.reader.trajectory(i, None, per) for i in ctx]
        none = np.zeros(0, dtype=np.int64)
        rows = self.cadence_rows(cadence, ctx) if ctx else fl.Rows(none, none)
        ctx_lab = self._labels_by_stay(prog, rows, ctx)    # one evaluation per episode part; a forecast's context too
        # (phase_f.3: its values after each cadence time are that row's label, as for every other type)
        qi = [idx[q["stay"]] for q in r["queries"]]
        # a query is thinned only beyond ``query_events``: thinning up to its last prediction time t_K is not causal
        # (what it keeps before an earlier t_k depends on how many events follow), so the budget is set high enough
        # that it almost never applies (phase_f.4; QA #129)
        q_tr = [self.reader.trajectory(i, int(max(q["times"])), bud.get("query_events", bud["events_per_trajectory"]))
                for i, q in zip(qi, r["queries"], strict=True)]
        tg = fl.Rows(np.concatenate([np.full(len(q["times"]), i, dtype=np.int64)
                                     for i, q in zip(qi, r["queries"], strict=True)]),
                     np.concatenate([np.asarray(q["times"], dtype=np.int64) for q in r["queries"]]))             if qi else fl.Rows(none, none)
        q_tg = self._labels_by_stay(prog, tg, qi)
        own = self.cadence_rows(cadence, qi) if qi else fl.Rows(none, none)
        last = dict(zip(qi, (max(q["times"]) for q in r["queries"]), strict=True))
        own = own.take(np.flatnonzero(own.t <= np.array([last[s] for s in own.stay.tolist()], dtype=np.int64)))
        if prog.label == "forecast":                     # its own past values are in its trajectory already
            own = own.take(none)
        q_own = self._labels_by_stay(prog, own, qi)
        variant, realism = r.get("site_variant") or {"kind": "plain"}, r.get("realism") or {}
        mrng = np.random.default_rng([self.seed, r["index"], 99])
        if realism or variant.get("transform") in ("thinning", "rounding"):
            ctx_tr = fsi.apply(ctx_tr, variant, realism, self.reader.thinnable, mrng)
            q_tr = fsi.apply(q_tr, variant, realism, self.reader.thinnable, mrng,
                             [int(max(q["times"])) for q in r["queries"]])
        if variant.get("transform") == "label_noise" and prog.label in ("binary", "hosp_death"):
            for _rows, blk, _x in [*ctx_lab, *q_tg, *q_own]:            # charting errors: every label alike
                blk.y = fsi.flip_labels(blk.y, blk.valid, variant["flip"], mrng)
        anon_map = None
        if r["anonymize"] is not None:
            both, anon_map = ftr.anonymize(ctx_tr + q_tr, np.random.default_rng(r["anonymize"]))
            ctx_tr, q_tr = both[:len(ctx_tr)], both[len(ctx_tr):]
        desc = fpr.descriptor(prog, self.store.channels) if r["descriptor"] else None
        if desc is not None and anon_map is not None:
            local = {int(g): k for k, g in enumerate(anon_map)}
            for term in desc["terms"]:
                if "channel" in term:
                    term["channel"] = local.get(self.store.channels.index[term["channel"]])
        shp = {"yh": (len(prog.bands),), "mh": (len(prog.bands),)} if prog.label == "multihorizon" else None
        if prog.label == "forecast":                     # [n, H, C]: the grid's steps (or next value and delay)
            _, chans, grid, mode = prog.target
            hc = (len(grid) if mode == "grid" else 2, len(chans))
            shp = {"yf": hc, "mf": hc}
        return {"recipe": r,
                "task": {"program": r["program"], "descriptor": desc, "label": prog.label,
                         "horizon_min": prog.horizon, "bands": list(prog.bands), "site": r["site"],
                         "cadence": r["cadence"], "stage": r["stage"], "anonymized": anon_map is not None,
                         "selection": {"policy": r["policy"], "case_odds": r.get("case_odds"),
                                       "min_cases": r.get("min_cases")}},
                "channels": {"global_ids": anon_map, "n": int(len(anon_map)) if anon_map is not None
                             else len(self.store.channels)},
                "context": {**_pack(ctx_tr), **_pack_labels(ctx_lab, shp),
                            "end_t": np.array([t["end_t"] for t in ctx_tr], dtype=np.int32),
                            "end_kind": [t["end_kind"] for t in ctx_tr]},
                "query": {**_pack(q_tr), **_pack_labels(q_tg, shp),
                          **{f"own_{k}": x for k, x in _pack_labels(q_own, shp).items()}}}

    # -- evaluation sets and bedside context banks (plan Phase F, Factory design 6 and 11) -------------------------
    def _fixed_recipe(self, prog, cadence: tuple, site: str, context: list, queries: list, kind: str) -> dict:
        return {"version": VERSION, "index": 0, "kind": kind, "seed": int(self.seed), "stage": self.stage,
                "program": {"spec": prog.spec(), "name": prog.name, "family": prog.family,
                            "flags": list(prog.flags), "hash": prog.hash()},
                "site": site, "query_site": site, "site_variant": {"kind": "plain"}, "realism": {},
                "difficulty_loss": None, "cadence": list(cadence), "policy": kind,
                "case_odds": 1.0 if context else None, "min_cases": 0 if context else None, "as_drawn": bool(context),
                "context": [self._id(i) for i in context], "queries": queries, "descriptor": True,
                "anonymize": None, "attempts": 0, "filter": {}}

    def context_bank(self, prog, cadence: tuple, site: str, n: int, seed: int = 0) -> dict:
        """A context bank (bedside inference: a site's labelled past patients, or a twin cohort for its profile): ``n``
        complete trajectories of the site's stays with a valid label (by weight), each with the program's label
        timeline, in an episode's layout without queries."""
        rng = np.random.default_rng([self.seed, seed])
        rows, b = self.site_labels(prog, cadence, site)
        cand = np.unique(rows.stay[b.valid])
        ctx = np.sort(rng.choice(cand, size=min(n, len(cand)), replace=False, p=self._p(cand))) if len(cand) else []
        return self.materialize(self._fixed_recipe(prog, cadence, site, [int(i) for i in ctx], [], "context_bank"))

    def full_queries(self, prog, cadence: tuple, stays: list) -> dict:
        """A full-trajectory query set (alarm metrics and lead time): each stay a query at every cadence time of its
        stay, with its own labels, in an episode's layout without context."""
        rows = self.cadence_rows(cadence, list(stays))
        queries = [{"stay": self._id(i), "times": rows.t[rows.stay == i].astype(int).tolist()}
                   for i in stays if (rows.stay == i).any()]
        site = str(self.datasets[int(stays[0])]) if len(stays) else ""
        return self.materialize(self._fixed_recipe(prog, cadence, site, [], queries, "full_queries"))

    def _labels_by_stay(self, prog, rows: fl.Rows, stays: list) -> list:
        """(rows, block, extras) per stay of ``stays`` (in order), from one evaluation of all their rows."""
        r, blk, ex = self._labels_of(prog, rows)
        out = []
        for i in stays:
            k = np.flatnonzero(r.stay == i)
            out.append((r.take(k), blk.take(k), {key: x[k] for key, x in ex.items()}))
        return out

    def _labels_of(self, prog, rows: fl.Rows) -> tuple:
        """(rows, block, extras) of a program: extras hold a multi-horizon label's ``yh``/``mh`` [n, H] and a
        forecast's ``yf``/``mf`` [n, H, C]."""
        if prog.label == "multihorizon":
            b, y, ok = self.ev.multi(prog, rows)
            return rows, b, {"yh": np.where(ok, y, np.nan).astype(np.float32), "mh": ok}
        if prog.label == "forecast":
            y, m, _ok = self.ev.forecast(prog, rows)
            return rows, self.ev.evaluate(prog, rows), {"yf": y, "mf": m}
        return rows, self.ev.evaluate(prog, rows), {}

    def episode(self, index: int) -> dict:
        return self.materialize(self.recipe(index))


def _program(d: dict) -> fpr.Program:
    s = d["spec"]
    return fpr.Program(d["family"], _tuple(s["target"]), s["label"], s.get("horizon", 0), tuple(s["eligibility"]),
                       tuple(s.get("bands", ())), d.get("name", ""), tuple(d.get("flags", ())))


def _tuple(x):
    return tuple(_tuple(y) for y in x) if isinstance(x, list) else x


DTYPES = {"ev_t": np.int32, "ev_c": np.int32, "ev_v": np.float32, "ev_z": np.float32, "ev_bin": np.uint8,
          "ev_f": np.uint8, "iv_c": np.int32, "iv_s": np.int32, "iv_e": np.int32, "iv_v": np.float32,
          "st_k": np.int16, "st_v": np.float32, "st_avail": np.int32}
LABEL_DTYPES = {"y": np.float32, "valid": bool, "reason": np.uint8, "t_event": np.float32, "cause": np.int8,
                "t_censor": np.float32, "t_resolved": np.int64}


def _pack(trajs: list[dict]) -> dict:
    """Ragged trajectories as flat arrays with offsets (``ev_off``, ``iv_off``, ``st_off``: n + 1 each)."""
    out = {}
    for g, keys in (("ev", ("ev_t", "ev_c", "ev_v", "ev_z", "ev_bin", "ev_f")),
                    ("iv", ("iv_c", "iv_s", "iv_e", "iv_v")), ("st", ("st_k", "st_v", "st_avail"))):
        n = [len(t[keys[0]]) for t in trajs]
        out[f"{g}_off"] = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
        for k in keys:
            out[k] = (np.concatenate([t[k] for t in trajs]) if trajs else np.zeros(0)).astype(DTYPES[k])
    return out


def _pack_labels(items: list, shapes: dict | None = None) -> dict:
    """Label blocks of several patients ((rows, block, extras) each) as flat arrays with offsets ``lab_off``;
    ``shapes``: the extras a label type always carries (their trailing shapes), empty when no patient has them."""
    rows, blocks, extras = [x[0] for x in items], [x[1] for x in items], [x[2] for x in items]
    n = [len(b.y) for b in blocks]
    out = {"lab_off": np.concatenate([[0], np.cumsum(n)]).astype(np.int64),
           "lab_t": np.concatenate([r.t for r in rows]).astype(np.int32) if rows else np.zeros(0, dtype=np.int32)}
    for f in ("y", "valid", "reason", "t_event", "cause", "t_censor", "t_resolved"):
        x = np.concatenate([getattr(b, f) for b in blocks]) if blocks else np.zeros(0)
        out[f"lab_{f}"] = x.astype(LABEL_DTYPES[f])
    out["lab_valid"] = out["lab_valid"].astype(bool)
    out["lab_t_resolved"] = np.maximum(out["lab_t_resolved"], out["lab_t"].astype(np.int64) + 1)   # never at t itself
    out["lab_y"] = np.where(out["lab_valid"], out["lab_y"], np.nan).astype(np.float32)   # labels only where they apply
    for k in sorted({k for e in extras for k in e}):
        parts = [e[k] for e in extras if k in e and len(e[k])]
        if parts:
            out[f"lab_{k}"] = np.concatenate(parts)
        else:
            shape = next((e[k].shape[1:] for e in extras if k in e), ())
            out[f"lab_{k}"] = np.zeros((0, *shape), dtype=bool if k in ("mh", "mf") else np.float32)
    for k, shape in (shapes or {}).items():          # no patient (an episode without context): empty, shaped
        if f"lab_{k}" not in out:
            out[f"lab_{k}"] = np.zeros((0, *shape), dtype=bool if k in ("mh", "mf") else np.float32)
    return out


def recipe_hash(r: dict) -> str:
    return hashlib.sha1(json.dumps(r, sort_keys=True).encode()).hexdigest()


__all__ = ["Sampler", "load_config", "recipe_hash", "fp"]
