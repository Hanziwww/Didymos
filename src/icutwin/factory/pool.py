"""The label sampler's candidate pool (plan Phase F, "Label sampling" 3-4): program instances enumerated over the label
grammar, their attributes measured on a fixed panel of a store, redundant label functions clustered by their
signatures, and weights raked to the clinical attribute targets P* (``configs/factory/targets.yaml``).

* Candidates: the clinical catalog (F1-F7); F6 threshold crossings over every physiological channel x direction x
  decile edge x sustained duration x horizon, as binaries and as times to the first crossing; functional heads (FH)
  with drawn prevalences, horizons and label types.
* Attributes (``attributes``): the label type, the clinical domain (``domain_of``), the horizon band, the prevalence
  per prediction time and per stay on the panel's hourly rows, the valid and censored shares.
* Signatures: the positive rows of a binary label on the panel's first ``SIG_ROWS`` rows; programs whose positive sets
  overlap with Jaccard similarity >= 0.9 share a cluster (MinHash, banded LSH, then the exact Jaccard), one unit of
  base mass (after MOTOR's selection of codes informative given their parents).
* Raking (``rake``): within each family group, bounded iterative proportional fitting (``generate.calibrate.rake``)
  of the base (uniform over clusters, split among their members) to P*'s margins on the axes every program of the group
  has; the clinical catalog keeps its natural mass (it is the anchor). ``cells``: per (domain, label, horizon,
  prevalence) cell the target and realised shares and the effective number of distinct label functions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from icutwin.factory import labels as fl
from icutwin.factory import programs as fpr

TARGETS_YAML = Path(__file__).resolve().parents[1] / "configs" / "factory" / "targets.yaml"
DOMAIN_RULES = (
    ("vasopressor", "circulatory"), ("//map", "circulatory"), ("//sbp", "circulatory"), ("//dbp", "circulatory"),
    ("heart_rate", "circulatory"), ("lactate", "circulatory"), ("HEMO//", "circulatory"),
    ("spo2", "respiratory"), ("respiratory_rate", "respiratory"), ("RESP//", "respiratory"), ("po2", "respiratory"),
    ("pco2", "respiratory"), ("imv", "respiratory"), ("nippv", "respiratory"), ("cpap", "respiratory"),
    ("nasal_cannula", "respiratory"), ("face_mask", "respiratory"), ("trach", "respiratory"), ("AIRWAY", "respiratory"),
    ("creatinine", "renal"), ("//bun", "renal"), ("kdigo", "renal"), ("rrt", "renal"), ("OUTPUT//", "renal"),
    ("CRRT", "renal"), ("IHD", "renal"),
    ("potassium", "metabolic"), ("sodium", "metabolic"), ("chloride", "metabolic"), ("glucose", "metabolic"),
    ("bicarbonate", "metabolic"), ("calcium", "metabolic"), ("magnesium", "metabolic"), ("phosphate", "metabolic"),
    ("//ph_", "metabolic"), ("base_excess", "metabolic"), ("anion", "metabolic"),
    ("platelet", "hematologic"), ("hemoglobin", "hematologic"), ("hematocrit", "hematologic"), ("inr", "hematologic"),
    ("//pt//", "hematologic"), ("ptt", "hematologic"), ("fibrinogen", "hematologic"),
    ("wbc", "infection"), ("temp", "infection"), ("ase_", "infection"), ("presumed_infection", "infection"),
    ("crp", "infection"), ("procalcitonin", "infection"), ("neutrophil", "infection"), ("lymphocyte", "infection"),
    ("gcs", "neurologic"), ("rass", "neurologic"), ("PA//", "neurologic"),
    ("bilirubin", "hepatic"), ("//alt//", "hepatic"), ("//ast//", "hepatic"), ("albumin", "hepatic"),
    ("alkaline", "hepatic"),
)
HORIZON_BINS = ((0, 120, "<=2h"), (121, 360, "2-6h"), (361, 720, "6-12h"), (721, 1440, "12-24h"),
                (1441, 2880, "24-48h"), (2881, 10080, "48h-7d"))
PREV_EDGES = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.30, 0.60)
PREV_BINS = ("<0.5%", "0.5-1%", "1-2%", "2-5%", "5-10%", "10-15%", "15-30%", "30-60%", ">60%")
GROUPS = {"clinical": ("F1", "F2", "F3", "F4", "F5"), "threshold": ("F6",), "values": ("F7", "FC"),
          "functional": ("FH",)}
SIG_ROWS = 5000
HOLD = {"vitals": 120, "hemodynamics": 120, "respiratory": 240, "assessments": 480, "fluids": 240, "labs": 1440,
        "renal_support": 1440, "mechanical_support": 1440}


def load_targets(path: Path = TARGETS_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


_names = fpr.names


def domain_of(prog: fpr.Program) -> str:
    if prog.target[0] in ("death", "discharge", "los") or prog.label == "hosp_death":
        return "global"
    for name in _names(prog.target)[1:] or _names(prog.target):
        for key, dom in DOMAIN_RULES:
            if key in name:
                return dom
    return "global"


def horizon_bin(prog: fpr.Program) -> str:
    t = prog.target
    if prog.label == "binary" and prog.horizon > 0 and t[0] != "fn":
        h = prog.horizon
    elif t[0] in ("value", "stage_max"):
        h = t[4] if t[0] == "value" else t[3]
    elif t[0] == "fn":
        h = abs(t[6]) if t[6] > 0 else abs(t[5])
    elif t[0] == "forecast":
        h = int(t[2][-1])
    elif prog.label == "multihorizon":
        h = int(max(prog.bands))
    else:
        return "end"
    for lo, hi, name in HORIZON_BINS:
        if lo <= h <= hi:
            return name
    return "48h-7d" if h > 0 else "end"


def prevalence_bin(p: float) -> str:
    return PREV_BINS[int(np.searchsorted(PREV_EDGES, p, side="right"))]


def group_of(family: str) -> str:
    return next(g for g, fams in GROUPS.items() if family in fams)


# -- candidates -------------------------------------------------------------------------------------------------
def f6_candidates(channels, horizons=(60, 120, 240, 360, 720, 1440, 2880, 4320, 10080), sustains=(0, 60),
                  tte: bool = True) -> list[fpr.Program]:
    out = []
    for i, name in enumerate(channels.names):
        if not channels.physiological[i] or len(channels.edges[i]) < 3:
            continue
        g = channels.group[i]
        hold = HOLD.get(g, 720)
        for direction in ("below", "above"):
            for j in range(len(channels.edges[i])):
                for sustain in (sustains if g in ("vitals", "hemodynamics", "respiratory") else (0,)):
                    tgt = ("cross", name, direction, j, int(sustain), hold)
                    out += [fpr.Program("F6", tgt, "binary", int(h)) for h in horizons]
                    if tte:
                        out.append(fpr.Program("F6", tgt, "tte"))
    return out


def _name_domain(name: str) -> str:
    return next((dom for key, dom in DOMAIN_RULES if key in name), "global")


def fh_candidates(channels, n: int, seed: int, targets: dict, horizons=(60, 120, 240, 360, 720, 1440, 2880),
                  nowcast_share: float = 0.2) -> list[fpr.Program]:
    """Functional heads drawn by P*: the label type, the horizon band (then a horizon in it), the domain (a first
    channel of it, which sets the head's domain, then 1-4 more channels) and the prevalence band, so fresh heads drawn
    at episode time follow the targets as the pool's raked heads do."""
    from icutwin.factory import functional

    rng = np.random.default_rng(seed)
    phys = [nm for i, nm in enumerate(channels.names) if channels.physiological[i]]
    by_dom: dict = {}
    for nm in phys:
        by_dom.setdefault(_name_domain(nm), []).append(nm)
    doms = p_star(targets, "domain", sorted(by_dom))
    hbins = {}
    for h in horizons:
        hbins.setdefault(next(name for lo_, hi_, name in HORIZON_BINS if lo_ <= h <= hi_), []).append(h)
    hb = p_star(targets, "horizon", sorted(hbins))
    labs = p_star(targets, "label", ["binary", "ordinal", "regression"])
    prev = targets["axes"]["prevalence"]
    lo = np.r_[0.001, PREV_EDGES]
    hi = np.r_[PREV_EDGES, 0.6]
    shares = np.asarray([prev[k] for k in PREV_BINS[:-1]], dtype=np.float64)
    out = []
    for _ in range(n):
        k = int(rng.choice(len(shares), p=shares / shares.sum()))
        q = float(np.exp(rng.uniform(np.log(lo[k]), np.log(hi[k]))))
        band = str(rng.choice(list(hb), p=np.array(list(hb.values())) / sum(hb.values())))
        h = int(rng.choice(hbins[band]))
        now = bool(rng.random() < nowcast_share)
        dom = str(rng.choice(list(doms), p=np.array(list(doms.values())) / sum(doms.values())))
        first = str(rng.choice(by_dom[dom]))
        tgt = functional.draw(rng, [x for x in phys if x != first], q, h, nowcast=now, first=first)
        label = str(rng.choice(list(labs), p=np.array(list(labs.values())) / sum(labs.values())))
        bands = tuple(range(int(rng.integers(2, 5)))) if label == "ordinal" else ()
        out.append(fpr.Program("FH", tgt, label, h if label == "binary" else 0, ("in_icu",), bands,
                               flags=("nowcast",) if now else ()))
    return out


FC_VITALS = ("VITAL//heart_rate", "VITAL//map//arterial", "VITAL//map//cuff", "VITAL//spo2", "VITAL//respiratory_rate",
             "VITAL//temp_c", "VITAL//sbp//arterial", "VITAL//sbp//cuff")
FC_GRIDS = (tuple(range(60, 361, 60)), tuple(range(60, 1441, 60)), tuple(range(360, 10081, 360)))
MH_HORIZONS = ((60, 240, 720, 1440), (240, 720, 1440, 2880), (720, 1440, 2880, 10080))


def fc_candidates(channels, n: int, seed: int) -> list[fpr.Program]:
    """Forecasts of the patient's own values: 1-4 vitals on an hourly (or 6-hourly) grid, or 1-3 labs' next values
    and delays within 24 or 72 hours."""
    rng = np.random.default_rng(seed)
    vit = [c for c in FC_VITALS if c in channels.index]
    labs = [nm for i, nm in enumerate(channels.names) if channels.group[i] == "labs" and channels.physiological[i]]
    out = []
    for _ in range(n):
        if rng.random() < 0.6 and vit:
            chans = tuple(sorted(rng.choice(vit, size=int(rng.integers(1, min(4, len(vit)) + 1)), replace=False)))
            grid = FC_GRIDS[int(rng.choice(len(FC_GRIDS), p=[0.4, 0.45, 0.15]))]
            tgt = ("forecast", chans, grid, "grid")
        else:
            chans = tuple(sorted(rng.choice(labs, size=int(rng.integers(1, 4)), replace=False)))
            tgt = ("forecast", chans, (int(rng.choice([1440, 4320])),), "next")
        out.append(fpr.Program("FC", tgt, "forecast", eligibility=("in_icu",)))
    return out


def mh_candidates(programs: list, n: int, seed: int) -> list[fpr.Program]:
    """Multi-horizon versions of event programs (catalog events, threshold crossings): one time-to-event label read
    at several horizons."""
    rng = np.random.default_rng(seed)
    events = [p for p in programs if p.label == "binary" and p.target[0] in ("event", "cross", "stage", "death",
                                                                               "or", "and")]
    out = []
    for k in rng.choice(len(events), size=min(n, len(events)), replace=False) if events else []:
        p = events[int(k)]
        hs = MH_HORIZONS[int(rng.integers(len(MH_HORIZONS)))]
        out.append(fpr.Program(p.family if p.family != "F6" else "F6", p.target, "multihorizon", 0, p.eligibility, hs,
                               p.name))
    return out


VALUE_WINDOWS = ((0, 120), (0, 360), (360, 720), (1080, 1440), (0, 1440), (1440, 2880), (2880, 4320),
                 (0, -1))                              # -1: to the end of the ICU stay
BAND_WINDOWS = ((0, 360), (1080, 1440), (0, 1440))
BAND_SETS = ((1, 4, 7), (2, 6), (0, 3, 5, 8))          # decile edges of an ordinal band set (4, 3 and 5 classes)


def _phys(channels, min_edges: int = 3) -> list:
    return [nm for i, nm in enumerate(channels.names)
            if channels.physiological[i] and len(channels.edges[i]) >= min_edges]


def f7_value_candidates(channels, aggs=("last", "min", "max", "mean")) -> list[fpr.Program]:
    """Regression targets ("Label sampling" 2, F7): a physiological channel's aggregate in a future window, and its
    last value's change from the last one in the day before t."""
    out = []
    for name in _phys(channels):
        for a, b in VALUE_WINDOWS:
            out += [fpr.Program("F7", ("value", (name,), agg, a, b, None), "regression", 0, ("in_icu",)) for agg in aggs]
            out.append(fpr.Program("F7", ("value", (name,), "last", a, b, 1440), "regression", 0, ("in_icu",)))
    return out


def f6_ordinal_candidates(channels, aggs=("last", "max", "min")) -> list[fpr.Program]:
    """Severity bands (F6, EHRSHOT-style lab and vital classes): a physiological channel's aggregate in a future
    window, banded at decile edges (exact under both decodes, as thresholds)."""
    out = []
    for i, name in enumerate(channels.names):
        e = channels.edges[i]
        if not channels.physiological[i] or len(e) < 9:
            continue
        for bands in BAND_SETS:
            edges = tuple(float(e[k]) for k in bands)
            for a, b in BAND_WINDOWS:
                out += [fpr.Program("F6", ("value", (name,), agg, a, b, None), "ordinal", 0, ("in_icu",), edges)
                        for agg in aggs]
    return out


def f7_band_candidates(channels) -> list[fpr.Program]:
    """F7 ordinal values: a physiological channel's mean in a future window, banded at decile edges (F6's bands read
    the last value and the extremes)."""
    out = []
    for i, name in enumerate(channels.names):
        e = channels.edges[i]
        if not channels.physiological[i] or len(e) < 9:
            continue
        for bands in BAND_SETS:
            edges = tuple(float(e[k]) for k in bands)
            out += [fpr.Program("F7", ("value", (name,), "mean", a, b, None), "ordinal", 0, ("in_icu",), edges)
                    for a, b in VALUE_WINDOWS[1:6]]
            out += [fpr.Program("F7", ("value", (name,), agg, 0, -1, None), "ordinal", 0, ("in_icu",), edges)
                    for agg in ("max", "min")]                 # the worst over the rest of the stay
    return out


def f7_event_candidates(programs: list, n_pairs: int, seed: int, crossings: list | None = None) -> list[fpr.Program]:
    """F7 times to event ("time to any F1-F6 event", death competing): the clinical events' targets, random OR
    composites of two of them, and of one with a threshold crossing (``crossings``: F6 targets; a deterioration seen
    either way); and F7 ordinal stages (the maximal KDIGO stage in a window, banded) and remaining-stay bands."""
    rng = np.random.default_rng(seed)
    tgts = list(dict.fromkeys(p.target for p in programs if p.family in GROUPS["clinical"] and p.label == "binary"
                              and p.target[0] in ("event", "stage", "point", "death", "or", "and")))
    out = [fpr.Program("F7", t, "tte", 0, ("in_icu", "not_in_event")) for t in tgts]
    for _ in range(min(n_pairs, len(tgts) * (len(tgts) - 1) // 2)):
        a, c = rng.choice(len(tgts), size=2, replace=False)
        out.append(fpr.Program("F7", ("or", tgts[min(a, c)], tgts[max(a, c)]), "tte", 0, ("in_icu", "not_in_event")))
    cross = list(dict.fromkeys(crossings or []))
    if tgts and cross:
        for op in ("or", "and"):                    # either (a deterioration seen either way) or both (rarer)
            for _ in range(n_pairs):
                t, x = tgts[int(rng.integers(len(tgts)))], cross[int(rng.integers(len(cross)))]
                out.append(fpr.Program("F7", (op, t, x), "tte", 0, ("in_icu", "not_in_event")))
    for kind in ("kdigo", "kdigo_creat", "kdigo_uo"):
        for a, c in ((0, 1440), (0, 2880), (1440, 2880), (0, 4320)):
            out += [fpr.Program("F7", ("stage_max", kind, a, c), "ordinal", 0, ("in_icu",), bands)
                    for bands in ((1, 2, 3), (1, 2), (2, 3))]
    out += [fpr.Program("F7", ("los",), "ordinal", 0, ("in_icu",), bands)
            for bands in ((24, 72), (48, 168), (12, 24, 48, 96), (24, 48, 96, 168, 336))]
    return out


# -- the panel and attributes -------------------------------------------------------------------------------------
@dataclass
class Panel:
    rows: fl.Rows
    sig: np.ndarray          # positions of the signature rows in ``rows``
    datasets: np.ndarray = None   # each row's dataset (the site support of ``attributes``)

    @classmethod
    def of(cls, store, n_stays: int = 3000, per_stay: int = 24, seed: int = 0, split: str | None = "train") -> Panel:
        rng = np.random.default_rng(seed)
        st = store.stays
        ok = np.ones(len(st), dtype=bool) if split is None or "split" not in st else (st["split"] == split).to_numpy()
        pool = np.flatnonzero(ok)
        band = np.digitize(store.los[pool] / 60, [24, 48, 96, 168])
        strata = st["dataset"].to_numpy()[pool].astype(str) + ":" + band.astype(str)
        keys, inv = np.unique(strata, return_inverse=True)
        take = []
        for k in range(len(keys)):                           # proportional, at least one per stratum
            m = np.flatnonzero(inv == k)
            n = max(1, int(round(n_stays * len(m) / len(pool))))
            take += rng.choice(pool[m], size=min(n, len(m)), replace=False).tolist()
        r = fl.rolling(store, 60, np.sort(np.asarray(take)))
        keep = []
        starts = np.r_[0, np.flatnonzero(np.diff(r.stay)) + 1]
        for a, b in zip(starts, np.r_[starts[1:], len(r)]):
            keep += rng.choice(np.arange(a, b), size=min(per_stay, b - a), replace=False).tolist()
        r = r.take(np.sort(np.asarray(keep)))
        sig = np.sort(rng.choice(len(r), size=min(SIG_ROWS, len(r)), replace=False))
        return cls(r, sig, st["dataset"].to_numpy()[r.stay].astype(str))


def attributes(ev: fpr.Evaluator, prog: fpr.Program, panel: Panel) -> dict:
    b = ev.evaluate(prog, panel.rows)
    ok = b.valid
    out = {"hash": prog.hash(), "family": prog.family, "group": group_of(prog.family), "name": prog.name,
           "label": prog.label, "domain": domain_of(prog), "horizon_bin": horizon_bin(prog),
           "valid_share": float(ok.mean()), "censored_share": float((b.reason & fl.REASONS["censored"] > 0).mean()),
           "spec": json.dumps({**prog.spec(), "family": prog.family, "name": prog.name, "flags": list(prog.flags)}),
           "prevalence": np.nan, "per_stay": np.nan, "sig": None, "sites": None}
    if not ok.any():
        return out
    ds = panel.datasets[ok]
    st_ok = panel.rows.stay[ok]
    event = prog.label in ("binary", "hosp_death", "tte", "multihorizon")
    pos_rows = (b.y[ok] > 0) if event else np.ones(int(ok.sum()), dtype=bool)
    out["sites"] = json.dumps({str(d): [len(np.unique(st_ok[ds == d])), len(np.unique(st_ok[(ds == d) & pos_rows]))]
                               for d in np.unique(ds)})
    if prog.label in ("binary", "hosp_death", "tte"):
        y = b.y[ok] > 0
        out["prevalence"] = float(y.mean())
        stays = panel.rows.stay[ok]
        pos = np.unique(stays[y])
        out["per_stay"] = float(len(pos) / len(np.unique(stays)))
        sig_y = np.zeros(len(panel.sig), dtype=bool)
        sv = ok[panel.sig]
        sig_y[sv] = b.y[panel.sig][sv] > 0
        out["sig"] = np.flatnonzero(sig_y).astype(np.int32).tobytes()
    elif prog.label == "multihorizon":
        out["prevalence"] = float((b.y[ok] > 0).mean())          # the event rate, for the prevalence axis
    elif prog.label == "ordinal":
        _k, n = np.unique(b.y[ok], return_counts=True)
        p = n / n.sum()
        out["prevalence"] = float(-(p * np.log(p)).sum())          # the class entropy, for ordinal labels
    return out


_W: dict = {}


def _attr_chunk(job: tuple) -> list[dict]:
    """Attributes of one chunk of candidates in a worker process (its own evaluator over the store)."""
    root, segments, panel, progs = job
    if _W.get("key") != (root, tuple(segments)):
        from icutwin.factory import store as fst

        _W.update(key=(root, tuple(segments)), ev=fpr.Evaluator(fst.Store(Path(root), list(segments))))
    return [attributes(_W["ev"], p, panel) for p in progs]


def build(store, v_path: Path | None = None, *, n_fh: int = 3000, seed: int = 0, n_stays: int = 3000,
          targets: dict | None = None, programs: list | None = None, workers: int = 1,
          chunk: int = 500, n_fc: int = 1000, n_mh: int = 4000, n_pairs: int = 3000) -> tuple:
    """(pool table, panel): every candidate's attributes on the store's panel, clustered and raked; ``workers``
    processes evaluate chunks of consecutive candidates (a channel's programs stay together), the table the same
    for any number of them."""
    import pandas as pd

    targets = targets or load_targets()
    panel = Panel.of(store, n_stays=n_stays, seed=seed)
    ev = fpr.Evaluator(store)
    if programs is None:
        cat = fpr.catalog(store.channels)
        f6 = f6_candidates(store.channels)
        base = cat + f6
        cross = [p.target for p in f6 if p.label == "tte"]
        progs = (base + fh_candidates(store.channels, n_fh, seed, targets)
                 + fc_candidates(store.channels, n_fc, seed) + mh_candidates(base, n_mh, seed)
                 + f6_ordinal_candidates(store.channels) + f7_value_candidates(store.channels)
                 + f7_band_candidates(store.channels) + f7_event_candidates(cat, n_pairs, seed, cross))
    else:
        progs = programs
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        segs = [s.name for s in store.segments]
        jobs = [(str(store.root), segs, panel, progs[k:k + chunk]) for k in range(0, len(progs), chunk)]
        with ProcessPoolExecutor(workers) as ex:
            rows = [r for part in ex.map(_attr_chunk, jobs) for r in part]
    else:
        rows = [attributes(ev, p, panel) for p in progs]
    df = pd.DataFrame(rows).drop_duplicates("hash").reset_index(drop=True)
    df["prevalence_bin"] = [prevalence_bin(p) if p == p else "na" for p in df["prevalence"]]
    df["cluster"] = clusters(df["sig"].tolist())
    for k, w in rake_stages(df, targets).items():
        df[k] = w
    return df, panel


def clusters(sigs: list, threshold: float = 0.9, perms: int = 32, bands: int = 8, seed: int = 0) -> np.ndarray:
    """Cluster ids: label functions whose positive sets overlap with Jaccard >= ``threshold`` share one (union-find
    over MinHash-LSH candidates checked exactly); programs without a signature are their own cluster."""
    rng = np.random.default_rng(seed)
    p = (1 << 61) - 1
    a = rng.integers(1, p, size=perms, dtype=np.int64).astype(np.uint64)
    b = rng.integers(0, p, size=perms, dtype=np.int64).astype(np.uint64)
    parent = np.arange(len(sigs))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sets = [np.frombuffer(s, dtype=np.int32) if s is not None else None for s in sigs]
    buckets: dict = {}
    rows = perms // bands
    for i, s in enumerate(sets):
        if s is None or len(s) == 0:
            continue
        h = (s.astype(np.uint64)[:, None] * a[None, :] + b[None, :]) % np.uint64(p)
        mh = h.min(0)
        for k in range(bands):
            buckets.setdefault((k, mh[k * rows:(k + 1) * rows].tobytes()), []).append(i)
    for members in buckets.values():
        reps: list = []                        # one member per cluster met in this bucket
        for j in members:
            sj = sets[j]
            for i0 in reps:
                if find(i0) == find(j):
                    break
                si = sets[i0]
                inter = len(np.intersect1d(si, sj, assume_unique=True))
                if inter / (len(si) + len(sj) - inter) >= threshold:
                    parent[find(j)] = find(i0)
                    break
            else:
                reps.append(j)
    return np.array([find(i) for i in range(len(sigs))])


def p_star(targets: dict, axis: str, have: list) -> dict:
    """P* on one axis over the categories ``have``: (1 - eps) tempered(P_clin, alpha) + eps uniform."""
    pc = {k: v for k, v in targets["axes"][axis].items() if k in have}
    if not pc:
        return {}
    a, e = targets["alpha"], targets["eps"]
    t = {k: v ** a for k, v in pc.items()}
    s = sum(t.values())
    return {k: (1 - e) * x / s + e / len(t) for k, x in t.items()}


AXES = (("domain", "domain"), ("horizon_bin", "horizon"), ("label", "label"), ("prevalence_bin", "prevalence"))


def axis_features(df) -> dict:
    """Each program's category per raked axis (the rake's and the report's), ``na`` where the axis does not apply: a
    time to event has no horizon (it is read at any), a label other than an event's has no prevalence."""
    lab = df["label"].to_numpy()
    return {"domain": df["domain"].to_numpy().astype(str),
            "horizon_bin": np.where(lab == "tte", "na", df["horizon_bin"].to_numpy().astype(str)),
            "label": lab.astype(str),
            "prevalence_bin": np.where(np.isin(lab, ["binary", "tte", "multihorizon"]),
                                       df["prevalence_bin"].to_numpy().astype(str), "na")}


def rake(df, targets: dict) -> np.ndarray:
    """Weights per program (module docstring); each group's weights sum to its count of programs, its mass among
    groups being the episode sampler's (the study lead's label mix). Catalog programs (named) are anchors: they keep
    their base mass. Generated programs are raked to P* on the axes of their group; on an axis a program has no category
    for (``axis_features``) it is free: the labels it does not apply to keep the label axis's target share. A
    generated program in a category P* leaves out (a prevalence above 60% per time), or an event never valid on the
    panel, gets no weight."""
    from icutwin.generate.calibrate import rake as ipf

    w = np.zeros(len(df))
    feats_all = axis_features(df)
    for g in GROUPS:
        idx = np.flatnonzero((df["group"] == g).to_numpy())
        if not len(idx):
            continue
        sub = df.iloc[idx]
        _u, inv, cnt = np.unique(sub["cluster"].to_numpy(), return_inverse=True, return_counts=True)
        base = 1.0 / cnt[inv]                                   # one unit of mass per cluster
        anchor = (sub["name"].fillna("") != "").to_numpy() | (g == "clinical")
        ww = base.copy()
        gen = np.flatnonzero(~anchor)
        if len(gen):
            feats = {ax: feats_all[ax][idx][gen] for ax, _c in AXES}
            out = np.zeros(len(gen), dtype=bool)
            for ax, col in AXES:
                support = set(targets["axes"].get(col, {}))
                out |= ~np.isin(feats[ax], [*support, "na"])
            event = np.isin(feats["label"], ["binary", "tte", "multihorizon"])
            out |= event & (feats["prevalence_bin"] == "na")      # an event never valid on the panel: degenerate
            ww[gen[out]] = 0.0
            keep = np.flatnonzero(~out)
            tg = {}
            bg = base[gen][keep]
            labs = feats["label"][keep]
            p_lab = p_star(targets, "label", sorted(set(labs.tolist())))
            for ax, col in AXES:
                cats = feats[ax][keep]
                have = sorted(set(cats.tolist()) - {"na"})
                ps = p_star(targets, col, have)
                if not ps or len(have) < 2:
                    continue
                na_labels = set(labs[cats == "na"].tolist())          # the labels this axis does not apply to:
                if ax != "label" and len(p_lab) > 1 and na_labels and                         not (set(labs[cats != "na"].tolist()) & na_labels):     # their share is the label axis's
                    free = float(sum(p_lab.get(x, 0.0) for x in na_labels))
                else:
                    free = float(bg[cats == "na"].sum() / bg.sum())
                t = {k: v * (1 - free) for k, v in ps.items()}
                if free > 0:
                    t["na"] = free
                tg[ax] = t
            if tg and len(keep):
                fl_ = [{ax: feats[ax][keep][j] for ax in tg} for j in range(len(keep))]
                rw, _rep = ipf(fl_, tg, base=bg, caps=tuple(targets["caps"]))
                ww[gen[keep]] = rw * bg.sum() / rw.sum()
        ww = ww * len(idx) / ww.sum() if ww.sum() > 0 else ww
        floor = float(targets.get("anchor_floor", 0.0)) * len(idx)
        named = (sub["name"].fillna("") != "").to_numpy()
        if g != "clinical" and floor > 0 and named.any():     # a catalog definition among generated programs is
            ww[named] = np.maximum(ww[named], floor)          # always reached (a floor: raking never removes it)
            ww = ww * len(idx) / ww.sum()
        w[idx] = ww
    return w


def stage_allowed(df, stage: int) -> np.ndarray:
    """The programs a stage draws (``configs/factory/stages.yaml``: its label types; forecasts up to its horizon)."""
    from icutwin.factory.episodes import load_config

    st = load_config()["stages"][stage]
    ok = df["label"].isin(st["labels"]).to_numpy()
    fc = (df["label"] == "forecast").to_numpy()
    if fc.any():
        ends = np.array([json.loads(s)["target"][2][-1] if f else 0 for s, f in zip(df["spec"], fc, strict=True)])
        ok &= ~fc | (ends <= st.get("forecast_max_min", 10080))
    return ok


def rake_stages(df, targets: dict) -> dict:
    """Weights raked on all programs (``weight``) and on stage 1's alone (``weight_s1``: without multi-horizon labels
    and long forecasts, its programs follow P* by themselves)."""
    out = {"weight": rake(df, targets)}
    m = stage_allowed(df, 1)
    w1 = np.zeros(len(df))
    if m.any():
        w1[m] = rake(df[m].reset_index(drop=True), targets)
    out["weight_s1"] = w1
    return out


def axis_report(df, targets: dict) -> dict:
    """Per raked group and axis: the realised shares of the programs' weights among the categories the axis applies to
    (``axis_features``), P* there, their total variation, and the KL divergence to P_clin."""
    feats_all = axis_features(df)
    out = {}
    for g in GROUPS:
        m = ((df["group"] == g) & (df["weight"] > 0)).to_numpy()
        if not m.any():
            continue
        w = df["weight"].to_numpy()[m]
        r = {}
        for ax, col in AXES:
            cats = feats_all[ax][m]
            on = cats != "na"
            if not on.any():
                continue
            share = {}
            for k in np.unique(cats[on]):
                share[str(k)] = float(w[on][cats[on] == k].sum() / w[on].sum())
            ps = p_star(targets, col, sorted(share))
            pc = {k: v for k, v in targets["axes"].get(col, {}).items() if k in share}
            tot = sum(pc.values()) or 1.0
            pc = {k: v / tot for k, v in pc.items()}
            kl = float(sum(v * np.log(v / pc[k]) for k, v in share.items() if v > 0 and pc.get(k, 0) > 0))
            r[ax] = {"realised": {k: round(v, 4) for k, v in share.items()},
                     "p_star": {k: round(v, 4) for k, v in ps.items()},
                     "tv_to_p_star": round(0.5 * sum(abs(share.get(k, 0) - ps.get(k, 0))
                                                     for k in set(share) | set(ps)), 4),
                     "kl_to_p_clin": round(kl, 4), "applies": round(float(w[on].sum() / w.sum()), 4)}
        out[g] = r
    return out


def load_support(df) -> dict:
    """Per program hash, its sites on the pool's panel: dataset -> (valid stays, positive stays) (the episode sampler's
    site draw and cohort size, ``episodes.Sampler``)."""
    if "sites" not in df:
        return {}
    return {h: {d: tuple(x) for d, x in json.loads(s).items()} for h, s in zip(df["hash"], df["sites"], strict=True)
            if isinstance(s, str)}


def cells(df, targets: dict) -> list[dict]:
    """Per (group, domain, label, horizon, prevalence) cell: its weight share and its effective number of distinct
    label functions (effective size over clusters)."""
    out = []
    key = ["group", "domain", "label", "horizon_bin", "prevalence_bin"]
    tot = df.groupby("group")["weight"].sum().to_dict()
    for k, sub in df.groupby(key):
        wc = sub.groupby("cluster")["weight"].sum().to_numpy()
        ess = float(wc.sum() ** 2 / (wc ** 2).sum()) if wc.sum() > 0 else 0.0
        out.append({**dict(zip(key, k)), "programs": len(sub), "clusters": int(len(wc)),
                    "share": float(sub["weight"].sum() / tot[k[0]]), "ess": ess})
    return out


def save(df, panel: Panel, out: Path, info: dict) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "pool.parquet", index=False)
    (out / "pool.json").write_text(json.dumps({**info, "programs": len(df), "panel_rows": len(panel.rows),
                                               "panel_stays": int(len(np.unique(panel.rows.stay)))}, indent=1),
                                   encoding="utf-8")


ACCEPT_KEYS = ("group", "label", "horizon_bin", "prevalence_bin")
ACCEPT_FLOOR = 0.05


def cell_key(row) -> str:
    return "|".join(str(row[k]) for k in ACCEPT_KEYS)


def acceptance_weights(df, acc: dict) -> np.ndarray:
    """The pool's weights divided by their cell's measured acceptance (``acceptance.json``: the share of the
    episode sampler's attempts with a program of the cell that became an episode, at least ``ACCEPT_FLOOR``), each
    group keeping its total: a redraw stays in the group, so the accepted programs then follow the raked weights
    ("Label sampling" 5: the cell's weight divided by its acceptance)."""
    w = df["weight"].to_numpy(dtype=np.float64).copy()
    keys = [cell_key(r) for r in df[list(ACCEPT_KEYS)].to_dict("records")]
    a = np.array([max(acc.get(k, 1.0), ACCEPT_FLOOR) for k in keys])
    out = w / a
    for g in GROUPS:
        m = (df["group"] == g).to_numpy()
        if m.any() and out[m].sum() > 0:
            out[m] *= w[m].sum() / out[m].sum()
    return out


def load(pool_dir: Path, stage: int = 1) -> tuple[list, np.ndarray, dict]:
    """A saved pool's programs, weights (the stage's raking when the pool has one; divided by their cells'
    acceptance when the pool has ``acceptance.json``) and site support (``Sampler(programs=, weights=, support=)``)."""
    import pandas as pd

    df = pd.read_parquet(Path(pool_dir) / "pool.parquet")
    if f"weight_s{stage}" in df:
        df["weight"] = df[f"weight_s{stage}"]
    progs, weights = load_programs(df)
    support = load_support(df)
    acc_path = Path(pool_dir) / "acceptance.json"
    if acc_path.exists():
        acc = json.loads(acc_path.read_text(encoding="utf-8"))
        weights = acceptance_weights(df, acc["cells"])
        if acc.get("sites"):                     # a site's draw scaled so episode use follows the store's shares
            support["_site_factors"] = acc["sites"]
    return progs, weights, support


def load_programs(df) -> tuple[list, np.ndarray]:
    """The pool's programs (from their specs) and weights."""
    from icutwin.factory.episodes import _program

    progs = []
    for s in df["spec"]:
        d = json.loads(s)
        progs.append(_program({"spec": d, "family": d["family"], "name": d.get("name", ""),
                               "flags": d.get("flags", [])}))
    return progs, df["weight"].to_numpy(dtype=np.float64)
