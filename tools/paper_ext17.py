"""Paper P amendment 17 (``docs/p1_paper_protocol.md``): adapting Didymos to RICD and zigongICU. Local, CPU; aggregates
only leave the machine.

``split``: the locked split by patient (17a): u = blake2b("17:<database>:<patient id>") over 2^64; TEST u < 0.50,
FT 0.50 <= u < 0.85, LABEL u >= 0.85. Writes ``<out>/split.json`` (counts and the stays' sha256),
``<out>/stays.parquet`` (stay, dataset, patient, slice; host only) and the anchor files the host reads with
``paper_rollouts.py --anchors-file``:
* ``test_R.parquet``, ``test_L24e.parquet``: amendment 16a's kept anchors whose patient is in TEST, at 16a's hours; R
  keeps its inclusion weights, L24e's case-cohort weights are recomputed within TEST (per database and stratum, the
  TEST landmark anchors of the stratum over the TEST anchors sampled from it);
* ``label_R.parquet``, ``label_L24e.parquet``: 16a's rules within the LABEL stays, seed 17 (the rolling design, 250 per
  database; the 24-hour landmark's case-cohort sample, up to 125 in-hospital deaths and 125 others per database),
  with F1's clock check.

    python tools/paper_ext17.py split --tokens data/tokens/ext_r4 --store data/factory/stores/ext_test \\
        --ext runs/paper/ext --out runs/paper/ext17

``trainroot``: F1's training data (17b), ``<out>/{ricd,zigong}/{train,tuning}.parquet`` from the improved encoding
(``ext_r4b``): the FT stays only, the held-out 10% of FT patients (v = blake2b("17v:<database>:<patient id>") over
2^64 < 0.10) as ``tuning`` (the choice of the grid's cell), the site token rewritten from ``other`` to unknown in
every sequence, goal block and window (the dataset token is unknown already); ``--member-out``: the FT train stays
as ``train`` and the TEST stays as ``held_out`` (the same rewrite), for the membership check (17b). The host links the seven databases'
shards from ``full_r4`` beside them. ``<split dir>/adapt.json``: the sizes, the target passes that mix FT 1:1 by
windows with the seven databases' train windows (q = E x FT windows / the seven's train windows, for E passes over the
FT pool), and each grid cell's steps (the tokens drawn over the tokens per step, as MUC32_main's steps were set).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_external import EXTERNAL, SPLITS, _clock_ok, _patients, labels  # noqa: E402

SEED = 17
TEST, FT = 0.50, 0.85
LABEL_R, LABEL_CASES = 250, 125
ANCHOR_FIELDS = ("anchor", "twin", "dataset", "hours", "los_hours", "band", "enriched", "flag", "weight")


def slice_of(dataset: str, patient: str) -> str:
    """TEST, FT or LABEL for a patient of an external database (17a)."""
    u = int.from_bytes(hashlib.blake2b(f"17:{dataset}:{patient}".encode(), digest_size=8).digest(), "big") / 2 ** 64
    return "test" if u < TEST else "ft" if u < FT else "label"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _anchor_frame(anchors) -> "object":
    import pandas as pd

    return pd.DataFrame({f: [getattr(x, f) for x in anchors] for f in ANCHOR_FIELDS})


def cmd_split(a) -> None:
    import pandas as pd

    from icutwin.eval import bench
    from icutwin.factory import twins
    from icutwin.tokenization.vocab import Vocab

    tokens, store, ext, out = Path(a.tokens), Path(a.store), Path(a.ext), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pat = _patients(Path(a.clif))
    rows = bench.real_rows(tokens, SPLITS, bench.store_stay_ids(store), EXTERNAL)
    st = pd.DataFrame({"stay": [r["stay_id"] for r in rows], "dataset": [r["dataset"] for r in rows]})
    st["patient"] = st["stay"].map(pat)
    st["slice"] = [slice_of(d, p.split(":", 1)[1]) for d, p in zip(st["dataset"], st["patient"], strict=True)]
    st.to_parquet(out / "stays.parquet", index=False)
    slice_of_stay = dict(zip(st["stay"], st["slice"], strict=True))
    rep: dict = {"seed": SEED, "bounds": {"test": TEST, "ft": FT}, "stays_sha256": _sha(out / "stays.parquet"),
                 "stays": {d: st[st["dataset"] == d]["slice"].value_counts().to_dict() for d in EXTERNAL},
                 "patients": {d: st[st["dataset"] == d].drop_duplicates("patient")["slice"].value_counts().to_dict()
                              for d in EXTERNAL}}
    # TEST anchors: 16a's kept anchors in TEST
    r16 = pd.read_parquet(ext / "R" / "host_anchors.parquet")
    t_r = r16[r16["twin"].map(slice_of_stay) == "test"][list(ANCHOR_FIELDS)].reset_index(drop=True)
    t_r.to_parquet(out / "test_R.parquet", index=False)
    l16 = pd.read_parquet(ext / "L24e" / "host_anchors.parquet")
    strata = pd.read_parquet(ext / "subsets" / "L24e.parquet").set_index("anchor")["stratum"]
    full = pd.read_parquet(ext / "L24e" / "anchors.parquet")                      # every landmark anchor (16a)
    full_t = full[full["twin"].map(slice_of_stay) == "test"].reset_index(drop=True)
    hd = labels(store, full_t, "L24")
    hd = hd[hd["program"] == "hospital_death@0"].set_index("anchor")["label"]
    t_l = l16[l16["twin"].map(slice_of_stay) == "test"][list(ANCHOR_FIELDS)].reset_index(drop=True)
    t_l["stratum"] = t_l["anchor"].map(strata)
    cc = {}
    for d in EXTERNAL:
        for stratum, y in (("case", 1.0), ("control", 0.0)):
            pool = int(((full_t["dataset"] == d) & (full_t["anchor"].map(hd) == y)).sum())
            m = (t_l["dataset"] == d) & (t_l["stratum"] == stratum)
            t_l.loc[m, "weight"] = pool / max(int(m.sum()), 1)
            cc[f"{d}_{stratum}"] = {"pool": pool, "sampled": int(m.sum()), "weight": pool / max(int(m.sum()), 1)}
    t_l.to_parquet(out / "test_L24e.parquet", index=False)
    rep["test"] = {"R": t_r["dataset"].value_counts().to_dict(), "L24e": cc}
    # LABEL anchors: 16a's rules within the LABEL stays, seed 17
    v = Vocab.load(tokens / "vocab.json")
    lab_rows = [r for r in rows if slice_of_stay[r["stay_id"]] == "label"]
    stays = [(r["stay_id"], r["dataset"], twins.decode_labels(r["ids"], v, r["stay_id"])) for r in lab_rows]
    ids = {r["stay_id"]: r["ids"] for r in lab_rows}

    def kept(anchors):
        an = _anchor_frame(anchors)
        prompts = [bench.unknown_site(bench.prompt(ids[x.twin], v, x.twin, x.hours), v) for x in anchors]
        an["t_min"] = [bench.anchor_time(p, v, x.twin) for p, x in zip(prompts, anchors, strict=True)]
        ok = _clock_ok(an, prompts, v)
        return an[ok].reset_index(drop=True), int((~ok).sum())

    l_r, ex_r = kept(bench.design_anchors(stays, v, "R", LABEL_R, SEED))
    l_r.drop(columns="t_min").to_parquet(out / "label_R.parquet", index=False)
    lm, ex_l = kept(bench.design_anchors(stays, v, "L24", 10 ** 9, SEED))
    y = labels(store, lm, "L24")
    y = y[y["program"] == "hospital_death@0"].set_index("anchor")["label"]
    rng = np.random.default_rng(SEED)
    parts, lcc = [], {}
    for d in EXTERNAL:
        x = lm[lm["dataset"] == d]
        yy = x["anchor"].map(y).to_numpy()
        for stratum, sel in (("case", yy == 1), ("control", yy == 0)):
            pool = x[sel]
            pick = pool.iloc[np.sort(rng.choice(len(pool), size=min(LABEL_CASES, len(pool)), replace=False))].copy()
            pick["weight"] = len(pool) / max(len(pick), 1)
            pick["stratum"] = stratum
            parts.append(pick)
            lcc[f"{d}_{stratum}"] = {"pool": int(len(pool)), "sampled": int(len(pick))}
    l_l = pd.concat(parts, ignore_index=True).drop(columns="t_min")
    l_l.to_parquet(out / "label_L24e.parquet", index=False)
    rep["label"] = {"R": l_r["dataset"].value_counts().to_dict(), "R_excluded_clock": ex_r,
                    "L24e": lcc, "L24e_excluded_clock": ex_l}
    rep["files"] = {f.name: _sha(f) for f in sorted(out.glob("*.parquet"))}
    (out / "split.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float))


VAL = 0.10
GRID = {"lr": (3e-5, 1e-4), "passes": (1.0, 3.0)}
STEP_TOKENS = 32768 * 4                        # one 5090 (32 GB): half MUC32_main's micro-batch, four per step
AUX = ("vocab.json", "vocabulary.parquet", "support.json", "rank_persistence.json", "goal_bindings.json",
       "los_bands.json", "calibration.json", "calibration_events.json")


def val_of(dataset: str, patient: str) -> bool:
    """FT patients held out for the choice of the grid's cell (17b)."""
    u = int.from_bytes(hashlib.blake2b(f"17v:{dataset}:{patient}".encode(), digest_size=8).digest(), "big") / 2 ** 64
    return u < VAL


def site_unknown(r: dict, pos: int, other: int, unknown: int) -> dict:
    """A stored sequence with its site token ``other`` set to unknown (every window starts with the prefix)."""
    seqs = [r["ids"], r["goal_ids"], *r["windows"]]
    if any(len(s) <= pos or s[pos] != other for s in seqs):
        raise ValueError(f"{r['stay_id']}: the site token is not 'other' at position {pos}")
    fix = lambda s: [*s[:pos], unknown, *s[pos + 1:]]             # noqa: E731
    return {**r, "ids": fix(r["ids"]), "goal_ids": fix(r["goal_ids"]), "windows": [fix(w) for w in r["windows"]]}


def cmd_trainroot(a) -> None:
    import shutil

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from icutwin.model.data import layout
    from icutwin.tokenization.vocab import Vocab

    tokens, split_dir, out = Path(a.tokens), Path(a.split_dir), Path(a.out)
    seven = json.loads(Path(a.seven_index).read_text())["shards"]
    v = Vocab.load(tokens / "vocab.json")
    lay = layout(v)
    other, unknown = v.id("PREFIX//site//other"), lay.site_unknown
    st = pd.read_parquet(split_dir / "stays.parquet")
    ft = st[st["slice"] == "ft"]
    part = {s: ("tuning" if val_of(d, p.split(":", 1)[1]) else "train")
            for s, d, p in zip(ft["stay"], ft["dataset"], ft["patient"], strict=True)}
    test = set(st.loc[st["slice"] == "test", "stay"])
    pat_of = dict(zip(st["stay"], st["patient"], strict=True))
    member = Path(a.member_out)                 # the membership check: FT train stays against TEST stays
    for root in (out, member):
        root.mkdir(parents=True, exist_ok=True)
        for f in AUX:
            shutil.copy2(tokens / f, root / f)
    sizes: dict = {}
    for d in EXTERNAL:
        got, held = {"train": [], "tuning": []}, []
        for sp in SPLITS.split(","):
            t = pq.read_table(tokens / d / f"{sp}.parquet")
            schema = t.schema
            for r in t.to_pylist():
                if r["stay_id"] in part:
                    got[part[r["stay_id"]]].append(site_unknown(r, lay.site_pos, other, unknown))
                elif r["stay_id"] in test:
                    held.append(site_unknown(r, lay.site_pos, other, unknown))
        (out / d).mkdir(exist_ok=True)
        (member / d).mkdir(exist_ok=True)
        pq.write_table(pa.Table.from_pylist(got["train"], schema=schema), member / d / "train.parquet")
        pq.write_table(pa.Table.from_pylist(held, schema=schema), member / d / "held_out.parquet")
        sizes[f"member/{d}"] = {"train": len(got["train"]), "held_out": len(held)}
        for sp, rs in got.items():
            pq.write_table(pa.Table.from_pylist(rs, schema=schema), out / d / f"{sp}.parquet")
            sizes[f"{d}/{sp}"] = {"stays": len(rs), "patients": len({pat_of[r["stay_id"]] for r in rs}),
                                  "windows": sum(len(r["windows"]) for r in rs),
                                  "tokens": sum(len(w) for r in rs for w in r["windows"]),
                                  "sha256": _sha(out / d / f"{sp}.parquet")}
    if set(part) != {r for d in EXTERNAL for sp in ("train", "tuning") for r in
                     pq.read_table(out / d / f"{sp}.parquet", columns=["stay_id"]).column(0).to_pylist()}:
        raise SystemExit("the FT stays and the shards differ")
    w_ext = sum(sizes[f"{d}/train"]["windows"] for d in EXTERNAL)
    t_ext = sum(sizes[f"{d}/train"]["tokens"] for d in EXTERNAL)
    w7 = sum(x["windows"] for k, x in seven.items() if k.endswith("/train"))
    t7 = sum(x["tokens"] for k, x in seven.items() if k.endswith("/train"))
    cells = {}
    for e in GRID["passes"]:
        q = e * w_ext / w7
        budget = e * t_ext + q * t7
        for lr in GRID["lr"]:
            cells[f"lr{lr:g}_p{e:g}"] = {"lr": lr, "ft_passes": e, "passes": {"default": q, **{d: e for d in EXTERNAL}},
                                         "tokens": budget, "max_steps": int(np.ceil(budget / STEP_TOKENS)),
                                         "warmup_steps": max(1, int(round(0.02 * np.ceil(budget / STEP_TOKENS))))}
    rep = {"split_sha256": _sha(split_dir / "split.json"), "val": VAL, "step_tokens": STEP_TOKENS, "sizes": sizes,
           "ft_train_windows": w_ext, "ft_train_tokens": t_ext, "seven_train_windows": w7, "seven_train_tokens": t7,
           "cells": cells}
    (split_dir / "adapt.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


# -- analysis (17b; local, CPU, after the host's outputs are downloaded) ----------------------------------------------
DESIGNS = ("R", "L24e")
ARMS = ("Z1", "F1")
MIN_RECAL = 10                                  # LABEL positives below which the recalibration is the intercept alone
MIN_TEST = 10                                   # TEST positives below which an endpoint is descriptive only
CURVE_N, CURVE_DRAWS = (25, 50, 100, 200), 20
CLIN = ("sofa", "news", "mews", "qsofa", "sirs")
RANKED = ("Z0", "Z1", "F1", "F1S", "lgbm_pooled", "logistic_pooled", "tabpfn_v3_5", *CLIN, "lgbm_label",
          "lgbm_ftlabel", "lgbm_local")
LOCAL = ("lgbm_label", "lgbm_ftlabel", "lgbm_local")


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def recal_fit(y, p, w) -> dict | None:
    """17b's recalibration on LABEL anchors: logit(p') = a + b logit(p) (``toolkit.recalibration.fit_logistic``), or
    the intercept alone (b = 1) below ``MIN_RECAL`` positives; None without a positive or a negative."""
    from icutwin.toolkit import recalibration as rc

    ok = np.isfinite(y) & np.isfinite(p)
    y, p, w = y[ok], p[ok], w[ok]
    if not 0 < y.sum() < len(y):
        return None
    if y.sum() >= MIN_RECAL:
        return {**rc.fit_logistic(y, p, w), "intercept_only": False}
    x, a = _logit(p), 0.0
    for _ in range(100):                        # Newton on the intercept, the logit as offset
        q = 1 / (1 + np.exp(-np.clip(a + x, -35, 35)))
        step = (w * (y - q)).sum() / max((w * q * (1 - q)).sum(), 1e-12)
        a += step
        if abs(step) < 1e-10:
            break
    return {"kind": "logistic", "a": float(a), "b": 1.0, "n": int(len(y)), "positives": int(y.sum()),
            "intercept_only": True}


def recal_apply(m: dict | None, p):
    from icutwin.toolkit import recalibration as rc

    return np.full(len(p), np.nan) if m is None else rc.transform(m, np.asarray(p, float))


def _estimate(dd: Path, prog: str, design: str):
    """The arm's estimate per anchor: E2 for amendment 16a's hazard endpoints, the 128-future share otherwise."""
    import pandas as pd

    sys_path_ext()
    from paper_external import HAZARD

    if prog in HAZARD[design]:
        e = pd.read_parquet(dd / "estimates.parquet")
        e = e[(e["program"] == prog) & (e["estimator"] == "e2")]
    else:
        e = pd.read_parquet(dd / "predictions.parquet")
        e = e[e["program"] == prog]
    return e.drop_duplicates("anchor").set_index("anchor")["p"]


def sys_path_ext() -> None:
    p = str(Path(__file__).resolve().parent)
    if p not in sys.path:
        sys.path.insert(0, p)


def _t_min(an, tokens: Path):
    """Each anchor's prediction time on amendment 16a's encoding (``paper_rollouts.py``'s ``t_min``)."""
    from icutwin.eval import bench
    from icutwin.tokenization.vocab import Vocab

    v = Vocab.load(tokens / "vocab.json")
    rows = bench.real_rows(tokens, SPLITS, set(an["twin"].astype(str)), EXTERNAL)
    ids = {r["stay_id"]: r["ids"] for r in rows}
    return np.array([bench.anchor_time(bench.unknown_site(bench.prompt(ids[t], v, t, h), v), v, t)
                     for t, h in zip(an["twin"].astype(str), an["hours"], strict=True)], dtype=np.int64)


class Local:
    """17b's local references on one store and design: LightGBM with A1's recipe (``eval.baselines``: features,
    training rows, fit) trained on the stays of a database's slices; early stopping on a tenth of the training stays'
    patients (seed 17), as the 16a local reference."""

    def __init__(self, store, design: str, slice_of_stay: dict, patient_of: dict, n_jobs: int):
        from icutwin.eval import baselines as bl
        from icutwin.factory import programs as fpr

        self.st, self.base, self.n_jobs = store, ("L24" if design.startswith("L") else design), n_jobs
        self.sid = store.stays["stay_id"].astype(str).to_numpy()
        self.sdb = store.stays["dataset"].astype(str).to_numpy()
        self.slice = np.array([slice_of_stay.get(s, "") for s in self.sid])
        self.patient = np.array([patient_of.get(s, s) for s in self.sid])
        self.qidx = {s: i for i, s in enumerate(self.sid)}
        self.fz = bl.Featurizer(bl.feature_channels(store, 64)).fit_levels(store)
        self.ev = fpr.Evaluator(store)
        self.progs = {_pkey(p): p for p in fpr.catalog(store.channels)}
        self.rows: dict = {}

    def lab(self, prog: str, rows):
        from icutwin.eval import baselines as bl

        st = self.st
        if prog == "remaining_los_gt_7d":
            rem = (st.los[rows.stay] - rows.t) / 60.0
            cens = np.isin(st.end_kind[rows.stay], ("censored", "unknown")) & (rem <= 168)
            return np.where(rem > 168, 1.0, np.where(cens, np.nan, 0.0))
        return bl.labels(st, self.progs[prog], rows, self.ev)

    def train_rows(self, d: str, slices: tuple):
        """The training rows (and features) of a database's slices, once per (database, slices)."""
        from icutwin.eval import baselines as bl

        key = (d, slices)
        if key not in self.rows:
            idx = np.flatnonzero((self.sdb == d) & np.isin(self.slice, slices))
            r = bl.training_rows(self.st, self.base, 4, seed=0, stays=idx)
            self.rows[key] = (r, self.fz(self.st, r, self.ev))
        return self.rows[key]

    def anchor_x(self, an):
        from icutwin.factory import labels as fl

        rows = fl.Rows(np.array([self.qidx[str(s)] for s in an["twin"]], dtype=np.int64),
                       an["t_min"].to_numpy(dtype=np.int64))
        return rows, self.fz(self.st, rows, self.ev)

    def fit(self, prog: str, d: str, slices: tuple, stays=None, seed: int = SEED):
        """A model on the slices' rows (``stays``: only these store rows' stays), or None (one class)."""
        from icutwin.eval import baselines as bl

        r, x = self.train_rows(d, slices)
        keep = np.ones(len(r.stay), bool) if stays is None else np.isin(r.stay, stays)
        y = self.lab(prog, r)
        pats = np.unique(self.patient[r.stay[keep]])
        rng = np.random.default_rng(seed)
        val = rng.choice(pats, size=max(1, len(pats) // 10), replace=False)
        is_val = np.isin(self.patient[r.stay], val)
        tr, va = keep & ~is_val & np.isfinite(y), keep & is_val & np.isfinite(y)
        if len(np.unique(y[tr])) < 2:
            return None
        return bl.fit_lgbm(x[tr], y[tr], x[va], y[va], seed=0, n_jobs=self.n_jobs)


def _pkey(p) -> str:
    sys_path_ext()
    from paper_external import _key

    return _key(p)


def cmd_analyze(a) -> None:
    """17b's arms on the TEST anchors, per design, database and endpoint amendment 16a scored (its availability rule):
    Z0 (16a's predictions), Z1 and F1 (the host's futures), Z1+R and F1+R (recalibrated on the LABEL anchors), F1+S
    (stacked with the LABEL-only LightGBM, cross-fitted on LABEL by patient), the local LightGBMs (LABEL; FT and
    LABEL), amendment 16a's transferred learners, TabPFN-3.5, clinical scores and its local cross-fitted reference.
    Writes ``<out>/scores.parquet`` (per TEST anchor and endpoint; local only), ``<out>/ext17.json`` (metrics with
    ``paper_comparators.evaluate``'s paired differences, F1+R as the reference, and each arm against Z0; O/E before and
    after recalibration; the maps; the label-efficiency curves), and the agreement, OOV, grid and membership read-outs."""
    import pandas as pd

    from icutwin.eval import bench
    from icutwin.factory import store as fst

    sys_path_ext()
    from paper_comparators import evaluate
    from paper_external import labels as ext_labels

    out, ext = Path(a.out), Path(a.ext)
    st_all = pd.read_parquet(out / "stays.parquet")
    slice_of_stay = dict(zip(st_all["stay"], st_all["slice"], strict=True))
    patient_of = dict(zip(st_all["stay"], st_all["patient"], strict=True))
    gate = json.loads((ext / "gate.json").read_text(encoding="utf-8"))
    clin = pd.read_parquet(ext / "clinical" / "scores.parquet")
    store = fst.Store(Path(a.store))
    rep: dict = {"git": __import__("paper_a1")._sha(), "seed": SEED, "boot": a.boot,
                 "split_sha256": _sha(out / "split.json"), "designs": {}, "curves": {}}
    frames = []
    for design in DESIGNS:
        base = "L24" if design.startswith("L") else design
        an16 = pd.read_parquet(ext / design / "anchors.parquet").drop_duplicates("anchor").set_index("anchor")
        test = pd.read_parquet(out / f"test_{design}.parquet")
        test["t_min"] = test["anchor"].map(an16["t_min"]).astype("int64")
        lab_an = pd.read_parquet(out / f"label_{design}.parquet")
        lab_an["t_min"] = _t_min(lab_an, Path(a.tokens))
        y_test = pd.read_parquet(ext / design / "labels.parquet")
        y_label = ext_labels(Path(a.store), lab_an, design)
        loc = Local(store, design, slice_of_stay, patient_of, a.n_jobs)
        rows_te, x_te = loc.anchor_x(test)
        rows_lb, x_lb = loc.anchor_x(lab_an)
        comp16 = pd.concat([pd.read_parquet(ext / design / f) for f in
                            ("baseline_predictions.parquet", "tabpfn_v3_5_predictions.parquet",
                             "local_predictions.parquet")], ignore_index=True)
        scd = clin[clin["design"] == base].set_index("anchor")
        for r in gate["endpoints"].values():
            if r["design"] != design or not r["available"]:
                continue
            d, prog = r["dataset"], r["program"]
            mt, ml = (test["dataset"] == d).to_numpy(), (lab_an["dataset"] == d).to_numpy()
            te, lb = test[mt].reset_index(drop=True), lab_an[ml].reset_index(drop=True)
            yt = y_test[y_test["program"] == prog].drop_duplicates("anchor").set_index("anchor")["label"]
            yl = y_label[y_label["program"] == prog].drop_duplicates("anchor").set_index("anchor")["label"]
            x = pd.DataFrame({"design": design, "dataset": d, "program": prog, "anchor": te["anchor"],
                              "stay": te["twin"].astype(str), "patient": te["twin"].astype(str).map(patient_of),
                              "y": te["anchor"].map(yt).to_numpy(float), "w": te["weight"].to_numpy(float)})
            ylb, wlb = lb["anchor"].map(yl).to_numpy(float), lb["weight"].to_numpy(float)
            x["Z0"] = x["anchor"].map(_estimate(ext / design, prog, design))
            res: dict = {"recalibration": {}}
            arm_lb = {}
            for arm in ARMS:
                x[arm] = x["anchor"].map(_estimate(out / arm / f"test_{design}", prog, design))
                arm_lb[arm] = lb["anchor"].map(_estimate(out / arm / f"label_{design}", prog, design)).to_numpy(float)
                m = recal_fit(ylb, arm_lb[arm], wlb)
                res["recalibration"][arm] = m
                x[f"{arm}R"] = recal_apply(m, x[arm].to_numpy(float))
            for c in ("lgbm_pooled", "logistic_pooled", "tabpfn_v3_5", "lgbm_local"):
                s = comp16[(comp16["program"] == prog) & (comp16["model"] == c)].drop_duplicates("anchor")
                x[c] = x["anchor"].map(s.set_index("anchor")["p"])
            for c in CLIN:
                if c in scd.columns:
                    x[c] = x["anchor"].map(scd[c])
            # the local LightGBMs: on LABEL; on FT and LABEL; LABEL-only cross-fitted on LABEL (for stacking)
            for name, slices in (("lgbm_label", ("label",)), ("lgbm_ftlabel", ("ft", "label"))):
                mdl = loc.fit(prog, d, slices)
                x[name] = mdl.predict_proba(x_te[mt])[:, 1] if mdl is not None else np.nan
            cf = np.full(len(lb), np.nan)
            pats = lb["twin"].astype(str).map(patient_of).to_numpy()
            fo = folds_of(pats, SEED)
            fold = np.array([fo[p] for p in pats])
            for k in range(5):
                tr_st = [loc.qidx[s] for s in lb["twin"].astype(str)[fold != k]]
                rest = np.flatnonzero((loc.sdb == d) & (loc.slice == "label")
                                      & ~np.isin(loc.patient, pats[fold == k]))
                mdl = loc.fit(prog, d, ("label",), stays=np.union1d(tr_st, rest))
                if mdl is not None:
                    cf[fold == k] = mdl.predict_proba(x_lb[ml][fold == k])[:, 1]
            okl = np.isfinite(ylb) & np.isfinite(arm_lb["F1"]) & np.isfinite(cf)
            x["F1S"] = np.nan
            if okl.any() and ylb[okl].sum() >= MIN_RECAL and np.isfinite(x["lgbm_label"]).all():
                from sklearn.linear_model import LogisticRegression

                z_lb = np.column_stack([_logit(arm_lb["F1"][okl]), _logit(cf[okl])])
                stk = LogisticRegression(C=1e6, max_iter=1000).fit(z_lb, ylb[okl], sample_weight=wlb[okl])
                z_te = np.column_stack([_logit(x["F1"]), _logit(x["lgbm_label"])])
                ok_te = np.isfinite(z_te).all(axis=1)
                x.loc[ok_te, "F1S"] = stk.predict_proba(z_te[ok_te])[:, 1]
                res["stacking"] = {"coef": stk.coef_[0].tolist(), "intercept": float(stk.intercept_[0]),
                                   "label_anchors": int(okl.sum()), "label_positives": int(ylb[okl].sum())}
            x = x[np.isfinite(x["y"])].reset_index(drop=True)
            y, w = x["y"].to_numpy(float), x["w"].to_numpy(float)
            stay, ds = x["stay"].to_numpy(), np.full(len(x), d)
            cols = [c for c in ("Z0", "Z1", "F1", "Z1R", "F1R", "F1S", "lgbm_pooled", "logistic_pooled",
                                "tabpfn_v3_5", *CLIN, "lgbm_label", "lgbm_ftlabel", "lgbm_local") if c in x.columns]
            preds = {c: x[c].to_numpy(float) for c in cols}
            res.update({"anchors": int(len(x)), "positives": int(y.sum()),
                        "label_anchors": int(np.isfinite(ylb).sum()), "label_positives": int(np.nansum(ylb)),
                        "descriptive_only": bool(y.sum() < MIN_TEST),
                        "predictors": evaluate(y, w, stay, ds, preds, "F1R", a.boot),
                        "arms_vs_z0": {arm: evaluate(y, w, stay, ds, {arm: preds[arm], "Z0": preds["Z0"]}, arm,
                                                     a.boot)["Z0"] for arm in ("Z1", "F1")}})
            res["arms_vs_z0"]["F1R"] = res["predictors"].get("Z0")
            res["calibration"] = {c: {k: bench.binary_metrics(y[np.isfinite(preds[c])],
                                                              np.clip(preds[c][np.isfinite(preds[c])], 0, 1),
                                                              w[np.isfinite(preds[c])]).get(k)
                                      for k in ("observed_over_expected", "calibration_slope", "brier")}
                                  for c in ("Z0", "Z1", "F1", "Z1R", "F1R", "F1S") if np.isfinite(preds[c]).sum() >= 30}
            rep["designs"].setdefault(design, {}).setdefault(d, {})[prog] = res
            rep["curves"].setdefault(design, {}).setdefault(d, {})[prog] = label_curve(
                loc, prog, d, lb, ylb, wlb, arm_lb["F1"], x, x_te[mt][np.isin(te["anchor"], x["anchor"])],
                a.curve_draws)
            frames.append(x)
            print(json.dumps({f"{design}:{d}:{prog}": {c: round(res["predictors"][c]["auroc"], 3)
                                                       for c in ("Z0", "Z1", "F1", "F1S", "lgbm_ftlabel")
                                                       if c in res["predictors"]}}), flush=True)
    pd.concat(frames, ignore_index=True).to_parquet(out / "scores.parquet", index=False)
    rep["readouts"] = readouts(out, Path(a.root))
    rep["ranks"] = ranks(rep)
    (out / "ext17.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


def label_curve(loc: Local, prog: str, d: str, lb, ylb, wlb, f1_lb, x, x_te, draws: int = CURVE_DRAWS) -> dict:
    """17b's label efficiency on the TEST anchors: n LABEL stays (25, 50, 100, 200, all), 20 draws each (seed 17):
    F1 recalibrated on the drawn stays' LABEL anchors (AUROC and O/E) and LightGBM trained on the drawn stays (AUROC);
    the median over draws; the n at which LightGBM's median reaches F1+R's AUROC (A1's interpolation in log10 n)."""
    from icutwin.eval import bench

    y, w = x["y"].to_numpy(float), x["w"].to_numpy(float)
    f1 = x["F1"].to_numpy(float)
    ok = np.isfinite(f1)
    stays = np.flatnonzero((loc.sdb == d) & (loc.slice == "label"))
    lb_stay = np.array([loc.qidx[s] for s in lb["twin"].astype(str)])
    rng = np.random.default_rng(SEED)
    ns = [n for n in CURVE_N if n < len(stays)] + [len(stays)]
    cur: dict = {}
    for n in ns:
        auc_l, auc_r, oe_r = [], [], []
        for k in range(draws if n < len(stays) else 1):
            pick = rng.choice(stays, size=n, replace=False)
            mdl = loc.fit(prog, d, ("label",), stays=pick, seed=SEED + k)
            if mdl is not None:
                p = mdl.predict_proba(x_te)[:, 1]
                auc_l.append(bench.auroc_or_none(y, p, w))
            sel = np.isin(lb_stay, pick)
            m = recal_fit(ylb[sel], f1_lb[sel], wlb[sel])
            if m is not None and ok.any():
                q = recal_apply(m, f1[ok])
                auc_r.append(bench.auroc_or_none(y[ok], q, w[ok]))
                oe_r.append(bench.binary_metrics(y[ok], q, w[ok]).get("observed_over_expected"))
        med = lambda v: float(np.median([u for u in v if u is not None])) if any(u is not None for u in v) else None  # noqa: E731
        cur[str(n)] = {"draws": draws if n < len(stays) else 1, "lgbm_auroc": med(auc_l),
                       "f1r_auroc": med(auc_r), "f1r_oe": med(oe_r), "lgbm_fitted": len(auc_l)}
    target = bench.auroc_or_none(y[ok], f1[ok], w[ok]) if ok.any() else None
    curve = [(int(n), v["lgbm_auroc"]) for n, v in cur.items() if v["lgbm_auroc"] is not None]
    status, n_match = "not reached", None
    if target is not None and curve:
        first = next((i for i, (_n, v) in enumerate(curve) if v >= target), None)
        if first == 0:
            status, n_match = "at most", float(curve[0][0])
        elif first is not None:
            (n0, a0), (n1, a1) = curve[first - 1], curve[first]
            status = "interpolated"
            n_match = float(10 ** (np.log10(n0) + (target - a0) / (a1 - a0) * (np.log10(n1) - np.log10(n0))))
    return {"label_stays": int(len(stays)), "f1_auroc": target, "points": cur, "status": status, "n_match": n_match}


def readouts(out: Path, root: Path) -> dict:
    """The label agreement under the improved encoding (paper_a1's alignment control per arm and anchor set), the OOV
    shares before and after the back-off, the grid (choice and forgetting) and the membership check."""
    def load(p: Path):
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    r: dict = {"alignment": {}, "oov": {}}
    for arm in ARMS:
        for s in ("test", "label"):
            for design in DESIGNS:
                a1 = load(out / arm / f"{s}_{design}" / "a1.json")
                if a1:
                    r["alignment"][f"{arm}:{s}_{design}"] = {"min": a1.get("alignment_control_min"),
                                                            "programs": a1.get("alignment_control")}
    sys_path_ext()
    from paper_external import _ids, _oov

    from icutwin.tokenization.vocab import Vocab

    for name in ("ext_r4", "ext_r4b"):                 # per database and domain: the share of events sent to OOV
        tokens = Path("data/tokens") / name
        if not (tokens / "vocab.json").exists():
            continue
        v = Vocab.load(tokens / "vocab.json")
        r["oov"][name] = {d: _oov(v, np.bincount(_ids(tokens, [d], SPLITS.split(",")), minlength=len(v.tokens)))
                          for d in EXTERNAL}
    r["grid"] = load(out / "grid.json")
    r["adapt"] = load(out / "adapt.json")
    r["membership"] = {who.name: load(who / "p1.json") for who in sorted((out / "membership").glob("*"))} \
        if (out / "membership").exists() else None
    return r


def ranks(rep: dict) -> dict:
    """Family 24: amendment 15's rank summary over the TEST pairs (``paper_stats.rank_summary``, Z1+R and F1+R share
    their arms' AUROC and are left out), with and without the local models; F1 is the reference."""
    sys_path_ext()
    from paper_stats import rank_summary

    tasks, rows = [], []
    for design in DESIGNS:
        for d, progs in sorted(rep["designs"].get(design, {}).items()):
            for p, x in sorted(progs.items()):
                if x["descriptive_only"]:
                    continue
                tasks.append({"design": design, "dataset": d, "program": p})
                rows.append([(x["predictors"].get(n) or {}).get("auroc") for n in RANKED])
    keep = [j for j in range(len(RANKED)) if rows and all(row[j] is not None for row in rows)]
    names = [RANKED[j] for j in keep]
    mat = np.array([[row[j] for j in keep] for row in rows], dtype=float)
    out = {"tasks": tasks, "dropped": [n for n in RANKED if n not in names], "names": names, "auroc": mat.tolist()}
    if len(tasks) >= 2:
        out["all"] = rank_summary(mat, names, "F1")
        nl = [j for j, n in enumerate(names) if n not in LOCAL]
        out["without_local"] = rank_summary(mat[:, nl], [names[j] for j in nl], "F1")
    return out


def folds_of(patients, seed: int) -> dict:
    pats = np.unique(np.asarray(patients))
    rng = np.random.default_rng(seed)
    return dict(zip(rng.permutation(pats), np.arange(len(pats)) % 5, strict=True))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("split")
    g.add_argument("--tokens", default="data/tokens/ext_r4")
    g.add_argument("--store", default="data/factory/stores/ext_test")
    g.add_argument("--ext", default="runs/paper/ext")
    g.add_argument("--clif", default="data/clif_canonical/full")
    g.add_argument("--out", default="runs/paper/ext17")
    g = sub.add_parser("trainroot")
    g.add_argument("--tokens", default="data/tokens/ext_r4b")
    g.add_argument("--split-dir", default="runs/paper/ext17")
    g.add_argument("--seven-index", default="data/tokens/full_r4/index.json")
    g.add_argument("--out", default="data/tokens/adapt_r4")
    g.add_argument("--member-out", default="data/tokens/adapt_r4_member")
    g = sub.add_parser("analyze")
    g.add_argument("--out", default="runs/paper/ext17")
    g.add_argument("--ext", default="runs/paper/ext")
    g.add_argument("--root", default="runs/paper")
    g.add_argument("--tokens", default="data/tokens/ext_r4")
    g.add_argument("--store", default="data/factory/stores/ext_test")
    g.add_argument("--boot", type=int, default=1000)
    g.add_argument("--n-jobs", type=int, default=6)
    g.add_argument("--curve-draws", type=int, default=CURVE_DRAWS)
    a = ap.parse_args(argv)
    {"split": cmd_split, "trainroot": cmd_trainroot, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
