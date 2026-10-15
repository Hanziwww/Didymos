"""Paper P amendment 16a (``docs/p1_paper_protocol.md``): transfer to two databases never seen in training (RICD 3.0,
zigongICU 1.1). Local, CPU; aggregates only leave the machine.

``design``: the anchors of both designs on the external store, with A1's functions and seed 16 (``eval.bench``):
* ``R``: A1's rolling design, 500 per database (``factory.oracle.choose``: length-of-stay bands, half enriched,
  inverse-probability weights);
* ``L24e``: the 24-hour landmark (every stay alive in the ICU at 24 h), then a case-cohort sample of it: up to 350
  in-hospital deaths and 350 other stays per database at random, each stratum weighted by the inverse of its sampling
  fraction (A1's ``hospital_death@0`` label);
* F1's clock check on both: an anchor whose prompt clock and hour disagree by more than 6 h and whose prompt continues
  a context window is left out and counted (``paper_f1_subsets.clock_ok``).
Writes ``<out>/<design>/{anchors.parquet,design.json}`` as ``paper_rollouts.py --dry`` writes them (the host redraws
the same anchors and checks them against these), ``<out>/subsets/<name>.parquet`` (anchor, weight, stratum; the
host's ``--subset``) and ``<out>/<name>/labels.parquet``: the real label of every program at every kept anchor.

``gate``: the pre-run gate, recorded before any future is drawn (``docs/paper/external_gate.md`` and
``<out>/gate.json``): grammar acceptance and the re-encode round trip (the tokenizer validation), OOV shares per
domain against the training databases' held-out stays, value bins per code against the training deciles, the
alignment control (each anchor's real stay decoded as a rollout member reproduces its labels), label prevalence per
endpoint and the availability decision under the pre-registered rule (the defining source has timing to the hour, and
at least 20 anchors of the design are positive).

    python tools/paper_external.py design --tokens data/tokens/ext_r4 --store data/factory/stores/ext_test \\
        --out runs/paper/ext
    python tools/paper_external.py gate --tokens data/tokens/ext_r4 --store data/factory/stores/ext_test \\
        --out runs/paper/ext --report docs/paper/external_gate.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXTERNAL = ("ricd", "zigong")
SPLITS = "train,tuning,held_out"
SEED = 16
PER_DATASET = {"R": 500}
CASES = CONTROLS = 350
PRIMARY = {"R": ("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880", "aki_2@2880",
                 "sepsis_ase@1440", "hypotension@240", "hyperkalemia_top@1440", "thrombocytopenia@2880"),
           "L24e": ("icu_death@-1", "hospital_death@0", "remaining_los_gt_7d", "decompensation@1440")}
MIN_POSITIVE = 20
# the pre-registered availability rule's first half: the source that defines each endpoint, and whether its timing is
# to the hour (adapters/{ricd,zigong}.py); decided from the data's structure, before any label was computed
SOURCES = {
    "zigong": {
        "decompensation@1440": (True, "death at the hospital discharge (hours from admission)"),
        "imv_onset@1440": (False, "ventilation from the nursing chart, which records dates only"),
        "vasopressor_onset@1440": (True, "vasopressor prescriptions (hours)"),
        "rrt_onset@2880": (True, "haemofiltration replacement-fluid prescriptions (hours)"),
        "aki_2@2880": (False, "creatinine (hours) with urine output from the nursing chart (dates only)"),
        "sepsis_ase@1440": (False, "no culture data"),
        "hypotension@240": (False, "blood pressure from the nursing chart (dates only)"),
        "hyperkalemia_top@1440": (True, "laboratory potassium (hours)"),
        "thrombocytopenia@2880": (True, "laboratory platelets (hours)"),
        "icu_death@-1": (True, "ICU transfers and the hospital discharge (hours)"),
        "hospital_death@0": (True, "the hospital discharge (hours)"),
        "remaining_los_gt_7d": (True, "ICU transfers (hours)"),
    },
    "ricd": {
        "decompensation@1440": (True, "death at the last department discharge (hours)"),
        "imv_onset@1440": (False, "a daily ventilation flag on the SOFA record only"),
        "vasopressor_onset@1440": (True, "vasopressor orders (hours)"),
        "rrt_onset@2880": (False, "no renal replacement source"),
        "aki_2@2880": (True, "creatinine (hours) and urine output from monitoring (minutes)"),
        "sepsis_ase@1440": (False, "culture results not harmonized"),
        "hypotension@240": (True, "blood pressure from monitoring (minutes)"),
        "hyperkalemia_top@1440": (True, "laboratory potassium (hours)"),
        "thrombocytopenia@2880": (True, "laboratory platelets (hours)"),
        "icu_death@-1": (True, "the patient pathway (hours)"),
        "hospital_death@0": (True, "the last department discharge (hours)"),
        "remaining_los_gt_7d": (True, "the patient pathway (hours)"),
    },
}


def _key(p) -> str:
    return f"{p.name}@{p.horizon}"


def _stays(tokens: Path, store: Path, v):
    from icutwin.eval import bench
    from icutwin.factory import twins

    rows = bench.real_rows(tokens, SPLITS, bench.store_stay_ids(store), EXTERNAL)
    return rows, [(r["stay_id"], r["dataset"], twins.decode_labels(r["ids"], v, r["stay_id"])) for r in rows]


def _design(rows, stays, v, design: str, per_dataset: int, cap_hours: float, out: Path, tokens: Path, store: Path):
    """The design's anchors, prompts' lengths and prediction times, and its caps, written as ``paper_rollouts.py
    --dry`` writes them."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from icutwin.eval import bench
    from icutwin.factory import oracle

    anchors = bench.design_anchors(stays, v, design, per_dataset, SEED)
    ids = {r["stay_id"]: r["ids"] for r in rows}
    prompts = [bench.unknown_site(bench.prompt(ids[x.twin], v, x.twin, x.hours), v) for x in anchors]
    seqs: dict = {}
    for r in rows:
        seqs.setdefault(r["dataset"], []).append(r["ids"])
    caps = oracle.token_caps(seqs, v, cap_hours, 0.95, SEED)
    d = {f: [getattr(x, f) for x in anchors] for f in oracle.Anchor.__dataclass_fields__}
    d["prompt_len"] = [len(p) for p in prompts]
    d["t_min"] = [bench.anchor_time(p, v, x.twin) for p, x in zip(prompts, anchors, strict=True)]
    d["prompt_dataset"] = [x.dataset for x in anchors]
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(d), out / "anchors.parquet")
    (out / "design.json").write_text(json.dumps({"tokens": str(tokens), "split": SPLITS, "store": str(store),
                                                 "design": design, "per_dataset": per_dataset, "seed": SEED,
                                                 "cap_hours": cap_hours, "cap_q": 0.95, "caps": caps,
                                                 "prompt_transform": "unknown", "datasets": list(EXTERNAL),
                                                 "anchors": len(anchors)}, indent=1), encoding="utf-8")
    return prompts


def _clock_ok(an, prompts, v) -> np.ndarray:
    """F1's clock check (``paper_f1_subsets.clock_ok``) on prompts already built."""
    off = np.abs(an["t_min"].to_numpy() / 60.0 - an["hours"].to_numpy()) > 6.0
    cont = v.id("[CONT]")
    return ~(off & np.array([cont in p for p in prompts]))


def labels(store: Path, an, design: str):
    """The real label of every program the design scores (``paper_a1.applies``) at each anchor's prediction time;
    with the remaining ICU stay over 7 days as A1 reads it."""
    import pandas as pd

    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst

    sys_path_tools()
    from paper_a1 import applies

    real = fst.Store(store)
    qidx = {str(s): i for i, s in enumerate(real.stays["stay_id"].tolist())}
    st_q = np.array([qidx[str(x)] for x in an["twin"]], dtype=np.int64)
    tt = an["t_min"].to_numpy(dtype=np.int64)
    rows_q = fl.Rows(st_q, tt)
    ev = fpr.Evaluator(real)
    out = []
    for p in fpr.catalog(real.channels):
        if not applies(p, "L24" if design.startswith("L") else design) or p.label not in ("binary", "hosp_death"):
            continue
        b = ev.evaluate(p, rows_q)
        out.append(pd.DataFrame({"anchor": an["anchor"], "dataset": an["dataset"], "weight": an["weight"],
                                 "program": _key(p), "label": np.where(b.valid, b.y, np.nan).astype(float)}))
    rem = (real.los[st_q] - tt) / 60.0
    cens = np.isin(real.end_kind[st_q], ("censored", "unknown"))
    y = np.where(rem > 168, 1.0, np.where(cens & (rem <= 168), np.nan, 0.0))
    out.append(pd.DataFrame({"anchor": an["anchor"], "dataset": an["dataset"], "weight": an["weight"],
                             "program": "remaining_los_gt_7d", "label": y}))
    return pd.concat(out, ignore_index=True)


def sys_path_tools() -> None:
    import sys

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)


def cmd_design(a) -> None:
    import pandas as pd

    from icutwin.tokenization.vocab import Vocab

    tokens, store, out = Path(a.tokens), Path(a.store), Path(a.out)
    v = Vocab.load(tokens / "vocab.json")
    rows, stays = _stays(tokens, store, v)
    rep = {"stays": {d: sum(1 for r in rows if r["dataset"] == d) for d in EXTERNAL}, "seed": SEED}
    (out / "subsets").mkdir(parents=True, exist_ok=True)
    # R: A1's rolling design
    prompts = _design(rows, stays, v, "R", PER_DATASET["R"], 72.0, out / "R", tokens, store)
    an = pd.read_parquet(out / "R" / "anchors.parquet")
    ok = _clock_ok(an, prompts, v)
    sub = pd.DataFrame({"anchor": an["anchor"][ok], "weight": an["weight"][ok], "stratum": "rolling"})
    sub.to_parquet(out / "subsets" / "R.parquet", index=False)
    rep["R"] = {"anchors": int(len(an)), "excluded_clock": int((~ok).sum()), "kept": int(ok.sum()),
                "by_dataset": an[ok]["dataset"].value_counts().sort_index().to_dict()}
    lab = labels(store, an[ok].reset_index(drop=True), "R")
    lab.to_parquet(out / "R" / "labels.parquet", index=False)
    # L24e: the 24-hour landmark, then its case-cohort sample per database
    prompts = _design(rows, stays, v, "L24", 10 ** 9, 168.0, out / "L24e", tokens, store)
    an = pd.read_parquet(out / "L24e" / "anchors.parquet")
    ok = _clock_ok(an, prompts, v)
    full = an[ok].reset_index(drop=True)
    hd = labels(store, full, "L24")
    hd = hd[hd["program"] == "hospital_death@0"].set_index("anchor")["label"]
    rng = np.random.default_rng(SEED)
    parts, cc = [], {}
    for d in EXTERNAL:
        x = full[full["dataset"] == d]
        y = hd.loc[x["anchor"]].to_numpy()
        for stratum, sel, n_max in (("case", y == 1, CASES), ("control", y == 0, CONTROLS)):
            pool = x[sel]
            pick = pool.iloc[np.sort(rng.choice(len(pool), size=min(n_max, len(pool)), replace=False))]
            parts.append(pd.DataFrame({"anchor": pick["anchor"], "weight": len(pool) / max(len(pick), 1),
                                       "stratum": stratum}))
            cc[f"{d}_{stratum}"] = {"pool": int(len(pool)), "sampled": int(len(pick)),
                                    "weight": len(pool) / max(len(pick), 1)}
        cc[f"{d}_unlabelled"] = int(np.isnan(y).sum())
    sub = pd.concat(parts, ignore_index=True)
    sub.to_parquet(out / "subsets" / "L24e.parquet", index=False)
    rep["L24e"] = {"landmark_anchors": int(len(an)), "excluded_clock": int((~ok).sum()), "case_cohort": cc,
                   "anchors": int(len(sub))}
    kept = an.merge(sub[["anchor", "weight"]].rename(columns={"weight": "w"}), on="anchor")
    kept["weight"] = kept["w"]
    lab = labels(store, kept.drop(columns="w").reset_index(drop=True), "L24")
    lab.to_parquet(out / "L24e" / "labels.parquet", index=False)
    (out / "subsets" / "subsets.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=float))


# -- the gate ----------------------------------------------------------------------------------------------------
def _ids(tokens: Path, datasets, splits) -> np.ndarray:
    import pyarrow.parquet as pq

    parts = []
    for d in datasets:
        for sp in splits:
            f = tokens / d / f"{sp}.parquet"
            if f.exists():
                parts.append(pq.read_table(f, columns=["ids"]).column("ids").combine_chunks().values.to_numpy())
    return np.concatenate(parts) if parts else np.zeros(0, np.int32)


def _oov(v, counts: np.ndarray) -> dict:
    """Per domain: events (event tokens of the domain) and the share that are its OOV token."""
    out: dict = {}
    for i, tok in enumerate(v.tokens):
        m = v.meta[i]
        if m.get("cls") != "event" or not counts[i]:
            continue
        dom = tok.split("//", 1)[0]
        r = out.setdefault(dom, {"events": 0, "oov": 0})
        r["events"] += int(counts[i])
        r["oov"] += int(counts[i]) if tok == f"{dom}//OOV" else 0
    return {d: {**r, "share": r["oov"] / r["events"]} for d, r in sorted(out.items())}


def _bins(v, counts: np.ndarray, min_n: int = 200) -> list:
    """Per numeric code (tokens ``<code>|q<k>``): the share of its values in the lowest and the highest bin. The bins
    are the training stays' quantiles, so each holds about 1/k of the training values; a share above one half in an
    end bin flags a likely unit or definition difference."""
    by: dict = {}
    for i, tok in enumerate(v.tokens):
        if "|q" not in tok:
            continue
        code, b = tok.rsplit("|q", 1)
        if b.isdigit():
            by.setdefault(code, {})[int(b)] = int(counts[i])
    out = []
    for code, bins in by.items():
        n = sum(bins.values())
        k = max(bins) + 1
        if n < min_n or k < 3:            # one or two bins (a value-less code, a flag): no end bin to read
            continue
        lo, hi = bins.get(0, 0) / n, bins.get(k - 1, 0) / n
        out.append({"code": code, "n": n, "bins": k, "low": lo, "high": hi, "flag": max(lo, hi) > 0.5})
    return sorted(out, key=lambda r: -max(r["low"], r["high"]))


def _urine(clif: Path) -> dict:
    """Urine output as charted (canonical CLIF): the hours between a stay's consecutive entries and the volume per
    entry, quartiles; the KDIGO urine criteria sum volumes over 6- to 24-hour windows."""
    import duckdb

    c = duckdb.connect()
    r = c.execute(f"""
        WITH u AS (SELECT hospitalization_id, recorded_dttm, output_volume FROM read_parquet('{clif.as_posix()}')
                   WHERE output_group = 'urine'),
             g AS (SELECT *, date_diff('minute', lag(recorded_dttm) OVER (PARTITION BY hospitalization_id
                                                                            ORDER BY recorded_dttm), recorded_dttm) AS gap
                   FROM u)
        SELECT quantile_cont(gap / 60.0, [0.25, 0.5, 0.75]), quantile_cont(output_volume, [0.25, 0.5, 0.75]), count(*)
        FROM g""").fetchone()
    return {"gap_hours_q": r[0], "volume_ml_q": r[1], "entries": r[2]}


def _alignment(tokens: Path, store: Path, out: Path, design: str, workers: int) -> dict:
    """A1's alignment control on the kept anchors: the real stay decoded as a rollout member on its own clock
    (``paper_a1.py``) against the real store's label, for every binary program without a clinical cut."""
    import pandas as pd

    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst

    sys_path_tools()
    from paper_a1 import applies

    an = pd.read_parquet(out / design / "anchors.parquet")
    sub = pd.read_parquet(out / "subsets" / f"{design}.parquet")
    an = an[an["anchor"].isin(sub["anchor"])].reset_index(drop=True)
    from icutwin.eval import bench

    rows = {r["stay_id"]: r["ids"] for r in bench.real_rows(tokens, SPLITS, set(an["twin"].astype(str)), EXTERNAL)}
    work = out / "work" / f"control_{design}"
    if not (work / "manifest.json").exists():
        seqs = [{"stay_id": f"{x}#real", "ids": np.asarray(rows[str(s)], np.int32), "clock": str(s),
                 "split": "control", "weight": 1.0, "cell": "control"}
                for x, s in zip(an["anchor"], an["twin"], strict=True)]
        fst.build(seqs, tokens / "vocab.json", work, "seg_0000", {"kind": "external_control", "design": design},
                  persistence=tokens / "rank_persistence.json", workers=workers, chunk=100)
    ro, real = fst.Store(work), fst.Store(store)
    ridx = {s: i for i, s in enumerate(ro.stays["stay_id"].tolist())}
    qidx = {str(s): i for i, s in enumerate(real.stays["stay_id"].tolist())}
    ctl = np.array([ridx[f"{x}#real"] for x in an["anchor"]], dtype=np.int64)
    st_q = np.array([qidx[str(x)] for x in an["twin"]], dtype=np.int64)
    tt = an["t_min"].to_numpy(dtype=np.int64)
    ev_r, ev_q = fpr.Evaluator(ro), fpr.Evaluator(real)
    res = {}
    for p in fpr.catalog(ro.channels):
        if not applies(p, "L24" if design.startswith("L") else design) or p.label not in ("binary", "hosp_death"):
            continue
        bc, bq = ev_r.evaluate(p, fl.Rows(ctl, tt)), ev_q.evaluate(p, fl.Rows(st_q, tt))
        both = bc.valid & bq.valid
        res[_key(p)] = {"valid_agreement": float(np.mean(bc.valid == bq.valid)),
                        "label_agreement": float(np.mean(bc.y[both] == bq.y[both])) if both.any() else None,
                        "clinical_cut": "clinical_cut" in p.flags}
    agree = [min(r["label_agreement"], r["valid_agreement"]) for r in res.values()
             if r["label_agreement"] is not None and not r["clinical_cut"]]
    return {"anchors": int(len(an)), "min": float(min(agree)) if agree else None, "programs": res}


def cmd_gate(a) -> None:
    import pandas as pd

    from icutwin.tokenization.vocab import Vocab

    tokens, store, out = Path(a.tokens), Path(a.store), Path(a.out)
    v = Vocab.load(tokens / "vocab.json")
    gate: dict = {"seed": SEED, "rule": {"min_positive": MIN_POSITIVE, "timing": "to the hour"}}
    tok = json.loads((Path(a.qa) / "tokenizer_report.json").read_text(encoding="utf-8"))["datasets"]
    gate["tokenizer"] = {d: {k: tok[d][k] for k in ("stays", "fsm_rejected", "parser_rejected", "round_trip_failed",
                                                    "reencode_failed", "sampled_reencode_failed", "goal_mismatch",
                                                    "continuity_failed", "separation_failed")} for d in EXTERNAL}
    n = len(v.tokens)
    ref = np.bincount(_ids(Path(a.train_tokens), a.train_datasets, ("held_out",)), minlength=n)
    gate["oov"] = {"training_held_out": _oov(v, ref)}
    gate["bins"] = {}
    for d in EXTERNAL:
        c = np.bincount(_ids(tokens, (d,), SPLITS.split(",")), minlength=n)
        gate["oov"][d] = _oov(v, c)
        b = _bins(v, c)
        gate["bins"][d] = {"codes": len(b), "flagged": [r for r in b if r["flag"]]}
    gate["urine_output"] = {d: _urine(Path(a.clif) / d / "clif_output.parquet") for d in EXTERNAL}
    gate["designs"] = json.loads((out / "subsets" / "subsets.json").read_text(encoding="utf-8"))
    gate["alignment_control"] = {d: _alignment(tokens, store, out, d, a.workers) for d in ("R", "L24e")}
    avail: dict = {}
    for design, progs in PRIMARY.items():
        lab = pd.read_parquet(out / design / "labels.parquet")
        for d in EXTERNAL:
            for p in progs:
                x = lab[(lab["dataset"] == d) & (lab["program"] == p)]
                ok = x["label"].notna()
                pos = int((x["label"][ok] == 1).sum())
                prev = float(np.average(x["label"][ok], weights=x["weight"][ok])) if ok.any() else None
                hourly, why = SOURCES[d][p]
                avail[f"{design}:{d}:{p}"] = {"design": design, "dataset": d, "program": p, "anchors": int(len(x)),
                                              "labelled": int(ok.sum()), "positive": pos, "prevalence": prev,
                                              "hourly_source": hourly, "source": why,
                                              "available": bool(hourly and pos >= MIN_POSITIVE)}
    gate["endpoints"] = avail
    (out / "gate.json").write_text(json.dumps(gate, indent=1, default=float), encoding="utf-8")
    Path(a.report).write_text(report(gate), encoding="utf-8")
    print(json.dumps({k: {d: [p for p, r in avail.items() if r["available"] and r["dataset"] == d and r["design"] == k]
                          for d in EXTERNAL} for k in PRIMARY}, indent=1))


def report(g: dict) -> str:
    """``docs/paper/external_gate.md``: aggregates only."""
    names = {"ricd": "RICD", "zigong": "zigongICU"}
    L = ["# Amendment 16a: the external databases' pre-run gate", "",
         "Recorded before any future was drawn on RICD or zigongICU (`tools/paper_external.py gate`; seed 16). Counts "
         "and shares only. The rule is pre-registered in `docs/p1_paper_protocol.md` (amendment 16a): an endpoint is "
         f"scored on a database when its defining source has timing to the hour and at least {MIN_POSITIVE} anchors of "
         "the design are positive.", "", "## Encoding with the frozen vocabulary", "",
         "| Database | Stays | Grammar rejections | Parser rejections | Round-trip failures | Re-encode failures "
         "| Goal mismatches |", "|---|---|---|---|---|---|---|"]
    for d, t in g["tokenizer"].items():
        L.append(f"| {names[d]} | {t['stays']:,} | {sum(t['fsm_rejected'].values())} | "
                 f"{sum(t['parser_rejected'].values())} | {t['round_trip_failed']} | "
                 f"{t['reencode_failed'] + t['sampled_reencode_failed']} | {sum(t['goal_mismatch'].values())} |")
    L += ["", "**Out-of-vocabulary events** (a domain's OOV token, share of the domain's events; the training "
          "databases' held-out stays for reference):", "",
          "| Domain | Training held-out | RICD | zigongICU |", "|---|---|---|---|"]
    doms = sorted(set().union(*(set(x) for x in g["oov"].values())))
    for dom in doms:
        cells = []
        for k in ("training_held_out", "ricd", "zigong"):
            r = g["oov"][k].get(dom)
            cells.append(f"{100 * r['share']:.1f}% of {r['events']:,}" if r else "none")
        L.append(f"| {dom} | " + " | ".join(cells) + " |")
    L += ["", "**Values against the training deciles:** for each numeric code with at least 200 values, the share in "
          "the lowest and the highest bin (each holds about one tenth of the training values). Codes with more than "
          "half in an end bin:", ""]
    for d, b in g["bins"].items():
        fl_ = b["flagged"]
        L.append(f"- {names[d]}: {len(fl_)} of {b['codes']} codes" + (":" if fl_ else "."))
        for r in fl_:
            L.append(f"  - `{r['code']}`: {100 * r['low']:.0f}% lowest, {100 * r['high']:.0f}% highest bin "
                     f"(n = {r['n']:,})")
    L += ["", "Intermittent doses are discrete: a standard dose at a bin edge fills one end bin without a unit error. "
          "Urine output is charted as totals over intervals (canonical CLIF; quartiles):", ""]
    for d, u in g["urine_output"].items():
        gq, vq = u["gap_hours_q"], u["volume_ml_q"]
        L.append(f"- {names[d]}: {u['entries']:,} entries, {gq[1]:.1f} h between entries (IQR {gq[0]:.1f}-{gq[2]:.1f}), "
                 f"{vq[1]:.0f} mL per entry (IQR {vq[0]:.0f}-{vq[2]:.0f}).")
    L += ["", "## Anchors", ""]
    ds = g["designs"]
    L.append(f"- **R** (A1's rolling design, 500 per database): {ds['R']['anchors']:,} anchors, "
             f"{ds['R']['excluded_clock']} left out by F1's clock check, {ds['R']['kept']:,} kept "
             f"({', '.join(f'{names[k]} {v_}' for k, v_ in ds['R']['by_dataset'].items())}).")
    cc = ds["L24e"]["case_cohort"]
    L.append(f"- **L24e** (the 24-hour landmark's case-cohort sample): {ds['L24e']['landmark_anchors']:,} landmark "
             f"anchors, {ds['L24e']['excluded_clock']} left out by the clock check; "
             + "; ".join(f"{names[d]}: {cc[f'{d}_case']['sampled']} of {cc[f'{d}_case']['pool']} in-hospital deaths "
                         f"and {cc[f'{d}_control']['sampled']} of {cc[f'{d}_control']['pool']} others (weight "
                         f"{cc[f'{d}_control']['weight']:.2f})" for d in ("ricd", "zigong")) + ".")
    L += ["", "**Alignment control** (each kept anchor's real stay decoded as a rollout member on its own clock "
          "reproduces the real store's label; the minimum agreement over binary programs without a clinical cut):", ""]
    for d, r in g["alignment_control"].items():
        L.append(f"- {d}: {r['min']:.4f} over {r['anchors']:,} anchors and "
                 f"{sum(1 for x in r['programs'].values() if not x['clinical_cut'])} programs.")
    L += ["", "## Endpoints: prevalence and the availability decision", "",
          "| Design | Endpoint | Database | Labelled anchors | Positive | Weighted prevalence | Source | Scored |",
          "|---|---|---|---|---|---|---|---|"]
    for r in g["endpoints"].values():
        prev = f"{100 * r['prevalence']:.1f}%" if r["prevalence"] is not None else "n/a"
        L.append(f"| {r['design']} | {r['program']} | {names[r['dataset']]} | {r['labelled']:,} | {r['positive']:,} | "
                 f"{prev} | {r['source']} | {'yes' if r['available'] else 'no'} |")
    L += ["", "An endpoint not scored on a database is reported as not defined there; it is not imputed."]
    return "\n".join(L) + "\n"


# -- the analysis -------------------------------------------------------------------------------------------------
HAZARD = {"R": ("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880"),
          "L24e": ("icu_death@-1", "decompensation@1440")}
LEARNED = ("lgbm_pooled", "logistic_pooled", "tabpfn_v3_5", "twin_probe", "lgbm_local")
# family 17's candidates: amendment 10's clinical scores (bedside indicators excluded) and RICD's recorded SOFA
CLINICAL_SCORES = ("sofa", "news", "mews", "qsofa", "sirs", "sapsii_prob", "oasis_prob", "ricd_recorded_sofa")
INTERNAL = {"R": "f1/R128", "L24e": "f1/L24x"}


def _patients(clif: Path) -> dict:
    import pandas as pd

    out = {}
    for d in EXTERNAL:
        s = pd.read_parquet(clif / d / "clif_ext_icu_stay.parquet", columns=["stay_id", "patient_id"])
        out.update(dict(zip(s["stay_id"].astype(str), d + ":" + s["patient_id"].astype(str), strict=True)))
    return out


def pairs(out: Path, clif: Path) -> dict:
    """Per (design, database, endpoint) the gate scores: the anchors with a real label and Didymos's estimate, as a
    frame (anchor, stay, patient, y, w, twin: Didymos's estimate, the hazard estimate E2 on HAZARD endpoints, else the
    sampled estimate; twin_sampled; every learned comparator; the clinical scores the endpoint is assigned; RICD's
    recorded SOFA)."""
    import pandas as pd

    sys_path_tools()
    from paper_comparators import scores_for

    gate = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    pat = _patients(clif)
    sc = pd.read_parquet(out / "clinical" / "scores.parquet")
    res = {}
    for design in PRIMARY:
        dd = out / design
        if not (dd / "predictions.parquet").exists():      # the design's rollouts are not scored yet
            continue
        lab = pd.read_parquet(dd / "labels.parquet")
        tw = pd.read_parquet(dd / "predictions.parquet")
        est = pd.read_parquet(dd / "estimates.parquet") if (dd / "estimates.parquet").exists() else None
        comp = [pd.read_parquet(f) for f in (dd / "baseline_predictions.parquet", dd / "tabpfn_v3_5_predictions.parquet",
                                             dd / "probe_predictions.parquet", dd / "local_predictions.parquet")
                if f.exists()]
        comp = pd.concat(comp, ignore_index=True) if comp else None
        own = pd.read_parquet(dd / "own_scores.parquet") if (dd / "own_scores.parquet").exists() else None
        an = pd.read_parquet(dd / "anchors.parquet")[["anchor", "twin"]]
        base = "L24" if design.startswith("L") else design
        scd = sc[sc["design"] == base].set_index("anchor")
        for r in gate["endpoints"].values():
            if r["design"] != design or not r["available"]:
                continue
            d, prog = r["dataset"], r["program"]
            x = lab[(lab["dataset"] == d) & (lab["program"] == prog) & lab["label"].notna()][["anchor", "weight", "label"]]
            x = x.merge(an, on="anchor").rename(columns={"label": "y", "weight": "w", "twin": "stay"})
            s_ = tw[tw["program"] == prog].set_index("anchor")["p"]
            x["twin_sampled"] = x["anchor"].map(s_)
            if prog in HAZARD[design] and est is not None:
                e2 = est[(est["program"] == prog) & (est["estimator"] == "e2")].set_index("anchor")["p"]
                x["twin"] = x["anchor"].map(e2)
            else:
                x["twin"] = x["twin_sampled"]
            for m in LEARNED:
                if comp is not None:
                    c = comp[(comp["program"] == prog) & (comp["model"] == m)].set_index("anchor")["p"]
                    if len(c):
                        x[m] = x["anchor"].map(c)
            for s in scores_for(base, prog):
                if s in scd.columns:
                    x[s] = x["anchor"].map(scd[s])
            if own is not None and d == "ricd":
                x["ricd_recorded_sofa"] = x["anchor"].map(own.set_index("anchor")["value"])
            x["patient"] = x["stay"].astype(str).map(pat)
            res[(design, d, prog)] = x[x["twin"].notna()].reset_index(drop=True)
    return res


def folds_of(patients, seed: int = SEED) -> dict:
    """The 5 cross-fitting folds of a database's patients (the local reference's: a permutation, then k mod 5)."""
    pats = np.unique(np.asarray(patients))
    rng = np.random.default_rng(seed)
    return dict(zip(rng.permutation(pats), np.arange(len(pats)) % 5, strict=True))


def recalibrate(y, p, w, fold) -> np.ndarray:
    """Logistic recalibration (intercept and slope on the logit of the risk), cross-fitted by fold."""
    from sklearn.linear_model import LogisticRegression

    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    out = np.full(len(y), np.nan)
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        if len(np.unique(y[tr])) < 2:
            continue
        m = LogisticRegression(C=1e6, max_iter=1000).fit(z[tr], y[tr], sample_weight=w[tr])
        out[te] = m.predict_proba(z[te])[:, 1]
    return out


def cmd_analyze(a) -> None:
    """``<out>/ext.json``: per design, database and scored endpoint, Didymos's AUROC and O/E with every comparator
    (``paper_comparators.evaluate``: 1,000 resamples of stays, paired differences, Didymos (``twin``) minus each), the
    recalibrated estimate (cross-fitted within the database by the local reference's folds; Didymos and transferred
    LightGBM), and Didymos's internal AUROC on the same endpoint (the follow-up anchors, descriptive)."""
    from icutwin.eval import bench

    sys_path_tools()
    from paper_comparators import evaluate

    out, root = Path(a.out), Path(a.root)
    rep: dict = {"git": __import__("paper_a1")._sha(), "seed": SEED, "boot": a.boot, "designs": {}}
    for (design, d, prog), x in sorted(pairs(out, Path(a.clif)).items()):
        y, w = x["y"].to_numpy(float), x["w"].to_numpy(float)
        stay, ds = x["stay"].astype(str).to_numpy(), np.full(len(x), d)
        cols = [c for c in x.columns if c not in ("anchor", "stay", "patient", "y", "w")]
        preds = {c: x[c].to_numpy(float) for c in cols}
        r = {"anchors": int(len(x)), "positives": int(y.sum()),
             "estimate": "hazard (E2)" if prog in HAZARD[design] else "sampled",
             "predictors": evaluate(y, w, stay, ds, preds, "twin", a.boot)}
        clin = {c: v for c, v in preds.items() if c in CLINICAL_SCORES}   # family 17: the sampled estimate's
        r["sampled_vs_scores"] = evaluate(y, w, stay, ds, {"twin_sampled": preds["twin_sampled"], **clin},
                                          "twin_sampled", a.boot)
        fold_of = folds_of(x["patient"])
        fold = x["patient"].map(fold_of).to_numpy()
        r["recalibrated"] = {}
        for m in ("twin", "lgbm_pooled"):
            if m not in preds:
                continue
            ok = np.isfinite(preds[m])
            pr = recalibrate(y[ok], preds[m][ok], w[ok], fold[ok])
            k = np.isfinite(pr)
            r["recalibrated"][m] = bench.binary_metrics(y[ok][k], pr[k], w[ok][k])
        f2 = load_json(root / INTERNAL[design] / "f2.json")
        if f2:
            hz = prog in HAZARD[design]
            v = (f2["f2"] if hz else f2.get("f1", {})).get(prog)
            r["internal_auroc"] = (v["estimators"]["e2"] if hz else v["twin_128"]).get("auroc") if v else None
        rep["designs"].setdefault(design, {}).setdefault(d, {})[prog] = r
        print(json.dumps({f"{design}:{d}:{prog}": {"twin": r["predictors"]["twin"].get("auroc"),
                                                   "lgbm": r["predictors"].get("lgbm_pooled", {}).get("auroc")}}),
              flush=True)
    (out / "ext.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# -- ranks (post hoc) ---------------------------------------------------------------------------------------------
RANKED = ("twin", "twin_probe", "lgbm_pooled", "logistic_pooled", "tabpfn_v3_5", "lgbm_local", "sofa", "news", "mews",
          "qsofa", "sirs")


def cmd_ranks(a) -> None:
    """Post hoc (the study lead's request, 2026-10-12; not in amendment 16): amendment 15's rank summary
    (``paper_stats.rank_summary``: mid-ranks by AUROC per task, mean ranks with a 10,000-resample bootstrap interval
    over tasks, firsts, Friedman, Didymos against each by Wilcoxon with Holm) over the 15 scored endpoint-database
    pairs, with amendment 15's predictors scored on all of them: Didymos's current estimate, the probe, the
    transferred learners, the local cross-fitted LightGBM and the five clinical scores (bedside indicators excluded).
    Twice: every predictor, and without the local model (only models that never saw the database). Writes
    ``<out>/ranks.json``."""
    sys_path_tools()
    from paper_stats import rank_summary

    out = Path(a.out)
    r = json.loads((out / "ext.json").read_text(encoding="utf-8"))
    tasks, rows = [], []
    for design in ("R", "L24e"):
        for d, progs in sorted(r["designs"].get(design, {}).items()):
            for p, x in sorted(progs.items()):
                tasks.append({"design": design, "dataset": d, "program": p})
                rows.append([(x["predictors"].get(n) or {}).get("auroc") for n in RANKED])
    keep = [j for j in range(len(RANKED)) if all(row[j] is not None for row in rows)]
    names = [RANKED[j] for j in keep]
    mat = np.array([[row[j] for j in keep] for row in rows], dtype=float)
    rep = {"post_hoc": "the study lead's request, 2026-10-12", "tasks": tasks,
           "dropped": [RANKED[j] for j in range(len(RANKED)) if j not in keep], "names": names, "auroc": mat.tolist(),
           "all": rank_summary(mat, names, "twin")}
    no_local = [j for j, n in enumerate(names) if n != "lgbm_local"]
    rep["transfer_only"] = rank_summary(mat[:, no_local], [names[j] for j in no_local], "twin")
    for d in EXTERNAL:                                 # per database, descriptive
        m = [i for i, t in enumerate(tasks) if t["dataset"] == d]
        rep[f"all_{d}"] = rank_summary(mat[m], names, "twin")
    (out / "ranks.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    for k in ("all", "transfer_only", "all_ricd", "all_zigong"):
        s = rep[k]
        print(k, s["tasks"], [(x["predictor"], round(x["mean_rank"], 2), x["firsts"]) for x in s["predictors"]],
              "Friedman P", round(s["friedman"]["p"], 4))


# -- RICD's own score ---------------------------------------------------------------------------------------------
def cmd_ownscore(a) -> None:
    """RICD's recorded SOFA (``detailed_sofa``), the database's own score: at each anchor the latest one recorded in the
    24 hours before the prediction time (window end: ICU admission + ``t_min`` + 1 minute, exclusive, as
    ``paper_clinical.py``), on the adapter's clock (``adapters/ricd.py``: whole hours at the end of their hour).
    Writes ``<out>/<design>/own_scores.parquet``."""
    import tempfile

    import pandas as pd

    from icutwin.clif.adapters.ricd import RicdAdapter

    out = Path(a.out)
    with tempfile.TemporaryDirectory() as tmp:
        ad = RicdAdapter(Path(a.parquet) / "ricd", Path(tmp), Path(tmp))
        rec = ad.con.execute(f"""
            SELECT b.hid, b.start_min + (CAST(x.post_admission_hours AS DOUBLE) + 1) * 60 AS m, x.sofa_score AS sofa
            FROM {ad.src('csv/detailed_sofa.parquet')} x JOIN base b ON b.hid = CAST(x.new_hosp_id AS VARCHAR)
            WHERE x.sofa_score IS NOT NULL AND x.post_admission_hours IS NOT NULL""").df()
    stays = pd.read_parquet(Path(a.clif) / "ricd" / "clif_ext_icu_stay.parquet",
                            columns=["stay_id", "hospitalization_id", "icu_in_dttm"])
    origin = pd.Timestamp("2020-01-01", tz="UTC")
    stays["in_min"] = (stays["icu_in_dttm"] - origin).dt.total_seconds() / 60.0
    by_h = {h: g.sort_values("m") for h, g in rec.groupby("hid")}
    for design in PRIMARY:
        an = pd.read_parquet(out / design / "anchors.parquet")
        sub = pd.read_parquet(out / "subsets" / f"{design}.parquet")
        an = an[an["anchor"].isin(sub["anchor"]) & (an["dataset"] == "ricd")].merge(
            stays, left_on="twin", right_on="stay_id")
        score = []
        for x in an.itertuples():
            end = x.in_min + x.t_min + 1
            g = by_h.get(str(x.hospitalization_id))
            s = None if g is None else g[(g["m"] < end) & (g["m"] >= end - 1440)]["sofa"]
            score.append(float(s.iloc[-1]) if s is not None and len(s) else np.nan)
        res = pd.DataFrame({"anchor": an["anchor"], "dataset": "ricd", "score": "ricd_recorded_sofa", "value": score})
        res.to_parquet(out / design / "own_scores.parquet", index=False)
        print(json.dumps({design: {"anchors": len(res), "with_score": int(res["value"].notna().sum())}}))


# -- the local reference ------------------------------------------------------------------------------------------
def cmd_local(a) -> None:
    """The local reference: LightGBM with A1's recipe (``eval.baselines``: its features, training rows and fit) trained
    on the database's own stays, 5-fold cross-fitted by patient, so each anchor is scored by a model that never saw
    its patient; early stopping on a tenth of the training folds' patients. The endpoints the gate scores. Writes
    ``<out>/<design>/local_predictions.parquet`` (host-only kind of file: per anchor) and ``<out>/local.json``."""
    import pandas as pd

    from icutwin.eval import baselines as bl
    from icutwin.eval import bench
    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst

    out, st = Path(a.out), fst.Store(Path(a.store))
    gate = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    sid = st.stays["stay_id"].astype(str).to_numpy()
    sdb = st.stays["dataset"].astype(str).to_numpy()
    pat = {}
    for d in EXTERNAL:
        s = pd.read_parquet(Path(a.clif) / d / "clif_ext_icu_stay.parquet", columns=["stay_id", "patient_id"])
        pat.update(dict(zip(s["stay_id"].astype(str), d + ":" + s["patient_id"].astype(str), strict=True)))
    patient = np.array([pat[s] for s in sid])
    fz = bl.Featurizer(bl.feature_channels(st, 64)).fit_levels(st)
    ev = fpr.Evaluator(st)
    progs = {_key(p): p for p in fpr.catalog(st.channels)}
    qidx = {s: i for i, s in enumerate(sid)}
    rep: dict = {"seed": SEED, "folds": 5, "designs": {}}

    def lab(prog, rows):
        if prog == "remaining_los_gt_7d":
            rem = (st.los[rows.stay] - rows.t) / 60.0
            cens = np.isin(st.end_kind[rows.stay], ("censored", "unknown")) & (rem <= 168)
            return np.where(rem > 168, 1.0, np.where(cens, np.nan, 0.0))
        return bl.labels(st, progs[prog], rows, ev)

    for design in PRIMARY:
        base = "L24" if design.startswith("L") else design
        an = pd.read_parquet(out / design / "anchors.parquet")
        sub = pd.read_parquet(out / "subsets" / f"{design}.parquet")
        an = an[an["anchor"].isin(sub["anchor"])].merge(sub[["anchor", "weight"]].rename(columns={"weight": "w"}),
                                                        on="anchor").reset_index(drop=True)
        rows_te = fl.Rows(np.array([qidx[str(s)] for s in an["twin"]], dtype=np.int64),
                          an["t_min"].to_numpy(dtype=np.int64))
        x_te = fz(st, rows_te, ev)
        preds, res = [], {}
        for d in EXTERNAL:
            todo = [r["program"] for r in gate["endpoints"].values()
                    if r["design"] == design and r["dataset"] == d and r["available"]]
            pats = np.unique(patient[sdb == d])
            rng = np.random.default_rng(SEED)
            fold_of = dict(zip(rng.permutation(pats), np.arange(len(pats)) % 5, strict=True))
            st_fold = np.array([fold_of.get(p, -1) for p in patient])
            an_fold = st_fold[rows_te.stay]
            folds = []
            for k in range(5):                  # each fold's training and early-stopping rows, once for all endpoints
                train_p = pats[np.array([fold_of[p] != k for p in pats])]
                val_p = rng.choice(train_p, size=max(1, len(train_p) // 10), replace=False)
                tr = np.flatnonzero((sdb == d) & (st_fold != k) & ~np.isin(patient, val_p))
                va = np.flatnonzero((sdb == d) & np.isin(patient, val_p))
                rows_tr = bl.training_rows(st, base, 4, seed=0, stays=tr)
                rows_va = bl.training_rows(st, base, 2, seed=1, stays=va)
                folds.append((rows_tr, rows_va, fz(st, rows_tr, ev), fz(st, rows_va, ev),
                              (an["dataset"].to_numpy() == d) & (an_fold == k)))
            for prog in todo:
                p_hat = np.full(len(an), np.nan)
                for rows_tr, rows_va, x_tr, x_va, test in folds:
                    y_tr, y_va = lab(prog, rows_tr), lab(prog, rows_va)
                    ok_tr, ok_va = np.isfinite(y_tr), np.isfinite(y_va)
                    if len(np.unique(y_tr[ok_tr])) < 2 or not test.any():
                        continue
                    m = bl.fit_lgbm(x_tr[ok_tr], y_tr[ok_tr], x_va[ok_va], y_va[ok_va], seed=0, n_jobs=a.n_jobs)
                    p_hat[test] = m.predict_proba(x_te[test])[:, 1]
                y = lab(prog, rows_te)
                ok = (an["dataset"].to_numpy() == d) & np.isfinite(p_hat) & np.isfinite(y)
                res[f"{d}:{prog}"] = bench.binary_metrics(y[ok], p_hat[ok], an["w"].to_numpy()[ok])
                for i in np.flatnonzero(ok):
                    preds.append({"anchor": an["anchor"][i], "dataset": d, "program": prog, "model": "lgbm_local",
                                  "p": float(p_hat[i])})
                print(json.dumps({design: {f"{d}:{prog}": res[f"{d}:{prog}"].get("auroc")}}), flush=True)
        pd.DataFrame(preds).to_parquet(out / design / "local_predictions.parquet", index=False)
        rep["designs"][design] = res
    (out / "local.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("analyze")
    g.add_argument("--out", default="runs/paper/ext")
    g.add_argument("--root", default="runs/paper")
    g.add_argument("--clif", default="data/clif_canonical/full")
    g.add_argument("--boot", type=int, default=1000)
    g = sub.add_parser("ranks")
    g.add_argument("--out", default="runs/paper/ext")
    g = sub.add_parser("ownscore")
    g.add_argument("--out", default="runs/paper/ext")
    g.add_argument("--parquet", default="data/parquet/full")
    g.add_argument("--clif", default="data/clif_canonical/full")
    g = sub.add_parser("local")
    g.add_argument("--store", default="data/factory/stores/ext_test")
    g.add_argument("--out", default="runs/paper/ext")
    g.add_argument("--clif", default="data/clif_canonical/full")
    g.add_argument("--n-jobs", type=int, default=6)
    for name in ("design", "gate"):
        g = sub.add_parser(name)
        g.add_argument("--tokens", default="data/tokens/ext_r4")
        g.add_argument("--store", default="data/factory/stores/ext_test")
        g.add_argument("--out", default="runs/paper/ext")
        if name == "gate":
            g.add_argument("--qa", default="data/qa/ext_r4")
            g.add_argument("--clif", default="data/clif_canonical/full")
            g.add_argument("--train-tokens", default="data/tokens/full_r4")
            g.add_argument("--train-datasets", nargs="+",
                           default=["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"])
            g.add_argument("--report", default="docs/paper/external_gate.md")
            g.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    {"design": cmd_design, "gate": cmd_gate, "local": cmd_local, "ownscore": cmd_ownscore,
     "analyze": cmd_analyze, "ranks": cmd_ranks}[a.cmd](a)


if __name__ == "__main__":
    main()
