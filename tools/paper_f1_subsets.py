"""Paper P amendment 7, F1's anchor subsets (``docs/p1_paper_protocol.md``): R128 (150 per dataset drawn from test_R's
anchors, their inclusion weights kept) and L24x (every test_L24 anchor whose stay died in hospital, plus 354 of the
others at random, weighted by the inverse inclusion probability: a case-cohort sample). Anchors whose prompt
continues a context window (its clock then starts at the window, not the ICU admission) are left out and counted.
Writes ``<out>/<name>.parquet`` (anchor, weight, stratum) and ``<out>/subsets.json``.

    python tools/paper_f1_subsets.py --a1 runs/paper/a1 --out runs/paper/f1/subsets

Amendment 16 (16b): one more case-cohort subset of a landmark design, drawn with its own seed, written alone and merged
into ``subsets.json`` (the R128 and L24x files are not touched):

    python tools/paper_f1_subsets.py --a1 runs/paper/a1 --out runs/paper/f1/subsets --design L48 --name L48x --seed 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def clock_ok(an, tokens: Path) -> np.ndarray:
    """False for anchors whose prompt continues a context window (its clock starts at the window, not the ICU
    admission): among those whose prompt clock and hour disagree by more than 6 h, the prompts with [CONT]."""
    from icutwin.eval import bench
    from icutwin.tokenization.vocab import Vocab

    off = np.abs(an["t_min"].to_numpy() / 60.0 - an["hours"].to_numpy()) > 6.0
    ok = np.ones(len(an), dtype=bool)
    if off.any():
        v = Vocab.load(tokens / "vocab.json")
        sub = an[off]
        rows = {r["stay_id"]: r for r in bench.real_rows(tokens, "held_out", set(sub["twin"].astype(str)))}
        for k, x in zip(np.flatnonzero(off), sub.itertuples(), strict=True):
            p = bench.prompt(rows[str(x.twin)]["ids"], v, str(x.twin), float(x.hours))
            ok[k] = v.id("[CONT]") not in p
    return ok


def case_cohort(design_dir: Path, controls: int, rng: np.random.Generator, tokens: Path):
    """Every anchor of a landmark design whose stay died in hospital (A1's ``hospital_death@0`` label), plus
    ``controls`` of the others at random, weighted by the inverse inclusion probability. Returns the subset and its
    report."""
    import pandas as pd

    an = pd.read_parquet(design_dir / "anchors.parquet").drop_duplicates("anchor")
    pr = pd.read_parquet(design_dir / "predictions.parquet")
    hd = pr[pr["program"] == "hospital_death@0"].set_index("anchor")["label"]
    ok = clock_ok(an, tokens) & an["anchor"].isin(hd.index).to_numpy()
    rep = {"excluded_continued_window_or_label": int((~ok).sum())}
    an = an[ok]
    y = hd.loc[an["anchor"]].to_numpy()
    pos, neg = an[y == 1], an[y == 0]
    pick = neg.iloc[np.sort(rng.choice(len(neg), size=min(controls, len(neg)), replace=False))]
    w_neg = len(neg) / len(pick)
    s = pd.concat([pd.DataFrame({"anchor": pos["anchor"], "weight": 1.0, "stratum": "case"}),
                   pd.DataFrame({"anchor": pick["anchor"], "weight": w_neg, "stratum": "control"})],
                  ignore_index=True)
    rep.update(anchors=int(len(s)), cases=int(len(pos)), controls=int(len(pick)), control_weight=w_neg)
    return s, rep


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-dataset", type=int, default=150)
    ap.add_argument("--controls", type=int, default=354)
    ap.add_argument("--tokens", default="data/tokens/full_r4")
    ap.add_argument("--design", default=None, help="amendment 16: one landmark design's case-cohort subset only")
    ap.add_argument("--name", default=None, help="its name (default <design>x)")
    a = ap.parse_args(argv)
    import pandas as pd

    a1, out = Path(a.a1), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    if a.design:                                     # one subset, merged into subsets.json
        name = a.name or f"{a.design}x"
        s, r = case_cohort(a1 / f"test_{a.design}", a.controls, rng, Path(a.tokens))
        s.to_parquet(out / f"{name}.parquet", index=False)
        r.update(design=a.design, seed=a.seed)
        f = out / "subsets.json"
        rep = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        rep[name] = r
        f.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps({name: r}, indent=1))
        return
    rep = {}
    an = pd.read_parquet(a1 / "test_R/anchors.parquet").drop_duplicates("anchor")
    ok = clock_ok(an, Path(a.tokens))
    rep["R128"] = {"excluded_continued_window": int((~ok).sum())}
    an = an[ok]
    parts = []
    for ds, g in an.groupby("dataset", sort=True):
        pick = g.iloc[np.sort(rng.choice(len(g), size=min(a.per_dataset, len(g)), replace=False))]
        parts.append(pd.DataFrame({"anchor": pick["anchor"], "weight": pick["weight"], "stratum": ds}))
    r = pd.concat(parts, ignore_index=True)
    r.to_parquet(out / "R128.parquet", index=False)
    rep["R128"].update(anchors=int(len(r)), per_dataset=r.groupby("stratum").size().to_dict())

    s, rep["L24x"] = case_cohort(a1 / "test_L24", a.controls, rng, Path(a.tokens))
    s.to_parquet(out / "L24x.parquet", index=False)
    (out / "subsets.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
