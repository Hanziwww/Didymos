"""Paper P, C2's analysis (``docs/p1_paper_protocol.md`` C2; ``tools/server/paper_attrib.py``,
``tools/server/paper_prompt_rollouts.py``). Per program:

* faithfulness: each method's deletion curve (the probe's logit after removing its top 1..5 groups) as the mean drop
  from the original logit, against a random ranking's; the area over the curve (mean drop over k);
* agreement with counterfactual editing: Spearman correlation and top-3 overlap of integrated gradients, attention
  rollout and attention x gradient with the edit effects; integrated gradients' completeness error;
* group importance across anchors: the mean edit effect of each family x block, and how often it is a top-3 group;
* the rollout-level check: the program's rollout risk of the original prompt and of its 3 highest and 3 lowest edits
  (labels on the decoded rollouts at the anchor's prediction time), and the agreement in sign and rank between the
  probe's edit effects and the rollouts' (the twin's own risk).

Writes ``<out>/c2.json`` (aggregates).

    python tools/paper_c2.py --attrib runs/paper/c2/test_L24 --work data/paper/stores/c2_L24 --out runs/paper/c2/test_L24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import _sha  # noqa: E402

METHODS = ("ig", "attn_rollout", "attn_x_grad")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("attrib", "work", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--tokens", default="data/tokens/full_r4")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-valid", type=int, default=8)
    a = ap.parse_args(argv)
    from icutwin.eval import attribution as at

    d = Path(a.attrib)
    recs = [json.loads(x) for x in (d / "attrib.jsonl").read_text().splitlines() if x.strip()]
    rep = {"git": _sha(), "programs": {}}
    for prog in sorted({r["program"] for r in recs}):
        rr = [r for r in recs if r["program"] == prog]
        res = {"anchors": len(rr), "faithfulness": {}, "agreement": {}, "groups": {}}
        for meth in ("edit", *METHODS, "random"):
            drops = [np.mean([r["f0"] - x for x in r["deletion"][meth]]) for r in rr
                     if meth in r.get("deletion", {}) and r["deletion"][meth]]
            if drops:
                res["faithfulness"][meth] = {"anchors": len(drops), "mean_drop": float(np.mean(drops)),
                                             "median_drop": float(np.median(drops))}
        for meth in METHODS:
            ag = [at.rank_agreement(r[meth], r["edit"]) for r in rr if meth in r]
            ag = [x for x in ag if x["spearman"] is not None]
            if ag:
                res["agreement"][meth] = {"anchors": len(ag), "spearman_median": float(np.median([x["spearman"]
                                                                                                 for x in ag])),
                                          "top3_overlap_mean": float(np.mean([x["top_overlap"] for x in ag]))}
        comp = [abs(r["ig_completeness"]["sum"] - r["ig_completeness"]["f_minus_f_zero"])
                / max(abs(r["ig_completeness"]["f_minus_f_zero"]), 1e-6) for r in rr]
        res["ig_completeness_relative_error_median"] = float(np.median(comp)) if comp else None
        eff: dict = {}
        top3: dict = {}
        for r in rr:
            ranked = sorted(r["edit"], key=lambda g: -r["edit"][g])
            for g, x in r["edit"].items():
                eff.setdefault(g, []).append(x)
            for g in ranked[:3]:
                top3[g] = top3.get(g, 0) + 1
        res["groups"] = {g: {"mean_edit_effect": float(np.mean(x)), "anchors": len(x),
                             "top3_share": top3.get(g, 0) / len(rr)} for g, x in sorted(eff.items())}
        rep["programs"][prog] = res
    rp = d / "rollout_prompts.parquet"
    if (d / "rollouts").exists() and rp.exists():
        rep["rollout_check"] = rollout_check(d, recs, Path(a.work), Path(a.tokens), a.workers, a.min_valid)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "c2.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({p: {"faith": {m: round(x["mean_drop"], 3) for m, x in r["faithfulness"].items()},
                          "agree": {m: round(x["spearman_median"], 3) for m, x in r["agreement"].items()}}
                      for p, r in rep["programs"].items()}, indent=1))


def rollout_check(d: Path, recs: list, work: Path, tokens: Path, workers: int, min_valid: int) -> dict:
    """The twin's own (rollout) risk of each original and edited prompt, against the probe's edit effects."""
    import pyarrow.parquet as pq
    from scipy.stats import spearmanr

    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst

    pr = pq.read_table(d / "rollout_prompts.parquet").to_pandas()
    pr = pr.drop_duplicates("id").set_index("id")   # an anchor with < 6 groups lists an edit among top and bottom 3
    rows, seen = [], set()
    for f in sorted((d / "rollouts").glob("*.parquet")):
        for r in pq.read_table(f).to_pylist():
            if (r["id"], r["rollout"]) not in seen:
                seen.add((r["id"], r["rollout"]))
                rows.append(r)
    if not (work / "manifest.json").exists():
        seqs = [{"stay_id": f"{r['id']}#{r['rollout']}", "ids": list(pr.loc[r["id"], "prompt"]) + list(r["continuation"]),
                 "clock": pr.loc[r["id"], "twin"], "split": "c2", "weight": 1.0, "cell": "c2"} for r in rows]
        fst.build(seqs, tokens / "vocab.json", work, "seg_0000", {"kind": "paper_c2", "attrib": str(d)},
                  persistence=tokens / "rank_persistence.json", workers=workers, chunk=250)
    st = fst.Store(work)
    ev = fpr.Evaluator(st)
    sid = st.stays["stay_id"].tolist()
    base = np.array([s.rsplit("#", 1)[0] for s in sid])
    catalog = {f"{p.name}@{p.horizon}": p for p in fpr.catalog(st.channels)}
    risk = {}
    for prog in sorted(set(pr["program"])):
        p = catalog.get(prog)
        if p is None:
            continue
        m = np.flatnonzero(np.isin(base, pr.index[pr["program"] == prog]))
        t = pr.loc[base[m], "t_min"].to_numpy(dtype=np.int64)
        b = ev.evaluate(p, fl.Rows(m.astype(np.int64), t))
        for i, ok, y in zip(base[m], b.valid, b.y, strict=True):
            if ok:
                risk.setdefault(i, []).append(float(y))
    probe = {(r["program"], r["anchor"]): r["edit"] for r in recs}
    out: dict = {}
    for prog in sorted(set(pr["program"])):
        signs, rhos, n = [], [], 0
        for anc in sorted(set(pr.loc[pr["program"] == prog, "anchor"])):
            o = risk.get(f"{prog}|{anc}|orig", [])
            if len(o) < min_valid:
                continue
            ro = np.mean(o)
            pe, re_ = [], []
            for g in pr.loc[(pr["program"] == prog) & (pr["anchor"] == anc), "group"]:
                if not g:
                    continue
                e = risk.get(f"{prog}|{anc}|{g}", [])
                if len(e) >= min_valid and g in probe.get((prog, anc), {}):
                    pe.append(probe[(prog, anc)][g])
                    re_.append(ro - np.mean(e))
            if len(pe) >= 4:
                n += 1
                signs += [np.sign(x) == np.sign(y) for x, y in zip(pe, re_, strict=True) if x != 0 and y != 0]
                rho = spearmanr(pe, re_).statistic
                if np.isfinite(rho):
                    rhos.append(rho)
        out[prog] = {"anchors": n, "sign_agreement": float(np.mean(signs)) if signs else None,
                     "spearman_median": float(np.median(rhos)) if rhos else None}
    return out


if __name__ == "__main__":
    main()
