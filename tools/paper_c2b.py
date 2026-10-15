"""Paper P amendment 7, F4's analysis (``docs/p1_paper_protocol.md``): C2's attributions against the generator's own
edit effects. The F4 design directory (``tools/server/paper_prompt_rollouts.py --design`` on ``paper_attrib.py
--rollout-all``'s prompts: every anchor's original prompt and each group's edit, 48 rollouts each) gives, per prompt,
the generator's risk over the endpoint's horizon after the prompt's end:

* death and new IMV: the mean over the rollouts of each one's summed hazard (``paper_hazards.py``: the compensator, a
  Rao-Blackwellized share), with its standard error;
* AKI and hypotension: the share of positive valid rollouts (the label engine on the rollouts' store), with its
  binomial standard error.

A group's generator effect is the original's risk minus the edit's. Reliability: the split-half (rollouts 0-23 against
24-47) Spearman correlation of the effects within an anchor, Spearman-Brown corrected (the ceiling of any agreement).
Agreement of each probe-level method (``attrib.jsonl``: editing, IG, attention rollout, attention x gradient) with the
generator: the Spearman correlation per anchor (median), top-3 overlap, and sign agreement on groups whose effect's 95%
interval excludes 0 (signed methods). Sanity checks: each method's rank correlation with itself on a random-weight
backbone (``--attrib-random``) and with a probe fit on permuted labels (``--attrib-perm``). Writes
``<out>/c2b.json`` (aggregates).

Amendment 16c: ``--knockout`` adds attention knockout (``tools/server/paper_knockout_f4.py``) as a method, and pairs it
with editing on the groups both score (per anchor: knockout's and editing's Spearman with the generator on the same
groups); ``--dump <dir>`` saves every group's generator effect with its halves and standard error
(``effects.parquet``), every method's per-anchor agreement (``agreement_per_anchor.parquet``) and the pairs
(``knockout_vs_edit.parquet``).

    python tools/paper_c2b.py --f4 runs/paper/f4/R --attrib runs/paper/f4/attrib/attrib.jsonl \\
        --attrib-random runs/paper/f4/attrib/attrib_random.jsonl --attrib-perm runs/paper/f4/attrib/attrib_perm.jsonl \\
        --work data/paper/stores/f4 --out runs/paper/f4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_a1 import _sha  # noqa: E402
from paper_f2 import load_hazards  # noqa: E402

HAZARD = {"decompensation@1440": ("death", 24), "imv_onset@1440": ("imv", 24)}
METHODS = ("edit", "ig", "attn_rollout", "attn_x_grad")
SIGNED = ("edit", "ig")


def spearman(x, y):
    from scipy.stats import spearmanr

    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return None
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    for k in ("f4", "attrib", "work", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--attrib-random", default=None)
    ap.add_argument("--attrib-perm", default=None)
    ap.add_argument("--tokens", default="data/tokens/full_r4")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--knockout", default=None, help="amendment 16c: paper_knockout_f4.py's knockout.jsonl")
    ap.add_argument("--dump", default=None, help="amendment 16c: a directory for per-group and per-anchor outputs")
    a = ap.parse_args(argv)
    import pandas as pd
    import pyarrow.parquet as pq

    from icutwin.factory import labels as fl
    from icutwin.factory import programs as fpr
    from icutwin.factory import store as fst

    d, tokens = Path(a.f4), Path(a.tokens)
    an = pq.read_table(d / "anchors.parquet").to_pandas().drop_duplicates("anchor").reset_index(drop=True)
    pos = {x: i for i, x in enumerate(an["anchor"])}
    risk = {}                                      # prompt id -> (all, half A, half B, standard error)
    H = load_hazards(d / "hazards") if (d / "hazards").exists() else {}
    if H:
        for prog, (cause, hor) in HAZARD.items():
            ids = an.index[an["program"] == prog].to_numpy()
            for i in ids:
                rr = np.flatnonzero((H["anchor_idx"] == i) & (H["rollout"] >= 0))
                if len(rr) < 8:
                    continue
                lam = H[f"inc_{cause}"][rr][:, :hor].sum(1)
                k = H["rollout"][rr]
                risk[an["anchor"][i]] = (float(lam.mean()), float(lam[k < 24].mean()), float(lam[k >= 24].mean()),
                                         float(lam.std(ddof=1) / np.sqrt(len(lam))))
    # AKI and hypotension: the label engine on the rollouts' store
    lab_progs = sorted(set(an["program"]) - set(HAZARD))
    if lab_progs:
        work = Path(a.work)
        if not (work / "manifest.json").exists():
            seqs = []
            want = set(an.loc[an["program"].isin(lab_progs), "anchor"])
            for f in sorted((d / "rollouts").rglob("*.parquet")):
                t = pq.read_table(f, columns=["anchor", "rollout", "prompt", "continuation"])
                for x, k, p, c in zip(t.column("anchor").to_pylist(), t.column("rollout").to_pylist(),
                                      t.column("prompt").to_pylist(), t.column("continuation").to_pylist(), strict=True):
                    if x in want:
                        seqs.append({"stay_id": f"{x}#{k}", "ids": np.asarray(list(p) + list(c), np.int32),
                                     "clock": str(an.loc[pos[x], "twin"]), "split": "c2b", "weight": 1.0, "cell": "c2b"})
            fst.build(seqs, tokens / "vocab.json", work, "seg_0000", {"kind": "paper_c2b", "f4": str(d)},
                      persistence=tokens / "rank_persistence.json", workers=a.workers, chunk=100)
        st = fst.Store(work)
        ev = fpr.Evaluator(st)
        sids = st.stays["stay_id"].tolist()
        base = np.array([s.rsplit("#", 1)[0] for s in sids])
        kk = np.array([int(s.rsplit("#", 1)[1]) for s in sids])
        catalog = {f"{p.name}@{p.horizon}": p for p in fpr.catalog(st.channels)}
        for prog in lab_progs:
            p = catalog.get(prog)
            if p is None:
                continue
            m = np.flatnonzero(np.isin(base, an.loc[an["program"] == prog, "anchor"]))
            t = an.loc[[pos[x] for x in base[m]], "t_min"].to_numpy(dtype=np.int64)
            b = ev.evaluate(p, fl.Rows(m.astype(np.int64), t))
            by: dict = {}
            for x, k, ok, y in zip(base[m], kk[m], b.valid, b.y, strict=True):
                if ok:
                    by.setdefault(x, []).append((k, float(y)))
            for x, vals in by.items():
                if len(vals) < 8:
                    continue
                kv, yv = np.array([v_[0] for v_ in vals]), np.array([v_[1] for v_ in vals])
                pm = float(yv.mean())
                risk[x] = (pm, float(yv[kv < 24].mean()) if (kv < 24).any() else np.nan,
                           float(yv[kv >= 24].mean()) if (kv >= 24).any() else np.nan,
                           float(np.sqrt(max(pm * (1 - pm), 1e-4) / len(yv))))

    def read(path):
        out = {}
        if path and Path(path).exists():
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                out[(r["program"], r["anchor"])] = r
        return out

    att, att_rand, att_perm = read(a.attrib), read(a.attrib_random), read(a.attrib_perm)
    kos = {k: r.get("knockout") or {} for k, r in read(a.knockout).items()} if a.knockout else {}
    methods = METHODS + (("knockout",) if kos else ())
    signed = SIGNED + (("knockout",) if kos else ())
    dump_eff, dump_agree, dump_pair = [], [], []
    rep = {"git": _sha(), "programs": {}}
    for prog in sorted(set(an["program"])):
        sub = an[an["program"] == prog]
        rel, rel_sb, agree, n_anchor = [], [], {m: {"rho": [], "top3": [], "sign": []} for m in methods}, 0
        pair_k, pair_e = [], []
        sanity = {f"{kind}_{m}": [] for kind in ("random", "perm") for m in METHODS}
        effects_all = []
        for base_a, g in sub.groupby("base_anchor"):
            orig = g.loc[g["group"].isin(["", "orig"]), "anchor"]
            if not len(orig) or orig.iloc[0] not in risk:
                continue
            ro = risk[orig.iloc[0]]
            eff = {}
            for x, grp in zip(g["anchor"], g["group"], strict=True):
                if grp in ("", "orig") or x not in risk:
                    continue
                re_ = risk[x]
                eff[grp] = (ro[0] - re_[0], ro[1] - re_[1], ro[2] - re_[2], float(np.hypot(ro[3], re_[3])))
            if len(eff) < 4:
                continue
            n_anchor += 1
            gs = sorted(eff)
            full = np.array([eff[x][0] for x in gs])
            ha, hb = np.array([eff[x][1] for x in gs]), np.array([eff[x][2] for x in gs])
            effects_all += list(full)
            r = spearman(ha, hb) if np.isfinite(ha).all() and np.isfinite(hb).all() else None
            if r is not None:
                rel.append(r)
                rel_sb.append(2 * r / (1 + r) if r > -1 else -1.0)
            if a.dump:
                dump_eff += [{"program": prog, "base_anchor": base_a, "group": x, "effect": eff[x][0],
                              "half_a": eff[x][1], "half_b": eff[x][2], "se": eff[x][3]} for x in gs]
            rec = att.get((prog, base_a))
            if rec is None:
                continue
            if kos:
                rec = {**rec, "knockout": kos.get((prog, base_a))}
                ko, ed = rec["knockout"] or {}, rec.get("edit") or {}
                both = [x for x in gs if x in ko and x in ed]  # the groups both methods score
                if len(both) >= 4:
                    e_ = np.array([eff[x][0] for x in both])
                    rk = spearman(np.array([ko[x] for x in both]), e_)
                    re2 = spearman(np.array([ed[x] for x in both]), e_)
                    if rk is not None and re2 is not None:
                        pair_k.append(rk)
                        pair_e.append(re2)
                        dump_pair.append({"program": prog, "base_anchor": base_a, "groups": len(both),
                                          "rho_knockout": rk, "rho_edit": re2})
            for m in methods:
                sc = rec.get(m)
                if not sc:
                    continue
                common = [x for x in gs if x in sc]
                if len(common) < 4:
                    continue
                s_ = np.array([sc[x] for x in common])
                e_ = np.array([eff[x][0] for x in common])
                rho = spearman(s_, e_)
                if rho is not None:
                    agree[m]["rho"].append(rho)
                    if a.dump:
                        dump_agree.append({"program": prog, "base_anchor": base_a, "method": m, "rho": rho,
                                           "groups": len(common)})
                top_s = set(np.array(common)[np.argsort(-s_)[:3]])
                top_e = set(np.array(common)[np.argsort(-e_)[:3]])
                agree[m]["top3"].append(len(top_s & top_e) / 3)
                if m in signed:
                    sig = [x for x in common if abs(eff[x][0]) > 1.96 * eff[x][3] and sc[x] != 0]
                    agree[m]["sign"] += [np.sign(sc[x]) == np.sign(eff[x][0]) for x in sig]
                for kind, src in (("random", att_rand), ("perm", att_perm)):
                    r2 = src.get((prog, base_a), {}).get(m)
                    if r2:
                        cc = [x for x in common if x in r2]
                        rr_ = spearman(np.array([sc[x] for x in cc]), np.array([r2[x] for x in cc]))
                        if rr_ is not None:
                            sanity[f"{kind}_{m}"].append(rr_)
        med = lambda x: float(np.median(x)) if len(x) else None  # noqa: E731
        rep["programs"][prog] = {
            "anchors": n_anchor, "reliability_split_half_median": med(rel), "reliability_sb_median": med(rel_sb),
            "effect_abs_median": med(np.abs(effects_all)),
            "agreement": {m: {"spearman_median": med(x["rho"]), "anchors": len(x["rho"]),
                              "top3_overlap_mean": float(np.mean(x["top3"])) if x["top3"] else None,
                              "sign_agreement": float(np.mean(x["sign"])) if x["sign"] else None,
                              "signed_groups": len(x["sign"])} for m, x in agree.items()},
            "sanity": {k: {"spearman_median": med(x), "anchors": len(x)} for k, x in sanity.items()}}
        if kos:
            rep["programs"][prog]["knockout_vs_edit"] = {
                "anchors": len(pair_k), "knockout_median": med(pair_k), "edit_median": med(pair_e),
                "difference_median": med(np.array(pair_k) - np.array(pair_e)) if pair_k else None}
        print(json.dumps({prog: {"anchors": n_anchor, "reliability_sb": med(rel_sb),
                                 "edit_rho": rep["programs"][prog]["agreement"]["edit"]["spearman_median"]}}), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "c2b.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    if a.dump:
        dd = Path(a.dump)
        dd.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dump_eff).to_parquet(dd / "effects.parquet", index=False)
        pd.DataFrame(dump_agree).to_parquet(dd / "agreement_per_anchor.parquet", index=False)
        if kos:
            pd.DataFrame(dump_pair).to_parquet(dd / "knockout_vs_edit.parquet", index=False)


if __name__ == "__main__":
    main()
