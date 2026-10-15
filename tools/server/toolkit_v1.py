"""icutwin on the GPU host: Paper P amendment 11, V1 (the toolkit's conformance; ``docs/p1_paper_protocol.md``).

* ``--part forecast``: the toolkit's batch forecast (``toolkit.forecast.batch``) on F1's R128 anchors (1,050 rolling
  held-out anchors, their inclusion weights), 32 rollouts each at A1's caps, E2 by the toolkit's teacher pass, the
  label engine's programs (``runs/toolkit/v1/forecast``: ``predictions.parquet``, ``e2.parquet``, ``hazards.npz``).
  The prompts are checked against the paper's prediction times.
* ``--part whatif``: F5's design rows for norepinephrine and furosemide (A/A withhold arms on C1's prefixes, starters
  and matched non-starters with give and withhold), 16 rollouts per arm at C1's 24-hour caps, read as F5 read them
  (``toolkit.whatif.response_values``: C1's channels, windows, aggregates and cap): ``runs/toolkit/v1/whatif``.
* ``--part analyse`` (CPU): the pass rules against A1's 32-rollout predictions on the same anchors, F2's E2 curve at
  32 rollouts and F5's intervals: ``runs/toolkit/v1/v1.json``.

Sampling runs in vLLM's environment through ``Session.sample`` (``--engines`` lanes); needs the M7 gate.
    python tools/server/toolkit_v1.py --part forecast --engines 3 --workers 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PRIMARY_R = ("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880", "aki_2@2880",
             "sepsis_ase@1440", "hypotension@240", "hyperkalemia_top@1440", "thrombocytopenia@2880")
E2_PROGS = ("decompensation@1440", "imv_onset@1440", "vasopressor_onset@1440", "rrt_onset@2880")
TREATS = {"norepinephrine": "map", "furosemide": "urine_output"}


def session(a):
    from icutwin.toolkit.session import Paths, Session

    return Session(Paths.default(), backend=a.backend, engines=a.engines, gpu_memory_utilization=a.gpu,
                   workers=a.workers, device=a.device)


def forecast_part(s, a, out: Path) -> None:
    import pandas as pd

    from icutwin.eval import bench
    from icutwin.toolkit import forecast, inputs

    an = pd.read_parquet(s.paths.paper / "f1/R128/anchors.parquet").drop_duplicates("anchor").reset_index(drop=True)
    if a.limit:
        an = an.groupby("dataset").head(a.limit).reset_index(drop=True)
    caps = json.loads((s.paths.paper / "a1/test_R/design.json").read_text())["caps"]
    if a.cap:
        caps = {d: a.cap for d in caps}
    rows = {r["stay_id"]: r for r in bench.real_rows(s.paths.tokens, "held_out", set(an["twin"].astype(str)))}
    v = s.vocab
    pats, anchors, w, bad = {}, [], {}, 0
    for x in an.itertuples():
        sid = str(x.twin)
        if sid not in pats:
            pats[sid] = inputs.from_ids(rows[sid]["ids"], sid, x.dataset, "held_out")
        anc = pats[sid].at(v, float(x.hours))
        bad += int(anc.t_min != int(x.t_min) or anc.id != x.anchor)
        anchors.append(anc)
        w[anc.id] = float(x.weight)
    t0 = time.time()
    forecast.batch(s, anchors, out / "forecast", n=a.n or 32, caps=caps, seed=11, design="R", weights=w)
    info = {"anchors": len(anchors), "prompt_mismatch": bad, "seconds": round(time.time() - t0, 1), "caps": caps}
    (out / "forecast" / "v1_forecast.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


def whatif_part(s, a, out: Path) -> None:
    import pandas as pd

    from icutwin.eval import bench
    from icutwin.generate.engine import Intervention, intervene
    from icutwin.toolkit import labels as tl
    from icutwin.toolkit import treatments as tr
    from icutwin.toolkit import whatif
    from icutwin.toolkit.jobs import Job, SimRequest

    d = pd.read_parquet(s.paths.paper / "f5/design.parquet")
    d = d[d["treatment"].isin(list(TREATS))].reset_index(drop=True)
    if a.limit:
        pairs = set(d[d["group"] != "aa"].drop_duplicates("pair").groupby("treatment").head(a.limit)["pair"])
        keep = set(d[d["group"] == "aa"].drop_duplicates("anchor").groupby("treatment").head(a.limit)["anchor"])
        d = d[d["pair"].isin(pairs) | d["anchor"].isin(keep)].reset_index(drop=True)
    caps = json.loads((s.paths.paper / "f5/rollouts_design.json").read_text())["caps"]
    if a.cap:
        caps = {k: a.cap for k in caps}
    v, t = s.vocab, s.tables
    rows = {r["stay_id"]: r for r in bench.real_rows(s.paths.tokens, "held_out", set(d["twin"].astype(str)))}
    prompts, tmin = {}, {}
    for x in d.drop_duplicates("anchor").itertuples():
        p = bench.prompt(rows[str(x.twin)]["ids"], v, str(x.twin), float(x.hours))
        prompts[x.anchor], tmin[x.anchor] = [int(y) for y in p], int(bench.anchor_time(p, v, str(x.twin)))
    fam = {k: tr.family_tokens(v, k) for k in TREATS}
    reqs, rejected, meta = [], {}, {}
    for i, x in enumerate(d.itertuples()):
        rid = f"{x.anchor}|{x.arm}"
        if x.arm == "give":
            try:
                intervene(prompts[x.anchor], Intervention(give=(int(x.give_token),)), v, t)
            except ValueError:
                rejected[x.treatment] = rejected.get(x.treatment, 0) + 1
                continue
            kw = {"give": (int(x.give_token),)}
        else:
            kw = {"withhold": fam[x.treatment]}
        reqs.append(SimRequest(rid, tuple(prompts[x.anchor]), a.n or 16, 50_000 + i, int(caps.get(x.dataset) or 4096),
                               **kw))
        meta[rid] = (x.anchor, x.arm, x.treatment, x.group, x.pair, str(x.twin))
    job = Job(out / "whatif")
    if not (job.root / "requests.parquet").exists():
        job.write_requests(reqs)
        job.write_spec({"tool": "v1_whatif", "requests": len(reqs), "rejected": rejected, "caps": caps,
                        "provenance": s.provenance()})
    s.sample(job)
    conts = job.rollouts()
    sets = [tl.RolloutSet(r.id, meta[r.id][5], tmin[meta[r.id][0]], np.asarray(r.prompt, np.int32), conts[r.id])
            for r in reqs]
    store = tl.build_store(sets, s.paths.tokens / "vocab.json", job.root / "store",
                           persistence=s.paths.tokens / "rank_persistence.json", workers=s.workers)
    vals = whatif.response_values(store, sets, key=lambda x: x.anchor)
    rec = []
    for rid, (anc, arm, trt, grp, pair, _tw) in meta.items():
        resp = TREATS[trt]
        x = vals[resp][rid]
        f = x[np.isfinite(x)]
        rec.append({"anchor": anc, "arm": arm, "treatment": trt, "group": grp, "pair": pair, "response": resp,
                    "mean": float(f.mean()) if len(f) else np.nan, "n_valid": len(f)})
    pd.DataFrame(rec).to_parquet(job.root / "arm_means.parquet")
    print(json.dumps({"requests": len(reqs), "rejected": rejected}), flush=True)


def boot_mean(x: np.ndarray, n: int = 1000, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    b = [np.mean(rng.choice(x, size=len(x))) for _ in range(n)] if len(x) > 1 else []
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if b else [None, None]


def analyse(s, a, out: Path) -> None:
    import pandas as pd

    from icutwin.eval import bench

    rep = {"forecast": {}, "e2": {}, "whatif": {}}
    a1 = pd.read_parquet(s.paths.paper / "a1/test_R/predictions.parquet")
    tk = pd.read_parquet(out / "forecast/predictions.parquet")
    e2 = pd.read_parquet(out / "forecast/e2.parquet")
    f2 = json.loads((s.paths.paper / "f1/R128/f2.json").read_text())["f2"]
    for prog in PRIMARY_R:
        x = a1[a1["program"] == prog][["anchor", "stay", "dataset", "weight", "label", "p"]].merge(
            tk[tk["program"] == prog][["anchor", "p"]].rename(columns={"p": "p_tk"}), on="anchor")
        x = x[np.isfinite(x["label"]) & np.isfinite(x["p"]) & np.isfinite(x["p_tk"])]
        y, w, st, ds = (x[c].to_numpy() for c in ("label", "weight", "stay", "dataset"))
        p1, p2 = x["p"].to_numpy(), x["p_tk"].to_numpy()

        def oe(yy, pp, ww):
            return float(np.sum(ww * yy) / max(np.sum(ww * pp), 1e-12))

        ci_au = bench.bootstrap(lambda yy, pp, ww: bench.auroc_or_none(yy, pp, ww), st, y, p1, w, strata=ds)
        ci_oe = bench.bootstrap(oe, st, y, p1, w, strata=ds)
        au_tk, oe_tk = bench.auroc_or_none(y, p2, w), oe(y, p2, w)
        known = None not in (*ci_au, *ci_oe, au_tk)
        rep["forecast"][prog] = {"anchors": len(x), "auroc_a1": bench.auroc_or_none(y, p1, w), "auroc_a1_ci": ci_au,
                                 "auroc_toolkit": au_tk, "oe_a1": oe(y, p1, w), "oe_a1_ci": ci_oe, "oe_toolkit": oe_tk,
                                 "pass": bool(known and ci_au[0] <= au_tk <= ci_au[1] and ci_oe[0] <= oe_tk <= ci_oe[1])}
    for prog in E2_PROGS:
        x = a1[a1["program"] == prog][["anchor", "label", "weight"]].merge(e2[e2["program"] == prog], on="anchor")
        x = x[np.isfinite(x["label"]) & np.isfinite(x["e2"])]
        au = bench.auroc_or_none(x["label"].to_numpy(), x["e2"].to_numpy(), x["weight"].to_numpy())
        ref = f2[prog]["curve"]["32"]["e2"]
        rep["e2"][prog] = {"anchors": len(x), "auroc_e2_toolkit": au, "f2_curve_32": ref,
                           "pass": bool(au is not None and abs(au - ref) <= 0.02)}
    am_f = out / "whatif/arm_means.parquet"
    if am_f.exists():
        am = pd.read_parquet(am_f)
        c1b = json.loads((s.paths.paper / "f5/c1b.json").read_text())
        for trt, resp in TREATS.items():
            m = am[(am["treatment"] == trt) & np.isfinite(am["mean"])].set_index(["anchor", "arm"])["mean"]
            g = am[am["treatment"] == trt].drop_duplicates("anchor").set_index("anchor")
            aa = np.array([m[(x, "withhold_a")] - m[(x, "withhold_b")] for x in g.index[g["group"] == "aa"]
                           if (x, "withhold_a") in m and (x, "withhold_b") in m])
            fac = []
            for _p, gg in g[g["group"] != "aa"].groupby("pair"):
                st_, ns = gg.index[gg["group"] == "starter"], gg.index[gg["group"] == "nonstarter"]
                if len(st_) == 1 and len(ns) == 1 and (st_[0], "give") in m and (ns[0], "withhold") in m:
                    fac.append(m[(st_[0], "give")] - m[(ns[0], "withhold")])
            fac = np.array(fac)
            ref = c1b["factual"][trt][resp]["twin_factual_contrast"]
            ci_aa = boot_mean(aa)
            ok = bool(len(fac) and ref["ci"][0] <= fac.mean() <= ref["ci"][1] and ci_aa[0] <= 0 <= ci_aa[1])
            rep["whatif"][f"{trt}:{resp}"] = {"pairs": len(fac), "factual_contrast": float(fac.mean()) if len(fac)
                                              else None, "factual_ci": boot_mean(fac), "f5": ref, "aa_prefixes": len(aa),
                                              "aa_mean": float(aa.mean()) if len(aa) else None, "aa_ci": ci_aa,
                                              "pass": ok}
    rep["pass"] = all(x["pass"] for part in ("forecast", "e2", "whatif") for x in rep[part].values())
    (out / "v1.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"pass": rep["pass"], **{k: {p: x["pass"] for p, x in rep[k].items()} for k in
                                              ("forecast", "e2", "whatif")}}, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("forecast", "whatif", "analyse"), required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default="vllm")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--engines", type=int, default=3)
    ap.add_argument("--gpu", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="a smoke run: at most this many anchors per dataset")
    ap.add_argument("--n", type=int, default=0, help="a smoke run: rollouts")
    ap.add_argument("--cap", type=int, default=0, help="a smoke run: one token cap")
    a = ap.parse_args()
    s = session(a)
    out = Path(a.out) if a.out else s.paths.out / "v1"
    out.mkdir(parents=True, exist_ok=True)
    {"forecast": forecast_part, "whatif": whatif_part, "analyse": analyse}[a.part](s, a, out)


if __name__ == "__main__":
    sys.exit(main())
