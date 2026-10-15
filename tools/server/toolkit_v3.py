"""icutwin on the GPU host: Paper P amendment 11, V3 (privacy of prompted and intervened sampling; gate v1's
``conditional_and_counterfactual_sampling: PENDING``; ``docs/p1_paper_protocol.md``).

Members are train-split stays, non-members held-out stays; per dataset at most 500 each of at least 30 hours, each cut
at a whole hour drawn uniformly from 4 h to its stay minus 25 h (``toolkit.inputs``: the paper's prompt). The real
future is the stay's encoding after the prompt (F3's D1), cut at 24 h by the interval tokens' median gaps.

* (a) loss: the mean NLL per token of the real future given the prompt (raw probabilities): membership AUC and TPR at
  1% FPR with bootstrap intervals, pooled and per dataset, judged by P1's rule (PASS: AUC upper bound <= 0.60 and
  TPR upper bound <= 0.03; FAIL: a lower bound above 0.60 or 0.05; else INCONCLUSIVE);
* (b) extraction: 16 rollouts per prompt (at most 150 per side and dataset, C1's 24-hour caps): whether any rollout
  reproduces the real future's first 32 tokens (member minus non-member rate <= 0.01), and the best rollout's Jaccard
  similarity of event-token multisets over 24 h (membership AUC, upper bound <= 0.60);
* (c) intervened: norepinephrine prefixes built as C1's (train members, held-out non-members, at most 100 each), the
  give-median arm, (b)'s statistics on the sampled tokens;
* controls: a copying generator (members' rollouts replaced by their real futures) must FAIL (b); a private reference
  (two halves of the non-members) must PASS (a) and (b).

Writes ``runs/toolkit/v3/`` (designs, jobs) and ``runs/audit/m7_<run>/prompted.json`` (aggregates).
    python tools/server/toolkit_v3.py --part design|loss|sample|analyse
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

DATASETS = ("mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire")


def session(a):
    from icutwin.toolkit.session import Paths, Session

    return Session(Paths.default(), backend="vllm", engines=a.engines, gpu_memory_utilization=a.gpu,
                   workers=a.workers)


def real_future(s, p, prompt: list, minutes: dict, hours: float = 24.0):
    """The stay's tokens after the prompt within ``hours`` (None for a continued window or a disagreement)."""
    from icutwin.tokenization.encode import prospective

    v = s.vocab
    tl, cont = v.id("[TL]"), v.id("[CONT]")
    st = p.stay(v)
    last_h = max((e[0] - st.icu_in).total_seconds() for e in st.timeline) / 3600 if st.timeline else 0.0
    full = prospective(st, v, last_h + 1.0)
    if tl not in prompt or tl not in full or cont in prompt or cont in full:
        return None
    ptl, ftl = prompt[prompt.index(tl) + 1:], full[full.index(tl) + 1:]
    if ftl[:len(ptl)] != ptl:
        return None
    fut, t = [], 0.0
    for tok in ftl[len(ptl):]:
        t += minutes.get(tok, 0.0)
        if t > hours * 60:
            break
        fut.append(int(tok))
    return fut


def design(s, a, out: Path) -> None:
    """Members and non-members per dataset, their anchors, prompts and real futures (``design.json`` + parquet)."""
    import pyarrow.parquet as pq

    from icutwin.eval.forecast import Reader
    from icutwin.toolkit import inputs
    from icutwin.toolkit import treatments as tr
    from icutwin.toolkit.jobs import write_parquet

    v = s.vocab
    rd = Reader(v)
    rng = np.random.default_rng(a.seed)
    rec, skipped = [], Counter()
    for ds in DATASETS:
        side: dict = {"member": [], "nonmember": []}
        for split, role in (("train", "member"), ("held_out", "nonmember")):
            f = s.paths.tokens / ds / f"{split}.parquet"
            if not f.exists():
                continue
            t = pq.read_table(f, columns=["stay_id", "ids"])
            kept = 0
            for i in rng.permutation(t.num_rows):
                if kept >= a.per_side:
                    break
                r = t.slice(int(i), 1).to_pylist()[0]
                p = inputs.from_ids(r["ids"], str(r["stay_id"]), ds, split)
                los = p.los_hours(v)
                if los < 30:
                    continue
                h = float(rng.integers(4, int(np.floor(los - 25)) + 1))
                anc = p.at(v, h)
                fut = real_future(s, p, anc.prompt, rd.minutes)
                if fut is None:
                    skipped[f"{ds}:{role}"] += 1
                    continue
                side[role].append({"id": f"{role}|{anc.id}", "role": role, "dataset": ds, "stay": p.stay_id,
                                   "split": split, "hours": h, "prompt": anc.prompt, "future": fut,
                                   "part": "b" if kept < a.per_side_b else "a"})
                kept += 1
        if a.balance:                                 # equal numbers per side (P1's design)
            k = min(len(side["member"]), len(side["nonmember"]))
            side = {r: xs[:k] for r, xs in side.items()}
        rec += side["member"] + side["nonmember"]
    # (c) norepinephrine prefixes as C1's (cut 0.5-6 h before a first start at least 2.5 h after admission)
    start_c = re.compile(tr.TREATMENTS["norepinephrine"][0])
    fam_c = re.compile(tr.TREATMENTS["norepinephrine"][1])
    for split, role in (("train", "member"), ("held_out", "nonmember")) if a.per_side_c else ():
        kept = 0
        for ds in rng.permutation(list(DATASETS)):
            f = s.paths.tokens / ds / f"{split}.parquet"
            if ds not in tr.MODAL["norepinephrine"] or not f.exists():
                continue
            t = pq.read_table(f, columns=["stay_id", "ids"])
            for i in rng.permutation(t.num_rows)[:4000]:
                if kept >= a.per_side_c:
                    break
                r = t.slice(int(i), 1).to_pylist()[0]
                p = inputs.from_ids(r["ids"], str(r["stay_id"]), ds, split)
                st = p.stay(v)
                first = None                         # the treatment's first record: its hour and whether a start
                for e in st.timeline:
                    if fam_c.match(e[2]) or start_c.match(e[2]):
                        first = ((e[0] - st.icu_in).total_seconds() / 3600, bool(start_c.match(e[2])))
                        break
                if first is None or not first[1] or first[0] < 2.5:
                    continue
                h = max(2.0, first[0] - rng.uniform(0.5, 6.0))
                anc = p.at(v, h)
                fut = real_future(s, p, anc.prompt, rd.minutes)
                if fut is None:
                    continue
                rec.append({"id": f"c_{role}|{anc.id}", "role": role, "dataset": ds, "stay": p.stay_id,
                            "split": split, "hours": h, "prompt": anc.prompt, "future": fut, "part": "c"})
                kept += 1
            if kept >= a.per_side_c:
                break
    from icutwin.toolkit.jobs import _ints

    write_parquet(out / "design.parquet", {k: ([x[k] for x in rec] if k not in ("prompt", "future") else
                                               _ints(x[k] for x in rec)) for k in rec[0]})
    info = {"records": len(rec), "skipped": dict(skipped),
            "by": Counter(f"{x['part']}:{x['role']}" for x in rec)}
    (out / "design.json").write_text(json.dumps(info, indent=1, default=dict))
    print(json.dumps(info, default=dict), flush=True)


def load_design(out: Path) -> list:
    import pyarrow.parquet as pq

    return pq.read_table(out / "design.parquet").to_pylist()


def loss(s, a, out: Path) -> None:
    """(a): the mean NLL per token of each real future given its prompt."""
    import torch

    rec = [x for x in load_design(out) if x["part"] in ("a", "b") and len(x["future"])]
    t = s.teacher
    m, W = t.model, t.model.get_output_embeddings().weight
    seqs = [list(x["prompt"]) + list(x["future"]) for x in rec]
    nll = np.full(len(rec), np.nan)
    with torch.no_grad():
        for idx in t._packs(np.array([len(q) for q in seqs])):
            x, att = t._batch(seqs, idx)
            hid = m.model(input_ids=x, attention_mask=att).last_hidden_state
            for r, i in enumerate(idx):
                P, F = len(rec[i]["prompt"]), len(rec[i]["future"])
                pos = torch.arange(P - 1, P + F - 1, device=t.device)
                tot = 0.0
                for c0 in range(0, F, 2048):
                    lp = torch.log_softmax((hid[r, pos[c0:c0 + 2048]] @ W.T).float(), -1)
                    tgt = x[r, pos[c0:c0 + 2048] + 1]
                    tot += float(-lp.gather(1, tgt[:, None]).sum())
                nll[i] = tot / F
            del hid
    from icutwin.toolkit.jobs import write_parquet

    write_parquet(out / "loss.parquet", {"id": [x["id"] for x in rec], "role": [x["role"] for x in rec],
                                          "dataset": [x["dataset"] for x in rec], "nll": nll.tolist(),
                                          "future_tokens": [len(x["future"]) for x in rec]})
    print(json.dumps({"scored": int(np.isfinite(nll).sum())}), flush=True)


def sample(s, a, out: Path) -> None:
    """(b) and (c): 16 rollouts per prompt at C1's 24-hour caps (the give-median arm for (c))."""
    from icutwin.generate.engine import Intervention, intervene
    from icutwin.toolkit import treatments as tr
    from icutwin.toolkit.jobs import Job, SimRequest

    caps = json.loads((s.paths.paper / "f5/rollouts_design.json").read_text())["caps"]
    v = s.vocab
    reqs, rejected = [], 0
    for i, x in enumerate(load_design(out)):
        if x["part"] == "b" and len(x["future"]) >= 32:
            reqs.append(SimRequest(x["id"], tuple(x["prompt"]), 16, 70_000 + i, int(caps[x["dataset"]])))
        elif x["part"] == "c":
            tok = tr.give_arms(v, "norepinephrine", x["dataset"])["give_median"]
            try:                                       # the grammar must accept the start (as C1 checked it)
                intervene(list(x["prompt"]), Intervention(give=(tok,)), v, s.tables)
            except ValueError:
                rejected += 1
                continue
            reqs.append(SimRequest(x["id"], tuple(x["prompt"]), 16, 70_000 + i, int(caps[x["dataset"]]), give=(tok,)))
    job = Job(out / "rollouts")
    if not (job.root / "requests.parquet").exists():
        job.write_requests(reqs)
        job.write_spec({"tool": "v3_sample", "requests": len(reqs), "rejected_interventions": rejected,
                        "provenance": s.provenance()})
    s.sample(job)


def mia(sm: np.ndarray, sn: np.ndarray, rng, boot: int = 1000) -> dict:
    """AUC (members scoring higher) and TPR at 1% FPR, with bootstrap intervals (members and non-members resampled)."""
    from icutwin.eval import bench

    def stats(m_, n_):
        y = np.r_[np.ones(len(m_)), np.zeros(len(n_))]
        au = bench.auroc_fast(y, np.r_[m_, n_])
        thr = np.quantile(n_, 0.99)
        return au, float(np.mean(m_ > thr))

    au, tpr = stats(sm, sn)
    bs = [stats(rng.choice(sm, len(sm)), rng.choice(sn, len(sn))) for _ in range(boot)]
    a_ci = [float(np.percentile([b[0] for b in bs], 2.5)), float(np.percentile([b[0] for b in bs], 97.5))]
    t_ci = [float(np.percentile([b[1] for b in bs], 2.5)), float(np.percentile([b[1] for b in bs], 97.5))]
    status = "PASS" if a_ci[1] <= 0.60 and t_ci[1] <= 0.03 else ("FAIL" if a_ci[0] > 0.60 or t_ci[0] > 0.05
                                                                  else "INCONCLUSIVE")
    return {"members": len(sm), "nonmembers": len(sn), "auc": au, "auc_ci": a_ci, "tpr_1fpr": tpr, "tpr_ci": t_ci,
            "status": status}


def extraction(rows: list, conts: dict, v, given: bool) -> dict:
    """Per prompt: any rollout reproducing the real future's first 32 tokens; the best event-multiset Jaccard."""
    from icutwin.eval.forecast import Reader

    rd = Reader(v)
    event = np.array([m.get("cls") == "event" for m in v.meta])
    out = {}
    for x in rows:
        cs = conts.get(x["id"])
        if not cs:
            continue
        fut = list(x["future"])
        sampled = [list(c[2:]) if given else list(c) for c in cs]     # given: the gap and the given token first

        def events24(seq):
            t, keep = 0.0, Counter()
            for tok in seq:
                t += rd.minutes.get(int(tok), 0.0)
                if t > 24 * 60:
                    break
                if event[int(tok)]:
                    keep[int(tok)] += 1
            return keep

        fe = events24(fut)
        best = 0.0
        for c in sampled:
            ce = events24(c)
            inter, union = sum((fe & ce).values()), sum((fe | ce).values())
            best = max(best, inter / union if union else 0.0)
        out[x["id"]] = {"exact32": any(c[:32] == fut[:32] for c in sampled) if len(fut) >= 32 else None,
                        "jaccard": best}
    return out


def analyse(s, a, out: Path) -> None:
    import pandas as pd

    from icutwin.toolkit.jobs import Job

    rng = np.random.default_rng(a.seed)
    v = s.vocab
    rep = {"a_loss": {}, "b_extraction": {}, "c_intervened": {}, "controls": {}}
    lo = pd.read_parquet(out / "loss.parquet")
    lo = lo[np.isfinite(lo["nll"])]
    score = -lo["nll"].to_numpy()                       # lower loss: member
    role, ds = lo["role"].to_numpy(), lo["dataset"].to_numpy()
    rep["a_loss"]["pooled"] = mia(score[role == "member"], score[role == "nonmember"], rng)
    for d in DATASETS:
        m = ds == d
        if (m & (role == "member")).sum() >= 50 and (m & (role == "nonmember")).sum() >= 50:
            rep["a_loss"][d] = mia(score[m & (role == "member")], score[m & (role == "nonmember")], rng)
    nm = np.flatnonzero(role == "nonmember")
    half = rng.permutation(nm)
    rep["controls"]["private_reference_loss"] = mia(score[half[: len(half) // 2]], score[half[len(half) // 2:]], rng)
    design = load_design(out)
    conts = Job(out / "rollouts").rollouts()
    for part, key, given in (("b", "b_extraction", False), ("c", "c_intervened", True)):
        rows = [x for x in design if x["part"] == part]
        ex = extraction(rows, conts, v, given)
        mem = [ex[x["id"]] for x in rows if x["id"] in ex and x["role"] == "member"]
        non = [ex[x["id"]] for x in rows if x["id"] in ex and x["role"] == "nonmember"]
        e_m = [r["exact32"] for r in mem if r["exact32"] is not None]
        e_n = [r["exact32"] for r in non if r["exact32"] is not None]
        diff = float(np.mean(e_m) - np.mean(e_n)) if e_m and e_n else None
        j = mia(np.array([r["jaccard"] for r in mem]), np.array([r["jaccard"] for r in non]), rng)
        rep[key] = {"exact32_member": float(np.mean(e_m)) if e_m else None,
                    "exact32_nonmember": float(np.mean(e_n)) if e_n else None, "exact32_difference": diff,
                    "exact32_status": None if diff is None else ("PASS" if diff <= 0.01 else "FAIL"),
                    "jaccard": j, "jaccard_status": "PASS" if j["auc_ci"][1] <= 0.60 else
                    ("FAIL" if j["auc_ci"][0] > 0.60 else "INCONCLUSIVE")}
        if part == "b":                              # controls on (b)
            copy = {x["id"]: [list(x["future"])] * 16 for x in rows if x["role"] == "member"}
            exc = extraction([x for x in rows if x["role"] == "member"], copy, v, False)
            cm = [r["exact32"] for r in exc.values() if r["exact32"] is not None]
            rep["controls"]["copying_generator"] = {
                "exact32_member": float(np.mean(cm)) if cm else None,
                "exact32_difference": float(np.mean(cm) - np.mean(e_n)) if cm and e_n else None,
                "status": "FAIL" if cm and e_n and np.mean(cm) - np.mean(e_n) > 0.01 else "PASS"}
            hn = rng.permutation([x["id"] for x in rows if x["role"] == "nonmember" and x["id"] in ex])
            ha, hb = hn[: len(hn) // 2], hn[len(hn) // 2:]
            ea = [ex[i]["exact32"] for i in ha if ex[i]["exact32"] is not None]
            eb = [ex[i]["exact32"] for i in hb if ex[i]["exact32"] is not None]
            jr = mia(np.array([ex[i]["jaccard"] for i in ha]), np.array([ex[i]["jaccard"] for i in hb]), rng)
            rep["controls"]["private_reference_extraction"] = {
                "exact32_difference": float(np.mean(ea) - np.mean(eb)) if ea and eb else None, "jaccard": jr}
    valid = rep["controls"]["copying_generator"]["status"] == "FAIL" and \
        rep["controls"]["private_reference_loss"]["status"] != "FAIL"
    rep["controls_valid"] = bool(valid)
    statuses = [rep["a_loss"]["pooled"]["status"]] + [x["status"] for k, x in rep["a_loss"].items() if k != "pooled"] \
        + [rep[k][f] for k in ("b_extraction", "c_intervened") for f in ("exact32_status", "jaccard_status")
           if rep[k].get(f)]
    rep["status"] = "FAIL" if "FAIL" in statuses else ("PASS" if all(x == "PASS" for x in statuses) else
                                                       "INCONCLUSIVE")
    rep["note"] = "Amendment 11 V3; the gate report is unchanged (the study lead decides whether it records this)."
    (out / "v3.json").write_text(json.dumps(rep, indent=1, default=float))
    dst = s.paths.audit / "prompted.json"
    dst.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"status": rep["status"], "controls_valid": rep["controls_valid"],
                      "a": rep["a_loss"]["pooled"]["status"], "b": (rep["b_extraction"]["exact32_status"],
                                                                   rep["b_extraction"]["jaccard_status"]),
                      "c": (rep["c_intervened"].get("exact32_status"), rep["c_intervened"].get("jaccard_status"))},
                     indent=1), flush=True)


def loss_status(a_loss: dict) -> str:
    st = [x["status"] for x in a_loss.values()]
    return "FAIL" if "FAIL" in st else ("PASS" if all(x == "PASS" for x in st) else "INCONCLUSIVE")


def analyse_loss(s, a, out: Path) -> None:
    """V3x (amendment 11, post hoc): (a) alone at P1's sizes, with the private reference on the loss; writes
    ``<out>/v3x.json`` (the audit file is written by ``combine``)."""
    import pandas as pd

    rng = np.random.default_rng(a.seed)
    lo = pd.read_parquet(out / "loss.parquet")
    lo = lo[np.isfinite(lo["nll"])]
    score, role, ds = -lo["nll"].to_numpy(), lo["role"].to_numpy(), lo["dataset"].to_numpy()
    rep = {"a_loss": {"pooled": mia(score[role == "member"], score[role == "nonmember"], rng)}, "controls": {}}
    for d in DATASETS:
        m = ds == d
        if (m & (role == "member")).sum() >= 50 and (m & (role == "nonmember")).sum() >= 50:
            rep["a_loss"][d] = mia(score[m & (role == "member")], score[m & (role == "nonmember")], rng)
    half = rng.permutation(np.flatnonzero(role == "nonmember"))
    rep["controls"]["private_reference_loss"] = mia(score[half[: len(half) // 2]], score[half[len(half) // 2:]], rng)
    rep["status_a"] = loss_status(rep["a_loss"])
    (out / "v3x.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"status_a": rep["status_a"], **{d: x["status"] for d, x in rep["a_loss"].items()},
                      "private_reference": rep["controls"]["private_reference_loss"]["status"]}, indent=1), flush=True)


def combine(s, a, out: Path) -> None:
    """V3's final verdict (amendment 11): V3x's (a) in place of V3's, V3's (b), (c) and copying control; the audit's
    ``prompted.json``."""
    v3 = json.loads((s.paths.out / "v3" / "v3.json").read_text())
    v3x = json.loads((s.paths.out / "v3x" / "v3x.json").read_text())
    final = json.loads(json.dumps(v3))
    final["a_loss_v3"] = v3["a_loss"]
    final["a_loss"] = v3x["a_loss"]
    final["controls"]["private_reference_loss_v3"] = v3["controls"]["private_reference_loss"]
    final["controls"]["private_reference_loss"] = v3x["controls"]["private_reference_loss"]
    final["controls_valid"] = bool(final["controls"]["copying_generator"]["status"] == "FAIL"
                                   and final["controls"]["private_reference_loss"]["status"] != "FAIL")
    statuses = [x["status"] for x in final["a_loss"].values()] + \
        [final[k][f] for k in ("b_extraction", "c_intervened") for f in ("exact32_status", "jaccard_status")
         if final[k].get(f)]
    final["status_v3"] = v3["status"]
    final["status"] = "FAIL" if "FAIL" in statuses else ("PASS" if all(x == "PASS" for x in statuses) else
                                                         "INCONCLUSIVE")
    final["note"] = ("Amendment 11 V3 with V3x's loss test at P1's sizes in place of V3's (the study lead's decision); "
                     "V3's (b), (c) and copying control unchanged.")
    (s.paths.out / "v3" / "v3_final.json").write_text(json.dumps(final, indent=1, default=float))
    (s.paths.audit / "prompted.json").write_text(json.dumps(final, indent=1, default=float))
    print(json.dumps({"status": final["status"], "status_v3": final["status_v3"],
                      "controls_valid": final["controls_valid"]}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("design", "loss", "sample", "analyse", "analyse_loss", "combine"),
                    required=True)
    ap.add_argument("--balance", action="store_true", help="equal members and non-members per dataset (V3x)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--engines", type=int, default=3)
    ap.add_argument("--gpu", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-side", type=int, default=500)
    ap.add_argument("--per-side-b", type=int, default=150)
    ap.add_argument("--per-side-c", type=int, default=100)
    a = ap.parse_args()
    s = session(a)
    out = Path(a.out) if a.out else s.paths.out / "v3"
    out.mkdir(parents=True, exist_ok=True)
    {"design": design, "loss": loss, "sample": sample, "analyse": analyse, "analyse_loss": analyse_loss,
     "combine": combine}[a.part](s, a, out)


if __name__ == "__main__":
    main()
