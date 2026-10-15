"""icutwin on the AutoDL host: Paper P amendment 16c (``docs/p1_paper_protocol.md``): attention knockout on F4's anchors,
for the comparison with the generator's own edit effects (``tools/paper_c2b.py --knockout``).

For each F4 anchor (``anchors_f4.parquet``, prompts of at most ``--attn-max`` tokens: all 160), the trained model scores
each event group by amendment 9's knockout (``paper_attn_diag.py``; Geva et al. 2023): attention from every position
outside the group to the group's tokens is blocked in every layer (one additive 4-D mask, eager attention), and the
score is the fall in the probe's logit, both logits pooling the mean over the positions outside the group. Groups are
``eval.attribution.token_groups`` of the prompt; the comparison keeps those that editing shares. A check: on the
anchors amendment 9 scored (prompts of at most 4,096 tokens), the per-anchor rank correlation of knockout with editing
is compared with amendment 9's stored one (``--check-against <attn_diag/per_anchor.jsonl>``).
Writes ``<out>/knockout.jsonl`` (one line per anchor: its group scores) and ``<out>/knockout.json`` (the run and the
check).
    python paper_knockout_f4.py --run MUC32_main --tokens <root> --split held_out --anchors-from <states/test_R> \\
        --probe <a1/test_R> --anchors-file <f4/attrib/anchors_f4.parquet> --attrib <f4/attrib/attrib.jsonl> --out <dir>"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/icutwin")


def spearman(a, b):
    from scipy.stats import spearmanr

    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
        return None
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("run", "tokens", "split", "anchors-from", "probe", "anchors-file", "attrib", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--attn-max", type=int, default=8192)
    ap.add_argument("--check-against", default=None, help="amendment 9's per_anchor.jsonl")
    a = ap.parse_args()
    t0 = time.time()
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM

    from icutwin.eval import attribution as at
    from icutwin.eval import bench
    from icutwin.tokenization.vocab import Vocab

    tokens, out = Path(a.tokens), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v = Vocab.load(tokens / "vocab.json")
    an = pq.read_table(Path(a.anchors_from) / "anchors.parquet").to_pandas().set_index("anchor")
    want = pq.read_table(a.anchors_file).to_pandas()
    edits = {(r["program"], r["anchor"]): r.get("edit") or {} for r in map(json.loads, open(a.attrib, encoding="utf-8"))}
    probes = np.load(Path(a.probe) / "probe_models.npz")
    rows = {r["stay_id"]: r for r in bench.real_rows(tokens, a.split, set(an.loc[want["anchor"], "twin"].astype(str)))}
    hf = R / "runs/train/full" / a.run / "final/hf"
    m = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda()
    m.eval()
    for p_ in m.parameters():
        p_.requires_grad_(False)
    neg = torch.finfo(torch.bfloat16).min

    def logit(h, w, b, keep=None):                   # as paper_attn_diag.py
        pool = h.mean(0) if keep is None else h[keep].mean(0)
        return b + (torch.cat([h[-1], pool]) * w).sum()

    @torch.no_grad()
    def knockout(p0, w, b, positions: dict):
        n = len(p0)
        ids = torch.as_tensor([p0]).cuda()
        h_ref = m(input_ids=ids, output_hidden_states=True).hidden_states[-1][0].float()
        causal = torch.ones(n, n, dtype=torch.bool, device="cuda").tril()       # True: may attend
        out_ = {}
        for g, pos in positions.items():
            pos = [q for q in pos if q != n - 1]
            if not pos:
                continue
            keep = torch.ones(n, dtype=torch.bool, device="cuda")
            keep[pos] = False
            allowed = causal & ~(keep[:, None] & ~keep[None, :])
            mask = torch.zeros(n, n, dtype=torch.bfloat16, device="cuda").masked_fill(~allowed, neg)
            hk = m(input_ids=ids, attention_mask=mask[None, None], output_hidden_states=True).hidden_states[-1][0]
            out_[g] = float(logit(h_ref, w, b, keep) - logit(hk.float(), w, b, keep))
            del mask, hk
        return out_, float(logit(h_ref, w, b))

    prior = {}
    if a.check_against and Path(a.check_against).exists():
        prior = {(r["program"], r["anchor"]): r for r in map(json.loads, open(a.check_against, encoding="utf-8"))}
    rep = {"anchors": 0, "skipped_long": 0, "groups": [], "check": []}
    with open(out / "knockout.jsonl", "w", encoding="utf-8") as fh:
        for prog, g_ in want.groupby("program"):
            coef = probes[prog.replace("@", "_at_")]
            b, w = float(coef[0]), torch.as_tensor(coef[1:], dtype=torch.float32).cuda()
            for anc in g_["anchor"]:
                x = an.loc[anc]
                sid, hours = str(x["twin"]), float(x["hours"])
                p0 = bench.prompt(rows[sid]["ids"], v, sid, hours)
                if len(p0) > a.attn_max:
                    rep["skipped_long"] += 1
                    continue
                positions: dict = {}
                for k, gname in enumerate(at.token_groups(p0, v)):
                    if gname is not None:
                        positions.setdefault(gname, []).append(k)
                ko, f = knockout(p0, w, b, positions)
                torch.cuda.empty_cache()
                fh.write(json.dumps({"program": prog, "anchor": anc, "n": len(p0), "logit": f, "knockout": ko}) + "\n")
                rep["anchors"] += 1
                rep["groups"].append(len(ko))
                old = prior.get((prog, anc))
                if old is not None and old.get("edit_knockout_sum") is not None:
                    ed = edits.get((prog, anc), {})
                    gs = sorted(set(ko) & set(ed))
                    now = spearman([ko[k] for k in gs], [ed[k] for k in gs])
                    if now is not None:
                        rep["check"].append(abs(now - old["edit_knockout_sum"]))
            print(json.dumps({"program": prog, "anchors_done": rep["anchors"], "seconds": round(time.time() - t0, 1)}),
                  flush=True)
    rep["groups_median"] = float(np.median(rep.pop("groups"))) if rep["anchors"] else None
    chk = rep.pop("check")
    rep["check_amendment9"] = ({"anchors": len(chk), "abs_diff_max": float(max(chk)), "abs_diff_median": float(np.median(chk))}
                               if chk else None)
    rep["seconds"] = round(time.time() - t0, 1)
    (out / "knockout.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
