"""icutwin on the AutoDL host: Paper P amendment 9 (``docs/p1_paper_protocol.md``; exploratory): why F4's attention
rollout ranks the event groups almost alike in the trained model and in a randomly initialized one (rank correlation
0.96-0.97), and candidate remedies, on F4's anchors.

For each F4 anchor (``anchors_f4.parquet``; prompts of at most ``--attn-max`` tokens), the trained model and F4's random
initialization (``--seed``; the same architecture, the trained probe kept, as ``paper_attrib.py --random-backbone``)
attribute the probe's logit to the prompt's tokens by:

* ``rollout``: Abnar & Zuidema (2020) as C2 (head-mean attention, residual 0.5 I, layers multiplied; the last token's
  row);
* ``raw_last``: the last layer's head-mean attention of the last token (no rollout);
* ``uniform``: the rollout of uniform causal attention (no model: the causal mask's geometry alone, the structural
  null);
* ``rollout_minus_uniform`` and ``rollout_lift``: the rollout less the null, and its ratio to it;
* ``attn_x_grad``: as C2 (attention times its gradient, positive part, head mean, summed over layers);
* ``grad_rollout``: Chefer et al. (2021): R <- R + E_h[(grad A * A)+] R from the identity (the last token's own 1
  removed), which depends on the target;
* ``norm_rollout``: Kobayashi et al. (2020): each head's weight times the norm of its transformed value
  ||W_O^h v_j||, head mean, rows renormalized, rolled out as C2;
* ``knockout``: attention knockout (Geva et al. 2023), an intervention: attention from every other position to a
  group's tokens blocked in every layer, and the probe logit's fall (the group's positions left out of the mean
  pooling in both runs).

Group scores are sums over each group's tokens (as C2) and, for token scores, means. Per anchor, Spearman over its
groups: trained against random (the sanity check), against the token counts, against the null, and against the
trained model's probe-level editing (``attrib.jsonl``, signed and absolute); the sanity and editing correlations also
with the token counts held fixed (partial rank correlations). Also token-level comparisons, the rollout's
mass on the statics, the last token's attention entropy per layer, and a check that ``rollout`` and ``attn_x_grad``
reproduce F4's stored scores. Writes ``<out>/attn_diag.json`` (medians and quartiles over anchors, pooled and per
program) and ``<out>/per_anchor.jsonl`` (aggregates per anchor; host only).
    python paper_attn_diag.py --run MUC32_main --tokens <root> --split held_out --anchors-from <states/test_R> \\
        --probe <a1/test_R> --anchors-file <f4/attrib/anchors_f4.parquet> --attrib <f4/attrib/attrib.jsonl> \\
        --attrib-random <f4/attrib_random/attrib.jsonl> --out <dir>"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/icutwin")
TOKEN_METHODS = ("rollout", "raw_last", "uniform", "rollout_minus_uniform", "rollout_lift", "attn_x_grad",
                 "grad_rollout", "norm_rollout")


def spearman(a, b):
    from scipy.stats import spearmanr

    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
        return None
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else None


def cmp(x: dict, y: dict, groups) -> float | None:
    g = [k for k in groups if k in x and k in y]
    return spearman([x[k] for k in g], [y[k] for k in g])


def partial(r_xy, r_xz, r_yz) -> float | None:
    """The partial correlation of x and y with z held fixed, from the three pairwise (rank) correlations."""
    if None in (r_xy, r_xz, r_yz):
        return None
    d = (1 - r_xz ** 2) * (1 - r_yz ** 2)
    return float((r_xy - r_xz * r_yz) / np.sqrt(d)) if d > 1e-9 else None


def tv(p, q) -> float:
    p, q = np.clip(np.asarray(p, float), 0, None), np.clip(np.asarray(q, float), 0, None)
    return float(0.5 * np.abs(p / p.sum() - q / q.sum()).sum())


def summary(xs: list) -> dict | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    q = np.quantile(xs, [0.25, 0.5, 0.75])
    return {"median": float(q[1]), "q25": float(q[0]), "q75": float(q[2]), "n": len(xs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("run", "tokens", "split", "anchors-from", "probe", "anchors-file", "attrib", "attrib-random", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--attn-max", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from icutwin.eval import attribution as at
    from icutwin.eval import bench
    from icutwin.tokenization.vocab import Vocab

    tokens, out = Path(a.tokens), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v = Vocab.load(tokens / "vocab.json")
    an = pq.read_table(Path(a.anchors_from) / "anchors.parquet").to_pandas().set_index("anchor")
    want = pq.read_table(a.anchors_file).to_pandas()
    if a.limit:
        want = want.groupby("program").head(a.limit)

    def stored(path):
        return {(r["program"], r["anchor"]): r for r in map(json.loads, open(path, encoding="utf-8"))}

    f4_tr, f4_rd = stored(a.attrib), stored(a.attrib_random)
    probes = np.load(Path(a.probe) / "probe_models.npz")
    rows = {r["stay_id"]: r for r in bench.real_rows(tokens, a.split, set(an.loc[want["anchor"], "twin"].astype(str)))}
    hf = R / "runs/train/full" / a.run / "final/hf"

    def load():
        return AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda()

    trained, rnd = load(), load()
    torch.manual_seed(a.seed)                        # as paper_attrib.py --random-backbone (F4)
    cfg0 = AutoConfig.from_pretrained(hf)
    rnd.load_state_dict(AutoModelForCausalLM.from_config(cfg0, torch_dtype=torch.bfloat16).state_dict())
    models = {"trained": trained, "random": rnd}
    for m in models.values():
        m.eval()
        for p_ in m.parameters():
            p_.requires_grad_(False)
    cfg = trained.config
    n_h, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    d_h = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_h
    neg = torch.finfo(torch.bfloat16).min
    uniform_cache: dict = {}

    def uniform_vec(n: int):
        if n not in uniform_cache:
            idx = torch.arange(n, device="cuda", dtype=torch.float32)
            u = torch.tril(torch.ones(n, n, device="cuda")) / (idx[:, None] + 1)
            vec = torch.zeros(n, device="cuda")
            vec[-1] = 1.0
            for _ in range(cfg.num_hidden_layers):
                vec = 0.5 * (vec @ u) + 0.5 * vec
            uniform_cache[n] = vec
        return uniform_cache[n]

    def logit(h, w, b, keep=None):
        pool = h.mean(0) if keep is None else h[keep].mean(0)
        return b + (torch.cat([h[-1], pool]) * w).sum()

    def analyse(m, p0, w, b):
        """Token-level attributions of the probe logit, its value, the final hidden states and the last token's
        attention entropy per layer (each head's, mean over heads; and of the head mean), for one model."""
        emb = m.get_input_embeddings()
        caps: list = []
        hooks = [layer.self_attn.v_proj.register_forward_hook(lambda _m, _i, o: caps.append(o.detach()))
                 for layer in m.model.layers]
        try:
            xe = emb(torch.as_tensor([p0]).cuda()).detach().requires_grad_(True)
            o = m(inputs_embeds=xe, output_attentions=True, output_hidden_states=True)
        finally:
            for h_ in hooks:
                h_.remove()
        h = o.hidden_states[-1][0].float()
        f = logit(h, w, b)
        atts = o.attentions
        grads = torch.autograd.grad(f, atts)
        n = len(p0)
        with torch.no_grad():
            e = torch.zeros(n, device="cuda")
            e[-1] = 1.0
            vr, vg, vn = e.clone(), e.clone(), e.clone()
            for li in range(len(atts) - 1, -1, -1):          # the last row of the product over layers
                A = atts[li][0].float()                       # [heads, n, n]
                vr = 0.5 * (vr @ A.mean(0)) + 0.5 * vr
                vg = vg + vg @ (grads[li][0].float() * A).clamp(min=0).mean(0)
                layer = m.model.layers[li]
                V = caps[li][0].float().view(n, n_kv, d_h)
                Wo = layer.self_attn.o_proj.weight.float()    # [hidden, heads * d]
                norms = torch.stack([(V[:, hh // (n_h // n_kv)] @ Wo[:, hh * d_h:(hh + 1) * d_h].T).norm(dim=-1)
                                     for hh in range(n_h)])   # [heads, n]
                An = (A * norms[:, None, :]).mean(0)
                vn = 0.5 * (vn @ (An / An.sum(-1, keepdim=True).clamp_min(1e-30))) + 0.5 * vn
                del A, An
            axg = torch.zeros(n, device="cuda")
            ent_h, ent_m = [], []
            for li in range(len(atts)):
                axg += (atts[li][0].float() * grads[li][0].float()).clamp(min=0).mean(0)[-1]
                row = atts[li][0, :, -1, :].float()           # [heads, n]
                ent_h.append(float((-(row * (row + 1e-30).log()).sum(-1)).mean() / np.log(n)))
                rm = row.mean(0)
                ent_m.append(float(-(rm * (rm + 1e-30).log()).sum() / np.log(n)))
            vg[-1] -= 1.0                                     # the identity's own term
            u = uniform_vec(n)
            res = {"rollout": vr, "raw_last": atts[-1][0].float().mean(0)[-1], "uniform": u,
                   "rollout_minus_uniform": vr - u, "rollout_lift": vr / u, "attn_x_grad": axg, "grad_rollout": vg,
                   "norm_rollout": vn}
            res = {k: x.cpu().numpy().astype(np.float64) for k, x in res.items()}
        caps.clear()
        del o, atts, grads
        return res, float(f.detach()), h.detach(), {"head": ent_h, "mean": ent_m}

    @torch.no_grad()
    def knockout(m, p0, w, b, positions: dict, h_ref):
        """Per group, the probe logit's fall when attention from every position outside the group to the group's
        tokens is blocked in every layer; both logits pool the mean over the positions outside the group."""
        n = len(p0)
        ids = torch.as_tensor([p0]).cuda()
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
        return out_

    @torch.no_grad()
    def mask_check(m, p0, w, b):
        """The probe logit with the plain causal mask passed as a 4D additive mask, against the default forward."""
        n = len(p0)
        ids = torch.as_tensor([p0]).cuda()
        causal = torch.ones(n, n, dtype=torch.bool, device="cuda").tril()
        mask = torch.zeros(n, n, dtype=torch.bfloat16, device="cuda").masked_fill(~causal, neg)
        f1 = logit(m(input_ids=ids, output_hidden_states=True).hidden_states[-1][0].float(), w, b)
        f2 = logit(m(input_ids=ids, attention_mask=mask[None, None], output_hidden_states=True)
                   .hidden_states[-1][0].float(), w, b)
        return float(f1), float(f2)

    recs = []
    per = open(out / "per_anchor.jsonl", "w")
    checked = False
    for prog, g_ in want.groupby("program"):
        coef = probes[prog.replace("@", "_at_")]
        b, w = float(coef[0]), torch.as_tensor(coef[1:], dtype=torch.float32).cuda()
        for anc in g_["anchor"]:
            x = an.loc[anc]
            sid, hours = str(x["twin"]), float(x["hours"])
            p0 = bench.prompt(rows[sid]["ids"], v, sid, hours)
            if len(p0) > a.attn_max:
                continue
            tg = at.token_groups(p0, v)
            if not checked:
                print(json.dumps({"first_tokens": [v.meta[int(t)].get("cls") or v.meta[int(t)].get("code")
                                                   for t in p0[:6]], "first_groups": tg[:6],
                                  **{f"mask_check_{k}": mask_check(m, p0, w, b) for k, m in models.items()}}),
                      flush=True)
                checked = True
            positions: dict = {}
            for k, gname in enumerate(tg):
                if gname is not None:
                    positions.setdefault(gname, []).append(k)
            size = {gname: len(ps) for gname, ps in positions.items()}
            ing = np.array([gname is not None for gname in tg])
            stat = np.array([gname == "statics" for gname in tg])
            n = len(p0)
            rec = {"program": prog, "anchor": anc, "n": n, "groups": len(positions),
                   "size_max_over_min": max(size.values()) / min(size.values())}
            scores, toks = {}, {}
            for name, m in models.items():
                tok, f, h_ref, ent = analyse(m, p0, w, b)
                toks[name] = tok
                for meth, vec in tok.items():
                    s_ = at.group_sums(vec, tg)
                    scores[(name, meth, "sum")] = s_
                    scores[(name, meth, "mean")] = {gname: s / size[gname] for gname, s in s_.items()}
                scores[(name, "knockout", "sum")] = knockout(m, p0, w, b, positions, h_ref)
                rec[f"{name}_logit"] = f
                rec[f"{name}_mass_statics"] = float(tok["rollout"][stat].sum())
                rec[f"{name}_mass_groups"] = float(tok["rollout"][ing].sum())
                rec[f"{name}_mass_first"] = float(tok["rollout"][0])
                rec[f"{name}_mass_last"] = float(tok["rollout"][-1])
                rec[f"{name}_raw_mass_first"] = float(tok["raw_last"][0])
                rec[f"{name}_entropy_head"] = ent["head"]
                rec[f"{name}_entropy_mean"] = ent["mean"]
                rec[f"{name}_tv_rollout_uniform"] = tv(tok["rollout"], tok["uniform"])
                rec[f"{name}_tv_raw_flat"] = tv(tok["raw_last"], np.ones(n))
                torch.cuda.empty_cache()
            rec["uniform_mass_statics"] = float(toks["trained"]["uniform"][stat].sum())
            for meth in ("rollout", "raw_last", "attn_x_grad", "norm_rollout"):
                rec[f"tok_sanity_{meth}"] = spearman(toks["trained"][meth][ing], toks["random"][meth][ing])
            for name in models:
                rec[f"tok_uniform_rollout_{name}"] = spearman(toks[name]["rollout"][ing], toks[name]["uniform"][ing])
            ed = f4_tr.get((prog, anc), {}).get("edit", {})
            ed_abs = {gname: abs(val) for gname, val in ed.items()}
            gs = sorted(set(positions) & set(ed))
            rec["edit_size"] = cmp(ed, size, gs)
            rec["editabs_size"] = cmp(ed_abs, size, gs)
            for meth in (*TOKEN_METHODS, "knockout"):
                for agg in (("sum", "mean") if meth != "knockout" else ("sum",)):
                    st, sr = scores[("trained", meth, agg)], scores[("random", meth, agg)]
                    un = scores[("trained", "uniform", agg)]
                    tag = f"{meth}_{agg}"
                    rec[f"sanity_{tag}"] = cmp(st, sr, gs)
                    rec[f"edit_{tag}"] = cmp(st, ed, gs)
                    rec[f"editabs_{tag}"] = cmp(st, ed_abs, gs)
                    rec[f"size_{tag}"] = cmp(st, size, gs)
                    rec[f"uniform_{tag}"] = cmp(st, un, gs)
                    rec[f"runiform_{tag}"] = cmp(sr, un, gs)
                    rec[f"rsize_{tag}"] = cmp(sr, size, gs)
                    rec[f"sanity_given_size_{tag}"] = partial(rec[f"sanity_{tag}"], rec[f"size_{tag}"],
                                                              rec[f"rsize_{tag}"])
                    rec[f"edit_given_size_{tag}"] = partial(rec[f"edit_{tag}"], rec[f"size_{tag}"], rec["edit_size"])
            rec["edit_uniform_sum"] = cmp(ed, scores[("trained", "uniform", "sum")], gs)
            for meth, key in (("rollout", "attn_rollout"), ("attn_x_grad", "attn_x_grad")):
                st4, sr4 = f4_tr.get((prog, anc), {}).get(key), f4_rd.get((prog, anc), {}).get(key)
                if st4:
                    rec[f"check_{meth}_trained"] = cmp(scores[("trained", meth, "sum")], st4, sorted(st4))
                if sr4:
                    rec[f"check_{meth}_random"] = cmp(scores[("random", meth, "sum")], sr4, sorted(sr4))
                if st4 and sr4:
                    rec[f"f4_sanity_{meth}"] = cmp(st4, sr4, sorted(set(st4) & set(ed)))
            per.write(json.dumps(rec) + "\n")
            per.flush()
            recs.append(rec)
            if len(recs) % 10 == 0:
                print(json.dumps({"done": len(recs), "seconds": round(time.time() - t0, 1)}), flush=True)
    per.close()
    scalar = sorted({k for r in recs for k, x in r.items() if k not in ("program", "anchor")
                     and (x is None or (isinstance(x, (int, float)) and not isinstance(x, bool)))})
    layered = [k for k in recs[0] if k.endswith(("_entropy_head", "_entropy_mean"))]

    def report(rr):
        d = {k: summary([r.get(k) for r in rr]) for k in scalar}
        for k in layered:
            d[k] = [float(np.median([r[k][li] for r in rr])) for li in range(len(rr[0][k]))]
        return {"anchors": len(rr), **d}

    rep = {"pooled": report(recs), "programs": {p: report([r for r in recs if r["program"] == p])
                                                for p in sorted({r["program"] for r in recs})},
           "seconds": round(time.time() - t0, 1), "args": vars(a)}
    (out / "attn_diag.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"done": len(recs), "seconds": rep["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
