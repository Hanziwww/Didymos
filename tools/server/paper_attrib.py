"""icutwin on the AutoDL host: Paper P's attribution (``docs/p1_paper_protocol.md`` C2; ``eval.attribution``). For one
design's held-out anchors (``--anchors-from``, with their states and the Twin probe of each program,
``tools/paper_probe.py``'s ``probe_models.npz``): per program, the ``--top`` anchors with the highest probe risk and
``--random`` at random (prompts of at most ``--max-tokens``). For each anchor, the probe's logit
f = b + w . [last state, mean state] of the frozen model, and its attribution to the event groups by
counterfactual editing (every group), integrated gradients (``--steps`` midpoint steps from the padding
embedding), attention
rollout and attention x gradient (prompts of at most ``--attn-max`` tokens); deletion curves of each ranking and of a
random one. Writes ``<out>/attrib.jsonl`` (one anchor and program per line: group scores, curves; host only) and
``<out>/rollout_prompts.parquet`` (the original prompt and the 3 highest and 3 lowest edits of the first ``--rollout``
anchors per program, for rollout-level checks). Needs the GPU to itself.

Amendment 7 (F4): ``--select tophalf`` picks ``--n-select`` anchors per program at random (``--seed``) among those
whose label is valid (``--labels``: A1's predictions) and whose probe risk is in the upper half; ``--anchors-file``
reuses a run's anchors; ``--rollout-all`` writes the original and every group's edit of every anchor for rollouts,
each with its own prediction time (its last event on the stay's clock: the generator predicts from a prompt's end);
the sanity checks run the same anchors with ``--random-backbone`` (the model's weights randomly initialized, the
probe kept) or ``--probe-file`` (a probe fit on permuted labels), ``--no-curves`` skipping the deletion curves.
    python paper_attrib.py --run MUC32_main --tokens <root> --split held_out --store <real_test> \\
        --anchors-from <design dir> --probe <dir with probe_models.npz> --programs icu_death@-1 --out <dir>"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/icutwin")


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("run", "tokens", "split", "store", "anchors-from", "probe", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--programs", nargs="+", required=True)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--random", type=int, default=100)
    ap.add_argument("--rollout", type=int, default=50)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--attn-max", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", choices=("top_random", "tophalf"), default="top_random")
    ap.add_argument("--n-select", type=int, default=40)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--anchors-file", default=None)
    ap.add_argument("--rollout-all", action="store_true")
    ap.add_argument("--random-backbone", type=int, default=None)
    ap.add_argument("--probe-file", default=None)
    ap.add_argument("--no-curves", action="store_true")
    ap.add_argument("--out-name", default="attrib.jsonl")
    a = ap.parse_args()
    t0 = time.time()
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM

    from icutwin.eval import attribution as at
    from icutwin.eval import bench
    from icutwin.factory import twins
    from icutwin.tokenization.encode import prospective_stay
    from icutwin.tokenization.vocab import Vocab

    tokens, out = Path(a.tokens), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v = Vocab.load(tokens / "vocab.json")
    an = pq.read_table(Path(a.anchors_from) / "anchors.parquet").to_pandas()
    st = np.load(Path(a.anchors_from) / "states.npz")
    feats = np.hstack([st["last"].astype(np.float32), st["mean"].astype(np.float32)])
    assert list(st["anchor"]) == an["anchor"].tolist()
    models = np.load(a.probe_file or Path(a.probe) / "probe_models.npz")
    rows = {r["stay_id"]: r for r in bench.real_rows(tokens, a.split, set(an["twin"].astype(str)))}
    hf = R / "runs/train/full" / a.run / "final/hf"
    m_sdpa = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    m_eager = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda()
    if a.random_backbone is not None:                # amendment 7 F4: the same architecture, random weights
        from transformers import AutoConfig

        torch.manual_seed(a.random_backbone)
        cfg = AutoConfig.from_pretrained(hf)
        rnd = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
        sd = rnd.state_dict()
        m_sdpa.load_state_dict(sd)
        m_eager.load_state_dict(sd)
        del rnd
    labels = None
    if a.labels:
        import pandas as pd

        labels = pd.read_parquet(a.labels)
    fixed = None
    if a.anchors_file:
        fixed = pq.read_table(a.anchors_file).to_pandas()
    for m in (m_sdpa, m_eager):
        m.eval()
        for p_ in m.parameters():
            p_.requires_grad_(False)
    emb = m_sdpa.get_input_embeddings()
    rng = np.random.default_rng(a.seed)

    def logit_of(model, ids=None, embeds=None, w=None, b=0.0, attentions=False):
        kw = {"inputs_embeds": embeds} if embeds is not None else {"input_ids": torch.as_tensor([ids]).cuda()}
        o = model(**kw, output_hidden_states=True, output_attentions=attentions)
        h = o.hidden_states[-1][0].float()
        z = torch.cat([h[-1], h.mean(0)])
        return b + (z * w).sum(), o

    @torch.no_grad()
    def risk_many(prompts, w, b):
        return [float(logit_of(m_sdpa, ids=p, w=w, b=b)[0]) for p in prompts]

    lines = open(out / a.out_name, "w")
    roll = []
    for prog in a.programs:
        key = prog.replace("@", "_at_")
        if key not in models:
            print(json.dumps({"skip": prog, "reason": "no probe"}), flush=True)
            continue
        coef = models[key]
        b, w = float(coef[0]), torch.as_tensor(coef[1:], dtype=torch.float32).cuda()
        risk = feats @ coef[1:] + coef[0]
        ok = (an["prompt_len"].to_numpy() <= a.max_tokens) & an["twin"].astype(str).isin(rows).to_numpy()
        if fixed is not None:                        # the anchors of an earlier run (the sanity checks)
            want = fixed.loc[fixed["program"] == prog, "anchor"].tolist()
            pos_of = {x: i for i, x in enumerate(an["anchor"])}
            pick, top = [pos_of[x] for x in want if x in pos_of], []
        elif a.select == "tophalf":                  # amendment 7 F4: at random among the upper half of probe risk
            if labels is not None:
                lab = labels[labels["program"] == prog].set_index("anchor")["label"]
                ok &= an["anchor"].map(lambda x: np.isfinite(lab.get(x, np.nan))).to_numpy()
            idx_ok = np.flatnonzero(ok)
            upper = idx_ok[risk[idx_ok] >= np.median(risk[idx_ok])]
            pick = list(np.sort(rng.choice(upper, size=min(a.n_select, len(upper)), replace=False)))
            top = []
        else:
            idx_ok = np.flatnonzero(ok)
            top = idx_ok[np.argsort(-risk[idx_ok])[:a.top]]
            rest = np.setdiff1d(idx_ok, top)
            pick = list(top) + list(rng.choice(rest, size=min(a.random, len(rest)), replace=False))
        for n_done, i in enumerate(pick):
            r = an.iloc[i]
            sid, hours = str(r["twin"]), float(r["hours"])
            s = prospective_stay(twins.decode_labels(rows[sid]["ids"], v, sid), hours)
            p0 = bench.prompt(rows[sid]["ids"], v, sid, hours)
            groups = at.stay_groups(s, hours)
            f0 = risk_many([p0], w, b)[0]
            names = [g for g, ix in groups.items() if ix or g == "statics"]
            edits = {g: at.edited_prompt(s, v, hours, g, groups) for g in names}
            fe = dict(zip(names, risk_many([edits[g] for g in names], w, b), strict=True))
            edit_effect = {g: f0 - fe[g] for g in names}            # > 0: the group raised the risk
            tg = at.token_groups(p0, v)
            # integrated gradients from the padding embedding (protocol C2; a zero baseline fails under RMSNorm:
            # the output barely changes along the path until its very start, which the steps miss), midpoint steps
            x = emb(torch.as_tensor([p0]).cuda()).float()
            base = emb(torch.full_like(torch.as_tensor([p0]), v.id("[PAD]")).cuda()).float()
            total = torch.zeros_like(x)
            for k in range(a.steps):
                xk = (base + (k + 0.5) / a.steps * (x - base)).to(torch.bfloat16).requires_grad_(True)
                fk, _o = logit_of(m_sdpa, embeds=xk, w=w, b=b)
                (g_,) = torch.autograd.grad(fk, xk)
                total += g_.float()
            ig_tok = ((x - base) * total / a.steps).sum(-1)[0].detach().cpu().numpy()
            with torch.no_grad():
                f_zero = float(logit_of(m_sdpa, embeds=base.to(torch.bfloat16), w=w, b=b)[0])
            ig = at.group_sums(ig_tok, tg)
            res = {"program": prog, "anchor": r["anchor"], "dataset": r["dataset"], "top": bool(n_done < len(top)),
                   "prompt_len": int(len(p0)), "f0": f0, "edit": edit_effect, "ig": ig,
                   "ig_completeness": {"sum": float(ig_tok.sum()), "f_minus_f_zero": f0 - f_zero}}
            if len(p0) <= a.attn_max:
                xe = emb(torch.as_tensor([p0]).cuda()).detach().requires_grad_(True)   # a graph to the attentions
                fa, o = logit_of(m_eager, embeds=xe, w=w, b=b, attentions=True)
                atts = o.attentions
                grads = torch.autograd.grad(fa, atts)
                n = len(p0)
                roll_m = torch.eye(n, device=xe.device)
                axg = torch.zeros(n, device=xe.device)
                for A, G in zip(atts, grads, strict=True):
                    Am = A[0].float().mean(0)
                    roll_m = (0.5 * Am + 0.5 * torch.eye(n, device=xe.device)) @ roll_m
                    axg += (A[0].float() * G[0].float()).clamp(min=0).mean(0)[-1]
                res["attn_rollout"] = at.group_sums(roll_m[-1].detach().cpu().numpy(), tg)
                res["attn_x_grad"] = at.group_sums(axg.detach().cpu().numpy(), tg)
                del atts, grads, o

            def risk_without(drop):
                if not drop:
                    return f0
                return risk_many([at.edited_prompt(s, v, hours, set(drop), groups)], w, b)[0]

            curves = {}
            if not a.no_curves:
                for meth in ("edit", "ig", "attn_rollout", "attn_x_grad"):
                    if meth in res:
                        order = sorted(res[meth], key=lambda g: -res[meth][g])
                        curves[meth] = at.deletion_curve(risk_without, order)
                curves["random"] = at.deletion_curve(risk_without, list(rng.permutation(names)))
            res["deletion"] = curves
            lines.write(json.dumps(res) + "\n")
            lines.flush()
            if a.rollout_all:                        # amendment 7 F4: the original and every group's edit
                for g, pr_ in [("", p0)] + [(g, edits[g]) for g in names]:
                    roll.append({"id": f"{prog}|{r['anchor']}|{g or 'orig'}", "program": prog, "anchor": r["anchor"],
                                 "twin": sid, "dataset": r["dataset"], "group": g, "prompt": pr_,
                                 "t_min": int(bench.anchor_time(pr_, v, sid))})
            elif n_done < a.rollout:
                ranked = sorted(edit_effect, key=lambda g: -edit_effect[g])
                sel = ranked[:3] + ranked[-3:]
                roll.append({"id": f"{prog}|{r['anchor']}|orig", "program": prog, "anchor": r["anchor"], "twin": sid,
                             "dataset": r["dataset"], "group": "", "prompt": p0, "t_min": int(r["t_min"])})
                for g in sel:
                    roll.append({"id": f"{prog}|{r['anchor']}|{g}", "program": prog, "anchor": r["anchor"],
                                 "twin": sid, "dataset": r["dataset"], "group": g, "prompt": edits[g],
                                 "t_min": int(r["t_min"])})
            if n_done % 20 == 0:
                print(json.dumps({"program": prog, "done": n_done + 1, "of": len(pick),
                                  "seconds": round(time.time() - t0, 1)}), flush=True)
    lines.close()
    pq.write_table(pa.table({k: [x[k] for x in roll] for k in ("id", "program", "anchor", "twin", "dataset", "group",
                                                                 "t_min")} |
                            {"prompt": pa.array([list(map(int, x["prompt"])) for x in roll], type=pa.list_(pa.int32()))}),
                   out / "rollout_prompts.parquet")
    print(json.dumps({"done": True, "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
