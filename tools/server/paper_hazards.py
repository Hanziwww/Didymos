"""icutwin on the AutoDL host: Paper P amendment 7 (F2, F3; ``docs/p1_paper_protocol.md``): the model's step hazards
along rollouts and along each anchor's real future (``eval.hazards``).

For a design directory (``anchors.parquet``, ``design.json``, ``rollouts/**.parquet``): every rollout (prompt and
continuation) and, with ``--real``, each anchor's real future appended to its prompt (the stay's timeline after the
cut, encoded as the prompt was; anchors whose encodings disagree or that continue a context window are left out and
counted) go through one teacher-forced pass of the gated model (bf16). At each position from the prompt's last on, the
probabilities of the ICU discharge dead and alive and of the first IMV, vasopressor and RRT tokens that the grammar
allows next, renormalized as the sampler draws (over the tokens the grammar's timeline rules and the dataset's
support allow: ``eval.hazards.masked_parts``, ``normalized``; ``--raw``: the model's raw probabilities, as the first
R128 scoring; zero before a clock block, and in a rollout's budget reserve, where the sampler could not draw them),
are summed per time bin while at risk (``eval.hazards.summarize``). Times: the labels'
decode clock of the anchor. ``--a1-caps``: A1's design.json, where each rollout would have been censored under A1's
cap (D3). Writes ``<out>/part_<k>.npz`` (per sequence: anchor index, rollout (-1: real), first events, end, the
increments) and ``<out>/hazards.json``. Needs the GPU to itself.
    python paper_hazards.py --run MUC32_main --tokens <root> --split held_out --design-dir <dir> [--real] [--long]"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/icutwin")
END_RESERVE = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("run", "tokens", "split", "design-dir"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--datasets", nargs="*", default=None, help="the real stays' databases (default: the seven)")
    ap.add_argument("--real", action="store_true", help="also each anchor's real future")
    ap.add_argument("--no-rollouts", action="store_true", help="only the real futures (D1)")
    ap.add_argument("--long", action="store_true", help="bins to 30 days (hourly to 48 h, then 6-hourly)")
    ap.add_argument("--a1-caps", default=None)
    ap.add_argument("--tokens-per-batch", type=int, default=65536)
    ap.add_argument("--part-size", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0, help="anchors (a smoke)")
    ap.add_argument("--check", type=int, default=200,
                    help="completed rollouts whose ICU discharge time is checked against the labels' decode")
    ap.add_argument("--raw", action="store_true",
                    help="the model's raw probabilities (the first R128 scoring), not the sampler's renormalized ones")
    ap.add_argument("--chunk", type=int, default=2048, help="positions per logits chunk")
    a = ap.parse_args()
    t0 = time.time()
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM

    from icutwin.eval import bench
    from icutwin.eval import hazards as hz
    from icutwin.factory import twins
    from icutwin.tokenization.encode import prospective
    from icutwin.tokenization.vocab import Vocab

    tokens, dd = Path(a.tokens), Path(a.design_dir)
    out = Path(a.out or dd / "hazards")
    out.mkdir(parents=True, exist_ok=True)
    v = Vocab.load(tokens / "vocab.json")
    g = hz.grammar(v)
    design = json.loads((dd / "design.json").read_text())
    caps = design.get("caps") or {}
    a1caps = json.loads(Path(a.a1_caps).read_text())["caps"] if a.a1_caps else None
    hf = R / "runs/train/full" / a.run / "final/hf"
    context = int(json.loads((hf / "config.json").read_text())["max_position_embeddings"])
    an = pd.read_parquet(dd / "anchors.parquet").drop_duplicates("anchor").reset_index(drop=True)
    if a.limit:
        an = an.iloc[:a.limit]
    a_idx = {x: i for i, x in enumerate(an["anchor"])}
    edges = np.r_[np.arange(0, 49), np.arange(54, 721, 6)].astype(np.float64) if a.long else \
        np.arange(0, 49).astype(np.float64)

    # rollouts: continuations as one flat int32 array with offsets; each anchor's prompt once
    prompts, members = {}, []                       # members: (anchor index, rollout, flat start, flat end)
    flats, base = [], 0
    if not a.no_rollouts:
        for f in sorted((dd / "rollouts").rglob("*.parquet")):
            t = pq.read_table(f, columns=["anchor", "rollout", "prompt", "continuation"])
            anc, rol = t.column("anchor").to_pylist(), t.column("rollout").to_pylist()
            co = t.column("continuation").combine_chunks()
            off, vals = co.offsets.to_numpy(), co.values.to_numpy().astype(np.int32)
            pr = t.column("prompt")
            for r, (x, k) in enumerate(zip(anc, rol, strict=True)):
                if x not in a_idx:
                    continue
                if x not in prompts:
                    prompts[x] = np.asarray(pr[r].as_py(), dtype=np.int32)
                members.append((a_idx[x], int(k), base + int(off[r]), base + int(off[r + 1])))
            flats.append(vals)
            base += len(vals)
            del t
    flat = np.concatenate(flats) if flats else np.zeros(0, np.int32)
    # the real futures: the prompt and the stay's timeline after it, encoded alike
    real_cont, excluded = {}, {"no_prompt": 0, "window": 0, "mismatch": 0}
    rows = {r["stay_id"]: r for r in bench.real_rows(tokens, a.split, set(an["twin"].astype(str)),
                                                     a.datasets or bench.DATASETS)}
    tl, cont_tok = g.tl, v.id("[CONT]")
    for i, x in enumerate(an.itertuples()):
        sid = str(x.twin)
        if x.anchor not in prompts:
            if sid not in rows:
                excluded["no_prompt"] += 1
                continue
            prompts[x.anchor] = np.asarray(bench.prompt(rows[sid]["ids"], v, sid, float(x.hours)), dtype=np.int32)
        if not a.real:
            continue
        p = prompts[x.anchor]
        st = twins.decode_labels(rows[sid]["ids"], v, sid)
        last_h = max((e[0] - st.icu_in).total_seconds() for e in st.timeline) / 3600 if st.timeline else 0.0
        full = np.asarray(prospective(st, v, last_h + 1.0), dtype=np.int32)      # the whole stay, as known at its end
        if tl not in p or tl not in full or cont_tok in p or cont_tok in full:
            excluded["window"] += 1                  # a continued context window: the encodings need not align
            continue
        ptl = p[int(np.flatnonzero(p == tl)[0]) + 1:]
        ftl = full[int(np.flatnonzero(full == tl)[0]) + 1:]
        if len(ftl) < len(ptl) or not np.array_equal(ftl[:len(ptl)], ptl):
            excluded["mismatch"] += 1
            continue
        real_cont[i] = ftl[len(ptl):][:max(0, context - len(p))]
    print(json.dumps({"anchors": len(an), "rollouts": len(members), "real": len(real_cont), "excluded": excluded,
                      "setup_s": round(time.time() - t0, 1)}), flush=True)

    # clocks: the prompt's draws once per anchor, the rng state after them shared by its members
    clocks: dict = {}
    ck_of: dict = {}
    for x in an.itertuples():
        if x.anchor not in prompts:
            continue
        p = prompts[x.anchor]
        ck = ck_of.setdefault(x.dataset, hz.Clock(v, x.dataset))
        rng = twins.clock_rng(str(x.twin), 0)
        tlp = int(np.flatnonzero(p == tl)[0]) if tl in p else 0
        mins = ck.minutes(p, rng, tlp)
        adm = np.flatnonzero(g.adm[p])
        clocks[x.anchor] = (ck, rng.getstate(), float(mins[-1]), float(mins[adm[0]]) if len(adm) else np.nan)
    checks = []                                     # the labels' decode against this clock (ICU discharge times)
    model = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.eval()
    W = model.get_output_embeddings().weight                       # [V, d]
    union = np.unique(np.concatenate([g.sets[c] for c in hz.CAUSES]))
    cols = {c: np.searchsorted(union, g.sets[c]) for c in hz.CAUSES}
    U = torch.as_tensor(union, device="cuda")
    # the sampler's renormalization: the grammar's timeline rules and the dataset's support (eval.hazards.Normalizer)
    from icutwin.generate.support import Support
    from icutwin.model.data import layout

    nz = hz.normalizer(v)
    lay = layout(v)
    support = Support.load(v, tokens / "support.json") if (tokens / "support.json").exists() else None
    sets_dev = {c: torch.as_tensor(g.sets[c], device="cuda") for c in hz.CAUSES}
    sup_cache: dict = {}

    def support_of(seq):
        tok = int(seq[lay.dataset_pos])
        if tok not in sup_cache:
            m = support.mask(tok) if support is not None else None
            sup_cache[tok] = torch.as_tensor(m if m is not None else np.ones(len(v.tokens), bool), device="cuda").float()
        return sup_cache[tok]

    seqs = [(ai, k, s0, s1) for ai, k, s0, s1 in members] + [(i, -1, None, None) for i in sorted(real_cont)]

    def tokens_of(m):
        ai, k, s0, s1 = m
        p = prompts[an["anchor"][ai]]
        c = flat[s0:s1] if k >= 0 else real_cont[ai]
        return p, c

    lengths = np.array([len(prompts[an["anchor"][m[0]]]) + (m[3] - m[2] if m[1] >= 0 else len(real_cont[m[0]]))
                        for m in seqs])
    order = np.argsort(lengths, kind="stable")
    res = []                                                        # per sequence: (ai, k, Path, minutes, tokens)
    part = 0
    stats = {"sequences": 0, "positions": 0}

    def flush(final=False):
        nonlocal res, part
        if not res or (len(res) < a.part_size and not final):
            return
        arr = {"anchor_idx": np.array([r[0] for r in res], np.int32), "rollout": np.array([r[1] for r in res], np.int16),
               "end_hour": np.array([r[2].end_hour for r in res], np.float32),
               "end_kind": np.array([r[2].end_kind for r in res], np.int8),
               "cut_hour": np.array([r[2].cut_hour for r in res], np.float32),
               "cont_minutes": np.array([r[3] for r in res], np.float32),
               "cont_tokens": np.array([r[4] for r in res], np.int32), "edges": edges}
        nb48 = 48
        for c in hz.CAUSES:
            arr[f"first_{c}"] = np.array([r[2].first[c] for r in res], np.float32)
            short = c in hz.TREATMENTS
            arr[f"inc_{c}"] = np.stack([r[2].inc[c][:nb48] if short else r[2].inc[c] for r in res]).astype(np.float32)
            if short:
                arr[f"comp_{c}"] = np.stack([r[2].comp[c][:nb48] for r in res]).astype(np.float32)
        if a1caps is not None:
            for c in ("death", "alive"):
                arr[f"inccut_{c}"] = np.stack([r[2].inc_cut[c] if r[2].inc_cut is not None else r[2].inc[c]
                                               for r in res]).astype(np.float32)
        np.savez_compressed(out / f"part_{part:04d}.npz", **arr)
        part += 1
        res = []

    k = 0
    with torch.no_grad():
        while k < len(order):
            n = 1
            while k + n < len(order) and (n + 1) * lengths[order[k + n]] <= a.tokens_per_batch:
                n += 1
            idx = order[k:k + n]
            width = int(lengths[idx].max())
            x = torch.zeros((len(idx), width), dtype=torch.long)
            att = torch.zeros((len(idx), width), dtype=torch.long)
            pcs = []
            for r, i in enumerate(idx):
                p, c = tokens_of(seqs[i])
                s = np.concatenate([p, c]).astype(np.int64)
                x[r, :len(s)] = torch.as_tensor(s)
                att[r, :len(s)] = 1
                pcs.append((p, c, s))
            hid = model.model(input_ids=x.cuda(), attention_mask=att.cuda()).last_hidden_state
            for r, i in enumerate(idx):
                ai, kk, _s0, _s1 = seqs[i]
                p, c, s = pcs[r]
                P, n_s = len(p), len(s)
                pos = np.arange(P - 1, n_s - 1)                    # predicting tokens P .. n_s - 1
                gf = hz.group_first(s, g)[pos]
                nxt_clock = g.clock[s[pos + 1]]
                haz = {}
                if a.raw:
                    probs = []
                    for c0 in range(0, len(pos), a.chunk):
                        h = hid[r, torch.as_tensor(pos[c0:c0 + a.chunk], device="cuda")]
                        lg = (h @ W.T).float()
                        lse = torch.logsumexp(lg, -1, keepdim=True)
                        probs.append(torch.exp(lg[:, U] - lse).cpu().numpy())
                    pu = np.concatenate(probs) if probs else np.zeros((0, len(union)))
                    per = {cz: hz.set_hazard(pu[:, cols[cz]], g.sets[cz], gf) for cz in hz.CAUSES}
                else:
                    sup = support_of(s)
                    gft = torch.as_tensor(gf, device="cuda")
                    acc: dict = {}
                    for c0 in range(0, len(pos), a.chunk):
                        h = hid[r, torch.as_tensor(pos[c0:c0 + a.chunk], device="cuda")]
                        prt = hz.masked_parts(h @ W.T, nz, sets_dev, sup, gft[c0:c0 + a.chunk])
                        for k_, x_ in prt.items():
                            acc.setdefault(k_, []).append(x_.cpu().numpy())
                    parts = {k_: np.concatenate(x_) for k_, x_ in acc.items()} if acc else None
                    per = hz.normalized(parts, s[pos], nz, hz.CAUSES, hz.running_state(s, pos, nz)) if parts \
                        else {cz: np.zeros(0) for cz in hz.CAUSES}
                for cz in hz.CAUSES:
                    h_ = np.asarray(per[cz], dtype=np.float64)
                    h_[nxt_clock] = 0.0
                    full_h = np.zeros(n_s)
                    full_h[pos] = h_
                    haz[cz] = full_h
                anchor = an["anchor"][ai]
                ck, state, m_prompt, m_adm = clocks[anchor]
                rng = random.Random()
                rng.setstate(state)
                mins = np.r_[np.zeros(P), ck.minutes(c, rng, 0)] + m_prompt
                mins[:P] = m_prompt                                 # the prompt's own times are not needed
                hours = (mins - m_prompt) / 60.0
                if kk >= 0 and len(checks) < a.check and v.id("[EOS]") in c:
                    dis = np.flatnonzero(np.isin(c, np.r_[g.sets["death"], g.sets["alive"]]))
                    if len(dis) and np.isfinite(m_adm):
                        st = twins.decode_labels([int(t_) for t_ in s], v, f"{anchor}#{kk}",
                                                 clock_id=str(an["twin"][ai]))
                        mine = mins[P + int(dis[0]) - 1] - m_adm
                        checks.append(abs((st.icu_out - st.icu_in).total_seconds() / 60 - mine))
                ds = an["dataset"][ai]
                stop = n_s - 1
                ends = np.flatnonzero(c == g.end)
                if len(ends):
                    stop = min(stop, P + int(ends[0]) - 1)
                cut = None
                if kk >= 0:
                    max_new = min(int(caps.get(ds) or 4096), context - P - 1)
                    stop = min(stop, P + max_new - END_RESERVE - 3)
                    if a1caps is not None:
                        mn1 = min(int(a1caps.get(ds) or 4096), context - P - 1)
                        cut = P + mn1 - END_RESERVE - 3
                path = hz.summarize(s, P - 1, haz, hours, edges, g, stop_pos=stop, cut_pos=cut)
                res.append((ai, kk, path, float(mins[-1] - m_prompt), int(len(c))))
                stats["sequences"] += 1
                stats["positions"] += int(len(pos))
            del hid
            k += n
            flush()
            if stats["sequences"] % 20000 < n:
                print(json.dumps({**stats, "seconds": round(time.time() - t0, 1)}), flush=True)
    flush(final=True)
    stats["clock_check"] = {"n": len(checks), "max_abs_minutes": float(max(checks)) if checks else None}
    (out / "hazards.json").write_text(json.dumps({**vars(a), **stats, "excluded": excluded, "parts": part,
                                                  "anchors": int(len(an)), "context": context,
                                                  "seconds": round(time.time() - t0, 1)}, indent=1))
    print(json.dumps({"done": stats, "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
