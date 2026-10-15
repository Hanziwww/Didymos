"""``icutwin.eval.cli``: the M7 privacy gate of the frozen candidate (``docs/m7_privacy.md``), step by step.

Steps (``step=``): ``p1`` (membership inference by loss, one GPU), ``p2`` (prefix-prompted extraction, one GPU),
``sample`` (audit samples: shard ``shard`` of ``shards``, one GPU each), ``synthetic`` (P3-P5 with both controls,
one GPU), ``report`` (statuses against the protocol's criteria). Results go to ``out_dir`` as JSON (aggregates);
audit samples stay in ``out_dir/audit_samples`` and are deleted by ``step=purge`` once the gate is decided.
``p5v2``: P5 of the proposed amendment v2 on its own sets and samples (``out_dir/audit_samples_p5v2``); it runs, and
the report uses it, only once the amendment is adopted (``p5v2.adopted``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from icutwin.eval import privacy as pv
from icutwin.model.data import layout
from icutwin.tokenization.vocab import Vocab

log = logging.getLogger(__name__)


def _device(cfg):
    import torch

    return torch.device(f"cuda:{int(cfg.gpu)}" if torch.cuda.is_available() else "cpu")


def _write(out: Path, name: str, obj) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    log.info("%s: %s", name, json.dumps(obj, default=float)[:3000])


def _model(cfg, dev):
    """The candidate's weights, with their sha256 checked against the run's FROZEN.json when there is one (the gate
    runs on the frozen candidate only)."""
    import hashlib

    model_dir = Path(cfg.model_dir)
    h = hashlib.sha256()
    with (model_dir / "model.safetensors").open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    info = {"model_dir": str(model_dir), "weights_sha256": h.hexdigest()}
    frozen = model_dir.parent.parent / "FROZEN.json"
    if frozen.exists():
        rel = (model_dir / "model.safetensors").relative_to(frozen.parent).as_posix()
        want = json.loads(frozen.read_text())["checkpoint_files_sha256"][rel]
        if want != info["weights_sha256"]:
            raise ValueError(f"{rel} differs from {frozen}: {info['weights_sha256']} != {want}")
        info["frozen_record"] = str(frozen)
    return pv.load_model(model_dir, dev), info


def _ctx(cfg):
    tokens = Path(cfg.tokens_root)
    v = Vocab.load(tokens / "vocab.json")
    return tokens, v, layout(v), Path(cfg.out_dir)


def step_p1(cfg) -> dict:
    tokens, v, lay, out = _ctx(cfg)
    dev = _device(cfg)
    model, weights = _model(cfg, dev)
    rng = np.random.default_rng(int(cfg.seed))
    members, nonmembers = {}, {}
    for ds in cfg.datasets:
        n_tr, n_ho = len(pv.stay_ids(tokens, ds, "train")), len(pv.stay_ids(tokens, ds, "held_out"))
        n = min(int(cfg.p1.cap), n_tr, n_ho)
        for split, total, dest in (("train", n_tr, members), ("held_out", n_ho, nonmembers)):
            rows = np.sort(rng.choice(total, size=n, replace=False))
            stays = pv.stay_windows(tokens, ds, split, rows)
            flat = [w for s in stays for w in s]
            nll = pv.window_nll(model, flat, lay, dev, int(cfg.tokens_per_batch))
            per, i = [], 0
            for s in stays:
                per.append(nll[i:i + len(s)])
                i += len(s)
            dest[ds] = pv.membership_scores(per)
        log.info("P1 %s: %d stays per side", ds, n)
    rep = pv.p1_report(members, nonmembers, float(cfg.p1.auc_limit), float(cfg.p1.tpr_limit), float(cfg.p1.tpr_fail),
                       int(cfg.seed))
    rep["model"] = weights
    _write(out, "p1", rep)
    return rep


def _end_ids(v: Vocab) -> set[int]:
    return {v.id("[END]"), v.id("[EOS]"), v.id("[CONT]")}


def step_p2(cfg) -> dict:
    tokens, v, lay, out = _ctx(cfg)
    dev = _device(cfg)
    model, weights = _model(cfg, dev)
    rng = np.random.default_rng(int(cfg.seed))
    eos = v.id("[EOS]")
    results = {}
    firsts = {(ds, sp): pv.first_windows(tokens, ds, sp) for ds in cfg.datasets for sp in ("train", "held_out")}
    for k, length in cfg.p2.settings:
        key = f"k{k}_L{length}"
        results[key] = {}
        for ds in cfg.datasets:
            sides = []
            for sp in ("train", "held_out"):
                items = pv.extraction_items(firsts[(ds, sp)], lay, int(k), int(length), _end_ids(v))
                sides.append(items)
            n = min(int(cfg.p2.cap), *(len(s) for s in sides))
            if n == 0:
                log.info("P2 %s %s: no first window with %d timeline tokens on both sides", key, ds, int(k) + int(length))
                continue
            stats = []
            for items in sides:
                pick = [items[i] for i in np.sort(rng.choice(len(items), size=n, replace=False))]
                prompts = np.stack([p for p, _t in pick])   # first-window heads have one length
                conts = []
                for a in range(0, n, int(cfg.p2.batch)):
                    conts += pv.generate(model, prompts[a:a + int(cfg.p2.batch)], int(length), dev, lay, eos,
                                         sample=False, min_new=int(length))
                stats.append(pv.match_stats(conts, [t for _p, t in pick]))
            results[key][ds] = (stats[0][0], stats[1][0], stats[0][1], stats[1][1])
            log.info("P2 %s %s: %d per side, exact %.4f / %.4f", key, ds, n, stats[0][0].mean(), stats[1][0].mean())
    rep = pv.p2_report(results, float(cfg.p2.limit), int(cfg.seed))
    rep["model"] = weights
    _write(out, "p2", rep)
    return rep


def step_sample(cfg) -> None:
    """Shard ``shard`` of ``shards`` of the audit samples (seed + shard), saved as an int32 list column."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tokens, v, lay, out = _ctx(cfg)
    dev = _device(cfg)
    model, _weights = _model(cfg, dev)
    n = int(cfg.sample.n) // int(cfg.shards) + (1 if int(cfg.shard) < int(cfg.sample.n) % int(cfg.shards) else 0)
    seqs, b = [], int(cfg.sample.batch)
    for a in range(0, n, b):
        m = min(b, n - a)
        prompts = np.full((m, 1), lay.bos, dtype=np.int32)
        conts = pv.generate(model, prompts, int(cfg.sample.max_new), dev, lay, v.id("[EOS]"), sample=True,
                            seed=int(cfg.seed) * 1000 + int(cfg.shard) * 100 + a // b)
        seqs += [np.r_[np.int32(lay.bos), c] for c in conts]
        log.info("sample shard %s: %d / %d", cfg.shard, len(seqs), n)
    d = out / "audit_samples"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"ids": pa.array([s.tolist() for s in seqs], type=pa.list_(pa.int32()))}),
                   d / f"shard_{int(cfg.shard)}.parquet")
    (d / "README.txt").write_text("AUDIT-ONLY samples of the frozen candidate for the M7 privacy gate "
                                  "(docs/m7_privacy.md). Not synthetic data: never decode, analyse otherwise, share or "
                                  "train on them; deleted when the gate is decided.\n", encoding="utf-8")


def _dataset_of(w: np.ndarray, v: Vocab, lay) -> str:
    tok = v.tokens[int(w[lay.dataset_pos])] if len(w) > lay.dataset_pos else ""
    return tok.split("//")[-1] if tok.startswith("PREFIX//dataset//") else "unknown"


def step_synthetic(cfg) -> dict:
    import pyarrow.parquet as pq
    import torch
    from sklearn.decomposition import TruncatedSVD

    tokens, v, lay, out = _ctx(cfg)
    dev = _device(cfg)
    rng = np.random.default_rng(int(cfg.seed))
    exclude = {i for i, m in enumerate(v.meta) if m.get("cls") == "special"}
    end, death = v.id("[END]"), v.id("MEDS_DEATH")
    samples = [np.asarray(x, dtype=np.int32) for f in sorted((out / "audit_samples").glob("shard_*.parquet"))
               for x in pq.read_table(f).column("ids").to_pylist()]
    from icutwin.tokenization.fsm import accepts, compile_tables

    tables = compile_tables(v)
    valid = float(np.mean([accepts(tables, s.tolist(), whole=False)[0] for s in samples]))
    # reference: every train first window; pools of held-out first windows
    train, train_ds = [], []
    held, held_ds = [], []
    for ds in cfg.datasets:
        tw = pv.first_windows(tokens, ds, "train")
        train += tw
        train_ds += [ds] * len(tw)
        hw = pv.first_windows(tokens, ds, "held_out")
        held += hw
        held_ds += [ds] * len(hw)
    n_pool = int(cfg.synthetic.pool)
    perm = rng.permutation(len(held))
    h_base, h_ref, h_priv = (perm[i * n_pool:(i + 1) * n_pool] for i in range(3))
    members = np.sort(rng.choice(len(train), size=n_pool, replace=False))
    x_train, idf = pv.tfidf(train, len(v.tokens), exclude)
    ref = pv.dense_rows(x_train, dev)
    train_hashes = {pv.window_hash(w) for w in train}

    def rep(ws):
        return pv.dense_rows(pv.tfidf(ws, len(v.tokens), exclude, idf)[0], dev)

    def p3_side(ws, ds_of):
        q = rep(ws)
        d1, d2, i1 = pv.nearest(q, ref)
        per = {}
        for d in sorted(set(ds_of)):
            m = np.array([x == d for x in ds_of])
            per[d] = ([pv.window_hash(w) for w, k in zip(ws, m) if k], d1[m], d2[m], i1[m])
        return per, i1

    base_ws = [held[i] for i in h_base]
    base, base_i = p3_side(base_ws, [held_ds[i] for i in h_base])
    gens = {"model": (samples, [_dataset_of(s, v, lay) for s in samples]),
            "private_reference": ([held[i] for i in h_priv], [held_ds[i] for i in h_priv]),
            "copying_generator": ([train[i] for i in members], [train_ds[i] for i in members])}
    sides = {name: p3_side(ws, ds_of) for name, (ws, ds_of) in gens.items()}
    # each nearest train window's distance to its own nearest other train window (authenticity)
    need = np.unique(np.concatenate([base_i, *(i1 for _p, i1 in sides.values())]))
    ref_nn = np.full(len(train), np.nan)
    with torch.no_grad():
        for a in range(0, len(need), 1024):
            idx = need[a:a + 1024]
            sim = (ref[torch.as_tensor(idx, device=dev)] @ ref.T).float()
            sim[torch.arange(len(idx), device=dev), torch.as_tensor(idx, device=dev)] = -float("inf")
            ref_nn[idx] = 1 - sim.max(1).values.cpu().numpy()
    p3 = {name: pv.p3_report(per, base, train_hashes, ref_nn, float(cfg.synthetic.copy_limit),
                             float(cfg.synthetic.share_limit), int(cfg.seed)) for name, (per, _i) in sides.items()}
    # P4 DOMIAS in a truncated SVD of the representation (fitted on a sample of train first windows)
    fit = np.sort(rng.choice(len(train), size=min(50000, len(train)), replace=False))
    svd = TruncatedSVD(n_components=int(cfg.synthetic.svd_dims), random_state=int(cfg.seed)).fit(x_train[fit])
    sparse_of = lambda ws: pv.tfidf(ws, len(v.tokens), exclude, idf)[0]  # noqa: E731
    x_ref, x_mem, x_non = sparse_of([held[i] for i in h_ref]), x_train[members], sparse_of(base_ws)
    p4 = {name: pv.p4_domias(sparse_of(ws), x_ref, x_mem, x_non, svd, float(cfg.synthetic.domias_limit), int(cfg.seed))
          for name, (ws, _d) in gens.items()}
    # P5 attribute inference: complete stays (end block) and samples that reach [EOS]
    def complete(ws):   # with a timeline and an end block (an unconstrained sample can lack either)
        keep = [(w, pv.end_flag(w, death, end)) for w in ws if lay.tl in w]
        return [w for w, y in keep if y is not None], np.array([int(y) for _w, y in keep if y is not None])

    def qi(ws):
        return pv.dense_rows(pv.tfidf([pv.qi_tokens(w, lay, int(cfg.synthetic.qi_tokens)) for w in ws], len(v.tokens),
                                      exclude, idf)[0], dev)

    mem_w, mem_y = complete([train[i] for i in members])
    non_w, non_y = complete(base_ws)
    p5 = {}
    for name, (ws, _d) in gens.items():
        syn_w, syn_y = complete(ws)
        if len(syn_w) < 2 or len(mem_w) < 2 or len(non_w) < 2:   # nothing to infer from (or about)
            p5[name] = {"status": "PENDING", "samples_complete": len(syn_w), "members": len(mem_w), "nonmembers": len(non_w)}
            continue
        p5[name] = pv.p5_attribute(qi(syn_w), syn_y, qi(mem_w), mem_y, qi(non_w), non_y,
                                   float(cfg.synthetic.attribute_limit), int(cfg.seed))
        p5[name]["samples_complete"] = len(syn_w)
    rep_ = {"samples": len(samples), "grammar_valid_share": valid,
            "datasets_of_samples": dict(sorted({d: sum(1 for s in samples if _dataset_of(s, v, lay) == d)
                                                for d in {_dataset_of(s, v, lay) for s in samples}}.items())),
            "p3": p3, "p4": p4, "p5": p5}
    _write(out, "synthetic", rep_)
    return rep_


def v1_sets(seed: int, n_held: int, n_train: int, n_pool: int) -> tuple:
    """The held-out pools (H_base, H_ref, H_priv) and the P4 members ``step_synthetic`` draws: its first two draws
    from ``default_rng(seed)``, repeated."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_held)
    h_base, h_ref, h_priv = (perm[i * n_pool:(i + 1) * n_pool] for i in range(3))
    members = np.sort(rng.choice(n_train, size=n_pool, replace=False))
    return h_base, h_ref, h_priv, members


def step_p5v2(cfg) -> dict:
    """P5 of the proposed amendment v2 (docs/m7_privacy.md): the v1 attack and criterion on fresh sets disjoint from
    every v1.1 set (members: train single-window stays outside the P4 members; non-members and the private
    reference: held-out single-window stays outside H_base, H_ref and H_priv) and on the v2 audit samples
    (``out_dir/audit_samples_p5v2``, drawn with the v2 seed). Runs only once the amendment is adopted."""
    import pyarrow.parquet as pq

    if not bool(cfg.p5v2.adopted):
        raise RuntimeError("amendment v2 is not adopted (docs/m7_privacy.md): p5v2 does not run")
    tokens, v, lay, out = _ctx(cfg)
    dev = _device(cfg)
    exclude = {i for i, m in enumerate(v.meta) if m.get("cls") == "special"}
    end, death = v.id("[END]"), v.id("MEDS_DEATH")
    samples = [np.asarray(x, dtype=np.int32) for f in sorted((out / "audit_samples_p5v2").glob("shard_*.parquet"))
               for x in pq.read_table(f).column("ids").to_pylist()]
    train, held = [], []
    for ds in cfg.datasets:
        train += pv.first_windows(tokens, ds, "train")
        held += pv.first_windows(tokens, ds, "held_out")
    h_base, h_ref, h_priv, members_v1 = v1_sets(int(cfg.seed), len(held), len(train), int(cfg.synthetic.pool))
    used_held = set(h_base.tolist()) | set(h_ref.tolist()) | set(h_priv.tolist())
    used_train = set(members_v1.tolist())

    def single(w):    # a single-window stay: a timeline and an end block in its first window
        return lay.tl in w and pv.end_flag(w, death, end) is not None

    free_train = np.array([i for i, w in enumerate(train) if i not in used_train and single(w)], dtype=np.int64)
    free_held = np.array([i for i, w in enumerate(held) if i not in used_held and single(w)], dtype=np.int64)
    rng = np.random.default_rng(int(cfg.p5v2.set_seed))
    mem = np.sort(rng.choice(free_train, size=min(int(cfg.p5v2.members), len(free_train)), replace=False))
    hp = rng.permutation(free_held)
    n_non = min(int(cfg.p5v2.nonmembers), len(hp))
    non = np.sort(hp[:n_non])
    priv = np.sort(hp[n_non:n_non + int(cfg.p5v2.private_reference)])
    _x, idf = pv.tfidf(train, len(v.tokens), exclude)

    def qi(ws):
        return pv.dense_rows(pv.tfidf([pv.qi_tokens(w, lay, int(cfg.synthetic.qi_tokens)) for w in ws], len(v.tokens),
                                      exclude, idf)[0], dev)

    def labelled(ws):
        keep = [(w, pv.end_flag(w, death, end)) for w in ws if lay.tl in w]
        return [w for w, y in keep if y is not None], np.array([int(y) for _w, y in keep if y is not None])

    mem_w, mem_y = labelled([train[i] for i in mem])
    non_w, non_y = labelled([held[i] for i in non])
    gens = {"model": samples, "private_reference": [held[i] for i in priv],
            "copying_generator": [train[i] for i in mem]}
    res = {"amendment": "v2", "sizes": {"members": len(mem_w), "nonmembers": len(non_w), "private_reference": len(priv),
                                        "samples": len(samples)},
           "seeds": {"sets": int(cfg.p5v2.set_seed), "samples": int(cfg.p5v2.sample_seed)}, "p5": {}}
    for name, ws in gens.items():
        syn_w, syn_y = labelled(ws)
        if len(syn_w) < 2 or len(mem_w) < 2 or len(non_w) < 2:
            res["p5"][name] = {"status": "PENDING", "samples_complete": len(syn_w)}
            continue
        res["p5"][name] = pv.p5_attribute(qi(syn_w), syn_y, qi(mem_w), mem_y, qi(non_w), non_y,
                                          float(cfg.synthetic.attribute_limit), int(cfg.seed))
        res["p5"][name]["samples_complete"] = len(syn_w)
    _write(out, "p5v2", res)
    return res


def step_report(cfg) -> dict:
    out = Path(cfg.out_dir)
    read = lambda n: json.loads((out / f"{n}.json").read_text()) if (out / f"{n}.json").exists() else None  # noqa: E731
    p1, p2, syn = read("p1"), read("p2"), read("synthetic")
    res = {"P1": p1["status"] if p1 else "PENDING", "P2": p2["status"] if p2 else "PENDING"}
    if syn:
        for t in ("p3", "p4", "p5"):
            model, priv, copy = (syn[t][g]["status"] for g in ("model", "private_reference", "copying_generator"))
            controls_ok = priv == "PASS" and copy == "FAIL"
            res[t.upper()] = model if controls_ok else "PENDING"
            res[f"{t.upper()}_controls"] = {"private_reference": priv, "copying_generator": copy, "valid": controls_ok}
    else:
        res.update({"P3": "PENDING", "P4": "PENDING", "P5": "PENDING"})
    v2 = read("p5v2")
    if v2 is not None and bool(cfg.get("p5v2", {}).get("adopted", False)):     # amendment v2: P5 on its own sets
        model, priv, copy = (v2["p5"][g]["status"] for g in ("model", "private_reference", "copying_generator"))
        controls_ok = priv == "PASS" and copy == "FAIL"
        res["P5_v1_1"], res["P5_v1_1_controls"] = res.get("P5"), res.get("P5_controls")
        res["P5"] = model if controls_ok else "PENDING"
        res["P5_controls"] = {"private_reference": priv, "copying_generator": copy, "valid": controls_ok}
        res["amendment"] = "v2 (P5)"
    res["not_in_v1"] = {"reference_model_attacks": "PENDING", "conditional_and_counterfactual_sampling": "PENDING"}
    res["gate"] = pv.combine([res[t] for t in ("P1", "P2", "P3", "P4", "P5")])
    _write(out, "report", res)
    return res


def step_purge(cfg) -> None:
    import shutil

    for d in sorted(Path(cfg.out_dir).glob("audit_samples*")):      # v1's and amendment v2's
        shutil.rmtree(d)
        log.info("audit samples deleted: %s", d)


STEPS = {"p1": step_p1, "p2": step_p2, "sample": step_sample, "synthetic": step_synthetic, "p5v2": step_p5v2,
         "report": step_report, "purge": step_purge}


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    STEPS[cfg.step](cfg)


if __name__ == "__main__":
    main()
