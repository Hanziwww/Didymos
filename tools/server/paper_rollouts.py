"""icutwin on the AutoDL host: Paper P's rollouts on real stays (``docs/p1_paper_protocol.md`` A1, B2). Anchors on real
stays of one split (restricted to the stays of a real store, whose labels A1 reads): a landmark design (``--design
L24`` / ``L48``: ``eval.bench.landmark_anchors``) or the rolling one (``R``: ``factory.oracle.choose``, enriched with
inclusion weights). Each anchor's prompt is the stay as known then, on its own clock (``--unknown-site``: dataset
and site tokens unknown, B2); ``n`` rollouts per anchor with the accepted model (the M7 gate must be PASS), capped at
the split's ``--cap-hours`` token quantile. Outputs ``<out>/anchors.parquet`` (the design, the prompts' lengths and
prediction times) and ``<out>/rollouts/<dataset>_<k>.parquet`` with their manifest (``factory.oracle.run``: atomic,
resumable). Runs in vLLM's environment.

Amendment 7 (F1, F9a): ``--subset`` keeps the anchors listed in a parquet (anchor, weight; the weights replace the
design's), ``--cap-mult`` multiplies the token caps (at most the context), ``--rollout-seed`` seeds the rollouts apart
from the design (``--seed`` keeps A1's anchors), ``--shard i --shards n`` draws one lane's share of the anchors into
``<out>/rollouts/shard<i>`` (the design files are written once, by ``--dry``), and ``--prompt-transform swap`` sets
each prompt's dataset token to another dataset's (drawn with ``--swap-seed``) and its site token to unknown.

Amendment 17: ``--anchors-file`` takes the anchors themselves (a parquet of ``factory.oracle.Anchor``'s fields, frozen
by ``tools/paper_ext17.py split``) instead of drawing the design (the rolling design's choice depends on the decoded
events, which a new encoding changes), and ``--caps-from`` takes the token caps of another design's ``design.json``
(before ``--cap-mult``), so a re-encoded database keeps amendment 16a's anchors and caps.
    python paper_rollouts.py --run MUC32_main --tokens <root> --split held_out --store <real_test> --design L24 \\
        --out <dir> [--per-dataset 500 --n 32 --cap-hours 168]"""
import argparse
import json
import time
from pathlib import Path

R = Path("/root/autodl-tmp/icutwin")
DATASETS = ("mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire")


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("run", "tokens", "split", "store", "design", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--per-dataset", type=int, default=500)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap-hours", type=float, default=None, help="default: 168 for landmarks, 72 for R")
    ap.add_argument("--cap-q", type=float, default=0.95)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--deadline-hours", type=float, default=20.0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--unknown-site", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--subset", default=None)
    ap.add_argument("--cap-mult", type=float, default=1.0)
    ap.add_argument("--rollout-seed", type=int, default=None)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--prompt-transform", choices=("none", "unknown", "swap"), default="none")
    ap.add_argument("--swap-seed", type=int, default=7)
    ap.add_argument("--anchors-file", default=None, help="amendment 17: the anchors, not drawn")
    ap.add_argument("--caps-from", default=None, help="amendment 17: another design.json's token caps")
    ap.add_argument("--gate-exemption", default=None,
                    help="amendment 17d: the study lead's recorded exemption for an INCONCLUSIVE gate "
                         "(these weights only)")
    ap.add_argument("--dry", action="store_true", help="the design files only (no GPU)")
    a = ap.parse_args()
    t_start = time.time()

    import pyarrow as pa
    import pyarrow.parquet as pq

    from icutwin.eval import bench
    from icutwin.factory import oracle, twins
    from icutwin.generate import producer
    from icutwin.generate.engine import TwinEngine
    from icutwin.tokenization.vocab import Vocab

    tokens, out = Path(a.tokens), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hf = R / "runs/train/full" / a.run / "final/hf"
    gate = producer.require_gate_or_exemption(R / "runs/audit" / f"m7_{a.run}", hf,
                                              Path(a.gate_exemption) if a.gate_exemption else None, "paper_rollouts")
    v = Vocab.load(tokens / "vocab.json")
    keep = set()                                    # the store's stays (pyarrow: vLLM's environment has no pandas)
    for f in Path(a.store).glob("seg_*/stays.parquet"):
        keep |= {str(x) for x in pq.read_table(f, columns=["stay_id"]).column(0).to_pylist()}
    if not keep:
        raise SystemExit(f"no stays in {a.store}")
    given = None
    if a.anchors_file:                              # amendment 17: the frozen anchors
        if a.subset:
            raise SystemExit("--anchors-file carries its own weights: no --subset")
        t = pq.read_table(a.anchors_file, columns=list(oracle.Anchor.__dataclass_fields__)).to_pylist()
        given = [oracle.Anchor(**{**r, "anchor": str(r["anchor"]), "twin": str(r["twin"]), "hours": float(r["hours"]),
                                  "los_hours": float(r["los_hours"]), "band": int(r["band"]),
                                  "enriched": bool(r["enriched"]), "flag": bool(r["flag"]),
                                  "weight": float(r["weight"])}) for r in t if r["dataset"] in a.datasets]
        if len({x.anchor for x in given}) != len(given):
            raise SystemExit("--anchors-file: duplicate anchors")
        if not a.caps_from:                         # the caps' quantile reads every stay of the split
            raise SystemExit("--anchors-file needs --caps-from")
        keep &= {x.twin for x in given}
    rows = []
    for ds in a.datasets:
        for sp in a.split.split(","):           # several splits, comma-separated (amendment 16a: every split)
            t = pq.read_table(tokens / ds / f"{sp}.parquet", columns=["stay_id", "ids"]).to_pylist()
            rows += [{"stay_id": str(r["stay_id"]), "dataset": ds, "ids": r["ids"]} for r in t
                     if str(r["stay_id"]) in keep]
    stays = [] if given is not None else [(r["stay_id"], r["dataset"], twins.decode_labels(r["ids"], v, r["stay_id"]))
                                          for r in rows]
    print(json.dumps({"stays": len(rows), "decoded_s": round(time.time() - t_start, 1), "gate": gate["gate"]}),
          flush=True)
    if given is not None:
        missing = {x.twin for x in given} - {r["stay_id"] for r in rows}
        if missing:
            raise SystemExit(f"--anchors-file: {len(missing)} stays not in the tokens and store")
        anchors = given
        cap_hours = a.cap_hours or (72.0 if a.design == "R" else 168.0)
    elif a.design in ("L24", "L48"):
        anchors = bench.landmark_anchors(stays, v, float(a.design[1:]), a.per_dataset, a.seed)
        cap_hours = a.cap_hours or 168.0
    elif a.design == "R":
        anchors = oracle.choose(stays, v, a.per_dataset, a.seed, horizon=48.0)
        cap_hours = a.cap_hours or 72.0
    else:
        raise SystemExit(f"unknown design {a.design}")
    if a.subset:                                    # amendment 7: a subset of the design's anchors, its own weights
        from dataclasses import replace

        sub = pq.read_table(a.subset, columns=["anchor", "weight"]).to_pydict()
        w_of = dict(zip(sub["anchor"], sub["weight"], strict=True))
        anchors = [replace(x, weight=float(w_of[x.anchor])) for x in anchors if x.anchor in w_of]
        if len(anchors) != len(w_of):
            raise SystemExit(f"subset: {len(w_of) - len(anchors)} anchors not in the design")
    ids = {r["stay_id"]: r["ids"] for r in rows}
    prompts = [bench.prompt(ids[x.twin], v, x.twin, x.hours) for x in anchors]
    if a.unknown_site or a.prompt_transform == "unknown":
        prompts = [bench.unknown_site(p, v) for p in prompts]
    prompt_ds = [x.dataset for x in anchors]
    if a.prompt_transform == "swap":                # F9a: another dataset's token, the site unknown
        import random

        from icutwin.model.data import layout

        lay = layout(v)
        ds_tok = {m.get("value"): i for i, m in enumerate(v.meta)
                  if m.get("cls") == "prefix" and m.get("field") == "dataset" and m.get("value") in DATASETS}
        rs = random.Random(a.swap_seed)
        for k, x in enumerate(anchors):
            other = rs.choice([d for d in DATASETS if d != x.dataset and d in ds_tok])
            p = list(prompts[k])
            p[lay.dataset_pos] = ds_tok[other]
            p[lay.site_pos] = lay.site_unknown
            prompts[k], prompt_ds[k] = p, other
    seqs: dict = {}
    for r in rows:
        seqs.setdefault(r["dataset"], []).append(r["ids"])
    if a.caps_from:                                 # amendment 17: another design's caps (before --cap-mult)
        src = json.loads(Path(a.caps_from).read_text())
        caps, cap_hours = {d: src["caps"][d] for d in a.datasets}, float(src["cap_hours"])
    else:
        caps = oracle.token_caps(seqs, v, cap_hours, a.cap_q, a.seed)
    if a.cap_mult != 1.0:                           # amendment 7: longer rollouts (at most the context)
        caps = {d: (int(c * a.cap_mult) if c else c) for d, c in caps.items()}
    if not (a.shard is not None and (out / "anchors.parquet").exists()):
        design = {f: [getattr(x, f) for x in anchors] for f in oracle.Anchor.__dataclass_fields__}
        design["prompt_len"] = [len(p) for p in prompts]
        design["t_min"] = [bench.anchor_time(p, v, x.twin) for p, x in zip(prompts, anchors, strict=True)]
        design["prompt_dataset"] = prompt_ds
        pq.write_table(pa.table(design), out / "anchors.parquet")
        (out / "design.json").write_text(json.dumps({**vars(a), "cap_hours": cap_hours, "caps": caps,
                                                     "anchors": len(anchors), "gate": gate["gate"]}, indent=1))
    print(json.dumps({"anchors": len(anchors), "enriched": sum(x.enriched for x in anchors),
                      "flagged": sum(x.flag for x in anchors), "caps": caps,
                      "setup_s": round(time.time() - t_start, 1)}), flush=True)
    if a.dry:
        return
    roll_dir, seed = out / "rollouts", a.seed if a.rollout_seed is None else a.rollout_seed
    if a.shard is not None:                         # one lane's share; seeds apart from the other lanes'
        keep_i = [k for k in range(len(anchors)) if k % a.shards == a.shard]
        anchors, prompts = [anchors[k] for k in keep_i], [prompts[k] for k in keep_i]
        roll_dir, seed = roll_dir / f"shard{a.shard}", seed * 10 + a.shard
    t0 = time.time()
    eng = TwinEngine(hf, tokens / "vocab.json", gpu_memory_utilization=a.gpu_memory_utilization)
    print(json.dumps({"engine_ready_s": round(time.time() - t0, 1)}), flush=True)
    man = oracle.run(eng, anchors, prompts, a.n, roll_dir, caps, seed, a.size,
                     deadline=t_start + a.deadline_hours * 3600)
    print(json.dumps({"complete": man.get("complete"), "batches": len(man["batches"]), "stopped": man.get("stopped"),
                      "seconds": round(time.time() - t_start, 1)}), flush=True)


if __name__ == "__main__":
    main()
