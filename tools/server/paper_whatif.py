"""icutwin on the AutoDL host: Paper P's what-if rollouts (``docs/p1_paper_protocol.md`` C1, amendment 2). For each
treatment, held-out stays whose first start of it in the ICU comes at least 2.5 hours after admission, with none of
its records before; each is cut at a time drawn uniformly 0.5-6 hours before that start (at least 2 hours after
admission): the treatment was plausible then. Arms (``generate.engine.Intervention``): *give* the start record at t
(the dataset's most common start code among these stays, at the median rate bin; for norepinephrine and insulin also
the low and high bins) against *withhold* every record of the treatment throughout. ``n`` rollouts per arm, capped at
the split's 24-hour token quantile. Outputs ``<out>/anchors.parquet``, ``<out>/arms.json`` and
``<out>/rollouts/<treatment>_<arm>_<k>.parquet`` (anchor, rollout, prompt, continuation; written atomically,
resumable). Runs in vLLM's environment.
    python paper_whatif.py --run MUC32_main --tokens <root> --split held_out --store <real_test> --out <dir>"""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/icutwin")
TREATMENTS = {   # name: (start-code regex, family-code regex or a treatment set of the engine, dose arms)
    "norepinephrine": (r"^MED_CON//norepinephrine//.+//start//iv$", r"^MED_CON//norepinephrine//", True),
    "insulin": (r"^MED_CON//insulin//.+//start//iv$", r"^MED_CON//insulin//", True),
    "furosemide": (r"^MED_(CON//furosemide//.+//start//iv|INT//furosemide//.+//given//iv)$",
                   r"^MED_(CON|INT)//furosemide//", False),
    "crrt": (r"^CRRT//crrt_mode_category//", "rrt", False),
    "imv": (r"^RESP//device_category//imv$", "imv", False),
    "fluid_bolus": (r"^MED_INT//(lactated_ringers_solution|sodium_chloride)//ml//given//iv$",
                    r"^MED_INT//(lactated_ringers_solution|sodium_chloride)//", False),
}
BINS = {"low": 1, "median": 4, "high": 8}


def _atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("run", "tokens", "split", "store", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--per-treatment", type=int, default=200)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--treatments", nargs="*", default=list(TREATMENTS))
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    import pyarrow as pa
    import pyarrow.parquet as pq

    from icutwin.eval import bench
    from icutwin.factory import oracle, twins
    from icutwin.generate import producer
    from icutwin.generate.engine import Intervention, TwinEngine, treatment_tokens
    from icutwin.tokenization.encode import prospective
    from icutwin.tokenization.vocab import Vocab

    tokens, out = Path(a.tokens), Path(a.out)
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    hf = R / "runs/train/full" / a.run / "final/hf"
    gate = producer.require_gate(R / "runs/audit" / f"m7_{a.run}", hf)
    v = Vocab.load(tokens / "vocab.json")
    code_of = [m.get("code") or "" for m in v.meta]
    rows = bench.real_rows(tokens, a.split, bench.store_stay_ids(a.store))
    rng = np.random.default_rng(a.seed)
    anchors, arms = [], {}
    stays = {r["stay_id"]: (r, twins.decode_labels(r["ids"], v, r["stay_id"])) for r in rows}
    for name in a.treatments:
        start_re, fam, dose = TREATMENTS[name]
        start_c = re.compile(start_re)
        fam_ids = treatment_tokens(v, fam) if fam in ("rrt", "imv") else \
            frozenset(i for i, c in enumerate(code_of) if re.match(fam, c))
        fam_codes = {code_of[i] for i in fam_ids}
        elig = []
        for sid, (r, s) in stays.items():
            for t, _rk, code, _n, _x in s.timeline:        # the treatment's first record of any kind
                if code in fam_codes or start_c.match(code):
                    h = (t - s.icu_in).total_seconds() / 3600
                    if start_c.match(code) and h >= 2.5:   # a new start in the ICU, none of its records before
                        elig.append((sid, r["dataset"], h, code))
                    break
        pick = [elig[k] for k in rng.permutation(len(elig))[:a.per_treatment]]
        modal: dict = {}
        for _sid, ds, _h, code in elig:
            modal.setdefault(ds, []).append(code)
        modal = {ds: max(set(c), key=c.count) for ds, c in modal.items()}
        arm_tokens: dict = {}
        for ds, code in modal.items():
            cand = [i for i, c in enumerate(code_of) if c == code]
            with_bin = {v.meta[i].get("bin"): i for i in cand if v.meta[i].get("kind") == "bin"}
            plain = [i for i in cand if v.meta[i].get("kind") == "na"] or cand
            if with_bin:                                 # dose arms only with three distinct rate bins
                distinct = len({with_bin.get(b) for b in BINS.values()} - {None}) == len(BINS)
                levels = BINS if dose and distinct else {"median": BINS["median"]}
                arm_tokens[ds] = {f"give_{k}": int(with_bin.get(b, next(iter(with_bin.values()))))
                                  for k, b in levels.items()}
            else:
                arm_tokens[ds] = {"give_median": int(plain[0])}
        arms[name] = {"eligible": len(elig), "picked": len(pick), "modal_code": modal,
                      "tokens": {ds: {k: v.meta[i].get("code") + f" bin {v.meta[i].get('bin')}" for k, i in x.items()}
                                 for ds, x in arm_tokens.items()}, "withhold_tokens": len(fam_ids)}
        for sid, ds, h, code in pick:
            t_h = max(2.0, h - rng.uniform(0.5, 6.0))
            r, s = stays[sid]
            p = prospective(s, v, t_h)
            anchors.append({"anchor": f"{name}:{sid}@{t_h:.3f}", "treatment": name, "twin": sid, "dataset": ds,
                            "hours": t_h, "start_hours": h, "prompt": p, "t_min": bench.anchor_time(p, v, sid),
                            "give": arm_tokens[ds], "withhold": sorted(fam_ids)})
    pq.write_table(pa.table({k: [x[k] for x in anchors] for k in ("anchor", "treatment", "twin", "dataset", "hours",
                                                                   "start_hours", "t_min")}),
                   out / "anchors.parquet")
    seqs: dict = {}
    for r in rows:
        seqs.setdefault(r["dataset"], []).append(r["ids"])
    caps = oracle.token_caps(seqs, v, 24.0, 0.95, a.seed)
    (out / "arms.json").write_text(json.dumps({"arms": arms, "caps": caps, "gate": gate["gate"], **vars(a)},
                                              indent=1, default=str))
    print(json.dumps({"anchors": len(anchors), "arms": {k: (x["eligible"], x["picked"]) for k, x in arms.items()},
                      "setup_s": round(time.time() - t_start, 1)}), flush=True)
    if a.dry:                                       # the design and its arms only (no GPU)
        return
    eng = TwinEngine(hf, tokens / "vocab.json", gpu_memory_utilization=a.gpu_memory_utilization)
    from icutwin.generate.engine import intervene

    dropped: dict = {}                               # an intervention the grammar rejects for a prefix: not run
    for x in anchors:
        for arm_name, tok in list(x["give"].items()):
            try:
                intervene(x["prompt"], Intervention(give=(tok,)), v, eng.t)
            except ValueError:
                del x["give"][arm_name]
                dropped[f"{x['treatment']}:{arm_name}"] = dropped.get(f"{x['treatment']}:{arm_name}", 0) + 1
    (out / "dropped.json").write_text(json.dumps(dropped, indent=1))
    print(json.dumps({"interventions_rejected_by_grammar": dropped}), flush=True)
    g = 0
    for name in a.treatments:
        group = [x for x in anchors if x["treatment"] == name]
        arm_names = sorted({k for x in group for k in x["give"]}) + ["withhold"]
        for arm in arm_names:
            items = [x for x in group if arm == "withhold" or arm in x["give"]]
            for k in range(0, len(items), a.size):
                g += 1
                path = out / "rollouts" / f"{name}_{arm}_{k // a.size:04d}.parquet"
                if path.exists():
                    continue
                batch = items[k:k + a.size]
                ivs = [Intervention(withhold=frozenset(x["withhold"])) if arm == "withhold"
                       else Intervention(give=(x["give"][arm],)) for x in batch]
                t0 = time.time()
                conts = eng.simulate([x["prompt"] for x in batch], a.n, seed=a.seed * 10_000 + g,
                                     interventions=ivs, max_new=min(caps.get(x["dataset"]) or 4096 for x in batch))
                recs = [(x, j, c) for x, cs in zip(batch, conts, strict=True) for j, c in enumerate(cs)]
                tab = pa.table({"anchor": [x["anchor"] for x, _j, _c in recs], "arm": [arm] * len(recs),
                                "rollout": [j for _x, j, _c in recs],
                                "prompt": pa.array([list(map(int, x["prompt"])) for x, _j, _c in recs],
                                                   type=pa.list_(pa.int32())),
                                "continuation": pa.array([list(map(int, c)) for _x, _j, c in recs],
                                                         type=pa.list_(pa.int32()))})
                sink = pa.BufferOutputStream()
                pq.write_table(tab, sink)
                data = sink.getvalue().to_pybytes()
                _atomic(path, data)
                print(json.dumps({"batch": path.name, "rollouts": len(recs), "seconds": round(time.time() - t0, 1),
                                  "sha256": hashlib.sha256(data).hexdigest()[:16]}), flush=True)
    print(json.dumps({"done": True, "seconds": round(time.time() - t_start, 1)}), flush=True)


if __name__ == "__main__":
    main()
