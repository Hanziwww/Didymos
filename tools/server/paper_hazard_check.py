"""icutwin on the AutoDL host: Paper P amendment 7, a check of the hazard extraction (``paper_hazards.py``) against
the sampler's exact probabilities. For a sample of rollouts per dataset, at a share of their timeline positions: the
probability of each cause's tokens under the generation mask (grammar and the dataset's support, renormalized over the
allowed tokens: ``eval.calibration.masked_probs``, as the sampler draws) against ``eval.hazards``' extraction (the
model's raw probability, zero where the group order forbids). Prints per dataset and cause the ratio of the summed
exact to the summed extracted hazards (the factor the self-check's shortfall should equal). Diagnostic only.
    python paper_hazard_check.py --run MUC32_main --tokens <root> --design-dir <dir> [--per-dataset 40 --share 0.2]"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/icutwin")


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("run", "tokens", "design-dir"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--per-dataset", type=int, default=40)
    ap.add_argument("--share", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    import pandas as pd
    import pyarrow.parquet as pq
    import torch

    from icutwin.eval import hazards as hz
    from icutwin.eval import privacy as pv
    from icutwin.eval.calibration import masked_probs
    from icutwin.generate.support import Support
    from icutwin.tokenization.fsm import compile_tables, mask_tables
    from icutwin.tokenization.vocab import Vocab

    tokens, dd = Path(a.tokens), Path(a.design_dir)
    v = Vocab.load(tokens / "vocab.json")
    g = hz.grammar(v)
    tables = compile_tables(v)
    masks = mask_tables(tables, len(v.tokens))
    support = Support.load(v, tokens / "support.json") if (tokens / "support.json").exists() else None
    dev = torch.device("cuda")
    model = pv.load_model(R / "runs/train/full" / a.run / "final/hf", dev)
    an = pd.read_parquet(dd / "anchors.parquet").drop_duplicates("anchor").set_index("anchor")
    rng = np.random.default_rng(a.seed)
    import pyarrow.compute as pc

    rows = []
    for f in sorted((dd / "rollouts").rglob("*.parquet")):
        t = pq.read_table(f, columns=["anchor", "rollout", "prompt", "continuation"])
        rows += [r for r in t.filter(pc.less(t["rollout"], 2)).to_pylist()
                 if len(r["prompt"]) + len(r["continuation"]) <= 6000]      # the loader's attention is quadratic
    by: dict = {}
    for r in rows:
        by.setdefault(an.loc[r["anchor"], "dataset"], []).append(r)
    index = {c: torch.as_tensor(g.sets[c], device=dev) for c in hz.CAUSES if len(g.sets[c])}
    nz = hz.normalizer(v)
    from icutwin.model.data import layout

    lay = layout(v)
    out = {}
    for ds, rs in sorted(by.items()):
        pick = [rs[i] for i in rng.permutation(len(rs))[:a.per_dataset]]
        tot = {c: [0.0, 0.0, 0.0] for c in index}
        for r in pick:
            p, c = list(r["prompt"]), list(r["continuation"])
            s = np.asarray(p + c, dtype=np.int64)
            ends = np.flatnonzero(np.asarray(c) == g.end)
            last = len(p) + (int(ends[0]) if len(ends) else len(c)) - 1
            pos = np.arange(len(p) - 1, last)
            pos = pos[~g.clock[s[pos + 1]]]
            pos = pos[np.random.default_rng(len(s)).random(len(pos)) < a.share]
            if not len(pos):
                continue
            exact = masked_probs(model, s.tolist(), index, tables, masks, support, dev, pos)
            with torch.no_grad():
                lg = model(input_ids=torch.as_tensor(s[None], device=dev)).logits[0].float()
                lse = torch.logsumexp(lg, -1)
            gf = hz.group_first(s, g)
            sm = support.mask(int(s[lay.dataset_pos])) if support is not None else None
            sup = torch.as_tensor(sm if sm is not None else np.ones(len(v.tokens), bool), device=dev).float()
            pt = torch.as_tensor(pos, device=dev)
            with torch.no_grad():
                parts = hz.masked_parts(lg[pt], nz, index, sup, torch.as_tensor(gf[pos], device=dev))
            parts = {k: x.cpu().numpy() for k, x in parts.items()}
            norm = hz.normalized(parts, s[pos], nz, list(index), hz.running_state(s, pos, nz))
            for cz, ix in index.items():
                raw = torch.exp(lg[pt][:, ix] - lse[pt, None])
                h = hz.set_hazard(raw.cpu().numpy(), g.sets[cz], gf[pos])
                e = exact[cz][pos]
                ok = np.isfinite(e)
                tot[cz][0] += float(np.sum(e[ok]))
                tot[cz][1] += float(np.sum(h[ok]))
                tot[cz][2] += float(np.sum(norm[cz][ok]))
        out[ds] = {cz: {"exact": x[0], "extracted": x[1], "normalized": x[2],
                        "ratio": x[0] / x[1] if x[1] else None, "ratio_normalized": x[0] / x[2] if x[2] else None}
                   for cz, x in tot.items()}
        print(json.dumps({ds: {cz: (round(y["ratio"], 3) if y["ratio"] else None,
                                    round(y["ratio_normalized"], 3) if y["ratio_normalized"] else None)
                               for cz, y in out[ds].items()}}), flush=True)
    (dd / "hazard_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    random.seed(0)
    main()
