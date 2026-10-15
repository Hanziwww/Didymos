"""icutwin on the AutoDL host: Paper P amendment 7, a debug of the death hazard's extraction in one dataset: the
timeline positions where the sampler's exact masked probability of the ICU discharge 'dead' exceeds the extraction's
most, with the automaton's state and the tokens before them. Diagnostic only (prints codes, no stay data).
    python paper_hazard_debug.py --run MUC32_main --tokens <root> --design-dir <dir> --dataset aumc"""
import argparse
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/icutwin")


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("run", "tokens", "design-dir", "dataset"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--rollouts", type=int, default=12)
    a = ap.parse_args()
    import pandas as pd
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import torch

    from icutwin.eval import hazards as hz
    from icutwin.eval import privacy as pv
    from icutwin.eval.calibration import masked_probs
    from icutwin.generate.support import Support
    from icutwin.model.data import layout
    from icutwin.tokenization.fsm import Automaton, compile_tables, mask_tables
    from icutwin.tokenization.vocab import Vocab

    tokens, dd = Path(a.tokens), Path(a.design_dir)
    v = Vocab.load(tokens / "vocab.json")
    g, nz, lay = hz.grammar(v), hz.normalizer(v), layout(v)
    tables = compile_tables(v)
    masks = mask_tables(tables, len(v.tokens))
    support = Support.load(v, tokens / "support.json")
    dev = torch.device("cuda")
    model = pv.load_model(R / "runs/train/full" / a.run / "final/hf", dev)
    an = pd.read_parquet(dd / "anchors.parquet").drop_duplicates("anchor").set_index("anchor")
    dead = g.sets["death"]
    print("dead tokens", [(int(t), v.tokens[int(t)], v.meta[int(t)].get("cls"), v.meta[int(t)].get("kind"),
                           int(t) in tables.event_first) for t in dead])
    sm = None
    rows = []
    for f in sorted((dd / "rollouts").rglob("*.parquet")):
        t = pq.read_table(f, columns=["anchor", "rollout", "prompt", "continuation"])
        t = t.filter(pc.less(t["rollout"], 2))
        for r in t.to_pylist():
            if an.loc[r["anchor"], "dataset"] == a.dataset and len(r["prompt"]) + len(r["continuation"]) <= 6000:
                rows.append(r)
        if len(rows) >= a.rollouts:
            break
    idx = {"death": torch.as_tensor(dead, device=dev)}
    worst = []
    for r in rows[:a.rollouts]:
        p, c = list(r["prompt"]), list(r["continuation"])
        s = np.asarray(p + c, dtype=np.int64)
        ends = np.flatnonzero(np.asarray(c) == g.end)
        last = len(p) + (int(ends[0]) if len(ends) else len(c)) - 1
        pos = np.arange(len(p) - 1, last)
        exact = masked_probs(model, s.tolist(), idx, tables, masks, support, dev, pos)["death"]
        with torch.no_grad():
            lg = model(input_ids=torch.as_tensor(s[None], device=dev)).logits[0].float()
        sm = support.mask(int(s[lay.dataset_pos]))
        sup = torch.as_tensor(sm, device=dev).float()
        gf = hz.group_first(s, g)
        parts = hz.masked_parts(lg[torch.as_tensor(pos, device=dev)], nz, {"death": idx["death"]}, sup,
                                torch.as_tensor(gf[pos], device=dev))
        parts = {k: x.cpu().numpy() for k, x in parts.items()}
        mine = hz.normalized(parts, s[pos], nz, ["death"], hz.running_state(s, pos, nz))["death"]
        e = exact[pos]
        ok = np.isfinite(e)
        print("rollout", r["anchor"][:12], "exact sum", float(np.nansum(e)), "mine", float(mine[ok].sum()))
        for j in np.flatnonzero(ok)[np.argsort(-(e[ok] - mine[ok]))[:3]]:
            q = int(pos[j])
            au = Automaton(tables)
            for tok in s[:q + 1]:
                au.step(int(tok))
            worst.append((float(e[j] - mine[j]), float(e[j]), float(mine[j]), q, au.phase, au.last, au.group_first,
                          au.hier, [v.tokens[int(x)] for x in s[max(0, q - 3):q + 1]], int(gf[q])))
    worst.sort(reverse=True)
    for w in worst[:15]:
        print(w)
    print("dead supported in", a.dataset, [bool(sm[int(t)]) for t in dead] if sm is not None else None)


if __name__ == "__main__":
    main()
