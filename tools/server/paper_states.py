"""icutwin on the AutoDL host: Paper P's model states at real anchors (``docs/p1_paper_protocol.md`` A1's Twin probe,
B3, C2's target, C4). Anchors of a design on a split's stays among a store's (``eval.bench.design_anchors``; the
same seed and sizes give A1's anchors), or the anchors of an existing design directory (``--anchors-from``); each
anchor's prompt (the stay as known then, on its own clock) through a checkpoint's frozen weights: the last layer's
hidden state at the prompt's last token and its mean over the prompt. Writes ``<out>/anchors.parquet`` (with each
prompt's prediction time) and ``<out>/states.npz`` (``last``, ``mean`` float16 [n, d], ``anchor``). Needs the GPU to
itself (never beside vLLM's engines). Amendment 7 (F7): ``--random-init`` replaces the weights by a random
initialization of the same architecture (a negative control).
    python paper_states.py --run MUC32_main --tokens <root> --split train --store <real_train> --design L24 \\
        --per-dataset 1500 --out <dir>"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

R = Path("/root/autodl-tmp/icutwin")


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("run", "tokens", "split", "store", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--design", default=None)
    ap.add_argument("--datasets", nargs="*", default=None, help="the real stays' databases (default: the seven)")
    ap.add_argument("--anchors-from", default=None)
    ap.add_argument("--per-dataset", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens-per-batch", type=int, default=65536)
    ap.add_argument("--random-init", type=int, default=None,
                    help="amendment 7 (F7): the same architecture with random weights (this seed), a negative control")
    a = ap.parse_args()
    t0 = time.time()
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM

    from icutwin.eval import bench
    from icutwin.factory import oracle, twins
    from icutwin.tokenization.vocab import Vocab

    tokens, out = Path(a.tokens), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v = Vocab.load(tokens / "vocab.json")
    rows = bench.real_rows(tokens, a.split, bench.store_stay_ids(a.store), a.datasets or bench.DATASETS)
    ids = {r["stay_id"]: r["ids"] for r in rows}
    if a.anchors_from:
        t = pq.read_table(Path(a.anchors_from) / "anchors.parquet").to_pylist()
        anchors = [oracle.Anchor(**{f: r[f] for f in oracle.Anchor.__dataclass_fields__}) for r in t]
        anchors = [x for x in anchors if x.twin in ids]
    else:
        stays = [(r["stay_id"], r["dataset"], twins.decode_labels(r["ids"], v, r["stay_id"])) for r in rows]
        anchors = bench.design_anchors(stays, v, a.design, a.per_dataset, a.seed)
    prompts = [bench.prompt(ids[x.twin], v, x.twin, x.hours) for x in anchors]
    design = {f: [getattr(x, f) for x in anchors] for f in oracle.Anchor.__dataclass_fields__}
    design["prompt_len"] = [len(p) for p in prompts]
    design["t_min"] = [bench.anchor_time(p, v, x.twin) for p, x in zip(prompts, anchors, strict=True)]
    pq.write_table(pa.table(design), out / "anchors.parquet")
    print(json.dumps({"anchors": len(anchors), "setup_s": round(time.time() - t0, 1)}), flush=True)
    hf = R / "runs/train/full" / a.run / "final/hf"
    model = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if a.random_init is not None:
        from transformers import AutoConfig

        torch.manual_seed(a.random_init)
        rnd = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(hf), torch_dtype=torch.bfloat16)
        model.load_state_dict(rnd.state_dict())
        del rnd
    model.eval()
    order = np.argsort([len(p) for p in prompts])
    last = [None] * len(prompts)
    mean = [None] * len(prompts)
    k = 0
    with torch.no_grad():
        while k < len(order):
            n = 1
            while k + n < len(order) and (n + 1) * len(prompts[order[k + n]]) <= a.tokens_per_batch:
                n += 1
            idx = order[k:k + n]
            width = max(len(prompts[i]) for i in idx)
            x = torch.zeros((len(idx), width), dtype=torch.long)
            att = torch.zeros((len(idx), width), dtype=torch.long)
            for r, i in enumerate(idx):
                x[r, :len(prompts[i])] = torch.as_tensor(prompts[i])
                att[r, :len(prompts[i])] = 1
            h = model(input_ids=x.cuda(), attention_mask=att.cuda(), output_hidden_states=True).hidden_states[-1]
            for r, i in enumerate(idx):
                m = int(att[r].sum())
                last[i] = h[r, m - 1].float().cpu().numpy().astype(np.float16)
                mean[i] = h[r, :m].float().mean(0).cpu().numpy().astype(np.float16)
            k += n
    np.savez_compressed(out / "states.npz", last=np.stack(last), mean=np.stack(mean),
                        anchor=np.array([x.anchor for x in anchors]))
    (out / "states.json").write_text(json.dumps({**vars(a), "anchors": len(anchors),
                                                 "seconds": round(time.time() - t0, 1)}, indent=1))
    print(json.dumps({"done": len(anchors), "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
