"""icutwin on the AutoDL host: where the membership signal (P1) comes from: per token family, the member and
non-member mean NLL and the membership AUC of a stay's mean NLL over that family alone (aggregates only).
Members: random training stays; non-members: random held-out stays (first windows, the P1 targets).
    python p1_families.py <run> <tokens root> <out.json> [--datasets mimic ...] [--n 1000]"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from icutwin.eval import privacy as pv
from icutwin.model.data import IGNORE, labels_for, layout
from icutwin.tokenization.vocab import Vocab


def families(v: Vocab) -> tuple[np.ndarray, list]:
    names = ["prefix_header", "timeline_event", "interval", "end_outcome", "end_diagnosis", "other"]
    fam = np.full(len(v.tokens), names.index("other"), dtype=np.int64)
    for i, m in enumerate(v.meta):
        cls, code = m.get("cls"), (m.get("code") or v.tokens[i])
        if cls in ("prefix", "header"):
            fam[i] = 0
        elif cls == "event":
            fam[i] = 1
        elif cls == "interval":
            fam[i] = 2
        elif cls == "end":
            fam[i] = 4 if code.startswith(("HOSP_DX", "PT_DX", "PROC")) else 3
        elif cls == "chunk":
            fam[i] = -1                          # a code's sub-token: the family of the code it continues
    return fam, names


def position_families(w: np.ndarray, fam: np.ndarray) -> np.ndarray:
    f = fam[w].copy()
    for j in range(len(f)):
        if f[j] < 0:
            f[j] = f[j - 1] if j else 5
    return f


def per_stay(model, windows, lay, fam, k, device) -> np.ndarray:
    """(stays, families, 2): summed NLL and target tokens per family."""
    out = np.zeros((len(windows), k, 2))
    with torch.no_grad():
        for j, w in enumerate(windows):
            ids = torch.from_numpy(w.astype(np.int64))[None].to(device)
            lab = torch.from_numpy(labels_for(w, lay))[None].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(input_ids=ids).logits
            tgt = lab[:, 1:]
            nll = torch.nn.functional.cross_entropy(logits[:, :-1].float().transpose(1, 2), tgt,
                                                    ignore_index=IGNORE, reduction="none")[0]
            keep = (tgt[0] != IGNORE)
            f = torch.from_numpy(position_families(w, fam)[1:]).to(device)
            for c in range(k):
                sel = keep & (f == c)
                out[j, c] = (float(nll[sel].double().sum()), float(sel.sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("tokens")
    ap.add_argument("out")
    ap.add_argument("--datasets", nargs="+", default=["mimic"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ckpt", default="final", help="a checkpoint directory of the run (step_1500, ...)")
    a = ap.parse_args()
    tokens = Path(a.tokens)
    v = Vocab.load(tokens / "vocab.json")
    lay = layout(v)
    fam, names = families(v)
    device = torch.device("cuda")
    model = pv.load_model(Path(f"/root/autodl-tmp/icutwin/runs/train/full/{a.run}/{a.ckpt}/hf"), device)
    rng = np.random.default_rng(0)
    report = {}
    for ds in a.datasets:
        sets = {}
        for split in ("train", "held_out"):
            w = pv.first_windows(tokens, ds, split)
            pick = rng.choice(len(w), size=min(a.n, len(w)), replace=False)
            sets[split] = per_stay(model, [w[i] for i in pick], lay, fam, len(names), device)
        mem, non = sets["train"], sets["held_out"]
        row = {}
        tot_m = mem[:, :, 0].sum(1) / np.maximum(mem[:, :, 1].sum(1), 1)
        tot_n = non[:, :, 0].sum(1) / np.maximum(non[:, :, 1].sum(1), 1)
        row["all"] = {"auc": pv._auc(-tot_m, -tot_n), "nll_members": float(tot_m.mean()),
                      "nll_nonmembers": float(tot_n.mean())}
        for c, name in enumerate(names):
            ok_m, ok_n = mem[:, c, 1] > 0, non[:, c, 1] > 0
            if ok_m.sum() < 20 or ok_n.sum() < 20:
                continue
            sm = mem[ok_m, c, 0] / mem[ok_m, c, 1]
            sn = non[ok_n, c, 0] / non[ok_n, c, 1]
            # the family's part of the per-stay mean NLL gap: its summed NLL difference over all target tokens
            share = (non[:, c, 0].sum() / non[:, :, 1].sum() - mem[:, c, 0].sum() / mem[:, :, 1].sum())
            row[name] = {"auc": pv._auc(-sm, -sn), "nll_members": float(sm.mean()), "nll_nonmembers": float(sn.mean()),
                         "tokens_per_stay": float(mem[:, c, 1].mean()), "gap_part": float(share)}
        report[ds] = row
        print(ds, json.dumps({k: {x: round(y, 4) for x, y in r.items()} for k, r in row.items()}), flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
