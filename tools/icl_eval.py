"""TrajPFN's learning confirmation checks L1-L6 (``icutwin.icl.eval``) for a run's checkpoint, on the frozen sets
(``tools/icl_train.py make-set``). Writes ``<run>/checks[_ema].json`` (aggregates only).

    python tools/icl_eval.py --run-dir runs/icl/M_pilot --ema
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sets", default="data/icl/sets")
    ap.add_argument("--ema", action="store_true", help="evaluate the EMA weights")
    ap.add_argument("--limit", type=int, default=None, help="episodes per set (default: all)")
    ap.add_argument("--max-rows", type=int, default=64, help="train rows per context patient, as in training")
    ap.add_argument("--checks", nargs="+", default=["L1", "L2", "L3", "L4", "L5", "L6"])
    a = ap.parse_args(argv)
    import torch

    from icutwin.icl import batch as B
    from icutwin.icl import eval as E
    from icutwin.icl import model as M
    from icutwin.icl import train as T

    run = Path(a.run_dir)
    conf = json.loads((run / "config.json").read_text(encoding="utf-8"))
    mcfg = M.Config(**conf["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.8)
    model = M.TrajPFN(mcfg).to(device)
    state = torch.load(run / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if a.ema and state.get("ema"):
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(state["ema"]["shadow"][n].to(p.dtype))
    names, n_static = T.channel_table(conf["train"]["store"], conf["train"]["vocab"])
    builder = B.Builder(names, n_static, train=False, max_rows=a.max_rows, d_id=mcfg.d_id, seed=conf["train"]["seed"])

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    def load(name):
        eps = T.load_set(str(Path(a.sets) / name))
        return eps[:a.limit] if a.limit else eps

    t0 = time.time()
    rep: dict = {"run": str(run), "step": state["step"], "ema": a.ema, "max_rows": a.max_rows}
    val = load("twin_val")
    val_rows = E.episode_results(model, builder, val, device, autocast)
    if "L1" in a.checks:
        rep["L1"] = E.check_l1(val_rows)
    if "L2" in a.checks:
        rep["L2"] = E.check_l2(model, builder, val, device, autocast)
    if "L3" in a.checks:
        rep["L3"] = E.check_l3(model, builder, val, device, autocast)
    if "L4" in a.checks:
        rep["L4"] = E.check_l4(val_rows)
    if "L5" in a.checks:
        real_rows = E.episode_results(model, builder, load("real_dev"), device, autocast)
        rep["L5"] = E.check_l5(real_rows, val_rows)
    if "L6" in a.checks:
        rep["L6"] = E.check_l6(E.episode_results(model, builder, load("twin_train"), device, autocast), val_rows)
    import numpy as np

    def save(name, rows):                             # per-row predictions for the paired baseline comparison
        keep = [r for r in rows if "y" in r and ("p" in r or "yhat" in r)]
        if keep:
            np.savez_compressed(run / f"preds_{name}{'_ema' if a.ema else ''}.npz",
                                episode=np.concatenate([np.full(len(r["y"]), r["index"]) for r in keep]),
                                label=np.concatenate([np.full(len(r["y"]), r["label"]) for r in keep]),
                                y=np.concatenate([r["y"] for r in keep]),
                                pred=np.concatenate([r.get("p", r.get("yhat")) for r in keep]))

    save("twin_val", val_rows)
    if "L5" in a.checks:
        save("real_dev", real_rows)
    rep["seconds"] = round(time.time() - t0, 1)
    out = run / ("checks_ema.json" if a.ema else "checks.json")
    out.write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in rep.items()}))


if __name__ == "__main__":
    main()
