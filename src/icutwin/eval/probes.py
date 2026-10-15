"""Linear probes of the model's representations (plan S8 *Representations*, the tokenization benchmark's protocol):
held-out stays as known 24 hours after ICU admission (``eval.utility.landmark``: every token before it), embedded by
the frozen model (the last layer's hidden state at the window's last token, and its mean over the window), and a
logistic regression per outcome (in-hospital death, ICU stay over 7 days, new IMV, vasopressor or RRT after the
landmark), cross-validated AUROC; the baseline is the same probe on the window's TF-IDF token counts.

    .venv/bin/python -m icutwin.eval.probes --model <hf dir> --tokens <tokens root> --out <probes.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from icutwin.eval import utility as ut
from icutwin.eval.forecast import Reader

DATASETS = ("mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire")


def embed(model, windows: list, device, batch: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """(last-token, mean) hidden states of the last layer for each window."""
    import torch

    last, mean = [], []
    order = np.argsort([len(w) for w in windows])
    out_last = [None] * len(windows)
    out_mean = [None] * len(windows)
    with torch.no_grad():
        for a in range(0, len(order), batch):
            idx = order[a:a + batch]
            n = max(len(windows[i]) for i in idx)
            ids = torch.zeros((len(idx), n), dtype=torch.long)
            att = torch.zeros((len(idx), n), dtype=torch.long)
            for r, i in enumerate(idx):
                w = torch.as_tensor(np.asarray(windows[i], dtype=np.int64))
                ids[r, :len(w)], att[r, :len(w)] = w, 1
            h = model(input_ids=ids.to(device), attention_mask=att.to(device), output_hidden_states=True)
            hs = h.hidden_states[-1].float()
            for r, i in enumerate(idx):
                k = int(att[r].sum())
                out_last[i] = hs[r, k - 1].cpu().numpy()
                out_mean[i] = hs[r, :k].mean(0).cpu().numpy()
    last, mean = np.stack(out_last), np.stack(out_mean)
    return last, mean


def probe_auroc(x, y: np.ndarray, seed: int = 0, min_pos: int = 10) -> float | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    k = ~np.isnan(y)
    yk = y[k]
    if min(yk.sum(), (1 - yk).sum()) < min_pos:
        return None
    xk = x[k]
    steps = [LogisticRegression(C=1.0, max_iter=3000)]
    if isinstance(xk, np.ndarray):
        steps.insert(0, StandardScaler())
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(make_pipeline(*steps), xk, yk, cv=cv, scoring="roc_auc")))


def evaluate(model, windows: list, labels: dict, n_vocab: int, device, seed: int = 0) -> dict:
    from sklearn.feature_extraction.text import TfidfTransformer

    last, mean = embed(model, windows, device)
    counts = ut._counts(windows, n_vocab)
    tfidf = TfidfTransformer(sublinear_tf=True).fit_transform(counts)
    out = {}
    for task, y in labels.items():
        out[task] = {"stays": int(np.sum(~np.isnan(y))), "prevalence": float(np.nanmean(y)) if len(y) else None,
                     "probe_last": probe_auroc(last, y, seed), "probe_mean": probe_auroc(mean, y, seed),
                     "tfidf_baseline": probe_auroc(tfidf, y, seed)}
    return out


def main(argv: list[str] | None = None) -> None:
    import torch

    from icutwin.eval import privacy as pv
    from icutwin.tokenization.vocab import Vocab

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-dataset", type=int, default=600)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    a = ap.parse_args(argv)
    tokens = Path(a.tokens)
    v = Vocab.load(tokens / "vocab.json")
    r = Reader(v)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pv.load_model(Path(a.model), dev)
    rng = np.random.default_rng(a.seed)
    report, pooled_w, pooled_l = {}, [], {k: [] for k in ut.TASKS}
    for ds in a.datasets:
        held = pv.first_windows(tokens, ds, "held_out")
        held = [held[i] for i in rng.choice(len(held), size=min(len(held), 3 * a.per_dataset), replace=False)]
        w, lab = ut.examples(held, r, a.hours)
        w, lab = w[:a.per_dataset], {k: y[:a.per_dataset] for k, y in lab.items()}
        report[ds] = evaluate(model, w, lab, len(v.tokens), dev, a.seed)
        pooled_w += w
        for k in ut.TASKS:
            pooled_l[k].append(lab[k])
        print(ds, json.dumps({t: [x["probe_last"], x["probe_mean"], x["tfidf_baseline"]] for t, x in report[ds].items()},
                             default=float), flush=True)
    report["pooled"] = evaluate(model, pooled_w, {k: np.concatenate(y) for k, y in pooled_l.items()}, len(v.tokens),
                                dev, a.seed)
    Path(a.out).write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
