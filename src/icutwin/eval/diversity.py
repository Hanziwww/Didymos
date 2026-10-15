"""S8 diversity (plan *Diversity assurance* 6): metrics of a candidate set C against a real reference set R
(held-out stays), each gated relative to the positive control (a second real set, as many stays).

Stays are token sequences, embedded as unigram TF-IDF rows (fitted on R, L2-normalized: dot products are cosine
similarities). Equal set sizes are assumed where a metric depends on the size (Vendi, distinct n-grams, rare codes,
nearest neighbours); the harness draws them so.

Twin level
* ``vendi_ratio``: Vendi score (Friedman & Dieng 2023: exp of the entropy of the eigenvalues of K / n, K the cosine
  kernel) of C over R's; ``vendi_ratio_by_dataset_min``: the same within each dataset, the smallest;
* ``rare_code_recall``: share of R's rare event codes (in 2 to 1% of R's stays) that occur in C;
* ``tail_share_diff``: |share of value events in a code's lowest or highest bin, C - R|;
* ``length_p99_logratio``: |log of the 99th percentile of events per stay, C over R|;
* ``distinct_ratio``: distinct event 3-grams per event 3-gram, C over R;
* ``duplicate_rate``: exact or near duplicates within C (``generate.bank``'s filters; real stays: about 0);
* ``nn_ratio``: median cosine distance from a C stay to its nearest other C stay, over the same within R
  (synthetic-to-synthetic against real-to-real; a collapsed generator goes to 0);
* ``copy_rate`` (with training stays): share of C closer to its nearest training stay than R's 5th percentile
  (R never trained on: about 0.05 for fresh stays, 1 for copies).
Population level
* ``hospital_tv``: total variation between C's and R's eICU hospitals (the site field of eICU stays: per-site
  heterogeneity, plan S8 *Heterogeneity*);
* ``stratum_tv``: total variation between C's and R's strata (dataset x sex x admission type);
  ``field_tv_max``: the largest total variation of one prefix or header field's values (site excluded: below);
* ``site_confusion_diff``: a dataset classifier (logistic regression on TF-IDF without the dataset and site tokens)
  trained on R; the mean total variation between R's confusion rows (cross-validated) and C's;
* ``mmd_corr``: Pearson correlation between the inter-dataset MMD matrices of R and C (linear kernel on the
  embeddings: the distance between the datasets' mean embeddings), over dataset pairs with enough stays in both.

Ratios are gated on both sides (|log ratio| against the positive control's); the rest one-sided (``flags``).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from icutwin.eval.fidelity import Vocabulary, features

DIVERGENCES = ("tail_share_diff", "length_p99_logratio", "duplicate_rate", "copy_rate", "stratum_tv",
               "field_tv_max", "site_confusion_diff", "hospital_tv")
AGREEMENTS = ("rare_code_recall", "mmd_corr")
RATIOS = ("vendi_ratio", "vendi_ratio_by_dataset_min", "distinct_ratio", "nn_ratio")
MIN_GROUP = 20
NOVELTY_Q = 0.05


class Embedding:
    """Unigram TF-IDF rows, fitted on the reference set; prefix dataset and site tokens can be left out."""

    def __init__(self, ref: list, voc: Vocabulary, drop_labels: bool = False):
        from sklearn.feature_extraction.text import TfidfTransformer

        self.n = len(voc.v.tokens)
        self.keep = np.ones(self.n, dtype=bool)
        if drop_labels:
            for i, m in enumerate(voc.v.meta):
                if m.get("cls") == "prefix" and m.get("field") in ("dataset", "site"):
                    self.keep[i] = False
        self.tf = TfidfTransformer(sublinear_tf=True).fit(self._counts(ref))

    def _counts(self, seqs: list):
        from scipy import sparse

        rows, cols, vals = [], [], []
        for r, s in enumerate(seqs):
            u, c = np.unique(np.asarray(s, dtype=np.int64), return_counts=True)
            k = self.keep[u]
            rows.append(np.full(int(k.sum()), r))
            cols.append(u[k])
            vals.append(c[k])
        return sparse.csr_matrix((np.concatenate(vals).astype(np.float32),
                                  (np.concatenate(rows), np.concatenate(cols))), shape=(len(seqs), self.n))

    def __call__(self, seqs: list):
        return self.tf.transform(self._counts(seqs)).astype(np.float32)


def vendi(x, cap: int = 3000) -> float:
    """Vendi score of L2-normalized rows (cosine kernel), from the eigenvalues of K / n (the first ``cap`` rows)."""
    x = x[:cap]
    n = x.shape[0]
    if n < 2:
        return float(n)
    k = (x @ x.T).toarray() if hasattr(x, "toarray") else x @ x.T
    lam = np.linalg.eigvalsh(np.asarray(k, dtype=np.float64) / n)
    lam = lam[lam > 1e-12]
    return float(np.exp(-np.sum(lam * np.log(lam))))


def _nn_within(x) -> np.ndarray:
    """Cosine distance from each row to its nearest other row."""
    s = (x @ x.T).toarray()
    np.fill_diagonal(s, -np.inf)
    return 1 - s.max(1)


def nn_distance(x, ref, chunk: int = 2048) -> np.ndarray:
    """Cosine distance from each row of ``x`` to its nearest row of ``ref``."""
    out = np.empty(x.shape[0])
    for i in range(0, x.shape[0], chunk):
        out[i:i + chunk] = 1 - (x[i:i + chunk] @ ref.T).toarray().max(1)
    return out


def _ngrams(f, n: int = 3) -> list:
    c = f.codes
    return [tuple(c[i:i + n]) for i in range(len(c) - n + 1)]


def _fields(ids: np.ndarray, voc: Vocabulary) -> dict:
    out = {}
    for t in np.asarray(ids)[:64]:
        m = voc.v.meta[int(t)]
        if m.get("cls") in ("prefix", "header") and m.get("field"):
            out.setdefault(f"{m['cls']}:{m['field']}", m.get("value"))
    return out


def _tv(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return 1.0
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in set(a) | set(b))


def _extremes(voc: Vocabulary) -> set:
    top = {}
    for m in voc.v.meta:
        if m.get("kind") == "bin":
            top[m["code"]] = max(top.get(m["code"], 0), int(m["bin"]))
    return {(c, 0) for c in top} | {(c, b) for c, b in top.items()}


def _confusion(labels: list, pred: np.ndarray, classes: list) -> dict:
    rows = {}
    for c in classes:
        k = np.array([y == c for y in labels])
        if k.sum() >= MIN_GROUP:
            p = Counter(pred[k].tolist())
            rows[c] = np.array([p.get(d, 0) / k.sum() for d in classes])
    return rows


def compare(real: list, cand: list, voc: Vocabulary, train: list | None = None, seed: int = 0) -> dict:
    """The diversity metrics of a candidate set against the real reference set (module docstring); ``train``: the
    model's training stays (a sample), for ``copy_rate``."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from icutwin.generate import bank

    fr, fc = [features(s, voc) for s in real], [features(s, voc) for s in cand]
    emb = Embedding(real, voc)
    xr, xc = emb(real), emb(cand)
    out = {"vendi_ratio": vendi(xc) / vendi(xr)}
    ds_r, ds_c = np.array([f.dataset for f in fr]), np.array([f.dataset for f in fc])
    by = {}
    for d in sorted(set(ds_r)):
        ir, ic = np.flatnonzero(ds_r == d), np.flatnonzero(ds_c == d)
        n = min(len(ir), len(ic))
        if n >= MIN_GROUP:
            by[d] = vendi(xc[ic[:n]]) / vendi(xr[ir[:n]])
    out["vendi_ratio_by_dataset"] = by
    out["vendi_ratio_by_dataset_min"] = min(by.values()) if by else None

    stays_with = Counter(c for f in fr for c in set(f.codes))
    rare = {c for c, n in stays_with.items() if 2 <= n <= max(2, 0.01 * len(fr))}
    seen = {c for f in fc for c in f.codes}
    out["rare_codes"] = len(rare)
    out["rare_code_recall"] = len(rare & seen) / len(rare) if rare else None
    ext = _extremes(voc)
    tail = [np.mean([p in ext for f in fs for p in f.pairs]) if any(f.pairs for f in fs) else 0.0 for fs in (fr, fc)]
    out["tail_share_diff"] = abs(float(tail[1] - tail[0]))
    p99 = [np.percentile([f.events for f in fs], 99) for fs in (fr, fc)]
    out["length_p99_logratio"] = abs(float(np.log(max(p99[1], 1) / max(p99[0], 1))))
    dist = []
    for fs in (fr, fc):
        g = [x for f in fs for x in _ngrams(f)]
        dist.append(len(set(g)) / len(g) if g else 0.0)
    out["distinct_ratio"] = dist[1] / dist[0] if dist[0] else None
    out["duplicate_rate"] = float(np.mean(bank.exact_duplicates(cand) | bank.near_duplicates(cand, voc.event)))
    nr, nc = np.median(_nn_within(xr)), np.median(_nn_within(xc))
    out["nn_ratio"] = float(nc / nr) if nr > 0 else None
    if train is not None:
        xt = emb(train)
        thr = np.quantile(nn_distance(xr, xt), NOVELTY_Q)
        out["copy_rate"] = float(np.mean(nn_distance(xc, xt) < thr))

    gr, gc = [_fields(s, voc) for s in real], [_fields(s, voc) for s in cand]
    sr = Counter((f.dataset, g.get("prefix:sex"), g.get("header:admission_type")) for f, g in zip(fr, gr))
    sc = Counter((f.dataset, g.get("prefix:sex"), g.get("header:admission_type")) for f, g in zip(fc, gc))
    out["stratum_tv"] = _tv(sr, sc)
    tvs = {k: _tv(Counter(g.get(k) for g in gr), Counter(g.get(k) for g in gc))
           for k in sorted({k for g in gr for k in g}) if k != "prefix:site"}
    out["field_tv"] = tvs
    out["field_tv_max"] = max(tvs.values()) if tvs else None
    hr = Counter(g.get("prefix:site") for f, g in zip(fr, gr) if f.dataset == "eicu")
    hc = Counter(g.get("prefix:site") for f, g in zip(fc, gc) if f.dataset == "eicu")
    out["hospital_tv"] = _tv(hr, hc) if sum(hr.values()) >= MIN_GROUP and sum(hc.values()) >= MIN_GROUP else None

    lab = Embedding(real, voc, drop_labels=True)
    lr, lc = lab(real), lab(cand)
    classes = sorted(set(ds_r))
    clf = LogisticRegression(C=1.0, max_iter=2000)
    groups = Counter(ds_r)
    if len(classes) > 1 and min(groups.values()) >= 5:
        pred_r = cross_val_predict(clf, lr, ds_r, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
        pred_c = clf.fit(lr, ds_r).predict(lc)
        cr, cc = _confusion(list(ds_r), pred_r, classes), _confusion(list(ds_c), pred_c, classes)
        both = sorted(set(cr) & set(cc))
        out["site_confusion_diff"] = (float(np.mean([0.5 * np.abs(cr[d] - cc[d]).sum() for d in both]))
                                      if both else None)
        out["site_accuracy"] = [float(np.mean(pred_r == ds_r)), float(np.mean(pred_c == ds_c))]
    means_r, means_c = {}, {}
    for d in classes:
        ir, ic = np.flatnonzero(ds_r == d), np.flatnonzero(ds_c == d)
        if len(ir) >= MIN_GROUP and len(ic) >= MIN_GROUP:
            means_r[d], means_c[d] = np.asarray(xr[ir].mean(0)).ravel(), np.asarray(xc[ic].mean(0)).ravel()
    ds = sorted(means_r)
    pairs = [(a, b) for i, a in enumerate(ds) for b in ds[i + 1:]]
    if len(pairs) >= 3:
        mr = np.array([np.sum((means_r[a] - means_r[b]) ** 2) for a, b in pairs])
        mc = np.array([np.sum((means_c[a] - means_c[b]) ** 2) for a, b in pairs])
        out["mmd_corr"] = float(np.corrcoef(mr, mc)[0, 1])
    return out


def flags(positive: dict, cand: dict, margin: float = 0.05, ratio_margin: float = 0.1) -> list:
    """The metrics on which a candidate is worse than the positive control: divergences above it by more than
    ``margin``, agreements below it by more than ``margin``, ratios farther from 1 (|log|) by more than
    ``ratio_margin``. ``copy_rate`` is held to its value for stays never trained on (``NOVELTY_Q``, by
    construction), not to the positive control: real training stays have near neighbours among other training
    stays (patients with several stays; 0.17 on the 2% subset)."""
    out = []
    positive = {**positive, "copy_rate": NOVELTY_Q}
    for k in DIVERGENCES:
        if positive.get(k) is not None and cand.get(k) is not None and cand[k] > positive[k] + margin:
            out.append(k)
    for k in AGREEMENTS:
        if positive.get(k) is not None and cand.get(k) is not None and cand[k] < positive[k] - margin:
            out.append(k)
    for k in RATIOS:
        p, c = positive.get(k), cand.get(k)
        if p is not None and c is not None and p > 0:
            if c <= 0 or abs(np.log(c)) > abs(np.log(p)) + ratio_margin:
                out.append(k)
    return out


def copying(train: list, n: int, rng: np.random.Generator) -> list:
    """The copying generator: training stays returned as they are."""
    return [np.array(train[i]) for i in rng.choice(len(train), size=n, replace=len(train) < n)]
