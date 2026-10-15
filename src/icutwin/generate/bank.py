"""The twin bank's plan and filters (plan S7 ``bank.py``, *Diversity assurance* 2 and 4).

* ``GenerationPlan``: quotas per dataset from a mix (``natural``: the train stays' shares; ``flattened``: shares to
  the power ``power`` < 1, renormalized; ``uniform``; or explicit counts), with each twin's importance weight
  p_natural / p_plan of its dataset, so the factory can restore natural prevalence.
* ``exact_duplicates``: twins whose token sequence repeats an earlier one (hash).
* ``near_duplicates``: MinHash over event n-grams with banded locality-sensitive hashing; pairs above a Jaccard
  threshold (estimated) are near-duplicates, the later one dropped.
* ``novelty``: twins closer to their nearest training stay (cosine distance of the TF-IDF, ``eval.privacy``'s
  representation) than the held-out stays' 5th percentile of their dataset are dropped: diversity must not come
  from copying.
A bank is generated only from a candidate that passed the M7 privacy gate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GenerationPlan:
    quotas: dict                 # dataset -> twins
    weights: dict                # dataset -> p_natural / p_plan

    @classmethod
    def from_mix(cls, natural: dict, total: int, mix: str = "natural", power: float = 0.5,
                 counts: dict | None = None) -> GenerationPlan:
        """``natural``: dataset -> train stays."""
        names = sorted(natural)
        p_nat = np.array([natural[d] for d in names], dtype=np.float64)
        p_nat /= p_nat.sum()
        if mix == "natural":
            p = p_nat
        elif mix == "flattened":
            p = p_nat ** power / (p_nat ** power).sum()
        elif mix == "uniform":
            p = np.full(len(names), 1 / len(names))
        elif mix == "counts":
            c = np.array([counts.get(d, 0) for d in names], dtype=np.float64)
            p = c / c.sum()
        else:
            raise ValueError(mix)
        q = np.floor(p * total).astype(int)
        q[np.argsort(-(p * total - q))[: total - q.sum()]] += 1          # largest remainders
        return cls({d: int(n) for d, n in zip(names, q)},
                   {d: float(pn / pp) if pp > 0 else 0.0 for d, pn, pp in zip(names, p_nat, p)})


def seq_hash(ids) -> str:
    return hashlib.sha1(np.ascontiguousarray(ids, dtype=np.int32).tobytes()).hexdigest()


def exact_duplicates(seqs: list) -> np.ndarray:
    """True for every sequence equal to an earlier one."""
    seen, dup = set(), np.zeros(len(seqs), dtype=bool)
    for i, s in enumerate(seqs):
        h = seq_hash(s)
        dup[i] = h in seen
        seen.add(h)
    return dup


def _shingles(ids: np.ndarray, event: np.ndarray, n: int) -> np.ndarray:
    ev = ids[event[ids]].astype(np.int64)
    if len(ev) < n:
        return ev if len(ev) else np.zeros(1, dtype=np.int64)
    k = np.zeros(len(ev) - n + 1, dtype=np.int64)
    for j in range(n):
        k = k * 1_000_003 + ev[j:len(ev) - n + 1 + j]
    return k


def near_duplicates(seqs: list, event: np.ndarray, n: int = 3, perms: int = 64, bands: int = 16,
                    threshold: float = 0.9, seed: int = 0) -> np.ndarray:
    """True for every sequence whose event n-gram set has an estimated Jaccard similarity of at least
    ``threshold`` with an earlier one (MinHash, ``perms`` permutations in ``bands`` LSH bands)."""
    rng = np.random.default_rng(seed)
    p = (1 << 61) - 1
    a = rng.integers(1, p, size=perms, dtype=np.int64)
    b = rng.integers(0, p, size=perms, dtype=np.int64)
    sig = np.empty((len(seqs), perms), dtype=np.uint64)
    for i, s in enumerate(seqs):
        sh = np.unique(_shingles(np.asarray(s), event, n)).astype(np.uint64)
        h = (sh[:, None] * a.astype(np.uint64)[None, :] + b.astype(np.uint64)[None, :]) % np.uint64(p)
        sig[i] = h.min(0)
    rows = perms // bands
    buckets: dict = {}
    dup = np.zeros(len(seqs), dtype=bool)
    for i in range(len(seqs)):
        cands = set()
        for bnd in range(bands):
            key = (bnd, sig[i, bnd * rows:(bnd + 1) * rows].tobytes())
            cands.update(buckets.get(key, ()))
            buckets.setdefault(key, []).append(i)
        dup[i] = any(np.mean(sig[i] == sig[j]) >= threshold for j in cands if j < i and not dup[j])
    return dup


class Novelty:
    """The novelty filter with its reference prepared once (a bank's shards share it): the training stays' TF-IDF
    rows and, per dataset, the held-out stays' ``q`` quantile of the distance to their nearest training stay (pooled
    where a dataset has fewer than 20 held-out stays). Called with synthetic stays and their datasets: True for every
    stay at least that far from its nearest training stay.

    ``sparse``: the reference as a sparse CSR matrix (float32 values, int32 indices) instead of dense rows (float16
    on a GPU): every training first window is about 1.5 GB instead of 16.8 GB, so it fits beside several sampling
    engines on one 32 GB GPU; the distances are the same cosine distances. ``save`` / ``load``: built once, shared by
    every engine process."""

    def __init__(self, train: list, held: list, held_ds: list, n_vocab: int, exclude: set, device, q: float = 0.05,
                 sparse: bool = False):
        from icutwin.eval import privacy as pv

        self.n_vocab, self.exclude, self.device, self.sparse = n_vocab, set(exclude), device, sparse
        x_train, self.idf = pv.tfidf(train, n_vocab, self.exclude)
        self.ref = self._csr(x_train) if sparse else pv.dense_rows(x_train, device)
        d_held = self.distances(held)
        held_ds = np.asarray(held_ds)
        self.pooled = float(np.quantile(d_held, q))
        self.thr = {d: float(np.quantile(d_held[held_ds == d], q)) if (held_ds == d).sum() >= 20 else self.pooled
                    for d in set(held_ds.tolist())}

    def _csr(self, x):
        import torch

        return torch.sparse_csr_tensor(torch.from_numpy(x.indptr.astype(np.int32)),
                                       torch.from_numpy(x.indices.astype(np.int32)),
                                       torch.from_numpy(x.data.astype(np.float32)), size=x.shape, device=self.device)

    def distances(self, windows: list, chunk: int = 64) -> np.ndarray:
        """Cosine distance of each window to its nearest training stay. ``chunk``: query rows per sparse product
        (64: the product and cuSPARSE's buffer stay near 0.25 GB beside the reference, against 1.3 GB at 512; same
        time, since the TF-IDF dominates)."""
        import torch

        from icutwin.eval import privacy as pv

        x = pv.tfidf(windows, self.n_vocab, self.exclude, self.idf)[0]
        if not self.sparse:
            return pv.nearest(pv.dense_rows(x, self.device), self.ref)[0]
        out = np.empty(x.shape[0])
        with torch.no_grad():
            for a in range(0, x.shape[0], chunk):
                q = torch.from_numpy(x[a:a + chunk].toarray()).to(self.device, torch.float32)
                sim = torch.sparse.mm(self.ref, q.T)                    # training stays x queries
                out[a:a + chunk] = 1 - sim.max(dim=0).values.cpu().numpy()
        return np.clip(out, 0, 2)

    def __call__(self, syn: list, syn_ds: list) -> np.ndarray:
        d_syn = self.distances(syn)
        return np.array([d >= self.thr.get(ds, self.pooled) for d, ds in zip(d_syn, syn_ds)])

    def save(self, path: Path) -> None:
        """The sparse reference with its IDF and thresholds (``np.savez``), for ``load`` in every engine process."""
        import json

        if not self.sparse:
            raise ValueError("only a sparse reference is saved")
        import os

        r = self.ref.to("cpu")
        path = Path(path)
        tmp = path.with_name(path.stem + ".partial.npz")    # renamed when complete: readers never see half a file
        np.savez(tmp, indptr=r.crow_indices().numpy(), indices=r.col_indices().numpy(), data=r.values().numpy(),
                 shape=np.asarray(r.shape), idf=self.idf, exclude=np.asarray(sorted(self.exclude), dtype=np.int64),
                 meta=np.asarray(json.dumps({"thr": self.thr, "pooled": self.pooled, "n_vocab": self.n_vocab})))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path, device) -> Novelty:
        import json

        import torch

        z = np.load(path)
        self = cls.__new__(cls)
        meta = json.loads(str(z["meta"]))
        self.n_vocab, self.exclude, self.device, self.sparse = meta["n_vocab"], set(z["exclude"].tolist()), device, True
        self.idf, self.thr, self.pooled = z["idf"], meta["thr"], meta["pooled"]
        self.ref = torch.sparse_csr_tensor(torch.from_numpy(z["indptr"]), torch.from_numpy(z["indices"]),
                                           torch.from_numpy(z["data"]), size=tuple(int(s) for s in z["shape"]),
                                           device=device)
        return self


def novelty(syn: list, syn_ds: list, train: list, held: list, held_ds: list, n_vocab: int, exclude: set,
            device, q: float = 0.05) -> np.ndarray:
    """True for every synthetic stay farther from its nearest training stay than the held-out stays' ``q``
    quantile of their dataset (pooled where a dataset has no held-out stays); ``Novelty`` prepared for one call."""
    return Novelty(train, held, held_ds, n_vocab, exclude, device, q)(syn, syn_ds)
