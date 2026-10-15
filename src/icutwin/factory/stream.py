"""The factory over generated twins (plan S9: the cache is built identically for bank shards and for real stays; a
PyTorch-style stream of episodes from the bank).

``cache_from_sequences`` decodes token sequences (``tokenization.decode``: values and gaps at their bins' medians, or
drawn with a seed; the decoder sets each stay's ICU window from its admission marker and discharge), keeps the stays
with an ICU admission marker, and builds the cache. With a work directory (``derived_dir``) the stays get the derived
concepts of real stays' definitions (``generate.clif_out.derived``: canonical CLIF -> ``meds.derived.derive``):
treatment episodes, KDIGO, the 24-hour SOFA; without one, event onsets come from the codes the ACES predicates read
(``factory.aces.code_onsets``) and KDIGO and SOFA are unknown. Real stays' stored token sequences go through the same
function, so a real benchmark is built exactly as the bank's cache is. ``cache_from_bank`` reads a producer bank's
shards (every part's) and keeps each twin's importance weight (``weights``), so the factory can restore natural
prevalence. ``episodes`` streams episodes from a cache without end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from icutwin.factory import aces, cache, tasks


def _with_times(s):
    """The decoded stay (the decoder sets its ICU window), or None without an ICU admission marker."""
    return s if any(c == "ICU//admission" for _t, _r, c, _n, _x in s.timeline) else None


def cache_from_sequences(seqs: list, v, ids: list | None = None, hours: int = 72, concepts: list | None = None,
                         seed: int | None = None, derived_dir: Path | None = None,
                         persistence: dict | None = None) -> tuple:
    """(cache, kept): the cache of the decodable stays among ``seqs`` (those with an ICU admission marker) and their
    positions in ``seqs``; ``concepts``: the grid's codes (use the real cache's to compare like with like);
    ``derived_dir``: where the stays' derived concepts are computed (module docstring). With a ``seed`` each stay's
    gaps and values are drawn within their bins (its own generators; ``persistence``: ``decode.load_persistence``),
    and its derived concepts come from a second decode on the same clock with values at their bins' medians: drawn
    values inflate change- and ratio-based labels (KDIGO: a low draw in a wide bin makes a tripling; the subsets'
    round trip, docs/m8_factory.md), medians keep them at the real stays' level."""
    import random

    from icutwin.tokenization.decode import decode

    stays, labelled, kept = [], [], []
    for k, ids_k in enumerate(seqs):
        ids_k = [int(x) for x in ids_k]
        if seed is None:
            s = lab = decode(ids_k, v)
        else:
            s = decode(ids_k, v, random.Random(f"t{seed}:{k}"), persistence, random.Random(f"v{seed}:{k}"))
            lab = decode(ids_k, v, random.Random(f"t{seed}:{k}"), value_rng=None) if derived_dir else None
        if _with_times(s) is None:
            continue
        s.stay_id = str(ids[k]) if ids is not None else f"twin:{k}"
        if lab is not None:
            lab.stay_id = s.stay_id
        stays.append(s)
        labelled.append(lab)
        kept.append(k)
    if derived_dir is None:
        c = cache.build(stays, {}, hours=hours, concepts=concepts)
        c.onset.update(aces.code_onsets(stays))
    else:
        from icutwin.generate import clif_out

        c = cache.build(stays, clif_out.derived(labelled, Path(derived_dir)), hours=hours, concepts=concepts)
    return c, np.array(kept, dtype=np.int64)


def cache_from_bank(bank_dir: Path, v, hours: int = 72, concepts: list | None = None, seed: int | None = None,
                    derived_dir: Path | None = None, persistence: dict | None = None) -> tuple:
    """(cache, weights): every shard of a producer bank (or of its parts, ``part<k>/``), with each twin's importance
    weight."""
    import pyarrow.parquet as pq

    seqs, ids, weights = [], [], []
    for f in sorted(Path(bank_dir).rglob("*.parquet")):
        t = pq.read_table(f, columns=["twin", "ids", "weight"]).to_pylist()
        seqs += [r["ids"] for r in t]
        ids += [r["twin"] for r in t]
        weights += [r["weight"] for r in t]
    c, kept = cache_from_sequences(seqs, v, ids, hours, concepts, seed, derived_dir, persistence)
    return c, np.asarray(weights, dtype=np.float64)[kept]


def episodes(c, seed: int = 0, **prior_kw):
    """An endless stream of episodes (spec, X, y, context rows, stays) from a cache."""
    prior = tasks.ICUPrior(c, seed=seed, **prior_kw)
    while True:
        yield prior.episode()


# -- Phase F production I/O (plan S7): the trajectory-episode stream, a reference collate, frozen dumps ----------
class EpisodeStream:
    """An endless (or bounded) stream of trajectory episodes (``factory.episodes``) for a training run: episode i is a
    function of (factory version, store, pool, seed, i) alone, so a run is reproducible and resumable from any index,
    whatever the number of workers. Under a PyTorch ``DataLoader`` each worker takes the indices congruent to its id
    (``torch.utils.data.get_worker_info``); ``iter_indices`` does the same for plain multiprocessing; ``as_dataset``
    wraps the stream as an ``IterableDataset``."""

    def __init__(self, store_dir: Path, vocab_path: Path, pool_dir: Path | None = None, stage: int = 1, seed: int = 0,
                 start: int = 0, stop: int | None = None, split: str = "train"):
        self.args = (str(store_dir), str(vocab_path), str(pool_dir) if pool_dir else None, stage, seed, split)
        self.start, self.stop = start, stop
        self._sampler = None

    def sampler(self):
        if self._sampler is None:

            from icutwin.factory import episodes, pool
            from icutwin.factory import store as fst
            from icutwin.tokenization.vocab import Vocab

            store_dir, vocab_path, pool_dir, stage, seed, split = self.args
            st = fst.Store(Path(store_dir))
            progs, weights, support = pool.load(Path(pool_dir)) if pool_dir else (None, None, None)
            self._sampler = episodes.Sampler(st, Vocab.load(Path(vocab_path)), stage=stage, seed=seed, programs=progs,
                                             weights=weights, targets=pool.load_targets(), split=split,
                                             support=support)
        return self._sampler

    def iter_indices(self, worker: int = 0, workers: int = 1):
        i = self.start + worker
        while self.stop is None or i < self.stop:
            yield i
            i += workers

    def __iter__(self):
        try:
            import torch

            info = torch.utils.data.get_worker_info()
        except ImportError:
            info = None
        from icutwin.factory.episodes import NoEpisode

        w, n = (info.id, info.num_workers) if info is not None else (0, 1)
        for i in self.iter_indices(w, n):
            try:
                yield self.sampler().episode(i)
            except NoEpisode:                      # deterministic: this index never yields an episode
                self.skipped = getattr(self, "skipped", 0) + 1

    def as_dataset(self):
        import torch

        stream = self

        class _DS(torch.utils.data.IterableDataset):
            def __iter__(self):
                return iter(stream)

        return _DS()


def collate(eps: list[dict]) -> dict:
    """Episodes as one batch (the reference collate, proving the format consumable; the architecture round may pad or
    bucket instead): each part's flat arrays concatenated, offsets shifted by the rows of the episodes before them,
    with per-episode offsets of patients (``<part>_patients``, B + 1); tasks, recipes and channel tables as lists."""
    out: dict = {"task": [e["task"] for e in eps], "recipe": [e["recipe"] for e in eps],
                 "channels": [e["channels"] for e in eps]}
    for part in ("context", "query"):
        keys = [k for k, x in eps[0][part].items() if isinstance(x, np.ndarray)]
        npat = [len(e[part]["ev_off"]) - 1 for e in eps]
        out[f"{part}_patients"] = np.concatenate([[0], np.cumsum(npat)]).astype(np.int64)
        for k in keys:
            if k.endswith("_off"):
                base, parts = 0, []
                for j, e in enumerate(eps):
                    o = np.asarray(e[part][k], dtype=np.int64)
                    parts.append((o if j == len(eps) - 1 else o[:-1]) + base)
                    base += int(o[-1])
                out[f"{part}.{k}"] = np.concatenate(parts)
            else:
                out[f"{part}.{k}"] = np.concatenate([np.asarray(e[part][k]) for e in eps])
    return out


def write_shard(eps: list[dict], path: Path) -> str:
    """Episodes as one compressed NPZ (arrays under ``<j>/<part>/<key>``; recipes, tasks, channel tables and list
    fields as JSON), written atomically; returns its SHA-256."""
    import hashlib
    import io
    import json
    import os

    def plain(o):
        return o.tolist() if hasattr(o, "tolist") else str(o)

    arrays = {}
    for j, e in enumerate(eps):
        for part in ("context", "query"):
            for k, x in e[part].items():
                if isinstance(x, np.ndarray):
                    arrays[f"{j}/{part}/{k}"] = x
            lists = {k: x for k, x in e[part].items() if not isinstance(x, np.ndarray)}
            arrays[f"{j}/{part}/__lists"] = np.array(json.dumps(lists, default=plain))
        meta = {"recipe": e["recipe"], "task": e["task"], "channels": e["channels"]}
        arrays[f"{j}/meta"] = np.array(json.dumps(meta, default=plain))
    buf = io.BytesIO()
    np.savez_compressed(buf, n=np.array(len(eps)), **arrays)
    data = buf.getvalue()
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def read_shard(path: Path) -> list[dict]:
    import json

    z = np.load(path, allow_pickle=False)
    out = []
    for j in range(int(z["n"])):
        meta = json.loads(str(z[f"{j}/meta"]))
        ep = {"recipe": meta["recipe"], "task": meta["task"], "channels": meta["channels"]}
        for part in ("context", "query"):
            pre = f"{j}/{part}/"
            ep[part] = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre) and not k.endswith("__lists")}
            ep[part].update(json.loads(str(z[f"{pre}__lists"])))
        if ep["channels"].get("global_ids") is not None:
            ep["channels"]["global_ids"] = np.asarray(ep["channels"]["global_ids"])
        out.append(ep)
    return out


def dump(stream: EpisodeStream, indices: list[int], out_dir: Path, shard_size: int = 256, name: str = "dump") -> dict:
    """A frozen dump (validation or evaluation sets): the episodes of ``indices`` in shards of ``shard_size`` with a
    manifest (factory version, store, pool, stage, seed, split, every shard's index range and SHA-256). Resumable:
    shards already written are kept."""
    import json

    from icutwin.factory.episodes import VERSION

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {"shards": {}}
    store_dir, _v, pool_dir, stage, seed, split = stream.args
    man.update({"name": name, "version": VERSION, "store": store_dir, "pool": pool_dir, "stage": stage, "seed": seed,
                "split": split, "episodes": len(indices)})
    for k in range(0, len(indices), shard_size):
        fname = f"{name}_{k // shard_size:05d}.npz"
        if fname in man["shards"] and (out_dir / fname).exists():
            continue
        idx = indices[k:k + shard_size]
        sha = write_shard([stream.sampler().episode(i) for i in idx], out_dir / fname)
        man["shards"][fname] = {"first": idx[0], "last": idx[-1], "episodes": len(idx), "sha256": sha}
        man_path.write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man
