"""Round 3 (``docs/m7_refinement.md``): the clock blocks (spec ``clock``) written into an encoded token root.

``transform_root(src, dst)`` makes a new token root from an old one: the vocabulary extended with the clock tokens
(``vocab.extend_clock``: every id unchanged, the new tokens last), and every stay's ``ids``, ``goal_ids`` and
``windows`` passed through ``fsm.insert_clock``, the transform the encoder now applies to new encodes (so the
stored shards and new encodes agree, and the sampler writes the same blocks from the automaton). Continuation
windows keep no blocks. The aggregates that read events only (``support.json``, ``los_bands.json``,
``calibration.json``) are copied: token ids are unchanged and the readers skip clock tokens. Workers per row range.

    python -m icutwin.tokenization.clock <src token root> <dst token root> [--workers 32]
"""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

COPIED = ("support.json", "los_bands.json", "calibration.json", "vocabulary.parquet")


def _rows(args: tuple) -> tuple:
    """Rows [start, stop) of one shard through insert_clock; returns (arrow table, longest window, blocks)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from icutwin.tokenization.fsm import compile_tables, insert_clock
    from icutwin.tokenization.vocab import Vocab

    src, start, stop, vocab = args
    t = compile_tables(Vocab.load(Path(vocab)))
    part = pq.read_table(src, memory_map=True).slice(start, stop - start)
    ids = [insert_clock(x, t) for x in part.column("ids").to_pylist()]
    goal = [insert_clock(x, t) for x in part.column("goal_ids").to_pylist()]
    windows = [[insert_clock(w, t) for w in ws] for ws in part.column("windows").to_pylist()]
    blocks = sum(sum(tok == t.clock.marker for tok in x) for x in ids)
    longest = max((len(w) for ws in windows for w in ws), default=0)
    out = pa.table({"subject_id": part.column("subject_id"), "stay_id": part.column("stay_id"),
                    "ids": pa.array(ids, type=pa.list_(pa.int32())), "goal_ids": pa.array(goal, type=pa.list_(pa.int32())),
                    "windows": pa.array(windows, type=pa.list_(pa.list_(pa.int32())))})
    return out, longest, blocks


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def transform_root(src: Path, dst: Path, workers: int = 8, rows_per_task: int = 2000) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from icutwin.tokenization.spec import spec_hash
    from icutwin.tokenization.vocab import Vocab, extend_clock

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    v = extend_clock(Vocab.load(src / "vocab.json"))
    v.save(dst / "vocab.json")
    for name in COPIED:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    shards = sorted(p for p in src.glob("*/*.parquet"))
    index = {"spec_sha256": spec_hash(), "vocab_sha256": _sha(dst / "vocab.json"), "shards": {},
             "clock": {"from": str(src), "tokens_appended": v.fitted_on.get("clock_tokens_appended", 0)}}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for shard in shards:
            n = pq.ParquetFile(shard).metadata.num_rows
            tasks = [(str(shard), a, min(a + rows_per_task, n), str(dst / "vocab.json"))
                     for a in range(0, n, rows_per_task)]
            parts = list(ex.map(_rows, tasks)) if tasks else []
            table = pa.concat_tables([p[0] for p in parts]) if parts else pq.read_table(shard)
            out = dst / shard.parent.name / shard.name
            out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out, compression="zstd")
            key = f"{shard.parent.name}/{shard.stem}"
            flat = table.column("windows").combine_chunks().flatten() if n else pa.array([], pa.list_(pa.int32()))
            index["shards"][key] = {"sha256": _sha(out), "subjects": n,
                                    "tokens": len(flat.flatten()), "windows": len(flat),
                                    "longest_window": max((p[1] for p in parts), default=0),
                                    "clock_blocks": int(sum(p[2] for p in parts))}
            print(key, json.dumps({k: x for k, x in index["shards"][key].items() if k != "sha256"}), flush=True)
    (dst / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    transform_root(Path(a.src), Path(a.dst), a.workers)


if __name__ == "__main__":
    main()
