"""``icutwin_tokenize``: S5 per the grammar spec.

Tasks: ``fit`` (vocabulary, bins and dequantization statistics from the train split of every dataset),
``encode`` (token shards per dataset and split, with windows and an index) and ``validate`` (A5 automaton and
parser acceptance, A6 round trip, A9 goal consistency, A10 report; fails on any rejection or mismatch).
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import hydra
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig

from icutwin.tokenization import backoff
from icutwin.tokenization.encode import encode
from icutwin.tokenization.fsm import compile_tables
from icutwin.tokenization.spec import spec_hash
from icutwin.tokenization.stays import iter_stays, load_derived, shard_files
from icutwin.tokenization.validate import Report, validate_stay
from icutwin.tokenization.vocab import Vocab, _Counts, accumulate, fit, fit_merged

log = logging.getLogger(__name__)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_part(task: tuple) -> tuple[dict, int, set]:
    meds_dir, derived_dir, part = task
    c = _Counts()
    n, datasets = accumulate(c, iter_stays(Path(meds_dir), "train", file=Path(part[0]), start=part[1], stop=part[2],
                                           derived=_derived(derived_dir)))
    return c.plain(), n, datasets


def fit_vocab(meds_root: Path, derived_root: Path, datasets: list[str], out_root: Path, workers: int = 1,
              chunk_stays: int = 2000) -> Vocab:
    """The vocabulary from the train split only (A7): held-out and tuning stays are never read. With ``workers``
    > 1 the statistics are gathered over chunks of subjects in parallel processes and merged (the same vocab)."""
    if workers > 1:
        tasks = [(str(meds_root / ds), str(derived_root / ds), part)
                 for ds in datasets for part in _chunks(meds_root / ds, "train", chunk_stays)]
        with _pool(workers) as pool:
            v = fit_merged(list(pool.map(_fit_part, tasks)))
    else:
        v = fit(s for ds in datasets for s in iter_stays(meds_root / ds, "train", derived_root / ds))
    out_root.mkdir(parents=True, exist_ok=True)
    v.save(out_root / "vocab.json")
    table = pa.table({"id": pa.array(range(len(v.tokens)), pa.int32()), "token": v.tokens,
                      "class": [m.get("cls") for m in v.meta], "field": [m.get("field") for m in v.meta],
                      "code": [m.get("code") for m in v.meta], "bin": [m.get("bin") for m in v.meta],
                      "category_kind": [m.get("category_kind") or None for m in v.meta]})
    pq.write_table(table, out_root / "vocabulary.parquet")
    return v


# -- parallel parts: chunks of a shard's subjects, in the order one process reads the stays --------------------
def _chunks(meds_dir: Path, split: str, size: int) -> list[tuple[str, int, int]]:
    out = []
    for f in shard_files(meds_dir, split):
        n = pl.scan_parquet(f).select(pl.col("subject_id").n_unique()).collect().item()
        out += [(str(f), a, min(a + size, n)) for a in range(0, n, size)]
    return out


@lru_cache(maxsize=2)
def _vocab(path: str) -> Vocab:
    return Vocab.load(Path(path))


@lru_cache(maxsize=1)
def _derived(path: str) -> dict:
    return load_derived(Path(path)) if Path(path).exists() else {}


def _encode_rows(meds_dir: Path, split: str, derived_dir: Path, v: Vocab, part: tuple | None = None) -> dict:
    rows = {"subject_id": [], "stay_id": [], "ids": [], "goal_ids": [], "windows": []}
    stays = (iter_stays(meds_dir, split, derived_dir) if part is None else
             iter_stays(meds_dir, split, file=Path(part[0]), start=part[1], stop=part[2],
                        derived=_derived(str(derived_dir))))
    for s in stays:
        s = backoff.rewrite(s, v)              # amendment 17: off unless ICUTWIN_BACKOFF is set
        plain, goal = encode(s, v), encode(s, v, goal=True)
        rows["subject_id"].append(s.subject_id)
        rows["stay_id"].append(s.stay_id)
        rows["ids"].append(plain.ids)
        rows["goal_ids"].append(goal.ids)
        rows["windows"].append(plain.windows)
    return rows


def _encode_part(task: tuple) -> dict:
    meds_dir, split, derived_dir, vocab_path, part = task
    return _encode_rows(Path(meds_dir), split, Path(derived_dir), _vocab(vocab_path), part)


def _validate_part(task: tuple) -> Report:
    meds_dir, split, derived_dir, vocab_path, part = task
    v = _vocab(vocab_path)
    rep, derived = Report(), _derived(derived_dir)
    for s in iter_stays(Path(meds_dir), split, file=Path(part[0]), start=part[1], stop=part[2], derived=derived):
        validate_stay(backoff.rewrite(s, v), v, _tables(vocab_path), rep, derived)
    return rep


@lru_cache(maxsize=2)
def _tables(vocab_path: str):
    return compile_tables(_vocab(vocab_path))


def _pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))


def encode_all(v: Vocab, meds_root: Path, derived_root: Path, datasets: list[str], splits: list[str],
               out_root: Path, workers: int = 1, chunk_stays: int = 2000) -> dict:
    """Token shards per dataset and split. With ``workers`` > 1, chunks of ``chunk_stays`` subjects are encoded
    in parallel processes and put together in the one-process order, so the shards are the same."""
    index = {"spec_sha256": spec_hash(), "vocab_sha256": _sha(out_root / "vocab.json"), "shards": {}}
    pool = _pool(workers) if workers > 1 else None
    for ds in datasets:
        for split in splits:
            if pool is None:
                rows = _encode_rows(meds_root / ds, split, derived_root / ds, v)
            else:
                tasks = [(str(meds_root / ds), split, str(derived_root / ds), str(out_root / "vocab.json"), part)
                         for part in _chunks(meds_root / ds, split, chunk_stays)]
                rows = {"subject_id": [], "stay_id": [], "ids": [], "goal_ids": [], "windows": []}
                for part in pool.map(_encode_part, tasks):
                    for k in rows:
                        rows[k] += part[k]
            table = pa.table({"subject_id": pa.array(rows["subject_id"], pa.int64()),
                              "stay_id": pa.array(rows["stay_id"], pa.string()),
                              "ids": pa.array(rows["ids"], pa.list_(pa.int32())),
                              "goal_ids": pa.array(rows["goal_ids"], pa.list_(pa.int32())),
                              "windows": pa.array(rows["windows"], pa.list_(pa.list_(pa.int32())))})
            path = out_root / ds / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path)
            index["shards"][f"{ds}/{split}"] = {"subjects": len(rows["ids"]),
                                                 "tokens": sum(len(i) for i in rows["ids"]),
                                                 "windows": sum(len(w) for w in rows["windows"]),
                                                 "sha256": _sha(path)}
    (out_root / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    if pool is not None:
        pool.shutdown()
    return index


def validate_all(v: Vocab, meds_root: Path, derived_root: Path, datasets: list[str], splits: list[str],
                 workers: int = 1, chunk_stays: int = 2000, vocab_path: Path | None = None) -> dict:
    """Per dataset the validation report; with ``workers`` > 1 over chunks of subjects in parallel processes
    (``vocab_path``: the fitted vocabulary they load), merged in the one-process order."""
    tables = compile_tables(v)
    out = {}
    pool = _pool(workers) if workers > 1 else None
    for ds in datasets:
        rep = Report()
        if pool is None:
            derived = load_derived(derived_root / ds) if (derived_root / ds).exists() else {}
            for split in splits:
                for s in iter_stays(meds_root / ds, split, derived_root / ds):
                    validate_stay(backoff.rewrite(s, v), v, tables, rep, derived)
        else:
            tasks = [(str(meds_root / ds), split, str(derived_root / ds), str(vocab_path), part)
                     for split in splits for part in _chunks(meds_root / ds, split, chunk_stays)]
            for part in pool.map(_validate_part, tasks):
                rep.merge(part)
        out[ds] = rep.summary()
        log.info("%s: %s", ds, {k: out[ds][k] for k in ("stays", "fsm_rejected", "parser_rejected", "round_trip_failed",
                                                         "reencode_failed", "sampled_reencode_failed", "goal_mismatch",
                                                         "continuity_failed", "separation_failed")})
    if pool is not None:
        pool.shutdown()
    return out


def vocabulary_summary(v: Vocab) -> dict:
    by_class: dict = {}
    for m in v.meta:
        by_class[m.get("cls")] = by_class.get(m.get("cls"), 0) + 1
    kinds: dict = {}
    for m in v.meta:
        if m.get("category_kind"):
            kinds[m["category_kind"]] = kinds.get(m["category_kind"], 0) + 1
    return {"tokens": len(v.tokens), "by_class": by_class, "category_kinds": kinds, "fitted_on": v.fitted_on}


def write_report(qa_root: Path, vocab: dict, per_dataset: dict) -> None:
    qa_root.mkdir(parents=True, exist_ok=True)
    (qa_root / "tokenizer_report.json").write_text(json.dumps({"vocabulary": vocab, "datasets": per_dataset},
                                                              indent=1, default=str), encoding="utf-8")
    lines = ["# Tokenizer validation (M4: A5, A6, A9, A10)", "",
             f"Vocabulary: {vocab['tokens']} tokens; by class {vocab['by_class']}; category kinds "
             f"{vocab['category_kinds']}; fitted on {vocab['fitted_on']}.", "",
             "| Dataset | Stays | Tokens/stay median (p99, max) | Stays in >1 window | Oversized windows | FSM rejected "
             "| Parser rejected | Round trip failed | Re-encode failed (median, sampled) | Goal mismatches "
             "| Window continuity failed | Separation failed |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds, r in per_dataset.items():
        t = r["tokens_per_stay"]
        lines.append(f"| {ds} | {r['stays']} | {t['median']:.0f} ({t['p99']:.0f}, {t['max']}) | "
                     f"{r['windows_per_stay']['>1']} | {r['oversized_windows']} | {sum(r['fsm_rejected'].values())} | "
                     f"{sum(r['parser_rejected'].values())} | {r['round_trip_failed']} | {r['reencode_failed']}, "
                     f"{r['sampled_reencode_failed']} | {sum(r['goal_mismatch'].values())} | {r['continuity_failed']} | "
                     f"{r['separation_failed']} |")
    lines += ["", "Events without a token of their own (by reason):", ""]
    for ds, r in per_dataset.items():
        lines.append(f"- {ds}: {r['events_without_own_token']}; measured: {r['measured']}")
    (qa_root / "tokenizer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: DictConfig) -> None:
    meds_root, derived_root, out_root = Path(cfg.meds_root), Path(cfg.derived_root), Path(cfg.out_root)
    datasets, splits = list(cfg.datasets), list(cfg.splits)
    if cfg.get("backoff"):                     # amendment 17: encoding outside the vocabulary's data, never a fit
        if cfg.task == "fit":
            raise ValueError("the code back-off applies to encoding only; a vocabulary is never fitted with it")
        os.environ[backoff.ENV] = str(cfg.backoff)   # read by every worker process (spawned after this)
    if cfg.task == "fit":
        v = fit_vocab(meds_root, derived_root, datasets, out_root, int(cfg.workers), int(cfg.chunk_stays))
        log.info("vocabulary: %s", vocabulary_summary(v))
        return
    v = Vocab.load(out_root / "vocab.json")
    if v.spec_hash != spec_hash():
        raise ValueError("the vocabulary was fitted with another grammar spec; run task=fit")
    if cfg.task == "encode":
        index = encode_all(v, meds_root, derived_root, datasets, splits, out_root, int(cfg.workers), int(cfg.chunk_stays))
        log.info("encoded: %s", {k: (s["subjects"], s["tokens"]) for k, s in index["shards"].items()})
    elif cfg.task == "validate":
        per = validate_all(v, meds_root, derived_root, datasets, splits, int(cfg.workers), int(cfg.chunk_stays),
                           out_root / "vocab.json")
        write_report(Path(cfg.qa_root), vocabulary_summary(v), per)
        bad = {ds: r for ds, r in per.items() if r["fsm_rejected"] or r["parser_rejected"] or r["round_trip_failed"]
               or r["reencode_failed"] or r["sampled_reencode_failed"] or r["goal_mismatch"]
               or r["continuity_failed"] or r["separation_failed"]}
        if bad:
            raise ValueError(f"tokenizer validation failed: {json.dumps(bad, default=str)[:3000]}")
        log.info("tokenizer validation passed; report in %s", cfg.qa_root)
    else:
        raise ValueError(f"unknown task {cfg.task!r}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="tokenize")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
