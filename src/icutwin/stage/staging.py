"""Stage one raw archive into its native on-disk layout (S1).

Two modes:
* ``subset`` — deterministic patient-hash sample (pass 1 reads the anchor table, pass 2 filters every
  data table to the selected patients' keys). Used for all local development runs.
* ``full``   — every member copied byte-exactly (server runs).

Selection is ``blake2b(salt|dataset|patient_id) < frac``, so it is stable across runs, machines and
dataset versions, and independent of the train/tuning/held_out split salt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from icutwin.stage.archive import iter_members
from icutwin.stage.csvstream import MemberStats, copy_member, filter_csv, read_all_strings
from icutwin.stage.spec import DatasetSpec

log = logging.getLogger(__name__)

MANIFEST = "_staging_manifest.json"


def patient_selected(patient_id: str, dataset: str, frac: float, salt: str) -> bool:
    digest = hashlib.blake2b(f"{salt}|{dataset}|{patient_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64 < frac


def anchor_key_sets(spec: DatasetSpec, archive: Path, frac: float, salt: str) -> tuple[dict[str, pa.Array], dict]:
    """Pass 1: read the anchor table and derive every key set used by filter rules."""
    for member, stream in iter_members(archive, spec.strip_prefix):
        if member.rel != spec.anchor.member:
            continue
        table = read_all_strings(stream, spec.encoding)
        patients = table.column(spec.anchor.patient_col)
        unique_patients = pc.unique(patients).to_pylist()
        chosen = [p for p in unique_patients if p != "" and patient_selected(p, spec.name, frac, salt)]
        keys: dict[str, pa.Array] = {"patient": pa.array(sorted(chosen), pa.string())}
        mask = pc.is_in(patients, value_set=keys["patient"])
        for name, col in spec.anchor.derived_keys.items():
            keys[name] = pc.unique(pc.filter(table.column(col), mask)).sort()
        counts = {
            "anchor_rows": table.num_rows,
            "patients_total": len(unique_patients),
            **{f"{k}_selected": len(v) for k, v in keys.items()},
        }
        return keys, counts
    raise FileNotFoundError(f"{spec.name}: anchor member {spec.anchor.member!r} not found in {archive}")


def _output_path(out_dir: Path, rel: str, compression: str) -> Path:
    if compression == "gzip" and rel.endswith(".csv"):
        return out_dir / f"{rel}.gz"
    return out_dir / rel


def stage_dataset(
    spec: DatasetSpec,
    archive_root: Path,
    out_root: Path,
    refs_root: Path,
    mode: str = "subset",
    frac: float = 0.02,
    salt: str = "icutwin-subset-v1",
    compression: str | None = None,
    include: tuple[str, ...] = (),
) -> dict:
    if mode not in {"subset", "full"}:
        raise ValueError(f"mode must be 'subset' or 'full', got {mode!r}")
    if mode == "subset" and spec.delimiter != ",":
        raise ValueError(f"{spec.name}: subset filtering parses ',' only; stage {spec.delimiter!r} CSVs in full mode")
    archive = archive_root / spec.archive
    out_dir = out_root / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    compression = compression or spec.out_compression
    started = time.time()

    keys: dict[str, pa.Array] | None = None
    selection: dict = {}
    if mode == "subset":
        keys, selection = anchor_key_sets(spec, archive, frac, salt)
        log.info("%s: selected %s", spec.name, selection)

    members: list[dict] = []
    for member, stream in iter_members(archive, spec.strip_prefix):
        rule = spec.match(member.rel)
        action = rule.action
        if rule.optional is not None and rule.optional not in include:
            action = "skip"
        if action == "checksums":
            action = "copy" if mode == "full" else "skip"
        t0 = time.time()
        stats: MemberStats | None = None
        dst: Path | None = None
        if action == "filter":
            dst = _output_path(out_dir, member.rel, compression)
            member_keys = keys[rule.key] if keys is not None else None
            if keys is None:  # full mode: byte-exact copy of data tables
                stats = copy_member(stream, dst, compression if member.rel.endswith(".csv") else "none")
            else:
                stats = filter_csv(stream, dst, compression, rule.key_col, member_keys, spec.encoding)
        elif action == "copy":
            dst = _output_path(out_dir, member.rel, compression)
            stats = copy_member(stream, dst, compression if member.rel.endswith(".csv") else "none")
        elif action == "refs":
            dst = refs_root / "physionet_bundled" / spec.name / member.rel
            stats = copy_member(stream, dst, "none")
        record = {"member": member.rel, "action": action, "size_in": member.size, "seconds": round(time.time() - t0, 2)}
        if stats is not None:
            record.update(asdict(stats))
            record["path"] = str(dst)
        members.append(record)
        if action in {"filter", "copy"}:
            log.info("%s: %-8s %s rows_out=%s (%.1fs)", spec.name, action, member.rel, record.get("rows_out"), record["seconds"])

    manifest = {
        "dataset": spec.name,
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "mode": mode,
        "frac": frac if mode == "subset" else 1.0,
        "salt": salt if mode == "subset" else None,
        "compression": compression,
        "encoding": spec.encoding,
        **({"delimiter": spec.delimiter} if spec.delimiter != "," else {}),
        "include": list(include),
        "selection": selection,
        "members": members,
        "seconds": round(time.time() - started, 1),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
