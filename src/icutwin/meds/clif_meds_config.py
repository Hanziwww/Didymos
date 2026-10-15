"""Generate the CLIF_MEDS concept-config directory: the tool's configs at the pinned commit + our overlay."""

from __future__ import annotations

import hashlib
import shutil
from importlib import resources
from pathlib import Path

import yaml


def _medication_block(cfg: dict) -> dict:
    table = next(k for k in cfg if k not in ("elf_version", "subject_id_col"))
    return cfg[table]["medication"]["unit_conversion"]


def build_config_dir(tool_config_dir: Path, out_dir: Path) -> dict:
    overlay_text = resources.files("icutwin.meds").joinpath("clif_meds_overlay.yaml").read_text(encoding="utf-8")
    overlay = yaml.safe_load(overlay_text)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(tool_config_dir, out_dir)
    if (overlay.pop("_all_", {}) or {}).get("disable_outlier_shaping"):
        for path in sorted(out_dir.glob("*.yaml")):
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(cfg, dict) and "outlier_shaping" in cfg:
                cfg["outlier_shaping"]["enabled"] = False
                path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    for fname, spec in overlay.items():
        path = out_dir / fname
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        uc = _medication_block(cfg)
        prefs = uc["preferred_units"]
        for old, news in (spec.get("rename") or {}).items():
            if old not in prefs:
                raise KeyError(f"{fname}: overlay renames {old!r}, which the tool config does not define")
            unit = prefs.pop(old)
            for new in news:
                prefs.setdefault(new, unit)
        for cat, unit in (spec.get("add") or {}).items():
            prefs.setdefault(cat, unit)
        if "allow_other_meds" in spec:
            uc["allow_other_meds"] = spec["allow_other_meds"]
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return {"overlay_sha256": hashlib.sha256(overlay_text.encode()).hexdigest(), "config_dir": str(out_dir)}
