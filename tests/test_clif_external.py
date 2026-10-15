"""Reference CLIF tools: the inputs our runners put in place for them (no tool is run)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from icutwin.clif.external import AMSTEL_FILES, stage_amstel


def test_amstel_mapping_is_staged_from_the_pinned_checkout(tmp_path: Path) -> None:
    """CLIF-AmsterdamUMCdb reads AMSTEL's procedure mapping from its own ignored ``ref/AMSTEL``: the runner copies
    it from the pinned AMSTEL checkout and records the commit and checksum (it used to be placed by hand)."""
    amstel, tool = tmp_path / "AMSTEL", tmp_path / "CLIF-AmsterdamUMCdb"
    src = amstel / AMSTEL_FILES[0]
    src.parent.mkdir(parents=True)
    src.write_text("sourceCode,targetConceptId\n1,2\n", encoding="utf-8")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", f"safe.directory={amstel.as_posix()}", "-C", str(amstel)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    tool.mkdir()
    placed = stage_amstel(tool, amstel)
    dst = tool / "ref" / "AMSTEL" / AMSTEL_FILES[0]
    assert dst.read_bytes() == src.read_bytes()
    assert placed[0]["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest() and len(placed[0]["amstel_commit"]) == 40
    assert stage_amstel(tool, amstel) == placed   # idempotent
