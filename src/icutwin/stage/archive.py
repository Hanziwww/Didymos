"""Streaming access to the raw ``.tar.zst`` archives, without extracting them to disk.

The archives are read strictly forward (tar stream mode), so every pass over an archive costs one
sequential decompression. Member file objects are wrapped as forward-only streams because tarfile's
stream-mode objects do not implement ``seekable()``, which pandas/pyarrow probe.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import zstandard

# zstd frames in these archives may use long-distance matching windows (zstd --long=31).
_MAX_WINDOW = 2**31
_READ_SIZE = 1 << 22  # 4 MiB


class ForwardOnlyReader(io.RawIOBase):
    """Adapts a stream-mode tar member (no ``seekable``) to a well-behaved raw reader."""

    def __init__(self, fileobj: IO[bytes]):
        self._f = fileobj

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, buffer) -> int:  # type: ignore[override]
        data = self._f.read(len(buffer))
        n = len(data)
        buffer[:n] = data
        return n


@dataclass(frozen=True)
class Member:
    """One regular file inside an archive; ``rel`` is the path with the archive prefix stripped."""

    rel: str
    name: str
    size: int


@contextmanager
def open_tar_zst(path: Path) -> Iterator[tarfile.TarFile]:
    dctx = zstandard.ZstdDecompressor(max_window_size=_MAX_WINDOW)
    with open(path, "rb") as fh, dctx.stream_reader(fh, read_size=_READ_SIZE, read_across_frames=True) as reader:
        with tarfile.open(fileobj=reader, mode="r|", bufsize=_READ_SIZE) as tar:
            yield tar


def strip_prefix(name: str, prefix: str) -> str:
    if prefix and name.startswith(prefix):
        return name[len(prefix) :]
    return name


def iter_members(path: Path, prefix: str = "") -> Iterator[tuple[Member, io.BufferedReader]]:
    """Yield every regular file in archive order with a buffered forward-only reader.

    The reader is only valid until the next iteration; unread bytes are skipped by tarfile.
    """
    with open_tar_zst(path) as tar:
        for info in tar:
            if not info.isfile():
                continue
            raw = tar.extractfile(info)
            if raw is None:
                continue
            member = Member(rel=strip_prefix(info.name, prefix), name=info.name, size=info.size)
            yield member, io.BufferedReader(ForwardOnlyReader(raw), buffer_size=_READ_SIZE)
