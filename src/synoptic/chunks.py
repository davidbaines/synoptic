"""Split/reassemble large files for artifact transport.

The ClearML file server kills uploads somewhere between 200 MB and 400 MB
(SSL EOF; probed 2026-07-25 — 200 MB passes, 400 MB fails, on every worker
tried). Run archives are ~1.5 GB, so ``train`` uploads them as numbered
parts under this size plus a checksum manifest, and ``fetch_weights``
reassembles and verifies them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PART_SIZE = 150 * 1024 * 1024  # comfortably under the ~200-400 MB failure band
MANIFEST_ARTIFACT = "run_manifest"
PART_PREFIX = "run.part"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_file(path: Path, out_dir: Path, part_size: int = PART_SIZE) -> tuple[list[Path], dict]:
    """Split ``path`` into numbered parts; return (part paths, manifest)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    with open(path, "rb") as f:
        i = 0
        while True:
            chunk = f.read(part_size)
            if not chunk:
                break
            part = out_dir / f"{PART_PREFIX}{i:03d}"
            part.write_bytes(chunk)
            parts.append(part)
            i += 1
    manifest = {
        "file_name": path.name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "parts": len(parts),
        "part_sha256": [sha256_file(p) for p in parts],
    }
    return parts, manifest


def join_files(parts: list[Path], manifest: dict, out_path: Path) -> Path:
    """Concatenate downloaded parts and verify against the manifest."""
    if len(parts) != manifest["parts"]:
        raise ValueError(f"expected {manifest['parts']} parts, got {len(parts)}")
    for i, part in enumerate(parts):
        got = sha256_file(part)
        want = manifest["part_sha256"][i]
        if got != want:
            raise ValueError(f"part {i} checksum mismatch: {got} != {want}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out:
        for part in parts:
            out.write(part.read_bytes())
    got = sha256_file(out_path)
    if got != manifest["sha256"]:
        raise ValueError(f"reassembled checksum mismatch: {got} != {manifest['sha256']}")
    return out_path
