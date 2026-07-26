"""Split/reassemble large files for artifact transport.

The ClearML file server kills large uploads unreliably (SSL EOF), with a
failure threshold that drifts between roughly 50 MB and 400 MB from day to
day. Run archives are ~1.5 GB, so ``train`` uploads them as numbered parts
small enough to stay reliable, plus a checksum manifest, and
``fetch_weights`` reassembles and verifies them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# The file server's tolerance drifts: a 200 MB probe passed on 2026-07-25,
# then 150 MB parts failed 4/4 retries on 2026-07-26, while every upload at
# or below ~50 MB has succeeded. Stay well under the observed failure zone.
PART_SIZE = 48 * 1024 * 1024
MANIFEST_ARTIFACT = "run_manifest"
PART_PREFIX = "run.part"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_file(path: Path, out_dir: Path, part_size: int = PART_SIZE) -> tuple[list[Path], dict]:
    """Split ``path`` into numbered parts; return (part paths, manifest).

    Single pass: whole-file and per-part digests are computed from the chunks
    as they are written (re-hashing a ~1.5 GB archive after writing would
    read everything twice more).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    part_hashes: list[str] = []
    whole = hashlib.sha256()
    with open(path, "rb") as f:
        i = 0
        while True:
            chunk = f.read(part_size)
            if not chunk:
                break
            whole.update(chunk)
            part_hashes.append(hashlib.sha256(chunk).hexdigest())
            part = out_dir / f"{PART_PREFIX}{i:03d}"
            part.write_bytes(chunk)
            parts.append(part)
            i += 1
    manifest = {
        "file_name": path.name,
        "size": path.stat().st_size,
        "sha256": whole.hexdigest(),
        "parts": len(parts),
        "part_sha256": part_hashes,
        # Exact artifact names, so a fetch never depends on the fetch-time
        # library agreeing with the upload-time naming scheme.
        "part_names": [p.name for p in parts],
    }
    return parts, manifest


def join_files(parts: list[Path], manifest: dict, out_path: Path) -> Path:
    """Concatenate downloaded parts and verify against the manifest.

    Streaming single pass: each part is read in 1 MB blocks that feed the
    per-part digest, the whole-file digest and the output file at once; a
    corrupt part aborts at that part, before the rest is read.
    """
    if len(parts) != manifest["parts"]:
        raise ValueError(f"expected {manifest['parts']} parts, got {len(parts)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    whole = hashlib.sha256()
    with open(out_path, "wb") as out:
        for i, part in enumerate(parts):
            per_part = hashlib.sha256()
            with open(part, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    per_part.update(block)
                    whole.update(block)
                    out.write(block)
            got, want = per_part.hexdigest(), manifest["part_sha256"][i]
            if got != want:
                raise ValueError(f"part {i} checksum mismatch: {got} != {want}")
    got = whole.hexdigest()
    if got != manifest["sha256"]:
        raise ValueError(f"reassembled checksum mismatch: {got} != {manifest['sha256']}")
    return out_path
