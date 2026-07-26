"""Download a remote run's chunked weight archive and reassemble it.

Counterpart of the chunked upload in ``train._upload_artifacts``: downloads
the checksum manifest and every ``run.partNNN`` artifact, verifies part and
whole-archive checksums, unzips the run directory, and sanity-loads the
model config.

    python -m synoptic.fetch_weights --task-id <id> --out weights/<run_name>
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from .chunks import MANIFEST_ARTIFACT, PART_PREFIX, join_files


def fetch(task_id: str, out_dir: Path) -> Path:
    from clearml import Task

    task = Task.get_task(task_id=task_id)
    artifacts = task.artifacts
    if MANIFEST_ARTIFACT not in artifacts:
        raise SystemExit(
            f"task {task_id} ({task.name}) has no {MANIFEST_ARTIFACT} artifact — "
            "either the upload failed (see its console log) or it predates "
            "chunked uploads"
        )
    manifest = json.loads(Path(artifacts[MANIFEST_ARTIFACT].get_local_copy()).read_text())
    part_names = sorted(a for a in artifacts if a.startswith(PART_PREFIX))
    print(f"{task.name}: {manifest['parts']} parts, "
          f"{manifest['size'] / 1e6:.0f} MB archive")
    parts = [Path(artifacts[name].get_local_copy()) for name in part_names]

    out_dir.mkdir(parents=True, exist_ok=True)
    archive = join_files(parts, manifest, out_dir / manifest["file_name"])
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_dir)
    archive.unlink()

    config = out_dir / "config.json"
    if config.exists():
        print(f"OK: run directory at {out_dir} (model config present)")
    else:
        print(f"WARNING: extracted to {out_dir} but no config.json found")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch and reassemble run weights")
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--out", required=True, help="directory for the run contents")
    args = ap.parse_args()
    fetch(args.task_id, Path(args.out))


if __name__ == "__main__":
    main()
