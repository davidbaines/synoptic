"""Download a remote run's chunked weight archive and reassemble it.

    python -m synoptic.fetch_weights --task-id <id> --out weights/<run_name>
    python -m synoptic.fetch_weights --name ms8_arabic_drafting

Counterpart of the chunked upload in ``train._upload_artifacts``: downloads
the checksum manifest and the part artifacts it names, verifies part and
whole-archive checksums while streaming into the reassembled zip, extracts
the run directory, and removes the cached part downloads. With ``--name``,
the most recent matching task in the current repo's ClearML project (the
repo directory name) is used. The extracted directory is what
``synoptic.publish`` expects as a local run.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from .chunks import MANIFEST_ARTIFACT, join_files
from .data import repo_root


def _local_copy(artifacts, name: str) -> Path:
    obj = artifacts[name]
    local = obj.get_local_copy()
    if local is None:
        raise SystemExit(
            f"artifact {name!r} failed to download (file server error?); retry"
        )
    return Path(local)


def fetch(task, out_dir: Path) -> Path:
    artifacts = task.artifacts
    if MANIFEST_ARTIFACT not in artifacts:
        raise SystemExit(
            f"task {task.id} ({task.name}) has no {MANIFEST_ARTIFACT} artifact — "
            "either the upload failed (see its console log) or it predates "
            "chunked uploads"
        )
    manifest = json.loads(_local_copy(artifacts, MANIFEST_ARTIFACT).read_text())
    print(f"{task.name}: {manifest['parts']} parts, "
          f"{manifest['size'] / 1e6:.0f} MB archive")
    # Manifests written before part names were recorded fall back to the
    # naming-scheme scan (numeric order; checksums still verify the result).
    part_names = manifest.get("part_names") or sorted(
        (n for n in artifacts if n.startswith("run.part")),
        key=lambda n: int(n.removeprefix("run.part")),
    )
    missing = [n for n in part_names if n not in artifacts]
    if missing:
        raise SystemExit(
            f"task {task.id} is missing part artifact(s) {missing} — the "
            "upload was incomplete (see its console log)"
        )
    parts = [_local_copy(artifacts, n) for n in part_names]

    out_dir.mkdir(parents=True, exist_ok=True)
    archive = join_files(parts, manifest, out_dir / manifest["file_name"])
    for part in parts:
        part.unlink()  # cached downloads: ~150 MB each, dead after the join
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_dir)
    archive.unlink()

    config = out_dir / "config.json"
    if config.exists():
        print(f"OK: run directory at {out_dir} (model config present)")
    else:
        print(f"WARNING: extracted to {out_dir} but no config.json found")
    return out_dir


def resolve_task(task_id: str | None, name: str | None):
    from clearml import Task

    if task_id:
        task = Task.get_task(task_id=task_id)
    else:
        task = Task.get_task(project_name=repo_root().name, task_name=name)
    if task is None:
        raise SystemExit(f"No ClearML task found for {task_id or name!r}")
    print(f"Task {task.id} '{task.name}' status={task.get_status()}")
    return task


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch and reassemble run weights")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task-id")
    g.add_argument("--name", help="task name (most recent match in this "
                                  "repo's ClearML project)")
    ap.add_argument("--out", default=None,
                    help="directory for the run contents "
                         "(default checkpoints/<task name>)")
    args = ap.parse_args()
    task = resolve_task(args.task_id, args.name)
    out = (Path(args.out) if args.out
           else repo_root() / "checkpoints" / task.name)
    fetch(task, out)


if __name__ == "__main__":
    main()
