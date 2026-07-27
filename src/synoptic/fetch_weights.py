"""Fetch a remote run's weights from the shared MinIO store.

    python -m synoptic.fetch_weights --run ms8_arabic_drafting
    python -m synoptic.fetch_weights --task-id <id> --out weights/<run>

Runs upload their directory to the store
(``nlp-research/MT/experiments/synoptic/<repo>/<run>/``, browsable at
``~/M/...``) with a manifest written last; this downloads it and verifies
every file against that manifest (``store.download_run``), so an incomplete
upload cannot masquerade as a good run. Tasks from the brief
chunked-artifact era fall back to manifest+parts reassembly via
``--task-id``. The resulting directory is what ``synoptic.publish`` expects.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from . import store
from .chunks import MANIFEST_ARTIFACT, PART_PREFIX, join_files
from .data import repo_root


def _local_copy(artifacts, name: str) -> Path:
    obj = artifacts[name]
    local = obj.get_local_copy()
    if local is None:
        raise SystemExit(
            f"artifact {name!r} failed to download (file server error?); retry"
        )
    return Path(local)


def fetch_chunked(task, out_dir: Path) -> Path:
    """Legacy path: reassemble a chunked-artifact upload (pre-store runs)."""
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
    part_names = manifest.get("part_names") or sorted(
        (n for n in artifacts if n.startswith(PART_PREFIX)),
        key=lambda n: int(n.removeprefix(PART_PREFIX)),
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
        part.unlink()  # cached downloads, dead after the join
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_dir)
    archive.unlink()
    print(f"OK: run directory at {out_dir}")
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
    ap = argparse.ArgumentParser(description="Fetch run weights")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="run name on the MinIO store (preferred)")
    g.add_argument("--task-id", help="chunked-artifact era task fallback")
    g.add_argument("--name", help="task name lookup for the fallback path")
    ap.add_argument("--repo", default=None,
                    help="experiment repo name on the store "
                         "(default: this repo's directory name)")
    ap.add_argument("--out", default=None,
                    help="directory for the run contents "
                         "(default checkpoints/<run>)")
    args = ap.parse_args()
    if args.run:
        # store.normalize_repo makes a local checkout and the worker's
        # <repo>.git clone address the same prefix without a manual --repo.
        repo = args.repo or repo_root().name
        out = Path(args.out) if args.out else repo_root() / "checkpoints" / args.run
        store.download_run(repo, args.run, out)
        return
    task = resolve_task(args.task_id, args.name)
    out = (Path(args.out) if args.out
           else repo_root() / "checkpoints" / task.name)
    fetch_chunked(task, out)


if __name__ == "__main__":
    main()
