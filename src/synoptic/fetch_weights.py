"""Download a remote run's weights.

    python -m synoptic.fetch_weights --run ms8_arabic_drafting
    python -m synoptic.fetch_weights --task-id <id> --out weights/<run>

Runs upload their directory to the shared MinIO store
(``nlp-research/MT/experiments/synoptic/<repo>/<run>/``, browsable at
``~/M/...``); this downloads it with the MINIO_* credentials. Tasks from the
brief chunked-artifact era fall back to manifest+parts reassembly via
``--task-id``. The resulting directory is what ``synoptic.publish`` expects
as a local run.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from .chunks import MANIFEST_ARTIFACT, join_files
from .data import repo_root

BUCKET = "nlp-research"


def fetch_s3(run: str, out_dir: Path, repo: str | None = None) -> Path:
    """Download a run directory from the MinIO store."""
    import os

    import boto3

    endpoint = os.environ.get("MINIO_ENDPOINT_URL")
    access = os.environ.get("MINIO_ACCESS_KEY")
    secret = os.environ.get("MINIO_SECRET_KEY")
    if not (endpoint and access and secret):
        raise SystemExit("MINIO_* environment variables are not set")
    s3 = boto3.client("s3", endpoint_url=endpoint,
                      aws_access_key_id=access, aws_secret_access_key=secret)
    repo = repo or repo_root().name
    prefix = f"MT/experiments/synoptic/{repo}/{run}/"
    paginator = s3.get_paginator("list_objects_v2")
    keys = [o["Key"] for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix)
            for o in page.get("Contents", [])]
    if not keys:
        raise SystemExit(f"nothing stored under {BUCKET}/{prefix}")
    print(f"{run}: {len(keys)} files from {BUCKET}/{prefix}")
    for key in keys:
        target = out_dir / key.removeprefix(prefix)
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(target))
    config = out_dir / "config.json"
    if config.exists():
        print(f"OK: run directory at {out_dir} (model config present)")
    else:
        print(f"WARNING: downloaded to {out_dir} but no config.json found")
    return out_dir


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
    ap = argparse.ArgumentParser(description="Fetch run weights")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="run name on the MinIO store (preferred)")
    g.add_argument("--task-id", help="chunked-artifact era task fallback")
    g.add_argument("--name", help="task name lookup for the fallback path")
    ap.add_argument("--repo", default=None,
                    help="experiment repo name on the store (default: this repo)")
    ap.add_argument("--out", default=None,
                    help="directory for the run contents "
                         "(default checkpoints/<run>)")
    args = ap.parse_args()
    if args.run:
        out = Path(args.out) if args.out else repo_root() / "checkpoints" / args.run
        fetch_s3(args.run, out, repo=args.repo)
        return
    task = resolve_task(args.task_id, args.name)
    out = (Path(args.out) if args.out
           else repo_root() / "checkpoints" / task.name)
    fetch(task, out)


if __name__ == "__main__":
    main()
