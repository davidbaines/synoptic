"""Run-directory transport to the shared MinIO store (the M: drive).

Weights and run outputs live in the ``nlp-research`` bucket on
``truenas.psonet.languagetechnology.org`` — the same store silnlp's rclone
results use, mounted locally at ``~/M`` — under
``MT/experiments/synoptic/<repo>/<run>/``. The ClearML file server is NOT
used for bulk data (it drops large uploads unreliably).

The transport must never leave an incomplete run looking complete, because
the worker copy is ephemeral. So an upload writes a manifest LAST listing
every file with its checksum; a download refuses any run whose files do not
match that manifest; and a run that cannot upload every file fails loudly
rather than completing. A cheap preflight write-probe runs before training,
so an unreachable store or bad credentials abort in seconds, not after
GPU-hours.

The cert is valid only for the hostname (no IP SAN), while ClearML agents
inject the store as a bare IP the workers route to but cannot resolve by
name — bridged by a ``--add-host`` entry set at enqueue (see
``train._maybe_clearml``); here we always address the store by that
hostname so TLS validates.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from .chunks import sha256_file

BUCKET = "nlp-research"
# Cert has a hostname-only SAN; always connect by this name.
MINIO_ENDPOINT = "https://truenas.psonet.languagetechnology.org:9000"
MANIFEST_NAME = "_synoptic_manifest.json"
STORE_ROOT = "MT/experiments/synoptic"
# Trainer checkpoints carry optimizer states; best/ duplicates the shipped
# weights (the best checkpoint is reloaded to the run root before saving).
SKIP_DIR_PREFIXES = ("checkpoint-", "best")


def normalize_repo(name: str) -> str:
    """Store segment for a repo: the dir name without a trailing ``.git``.

    Upload runs from the agent's clone (``<repo>.git``); fetch runs from a
    local checkout (``<repo>``). Stripping ``.git`` makes both address the
    same prefix.
    """
    return name[:-4] if name.endswith(".git") else name


def run_prefix(repo: str, run: str) -> str:
    return f"{STORE_ROOT}/{normalize_repo(repo)}/{run}/"


def client():
    """boto3 S3 client for the store, or None if credentials are absent."""
    access = os.environ.get("MINIO_ACCESS_KEY")
    secret = os.environ.get("MINIO_SECRET_KEY")
    if not (access and secret):
        return None
    import boto3

    return boto3.client(
        "s3", endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=access, aws_secret_access_key=secret,
    )


def run_files(output: Path) -> list[Path]:
    """Files of a run directory worth storing (skips checkpoints and best/)."""
    return [
        f for f in sorted(output.rglob("*"))
        if f.is_file() and not any(
            part.startswith(skip)
            for part in f.relative_to(output).parts for skip in SKIP_DIR_PREFIXES
        )
    ]


def build_manifest(output: Path) -> dict:
    """Manifest of a run directory: every stored file with sha256 and size."""
    files = run_files(output)
    entries = [
        {"path": str(f.relative_to(output)), "sha256": sha256_file(f),
         "size": f.stat().st_size}
        for f in files
    ]
    return {
        "count": len(entries),
        "total_bytes": sum(e["size"] for e in entries),
        "files": entries,
    }


def manifest_problems(out_dir: Path, manifest: dict) -> list[str]:
    """Reasons a downloaded directory fails to match its manifest (empty = ok)."""
    problems = []
    for e in manifest["files"]:
        f = out_dir / e["path"]
        if not f.is_file():
            problems.append(f"missing {e['path']}")
        elif f.stat().st_size != e["size"]:
            problems.append(f"size mismatch {e['path']}")
        elif sha256_file(f) != e["sha256"]:
            problems.append(f"checksum mismatch {e['path']}")
    return problems


def _retry(fn, what: str, attempts: int = 4) -> bool:
    """Run ``fn`` with jittered backoff; True on success, False on give-up."""
    for attempt in range(1, attempts + 1):
        try:
            fn()
            return True
        except Exception as e:  # noqa: BLE001 - any transport error
            print(f"  WARNING: {what} attempt {attempt}/{attempts} failed: {e}")
        if attempt < attempts:
            # Jitter so concurrent fleet uploads don't retry in lockstep.
            time.sleep(20 * attempt + random.uniform(0, 10))
    return False


def preflight(repo: str, run: str) -> None:
    """Write-probe the store before training; raise loudly if unusable.

    Catches missing credentials, an unreachable endpoint, or a bad
    ``--add-host`` mapping in seconds, instead of after GPU-hours followed by
    a silent failure to store the model.
    """
    s3 = client()
    if s3 is None:
        raise SystemExit(
            "MINIO_* credentials are not set on this worker; the run would "
            "train and then be unable to store its weights"
        )
    key = f"{run_prefix(repo, run)}.preflight"
    try:
        s3.put_object(Bucket=BUCKET, Key=key, Body=b"ok")
        s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        s3.delete_object(Bucket=BUCKET, Key=key)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"cannot reach the weights store ({MINIO_ENDPOINT}, bucket "
            f"{BUCKET}): {e}. Weights would be lost; aborting before training."
        )
    print(f"  store preflight OK: {BUCKET}/{run_prefix(repo, run)}")


def upload_run(output: Path, repo: str, run: str) -> None:
    """Upload a run directory, then a manifest LAST; raise if incomplete.

    Clears the destination prefix first so a re-run cannot mix new files
    with stale shards from a previous attempt.
    """
    s3 = client()
    if s3 is None:
        raise SystemExit("MINIO_* credentials are not set; cannot store weights")
    prefix = run_prefix(repo, run)

    # Clear any prior attempt's objects under this prefix.
    existing = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", [])
    if existing:
        s3.delete_objects(
            Bucket=BUCKET,
            Delete={"Objects": [{"Key": o["Key"]} for o in existing]},
        )
        print(f"  cleared {len(existing)} stale object(s) under {prefix}")

    manifest = build_manifest(output)
    print(f"  uploading {manifest['count']} files "
          f"({manifest['total_bytes'] / 1e6:.0f} MB) to {BUCKET}/{prefix} ...")
    for e in manifest["files"]:
        src = output / e["path"]
        key = f"{prefix}{e['path']}"
        if not _retry(lambda: s3.upload_file(str(src), BUCKET, key),
                      f"upload {e['path']}"):
            raise SystemExit(
                f"failed to upload {e['path']} to the store after retries; "
                f"the run is incomplete on {BUCKET}/{prefix} — failing the "
                "task so it is not mistaken for a stored model (scores remain "
                "in the console log)"
            )
    # Manifest last: its presence is the signal that the run is complete.
    body = json.dumps(manifest, indent=2).encode("utf-8")
    if not _retry(lambda: s3.put_object(Bucket=BUCKET, Key=f"{prefix}{MANIFEST_NAME}",
                                        Body=body), "upload manifest"):
        raise SystemExit("failed to upload the run manifest; run marked incomplete")
    print(f"  run directory stored: {BUCKET}/{prefix} (browsable at ~/M/{prefix})")


def download_run(repo: str, run: str, out_dir: Path) -> Path:
    """Download a run directory and verify it against its manifest."""
    s3 = client()
    if s3 is None:
        raise SystemExit("MINIO_* credentials are not set; cannot fetch weights")
    prefix = run_prefix(repo, run)
    try:
        body = s3.get_object(Bucket=BUCKET, Key=f"{prefix}{MANIFEST_NAME}")["Body"].read()
    except Exception:  # noqa: BLE001
        raise SystemExit(
            f"no manifest at {BUCKET}/{prefix}{MANIFEST_NAME}: the run is "
            "absent or its upload never completed (an incomplete run has no "
            "manifest by design)"
        )
    manifest = json.loads(body)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{run}: {manifest['count']} files "
          f"({manifest['total_bytes'] / 1e6:.0f} MB) from {BUCKET}/{prefix}")
    for e in manifest["files"]:
        target = out_dir / e["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not _retry(lambda: s3.download_file(BUCKET, f"{prefix}{e['path']}", str(target)),
                      f"download {e['path']}"):
            raise SystemExit(f"failed to download {e['path']} after retries")
    problems = manifest_problems(out_dir, manifest)
    if problems:
        raise SystemExit(f"downloaded run does not match its manifest: {problems}")
    print(f"OK: {len(manifest['files'])} files verified against the manifest "
          f"in {out_dir}")
    return out_dir
