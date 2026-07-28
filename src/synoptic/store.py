"""Run-directory transport to the shared MinIO store (the M: drive), via rclone.

Weights and run outputs live in the ``nlp-research`` bucket on
``truenas.psonet.languagetechnology.org`` — the same store silnlp's rclone
results use, mounted locally at ``~/M`` — under
``MT/experiments/synoptic/<repo>/<run>/``. The ClearML file server is NOT
used for bulk data (it drops large uploads unreliably).

Transport is **rclone**: it streams 5 GB checkpoint files in one pass, so no
chunking is needed (the legacy chunked-ClearML-artifact path in ``chunks.py``
/ ``fetch_weights.fetch_chunked`` survives only to recover pre-store runs).
rclone is addressed as an on-the-fly ``:s3:`` remote configured entirely from
the ``MINIO_ACCESS_KEY`` / ``MINIO_SECRET_KEY`` env vars already present both
locally and on the worker (via ``RCLONE_S3_*`` in the child environment, so no
``rclone.conf`` is needed anywhere and no secret reaches the argv). The binary
is found on ``PATH`` or downloaded once at runtime (``_ensure_rclone``), so the
stock worker image needs no rclone baked in.

The transport must never leave an incomplete run looking complete, because
the worker copy is ephemeral. So an upload copies every file, then writes a
manifest LAST listing every file with its checksum; a download refuses any run
whose files do not match that manifest; and a run that cannot upload every file
fails loudly rather than completing. A cheap preflight write-probe runs before
training, so an unreachable store, bad credentials, or a missing rclone binary
abort in seconds, not after GPU-hours.

The cert is valid only for the hostname (no IP SAN), while ClearML agents
inject the store as a bare IP the workers route to but cannot resolve by
name — bridged by a ``--add-host`` entry set at enqueue (see
``train._maybe_clearml``); here we always address the store by that
hostname so TLS validates.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

from .chunks import sha256_file

BUCKET = "nlp-research"
# Cert has a hostname-only SAN; always connect by this name.
MINIO_ENDPOINT = "https://truenas.psonet.languagetechnology.org:9000"
# region of the miniosilnlp remote; MinIO ignores it but the S3 backend wants one.
MINIO_REGION = "us-east-2"
MANIFEST_NAME = "_synoptic_manifest.json"
STORE_ROOT = "MT/experiments/synoptic"
# Trainer checkpoints carry optimizer states; best/ duplicates the shipped
# weights (the best checkpoint is reloaded to the run root before saving).
SKIP_DIR_PREFIXES = ("checkpoint-", "best")
# rclone glob excludes matching SKIP_DIR_PREFIXES, so the copy and the manifest
# (built from run_files) cover exactly the same files.
_EXCLUDES = ("checkpoint-*/**", "best/**", "best")


def normalize_repo(name: str) -> str:
    """Store segment for a repo: the dir name without a trailing ``.git``.

    Upload runs from the agent's clone (``<repo>.git``); fetch runs from a
    local checkout (``<repo>``). Stripping ``.git`` makes both address the
    same prefix.
    """
    return name[:-4] if name.endswith(".git") else name


def run_prefix(repo: str, run: str) -> str:
    return f"{STORE_ROOT}/{normalize_repo(repo)}/{run}/"


# ---------------------------------------------------------------------------
# rclone plumbing
# ---------------------------------------------------------------------------

def _rclone_env() -> dict | None:
    """Child env with the store's :s3: backend configured, or None if no creds.

    Secrets go through the environment (``RCLONE_S3_*``), never the argv, so
    they do not show up in ``ps`` or ClearML's captured command line.
    """
    access = os.environ.get("MINIO_ACCESS_KEY")
    secret = os.environ.get("MINIO_SECRET_KEY")
    if not (access and secret):
        return None
    env = dict(os.environ)
    env.update(
        RCLONE_S3_PROVIDER="Minio",
        RCLONE_S3_ENDPOINT=MINIO_ENDPOINT,
        RCLONE_S3_ACCESS_KEY_ID=access,
        RCLONE_S3_SECRET_ACCESS_KEY=secret,
        RCLONE_S3_REGION=MINIO_REGION,
    )
    return env


def _ensure_rclone() -> str:
    """Path to an rclone binary: found on PATH, or downloaded once at runtime.

    The stock worker image has no rclone; rather than bake a custom image we
    fetch the official static binary (a single ~18 MB zip) into a cache dir.
    Raises SystemExit if it is neither present nor fetchable, so preflight can
    abort before training.
    """
    found = shutil.which("rclone")
    if found:
        return found
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "synoptic"
    dest = cache / "rclone"
    if dest.is_file() and os.access(dest, os.X_OK):
        return str(dest)
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64",
            "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if arch is None:
        raise SystemExit(
            f"no rclone on PATH and no known download for arch {machine!r}; "
            "install rclone on this host or bake it into the worker image"
        )
    url = f"https://downloads.rclone.org/rclone-current-linux-{arch}.zip"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"  rclone not on PATH; downloading {url} ...")
    try:
        with tempfile.TemporaryDirectory() as td:
            zpath = Path(td) / "rclone.zip"
            with urlopen(url, timeout=120) as r, open(zpath, "wb") as f:
                shutil.copyfileobj(r, f)
            with zipfile.ZipFile(zpath) as zf:
                member = next(n for n in zf.namelist() if n.endswith("/rclone"))
                with zf.open(member) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
        dest.chmod(0o755)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"could not obtain rclone ({e}); install it on this host or bake it "
            "into the worker image so weights can be stored"
        )
    return str(dest)


def _remote(prefix: str = "") -> str:
    """On-the-fly ``:s3:`` remote path for a store prefix (config comes from env)."""
    return f":s3:{BUCKET}/{prefix}"


def _run_rclone(args: list[str], env: dict, what: str, capture: bool = False,
                attempts: int = 4) -> subprocess.CompletedProcess:
    """Run rclone with retries; raise SystemExit after ``attempts`` failures.

    rclone retries individual transfers internally; this wraps the whole
    invocation so a wholesale failure (endpoint down between checkpoints) still
    backs off instead of failing the run on the first blip.
    """
    rclone = _ensure_rclone()
    # --low-level-retries covers packet-level blips; the outer loop covers the
    # rare case of the endpoint being unreachable for a stretch.
    cmd = [rclone, *args, "--s3-no-check-bucket", "--retries", "5",
           "--low-level-retries", "10"]
    last = ""
    for attempt in range(1, attempts + 1):
        proc = subprocess.run(cmd, env=env, text=True, capture_output=capture)
        if proc.returncode == 0:
            return proc
        last = (proc.stderr or proc.stdout or "").strip() if capture else \
            f"exit {proc.returncode}"
        print(f"  WARNING: {what} attempt {attempt}/{attempts} failed: {last}")
        if attempt < attempts:
            # Linear backoff; concurrent fleet uploads already stagger on their
            # differing start times and run prefixes.
            time.sleep(20 * attempt)
    raise SystemExit(f"{what} failed after {attempts} attempts: {last}")


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


def preflight(repo: str, run: str) -> None:
    """Write-probe the store before training; raise loudly if unusable.

    Catches missing credentials, a missing/unfetchable rclone, an unreachable
    endpoint, or a bad ``--add-host`` mapping in seconds, instead of after
    GPU-hours followed by a silent failure to store the model.
    """
    env = _rclone_env()
    if env is None:
        raise SystemExit(
            "MINIO_* credentials are not set on this worker; the run would "
            "train and then be unable to store its weights"
        )
    prefix = run_prefix(repo, run)
    key = f"{prefix}.preflight"
    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / ".preflight"
        probe.write_bytes(b"ok")
        try:
            _run_rclone(["copyto", str(probe), _remote(key)], env,
                        "store preflight write", capture=True, attempts=2)
            _run_rclone(["deletefile", _remote(key)], env,
                        "store preflight cleanup", capture=True, attempts=2)
        except SystemExit as e:
            raise SystemExit(
                f"cannot reach the weights store ({MINIO_ENDPOINT}, bucket "
                f"{BUCKET}): {e}. Weights would be lost; aborting before training."
            )
    print(f"  store preflight OK: {BUCKET}/{prefix}")


def upload_run(output: Path, repo: str, run: str) -> None:
    """Upload a run directory, then a manifest LAST; raise if incomplete.

    Purges the destination prefix first so a re-run cannot mix new files with
    stale objects from a previous attempt. The manifest is written only after
    every file is copied, so its presence is the signal that the run is
    complete; ``download_run`` refuses a run without one.
    """
    env = _rclone_env()
    if env is None:
        raise SystemExit("MINIO_* credentials are not set; cannot store weights")
    prefix = run_prefix(repo, run)

    # Clear any prior attempt's objects under this prefix. purge fails if the
    # path does not exist yet (a first run); that is fine, so it is best-effort.
    proc = subprocess.run(
        [_ensure_rclone(), "purge", _remote(prefix), "--s3-no-check-bucket"],
        env=env, text=True, capture_output=True,
    )
    if proc.returncode == 0:
        print(f"  cleared stale objects under {prefix}")

    manifest = build_manifest(output)
    excludes = [a for x in _EXCLUDES for a in ("--exclude", x)]
    print(f"  uploading {manifest['count']} files "
          f"({manifest['total_bytes'] / 1e6:.0f} MB) to {BUCKET}/{prefix} ...")
    _run_rclone(["copy", str(output), _remote(prefix), *excludes,
                 "--transfers", "4", "--checkers", "8"],
                env, f"upload run {run}", capture=True)

    # Manifest last: its presence is the signal that the run is complete. rcat
    # streams stdin straight to the object.
    body = json.dumps(manifest, indent=2).encode("utf-8")
    rclone = _ensure_rclone()
    proc = subprocess.run(
        [rclone, "rcat", _remote(f"{prefix}{MANIFEST_NAME}"), "--s3-no-check-bucket"],
        input=body, env=env, capture_output=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            "failed to upload the run manifest "
            f"({(proc.stderr or b'').decode(errors='replace').strip()}); "
            "run marked incomplete"
        )
    print(f"  run directory stored: {BUCKET}/{prefix} (browsable at ~/M/{prefix})")


def download_run(repo: str, run: str, out_dir: Path) -> Path:
    """Download a run directory and verify it against its manifest."""
    env = _rclone_env()
    if env is None:
        raise SystemExit("MINIO_* credentials are not set; cannot fetch weights")
    prefix = run_prefix(repo, run)
    rclone = _ensure_rclone()
    proc = subprocess.run(
        [rclone, "cat", _remote(f"{prefix}{MANIFEST_NAME}"), "--s3-no-check-bucket"],
        env=env, capture_output=True,
    )
    # rclone cat of a missing object may exit 0 with empty stdout (prefix
    # present, manifest absent) as well as non-zero (prefix absent); both mean
    # the run never completed, since the manifest is written last.
    absent = SystemExit(
        f"no manifest at {BUCKET}/{prefix}{MANIFEST_NAME}: the run is "
        "absent or its upload never completed (an incomplete run has no "
        "manifest by design)"
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise absent
    try:
        manifest = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise absent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{run}: {manifest['count']} files "
          f"({manifest['total_bytes'] / 1e6:.0f} MB) from {BUCKET}/{prefix}")
    # Exclude the manifest so out_dir matches the manifest's file list exactly.
    _run_rclone(["copy", _remote(prefix), str(out_dir),
                 "--exclude", MANIFEST_NAME, "--transfers", "4", "--checkers", "8"],
                env, f"download run {run}", capture=True)
    problems = manifest_problems(out_dir, manifest)
    if problems:
        raise SystemExit(f"downloaded run does not match its manifest: {problems}")
    print(f"OK: {len(manifest['files'])} files verified against the manifest "
          f"in {out_dir}")
    return out_dir
