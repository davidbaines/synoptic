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
is found on ``PATH`` or downloaded once at runtime (``_ensure_rclone``) — a
pinned version, verified against a hardcoded SHA-256 before it is executed —
so the stock worker image needs nothing baked in.

The transport must never leave an incomplete run looking complete, because
the worker copy is ephemeral. So an upload purges the destination, copies
exactly the manifest's files, then writes the manifest LAST; a download reads
the manifest first, fetches exactly its files, and refuses any run whose files
do not match it; and a run that cannot upload every file fails loudly rather
than completing. The manifest's file list is the single source of truth for
"what belongs to this run" — both the upload copy and the download fetch use
it via rclone ``--files-from``, so the stored set and the manifest cannot drift
apart. A cheap preflight write/read/delete probe runs before training, so an
unreachable store, bad credentials, or a missing rclone binary abort in
seconds, not after GPU-hours.

The cert is valid only for the hostname (no IP SAN), while ClearML agents
inject the store as a bare IP the workers route to but cannot resolve by
name — bridged by a ``--add-host`` entry set at enqueue (see
``train._maybe_clearml``); here we always address the store by that
hostname so TLS validates.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
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
# run_files applies this as a startswith on every path part; it is the ONLY
# definition of what a run stores — the rclone copy is driven from the manifest
# built off it (--files-from), so there is no second, divergent glob rule.
SKIP_DIR_PREFIXES = ("checkpoint-", "best")

# Flags applied to every rclone invocation against the store. Centralised so a
# store-wide change (endpoint option, retry budget, a CA flag on cert rotation)
# is made in exactly one place. --s3-no-check-bucket avoids a HeadBucket the
# credentials may not be allowed to make; the retry flags cover packet-level
# blips (the outer _rclone loop covers a longer endpoint outage).
STORE_FLAGS = ("--s3-no-check-bucket", "--retries", "5", "--low-level-retries", "10")

# Pinned rclone for the runtime fetch: a fixed version and a hardcoded SHA-256
# of the EXTRACTED BINARY per arch, so a truncated, regressed, or tampered
# download is rejected before it is ever executed (the binary runs with the
# store credentials in its env). Hashing the binary (not the zip) means the
# on-disk cache can be re-verified against the same constant on every reuse.
# Derived from the official downloads.rclone.org release (zip verified against
# the published SHA256SUMS, then the unpacked `rclone` hashed).
RCLONE_VERSION = "v1.74.3"
RCLONE_SHA256 = {
    "amd64": "9700aa1273ac73d6d0833c43ba63fe830516422cb131960b8c1a24ced789cba0",
    "arm64": "646d2db7e701a4d41d39ed38a71f63373ab051b270ee5f0d6ae14b24cc17c923",
}

_RCLONE_PATH: str | None = None  # process-lifetime cache for _ensure_rclone


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
    """Path to a trusted rclone binary: found on PATH, or fetched once at runtime.

    The stock worker image has no rclone; rather than bake a custom image we
    fetch a PINNED version and verify it against a hardcoded SHA-256 before
    executing it. The download is written to a temp file and atomically renamed
    onto the cache path, so concurrent fleet workers on a shared cache can never
    exec a half-written or unverified binary. Raises SystemExit if rclone is
    neither present nor obtainable-and-verified, so preflight aborts early.
    """
    global _RCLONE_PATH
    if _RCLONE_PATH:
        return _RCLONE_PATH
    found = shutil.which("rclone")
    if found:
        _RCLONE_PATH = found
        return found
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "synoptic"
    dest = cache / f"rclone-{RCLONE_VERSION}"
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64",
            "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if arch is None or arch not in RCLONE_SHA256:
        raise SystemExit(
            f"no rclone on PATH and no pinned download for arch {machine!r}; "
            "install rclone on this host or bake it into the worker image"
        )
    if dest.is_file() and os.access(dest, os.X_OK) and \
            sha256_file(dest) == RCLONE_SHA256[arch]:
        _RCLONE_PATH = str(dest)
        return _RCLONE_PATH
    cache.mkdir(parents=True, exist_ok=True)
    url = (f"https://downloads.rclone.org/{RCLONE_VERSION}/"
           f"rclone-{RCLONE_VERSION}-linux-{arch}.zip")
    print(f"  rclone not on PATH; downloading pinned {RCLONE_VERSION} ({url}) ...")
    try:
        with tempfile.TemporaryDirectory(dir=cache) as td:
            tdp = Path(td)
            zpath = tdp / "rclone.zip"
            with urlopen(url, timeout=120) as r, open(zpath, "wb") as f:
                shutil.copyfileobj(r, f)
            with zipfile.ZipFile(zpath) as zf:
                member = next(n for n in zf.namelist() if n.endswith("/rclone"))
                tmp_bin = tdp / "rclone"
                with zf.open(member) as src, open(tmp_bin, "wb") as out:
                    shutil.copyfileobj(src, out)
            got = sha256_file(tmp_bin)
            if got != RCLONE_SHA256[arch]:
                raise SystemExit(
                    f"downloaded rclone failed its SHA-256 check "
                    f"(expected {RCLONE_SHA256[arch]}, got {got}); refusing to "
                    "execute an unverified binary"
                )
            tmp_bin.chmod(0o755)
            os.replace(tmp_bin, dest)  # atomic; concurrent writers can't tear it
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"could not obtain rclone ({e}); install it on this host or bake it "
            "into the worker image so weights can be stored"
        )
    _RCLONE_PATH = str(dest)
    return _RCLONE_PATH


def _remote(prefix: str = "") -> str:
    """On-the-fly ``:s3:`` remote path for a store prefix (config comes from env)."""
    return f":s3:{BUCKET}/{prefix}"


def _rclone_once(args: list[str], env: dict, input: bytes | None = None
                 ) -> subprocess.CompletedProcess:
    """One rclone invocation against the store (binary + STORE_FLAGS), captured."""
    cmd = [_ensure_rclone(), *args, *STORE_FLAGS]
    return subprocess.run(cmd, env=env, input=input, capture_output=True)


def _decode(proc: subprocess.CompletedProcess) -> str:
    """rclone's stderr (or stdout) as a stripped string for messages."""
    return (proc.stderr or proc.stdout or b"").decode(errors="replace").strip()


def _rclone(args: list[str], env: dict, what: str, *, input: bytes | None = None,
            attempts: int = 4, tolerate: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    """Run rclone with jittered-backoff retries; raise SystemExit on give-up.

    rclone retries individual transfers internally (STORE_FLAGS); this wraps the
    whole invocation so a wholesale failure (endpoint down for a stretch) backs
    off instead of failing the run on the first blip. ``tolerate`` lets a caller
    accept a specific non-zero outcome (e.g. purge of an absent prefix) without
    retrying or raising.
    """
    last = ""
    for attempt in range(1, attempts + 1):
        proc = _rclone_once(args, env, input=input)
        if proc.returncode == 0:
            return proc
        last = _decode(proc)
        if tolerate and any(s in last.lower() for s in tolerate):
            return proc
        print(f"  WARNING: {what} attempt {attempt}/{attempts} failed: {last}")
        if attempt < attempts:
            # Jitter so a shared-endpoint outage does not make the whole fleet
            # retry in lockstep (thundering herd on recovery).
            time.sleep(20 * attempt + random.uniform(0, 10))
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


def _write_file_list(paths: list[str], directory: Path) -> Path:
    """Write an rclone --files-from list (one path per line) into ``directory``."""
    listfile = directory / "_files_from.txt"
    listfile.write_text("\n".join(paths) + "\n", encoding="utf-8")
    return listfile


def preflight(repo: str, run: str) -> None:
    """Write/read/delete-probe the store before training; raise loudly if unusable.

    Catches missing credentials, a missing/unverifiable rclone, an unreachable
    endpoint, or a bad ``--add-host`` mapping in seconds, instead of after
    GPU-hours followed by a silent failure to store the model. The read-back
    catches a store that accepts writes but cannot serve reads.
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
            _rclone(["copyto", str(probe), _remote(key)], env,
                    "store preflight write", attempts=2)
            back = _rclone(["cat", _remote(key)], env,
                           "store preflight read-back", attempts=2)
            if back.stdout.strip() != b"ok":
                raise SystemExit(
                    f"store preflight read-back returned {back.stdout!r}, not b'ok'"
                )
            _rclone(["deletefile", _remote(key)], env,
                    "store preflight cleanup", attempts=2)
        except SystemExit as e:
            raise SystemExit(
                f"cannot use the weights store ({MINIO_ENDPOINT}, bucket "
                f"{BUCKET}): {e}. Weights would be lost; aborting before training."
            )
    print(f"  store preflight OK: {BUCKET}/{prefix}")


def upload_run(output: Path, repo: str, run: str) -> None:
    """Upload a run directory, then a manifest LAST; raise if incomplete.

    Purges the destination prefix first (retried, fail-loud) so a re-run cannot
    mix new files with stale objects from a previous attempt, copies exactly the
    manifest's files (``--files-from``), then writes the manifest — its presence
    is the signal that the run is complete, so ``download_run`` refuses a run
    without one.
    """
    env = _rclone_env()
    if env is None:
        raise SystemExit("MINIO_* credentials are not set; cannot store weights")
    prefix = run_prefix(repo, run)

    # Clear any prior attempt's objects. A purge of an absent prefix is fine
    # (first run); any other failure is retried and then fails the run, because
    # a half-cleared prefix could leak stale objects into an otherwise-complete
    # run.
    purge = _rclone(["purge", _remote(prefix)], env, f"clear prefix {prefix}",
                    tolerate=("not found", "no such", "does not exist", "empty"))
    if purge.returncode == 0:
        print(f"  cleared stale objects under {prefix}")

    manifest = build_manifest(output)
    print(f"  uploading {manifest['count']} files "
          f"({manifest['total_bytes'] / 1e6:.0f} MB) to {BUCKET}/{prefix} ...")
    with tempfile.TemporaryDirectory() as td:
        listfile = _write_file_list([e["path"] for e in manifest["files"]], Path(td))
        _rclone(["copy", str(output), _remote(prefix),
                 "--files-from", str(listfile), "--transfers", "4", "--checkers", "8"],
                env,
                f"upload run {run} (run incomplete; scores remain in the console log)")

    # Manifest last, with retries: it is the completeness signal and the single
    # most consequential write, yet the cheapest — it must not be the one op
    # without the backoff every transfer around it gets.
    body = json.dumps(manifest, indent=2).encode("utf-8")
    _rclone(["rcat", _remote(f"{prefix}{MANIFEST_NAME}")], env,
            f"upload run {run} manifest (run marked incomplete)", input=body)
    print(f"  run directory stored: {BUCKET}/{prefix} (browsable at ~/M/{prefix})")


def download_run(repo: str, run: str, out_dir: Path) -> Path:
    """Download a run directory and verify it against its manifest."""
    env = _rclone_env()
    if env is None:
        raise SystemExit("MINIO_* credentials are not set; cannot fetch weights")
    prefix = run_prefix(repo, run)
    # Read the manifest first. rclone cat of a genuinely-absent object exits 0
    # with empty stdout AND empty stderr; a store/credential/network error exits
    # non-zero or leaves an error on stderr. Distinguish the two so an auth or
    # DNS failure is not misreported as "run never uploaded".
    proc = _rclone_once(["cat", _remote(f"{prefix}{MANIFEST_NAME}")], env)
    if not proc.stdout.strip():
        err = _decode(proc)
        if proc.returncode != 0 or err:
            raise SystemExit(
                f"cannot read the run manifest at {BUCKET}/{prefix}{MANIFEST_NAME}: "
                f"{err or 'unknown rclone error'} — a store/credential/network "
                "error, not a missing run"
            )
        raise SystemExit(
            f"no manifest at {BUCKET}/{prefix}{MANIFEST_NAME}: the run is "
            "absent or its upload never completed (an incomplete run has no "
            "manifest by design)"
        )
    try:
        manifest = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"run manifest at {BUCKET}/{prefix}{MANIFEST_NAME} is not valid JSON "
            f"({e}); the store object is corrupt"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{run}: {manifest['count']} files "
          f"({manifest['total_bytes'] / 1e6:.0f} MB) from {BUCKET}/{prefix}")
    # Fetch exactly the manifest's files (--files-from), so stale objects that
    # somehow survive under the prefix are never pulled into the run directory.
    with tempfile.TemporaryDirectory() as td:
        listfile = _write_file_list([e["path"] for e in manifest["files"]], Path(td))
        _rclone(["copy", _remote(prefix), str(out_dir),
                 "--files-from", str(listfile), "--transfers", "4", "--checkers", "8"],
                env, f"download run {run}")
    problems = manifest_problems(out_dir, manifest)
    if problems:
        raise SystemExit(f"downloaded run does not match its manifest: {problems}")
    print(f"OK: {len(manifest['files'])} files verified against the manifest "
          f"in {out_dir}")
    return out_dir
