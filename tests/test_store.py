import shutil
import subprocess
from pathlib import Path

import pytest

from synoptic import store


def test_normalize_repo_strips_git():
    assert store.normalize_repo("bible-mt-same-script.git") == "bible-mt-same-script"
    assert store.normalize_repo("bible-mt-same-script") == "bible-mt-same-script"


def test_run_prefix_uses_normalized_repo():
    # A worker's <repo>.git clone and a local <repo> checkout must resolve to
    # the same store prefix (the upload/fetch mismatch the review flagged).
    assert store.run_prefix("x.git", "r") == store.run_prefix("x", "r")
    assert store.run_prefix("x", "r") == "MT/experiments/synoptic/x/r/"


def _make_run(tmp_path: Path) -> Path:
    out = tmp_path / "run"
    (out / "tokenizer").mkdir(parents=True)
    (out / "config.json").write_text("{}")
    (out / "model.safetensors").write_bytes(b"weights" * 1000)
    (out / "tokenizer" / "spm.model").write_bytes(b"sp")
    # excluded: trainer checkpoints and the best/ duplicate
    (out / "checkpoint-500").mkdir()
    (out / "checkpoint-500" / "optimizer.pt").write_bytes(b"opt")
    (out / "best").mkdir()
    (out / "best" / "model.safetensors").write_bytes(b"dup")
    return out


def test_run_files_excludes_checkpoints_and_best(tmp_path):
    out = _make_run(tmp_path)
    rels = {str(f.relative_to(out)) for f in store.run_files(out)}
    assert rels == {"config.json", "model.safetensors", "tokenizer/spm.model"}


def test_build_manifest_counts_and_hashes(tmp_path):
    out = _make_run(tmp_path)
    m = store.build_manifest(out)
    assert m["count"] == 3
    assert m["total_bytes"] == sum(e["size"] for e in m["files"])
    assert all(len(e["sha256"]) == 64 for e in m["files"])


def test_manifest_problems_detects_missing_and_corrupt(tmp_path):
    out = _make_run(tmp_path)
    m = store.build_manifest(out)

    # A faithful copy passes.
    dest = tmp_path / "dl"
    for e in m["files"]:
        t = dest / e["path"]
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes((out / e["path"]).read_bytes())
    assert store.manifest_problems(dest, m) == []

    # Missing weights (the exact partial-upload blind spot) is caught.
    (dest / "model.safetensors").unlink()
    assert any("model.safetensors" in p for p in store.manifest_problems(dest, m))

    # A corrupt file (right size, wrong bytes) is caught by checksum.
    good = out / "config.json"
    (dest / "model.safetensors").write_bytes((out / "model.safetensors").read_bytes())
    (dest / "config.json").write_text("x" * len(good.read_text()))
    assert any("config.json" in p for p in store.manifest_problems(dest, m))


def test_rclone_env_none_without_credentials(monkeypatch):
    monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
    assert store._rclone_env() is None


def test_rclone_env_configures_s3_backend_from_minio_creds(monkeypatch):
    monkeypatch.setenv("MINIO_ACCESS_KEY", "AK")
    monkeypatch.setenv("MINIO_SECRET_KEY", "SK")
    env = store._rclone_env()
    # Secrets travel in the child env (RCLONE_S3_*), never on the argv.
    assert env["RCLONE_S3_ACCESS_KEY_ID"] == "AK"
    assert env["RCLONE_S3_SECRET_ACCESS_KEY"] == "SK"
    assert env["RCLONE_S3_PROVIDER"] == "Minio"
    assert env["RCLONE_S3_ENDPOINT"] == store.MINIO_ENDPOINT


def test_remote_addresses_bucket_and_prefix():
    # On-the-fly :s3: remote (backend config comes from the env, not the path).
    assert store._remote("MT/x/r/") == ":s3:nlp-research/MT/x/r/"
    assert store._remote() == ":s3:nlp-research/"


def test_ensure_rclone_prefers_path(monkeypatch):
    monkeypatch.setattr(store, "_RCLONE_PATH", None)  # bypass the process cache
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rclone")
    assert store._ensure_rclone() == "/usr/bin/rclone"


def test_pinned_rclone_hashes_present_for_common_arches():
    # The runtime fetch verifies against these; a missing arch means we would
    # refuse to install rather than run an unverified binary.
    assert set(store.RCLONE_SHA256) >= {"amd64", "arm64"}
    assert all(len(h) == 64 for h in store.RCLONE_SHA256.values())


def test_write_file_list_one_path_per_line(tmp_path):
    lf = store._write_file_list(["a.txt", "sub/b.bin"], tmp_path)
    assert lf.read_text().splitlines() == ["a.txt", "sub/b.bin"]


def _proc(rc, out=b"", err=b""):
    return subprocess.CompletedProcess(args=["rclone"], returncode=rc,
                                       stdout=out, stderr=err)


def test_download_run_reports_absent_run_cleanly(monkeypatch, tmp_path):
    # rclone cat of a genuinely-absent manifest: rc 0, empty stdout AND stderr.
    monkeypatch.setattr(store, "_rclone_env", lambda: {})
    monkeypatch.setattr(store, "_rclone_once", lambda *a, **k: _proc(0, b"", b""))
    with pytest.raises(SystemExit) as ei:
        store.download_run("repo", "run", tmp_path)
    assert "absent" in str(ei.value) or "never completed" in str(ei.value)


def test_download_run_distinguishes_store_error_from_absent(monkeypatch, tmp_path):
    # A 403/auth or DNS error must NOT be misreported as a missing run. The
    # manifest read retries (via _rclone), so stub the backoff sleep away.
    monkeypatch.setattr(store, "_rclone_env", lambda: {})
    monkeypatch.setattr(store.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        store, "_rclone_once",
        lambda *a, **k: _proc(1, b"", b"CRITICAL: api error Forbidden: Forbidden"))
    with pytest.raises(SystemExit) as ei:
        store.download_run("repo", "run", tmp_path)
    msg = str(ei.value)
    assert "store/credential/network error" in msg and "Forbidden" in msg


def test_download_run_rejects_corrupt_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_rclone_env", lambda: {})
    monkeypatch.setattr(store, "_rclone_once", lambda *a, **k: _proc(0, b"not json{", b""))
    with pytest.raises(SystemExit) as ei:
        store.download_run("repo", "run", tmp_path)
    assert "not valid JSON" in str(ei.value)


def test_run_files_keeps_best_prefixed_files_but_drops_best_dir(tmp_path):
    # 'best' is skipped as an EXACT path component (the best/ duplicate), not as
    # a prefix — a real artifact like best_metrics.json must be kept, else it is
    # silently dropped from both the manifest and the --files-from upload.
    out = tmp_path / "run"
    (out / "best").mkdir(parents=True)
    (out / "best" / "model.safetensors").write_bytes(b"dup")   # dropped: best/ dir
    (out / "checkpoint-9").mkdir()
    (out / "checkpoint-9" / "opt.pt").write_bytes(b"o")        # dropped: checkpoint dir
    (out / "best_metrics.json").write_text("{}")               # KEPT: not exactly 'best'
    (out / "model.safetensors").write_bytes(b"w")
    rels = {str(f.relative_to(out)) for f in store.run_files(out)}
    assert rels == {"best_metrics.json", "model.safetensors"}


def _recording_rclone(out_dir, calls):
    """A fake store._rclone that records subcommands and satisfies the lsf probe."""
    def fake(args, env, what, *, input=None, attempts=4, tolerate=()):
        calls.append(args[0])
        stdout = b""
        if args[0] == "lsf":  # completeness probe: report every stored file present
            rels = [str(f.relative_to(out_dir)) for f in store.run_files(out_dir)]
            stdout = ("\n".join(rels)).encode()
        return _proc(0, stdout, b"")
    return fake


def test_upload_run_writes_manifest_last(tmp_path, monkeypatch):
    out = tmp_path / "run"
    (out / "tokenizer").mkdir(parents=True)
    (out / "config.json").write_text("{}")
    (out / "model.safetensors").write_bytes(b"w" * 20)
    (out / "tokenizer" / "spm.model").write_bytes(b"sp")
    calls: list[str] = []
    monkeypatch.setattr(store, "_rclone_env", lambda: {})
    monkeypatch.setattr(store, "_rclone", _recording_rclone(out, calls))
    store.upload_run(out, "repo", "run")
    assert calls[0] == "purge"                       # destination cleared first
    assert "copy" in calls and "lsf" in calls
    assert calls[-1] == "rcat"                       # manifest is the LAST write
    assert calls.index("copy") < calls.index("rcat")
    assert calls.index("lsf") < calls.index("rcat")  # completeness verified before manifest


def test_upload_run_fails_loud_when_a_file_missing_from_store(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    (out / "config.json").write_text("{}")
    (out / "model.safetensors").write_bytes(b"w" * 20)
    calls: list[str] = []

    def fake(args, env, what, *, input=None, attempts=4, tolerate=()):
        calls.append(args[0])
        stdout = b""
        if args[0] == "lsf":  # copy "succeeded" but a file never reached the store
            stdout = b"config.json"
        return _proc(0, stdout, b"")

    monkeypatch.setattr(store, "_rclone_env", lambda: {})
    monkeypatch.setattr(store, "_rclone", fake)
    with pytest.raises(SystemExit) as ei:
        store.upload_run(out, "repo", "run")
    assert "incomplete" in str(ei.value)
    assert "rcat" not in calls  # manifest must NOT be written for an incomplete upload
