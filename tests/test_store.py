from pathlib import Path

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
