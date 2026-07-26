import pytest

from synoptic.chunks import join_files, split_file


def test_split_join_roundtrip(tmp_path):
    data = bytes(range(256)) * 4321  # ~1.1 MB, not a multiple of the part size
    src = tmp_path / "archive.zip"
    src.write_bytes(data)
    parts, manifest = split_file(src, tmp_path / "parts", part_size=100_000)
    assert manifest["parts"] == len(parts) == 12
    assert manifest["size"] == len(data)
    out = join_files(parts, manifest, tmp_path / "out" / "archive.zip")
    assert out.read_bytes() == data


def test_join_detects_corrupt_part(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"x" * 50_000)
    parts, manifest = split_file(src, tmp_path / "parts", part_size=20_000)
    parts[1].write_bytes(b"y" * 20_000)
    with pytest.raises(ValueError, match="part 1 checksum"):
        join_files(parts, manifest, tmp_path / "out.bin")


def test_join_detects_missing_part(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"x" * 50_000)
    parts, manifest = split_file(src, tmp_path / "parts", part_size=20_000)
    with pytest.raises(ValueError, match="expected 3 parts"):
        join_files(parts[:-1], manifest, tmp_path / "out.bin")
