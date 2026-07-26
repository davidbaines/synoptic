import pytest

from synoptic.config import ExperimentConfig


def test_load_smoke_config():
    from pathlib import Path

    cfg = ExperimentConfig.load(Path(__file__).parent / "fixtures" / "smoke.yaml")
    assert cfg.name == "smoke"
    assert cfg.data.source == "hin2017"
    assert cfg.data.pairing == "multi-source"
    assert cfg.data.companion_ranking == "coverage"
    assert cfg.model.d_model == 256
    assert cfg.tokenizer.vocab_size == 4000
    assert cfg.training.per_device_batch_size == 64
    assert cfg.inference.beam == 5
    assert cfg.validation is None


def test_unknown_yaml_keys_error(tmp_path):
    # A mistyped or unsupported key must fail before compute is spent, not be
    # silently ignored (code-review finding: `pretrained:` trained from scratch).
    p = tmp_path / "c.yaml"
    p.write_text(
        "name: x\nphase: one-to-many\n"
        "data:\n  selection: a.csv\n  holdouts: b.yaml\n  future_key: 1\n"
        "model:\n  d_model: 128\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="future_key"):
        ExperimentConfig.load(p)


def test_unknown_pairing_errors(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "name: x\nphase: x\n"
        "data:\n  selection: a.csv\n  holdouts: b.yaml\n  pairing: many-to-many\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pairing"):
        ExperimentConfig.load(p)


def test_unknown_companion_ranking_errors(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "name: x\nphase: x\n"
        "data:\n  selection: a.csv\n  holdouts: b.yaml\n  companion_ranking: branchy\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="companion_ranking"):
        ExperimentConfig.load(p)
