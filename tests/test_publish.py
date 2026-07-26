import pandas as pd
import pytest

from synoptic.hf_export import build_model_card, package_tokenizer
from synoptic.publish import check_gate, default_repo_id
from synoptic.tokenizer import train_tokenizer

TAGS = ["<2spa>", "<2deu>"]
CORPUS = (
    ["<2spa> en el principio creo dios los cielos y la tierra"] * 50
    + ["<2deu> am anfang schuf gott himmel und erde"] * 50
)


def test_default_repo_id():
    assert default_repo_id("ie_base") == "DavidCBaines/ebible_m2m-ie-base"


def test_gate_passes_when_all_beat_baseline():
    m = pd.DataFrame({"translation": ["a", "b"], "book": ["OT", "OT"],
                      "verses": [100, 100],
                      "chrF3": [20.0, 15.0], "copy_chrF3": [1.0, 2.0]})
    ok, verdict = check_gate(m)
    assert ok
    assert verdict["beats_baseline"].all()


def test_gate_fails_when_one_loses_to_baseline():
    m = pd.DataFrame({"translation": ["a", "b"], "book": ["OT", "OT"],
                      "verses": [100, 100],
                      "chrF3": [20.0, 1.5], "copy_chrF3": [1.0, 2.0]})
    ok, verdict = check_gate(m)
    assert not ok
    assert not verdict["beats_baseline"].all()


def test_gate_fails_when_language_loses_to_other_language_baseline():
    # model beats source-copy everywhere, but loses to a relative on average
    m = pd.DataFrame({
        "translation": ["a", "a"], "book": ["GEN", "EXO"], "verses": [100, 100],
        "chrF3": [20.0, 22.0], "copy_chrF3": [1.0, 1.0],
        "other_chrF3": [25.0, 24.0],
    })
    ok, _ = check_gate(m)
    assert not ok


def test_gate_passes_over_other_language_in_aggregate():
    # loses on one book but wins the verse-weighted language average
    m = pd.DataFrame({
        "translation": ["a", "a"], "book": ["GEN", "EXO"], "verses": [50, 150],
        "chrF3": [24.0, 40.0], "copy_chrF3": [1.0, 1.0],
        "other_chrF3": [25.0, 20.0],
    })
    ok, _ = check_gate(m)
    assert ok


def test_package_tokenizer_roundtrips(tmp_path):
    from transformers import MarianTokenizer

    prefix = tmp_path / "spm"
    train_tokenizer(CORPUS, prefix, tags=TAGS, vocab_size=400)
    out = tmp_path / "tok"
    package_tokenizer(prefix.with_suffix(".model"), out, source_lang="hin")
    tok = MarianTokenizer.from_pretrained(str(out))
    ids = tok("<2spa> en el principio").input_ids
    assert tok.convert_ids_to_tokens(ids)[0] == "<2spa>"      # tag stays atomic
    assert "en el principio" in tok.decode(ids, skip_special_tokens=True)


def test_model_card_has_key_sections():
    lic = pd.DataFrame({"translationId": ["spablm"], "languageCode": ["spa"],
                        "licence": ["Public Domain"]})
    metrics = pd.DataFrame({"translation": ["engbsb"], "language": ["eng"],
                            "book": ["GEN"], "verses": [100],
                            "chrF3": [30.0], "copy_chrF3": [1.0],
                            "other_chrF3": [12.0], "other_lang": ["fra"]})
    card = build_model_card(
        repo_id="DavidCBaines/ebible_m2m-ie-base", experiment="ie_base",
        n_params=61_000_000, licences=lic, holdouts={"engbsb": ["OT"]},
        metrics=metrics, model_licence="cc0-1.0", git_commit="abc123", seed=13,
        source_id="hin2017", source_lang="hin", source_name="Hindi",
    )
    assert "license: cc0-1.0" in card
    assert "from_pretrained" in card
    assert "engbsb" in card
    assert "abc123" in card
