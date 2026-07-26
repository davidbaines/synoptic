import pytest

from synoptic.tokenizer import load_tokenizer, train_tokenizer

TAGS = ["<2deu>", "<2fin>", "<2hin>"]
CORPUS = (
    ["<2deu> the quick brown fox jumps over the lazy dog"] * 50
    + ["<2fin> nopea ruskea kettu hyppää laiskan koiran yli"] * 50
    + ["<2hin> तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है"] * 50
)


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("spm") / "test-bpe"
    model = train_tokenizer(CORPUS, prefix, tags=TAGS, vocab_size=400)
    return load_tokenizer(model)


def test_tags_are_atomic(tokenizer):
    for tag in TAGS:
        pieces = tokenizer.encode(f"{tag} hello", out_type=str)
        assert pieces[0] == tag, pieces


def test_round_trip_lossless(tokenizer):
    for text in [
        "<2deu> the quick brown fox",
        "<2hin> तेज़ भूरी लोमड़ी",
        "<2fin> kettu hyppää",
    ]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_fallback_handles_unseen_script(tokenizer):
    # Greek never appeared in training; byte fallback must still round-trip it
    text = "εν αρχη εποιησεν ο θεος"
    assert tokenizer.decode(tokenizer.encode(text)) == text
