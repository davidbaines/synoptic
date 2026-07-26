import pandas as pd
import pytest

from synoptic.config import ModelConfig
from synoptic.dataset import Collator, PairDataset
from synoptic.model import build_model
from synoptic.preprocess import SRC_COLUMN, TGT_COLUMN
from synoptic.tokenizer import load_tokenizer, train_tokenizer

TAGS = ["<2spa>", "<2swh>"]
CORPUS = (
    ["<2spa> en el principio creo dios los cielos y la tierra"] * 50
    + ["<2swh> hapo mwanzo mungu aliumba mbingu na nchi"] * 50
)


@pytest.fixture(scope="module")
def sp(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("spm") / "smoke"
    return load_tokenizer(train_tokenizer(CORPUS, prefix, tags=TAGS, vocab_size=400))


TINY = ModelConfig(
    encoder_layers=2, decoder_layers=2, d_model=64,
    encoder_attention_heads=2, decoder_attention_heads=2,
    encoder_ffn_dim=128, decoder_ffn_dim=128,
)


def test_special_tokens_wired(sp):
    model = build_model(TINY, sp, max_position_embeddings=64)
    assert model.config.pad_token_id == sp.pad_id()
    assert model.config.eos_token_id == sp.eos_id()
    assert model.config.decoder_start_token_id == sp.pad_id()
    assert model.config.vocab_size == sp.get_piece_size()


def test_forward_produces_finite_loss(sp):
    import torch

    model = build_model(TINY, sp, max_position_embeddings=64)
    pairs = pd.DataFrame(
        {
            SRC_COLUMN: ["<2spa> en el principio", "<2swh> hapo mwanzo mungu"],
            TGT_COLUMN: ["en el principio creo", "hapo mwanzo"],
        }
    )
    ds = PairDataset(pairs, sp, max_len=64)
    batch = Collator(pad_id=sp.pad_id())([ds[0], ds[1]])
    out = model(**batch)
    assert torch.isfinite(out.loss)
    assert out.loss.item() > 0
