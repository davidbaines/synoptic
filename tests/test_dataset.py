import pandas as pd
import pytest

from synoptic.dataset import LABEL_PAD_ID, Collator, PairDataset
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
    model = train_tokenizer(CORPUS, prefix, tags=TAGS, vocab_size=400)
    return load_tokenizer(model)


@pytest.fixture
def pairs():
    return pd.DataFrame(
        {
            SRC_COLUMN: ["<2spa> en el principio", "<2swh> hapo mwanzo mungu aliumba"],
            TGT_COLUMN: ["en el principio creo dios", "hapo mwanzo"],
        }
    )


def test_pad_token_enabled(sp):
    assert sp.pad_id() == 3
    assert sp.eos_id() == 2


def test_items_end_with_eos(sp, pairs):
    ds = PairDataset(pairs, sp, max_len=64)
    item = ds[0]
    assert item["input_ids"][-1] == sp.eos_id()
    assert item["labels"][-1] == sp.eos_id()


def test_collator_pads_and_masks(sp, pairs):
    ds = PairDataset(pairs, sp, max_len=64)
    batch = Collator(pad_id=sp.pad_id())([ds[0], ds[1]])
    b, s = batch["input_ids"].shape
    assert b == 2
    # padded rows: attention_mask 0 exactly where input is pad
    assert (batch["input_ids"][batch["attention_mask"] == 0] == sp.pad_id()).all()
    # label padding is the ignore index, never the real pad id
    assert (batch["labels"] == LABEL_PAD_ID).any()
    assert (batch["labels"] != sp.pad_id()).all() or LABEL_PAD_ID != sp.pad_id()


def test_max_len_truncation(sp):
    long = " ".join(["palabra"] * 200)
    pairs = pd.DataFrame({SRC_COLUMN: [f"<2spa> {long}"], TGT_COLUMN: [long]})
    ds = PairDataset(pairs, sp, max_len=16)
    assert len(ds[0]["input_ids"]) == 16
    assert ds[0]["input_ids"][-1] == sp.eos_id()
