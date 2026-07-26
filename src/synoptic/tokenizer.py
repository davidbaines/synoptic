"""SentencePiece tokeniser training (spec.md, "Tokenisation").

BPE with byte fallback; language tags registered as user-defined symbols so
they stay atomic. The training corpus must come from the training split only —
callers pass the text; ``train_tokenizer`` never reads the corpus itself, so
the leakage test in tests/test_tokenizer.py can verify what goes in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import sentencepiece as spm


def train_tokenizer(
    texts: Iterable[str],
    model_prefix: Path,
    tags: list[str],
    vocab_size: int = 32000,
    model_type: str = "bpe",
    pad_id: int = 3,
) -> Path:
    """Train a SentencePiece model; returns the .model path.

    ``pad_id`` enables an explicit ``<pad>`` control symbol (disabled by
    default in SentencePiece) so the seq2seq collator has a padding token;
    ids stay unk=0, bos=1, eos=2, pad=3, then the atomic tags.
    """
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    corpus_path = model_prefix.with_suffix(".txt")
    with corpus_path.open("w", encoding="utf-8", newline="\n") as f:
        for line in texts:
            f.write(line + "\n")
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        byte_fallback=True,
        character_coverage=0.9999,
        user_defined_symbols=tags,
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
        normalization_rule_name="identity",  # we NFC-normalise ourselves
        add_dummy_prefix=False,  # the language tag must be token 0
        pad_id=pad_id,
    )
    return model_prefix.with_suffix(".model")


def load_tokenizer(model_path: Path) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp
