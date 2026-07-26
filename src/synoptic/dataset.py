"""Torch datasets and a padding collator over preprocessed verse pairs.

The tokeniser is a raw SentencePiece processor (see ``tokenizer.py``); rather
than wrap it in a HF tokenizer class, we tokenise here and let a small collator
pad batches. Source text already carries the target-language tag as token 0
(``preprocess.build_pairs``); EOS is appended to both sides. Labels use -100 for
padding so they are ignored by the loss.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from .preprocess import SRC_COLUMN, TGT_COLUMN

LABEL_PAD_ID = -100


class PairDataset(Dataset):
    """Tokenised (source, target) verse pairs.

    Each item is a dict of python int lists: ``input_ids`` (tagged source +
    EOS) and ``labels`` (target + EOS). Padding is deferred to the collator.
    """

    def __init__(self, pairs, sp, max_len: int = 192, max_src_len: int | None = None):
        self.src = pairs[SRC_COLUMN].tolist()
        self.tgt = pairs[TGT_COLUMN].tolist()
        self.sp = sp
        self.max_len = max_len
        # Multi-source concatenations need a larger source-side cap; the
        # target side keeps max_len (plan.md, "Multi-source format").
        self.max_src_len = max_src_len or max_len
        self.eos = sp.eos_id()

    def __len__(self) -> int:
        return len(self.src)

    def _encode(self, text: str, cap: int) -> list[int]:
        ids = self.sp.encode(text, out_type=int)[: cap - 1]
        ids.append(self.eos)
        return ids

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return {
            "input_ids": self._encode(self.src[idx], self.max_src_len),
            "labels": self._encode(self.tgt[idx], self.max_len),
        }


class Collator:
    """Pad a batch of {input_ids, labels} to rectangular tensors.

    Also emits ``decoder_input_ids`` (labels shifted right, starting from
    ``decoder_start_id``). The Trainer's label-smoothing path pops ``labels``
    before the model runs, so the model cannot build the decoder inputs itself
    — the collator must provide them. ``decoder_start_id`` defaults to the pad
    id, matching Marian's convention.
    """

    def __init__(self, pad_id: int, decoder_start_id: int | None = None):
        self.pad_id = pad_id
        self.decoder_start_id = pad_id if decoder_start_id is None else decoder_start_id

    def __call__(self, batch: Sequence[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        src_max = max(len(b["input_ids"]) for b in batch)
        tgt_max = max(len(b["labels"]) for b in batch)
        input_ids, attention_mask, labels, decoder_input_ids = [], [], [], []
        for b in batch:
            s, t = b["input_ids"], b["labels"]
            input_ids.append(s + [self.pad_id] * (src_max - len(s)))
            attention_mask.append([1] * len(s) + [0] * (src_max - len(s)))
            labels.append(t + [LABEL_PAD_ID] * (tgt_max - len(t)))
            shifted = ([self.decoder_start_id] + t)[:tgt_max]
            decoder_input_ids.append(shifted + [self.pad_id] * (tgt_max - len(shifted)))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
        }
