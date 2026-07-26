"""Verse-pair preprocessing (spec.md, "Preprocessing").

Native scripts, case preserved, Unicode NFC only. One training example is one
verse: the source text with the target-language tag prepended, paired with the
target text. Length filtering happens after tokenisation, so it lives here as
a separate step taking an encode function.
"""

from __future__ import annotations

import unicodedata
from typing import Callable, Mapping

import pandas as pd

from .data import RANGE_MARKER, VREF_COLUMN

SRC_COLUMN = "src"
TGT_COLUMN = "tgt"


def normalise(text: str) -> str:
    """NFC-normalise and collapse internal whitespace."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def target_tag(language_code: str) -> str:
    return f"<2{language_code}>"


def source_tag(language_code: str) -> str:
    """Source-language tag for many-to-many training (spec.md phase 4).

    One-to-many uses only the target tag; many-to-many prepends the source tag
    as well, so the model knows both which language it is reading and which it
    should produce. Kept distinct from the target tag (`<1..>` vs `<2..>`).
    """
    return f"<1{language_code}>"


def build_pairs(
    manifest: pd.DataFrame,
    verses: pd.DataFrame,
    source: pd.Series,
    language_of: Mapping[str, str],
) -> pd.DataFrame:
    """Join a split manifest with source and target text.

    ``manifest`` has columns vref, translation (a Splits frame); ``verses`` is
    the vref-indexed wide table; ``source`` the source translation's Series.
    Pairs whose source text is empty or a ``<range>`` marker are dropped —
    neither can be translated from. Returns columns: vref, translation, src
    (tagged), tgt.
    """
    frame = manifest.copy()
    frame[TGT_COLUMN] = [
        verses.at[v, t] for v, t in zip(frame[VREF_COLUMN], frame["translation"])
    ]
    frame["source_text"] = source.reindex(frame[VREF_COLUMN]).fillna("").to_numpy()
    frame = frame[
        (frame["source_text"] != "")
        & (frame["source_text"] != RANGE_MARKER)
        & (frame[TGT_COLUMN] != "")
    ]

    tags = frame["translation"].map(lambda t: target_tag(language_of[t]))
    frame[SRC_COLUMN] = tags + " " + frame["source_text"].map(normalise)
    frame[TGT_COLUMN] = frame[TGT_COLUMN].map(normalise)
    return frame[[VREF_COLUMN, "translation", SRC_COLUMN, TGT_COLUMN]].reset_index(
        drop=True
    )


def build_target_frame(
    manifest: pd.DataFrame,
    verses: pd.DataFrame,
    language_of: Mapping[str, str],
) -> pd.DataFrame:
    """Target-side pairs with a tag-only src placeholder (multi-source eval).

    Multi-source valid/test frames must keep every manifest row regardless of
    the forced source's coverage — the renderings are filled in later by
    ``to_ms_sources`` from whichever pool members cover each verse. The
    manifest guarantees non-empty, non-range target text.
    """
    frame = manifest.copy()
    frame[TGT_COLUMN] = [
        normalise(verses.at[v, t])
        for v, t in zip(frame[VREF_COLUMN], frame["translation"])
    ]
    frame[SRC_COLUMN] = frame["translation"].map(
        lambda t: target_tag(language_of[t])
    )
    return frame[[VREF_COLUMN, "translation", SRC_COLUMN, TGT_COLUMN]].reset_index(
        drop=True
    )


def length_filter(
    pairs: pd.DataFrame,
    encode: Callable[[str], list],
    max_len: int = 192,
    max_ratio: float = 2.0,
    max_src_len: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop pairs that are too long or too unbalanced after tokenisation.

    Returns the kept pairs and counts of what was dropped, so pipelines can
    log truncation instead of silently shrinking the data (spec.md,
    "no silent caps"). ``max_ratio`` 0 (or None) disables the ratio filter —
    a vref source is always a few tokens, so every pair would fail it
    (spec-vref.md, "Data"). ``max_src_len`` caps the source side separately
    (multi-source concatenations are much longer than any target).
    """
    src_len = pairs[SRC_COLUMN].map(lambda s: len(encode(s)))
    tgt_len = pairs[TGT_COLUMN].map(lambda s: len(encode(s)))
    too_long = (src_len > (max_src_len or max_len)) | (tgt_len > max_len)
    if max_ratio:
        ratio = pd.concat([src_len / tgt_len, tgt_len / src_len], axis=1).max(axis=1)
        bad_ratio = ratio > max_ratio
    else:
        bad_ratio = pd.Series(False, index=pairs.index)
    kept = pairs[~too_long & ~bad_ratio].reset_index(drop=True)
    stats = {
        "input": len(pairs),
        "dropped_too_long": int(too_long.sum()),
        "dropped_ratio": int((bad_ratio & ~too_long).sum()),
        "kept": len(kept),
    }
    return kept, stats
