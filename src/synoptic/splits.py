"""Train/valid/test splits over (vref, translation) pairs.

Holdouts are whole books per translation, plus (new in this series) optional
verse-level holdout sets — fixed vref lists like the committed Genesis-250
test set. Held-out books and verses form the test set,
a small random pair sample forms the validation set, and everything else
trains.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from .canon import NT_BOOKS, OT_BOOKS
from .data import RANGE_MARKER, VREF_COLUMN

PAIR_COLUMNS = [VREF_COLUMN, "translation"]


def expand_books(books: Iterable[str]) -> set[str]:
    """Expand a holdout book list; ``OT``/``NT`` expand to the whole testament."""
    out: set[str] = set()
    for b in books:
        if b == "OT":
            out.update(OT_BOOKS)
        elif b == "NT":
            out.update(NT_BOOKS)
        elif b in OT_BOOKS or b in NT_BOOKS:
            out.add(b)
        else:
            raise ValueError(f"Unknown book code {b!r}")
    return out


@dataclass(frozen=True)
class Splits:
    """Pair manifests; each frame has columns vref, translation, book."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def build_splits(
    verses: pd.DataFrame,
    holdouts: Mapping[str, list[str]],
    valid_size: int = 5000,
    seed: int = 13,
    verse_holdouts: Mapping[str, Iterable[str]] | None = None,
    reserve_per_holdout: int = 0,
) -> Splits:
    """Split a vref-indexed wide verse table into train/valid/test pairs.

    ``holdouts`` maps translationId -> book codes (whole books);
    ``verse_holdouts`` maps translationId -> explicit vrefs (partial-book test
    sets such as Genesis-250). Empty cells and ``<range>`` markers are dropped
    before splitting, so every pair in the result has usable text.

    ``reserve_per_holdout`` guarantees the per-target validation set is
    satisfiable. The validation set (``validation.build_validation_set``) draws
    ``verses_per_language`` verses per *held-out* translation from the valid
    split; a purely global ``valid_size`` sample under-represents those targets
    whenever the pool is dominated by fully-present companion translations (a
    14-translation family with 2 held-out targets gives each target ~1.8% of the
    sample, far below 250). When >0, that many of each holdout translation's
    non-test verses are reserved into the valid split first; the remaining
    ``valid_size`` (for logged eval loss) is then drawn from what is left. The
    default 0 keeps the original global-only behaviour for callers that do not
    build a per-target validation set.

    Holdout keys and vrefs are validated: an unknown translation id, a vref
    absent from the corpus index, or a verse holdout that matches nothing
    raises instead of silently leaving test verses in training.
    """
    known = set(verses.columns)
    for translation in list(holdouts) + list(verse_holdouts or {}):
        if translation not in known:
            raise ValueError(
                f"Holdout translation {translation!r} is not in the selection "
                f"(check the holdout YAML uses translationIds, not language codes)"
            )
    index = set(verses.index)
    for translation, vrefs in (verse_holdouts or {}).items():
        unknown = [v for v in vrefs if v not in index]
        if unknown:
            raise ValueError(
                f"Verse holdout for {translation}: {len(unknown)} vref(s) not in "
                f"the corpus index, e.g. {unknown[:3]}"
            )

    long = verses.reset_index().melt(
        id_vars=VREF_COLUMN, var_name="translation", value_name="text"
    )
    long = long[long["text"].notna() & (long["text"] != "") & (long["text"] != RANGE_MARKER)]
    long = long.drop(columns="text")
    long["book"] = long[VREF_COLUMN].str.split(" ").str[0]

    test_mask = pd.Series(False, index=long.index)
    for translation, books in holdouts.items():
        wanted = expand_books(books)
        test_mask |= (long["translation"] == translation) & long["book"].isin(wanted)
    for translation, vrefs in (verse_holdouts or {}).items():
        vref_set = set(vrefs)
        mask = (long["translation"] == translation) & long[VREF_COLUMN].isin(vref_set)
        matched = int(mask.sum())
        if matched == 0:
            raise ValueError(
                f"Verse holdout for {translation} matched no usable verses "
                f"({len(vref_set)} vrefs listed) — the test set would be empty "
                f"and its verses would train"
            )
        print(f"  verse holdout {translation}: {matched}/{len(vref_set)} vrefs "
              f"held out (the rest are empty/range cells)")
        test_mask |= mask

    test = long[test_mask].reset_index(drop=True)
    rest = long[~test_mask]

    # Reserve per-target validation verses first (see reserve_per_holdout in the
    # docstring), so a global sample can never starve a held-out target.
    reserved = rest.iloc[0:0]
    if reserve_per_holdout > 0:
        holdout_ids = set(holdouts) | set(verse_holdouts or {})
        parts = []
        for translation in sorted(holdout_ids):
            sub = rest[rest["translation"] == translation]
            n = min(reserve_per_holdout, len(sub))
            if n:
                parts.append(sub.sample(n=n, random_state=seed))
        if parts:
            reserved = pd.concat(parts)

    remainder = rest.drop(reserved.index)
    if valid_size >= len(remainder):
        raise ValueError(
            f"valid_size {valid_size} would consume all {len(remainder)} training "
            f"pairs (after reserving {len(reserved)} per-target validation verses)"
        )
    valid = pd.concat([reserved, remainder.sample(n=valid_size, random_state=seed)])
    train = rest.drop(valid.index).reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    return Splits(train=train, valid=valid, test=test)


def manifest_checksum(pairs: pd.DataFrame) -> str:
    """Order-independent sha256 of a pair manifest, for reproducibility checks."""
    lines = sorted(pairs[VREF_COLUMN] + "\t" + pairs["translation"])
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def assert_no_leakage(
    splits: Splits,
    holdouts: Mapping[str, list[str]],
    verse_holdouts: Mapping[str, Iterable[str]] | None = None,
) -> None:
    """Raise AssertionError if any held-out material could reach training.

    Checks that the three splits are pairwise disjoint and that no held-out
    (book, translation) or (vref, translation) pair appears in train or
    valid. Cheap enough to run inside the training script before every run.
    """
    as_set = lambda df: set(zip(df[VREF_COLUMN], df["translation"]))
    train, valid, test = as_set(splits.train), as_set(splits.valid), as_set(splits.test)
    assert not train & test, "train/test overlap"
    assert not valid & test, "valid/test overlap"
    assert not train & valid, "train/valid overlap"
    for translation, books in holdouts.items():
        wanted = expand_books(books)
        for name, df in (("train", splits.train), ("valid", splits.valid)):
            bad = df[(df["translation"] == translation) & df["book"].isin(wanted)]
            assert bad.empty, (
                f"{len(bad)} held-out verses of {translation} leaked into {name}"
            )
    for translation, vrefs in (verse_holdouts or {}).items():
        wanted_vrefs = set(vrefs)
        for name, df in (("train", splits.train), ("valid", splits.valid)):
            bad = df[(df["translation"] == translation) & df[VREF_COLUMN].isin(wanted_vrefs)]
            assert bad.empty, (
                f"{len(bad)} verse-held-out verses of {translation} leaked into {name}"
            )
