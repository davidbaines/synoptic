"""Shared data assembly for training and generation.

Both the training script and the generation utility need the exact same
train/valid/test pairs for a given experiment config, built the same way, so
that generation scores the held-out material the model was actually kept away
from. This module is that single source of truth.

Unlike bible-interlingua there is no composite Greek: ``cfg.data.source`` is
always a translation id — the alignment-chosen in-script source (spec.md,
"Source"). Holdout YAMLs may carry a ``verse_holdouts`` section mapping a
translation to committed vref-list files (e.g. Genesis-250).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import yaml

from .config import ExperimentConfig
from .data import load_verses, repo_root
from .preprocess import build_pairs, build_target_frame
from .splits import Splits, build_splits


@dataclass
class PreparedData:
    selection: pd.DataFrame
    verses: pd.DataFrame
    source: pd.Series
    source_id: str
    language_of: dict[str, str]
    holdouts: dict[str, list[str]]
    splits: Splits
    train_pairs: pd.DataFrame
    valid_pairs: pd.DataFrame
    test_pairs: pd.DataFrame
    verse_holdouts: dict[str, list[str]] = field(default_factory=dict)

    @property
    def holdout_translations(self) -> list[str]:
        """Every translation with held-out material (book- or verse-level)."""
        return sorted(set(self.holdouts) | set(self.verse_holdouts))


def load_vref_list(path) -> list[str]:
    """Read a committed vref-list file (one vref per line, # comments)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def load_holdouts(
    cfg: ExperimentConfig,
) -> tuple[dict[str, list[str]], dict[str, list[str]], int, int]:
    raw = yaml.safe_load(cfg.resolve(cfg.data.holdouts).read_text(encoding="utf-8"))
    holdouts = {k: list(v) for k, v in raw["holdouts"].items()}
    verse_holdouts: dict[str, list[str]] = {}
    for translation, files in (raw.get("verse_holdouts") or {}).items():
        vrefs: list[str] = []
        for f in files:
            vrefs.extend(load_vref_list(repo_root() / f))
        verse_holdouts[translation] = vrefs
    return holdouts, verse_holdouts, int(raw.get("valid_size", 5000)), int(raw.get("seed", 13))


def apply_nt_truncation(verses: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """Blank non-NT cells of every selection member marked ``ntOnly``.

    The Latin control truncates non-source, non-target members to their NT so
    the pool structure mirrors Devanagari (spec.md, "Scripts and pools").
    Applied before splitting, so truncated cells can never train, validate,
    or feed the source side.
    """
    if "ntOnly" not in selection.columns:
        return verses
    from .canon import NT_BOOKS

    nt_only = selection.loc[
        selection["ntOnly"].astype(str).str.lower() == "true", "translationId"
    ]
    if not len(nt_only):
        return verses
    verses = verses.copy()
    outside_nt = ~verses.index.str.split(" ").str[0].isin(NT_BOOKS)
    for tid in nt_only:
        verses.loc[outside_nt, tid] = ""
    return verses


def prepare(cfg: ExperimentConfig) -> PreparedData:
    """Assemble selection, source, splits and text pairs for ``cfg``."""
    selection = pd.read_csv(cfg.resolve(cfg.data.selection), dtype=str)
    target_ids = selection["translationId"].tolist()
    language_of = dict(zip(selection["translationId"], selection["languageCode"]))

    verses = apply_nt_truncation(load_verses(target_ids), selection)
    source_id = cfg.data.source
    if source_id in verses.columns:
        source = verses[source_id]
    else:
        # A source outside the selection (single-source baselines exclude it
        # from the target pool so no identity pairs are built).
        source = load_verses([source_id])[source_id]

    holdouts, verse_holdouts, valid_size, seed = load_holdouts(cfg)
    if source_id in holdouts or source_id in verse_holdouts:
        # A held-out source is a config error: its held-out text would feed
        # the source side of every pair at those vrefs (one-to-many builds
        # pairs straight from the raw source column).
        raise ValueError(
            f"The source translation {source_id!r} appears in the holdouts — "
            f"a held-out source would leak its test text into every pair"
        )
    splits = build_splits(
        verses, holdouts, valid_size=valid_size, seed=seed,
        verse_holdouts=verse_holdouts,
    )
    if cfg.data.pairing == "one-to-many" and source_id in verses.columns:
        # Single-source baselines with an in-pool source: the source cannot be
        # a target (source->source identity pairs would train copying, and its
        # test rows could only ever be scored against itself).
        drop = lambda df: df[df["translation"] != source_id].reset_index(drop=True)
        splits = Splits(
            train=drop(splits.train), valid=drop(splits.valid), test=drop(splits.test)
        )

    def pairs(frame: pd.DataFrame) -> pd.DataFrame:
        # Multi-source eval frames must not shrink to the forced source's
        # coverage: renderings come from the whole pool, so keep every
        # manifest row (to_ms_sources fills the sources and drops only verses
        # nothing in the pool covers). One-to-many genuinely needs the source
        # text, so build_pairs' source filtering is correct there.
        if cfg.data.pairing == "multi-source":
            return build_target_frame(frame, verses, language_of)
        return build_pairs(frame, verses, source, language_of)

    return PreparedData(
        selection=selection,
        verses=verses,
        source=source,
        source_id=source_id,
        language_of=language_of,
        holdouts=holdouts,
        verse_holdouts=verse_holdouts,
        splits=splits,
        train_pairs=pairs(splits.train),
        valid_pairs=pairs(splits.valid),
        test_pairs=pairs(splits.test),
    )
