"""Tests for the 2026-07-25 code-review fixes
(experiments/code-review-findings.md)."""

import pandas as pd
import pytest

from synoptic.data import RANGE_MARKER, VREF_COLUMN
from synoptic.multisource import to_ms_sources
from synoptic.preprocess import SRC_COLUMN, TGT_COLUMN, build_pairs, build_target_frame
from synoptic.splits import build_splits


def _verses():
    vrefs = ["GEN 1:1", "GEN 1:2", "GEN 1:3", "MRK 1:1"]
    return pd.DataFrame(
        {
            "eng": ["e1", "e2", "e3", "e4"],
            "deu": ["d1", "", RANGE_MARKER, "d4"],
            "hin": ["h1", "h2", "", "h4"],
        },
        index=pd.Index(vrefs, name=VREF_COLUMN),
    )


# --- finding 1.4: verse-holdout validation -------------------------------

def test_unknown_holdout_translation_errors():
    with pytest.raises(ValueError, match="translationIds"):
        build_splits(_verses(), {"nope": ["GEN"]}, valid_size=2)


def test_unknown_verse_holdout_vref_errors():
    with pytest.raises(ValueError, match="not in"):
        build_splits(
            _verses(), {}, valid_size=2,
            verse_holdouts={"eng": ["GEN 99:99"]},
        )


def test_zero_match_verse_holdout_errors():
    # deu is empty/range at GEN 1:2-3: a Genesis-250-style list matching no
    # usable verses must fail, not silently leave the verses in training.
    with pytest.raises(ValueError, match="matched no usable verses"):
        build_splits(
            _verses(), {}, valid_size=2,
            verse_holdouts={"deu": ["GEN 1:2", "GEN 1:3"]},
        )


# --- finding 1.3: <range> markers never feed the source side --------------

def test_build_pairs_drops_range_marker_sources():
    verses = _verses()
    manifest = pd.DataFrame(
        {VREF_COLUMN: ["GEN 1:1", "GEN 1:3", "MRK 1:1"], "translation": ["eng"] * 3}
    )
    out = build_pairs(manifest, verses, verses["deu"], {"eng": "eng"})
    # GEN 1:3 has a <range> source cell: dropped, like the empty GEN 1:2
    assert out[VREF_COLUMN].tolist() == ["GEN 1:1", "MRK 1:1"]
    assert not out[SRC_COLUMN].str.contains(RANGE_MARKER).any()


# --- finding 1.2: eval sets must not shrink to the source's coverage ------

def test_build_target_frame_keeps_all_manifest_rows():
    verses = _verses()
    manifest = pd.DataFrame(
        {VREF_COLUMN: ["GEN 1:1", "GEN 1:2", "GEN 1:3"], "translation": ["eng"] * 3}
    )
    out = build_target_frame(manifest, verses, {"eng": "eng"})
    # deu (a would-be source) is empty/range at GEN 1:2-3 — irrelevant here
    assert len(out) == 3
    assert out[SRC_COLUMN].tolist() == ["<2eng>"] * 3
    assert out[TGT_COLUMN].tolist() == ["e1", "e2", "e3"]


def test_to_ms_sources_drops_only_uncovered_verses():
    verses = _verses()
    lang = {c: c for c in verses.columns}
    frame = build_target_frame(
        pd.DataFrame(
            {VREF_COLUMN: ["GEN 1:1", "GEN 1:2", "GEN 1:3"], "translation": ["eng"] * 3}
        ),
        verses, lang,
    )
    # hin covers GEN 1:1-2 but not 1:3; deu covers only 1:1; eng is the target
    present = {"GEN 1:1": ["deu", "hin"], "GEN 1:2": ["hin"]}
    ranking = {"eng": ["hin", "deu"]}
    out = to_ms_sources(frame, verses, lang, present, ranking, k=2, source_id="deu")
    # GEN 1:3 has no rendering anywhere -> dropped; the others survive with
    # whatever the pool covers (not just the forced source deu)
    assert out[VREF_COLUMN].tolist() == ["GEN 1:1", "GEN 1:2"]
    assert out[SRC_COLUMN].tolist()[0].startswith("<2eng> <1deu> d1")
    assert "<1hin> h2" in out[SRC_COLUMN].tolist()[1]


# --- finding 1.5: source-side leakage is asserted, not assumed ------------

def test_to_ms_sources_asserts_forbidden_cells():
    verses = _verses()
    lang = {c: c for c in verses.columns}
    frame = build_target_frame(
        pd.DataFrame({VREF_COLUMN: ["GEN 1:1"], "translation": ["eng"]}),
        verses, lang,
    )
    # A (bugged) present map offering a held-out cell must trip the assertion
    present = {"GEN 1:1": ["hin"]}
    ranking = {"eng": ["hin"]}
    with pytest.raises(AssertionError, match="held-out cell"):
        to_ms_sources(
            frame, verses, lang, present, ranking, k=2,
            forbidden={("GEN 1:1", "hin")},
        )
