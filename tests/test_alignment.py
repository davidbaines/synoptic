"""Integration tests against the real corpus (network; run with -m integration)."""

import pytest

from synoptic.data import load_metadata, load_verses, resolve_column

pytestmark = pytest.mark.integration


def test_metadata_shape():
    meta = load_metadata()
    assert len(meta) == 1253
    assert meta["languageCode"].nunique() == 997


def test_known_verse_text():
    verses = load_verses(["engbsb"])
    assert verses.loc["GEN 1:1", "engbsb"].startswith("In the beginning God created")
    assert len(verses) == 41899


def test_coverage_matches_metadata():
    meta = load_metadata().set_index("translationId")
    verses = load_verses(["engbsb"])
    non_empty = int((verses["engbsb"] != "").sum())
    expected = int(meta.loc["engbsb", "totalVerses"])
    # <range> markers and merged verses allow small drift
    assert abs(non_empty - expected) / expected < 0.02


def test_series_columns_resolve():
    # One source / target id per script pool of this series (spec.md table).
    for tid in ("hin2017", "hne", "arbnav", "ckb", "gmve", "tel2017", "nde"):
        assert resolve_column(tid)


def test_real_holdout_splits_have_no_leakage():
    from synoptic.splits import assert_no_leakage, build_splits

    # A drafting-condition target (whole OT + the NT test-set books) next to
    # an untouched pool member, on real Devanagari texts.
    holdouts = {"hne": ["OT", "MRK", "JAS", "1PE", "2PE"]}
    verses = load_verses(["hin2017", "hne", "mar"])
    splits = build_splits(verses, holdouts, valid_size=100, seed=13)
    assert_no_leakage(splits, holdouts)
    hne_test = splits.test[splits.test["translation"] == "hne"]
    assert len(hne_test) > 20000  # whole OT plus Mark and the epistles

    # The Hindi source must cover the vast majority of the held-out verses.
    covered = (verses.loc[hne_test["vref"], "hin2017"] != "").mean()
    assert covered > 0.95
