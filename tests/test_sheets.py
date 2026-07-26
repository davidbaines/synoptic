import pandas as pd

from synoptic.sheets import make_sheet, passage_vrefs


def _series(values: dict) -> pd.Series:
    return pd.Series(values)


PASSAGES = [{"name": "Genesis 1:1–2", "start": "GEN 1:1", "end": "GEN 1:2"}]


def test_passage_vrefs_inclusive():
    index = pd.Index(["GEN 1:1", "GEN 1:2", "GEN 1:3"])
    assert passage_vrefs(PASSAGES[0], index) == ["GEN 1:1", "GEN 1:2"]


def test_make_sheet_renders_rows():
    hyp = _series({"GEN 1:1": "generated one", "GEN 1:2": "generated | two"})
    ref = _series({"GEN 1:1": "reference one", "GEN 1:2": "reference two"})
    sheet = make_sheet("German", hyp, ref, PASSAGES)
    assert "## Genesis 1:1–2" in sheet
    assert "| GEN 1:1 | generated one | reference one |" in sheet
    assert "generated \\| two" in sheet  # pipes escaped


def test_missing_passage_skipped():
    hyp = _series({"GEN 1:1": "x", "GEN 1:2": "y"})
    ref = _series({"GEN 1:1": "x", "GEN 1:2": "y"})
    passages = PASSAGES + [{"name": "Psalm 23", "start": "PSA 23:1", "end": "PSA 23:6"}]
    sheet = make_sheet("German", hyp, ref, passages)
    assert "Psalm 23" not in sheet
