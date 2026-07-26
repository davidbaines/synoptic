import pandas as pd

from synoptic.licensing import (
    check_shareable,
    is_shareable,
    model_licence_for,
    selection_licences,
)


META = pd.DataFrame(
    {
        "translationId": ["pd1", "by1", "sa1", "nd1", "ncnd1", "nc1", "unk1"],
        "licence_Licence_Type": [
            "Public Domain", "by", "by-sa", "by-nd", "by-nc-nd", "by-nc", "Unknown",
        ],
    }
)


def test_is_shareable_policy():
    assert is_shareable("Public Domain")
    assert is_shareable("by")
    assert is_shareable("by-sa")
    assert not is_shareable("by-nd")
    assert not is_shareable("by-nc-nd")
    assert not is_shareable("Unknown")
    assert not is_shareable("by-nc")
    assert is_shareable("by-nc", allow_nc=True)


def test_model_licence_precedence():
    assert model_licence_for(["Public Domain"]) == "cc0-1.0"
    assert model_licence_for(["Public Domain", "by"]) == "cc-by-4.0"
    assert model_licence_for(["by", "by-sa", "Public Domain"]) == "cc-by-sa-4.0"
    assert model_licence_for(["by-nc"], allow_nc=True) == "cc-by-nc-4.0"


def test_check_shareable_flags_offenders():
    sel = pd.DataFrame(
        {"translationId": ["pd1", "sa1", "nd1", "ncnd1"], "languageCode": ["a", "b", "c", "d"]}
    )
    ok, offenders = check_shareable(sel, metadata=META)
    assert not ok
    assert set(offenders["translationId"]) == {"nd1", "ncnd1"}


def test_check_shareable_all_clean():
    sel = pd.DataFrame({"translationId": ["pd1", "by1", "sa1"], "languageCode": ["a", "b", "c"]})
    ok, offenders = check_shareable(sel, metadata=META)
    assert ok
    assert offenders.empty


def test_selection_licences_annotates():
    sel = pd.DataFrame({"translationId": ["pd1", "nd1"], "languageCode": ["a", "b"]})
    ann = selection_licences(sel, metadata=META)
    assert list(ann["licence"]) == ["Public Domain", "by-nd"]
    assert list(ann["shareable"]) == [True, False]
