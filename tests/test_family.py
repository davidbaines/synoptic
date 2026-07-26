import pandas as pd
import pytest

import synoptic.family as family_mod
from synoptic.family import build_family_selection, load_family_codes


def _meta(rows):
    meta = pd.DataFrame(
        rows,
        columns=[
            "languageCode", "translationId", "languageNameInEnglish", "script",
            "OTverses", "NTverses", "DCverses", "NTbooks", "licence_Licence_Type",
        ],
    )
    meta["totalVerses"] = meta[["OTverses", "NTverses", "DCverses"]].sum(axis=1)
    return meta


def _force(monkeypatch, ids):
    monkeypatch.setattr(family_mod, "load_holdout_ids", lambda f: list(ids))


def test_ie_map_covers_expected_branches():
    codes = load_family_codes("indo_european")
    assert codes["deu"] == "Germanic"
    assert codes["hin"] == "Indo-Aryan"
    assert codes["rus"] == "Slavic"
    assert codes["spa"] == "Romance"
    # the Greek source and ancient Hebrew must never be selectable as targets
    assert "grc" not in codes
    assert "hbo" not in codes


def test_family_selection_restricts_and_forces_holdouts(monkeypatch):
    meta = _meta(
        [
            # IE languages (kept)
            ("deu", "deuelbbk", "German", "Latin", 23000, 7900, 0, 27, "by-sa"),
            ("deu", "deusmall", "German (short)", "Latin", 0, 5000, 0, 27, "by-sa"),
            ("nld", "nld1939", "Dutch", "Latin", 22000, 7900, 0, 27, "by-sa"),
            ("hin", "hin2017", "Hindi", "Devanagari", 23000, 7900, 0, 27, "by-sa"),
            ("eng", "engbsb", "English", "Latin", 23000, 7900, 0, 27, "by-sa"),
            # non-IE languages (dropped unless forced as targets)
            ("swh", "swhonen", "Swahili", "Latin", 23000, 7900, 0, 27, "by-sa"),
            ("cmn", "cmncu", "Chinese", "CJK", 23000, 7900, 0, 27, "by-sa"),
        ]
    )
    # swhonen is outside the IE membership list: gradient-step pools exclude
    # the target's family by design, so targets are forced from full metadata.
    _force(monkeypatch, ["hin2017", "swhonen"])
    sel = build_family_selection(
        family="indo_european", holdouts_file="x.yaml", metadata=meta,
    )
    langs = set(sel["languageCode"])
    assert langs == {"deu", "nld", "hin", "eng", "swh"}
    assert "cmn" not in langs
    # one translation per language, best coverage wins
    assert (sel["languageCode"] == "deu").sum() == 1
    assert "deusmall" not in set(sel["translationId"])
    # branch column populated for members, empty for the out-of-family target
    assert set(sel.loc[sel["languageCode"] != "swh", "branch"]) == {
        "Germanic", "Indo-Aryan",
    }
    assert (sel.loc[sel["languageCode"] == "swh", "branch"] == "").all()


def test_complete_nt_rule_governs_membership_and_dedup(monkeypatch):
    meta = _meta(
        [
            # OT-heavy but incomplete NT vs smaller complete NT: complete wins
            ("deu", "deu_bigot", "German big OT", "Latin", 23000, 6000, 0, 26, "by-sa"),
            ("deu", "deu_nt", "German NT", "Latin", 0, 7900, 0, 27, "by-sa"),
            # NT-portions-only language: excluded entirely
            ("nld", "nld_part", "Dutch portions", "Latin", 0, 6000, 0, 12, "by-sa"),
            ("hin", "hin2017", "Hindi", "Devanagari", 23000, 7900, 0, 27, "by-sa"),
        ]
    )
    _force(monkeypatch, ["hin2017"])
    sel = build_family_selection(
        family="indo_european", holdouts_file="x.yaml", metadata=meta,
    )
    assert set(sel["translationId"]) == {"deu_nt", "hin2017"}


def test_shareable_filter_drops_non_derivative_licences(monkeypatch):
    meta = _meta(
        [
            ("deu", "deu_pd", "German PD", "Latin", 23000, 7900, 0, 27, "Public Domain"),
            ("nld", "nld_nd", "Dutch ND", "Latin", 23000, 7900, 0, 27, "by-nc-nd"),
            ("hin", "hin_sa", "Hindi SA", "Devanagari", 23000, 7900, 0, 27, "by-sa"),
            ("fra", "fra_unk", "French ?", "Latin", 23000, 7900, 0, 27, "Unknown"),
        ]
    )
    _force(monkeypatch, ["hin_sa"])
    sel = build_family_selection(
        family="indo_european", holdouts_file="x.yaml", metadata=meta,
    )
    # only derivative-permitting licences survive
    assert set(sel["translationId"]) == {"deu_pd", "hin_sa"}


def test_missing_holdout_id_errors(monkeypatch):
    meta = _meta([("deu", "deuelbbk", "German", "Latin", 23000, 7900, 0, 27, "by-sa")])
    _force(monkeypatch, ["hin2017"])
    with pytest.raises(ValueError, match="hin2017"):
        build_family_selection(
            family="indo_european", holdouts_file="x.yaml", metadata=meta,
        )


def test_non_shareable_holdout_errors(monkeypatch):
    meta = _meta(
        [
            ("deu", "deuelbbk", "German", "Latin", 23000, 7900, 0, 27, "by-sa"),
            ("hin", "hin_nd", "Hindi ND", "Devanagari", 23000, 7900, 0, 27, "by-nc-nd"),
        ]
    )
    _force(monkeypatch, ["hin_nd"])
    with pytest.raises(ValueError, match="non-shareable"):
        build_family_selection(
            family="indo_european", holdouts_file="x.yaml", metadata=meta,
        )


def test_holdouts_file_required():
    with pytest.raises(ValueError, match="holdouts_file"):
        build_family_selection(family="indo_european", holdouts_file="")
