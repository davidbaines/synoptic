import pandas as pd
import pytest

from synoptic.data import VREF_COLUMN
from synoptic.data_pipeline import apply_nt_truncation
from synoptic.script_pool import PoolSpec, build_pool


def _meta():
    rows = [
        # code, id, name, script, OT, NT, licence
        ("hin", "hin2017", "Hindi", "Devanagari", 23145, 7959, "by-sa"),
        ("hne", "hne", "Chhattisgarhi", "Devanagari", 23145, 7958, "by-sa"),
        ("mar", "mar", "Marathi", "Devanagari", 23143, 7944, "by-sa"),
        ("mar", "marc", "Marathi 2", "Devanagari", 23129, 7948, "by-sa"),
        ("san", "sandev", "Sanskrit", "Devanagari", 0, 7959, "by-sa"),
        ("tcn", "tcn", "Tichurong", "Devanagari", 0, 1886, "by-sa"),   # < 7000
        ("bad", "baddev", "NoShare", "Devanagari", 23000, 7900, "by-nc-nd"),
        ("nya", "nya", "Chichewa", "Latin", 23145, 7959, "by-sa"),
        ("nde", "nde", "Ndebele", "Latin", 23145, 7958, "by-sa"),
        ("swh", "swhbig", "Swahili", "Latin", 23145, 7958, "by-sa"),
        ("swh", "swhsmall", "Swahili NT", "Latin", 0, 7853, "by-sa"),
        ("kik", "kik", "Gikuyu", "Latin", 23145, 7956, "by-sa"),
        ("eng", "engbsb", "English", "Latin", 23145, 7941, "by"),      # not Bantu
    ]
    meta = pd.DataFrame(
        rows,
        columns=["languageCode", "translationId", "languageNameInEnglish",
                 "script", "OTverses", "NTverses", "licence_Licence_Type"],
    )
    meta["DCverses"] = 0
    meta["totalVerses"] = meta["OTverses"] + meta["NTverses"]
    return meta


DEVANAGARI = PoolSpec(script="Devanagari", targets=["hne", "mar"], exclude=["marc"])
BANTU = PoolSpec(
    script="Latin", targets=["nde", "nya"],
    languages=["kik", "nde", "nya", "swh"], one_per_language=True,
)


def test_pool_excludes_duplicates_fragments_and_unshareable():
    pool = build_pool(DEVANAGARI, metadata=_meta())
    assert set(pool["translationId"]) == {"hin2017", "hne", "mar", "sandev"}
    assert not pool["ntOnly"].any()  # no truncation requested


def test_one_per_language_and_truncation_marks():
    pool = build_pool(BANTU, metadata=_meta(), nt_only_except=["swhbig"])
    ids = set(pool["translationId"])
    assert "engbsb" not in ids                 # language filter
    assert "swhsmall" not in ids               # one text per language, best coverage
    marked = dict(zip(pool["translationId"], pool["ntOnly"]))
    assert marked["kik"]                       # neither source nor target
    assert not marked["swhbig"]                # the source keeps its OT
    assert not marked["nde"] and not marked["nya"]  # targets keep theirs


def test_missing_target_raises():
    meta = _meta()
    meta = meta[meta["translationId"] != "hne"]
    with pytest.raises(ValueError, match="hne"):
        build_pool(DEVANAGARI, metadata=meta)


def test_apply_nt_truncation_blanks_ot_cells():
    verses = pd.DataFrame(
        {"a": ["ot", "nt"], "b": ["ot", "nt"]},
        index=pd.Index(["GEN 1:1", "MAT 1:1"], name=VREF_COLUMN),
    )
    selection = pd.DataFrame(
        {"translationId": ["a", "b"], "ntOnly": ["True", "False"]}
    )
    out = apply_nt_truncation(verses, selection)
    assert out.at["GEN 1:1", "a"] == ""        # truncated member loses OT
    assert out.at["MAT 1:1", "a"] == "nt"      # keeps NT
    assert out.at["GEN 1:1", "b"] == "ot"      # exempt member untouched
    assert verses.at["GEN 1:1", "a"] == "ot"   # input not mutated
