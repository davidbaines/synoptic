import pandas as pd
import pytest


@pytest.fixture
def synthetic_verses() -> pd.DataFrame:
    """Tiny wide verse table covering OT, NT and awkward cells."""
    vrefs = ["GEN 1:1", "GEN 1:2", "EXO 1:1", "MAT 1:1", "MAT 1:2", "REV 22:21", "TOB 1:1"]
    frame = pd.DataFrame(
        {
            "vref": vrefs,
            "alpha": ["a-gen1", "a-gen2", "a-exo1", "a-mat1", "a-mat2", "a-rev", "a-tob"],
            "beta": ["b-gen1", "", "b-exo1", "b-mat1", "<range>", "b-rev", ""],
            "grcbrent": ["g-gen1", "g-gen2", "g-exo1", "", "", "", "g-tob"],
            "grc-tisch": ["", "", "", "t-mat1", "t-mat2", "t-rev", ""],
        }
    )
    return frame.set_index("vref")


@pytest.fixture
def synthetic_metadata() -> pd.DataFrame:
    """Metadata covering dedup, coverage-floor and diversity cases."""
    rows = [
        # code, id, name, script, OT, NT, DC
        ("swh", "swhbig", "Kiswahili", "Latin", 23000, 7900, 0),
        ("swh", "swhsmall", "Kiswahili", "Latin", 0, 7000, 0),
        ("ceb", "cebulb", "Cebuano", "Latin", 23000, 7900, 0),
        ("hin", "hin2017", "Hindi", "Devanagari", 23000, 7900, 0),
        ("tur", "turytc", "Turkish", "Latin", 22500, 7900, 0),
        ("vie", "vie1934", "Vietnamese", "Latin", 23000, 7900, 0),
        ("eng", "engbsb", "English", "Latin", 23145, 7941, 0),
        ("xxx", "xxxfrag", "Fragmentary", "Latin", 0, 300, 0),
    ]
    meta = pd.DataFrame(
        rows,
        columns=[
            "languageCode", "translationId", "languageNameInEnglish", "script",
            "OTverses", "NTverses", "DCverses",
        ],
    )
    meta["totalVerses"] = meta[["OTverses", "NTverses", "DCverses"]].sum(axis=1)
    return meta


@pytest.fixture
def families() -> dict[str, str]:
    return {
        "swh": "Bantu",
        "ceb": "Austronesian",
        "hin": "Indo-Aryan",
        "tur": "Turkic",
        "vie": "Austroasiatic",
        "eng": "Indo-European",
    }
