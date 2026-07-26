"""USFM book codes for the Protestant canon, in vref order.

The eBible corpus is keyed by `vref` values such as ``GEN 1:1``. Deuterocanon
books also appear in the 41,899-line vref list but are outside the composite
Greek source, so they are not listed here; anything not in OT_BOOKS or
NT_BOOKS is treated as deuterocanon.
"""

from __future__ import annotations

OT_BOOKS: tuple[str, ...] = (
    "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT",
    "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EZR", "NEH", "EST",
    "JOB", "PSA", "PRO", "ECC", "SNG",
    "ISA", "JER", "LAM", "EZK", "DAN",
    "HOS", "JOL", "AMO", "OBA", "JON", "MIC", "NAM", "HAB", "ZEP",
    "HAG", "ZEC", "MAL",
)

NT_BOOKS: tuple[str, ...] = (
    "MAT", "MRK", "LUK", "JHN", "ACT", "ROM",
    "1CO", "2CO", "GAL", "EPH", "PHP", "COL",
    "1TH", "2TH", "1TI", "2TI", "TIT", "PHM",
    "HEB", "JAS", "1PE", "2PE", "1JN", "2JN", "3JN", "JUD", "REV",
)

CANON_BOOKS: tuple[str, ...] = OT_BOOKS + NT_BOOKS


def book_of(vref: str) -> str:
    """Return the USFM book code of a vref like ``GEN 1:1``."""
    return vref.split(" ", 1)[0]
