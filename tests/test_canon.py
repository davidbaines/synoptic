from synoptic.canon import CANON_BOOKS, NT_BOOKS, OT_BOOKS, book_of


def test_book_counts():
    assert len(OT_BOOKS) == 39
    assert len(NT_BOOKS) == 27
    assert len(CANON_BOOKS) == 66


def test_no_overlap_or_duplicates():
    assert len(set(OT_BOOKS) & set(NT_BOOKS)) == 0
    assert len(set(CANON_BOOKS)) == len(CANON_BOOKS)


def test_book_of():
    assert book_of("GEN 1:1") == "GEN"
    assert book_of("1CO 13:4") == "1CO"
