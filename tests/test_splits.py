import pytest

from synoptic.splits import (
    Splits,
    assert_no_leakage,
    build_splits,
    expand_books,
    manifest_checksum,
)


def test_expand_books_aliases():
    assert len(expand_books(["OT"])) == 39
    assert len(expand_books(["NT"])) == 27
    assert expand_books(["GEN", "MAT"]) == {"GEN", "MAT"}
    with pytest.raises(ValueError):
        expand_books(["NOPE"])


def test_holdout_books_go_to_test(synthetic_verses):
    holdouts = {"alpha": ["GEN"], "beta": ["MAT"]}
    splits = build_splits(synthetic_verses, holdouts, valid_size=1, seed=0)

    test_pairs = set(zip(splits.test["vref"], splits.test["translation"]))
    assert ("GEN 1:1", "alpha") in test_pairs
    assert ("GEN 1:2", "alpha") in test_pairs
    assert ("MAT 1:1", "beta") in test_pairs
    # beta's GEN and alpha's MAT stay trainable
    trainable = set(zip(splits.train["vref"], splits.train["translation"])) | set(
        zip(splits.valid["vref"], splits.valid["translation"])
    )
    assert ("GEN 1:1", "beta") in trainable
    assert ("MAT 1:1", "alpha") in trainable


def test_empty_and_range_cells_dropped(synthetic_verses):
    splits = build_splits(synthetic_verses, {}, valid_size=1, seed=0)
    all_pairs = set()
    for frame in (splits.train, splits.valid, splits.test):
        all_pairs |= set(zip(frame["vref"], frame["translation"]))
    assert ("GEN 1:2", "beta") not in all_pairs  # empty
    assert ("MAT 1:2", "beta") not in all_pairs  # <range>


def test_valid_disjoint_and_seeded(synthetic_verses):
    a = build_splits(synthetic_verses, {"alpha": ["GEN"]}, valid_size=2, seed=13)
    b = build_splits(synthetic_verses, {"alpha": ["GEN"]}, valid_size=2, seed=13)
    assert manifest_checksum(a.valid) == manifest_checksum(b.valid)
    assert manifest_checksum(a.train) == manifest_checksum(b.train)
    assert_no_leakage(a, {"alpha": ["GEN"]})


def test_valid_size_guard(synthetic_verses):
    with pytest.raises(ValueError):
        build_splits(synthetic_verses, {}, valid_size=10_000, seed=0)


def test_checksum_order_independent(synthetic_verses):
    splits = build_splits(synthetic_verses, {}, valid_size=1, seed=0)
    shuffled = splits.train.sample(frac=1.0, random_state=99)
    assert manifest_checksum(splits.train) == manifest_checksum(shuffled)


def test_leakage_detected():
    import pandas as pd

    leaky = pd.DataFrame(
        {"vref": ["GEN 1:1"], "translation": ["alpha"], "book": ["GEN"]}
    )
    empty = leaky.iloc[0:0]
    splits = Splits(train=leaky, valid=empty, test=leaky.copy())
    with pytest.raises(AssertionError):
        assert_no_leakage(splits, {"alpha": ["GEN"]})


def test_verse_holdouts_go_to_test(synthetic_verses):
    verse_holdouts = {"alpha": ["GEN 1:2"]}
    splits = build_splits(
        synthetic_verses, {}, valid_size=1, seed=0, verse_holdouts=verse_holdouts
    )
    test_pairs = set(zip(splits.test["vref"], splits.test["translation"]))
    assert test_pairs == {("GEN 1:2", "alpha")}
    trainable = set(zip(splits.train["vref"], splits.train["translation"])) | set(
        zip(splits.valid["vref"], splits.valid["translation"])
    )
    assert ("GEN 1:1", "alpha") in trainable  # same book, not held out
    assert ("GEN 1:1", "beta") in trainable   # same vref, other translation
    assert_no_leakage(splits, {}, verse_holdouts)


def test_verse_holdouts_combine_with_book_holdouts(synthetic_verses):
    holdouts = {"alpha": ["MAT"]}
    verse_holdouts = {"alpha": ["GEN 1:1"]}
    splits = build_splits(
        synthetic_verses, holdouts, valid_size=1, seed=0, verse_holdouts=verse_holdouts
    )
    test_pairs = set(zip(splits.test["vref"], splits.test["translation"]))
    assert ("GEN 1:1", "alpha") in test_pairs
    assert ("MAT 1:1", "alpha") in test_pairs
    assert ("GEN 1:2", "alpha") not in test_pairs
    assert_no_leakage(splits, holdouts, verse_holdouts)


def test_verse_leakage_detected():
    import pandas as pd

    leaky = pd.DataFrame(
        {"vref": ["GEN 1:1"], "translation": ["alpha"], "book": ["GEN"]}
    )
    empty = leaky.iloc[0:0]
    splits = Splits(train=leaky, valid=empty, test=leaky.copy())
    with pytest.raises(AssertionError):
        assert_no_leakage(splits, {}, {"alpha": ["GEN 1:1"]})


def _diluted_verses():
    """A held-out target that is a small fraction of a large companion pool.

    Mirrors the family setup: 2 targets hold out books, 8 companions are fully
    present, so a global valid sample under-represents the targets — the bug
    reserve_per_holdout fixes.
    """
    import pandas as pd

    ot = [f"GEN 1:{i}" for i in range(1, 21)]          # companions only
    mrk = [f"MRK 1:{i}" for i in range(1, 6)]          # target test (held out)
    mat = [f"MAT 1:{i}" for i in range(1, 41)]         # target train/valid pool
    vrefs = ot + mrk + mat
    data = {"vref": vrefs}
    # target: empty in OT, present in MRK (test) and MAT (rest)
    data["tgt"] = ["" for _ in ot] + [f"t-{v}" for v in mrk + mat]
    for c in range(8):
        data[f"comp{c}"] = [f"c{c}-{v}" for v in vrefs]  # present everywhere
    return pd.DataFrame(data).set_index("vref")


def test_reserve_per_holdout_guarantees_target_validation_verses():
    verses = _diluted_verses()
    holdouts = {"tgt": ["MRK"]}

    def tgt_valid(reserve):
        s = build_splits(verses, holdouts, valid_size=100, seed=13,
                         reserve_per_holdout=reserve)
        return s, int((s.valid["translation"] == "tgt").sum())

    # Without reservation the global sample starves the diluted target...
    _, without = tgt_valid(0)
    assert without < 30, without
    # ...with it, the target always clears the reserved floor.
    s, with_reserve = tgt_valid(30)
    assert with_reserve >= 30, with_reserve
    # Reserved validation verses never leak into training.
    assert_no_leakage(s, holdouts)


def test_reserve_per_holdout_is_seeded():
    verses = _diluted_verses()
    holdouts = {"tgt": ["MRK"]}
    a = build_splits(verses, holdouts, valid_size=100, seed=13, reserve_per_holdout=30)
    b = build_splits(verses, holdouts, valid_size=100, seed=13, reserve_per_holdout=30)
    assert manifest_checksum(a.valid) == manifest_checksum(b.valid)
    assert manifest_checksum(a.train) == manifest_checksum(b.train)
