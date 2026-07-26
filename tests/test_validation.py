import pandas as pd

from synoptic.validation import build_validation_set, validation_scores, should_stop
from synoptic.preprocess import SRC_COLUMN, TGT_COLUMN


def _test_pairs(n_per_lang=100):
    rows = []
    for translation, lang in [("engbsb", "eng"), ("deuelbbk", "deu")]:
        for i in range(n_per_lang):
            rows.append({
                "vref": f"GEN {i // 10 + 1}:{i % 10 + 1}",
                "translation": translation,
                SRC_COLUMN: f"<2{lang}> <GEN> <c{i}> <v{i}>",
                TGT_COLUMN: f"verse {i} in {lang}",
            })
    return pd.DataFrame(rows)


def test_validation_set_is_deterministic_and_order_independent():
    pairs = _test_pairs()
    lang_of = {"engbsb": "eng", "deuelbbk": "deu"}
    a = build_validation_set(pairs, lang_of, verses_per_language=25, seed=13)
    # shuffle the input rows: sampling must be unaffected (verification #5b)
    shuffled = pairs.sample(frac=1.0, random_state=99).reset_index(drop=True)
    b = build_validation_set(shuffled, lang_of, verses_per_language=25, seed=13)
    key = ["vref", "translation"]
    assert (
        set(map(tuple, a[key].itertuples(index=False)))
        == set(map(tuple, b[key].itertuples(index=False)))
    )
    assert len(a) == 50  # 25 per language, 2 languages


def test_validation_set_translations_filter_for_seen_probe():
    # the seen-verse set restricts to specific translation ids
    pairs = _test_pairs()
    lang_of = {"engbsb": "eng", "deuelbbk": "deu"}
    seen = build_validation_set(pairs, lang_of, verses_per_language=25, seed=13,
                           translations=["engbsb"])
    assert set(seen["translation"]) == {"engbsb"}
    assert len(seen) == 25


def test_validation_set_respects_small_languages():
    pairs = _test_pairs(n_per_lang=10)
    lang_of = {"engbsb": "eng", "deuelbbk": "deu"}
    validation = build_validation_set(pairs, lang_of, verses_per_language=250, seed=13)
    assert len(validation) == 20  # capped at what exists


def test_validation_scores_perfect_and_shape():
    pairs = _test_pairs(n_per_lang=5)
    validation = build_validation_set(pairs, {"engbsb": "eng", "deuelbbk": "deu"},
                            verses_per_language=5, seed=13)
    # perfect hypotheses -> chrF3 100 for both languages and macro
    row = validation_scores(validation[TGT_COLUMN].tolist(), validation)
    assert row["chrF3_eng"] == 100.0
    assert row["chrF3_deu"] == 100.0
    assert row["chrF3_macro"] == 100.0
    assert "BLEU_macro" in row


def test_should_stop_no_history():
    assert should_stop([], patience_steps=20000, min_gain=1.0) is False


def test_should_stop_waits_for_full_window():
    # only 10k of the 20k window elapsed -> cannot stop yet
    hist = [(1000, 10.0), (5000, 12.0), (10000, 13.0)]
    assert should_stop(hist, patience_steps=20000, min_gain=1.0) is False


def test_should_stop_triggers_on_plateau():
    # from step 21000 back to <=1000 the best gained only 0.3 (< 1.0)
    hist = [(1000, 30.0), (11000, 30.1), (21000, 30.3)]
    assert should_stop(hist, patience_steps=20000, min_gain=1.0) is True


def test_should_stop_holds_while_improving():
    # gained 3.0 across the window -> keep going
    hist = [(1000, 30.0), (11000, 31.5), (21000, 33.0)]
    assert should_stop(hist, patience_steps=20000, min_gain=1.0) is False


def test_should_stop_uses_best_not_last():
    # a late dip must not trigger a stop if an earlier point already cleared
    # the gain bar; "best so far" is what matters.
    hist = [(1000, 30.0), (11000, 34.0), (21000, 31.0)]
    assert should_stop(hist, patience_steps=20000, min_gain=1.0) is False
