import pandas as pd

from synoptic.data import VREF_COLUMN
from synoptic.multisource import (
    build_ms_pairs,
    inference_source_ranking,
    present_by_vref,
    strip_tags,
    to_ms_sources,
)
from synoptic.preprocess import SRC_COLUMN, TGT_COLUMN

SOURCE_ID = "hin"


def _tags(src):
    return [t for t in src.split(" ") if t.startswith("<") and t.endswith(">")]


def _setup():
    """Pool of four translations; ``hin`` is the alignment-chosen source.

    Unlike the Greek composite of the previous series, the source is a pool
    member: present in the verse table AND in the manifests like any other.
    """
    vrefs = ["GEN 1:1", "GEN 1:2"]
    verses = pd.DataFrame(
        {"eng": ["e1", "e2"], "spa": ["s1", "s2"], "deu": ["d1", "d2"],
         "hin": ["h1", "h2"]},
        index=pd.Index(vrefs, name=VREF_COLUMN),
    )
    language_of = {"eng": "eng", "spa": "spa", "deu": "deu", "hin": "hin"}
    train = pd.DataFrame(
        {VREF_COLUMN: ["GEN 1:1", "GEN 1:2", "GEN 1:1", "GEN 1:2"],
         "translation": ["eng", "spa", "hin", "hin"]}
    )
    valid = pd.DataFrame({VREF_COLUMN: ["GEN 1:1", "GEN 1:2"], "translation": ["deu", "deu"]})
    return train, valid, verses, language_of


def _selection():
    return pd.DataFrame(
        {
            "translationId": ["eng", "spa", "deu", "hin"],
            "languageCode": ["eng", "spa", "deu", "hin"],
            "branch": ["Germanic", "Romance", "Germanic", "Indic"],
            "totalVerses": ["100", "200", "150", "300"],
        }
    )


def test_one_pair_per_target_and_format():
    train, valid, verses, lang = _setup()
    out = build_ms_pairs(train, valid, verses, lang, k=4, k_min=1, seed=1,
                         source_id=SOURCE_ID)
    # one example per (vref, target) — no K-fold expansion
    assert len(out) == len(train)
    for _, r in out.iterrows():
        tags = _tags(r[SRC_COLUMN])
        assert tags[0].startswith("<2")                 # target tag first
        assert all(t.startswith("<1") for t in tags[1:])  # then source tags
        # never uses the target itself as a source
        own = tags[0].replace("<2", "<1")
        assert own not in tags[1:]
        if tags[0] != f"<2{SOURCE_ID}>":
            # source forced first among renderings (usable at every vref here)
            assert tags[1] == f"<1{SOURCE_ID}>"


def test_source_translation_as_target_gets_no_self_source():
    train, valid, verses, lang = _setup()
    out = build_ms_pairs(train, valid, verses, lang, k=4, k_min=4, seed=3,
                         source_id=SOURCE_ID)
    own = out[out["translation"] == SOURCE_ID]
    assert len(own) == 2                              # hin is a target too
    for _, r in own.iterrows():
        assert f"<1{SOURCE_ID}>" not in _tags(r[SRC_COLUMN])


def test_source_not_forced_where_held_out():
    train, valid, verses, lang = _setup()
    # Remove hin's GEN 1:2 from the manifests: its cell there is held out.
    train = train[~((train["translation"] == SOURCE_ID) & (train[VREF_COLUMN] == "GEN 1:2"))]
    out = build_ms_pairs(train, valid, verses, lang, k=10, k_min=10, seed=2,
                         source_id=SOURCE_ID)
    at_v2 = out[(out[VREF_COLUMN] == "GEN 1:2") & (out["translation"] != SOURCE_ID)]
    assert len(at_v2)
    for _, r in at_v2.iterrows():
        assert f"<1{SOURCE_ID}>" not in _tags(r[SRC_COLUMN])
        assert "h2" not in r[SRC_COLUMN]


def test_determinism_same_seed():
    train, valid, verses, lang = _setup()
    a = build_ms_pairs(train, valid, verses, lang, k=3, seed=7, source_id=SOURCE_ID)
    b = build_ms_pairs(train, valid, verses, lang, k=3, seed=7, source_id=SOURCE_ID)
    assert a.equals(b)


def test_k_min_dropout_produces_variable_counts():
    # With many vrefs, n ~ Uniform{1..k} must produce both 1-rendering and
    # k-rendering examples.
    n = 200
    vrefs = [f"GEN 1:{i}" for i in range(1, n + 1)]
    verses = pd.DataFrame(
        {c: [f"{c}{i}" for i in range(n)] for c in ["eng", "spa", "deu", "fra", "hin"]},
        index=pd.Index(vrefs, name=VREF_COLUMN),
    )
    lang = {c: c for c in ["eng", "spa", "deu", "fra", "hin"]}
    train = pd.DataFrame({VREF_COLUMN: vrefs, "translation": ["eng"] * n})
    valid = pd.DataFrame(
        {VREF_COLUMN: vrefs * 4,
         "translation": ["spa"] * n + ["deu"] * n + ["fra"] * n + ["hin"] * n}
    )
    out = build_ms_pairs(train, valid, verses, lang, k=4, k_min=1, seed=13,
                         source_id=SOURCE_ID)
    counts = out[SRC_COLUMN].map(lambda s: len(_tags(s)) - 1)  # renderings per example
    assert counts.min() == 1
    assert counts.max() == 4


def test_held_out_translation_never_used_as_source():
    train, valid, verses, lang = _setup()
    verses["sec"] = ["x1", "x2"]  # held-out translation: text exists, not in manifests
    lang["sec"] = "sec"
    out = build_ms_pairs(train, valid, verses, lang, k=10, seed=5, source_id=SOURCE_ID)
    assert not out[SRC_COLUMN].str.contains("<1sec>").any()
    assert not out[SRC_COLUMN].str.contains("x1|x2").any()


def test_inference_ranking_branch_first_then_coverage():
    ranking = inference_source_ranking(_selection(), policy="branch-first")
    # For eng (Germanic): deu (same branch) first, then the rest by coverage
    assert ranking["eng"] == ["deu", "hin", "spa"]
    # For spa (Romance, alone): others by coverage desc: hin(300) > deu(150) > eng(100)
    assert ranking["spa"] == ["hin", "deu", "eng"]


def test_inference_ranking_coverage_ignores_branch():
    # The default policy is pure coverage order — identical across selections
    # with and without a branch column, so the companion-selection policy
    # cannot silently vary with CSV schema across compared runs.
    ranking = inference_source_ranking(_selection(), policy="coverage")
    assert ranking["eng"] == ["hin", "spa", "deu"]
    no_branch = _selection().drop(columns="branch")
    assert inference_source_ranking(no_branch, policy="coverage") == ranking


def test_inference_ranking_branch_first_requires_branch_column():
    import pytest

    with pytest.raises(ValueError, match="branch"):
        inference_source_ranking(
            _selection().drop(columns="branch"), policy="branch-first"
        )


def test_to_ms_sources_deterministic_and_leakage_safe():
    train, valid, verses, lang = _setup()
    verses["sec"] = ["x1", "x2"]
    lang["sec"] = "sec"
    present = present_by_vref(train, valid)
    sel = _selection()
    ranking = inference_source_ranking(sel)
    frame = pd.DataFrame(
        {
            VREF_COLUMN: ["GEN 1:1"],
            "translation": ["eng"],
            SRC_COLUMN: ["<2eng> h1"],
            TGT_COLUMN: ["e1"],
        }
    )
    out1 = to_ms_sources(frame, verses, lang, present, ranking, k=3, source_id=SOURCE_ID)
    out2 = to_ms_sources(frame, verses, lang, present, ranking, k=3, source_id=SOURCE_ID)
    assert out1.equals(out2)
    src = out1[SRC_COLUMN].iloc[0]
    tags = _tags(src)
    assert tags[0] == "<2eng>"
    assert tags[1] == f"<1{SOURCE_ID}>"        # forced source first
    assert "<1deu>" in tags                    # same-branch candidate chosen
    assert tags.count(f"<1{SOURCE_ID}>") == 1  # ranking never re-adds the source
    assert "<1sec>" not in tags                # held-out cell never selected
    assert "<1eng>" not in tags                # never the target itself


def test_to_ms_sources_source_translation_as_target():
    train, valid, verses, lang = _setup()
    present = present_by_vref(train, valid)
    ranking = inference_source_ranking(_selection())
    frame = pd.DataFrame(
        {
            VREF_COLUMN: ["GEN 1:1"],
            "translation": [SOURCE_ID],
            SRC_COLUMN: [f"<2{SOURCE_ID}> h1"],
            TGT_COLUMN: ["h1"],
        }
    )
    out = to_ms_sources(frame, verses, lang, present, ranking, k=3, source_id=SOURCE_ID)
    tags = _tags(out[SRC_COLUMN].iloc[0])
    assert f"<1{SOURCE_ID}>" not in tags


def test_ranking_without_branch_or_coverage_columns():
    sel = pd.DataFrame({"translationId": ["a", "b", "c"], "languageCode": ["a", "b", "c"]})
    ranking = inference_source_ranking(sel)
    assert ranking["a"] == ["b", "c"]  # stable id order when no other signal


def test_strip_tags():
    assert strip_tags("<2eng> <1hin> alpha <1deu> beta") == "alpha beta"
