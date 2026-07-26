from synoptic.selection import SelectionConfig, select_translations


def test_dedup_prefers_coverage(synthetic_metadata, families):
    config = SelectionConfig(min_verses=1000)
    chosen = select_translations(config, synthetic_metadata, families)
    swh = chosen[chosen["languageCode"] == "swh"]
    assert list(swh["translationId"]) == ["swhbig"]


def test_min_verses_floor(synthetic_metadata, families):
    config = SelectionConfig(min_verses=1000)
    chosen = select_translations(config, synthetic_metadata, families)
    assert "xxxfrag" not in set(chosen["translationId"])


def test_exclude_and_include(synthetic_metadata, families):
    config = SelectionConfig(
        min_verses=1000, exclude=["cebulb"], include=["xxxfrag"]
    )
    chosen = select_translations(config, synthetic_metadata, families)
    ids = set(chosen["translationId"])
    assert "cebulb" not in ids
    assert "xxxfrag" in ids  # forced in despite failing the floor


def test_target_size_spreads_across_buckets(synthetic_metadata, families):
    config = SelectionConfig(min_verses=1000, target_size=4)
    chosen = select_translations(config, synthetic_metadata, families)
    assert len(chosen) == 4
    # Round-robin over buckets: four picks must span four distinct families
    assert chosen["family"].nunique() == 4
