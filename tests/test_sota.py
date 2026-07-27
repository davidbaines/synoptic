import pandas as pd

from synoptic import sota


def test_extract_stem():
    assert sota.extract_stem("hin", "hin2017") == "hin-hin2017"


def test_flores_tag_known_and_new(monkeypatch):
    # Known NLLB language -> its FLORES tag; unknown -> <iso>_<script> new code.
    import synoptic.sota as s
    # hin is in NLLB; a made-up iso is not.
    assert s.flores_tag("hin", "Deva") == "hin_Deva"
    assert s.flores_tag("zzz", "Ethi") == "zzz_Ethi"


def _spec():
    return sota.SotaSpec(
        name="ms8_devanagari_drafting_hne", src_iso="hin", src_project="hin2017",
        src_script="Deva", trg_iso="hne", trg_project="hne", trg_script="Deva",
        test_vrefs=["MRK 1:1", "GEN 1:3"],
    )


def test_build_config_shape():
    cfg = sota.build_config(_spec(), use_test_set_from="synoptic-sota/x/ms8_devanagari_drafting_hne")
    assert cfg["model"] == sota.NLLB_DISTILLED_1_3B
    assert cfg["data"]["seed"] == 111
    pair = cfg["data"]["corpus_pairs"][0]
    assert pair["src"] == "hin-hin2017"
    assert pair["trg"] == "hne-hne"
    assert pair["use_test_set_from"] == "synoptic-sota/x/ms8_devanagari_drafting_hne"
    assert pair["type"] == "train,test"
    assert cfg["data"]["lang_codes"] == {"hin": "hin_Deva", "hne": "hne_Deva"}


def test_write_experiment_emits_config_and_vref(tmp_path):
    exp = sota.write_experiment(_spec(), tmp_path, rel_root="synoptic-sota/x")
    assert (exp / "config.yml").exists()
    vrefs = (exp / "test.vref.txt").read_text().splitlines()
    assert vrefs == ["MRK 1:1", "GEN 1:3"]
    import yaml
    cfg = yaml.safe_load((exp / "config.yml").read_text())
    # self-reference points at this experiment's own folder
    assert cfg["data"]["corpus_pairs"][0]["use_test_set_from"].endswith(
        "/ms8_devanagari_drafting_hne")


def test_run_argv_has_by_book_chrf3_and_queue():
    argv = sota.run("exp1", "synoptic-sota/x", queue="jobs_backlog")
    assert "--by-book" in argv and "chrf3" in argv
    assert argv[-1] == "synoptic-sota/x/exp1"
    assert "--clearml-queue" in argv and "jobs_backlog" in argv


def test_export_scripture_writes_vref_aligned(tmp_path, monkeypatch):
    # Stub the corpus so the test is offline and deterministic.
    idx = pd.Index(["GEN 1:1", "GEN 1:2", "MAT 1:1"], name="vref")
    verses = pd.DataFrame({"hin2017": ["adi", "", "<range>"],
                           "hne": ["a", "b", "c"]}, index=idx)
    meta = pd.DataFrame({"translationId": ["hin2017", "hne"],
                         "languageCode": ["hin", "hne"]})
    monkeypatch.setattr(sota, "load_verses", lambda ids: verses[list(ids)])
    monkeypatch.setattr(sota, "load_metadata", lambda: meta)
    paths = sota.export_scripture(["hin2017", "hne"], tmp_path)
    names = {p.name for p in paths}
    assert names == {"hin-hin2017.txt", "hne-hne.txt"}
    # vref-aligned: one line per master vref; missing -> blank; <range> kept
    assert (tmp_path / "hin-hin2017.txt").read_text().splitlines() == ["adi", "", "<range>"]
