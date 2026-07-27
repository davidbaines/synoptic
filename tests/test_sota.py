import pandas as pd

from synoptic import sota


def test_extract_stem():
    assert sota.extract_stem("hin", "hin2017") == "hin-hin2017"


def test_sota_stem_has_suffix():
    assert sota.sota_stem("hin", "hin2017") == "hin-hin2017_synsota"


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
    donor = sota.donor_relref("synoptic-sota/x", "ms8_devanagari_drafting_hne")
    cfg = sota.build_config(_spec(), use_test_set_from=donor)
    assert cfg["model"] == sota.NLLB_DISTILLED_1_3B
    assert cfg["data"]["seed"] == 111
    pair = cfg["data"]["corpus_pairs"][0]
    assert pair["src"] == "hin-hin2017_synsota"
    assert pair["trg"] == "hne-hne_synsota"
    assert pair["use_test_set_from"] == donor
    assert pair["type"] == "train,test,val"      # val split = silnlp default
    assert cfg["data"]["lang_codes"] == {"hin": "hin_Deva", "hne": "hne_Deva"}


def test_write_experiment_puts_testset_in_separate_donor(tmp_path):
    exp = sota.write_experiment(_spec(), tmp_path, rel_root="synoptic-sota/x")
    # exp dir must NOT contain a test.*.txt (silnlp preprocess would delete it)
    assert (exp / "config.yml").exists()
    assert not list(exp.glob("test.*.txt"))
    # donor sibling holds the per-pair-named vref file
    donor = tmp_path / f"{_spec().name}__testset"
    vref_file = donor / "test.hin.hne.vref.txt"
    assert vref_file.read_text().splitlines() == ["MRK 1:1", "GEN 1:3"]
    import yaml
    cfg = yaml.safe_load((exp / "config.yml").read_text())
    assert cfg["data"]["corpus_pairs"][0]["use_test_set_from"].endswith("__testset")


def test_run_argv_flags_and_ordering():
    argv = sota.run("exp1", "synoptic-sota/x", queue="jobs_backlog")
    assert "--score-by-book" in argv and "--by-book" not in argv
    # experiment path precedes --scorers (nargs=* would otherwise eat it)
    assert argv.index("synoptic-sota/x/exp1") < argv.index("--scorers")
    assert argv[-2:] == ["--scorers", "chrf3"]
    assert "--clearml-queue" in argv and "jobs_backlog" in argv
    assert argv[argv.index("--clearml-tag") + 1] == "research"


def test_export_scripture_writes_vref_aligned(tmp_path, monkeypatch):
    # Stub the corpus so the test is offline and deterministic.
    idx = pd.Index(["GEN 1:1", "GEN 1:2", "MAT 1:1"], name="vref")  # starts GEN 1:1 (alignment guard)
    verses = pd.DataFrame({"hin2017": ["adi", "", "<range>"],
                           "hne": ["a", "b", "c"]}, index=idx)
    meta = pd.DataFrame({"translationId": ["hin2017", "hne"],
                         "languageCode": ["hin", "hne"]})
    monkeypatch.setattr(sota, "load_verses", lambda ids: verses[list(ids)])
    monkeypatch.setattr(sota, "load_metadata", lambda: meta)
    paths = sota.export_scripture(["hin2017", "hne"], tmp_path)
    names = {p.name for p in paths}
    assert names == {"hin-hin2017_synsota.txt", "hne-hne_synsota.txt"}
    # vref-aligned: one line per master vref; missing -> blank; <range> kept
    assert (tmp_path / "hin-hin2017_synsota.txt").read_text().splitlines() == ["adi", "", "<range>"]
