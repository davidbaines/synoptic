"""SOTA baselines: mirror a synoptic experiment as a silnlp NLLB-1.3B run.

Each synoptic experiment designates a primary source and a target and scores a
fixed set of test verses. To judge how close the from-scratch multi-source
models are to a strong pretrained model, we fine-tune
``facebook/nllb-200-distilled-1.3B`` with silnlp on the SAME source->target
parallel data and score it on the SAME verses, per book. silnlp's chrF3 is
sacrebleu ``char_order=6, beta=3`` — identical to :mod:`synoptic.evaluate` —
so the numbers compare directly.

This module is series-agnostic: a caller (one per experiment repo) supplies a
list of :class:`SotaSpec` — source/target translations and the exact test
vrefs — and this builds silnlp experiment folders, runs them, and collects
per-book scores. The concrete source/target/test mapping lives in the
experiment repo, not here.

Design notes (verified against silnlp 2026-07):
- A silnlp experiment is any folder under ``$SIL_NLP_DATA_PATH/MT/experiments``
  with a ``config.yml``; ``silnlp.nmt.experiment`` runs it. We place ours under
  one collected dir, bypassing ``create_onboarding_experiments`` and its
  country/language nesting.
- Exact test sets come from a ``test.vref.txt`` referenced by
  ``use_test_set_from`` (silnlp/nmt/config.py). A verse is kept only if present
  in both the src and trg extracts.
- Extracts are ``<iso>-<project>.txt``, vref-aligned to silnlp's master
  ``vref.txt``. We export our own from the corpus so the reference text is
  byte-identical to what synoptic scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .data import RANGE_MARKER, VREF_COLUMN, load_metadata, load_verses

NLLB_DISTILLED_1_3B = "facebook/nllb-200-distilled-1.3B"
SILNLP_SEED = 111  # silnlp's default split seed
# Our extracts live in the SHARED silnlp scripture dir (the only one the remote
# workers read — a local env override would not reach them), so they carry a
# distinct project suffix: it avoids overwriting silnlp's own extracts (e.g.
# an existing hin-hin2017.txt from a different eBible snapshot) and guarantees
# the reference text is exactly ours.
SOTA_PROJECT_SUFFIX = "_synsota"


def extract_stem(iso: str, project: str) -> str:
    """silnlp corpus file stem ``<iso>-<project>``."""
    return f"{iso}-{project}"


def sota_stem(iso: str, translation_id: str) -> str:
    """The extract stem for one of our exported corpus files (suffixed)."""
    return extract_stem(iso, f"{translation_id}{SOTA_PROJECT_SUFFIX}")


def flores_tag(iso: str, script: str) -> str:
    """FLORES-200 language token for an iso.

    Reuses silnlp's NLLB tag table where the language is in NLLB; otherwise
    builds ``<iso>_<script>`` — a new token silnlp registers and fine-tunes
    when ``add_new_lang_code`` is on (its default). ``script`` is a
    four-letter code (Latn, Arab, Deva, Ethi, Telu).
    """
    try:
        from silnlp.common.iso_info import NLLB_TAG_FROM_ISO
    except ImportError:  # silnlp only present where SOTA runs are built
        NLLB_TAG_FROM_ISO = {}
    return NLLB_TAG_FROM_ISO.get(iso) or f"{iso}_{script}"


@dataclass
class SotaSpec:
    """One silnlp NLLB baseline mirroring a synoptic source->target + test set."""

    name: str                       # experiment folder name (matches synoptic run family)
    src_iso: str
    src_project: str                # translationId, e.g. hin2017
    src_script: str
    trg_iso: str
    trg_project: str                # translationId, e.g. hne
    trg_script: str
    test_vrefs: list[str]           # exact verse refs to score (e.g. "GEN 1:3")
    # translations whose extracts this run needs (defaults to src+trg projects).
    extra_projects: list[tuple[str, str]] = field(default_factory=list)

    def scripture_translations(self) -> list[str]:
        return [self.src_project, self.trg_project, *[p for _, p in self.extra_projects]]


def _silnlp_master_vrefs() -> list[str] | None:
    """silnlp's master ``vref.txt`` lines, or None if silnlp isn't importable."""
    try:
        import silnlp
    except ImportError:
        return None
    vref = Path(silnlp.__file__).parent / "assets" / "vref.txt"
    if not vref.exists():
        return None
    return vref.read_text(encoding="utf-8").splitlines()


def _assert_vref_alignment(index: list[str]) -> None:
    """Fail loudly if the corpus vref order won't line up with silnlp's.

    silnlp maps each test/train vref to a line number in its master
    ``vref.txt``, so an extract written in corpus-index order is only correct
    if that order matches silnlp's exactly. Where silnlp is importable (the box
    that builds SOTA runs) we compare the full list; otherwise we fall back to
    the standard eBible invariants (a mismatch means re-verify against silnlp).
    """
    master = _silnlp_master_vrefs()
    if master is not None:
        if index != master:
            n = min(len(index), len(master))
            first_diff = next((i for i in range(n) if index[i] != master[i]), n)
            raise SystemExit(
                f"corpus vref order differs from silnlp's vref.txt "
                f"(len {len(index)} vs {len(master)}; first diff at line "
                f"{first_diff}); extracts would misalign — do not export"
            )
        return
    if not index or index[0] != "GEN 1:1":
        raise SystemExit(
            f"corpus vref index looks wrong (len={len(index)}, "
            f"first={index[:1]}); cannot confirm alignment to silnlp's vref.txt"
        )


def export_scripture(translation_ids: list[str], scripture_dir: Path) -> list[Path]:
    """Write vref-aligned ``<iso>-<project>.txt`` extracts from the corpus.

    One line per master vref in corpus order (blank where the translation is
    missing the verse; ``<range>`` markers preserved, as silnlp expects). Text
    is taken from :func:`synoptic.data.load_verses`, so it is byte-identical to
    what synoptic trained and scored on. Returns the written paths.
    """
    scripture_dir.mkdir(parents=True, exist_ok=True)
    meta = load_metadata().set_index("translationId")
    verses = load_verses(sorted(set(translation_ids)))
    _assert_vref_alignment(list(verses.index))
    written = []
    for tid in sorted(set(translation_ids)):
        iso = meta.at[tid, "languageCode"]
        # fillna guards against a NaN cell reaching the extract as "nan"
        # (load_verses already fills "", but this keeps export self-contained).
        col = verses[tid].fillna("")
        lines = [str(v) for v in col.tolist()]  # keep "" (missing) and <range>
        path = scripture_dir / f"{sota_stem(iso, tid)}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)
    return written


def exp_relref(rel_root: str, name: str) -> str:
    """Experiment path relative to MT/experiments (what silnlp addresses)."""
    return f"{rel_root}/{name}"


def donor_relref(rel_root: str, name: str) -> str:
    """Path of the run's test-set donor dir (a sibling, never preprocessed).

    silnlp's preprocess deletes ``test.*.txt`` in the experiment dir before it
    reads the pin, so the exact test verses must live in a separate directory
    referenced by ``use_test_set_from`` (silnlp/nmt/config.py). This one holds
    only ``test.<src_iso>.<trg_iso>.vref.txt``.
    """
    return f"{rel_root}/{name}__testset"


def donor_vref_name(spec: "SotaSpec") -> str:
    """Per-pair donor filename; silnlp maps the pin by (src_iso, trg_iso)."""
    return f"test.{spec.src_iso}.{spec.trg_iso}.vref.txt"


def build_config(spec: SotaSpec, use_test_set_from: str,
                 model: str = NLLB_DISTILLED_1_3B) -> dict:
    """The silnlp ``config.yml`` contents for one baseline (pure; unit-tested).

    ``use_test_set_from`` is the donor experiment path (relative to
    MT/experiments) whose ``test.<src>.<trg>.vref.txt`` pins the exact test
    set. With no ``corpus_books``/``test_books`` the pair spans the whole
    Bible and silnlp holds out exactly the pinned verses (removing them from
    train, silnlp/nmt/config.py), so the train set is exactly the complement —
    which is how each condition is encoded purely through ``test_vrefs``.
    ``type`` includes ``val`` so silnlp does its default 250-verse
    validation split and best-checkpoint selection (the routine fine-tune).
    """
    lang_codes = {
        spec.src_iso: flores_tag(spec.src_iso, spec.src_script),
        spec.trg_iso: flores_tag(spec.trg_iso, spec.trg_script),
    }
    return {
        "model": model,
        "data": {
            "seed": SILNLP_SEED,
            "lang_codes": lang_codes,
            "corpus_pairs": [{
                "src": sota_stem(spec.src_iso, spec.src_project),
                "trg": sota_stem(spec.trg_iso, spec.trg_project),
                "type": "train,test,val",
                "mapping": "mixed_src",
                "use_test_set_from": use_test_set_from,
            }],
        },
    }


def write_experiment(spec: SotaSpec, collect_dir: Path,
                     rel_root: str, model: str = NLLB_DISTILLED_1_3B) -> Path:
    """Write the baseline's donor test-set dir and its ``config.yml``.

    Creates two folders under ``collect_dir``: ``<name>__testset/`` holding the
    exact test verses (never preprocessed), and ``<name>/`` with the config
    whose ``use_test_set_from`` points at the donor. ``rel_root`` is
    ``collect_dir`` relative to MT/experiments, so the reference resolves on
    the worker.
    """
    donor_dir = collect_dir / f"{spec.name}__testset"
    donor_dir.mkdir(parents=True, exist_ok=True)
    (donor_dir / donor_vref_name(spec)).write_text(
        "\n".join(spec.test_vrefs) + "\n", encoding="utf-8",
    )
    exp_dir = collect_dir / spec.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.yml").write_text(
        yaml.safe_dump(
            build_config(spec, donor_relref(rel_root, spec.name), model),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return exp_dir


def run(exp_name: str, rel_root: str, queue: str | None = None,
        clearml_tag: str = "research") -> list[str]:
    """Build the silnlp argv to preprocess/train/test one baseline, per book.

    The experiment path comes first (before ``--scorers``, whose nargs='*'
    would otherwise swallow it) and ``--scorers chrf3`` comes last. silnlp
    requires ``--clearml-tag`` (one of research/dev/eitl/onboarding) on any
    queued run; these SOTA baselines are tagged ``research``. Returns the
    argv; the caller runs it, gated on free workers.
    """
    argv = [
        "python", "-m", "silnlp.nmt.experiment",
        exp_relref(rel_root, exp_name),
        "--preprocess", "--train", "--test", "--score-by-book",
    ]
    if queue:
        argv += ["--clearml-queue", queue, "--clearml-tag", clearml_tag]
    argv += ["--scorers", "chrf3"]
    return argv
