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


def extract_stem(iso: str, project: str) -> str:
    """silnlp corpus file stem ``<iso>-<project>`` (project = our translationId)."""
    return f"{iso}-{project}"


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
    written = []
    for tid in sorted(set(translation_ids)):
        iso = meta.at[tid, "languageCode"]
        col = verses[tid]
        # Corpus already gives "" for missing; keep <range> as silnlp does.
        lines = ["" if v is None else str(v) for v in col.tolist()]
        path = scripture_dir / f"{extract_stem(iso, tid)}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)
    return written


def build_config(spec: SotaSpec, use_test_set_from: str,
                 model: str = NLLB_DISTILLED_1_3B) -> dict:
    """The silnlp ``config.yml`` contents for one baseline (pure; unit-tested).

    ``use_test_set_from`` is the experiment path (relative to MT/experiments)
    whose ``test.vref.txt`` pins the exact test set — normally this run itself.
    Training is everything else: with no ``corpus_books`` the pair spans the
    whole Bible and the test verses (and only those) are held out, so the train
    set is exactly the complement — which is how we encode each condition
    purely through ``test_vrefs``.
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
                "src": extract_stem(spec.src_iso, spec.src_project),
                "trg": extract_stem(spec.trg_iso, spec.trg_project),
                "type": "train,test",
                "mapping": "mixed_src",
                "use_test_set_from": use_test_set_from,
            }],
        },
    }


def write_experiment(spec: SotaSpec, collect_dir: Path,
                     rel_root: str, model: str = NLLB_DISTILLED_1_3B) -> Path:
    """Write ``config.yml`` + ``test.vref.txt`` for one baseline.

    ``collect_dir`` is the on-disk experiment folder
    (``.../MT/experiments/<rel_root>/<name>``); ``rel_root`` is that folder's
    path relative to ``MT/experiments`` so the config can self-reference its
    own test set via ``use_test_set_from``.
    """
    exp_dir = collect_dir / spec.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    self_ref = f"{rel_root}/{spec.name}"
    (exp_dir / "config.yml").write_text(
        yaml.safe_dump(build_config(spec, self_ref, model), sort_keys=False),
        encoding="utf-8",
    )
    (exp_dir / "test.vref.txt").write_text(
        "\n".join(spec.test_vrefs) + "\n", encoding="utf-8",
    )
    return exp_dir


def run(exp_name: str, rel_root: str, queue: str | None = None,
        max_steps: int | None = None) -> list[str]:
    """Build the silnlp command to preprocess/train/test one baseline per-book.

    Returns the argv (the caller runs it, gated on free workers). ``max_steps``
    overrides silnlp's default only for smoke checks (passed via env the config
    would normally hold; here we keep it explicit for the caller to inject).
    """
    argv = [
        "python", "-m", "silnlp.nmt.experiment",
        "--preprocess", "--train", "--test",
        "--by-book", "--scorers", "chrf3",
        f"{rel_root}/{exp_name}",
    ]
    if queue:
        argv[3:3] = ["--clearml-queue", queue]
    return argv
