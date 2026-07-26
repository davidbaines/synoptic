"""Metrics per silnlp/machine.py conventions.

chrF3 (headline), chrF3+, chrF3++, spBLEU (Flores-200 tokeniser) and BLEU,
all via sacreBLEU. Scores are corpus-level per held-out book per language.
"""

from __future__ import annotations

from typing import Sequence

from sacrebleu.metrics import BLEU, CHRF

METRIC_NAMES = ("chrF3", "chrF3+", "chrF3++", "spBLEU", "BLEU")

# Constructed once: sacreBLEU metrics keep no per-call state across
# corpus_score, and BLEU(tokenize="flores200") loads the Flores-200
# SentencePiece model from disk at construction — per-call construction cost
# ~90 scored rows x 2 baselines per run at this series' per-OT-book scale.
_METRICS = {
    "chrF3": CHRF(char_order=6, word_order=0, beta=3),
    "chrF3+": CHRF(char_order=6, word_order=1, beta=3),
    "chrF3++": CHRF(char_order=6, word_order=2, beta=3),
    "spBLEU": BLEU(tokenize="flores200"),
    "BLEU": BLEU(),
}
_CHRF3 = _METRICS["chrF3"]


def score(hypotheses: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Corpus-level scores for one (book, language) pair.

    ``hypotheses`` and ``references`` are verse-aligned lists of equal length.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"{len(hypotheses)} hypotheses vs {len(references)} references"
        )
    refs = [list(references)]
    return {
        name: round(metric.corpus_score(list(hypotheses), refs).score, 2)
        for name, metric in _METRICS.items()
    }


def trivial_baselines(
    sources: Sequence[str], references: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Baselines any real system must beat.

    ``source-copy``: emit the (untagged) source verse unchanged.
    """
    return {"source-copy": score(sources, references)}


def best_reference_baseline(
    references: Sequence[str], candidates: dict[str, Sequence[str]]
) -> tuple[str, float]:
    """Strongest "copy another language" baseline.

    ``candidates`` maps a language label to that language's text for the same
    verses, aligned to ``references``. Returns the label and chrF3 of the
    best-scoring candidate. This is a far more demanding floor than source-copy,
    because a close relative shares script and vocabulary with the target.
    Candidates with no text at all are skipped (scoring an empty corpus is
    wasted work — most pool members have no OT). Returns ("", 0.0) if there
    are no candidates.
    """
    refs = [list(references)]
    best_lang, best_score = "", 0.0
    for lang, texts in candidates.items():
        texts = list(texts)
        if not any(texts):
            continue
        s = _CHRF3.corpus_score(texts, refs).score
        if s > best_score:
            best_lang, best_score = lang, s
    return best_lang, round(best_score, 2)
