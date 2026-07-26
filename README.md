# synoptic

Shared toolkit for the closed-text Bible machine-translation series
(`m2m_bible_mt` → `ebible-mt` → `bible-interlingua` → `bible-mt-same-script`
→ `bible-mt-family-transfer`). The purpose of the line of work is machine
translation for languages whose only available text is parts of the Bible;
everything trains only on the eBible corpus
(`DavidCBaines/ebible_corpus`), shareable licences only.

The name: multi-source fusion reads several renderings of the same verse
side by side — seeing them together, synoptically. The package descends from
code originally inspired by Sami Liedes' Bible-MT experiments.

## What it provides

- **Data**: eBible corpus loading (`data`), licence gates (`licensing`),
  selection by criteria, family or script (`selection`, `family`,
  `script_pool`), book- and verse-level test-set holdouts with leakage
  assertions (`splits`, `data_pipeline`).
- **Pairing**: one-to-many (single source) and multi-source K-rendering
  fusion (`multisource`), with `<range>`-marker filtering and source-side
  leakage checks.
- **Training**: HF Seq2Seq training with a validation set that is disjoint
  from both the training and test material, silnlp-style early stopping
  (chrF3 every 1000 steps, stop when no +0.2 gain within 4000 steps), and
  best-checkpoint selection (`train`, `validation`).
- **Evaluation**: chrF3-family metrics, copy and best-other-language
  baselines, per-book and named-test-set reporting (`evaluate`, `generate`,
  `sheets`).
- **Run plumbing**: ClearML remote execution, chunked artifact upload (the
  file server rejects uploads above ~200 MB), score and weight recovery
  (`fetch_scores`, `fetch_weights`), HF export with correct source-language
  metadata (`hf_export`, `publish`).

## Use from an experiment repo

```toml
[project]
dependencies = [
  "synoptic @ git+https://github.com/davidbaines/synoptic@v0.1.0",
]
```

Pin a tag: a fix must never silently change a running series. Experiment
repos hold their own selections, holdout YAMLs, experiment configs and
result files; this package holds everything two experiments would otherwise
duplicate.

## History

Extracted 2026-07-26 from the (fixed) vendored copy in
`bible-mt-family-transfer` after a code review found defects that had been
live during the same-script series v1; the findings and their fixes are
documented in that repo's `experiments/code-review-findings.md`. Earlier
vendored copies in `bible-interlingua` and `m2m_bible_mt` predate the fixes.

Licence: Apache-2.0. Models trained on eBible shareable selections publish
under cc-by-sa-4.0 (ShareAlike propagates from by-sa sources).
