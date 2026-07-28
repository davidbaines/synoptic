"""Hugging Face packaging helpers.

Two jobs the publish command delegates here: turning our raw SentencePiece
model into a ``MarianTokenizer`` that loads with ``from_pretrained``, and
rendering the model card. Keeping these pure and file-based makes them testable
without touching the Hub.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


def package_tokenizer(spm_path: Path, out_dir: Path, source_lang: str) -> Path:
    """Write MarianTokenizer files (vocab.json, source/target.spm, config).

    The same SentencePiece model serves both sides. Language tags stay atomic
    because they were trained as user-defined symbols, so they already exist as
    single vocab entries. ``source_lang`` is the run's actual source language
    code (from the experiment config) — a hardcoded value here once shipped
    "grc" on models with no Greek in them.
    """
    import sentencepiece as spm
    from transformers import MarianTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_path))
    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    (out_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
    )
    shutil.copy(spm_path, out_dir / "source.spm")
    shutil.copy(spm_path, out_dir / "target.spm")
    tok = MarianTokenizer(
        vocab=str(out_dir / "vocab.json"),
        source_spm=str(out_dir / "source.spm"),
        target_spm=str(out_dir / "target.spm"),
        source_lang=source_lang,
        target_lang="mul",
        unk_token="<unk>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    tok.save_pretrained(str(out_dir))
    return out_dir


def _yaml_metadata(languages, licence, tags) -> str:
    lang_lines = "\n".join(f"  - {c}" for c in sorted(set(languages)))
    tag_lines = "\n".join(f"  - {t}" for t in tags)
    return (
        "---\n"
        "library_name: transformers\n"
        "pipeline_tag: translation\n"
        f"license: {licence}\n"
        "language:\n"
        f"{lang_lines}\n"
        "tags:\n"
        f"{tag_lines}\n"
        "---\n"
    )


def _summary_metric_table(metrics: pd.DataFrame) -> str:
    """Verse-weighted per-language summary; per-book detail stays in the CSV."""
    if metrics is None or not len(metrics):
        return "Metrics were not recorded for this run."
    import numpy as np

    metric_cols = [c for c in ("chrF3", "spBLEU", "BLEU", "copy_chrF3", "other_chrF3")
                   if c in metrics.columns]
    rows = []
    for (t, lang), g in metrics.groupby(["translation", "language"]):
        row = {"translation": t, "language": lang,
               "books": len(g), "verses": int(g["verses"].sum())}
        for c in metric_cols:
            row[c] = round(float(np.average(g[c], weights=g["verses"])), 2)
        if "other_lang" in g.columns and len(g["other_lang"].mode()):
            row["other_lang"] = g["other_lang"].mode().iloc[0]
        rows.append(row)
    table = pd.DataFrame(rows).to_markdown(index=False)
    return (
        table
        + "\n\nScores are verse-weighted across each language's held-out books. "
        "Per-book metrics for every book are in `generated/metrics.csv`."
    )


def build_model_card(
    *,
    repo_id: str,
    experiment: str,
    n_params: int,
    licences: pd.DataFrame,
    holdouts: dict,
    verse_holdouts: dict | None = None,
    metrics: pd.DataFrame,
    model_licence: str,
    git_commit: str,
    seed: int,
    source_id: str,
    source_lang: str,
    source_name: str,
) -> str:
    """Render the model card (README.md) for a published run.

    ``source_id``/``source_lang``/``source_name`` describe the run's actual
    forced-first source translation (from the experiment config) — the card
    must never claim a source the training data does not contain.
    """
    from . import __version__ as synoptic_version
    languages = sorted(set(licences["languageCode"]))
    tags = ["translation", "bible", "marian", "multilingual", "from-scratch"]
    header = _yaml_metadata(languages, model_licence, tags)

    # The model emits a target tag <2<iso>>, where <iso> is the target's
    # languageCode (NOT its translationId). Derive the real target tag(s) from
    # the held-out targets so the usage example is correct for this model —
    # never a generic placeholder that would make copy-paste output wrong.
    tid_to_lang = dict(zip(licences["translationId"], licences["languageCode"]))
    target_langs = []
    for t in list(holdouts) + list(verse_holdouts or {}):
        lc = tid_to_lang.get(t)
        if lc and lc not in target_langs:
            target_langs.append(lc)
    example_target = target_langs[0] if target_langs else "xxx"
    target_note = (
        f"this model's target tag `<2{example_target}>`"
        if len(target_langs) <= 1
        else "one of this model's target tags "
        + ", ".join(f"`<2{lc}>`" for lc in target_langs)
    )

    holdout_lines = "\n".join(
        f"| `{t}` | {', '.join(b)} |" for t, b in holdouts.items()
    )
    for t, vrefs in (verse_holdouts or {}).items():
        holdout_lines += f"\n| `{t}` | {len(vrefs)}-verse test set |"
    src_table = licences.rename(
        columns={"translationId": "translation", "languageCode": "language"}
    )[["translation", "language", "licence"]].to_markdown(index=False)
    metric_table = _summary_metric_table(metrics)

    body = f"""# {repo_id.split('/')[-1]}

A multilingual, from scratch encoder decoder transformer that drafts Bible
verses in a target language from one or more source renderings of the same
verse. It belongs to the closed text Bible translation series that began as a
reproduction of Sami Liedes' 2018 experiment and now studies multi-source
fusion across language pools. This checkpoint comes from the `{experiment}`
experiment.

The model is trained on verse aligned Bible translations from the eBible corpus.
Whole books (and fixed verse test sets) are withheld from the target languages
during training, and the model then generates that withheld material. The
practical aim is to draft scripture in a language for which parts of the Bible
do not yet exist, using the many translations the model has already seen as
context.

## How to use

```python
from transformers import MarianMTModel, MarianTokenizer

model = MarianMTModel.from_pretrained("{repo_id}")
tokenizer = MarianTokenizer.from_pretrained("{repo_id}")

# Prepend {target_note}, then one or more language-tagged source renderings
# of the verse. This run's forced-first source is {source_name}
# (`{source_id}`, tag <1{source_lang}>).
source = "<2{example_target}> <1{source_lang}> ...verse text in {source_name}..."
batch = tokenizer([source], return_tensors="pt")
generated = model.generate(**batch, num_beams=5, max_length=192)
print(tokenizer.decode(generated[0], skip_special_tokens=True))
```

## Model details

- Architecture: MarianMT style encoder decoder, randomly initialised (no
  pretrained weights).
- Parameters: approximately {n_params / 1e6:.0f} million.
- Source: {source_name} (`{source_id}`), the pool member that aligns best
  with the target languages, always presented first; further renderings from
  other pool members may follow, each behind its own `<1xxx>` tag.
- Target languages: indicated by an atomic `<2xxx>` tag prepended to the
  source renderings.
- Tokeniser: SentencePiece, trained on the training split only.

## Held out material

The following material was withheld from training and generated by the model.

| Translation | Held out |
|---|---|
{holdout_lines}

## Evaluation

Scores are corpus level per held out book, following the silnlp and sacreBLEU
conventions. chrF3 is the headline metric. Two baselines are reported for
context. The source copy baseline (`copy_chrF3`) returns the source verse
unchanged. The other language baseline (`other_chrF3`, with the language in
`other_lang`) is the score of the single most similar training language's own
text for the same verses, which is a demanding floor because a close relative
shares script and vocabulary with the target.

{metric_table}

## Training data and licensing

This model was trained only on translations whose licences permit derivative
works, so that the model may itself be shared. The published model is released
under `{model_licence}`. Every source translation and its licence is listed
below.

{src_table}

## Reproducibility

- Experiment: `{experiment}`
- Experiment repo commit: `{git_commit}`
- Training code: synoptic `{synoptic_version}`
- Random seed: `{seed}`

The full training and evaluation code, configuration and selection list are in
the project repository.

## Acknowledgement

This work reproduces the closed text Bible translation experiment described by
Sami Liedes in his 2018 blog post "Machine translating the Bible into new
languages". The reproduction and this model are independent work and are not
endorsed by him.
"""
    return header + "\n" + body
