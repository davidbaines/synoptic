"""Generation, scoring and sample sheets.

    uv run python -m synoptic.generate --run checkpoints/pilot

Loads a trained checkpoint + its tokeniser, regenerates the exact held-out
books via the shared data pipeline, and writes per-holdout artefacts:
verse-referenced generated books, a metrics table (chrF3 family, spBLEU, BLEU,
plus the source-copy baseline) and a side-by-side sample sheet. Beam search with
a hard length cap; truncations (max-length hits without EOS) are counted and
reported, never silently dropped.

Template mode (``--book`` / ``--lang``) generates any book into any training
language on demand, mirroring Liedes' ``TGT_TEMPLATE`` trick.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .canon import book_of
from .config import ExperimentConfig
from .data import RANGE_MARKER, VREF_COLUMN
from .data_pipeline import prepare
from .evaluate import best_reference_baseline, score, trivial_baselines
from .preprocess import SRC_COLUMN, TGT_COLUMN, normalise, target_tag
from .sheets import make_sheet
from .tokenizer import load_tokenizer


def load_run(run_dir: Path):
    """Load the config, model and tokeniser saved by ``train.py``."""
    from transformers import MarianMTModel

    cfg = ExperimentConfig.load(run_dir / "config.yaml")
    sp = load_tokenizer(run_dir / "tokenizer" / "spm.model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MarianMTModel.from_pretrained(run_dir).to(device).eval()
    return cfg, sp, model, device


@torch.no_grad()
def generate_texts(
    model,
    sp,
    device: str,
    sources: list[str],
    beam: int,
    length_penalty: float,
    max_length: int,
    batch_size: int = 32,
    src_max_length: int | None = None,
) -> tuple[list[str], int]:
    """Beam-decode tagged source strings. Returns (hypotheses, truncations).

    ``max_length`` caps generation; ``src_max_length`` (when set) caps the
    source encoding separately — multi-source inputs are much longer than any
    generated verse.
    """
    pad, eos, bos = sp.pad_id(), sp.eos_id(), sp.bos_id()
    special = {pad, eos, bos}
    hyps: list[str] = []
    truncated = 0
    src_cap = src_max_length or max_length
    for start in range(0, len(sources), batch_size):
        chunk = sources[start : start + batch_size]
        enc = [sp.encode(s, out_type=int)[: src_cap - 1] + [eos] for s in chunk]
        width = max(len(e) for e in enc)
        input_ids = torch.tensor(
            [e + [pad] * (width - len(e)) for e in enc], device=device
        )
        attn = torch.tensor(
            [[1] * len(e) + [0] * (width - len(e)) for e in enc], device=device
        )
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            num_beams=beam,
            length_penalty=length_penalty,
            max_length=max_length,
            early_stopping=True,
        )
        for row in out.tolist():
            truncated += int(len(row) >= max_length and eos not in row)
            ids = [i for i in row if i not in special]
            hyps.append(sp.decode(ids))
    return hyps, truncated


def generate_holdouts(run_dir: Path, out_dir: Path, args) -> pd.DataFrame:
    cfg, sp, model, device = load_run(run_dir)
    beam = args.beam or cfg.inference.beam
    max_length = args.max_length or cfg.inference.max_length
    data = prepare(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.data.pairing == "multi-source":
        # Deterministic multi-source inputs for the held-out material: the
        # forced source + top-ranked non-held-out renderings, exactly as
        # during training eval.
        from .multisource import (
            inference_source_ranking, present_by_vref, to_ms_sources,
        )

        forbidden = set(
            zip(data.splits.test[VREF_COLUMN], data.splits.test["translation"])
        )
        present = present_by_vref(data.splits.train, data.splits.valid)
        ranking = inference_source_ranking(
            data.selection, policy=cfg.data.companion_ranking
        )
        data.test_pairs = to_ms_sources(
            data.test_pairs, data.verses, data.language_of,
            present, ranking, k=cfg.data.k, source_id=data.source_id,
            forbidden=forbidden,
        )

    rows = []
    for translation in sorted(data.test_pairs["translation"].unique()):
        sub = data.test_pairs[data.test_pairs["translation"] == translation]
        if sub.empty:
            continue
        generated: dict[str, tuple[list[str], list[str], int]] = {}
        sub = sub.assign(book=sub[VREF_COLUMN].map(book_of))
        for book, book_pairs in sub.groupby("book", sort=True):
            book_pairs = book_pairs.reset_index(drop=True)
            print(f"Generating {translation}/{book}: {len(book_pairs)} verses ...")
            srcs = book_pairs[SRC_COLUMN].tolist()
            hyps, truncated = generate_texts(
                model, sp, device, srcs,
                beam=beam, length_penalty=cfg.inference.length_penalty,
                max_length=max_length, batch_size=args.batch_size,
                src_max_length=cfg.data.max_src_len,
            )
            refs = book_pairs[TGT_COLUMN].tolist()
            vrefs = book_pairs[VREF_COLUMN].tolist()
            lang = data.language_of.get(translation, translation)
            generated[book] = (vrefs, hyps, truncated)
            _write_book(out_dir, translation, book, vrefs, hyps)
            _write_sheet(out_dir, translation, book, lang, vrefs, hyps, refs)

            row = score_book(
                translation=translation, book=book, vrefs=vrefs, hyps=hyps,
                data=data, truncated=truncated,
            )
            rows.append(row)
            print(f"  chrF3={row['chrF3']} (copy={row['copy_chrF3']}, "
                  f"other={row['other_chrF3']} [{row['other_lang']}]) "
                  f"truncated={truncated}")
        rows.extend(_group_rows(translation, generated, data))

    table = pd.DataFrame(rows)
    if not table.empty:
        table.to_csv(out_dir / "metrics.csv", index=False)
        (out_dir / "metrics.md").write_text(
            table.to_markdown(index=False), encoding="utf-8"
        )
        # Echo the whole table to the console: artifact uploads have failed
        # before (SSL EOF at the file server), and the console log is the one
        # artefact that always survives a remote run.
        print("METRICS_CSV_BEGIN")
        print(table.to_csv(index=False), end="")
        print("METRICS_CSV_END")
    return table


# Named test-set aggregates: the epistle set spans
# three books; the whole-OT aggregate spans every generated OT book. Mark and
# Genesis-250 are single-book test sets, so their plain book rows suffice.
EPISTLE_BOOKS = ("JAS", "1PE", "2PE")


def _group_rows(translation, generated, data) -> list[dict]:
    """Corpus-level rows for multi-book test sets ([epistles], [OT])."""
    from .canon import OT_BOOKS

    groups = []
    epistles = [b for b in EPISTLE_BOOKS if b in generated]
    if len(epistles) > 1:
        groups.append(("[epistles]", epistles))
    ot_books = [b for b in generated if b in OT_BOOKS]
    if len(ot_books) > 1:
        groups.append(("[OT]", ot_books))

    rows = []
    for label, books in groups:
        vrefs = [v for b in books for v in generated[b][0]]
        hyps = [h for b in books for h in generated[b][1]]
        truncated = sum(generated[b][2] for b in books)
        row = score_book(
            translation=translation, book=label, vrefs=vrefs, hyps=hyps,
            data=data, truncated=truncated,
        )
        rows.append(row)
        print(f"  {label} ({len(vrefs)} verses): chrF3={row['chrF3']} "
              f"(copy={row['copy_chrF3']}, other={row['other_chrF3']} "
              f"[{row['other_lang']}])")
    return rows


def other_language_candidates(vrefs, verses, translation, language_of):
    """Text of every other selected translation for these verses.

    Returns {language: [normalised text per vref]}, the pool for the
    best-other-language baseline. Missing verses become empty strings.
    """
    from .preprocess import normalise

    candidates = {}
    index = set(verses.index)
    for other in verses.columns:
        if other == translation:
            continue
        cells = [
            str(verses.at[v, other]) if v in index else "" for v in vrefs
        ]
        texts = [
            "" if c == RANGE_MARKER else normalise(c) for c in cells
        ]
        candidates[language_of.get(other, other)] = texts
    return candidates


def score_book(*, translation, book, vrefs, hyps, data, truncated):
    """Score one generated book against the reference and both baselines."""
    sub = data.test_pairs[
        (data.test_pairs["translation"] == translation)
        & (data.test_pairs[VREF_COLUMN].isin(vrefs))
    ].set_index(VREF_COLUMN)
    refs = [sub.at[v, TGT_COLUMN] for v in vrefs]
    sample_src = sub.at[vrefs[0], SRC_COLUMN] if len(vrefs) else ""
    if " <1" in sample_src:
        # Multi-source lines carry language-tagged renderings; copying the
        # whole line would leak other-language text into the baseline. Keep
        # the baseline as plain source-copy, as in the one-to-many runs.
        # <range> markers copy as empty — they are not text.
        from .preprocess import normalise

        src_plain = [
            "" if (cell := data.source.get(v, "")) == RANGE_MARKER
            else normalise(cell)
            for v in vrefs
        ]
    else:
        src_plain = [
            (sub.at[v, SRC_COLUMN].split(" ", 1)[1] if " " in sub.at[v, SRC_COLUMN]
             else sub.at[v, SRC_COLUMN])
            for v in vrefs
        ]
    metrics = score(hyps, refs)
    copy = trivial_baselines(src_plain, refs)["source-copy"]
    candidates = other_language_candidates(vrefs, data.verses, translation, data.language_of)
    other_lang, other_chrf = best_reference_baseline(refs, candidates)
    return {
        "translation": translation,
        "language": data.language_of.get(translation, translation),
        "book": book,
        "verses": len(vrefs),
        "truncated": truncated,
        **metrics,
        **{f"copy_{k}": v for k, v in copy.items()},
        "other_chrF3": other_chrf,
        "other_lang": other_lang,
    }


def _write_book(out_dir, translation, book, vrefs, hyps):
    lines = [f"{v}\t{h}" for v, h in zip(vrefs, hyps)]
    path = out_dir / f"{translation}-{book}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sheet(out_dir, translation, book, lang, vrefs, hyps, refs):
    hyp_s = pd.Series(hyps, index=vrefs)
    ref_s = pd.Series(refs, index=vrefs)
    sheet = make_sheet(f"{lang} — {book}", hyp_s, ref_s)
    if "|" not in sheet:
        # No configured passage falls in this book (e.g. a Gospel or minor
        # prophet holdout); show the opening verses so the sheet is never blank.
        head = [{"name": f"{book} (opening)", "start": vrefs[0],
                 "end": vrefs[min(9, len(vrefs) - 1)]}]
        sheet = make_sheet(f"{lang} — {book}", hyp_s, ref_s, passages=head)
    (out_dir / f"sheet-{translation}-{book}.md").write_text(sheet, encoding="utf-8")


def generate_template(run_dir: Path, out_dir: Path, args) -> None:
    """Generate one book into one language on demand (no reference needed)."""
    cfg, sp, model, device = load_run(run_dir)
    beam = args.beam or cfg.inference.beam
    max_length = args.max_length or cfg.inference.max_length
    from .data import load_verses

    source = load_verses([cfg.data.source])[cfg.data.source]

    book_mask = source.index.map(book_of) == args.book
    vrefs = [v for v, keep in zip(source.index, book_mask) if keep and source[v]]
    tag = target_tag(args.lang)
    if cfg.data.pairing == "multi-source":
        # Single source rendering; in-distribution thanks to the k_min=1
        # source-dropout during training.
        from .data import load_metadata
        from .preprocess import source_tag

        meta = load_metadata().set_index("translationId")
        src_lang = meta.at[cfg.data.source, "languageCode"]
        stag = source_tag(src_lang)
        srcs = [f"{tag} {stag} {normalise(source[v])}" for v in vrefs]
    else:
        srcs = [f"{tag} {normalise(source[v])}" for v in vrefs]
    print(f"Template: {args.book} -> {tag} ({len(srcs)} verses)")
    hyps, truncated = generate_texts(
        model, sp, device, srcs, beam=beam,
        length_penalty=cfg.inference.length_penalty,
        max_length=max_length, batch_size=args.batch_size,
        src_max_length=cfg.data.max_src_len,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_book(out_dir, f"template-{args.lang}", args.book, vrefs, hyps)
    print(f"Wrote template-{args.lang}-{args.book}.txt (truncated={truncated})")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate and score held-out books")
    p.add_argument("--run", required=True, help="training output dir")
    p.add_argument("--out", default=None, help="results dir (default <run>/generated)")
    p.add_argument("--beam", type=int, default=0)
    p.add_argument("--max-length", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--book", default=None, help="template mode: book code")
    p.add_argument("--lang", default=None, help="template mode: target language code")
    args = p.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out) if args.out else run_dir / "generated"
    if args.book and args.lang:
        generate_template(run_dir, out_dir, args)
    else:
        table = generate_holdouts(run_dir, out_dir, args)
        print(f"\nWrote {len(table)} holdout-book results to {out_dir}")


if __name__ == "__main__":
    main()
