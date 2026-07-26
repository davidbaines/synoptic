"""Held-out validation evaluation during training.

Every ``validation.every_steps`` optimiser steps, a fixed seeded sample of held-out
verses is generated greedily and scored (chrF3 + BLEU per holdout language and
macro-averaged). The scores drive early stopping — training halts once macro
chrF3 has gained less than ``min_gain`` over the last ``patience_steps`` — and
best-checkpoint selection. Everything lands in ``validation.csv`` in the run
directory, from which curves are plotted; no external tracker is involved.

The validation verses are sampled from the *validation* split, restricted to
the target languages — never from the test pairs, so stopping and checkpoint
selection are blind to the verses reported as scores (the prior two series
drew this set from test verses; 2026-07-25 review, finding 1.1). Under the
drafting condition the target's validation verses are NT verses, so the
curve watches in-domain transfer while the reported result is
cross-testament — stopping slightly early is possible but unbiased.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

from .config import ValidationConfig
from .data import VREF_COLUMN
from .preprocess import SRC_COLUMN, TGT_COLUMN

try:  # transformers only exists in the train extra; the pure pieces
    from transformers import TrainerCallback  # (sampling, stop rule, plot)
except ImportError:  # remain importable and testable without it.
    TrainerCallback = object

VALIDATION_CSV = "validation.csv"
VALIDATION_PNG = "validation.png"
BEST_DIR = "best"


def build_validation_set(
    pairs: pd.DataFrame,
    language_of: dict[str, str],
    verses_per_language: int,
    seed: int,
    translations: list[str] | None = None,
    min_required: int | None = None,
) -> pd.DataFrame:
    """A fixed, seeded sample of verses per translation.

    Deterministic regardless of the incoming row order: rows are sorted by
    (translation, vref) before sampling, so every run of a series draws the
    same (vref, translation) pairs. Pass ``translations`` to restrict to
    specific translation ids (used for the seen-verse set: the holdout
    languages' *trained* verses).

    ``min_required`` makes a shortfall an error instead of a silent smaller
    sample — the held-out validation set must be full size or the run's
    early stopping is not comparable to its peers. When None, shortfalls are
    allowed but reported.
    """
    frame = pairs
    if translations is not None:
        frame = frame[frame["translation"].isin(translations)]
    counts = frame["translation"].value_counts()
    wanted = translations if translations is not None else sorted(counts.index)
    short = {t: int(counts.get(t, 0)) for t in wanted
             if counts.get(t, 0) < verses_per_language}
    if short and min_required is not None:
        raise ValueError(
            f"validation set needs {min_required} verses per language but "
            f"only found {short}; raise valid_size in the holdouts YAML or "
            f"lower validation.verses_per_language"
        )
    if short:
        print(f"  note: validation sample smaller than requested for {short}")
    out = []
    for translation, sub in frame.groupby("translation", sort=True):
        sub = sub.sort_values(VREF_COLUMN, kind="stable").reset_index(drop=True)
        n = min(verses_per_language, len(sub))
        if n == 0:
            continue
        out.append(sub.sample(n=n, random_state=seed).sort_index())
    if not out:
        return pairs.iloc[0:0].assign(language=[])
    validation = pd.concat(out).reset_index(drop=True)
    validation["language"] = validation["translation"].map(lambda t: language_of.get(t, t))
    return validation


def validation_scores(hyps: Sequence[str], validation: pd.DataFrame) -> dict[str, float]:
    """chrF3 and BLEU per language plus macro averages, as one flat row."""
    chrf = CHRF(char_order=6, word_order=0, beta=3)
    bleu = BLEU()
    frame = validation.assign(hyp=list(hyps))
    row: dict[str, float] = {}
    per_lang_chrf, per_lang_bleu = [], []
    for lang, sub in frame.groupby("language", sort=True):
        refs = [sub[TGT_COLUMN].tolist()]
        c = round(chrf.corpus_score(sub["hyp"].tolist(), refs).score, 2)
        b = round(bleu.corpus_score(sub["hyp"].tolist(), refs).score, 2)
        row[f"chrF3_{lang}"] = c
        row[f"BLEU_{lang}"] = b
        per_lang_chrf.append(c)
        per_lang_bleu.append(b)
    row["chrF3_macro"] = round(sum(per_lang_chrf) / len(per_lang_chrf), 2)
    row["BLEU_macro"] = round(sum(per_lang_bleu) / len(per_lang_bleu), 2)
    return row


def should_stop(
    history: Sequence[tuple[int, float]], patience_steps: int, min_gain: float
) -> bool:
    """The stopping rule: best macro chrF3 must have gained ``min_gain``
    over the value it had ``patience_steps`` ago.

    ``history`` is [(step, macro chrF3)] in step order. Never stops before a
    full patience window exists.
    """
    if not history:
        return False
    current_step = history[-1][0]
    earlier = [m for s, m in history if s <= current_step - patience_steps]
    if not earlier:
        return False
    best_now = max(m for _, m in history)
    return best_now - max(earlier) < min_gain


class ValidationStopper(TrainerCallback):
    """Validation + (optional) early-stop + best-checkpoint callback.

    ``validation_sets`` maps a column prefix to a validation frame: the held-out set uses
    prefix ``""`` and drives best-checkpoint selection and (if enabled) early
    stopping; the seen-verse set uses prefix ``"seen_"`` and is logged only, to
    watch memorisation separately from transfer. Set ``cfg.early_stop`` False to
    run to ``max_steps`` regardless (diagnostic re-runs).
    """

    STOP_KEY = "chrF3_macro"  # column (in the held-out, unprefixed set) to stop on

    def __init__(self, validation_sets: dict[str, pd.DataFrame], sp, cfg: ValidationConfig,
                 output: Path, max_length: int, src_max_length: int | None = None):
        self.validation_sets = validation_sets
        self.sp = sp
        self.cfg = cfg
        self.output = Path(output)
        self.max_length = max_length
        self.src_max_length = src_max_length  # multi-source: longer source cap
        self.csv_path = self.output / VALIDATION_CSV
        self.history: list[tuple[int, float]] = []
        # (step, held-out macro chrF3) of the checkpoint saved in best/.
        self.best: tuple[int, float] | None = None
        # No resume-from-CSV: a leftover curve would seed self.best with the
        # previous run's peak (premature stopping, stale best/ weights).
        # train.py refuses to start into a directory that has one.

    def _generate(self, model, validation: pd.DataFrame) -> list[str]:
        import torch

        from .generate import generate_texts

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            hyps, _ = generate_texts(
                model, self.sp, device, validation[SRC_COLUMN].tolist(),
                beam=1, length_penalty=1.0, max_length=self.max_length,
                batch_size=self.cfg.batch_size,
                src_max_length=self.src_max_length,
            )
        if was_training:
            model.train()
        return hyps

    def run_validation(self, model, step: int) -> dict[str, float]:
        started = time.time()
        row: dict[str, float] = {"step": step}
        for prefix, validation in self.validation_sets.items():
            if validation is None or len(validation) == 0:
                continue
            scored = validation_scores(self._generate(model, validation), validation)
            row.update({f"{prefix}{k}": v for k, v in scored.items()})
        row["seconds"] = round(time.time() - started, 1)
        pd.DataFrame([row]).to_csv(
            self.csv_path, mode="a", header=not self.csv_path.exists(), index=False
        )
        held_out_macro = row[self.STOP_KEY]
        self.history.append((step, held_out_macro))
        # ``best`` always describes the checkpoint in best/ on disk, so what
        # train.py reloads and reports is exactly what was saved. The ~GB
        # checkpoint is rewritten only when the gain clears save_min_gain
        # (early training improves at nearly every evaluation; unconditional
        # saves hammer the disk for nothing). The stop rule reads ``history``,
        # so small unsaved improvements still count toward patience.
        if self.best is None or held_out_macro - self.best[1] >= self.cfg.save_min_gain:
            self.best = (step, held_out_macro)
            best_dir = self.output / BEST_DIR
            model.save_pretrained(best_dir)
            (best_dir / "best.json").write_text(
                json.dumps({"step": step, "chrF3_macro": held_out_macro}),
                encoding="utf-8",
            )
        return row

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.cfg.every_steps:
            return control
        if not state.is_world_process_zero:
            return control
        row = self.run_validation(model, state.global_step)
        seen = row.get("seen_chrF3_macro")
        seen_str = f" seen_chrF3={seen}" if seen is not None else ""
        print(
            f"  validation @ {state.global_step}: chrF3_macro={row['chrF3_macro']}"
            f"{seen_str} best={self.best[1]}@{self.best[0]} ({row['seconds']}s)"
        )
        if self.cfg.early_stop and should_stop(
            self.history, self.cfg.patience_steps, self.cfg.min_gain
        ):
            print(
                f"  validation early stop: chrF3_macro gained <{self.cfg.min_gain} "
                f"over the last {self.cfg.patience_steps} steps"
            )
            control.should_training_stop = True
        return control


def plot_curves(csv_path: Path, png_path: Path) -> Path | None:
    """chrF3 (solid) and BLEU (dashed) vs steps, per language + macro.

    Returns None (with a message) if matplotlib is not installed — the CSV is
    the artefact of record; the PNG is a convenience.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping validation plot")
        return None

    frame = pd.read_csv(csv_path)
    langs = sorted(c[len("chrF3_"):] for c in frame.columns
                   if c.startswith("chrF3_") and c != "chrF3_macro")
    has_seen = "seen_chrF3_macro" in frame.columns
    fig, ax = plt.subplots(figsize=(9, 5))
    for lang in langs:
        ax.plot(frame["step"], frame[f"chrF3_{lang}"],
                label=f"held-out {lang}", alpha=0.6)
        if has_seen and f"seen_chrF3_{lang}" in frame.columns:
            ax.plot(frame["step"], frame[f"seen_chrF3_{lang}"], linestyle=":",
                    label=f"seen {lang}", alpha=0.5)
    ax.plot(frame["step"], frame["chrF3_macro"], label="held-out macro",
            color="black", linewidth=2)
    if has_seen:
        ax.plot(frame["step"], frame["seen_chrF3_macro"], label="seen macro",
                color="black", linewidth=2, linestyle=":")
    ax.set_xlabel("optimiser steps")
    ax.set_ylabel("chrF3")
    ax.set_title(csv_path.parent.name)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return png_path
