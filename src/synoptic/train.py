"""Training entry point (spec.md, "Training").

    uv run python -m synoptic.train --config configs/experiments/pilot.yaml

HF ``Seq2SeqTrainer`` on a randomly-initialised MarianMT model, bf16, label
smoothing, inverse-sqrt schedule. The tokeniser is trained here on the training
split only and saved beside the checkpoint so generation reuses it. ClearML
logging is opt-in (``--clearml``) and degrades gracefully if unavailable.

Two sanity modes back the verification plan (spec.md #4):
  --overfit N   train on N pairs only; final loss must fall near zero.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .config import ExperimentConfig
from .data import VREF_COLUMN, repo_root
from .data_pipeline import prepare
from .dataset import Collator, PairDataset
from .model import build_model
from .preprocess import SRC_COLUMN, TGT_COLUMN
from .splits import assert_no_leakage, manifest_checksum
from .tokenizer import load_tokenizer, train_tokenizer


DEFAULT_DOCKER_IMAGE = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime"


def _maybe_clearml(
    cfg: ExperimentConfig,
    enable: bool,
    remote_queue: str | None,
    docker_image: str | None = None,
):
    """Init a ClearML task; optionally enqueue it to run on a remote agent.

    With ``remote_queue`` set, ``execute_remotely`` captures this repo's git
    commit, the entry point and args, enqueues the task, and stops the local
    process. A worker on that queue then reruns the same command. Returns the
    Task (or None if ClearML is unavailable and not required for remote).
    """
    if not enable and not remote_queue:
        return None
    try:
        from clearml import Task
    except ImportError:
        if remote_queue:
            raise SystemExit("clearml is required for --remote-queue; install the train extra")
        print("clearml not installed; skipping experiment tracking")
        return None
    # The ClearML project is the repo directory name, so vendored copies of
    # this file log to their own series' project without an edit.
    task = Task.init(project_name=repo_root().name, task_name=cfg.name)
    if remote_queue:
        # The queue's default image (python:3.12-bullseye) breaks the agent
        # bootstrap (clearml-agent 2.0.4 imports pkg_resources, dropped by
        # setuptools>=81); a task-specified image sidesteps it. PYTHONPATH=src
        # makes the src-layout package importable in the agent's repo clone
        # without installing it.
        task.set_base_docker(
            docker_image=docker_image or DEFAULT_DOCKER_IMAGE,
            docker_arguments=(
                "-e PYTHONPATH=src "
                # ie_big at batch 256 died with 5.6 GiB reserved-but-unallocated;
                # expandable segments avoids that fragmentation on long batches.
                "-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ),
        )
        # Enqueue and exit locally; the agent reruns the whole script remotely.
        task.execute_remotely(queue_name=remote_queue, exit_process=True)
    return task


def _upload_artifacts(output: Path) -> None:
    """If running under a ClearML task, upload the output dir for retrieval."""
    try:
        from clearml import Task
    except ImportError:
        return
    task = Task.current_task()
    if task is None:
        return
    # Upload the run directory minus intermediate trainer checkpoints:
    # checkpoint-*/ dirs carry optimizer states (~2.4 GB each at big scale) and
    # the full 5.95 GB ie_big zip hit ENOSPC on the file server, losing the
    # whole artifact. The final model, best/, tokenizer, validation curve and
    # generated books all survive the filter. Staged copy is temporary, so the
    # upload must complete before it is deleted.
    import tempfile

    # The file server kills uploads above the 200-400 MB band (SSL EOF; size
    # probe 2026-07-25), so the archive travels as numbered parts under that
    # size plus a checksum manifest (chunks.py); fetch_weights reassembles.
    # Per-part retry; on a part's final failure WARN and return rather than
    # failing the task — scores are already in the console log.
    import json
    import time

    from .chunks import MANIFEST_ARTIFACT, split_file

    def upload(name: str, obj) -> bool:
        for attempt in range(1, 5):
            try:
                task.upload_artifact(name, artifact_object=obj, wait_on_upload=True)
                return True
            except Exception as e:  # noqa: BLE001 - any transport error
                print(f"  WARNING: upload of {name} attempt {attempt}/4 failed: {e}")
                time.sleep(30 * attempt)
        return False

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / output.name
        shutil.copytree(output, staged, ignore=shutil.ignore_patterns("checkpoint-*"))
        archive = Path(shutil.make_archive(str(Path(tmp) / output.name), "zip", staged))
        parts, manifest = split_file(archive, Path(tmp) / "parts")
        manifest_path = Path(tmp) / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if not upload(MANIFEST_ARTIFACT, str(manifest_path)):
            print("  WARNING: manifest upload abandoned; weights live only on "
                  "the worker. Scores remain in the console log.")
            return
        for part in parts:
            if not upload(part.name, str(part)):
                print(f"  WARNING: {part.name} abandoned; weights incomplete. "
                      "Scores remain in the console log.")
                return
        print(f"  uploaded run archive as {len(parts)} parts + manifest "
              f"to ClearML task {task.id}")


def _leading_tags(src: str) -> list[str]:
    """The leading `<1..>`/`<2..>` language tags on a source line.

    The body after the ``1``/``2`` must be a lowercase language code, which
    distinguishes a real tag (``<2eng>``) from a vref *book* token that also
    starts with a digit (``<1CH>``, ``<2CO>``) or a vtok verse symbol
    (``<1CH_10:10>``) — those are source symbols, not language tags, and must
    not reach the base tokeniser as user-defined pieces.
    """
    out = []
    for tok in src.split(" "):
        body = tok[2:-1]
        if (
            len(tok) > 2
            and tok[0] == "<"
            and tok[-1] == ">"
            and tok[1] in "12"
            and body.isalpha()
            and body.islower()
        ):
            out.append(tok)
        else:
            break
    return out


def train_tokenizer_for(cfg: ExperimentConfig, train_pairs, output: Path):
    """Train (or reuse) the SentencePiece model on the training split only."""
    tok_dir = output / "tokenizer"
    model_path = tok_dir / "spm.model"
    tags = set()
    for s in train_pairs[SRC_COLUMN]:
        tags.update(_leading_tags(s))
    tags = sorted(tags)
    corpus = train_pairs[SRC_COLUMN].tolist() + train_pairs[TGT_COLUMN].tolist()
    train_tokenizer(
        corpus,
        tok_dir / "spm",
        tags=tags,
        vocab_size=cfg.tokenizer.vocab_size,
        model_type=cfg.tokenizer.type,
    )
    return load_tokenizer(model_path)


def build_training_args(cfg: ExperimentConfig, output: Path, args):
    from transformers import Seq2SeqTrainingArguments

    per_device = cfg.training.per_device_batch_size or 16
    max_steps = args.max_steps if args.max_steps else cfg.training.max_steps
    eval_steps = min(cfg.training.eval_every_steps, max_steps)
    # When validation eval drives stopping and best-checkpoint selection, we handle
    # "best" ourselves (validation.ValidationStopper), so HF must not also try to reload
    # an eval_loss-best checkpoint at the end.
    validation_driven = cfg.validation is not None and not args.overfit
    # Label smoothing floors the loss well above zero, which hides whether the
    # loop is actually memorising the overfit subset; turn it off there so the
    # near-zero-loss check (spec.md #4) is meaningful.
    label_smoothing = 0.0 if args.overfit else cfg.model.label_smoothing
    return Seq2SeqTrainingArguments(
        output_dir=str(output),
        max_steps=max_steps,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=per_device,
        gradient_accumulation_steps=cfg.training.gradient_accumulation,
        learning_rate=cfg.training.lr,
        warmup_steps=cfg.training.warmup_steps,
        lr_scheduler_type=cfg.training.lr_scheduler,
        max_grad_norm=cfg.training.max_grad_norm,
        label_smoothing_factor=label_smoothing,
        bf16=cfg.training.bf16,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=not validation_driven,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=max(1, min(50, max_steps // 10)),
        seed=cfg.training.seed,
        report_to=["clearml"] if args.clearml else [],
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        predict_with_generate=False,
    )


def run(args) -> None:
    cfg = ExperimentConfig.load(args.config)
    output = Path(args.output_dir) if args.output_dir else repo_root() / "checkpoints" / cfg.name
    output.mkdir(parents=True, exist_ok=True)

    _maybe_clearml(cfg, args.clearml, args.remote_queue, args.docker_image)

    print(f"Preparing data for '{cfg.name}' ...")
    data = prepare(cfg)
    assert_no_leakage(data.splits, data.holdouts, data.verse_holdouts)

    train_manifest = manifest_checksum(data.train_pairs)
    print(f"  train manifest checksum: {train_manifest}")

    if cfg.data.pairing == "multi-source":
        from .multisource import (
            build_ms_pairs, inference_source_ranking, present_by_vref, to_ms_sources,
        )

        # Held-out cells, asserted against every source-side pick below
        # (spec.md, Verification: source-side leakage).
        forbidden = set(
            zip(data.splits.test[VREF_COLUMN], data.splits.test["translation"])
        )
        train_source_pairs = build_ms_pairs(
            data.splits.train, data.splits.valid, data.verses,
            data.language_of, k=cfg.data.k, k_min=cfg.data.k_min,
            seed=cfg.training.seed, source_id=data.source_id,
            forbidden=forbidden,
        )
        # Valid/test (and thus the validation sets built from them) get
        # deterministic multi-source inputs so eval loss and validation generation
        # match the training format.
        present = present_by_vref(data.splits.train, data.splits.valid)
        ranking = inference_source_ranking(
            data.selection, policy=cfg.data.companion_ranking
        )
        data.valid_pairs = to_ms_sources(
            data.valid_pairs, data.verses, data.language_of,
            present, ranking, k=cfg.data.k, source_id=data.source_id,
            forbidden=forbidden,
        )
        data.test_pairs = to_ms_sources(
            data.test_pairs, data.verses, data.language_of,
            present, ranking, k=cfg.data.k, source_id=data.source_id,
            forbidden=forbidden,
        )
        print(f"  multi-source (K={cfg.data.k}, k_min={cfg.data.k_min}, "
              f"companions={cfg.data.companion_ranking}): "
              f"{len(train_source_pairs)} training pairs")
    else:
        train_source_pairs = data.train_pairs
    print(
        f"  pairs: train={len(train_source_pairs)} valid={len(data.valid_pairs)} "
        f"test={len(data.test_pairs)}"
    )

    sp = train_tokenizer_for(cfg, train_source_pairs, output)
    print(f"  tokenizer: {sp.get_piece_size()} pieces "
          f"(unk={sp.unk_id()} bos={sp.bos_id()} eos={sp.eos_id()} pad={sp.pad_id()})")

    encode = lambda s: sp.encode(s, out_type=int)
    from .preprocess import length_filter

    train_pairs, train_stats = length_filter(
        train_source_pairs, encode, cfg.data.max_len, cfg.data.max_ratio,
        max_src_len=cfg.data.max_src_len,
    )
    valid_pairs, _ = length_filter(
        data.valid_pairs, encode, cfg.data.max_len, cfg.data.max_ratio,
        max_src_len=cfg.data.max_src_len,
    )
    print(f"  length filter (train): {train_stats}")

    if args.overfit:
        train_pairs = train_pairs.head(args.overfit).reset_index(drop=True)
        valid_pairs = train_pairs  # overfit: eval on the same pairs
        print(f"  OVERFIT mode: {len(train_pairs)} pairs")

    train_ds = PairDataset(train_pairs, sp, cfg.data.max_len, cfg.data.max_src_len)
    valid_ds = PairDataset(valid_pairs, sp, cfg.data.max_len, cfg.data.max_src_len)
    collator = Collator(pad_id=sp.pad_id())

    max_pos = max(cfg.data.max_src_len or 0, cfg.data.max_len) + 4
    model = build_model(cfg.model, sp, max_position_embeddings=max_pos)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {cfg.model.arch} {n_params/1e6:.1f}M params")

    from transformers import EarlyStoppingCallback, Seq2SeqTrainer

    targs = build_training_args(cfg, output, args)
    validation_driven = cfg.validation is not None and not args.overfit
    callbacks = []
    stopper = None
    if validation_driven:
        from .validation import ValidationStopper, build_validation_set

        # Probe verses come from the VALIDATION split (restricted to the
        # target languages), never from the test pairs: stopping and
        # checkpoint selection must not see the verses reported as scores.
        # This deviates from the prior two series, which drew this set from test verses
        # (experiments/code-review-findings.md, finding 1.1).
        available = valid_pairs[
            valid_pairs["translation"].isin(data.holdout_translations)
        ]["translation"].value_counts()
        short = {
            t: int(available.get(t, 0))
            for t in data.holdout_translations
            if available.get(t, 0) < cfg.validation.verses_per_language
        }
        if short:
            raise ValueError(
                f"The validation set needs {cfg.validation.verses_per_language} validation "
                f"verses per target language but only found {short}; raise "
                f"valid_size in the holdouts YAML or lower "
                f"validation.verses_per_language"
            )
        held_out = build_validation_set(
            valid_pairs, data.language_of,
            cfg.validation.verses_per_language, cfg.validation.seed,
            translations=data.holdout_translations,
        )
        validation_sets = {"": held_out}
        msg = (f"  validation set: {len(held_out)} verses / "
               f"{held_out['language'].nunique()} langs")
        if cfg.validation.seen_verses_per_language:
            # SEEN validation set: the holdout languages' *trained* verses (their NT),
            # to watch memorisation apart from held-out transfer.
            seen = build_validation_set(
                train_pairs, data.language_of,
                cfg.validation.seen_verses_per_language, cfg.validation.seed,
                translations=data.holdout_translations,
            )
            validation_sets["seen_"] = seen
            msg += f"; seen {len(seen)} verses / {seen['language'].nunique()} langs"
        stop_mode = "early-stop ON" if cfg.validation.early_stop else "run-to-max_steps"
        print(f"{msg}; every {cfg.validation.every_steps} steps; {stop_mode}")
        stopper = ValidationStopper(validation_sets, sp, cfg.validation, output,
                               cfg.inference.max_length, cfg.data.max_src_len)
        callbacks.append(stopper)
    elif not args.overfit:
        callbacks.append(EarlyStoppingCallback(cfg.training.early_stopping_patience))

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"  final eval: {metrics}")

    # For validation-driven runs the checkpoint used downstream is the best-validation
    # one saved by ValidationStopper, not the final-step weights; load it back so
    # save_model writes it as the run's model (spec-vref.md, "Checkpointing").
    if stopper is not None and stopper.best is not None:
        from .validation import BEST_DIR, plot_curves

        best_dir = output / BEST_DIR
        trainer.model = type(model).from_pretrained(str(best_dir)).to(model.device)
        print(f"  using best-validation checkpoint: chrF3_macro={stopper.best[1]} "
              f"@ step {stopper.best[0]}")
        plot_curves(stopper.csv_path, output / "validation.png")

    trainer.save_model(str(output))
    shutil.copy(cfg.path, output / "config.yaml")
    (output / "train_summary.json").write_text(
        json.dumps(
            {
                "name": cfg.name,
                "eval_loss": metrics.get("eval_loss"),
                "n_params": n_params,
                "train_pairs": len(train_pairs),
                "train_manifest": train_manifest,
                "length_filter": train_stats,
                "overfit": args.overfit,
                "validation_best_step": stopper.best[0] if stopper and stopper.best else None,
                "validation_best_chrF3_macro": stopper.best[1] if stopper and stopper.best else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved model + tokenizer to {output}")

    if args.generate_after and not args.overfit:
        from types import SimpleNamespace

        from .generate import generate_holdouts

        print("Generating and scoring held-out books ...")
        gargs = SimpleNamespace(beam=0, max_length=0, batch_size=args.gen_batch_size)
        generate_holdouts(output, output / "generated", gargs)

    _upload_artifacts(output)

    if args.overfit:
        loss = metrics.get("eval_loss", float("inf"))
        threshold = args.overfit_threshold
        status = "PASS" if loss < threshold else "FAIL"
        print(f"OVERFIT {status}: eval_loss={loss:.4f} (threshold {threshold})")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the synoptic MT model")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--overfit", type=int, default=0, help="train on N pairs only")
    p.add_argument("--overfit-threshold", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--clearml", action="store_true")
    p.add_argument("--remote-queue", default=None,
                   help="enqueue this run on a ClearML queue and exit (e.g. jobs_backlog)")
    p.add_argument("--docker-image", default=None,
                   help="docker image for the remote agent (the queue's default "
                        "image breaks agent bootstrap; see spec.md Infrastructure)")
    p.add_argument("--generate-after", action="store_true",
                   help="after training, generate + score held-out books and upload them")
    p.add_argument("--gen-batch-size", type=int, default=32)
    run(p.parse_args())


if __name__ == "__main__":
    main()
