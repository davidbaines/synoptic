"""Training entry point.

    .venv/bin/python scripts/train.py --config configs/experiments/pilot.yaml

(Experiment repos wrap ``synoptic.train.main`` in a thin ``scripts/train.py``
so ClearML remote execution captures the EXPERIMENT repo — running
``python -m synoptic.train --remote-queue ...`` would make ClearML detect the
synoptic checkout instead, and the agent's clone would lack the configs.)

HF ``Seq2SeqTrainer`` on a randomly-initialised MarianMT model, bf16, label
smoothing, inverse-sqrt schedule. The tokeniser is trained here on the training
split only and saved beside the checkpoint so generation reuses it. ClearML
logging is opt-in (``--clearml``) and degrades gracefully if unavailable.

One sanity mode backs the verification plan:
  --overfit N   train on N pairs only; final loss must fall near zero.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    # Imported for its side effect BEFORE any parse_args call: ClearML
    # patches argparse at import, which (a) records the parsed arguments
    # into the task at enqueue and (b) injects them back when the agent
    # reruns the script with an empty command line. A plain-script entry
    # point carries no argv (the agent does not shell-split it), so without
    # this the remote rerun dies at "--config is required".
    import clearml  # noqa: F401
except ImportError:  # local training without the ClearML stack is fine
    pass

from .config import ExperimentConfig
from .data import VREF_COLUMN, repo_root
from .data_pipeline import prepare
from .dataset import Collator, PairDataset
from .model import build_model
from .preprocess import SRC_COLUMN, TGT_COLUMN
from .splits import assert_no_leakage, manifest_checksum
from .tokenizer import load_tokenizer, train_tokenizer


# Matches the pinned torch (2.6.0+cu124) so the preinstalled torch and its
# native CUDA libraries satisfy the requirement as-is: the agent's venv
# gets torch from the image instead of a wheel whose nvidia-* library
# packages land where the venv's loader cannot see them (v2 smoke attempts
# 5-6 died on libcudnn.so.9 exactly that way).
DEFAULT_DOCKER_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"


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
    if remote_queue:
        # ClearML detects the repo to clone from the entry-point file
        # (sys.argv[0]); with synoptic installed as a dependency that file
        # must live in the EXPERIMENT repo or the agent clones the wrong one.
        entry = Path(sys.argv[0]).resolve()
        if cfg.root.resolve() not in entry.parents:
            raise SystemExit(
                f"--remote-queue requires an entry script inside the "
                f"experiment repo {cfg.root} so ClearML captures that repo; "
                f"run via the repo's scripts/train.py wrapper "
                f"(entry point was {entry})"
            )
        # The requirements capture records synoptic as a bare name==version,
        # which pip would resolve from PyPI (a different project). Force the
        # git source instead; keying by package name makes ClearML REPLACE
        # the captured line rather than adding a second, conflicting one.
        from . import __version__

        if __version__ == "0.0.0":
            raise SystemExit(
                "synoptic is running from a raw checkout (version 0.0.0); "
                "remote agents install it by git tag, so enqueue from an "
                "environment with a released synoptic installed"
            )
        Task.add_requirements(
            "synoptic",
            f"@ git+https://github.com/davidbaines/synoptic@v{__version__}",
        )
        # The capture analyzes the EXPERIMENT repo's imports only; the train
        # stack is imported lazily inside this package, so it never appears
        # there. Force it with the locally installed versions (the local venv
        # tracks the workers' Python for exactly this reason). torch's
        # +cu124 local-version pin goes through clearml-agent's PyTorch
        # index resolution, as the v1 captures did.
        from importlib import metadata

        for pkg in ("torch", "transformers", "accelerate", "matplotlib", "boto3"):
            try:
                Task.add_requirements(pkg, metadata.version(pkg))
            except metadata.PackageNotFoundError:
                raise SystemExit(
                    f"{pkg} is not installed locally; remote runs capture "
                    "requirement versions from this environment"
                )
        # The agent installs the torch wheel without resolving its
        # dependencies, so the CUDA runtime wheels (libcudnn & co) must be
        # pinned explicitly as well — v1's full-freeze capture listed them.
        for dist in metadata.distributions():
            name = (dist.metadata["Name"] or "").lower()
            if name.startswith("nvidia-") or name == "triton":
                Task.add_requirements(name, dist.version)
    # The ClearML project is the experiment repo's directory name.
    task = Task.init(project_name=cfg.root.name, task_name=cfg.name)
    if remote_queue:
        # The queue's default image (python:3.12-bullseye) breaks the agent
        # bootstrap (clearml-agent 2.0.4 imports pkg_resources, dropped by
        # setuptools>=81); a task-specified image sidesteps it.
        docker_args = [
            # ie_big at batch 256 died with 5.6 GiB reserved-but-unallocated;
            # expandable segments avoids that fragmentation on long batches.
            "-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        ]
        # The workers route to the MinIO store's LAN IP but have no DNS for
        # its hostname, while the store cert is valid ONLY for the hostname.
        # Map the hostname to its IP in the container's hosts file (resolved
        # here, where DNS works) so the upload connects by the cert-valid
        # hostname to the routable IP — full TLS, no hardcoded IP.
        import socket
        from urllib.parse import urlparse

        from . import store

        host = urlparse(store.MINIO_ENDPOINT).hostname
        try:
            ip = socket.gethostbyname(host)
        except OSError as e:
            raise SystemExit(
                f"cannot resolve the weights store {host} at enqueue "
                f"({e}); the worker would have no route to it and lose its "
                "weights. Enqueue from a host that resolves the store."
            )
        docker_args.append(f"--add-host {host}:{ip}")
        task.set_base_docker(
            docker_image=docker_image or DEFAULT_DOCKER_IMAGE,
            docker_arguments=" ".join(docker_args),
        )
        # Enqueue and exit locally; the agent reruns the whole script remotely.
        task.execute_remotely(queue_name=remote_queue, exit_process=True)
    return task


def _on_agent() -> bool:
    """True when running inside a ClearML agent (an ephemeral worker)."""
    import os

    return bool(os.environ.get("CLEARML_TASK_ID"))


def _upload_artifacts(output: Path, cfg: ExperimentConfig) -> None:
    """Store the run directory on the shared MinIO store, or fail loudly.

    Only agent runs upload — a local run already has its weights on disk.
    The store transport and its completeness guarantee live in ``store`` and
    raise on an incomplete upload, so a "completed" task always has a full,
    manifest-verified model on the store (store.upload_run)."""
    if not _on_agent():
        return
    from . import store

    store.upload_run(output, repo=cfg.root.name, run=cfg.name)


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


def build_training_args(cfg: ExperimentConfig, output: Path, args,
                        validation_driven: bool):
    from transformers import Seq2SeqTrainingArguments

    per_device = cfg.training.per_device_batch_size or 16
    max_steps = args.max_steps if args.max_steps else cfg.training.max_steps
    eval_steps = min(cfg.training.eval_every_steps, max_steps)
    # Label smoothing floors the loss well above zero, which hides whether the
    # loop is actually memorising the overfit subset; turn it off there so the
    # near-zero-loss check is meaningful.
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
        # When validation drives stopping and best-checkpoint selection we
        # handle "best" ourselves (ValidationStopper), so HF must not also
        # reload an eval_loss-best checkpoint at the end.
        load_best_model_at_end=not validation_driven,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        logging_steps=max(1, min(50, max_steps // 10)),
        seed=cfg.training.seed,
        report_to=["clearml"] if args.clearml else [],
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        predict_with_generate=False,
    )


def run(args) -> None:
    cfg = ExperimentConfig.load(args.config)
    output = Path(args.output_dir) if args.output_dir else cfg.root / "checkpoints" / cfg.name
    output.mkdir(parents=True, exist_ok=True)

    _maybe_clearml(cfg, args.clearml, args.remote_queue, args.docker_image)
    # (With --remote-queue the process has already enqueued and exited by
    # here unless it IS the remote worker, whose clone starts clean.)

    if _on_agent() and not args.overfit:
        # Fail in seconds if the store is unreachable or credentials are
        # missing, rather than after GPU-hours followed by a lost model.
        from . import store

        store.preflight(repo=cfg.root.name, run=cfg.name)

    from .validation import BEST_DIR, VALIDATION_CSV

    stale = [n for n in (VALIDATION_CSV, BEST_DIR) if (output / n).exists()]
    if stale and not args.overfit:
        # A leftover validation curve or best/ checkpoint would be read as
        # this run's state: the stopper would inherit the old best score and
        # save_model could ship the previous run's weights. (--overfit never
        # reads or writes that state, so it may rerun in place.)
        raise SystemExit(
            f"{output} contains a previous run's state ({', '.join(stale)}); "
            "delete the directory or pass a fresh --output-dir"
        )

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
        # (verification: source-side leakage).
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
        for frame_attr in ("valid_pairs", "test_pairs"):
            setattr(data, frame_attr, to_ms_sources(
                getattr(data, frame_attr), data.verses, data.language_of,
                present, ranking, k=cfg.data.k, source_id=data.source_id,
                forbidden=forbidden,
            ))
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

    # Position table must cover the longest sequence any phase produces:
    # training sources/targets AND inference-time decoding.
    max_pos = max(cfg.data.max_src_len or 0, cfg.data.max_len,
                  cfg.inference.max_length) + 4
    model = build_model(cfg.model, sp, max_position_embeddings=max_pos)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {cfg.model.arch} {n_params/1e6:.1f}M params")

    from transformers import EarlyStoppingCallback, Seq2SeqTrainer

    validation_driven = cfg.validation is not None and not args.overfit
    targs = build_training_args(cfg, output, args, validation_driven)
    callbacks = []
    stopper = None
    if validation_driven:
        from .validation import ValidationStopper, build_validation_set

        # Validation verses come from the VALIDATION split (restricted to
        # the target languages), never from the test pairs: stopping and
        # checkpoint selection must not see the verses reported as scores.
        # Sampled from the PRE-length-filter frame so every run of a series
        # validates the identical (vref, translation) pairs regardless of
        # its tokenizer (the length filter is tokenizer-dependent).
        held_out = build_validation_set(
            data.valid_pairs, data.language_of,
            cfg.validation.verses_per_language, cfg.validation.seed,
            translations=data.holdout_translations,
            min_required=cfg.validation.verses_per_language,
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

    # For validation-driven runs the checkpoint shipped downstream is the
    # best-validation one saved by ValidationStopper, not the final-step
    # weights; load it back BEFORE the final evaluation so the recorded
    # eval_loss belongs to the weights save_model actually writes.
    if stopper is not None and stopper.best is not None:
        from .validation import VALIDATION_PNG, plot_curves

        best_dir = output / BEST_DIR
        trainer.model = type(model).from_pretrained(str(best_dir)).to(model.device)
        print(f"  using best-validation checkpoint: chrF3_macro={stopper.best[1]} "
              f"@ step {stopper.best[0]}")
        plot_curves(stopper.csv_path, output / VALIDATION_PNG)

    metrics = trainer.evaluate()
    print(f"  final eval (shipped weights): {metrics}")

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

    if not args.overfit:
        # Overfit is a debug mode (N pairs); its weights must never reach the
        # store — upload_run clears the prefix first and would replace a real
        # same-named model with garbage. Gated exactly like preflight.
        _upload_artifacts(output, cfg)

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
                        "image breaks agent bootstrap)")
    p.add_argument("--generate-after", action="store_true",
                   help="after training, generate + score held-out books and upload them")
    p.add_argument("--gen-batch-size", type=int, default=32)
    run(p.parse_args())


if __name__ == "__main__":
    main()
