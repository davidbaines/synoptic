"""Publish a trained run to the Hugging Face Hub.

    uv run python -m synoptic.publish --run checkpoints/ie_base --dry-run

The command is deliberately gated. It refuses to publish unless:

1. the generated held out books beat the source copy baseline on chrF3 for
   every holdout (run ``synoptic.generate`` first to produce metrics), and
2. every training translation carries a licence that permits sharing a derived
   model (see ``synoptic.licensing``).

If both pass it assembles a self contained repository (model, loadable
tokeniser, generated books, sample sheets, metrics and a model card), shows a
summary, asks for confirmation, and pushes. ``--dry-run`` stops before the push
and leaves the staging folder for inspection.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .data import repo_root
from .data_pipeline import load_holdouts
from .hf_export import build_model_card, package_tokenizer
from .licensing import check_shareable, model_licence_for, selection_licences

MODEL_FILE_GLOBS = ("config.json", "generation_config.json", "model*.safetensors",
                    "pytorch_model*.bin")


def default_repo_id(name: str) -> str:
    return f"DavidCBaines/ebible_m2m-{name.replace('_', '-')}"


def git_commit() -> str:
    """The EXPERIMENT repo's commit (configs, selections, holdouts).

    The training-code version is the synoptic package version, recorded
    separately on the model card — repo_root() is the caller's repo, not
    this package's checkout.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True
        ).strip()
    except Exception:
        return "unknown"


def check_gate(metrics: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    """Quality gate on the generated held-out books.

    Two conditions must hold. Every book must beat the source-copy baseline on
    chrF3 (a floor against degenerate output). And, where the stronger
    other-language baseline is recorded, each held-out language must beat it on
    a verse-weighted average (a real bar, checked per language rather than per
    book so an occasional near-tie on a single book does not block a model that
    is clearly better overall).
    """
    verdict = metrics.copy()
    verdict["beats_baseline"] = verdict["chrF3"] > verdict["copy_chrF3"]
    passed = bool(verdict["beats_baseline"].all())

    if "other_chrF3" in verdict.columns:
        import numpy as np

        for _, g in verdict.groupby("translation"):
            model = np.average(g["chrF3"], weights=g["verses"])
            other = np.average(g["other_chrF3"], weights=g["verses"])
            if model <= other:
                passed = False
    return passed, verdict


def _copy_model_files(run: Path, staging: Path) -> None:
    for pattern in MODEL_FILE_GLOBS:
        for src in run.glob(pattern):
            shutil.copy(src, staging / src.name)


def assemble(run: Path, staging: Path, allow_nc: bool) -> dict:
    """Build the staging folder and return a summary; raises on gate failure."""
    cfg = ExperimentConfig.load(run / "config.yaml")
    metrics_path = run / "generated" / "metrics.csv"
    if not metrics_path.exists():
        raise SystemExit(
            f"No metrics at {metrics_path}. Run synoptic.generate on this run first."
        )
    metrics = pd.read_csv(metrics_path)

    passed, verdict = check_gate(metrics)
    if not passed:
        bad = verdict[~verdict["beats_baseline"]]
        raise SystemExit(
            "Quality gate FAILED: these holdouts do not beat the source-copy "
            f"baseline on chrF3:\n{bad.to_string(index=False)}"
        )

    selection = pd.read_csv(cfg.resolve(cfg.data.selection), dtype=str)
    ok, offenders = check_shareable(selection, allow_nc=allow_nc)
    if not ok:
        raise SystemExit(
            "Licence gate FAILED: these training translations do not permit "
            f"sharing a derived model:\n{offenders.to_string(index=False)}\n"
            "Build a selection restricted to redistributable licences and "
            "retrain before publishing."
        )

    lic = selection_licences(selection, allow_nc=allow_nc)
    model_licence = model_licence_for(lic["licence"], allow_nc=allow_nc)
    holdouts, verse_holdouts, _, seed = load_holdouts(cfg)

    # The run's actual source translation: never a hardcoded language
    # (2026-07-25 review, finding 1.6).
    source_id = cfg.data.source
    src_rows = selection[selection["translationId"] == source_id]
    if len(src_rows):
        source_lang = src_rows["languageCode"].iloc[0]
        source_name = src_rows["languageNameInEnglish"].iloc[0]
    else:
        from .data import load_metadata

        meta = load_metadata().set_index("translationId")
        source_lang = meta.at[source_id, "languageCode"]
        source_name = meta.at[source_id, "languageNameInEnglish"]

    summary = json.loads((run / "train_summary.json").read_text())
    repo_id = default_repo_id(cfg.name)

    # assemble staging folder
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    _copy_model_files(run, staging)
    package_tokenizer(run / "tokenizer" / "spm.model", staging, source_lang=source_lang)
    shutil.copytree(run / "generated", staging / "generated")
    (staging / "experiment").mkdir()
    shutil.copy(run / "config.yaml", staging / "experiment" / "config.yaml")
    shutil.copy(cfg.resolve(cfg.data.selection), staging / "experiment" / "selection.csv")

    card = build_model_card(
        repo_id=repo_id,
        experiment=cfg.name,
        n_params=summary.get("n_params", 0),
        licences=lic,
        holdouts=holdouts,
        verse_holdouts=verse_holdouts,
        metrics=metrics,
        model_licence=model_licence,
        git_commit=git_commit(),
        seed=seed,
        source_id=source_id,
        source_lang=source_lang,
        source_name=source_name,
    )
    (staging / "README.md").write_text(card, encoding="utf-8")

    return {
        "repo_id": repo_id,
        "experiment": cfg.name,
        "model_licence": model_licence,
        "languages": sorted(set(lic["languageCode"])),
        "holdouts": list(holdouts),
        "metrics": metrics,
        "staging": staging,
    }


def push(staging: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(folder_path=str(staging), repo_id=repo_id, repo_type="model")


def run(args) -> None:
    run_dir = Path(args.run)
    staging = Path(args.staging_dir) if args.staging_dir else run_dir / "hf_staging"
    info = assemble(run_dir, staging, allow_nc=args.allow_nc)
    repo_id = args.repo or info["repo_id"]

    print(f"\nReady to publish '{info['experiment']}' -> {repo_id}")
    print(f"  visibility : {'private' if args.private else 'public'}")
    print(f"  licence    : {info['model_licence']}")
    print(f"  languages  : {len(info['languages'])}")
    print(f"  holdouts   : {', '.join(info['holdouts'])}")
    print(f"  staging    : {staging}")
    import numpy as np

    print("  chrF3 by language (model vs baselines):")
    for t, g in info["metrics"].groupby("translation"):
        w = lambda c: round(np.average(g[c], weights=g["verses"]), 2)
        other = f", other-lang {w('other_chrF3')}" if "other_chrF3" in g else ""
        print(f"    {t}: model {w('chrF3')} (source-copy {w('copy_chrF3')}{other})")

    if args.dry_run:
        print("\nDry run: staging built, nothing pushed.")
        return
    if not args.yes:
        reply = input("\nPush to the Hub? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            return
    push(staging, repo_id, private=args.private)
    print(f"Published: https://huggingface.co/{repo_id}")


def main() -> None:
    p = argparse.ArgumentParser(description="Publish a run to the Hugging Face Hub")
    p.add_argument("--run", required=True, help="training output dir")
    p.add_argument("--repo", default=None, help="override the repo id")
    p.add_argument("--staging-dir", default=None)
    p.add_argument("--private", action="store_true", help="push to a private repo")
    p.add_argument("--allow-nc", action="store_true",
                  help="also treat by-nc sources as shareable (non-commercial model)")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--dry-run", action="store_true", help="build staging, do not push")
    run(p.parse_args())


if __name__ == "__main__":
    main()
