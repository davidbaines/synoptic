"""Criteria-driven translation selection (spec.md, "Translation selection").

Selection is reproducible: criteria live in a YAML config, the curated
language-family map lives in ``configs/language_families.csv``, and the
resulting translation-ID list is written under ``experiments/`` and committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from .data import load_metadata, repo_root, resource_path

FAMILIES_CSV = "configs/language_families.csv"


@dataclass
class SelectionConfig:
    """Criteria plus manual overrides for one selection run."""

    target_size: int | None = None
    include: list[str] = field(default_factory=list)  # translationIds forced in
    exclude: list[str] = field(default_factory=list)  # translationIds forced out
    min_verses: int = 1000
    one_per_language: bool = True
    # Complete-NT pool rule (spec.md, "Family pools"): only translations with
    # at least this many NT books qualify, and the one-per-language dedupe
    # runs after the filter so an OT-heavy incomplete-NT edition can never
    # shadow a complete-NT one. None disables (prior series' behaviour).
    min_nt_books: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SelectionConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**raw)


def load_families() -> dict[str, str]:
    """Curated ISO 639-3 -> family map; advisory, used for diversity buckets."""
    path = resource_path(FAMILIES_CSV)
    df = pd.read_csv(path, comment="#")
    return dict(zip(df["languageCode"], df["family"]))


def _bucket(row: pd.Series, families: dict[str, str]) -> str:
    """Diversity bucket: known family, else script as a weak proxy."""
    fam = families.get(row["languageCode"])
    return fam if fam else f"script:{row.get('script', 'unknown')}"


def select_translations(
    config: SelectionConfig,
    metadata: pd.DataFrame | None = None,
    families: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Apply the selection criteria and return the chosen metadata rows.

    Order of operations: forced excludes, coverage floor, one translation per
    language (highest totalVerses wins), then — if target_size caps the list —
    a round-robin over diversity buckets, most-covered language first, so no
    family or script dominates. Forced includes always survive.
    """
    meta = load_metadata() if metadata is None else metadata
    families = load_families() if families is None else families

    meta = meta[~meta["translationId"].isin(config.exclude)]
    forced = meta[meta["translationId"].isin(config.include)]
    missing = sorted(set(config.include) - set(forced["translationId"]))
    if missing:
        raise ValueError(
            f"Forced include(s) {missing} not in the candidate metadata — a "
            f"target would silently vanish from the pool (check ids, licence "
            f"filtering, and family membership)"
        )
    pool = meta[
        ~meta["translationId"].isin(config.include)
        & (meta["totalVerses"] >= config.min_verses)
    ]
    if config.min_nt_books is not None:
        pool = pool[
            pd.to_numeric(pool["NTbooks"], errors="coerce").fillna(0)
            >= config.min_nt_books
        ]

    if config.one_per_language:
        taken_langs = set(forced["languageCode"])
        pool = pool[~pool["languageCode"].isin(taken_langs)]
        pool = (
            pool.sort_values("totalVerses", ascending=False)
            .groupby("languageCode", as_index=False)
            .first()
        )

    if config.target_size is None:
        chosen = pool
    else:
        remaining = config.target_size - len(forced)
        if remaining <= 0:
            chosen = pool.iloc[0:0]
        else:
            pool = pool.copy()
            pool["bucket"] = pool.apply(_bucket, axis=1, families=families)
            pool = pool.sort_values("totalVerses", ascending=False)
            picks: list[int] = []
            queues = {b: list(g.index) for b, g in pool.groupby("bucket", sort=False)}
            while len(picks) < remaining and any(queues.values()):
                for b in list(queues):
                    if queues[b]:
                        picks.append(queues[b].pop(0))
                        if len(picks) >= remaining:
                            break
            chosen = pool.loc[picks].drop(columns="bucket")

    result = pd.concat([forced, chosen]).reset_index(drop=True)
    result["family"] = [
        families.get(code, "") for code in result["languageCode"]
    ]
    return result


def write_selection(selection: pd.DataFrame, name: str) -> Path:
    """Write a committed selection list under experiments/."""
    out_dir = repo_root() / "experiments"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"selection-{name}.csv"
    cols = [
        "translationId", "languageCode", "languageNameInEnglish", "family",
        "script", "OTverses", "NTverses", "DCverses", "totalVerses",
    ]
    selection[[c for c in cols if c in selection.columns]].to_csv(
        out_path, index=False
    )
    return out_path
