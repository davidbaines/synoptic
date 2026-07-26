"""By-script pool selection.

A pool is every shareable text of one script above a coverage floor, minus
explicit exclusions (same-language duplicates that would leak into a
target's test sets), optionally restricted to a language list and one text
per language. ``nt_only_except`` marks every non-target, non-exempt member
``ntOnly`` — its OT is blanked by ``data_pipeline.apply_nt_truncation`` — so
a control pool can mirror another pool's coverage profile.

This module only pre-filters the metadata (script, languages, the OT+NT
coverage floor, shareable licences) and then delegates to
``selection.select_translations``, which owns the dedupe and forced-include
mechanics — targets survive one-per-language dedupe there even when a
same-language sibling has more verses. The concrete pool definitions (which
script, which targets) belong to the experiment repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import load_metadata
from .licensing import selection_licences
from .selection import SelectionConfig, select_translations, write_selection

MIN_OT_NT_VERSES = 7000


@dataclass
class PoolSpec:
    script: str                                  # metadata `script` value
    targets: list[str]                           # target translationIds (holdout YAMLs name them too)
    exclude: list[str] = field(default_factory=list)
    languages: list[str] | None = None           # restrict to these ISO codes
    one_per_language: bool = False               # True where duplicates exist and none is wanted


def build_pool(
    spec: PoolSpec,
    metadata: pd.DataFrame | None = None,
    nt_only_except: list[str] | None = None,
    min_ot_nt_verses: int = MIN_OT_NT_VERSES,
) -> pd.DataFrame:
    """Select one script pool; shareable licences and the coverage floor only.

    ``nt_only_except``: when given, every pool member NOT in the list and NOT
    a target is marked ``ntOnly``. Pass the chosen source id once alignment
    has picked it.
    """
    meta = load_metadata() if metadata is None else metadata

    sub = meta[meta["script"] == spec.script].copy()
    if spec.languages is not None:
        sub = sub[sub["languageCode"].isin(spec.languages)]
    sub = sub[(sub["OTverses"] + sub["NTverses"]) >= min_ot_nt_verses]

    lic = selection_licences(sub, meta)
    sub = sub.merge(lic[["translationId", "licence", "shareable"]], on="translationId")
    sub = sub[sub["shareable"]].drop(columns="shareable")

    missing_targets = set(spec.targets) - set(sub["translationId"])
    if missing_targets:
        raise ValueError(
            f"targets {sorted(missing_targets)} not in the {spec.script} pool"
        )

    pool = select_translations(
        SelectionConfig(
            include=spec.targets,
            exclude=spec.exclude,
            min_verses=0,          # the OT+NT floor above already applied
            one_per_language=spec.one_per_language,
        ),
        metadata=sub,
    )
    # select_translations re-derives family from the curated map and keeps
    # the licence column from the pre-filtered frame.

    keep = set(spec.targets) | set(nt_only_except or [])
    if nt_only_except is not None:
        pool["ntOnly"] = ~pool["translationId"].isin(keep)
    else:
        pool["ntOnly"] = False
    return pool.sort_values("translationId").reset_index(drop=True)


def write_pool(selection: pd.DataFrame, name: str) -> Path:
    return write_selection(selection, name, extra_cols=["licence", "ntOnly"])
