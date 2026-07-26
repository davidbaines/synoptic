"""By-script pool selection (same-script series, spec "Scripts and pools").

A pool is every shareable text of one script above a coverage floor, minus
explicit exclusions (same-language duplicates that would leak into a
target's test sets), optionally restricted to a language list and one text
per language. ``nt_only_except`` marks every non-target, non-exempt member
``ntOnly`` — its OT is blanked at data-assembly time — so a control pool can
mirror another pool's coverage profile.

The concrete pool definitions (which script, which targets, which
exclusions) belong to the experiment repo; this module holds the shared
mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import load_metadata, repo_root
from .licensing import is_shareable, licence_of
from .selection import load_families

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
    a target is marked ``ntOnly`` (its OT is blanked by
    ``data_pipeline.apply_nt_truncation``). Pass the chosen source id once
    alignment has picked it.
    """
    meta = load_metadata() if metadata is None else metadata

    sub = meta[meta["script"] == spec.script].copy()
    if spec.languages is not None:
        sub = sub[sub["languageCode"].isin(spec.languages)]
    sub = sub[~sub["translationId"].isin(spec.exclude)]
    sub = sub[(sub["OTverses"] + sub["NTverses"]) >= min_ot_nt_verses]

    lic = licence_of(sub["translationId"], meta)
    sub["licence"] = sub["translationId"].map(lic)
    sub = sub[sub["licence"].map(lambda l: is_shareable(l, allow_nc=False))]

    if spec.one_per_language:
        sub = (
            sub.sort_values(["totalVerses", "translationId"], ascending=[False, True])
            .groupby("languageCode", as_index=False)
            .first()
        )

    missing_targets = set(spec.targets) - set(sub["translationId"])
    if missing_targets:
        raise ValueError(
            f"targets {sorted(missing_targets)} not in the {spec.script} pool"
        )

    keep = set(spec.targets) | set(nt_only_except or [])
    if nt_only_except is not None:
        sub["ntOnly"] = ~sub["translationId"].isin(keep)
    else:
        sub["ntOnly"] = False

    families = load_families()
    sub["family"] = [families.get(c, "") for c in sub["languageCode"]]
    return sub.sort_values("translationId").reset_index(drop=True)


def write_pool(selection: pd.DataFrame, name: str) -> Path:
    out_dir = repo_root() / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"selection-{name}.csv"
    cols = [
        "translationId", "languageCode", "languageNameInEnglish", "family",
        "script", "OTverses", "NTverses", "DCverses", "totalVerses",
        "licence", "ntOnly",
    ]
    selection[[c for c in cols if c in selection.columns]].to_csv(out_path, index=False)
    return out_path
