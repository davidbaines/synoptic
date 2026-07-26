"""Licence policy for publishing derived models (spec.md, "Publishing").

The eBible metadata marks every translation ``Redistributable = True``, which
only means the source text may be redistributed. It says nothing about whether
a model trained on that text may be shared, so the meaningful field is
``licence_Licence_Type``. A trained model is treated as a derivative work, so
only licences that permit derivatives are safe to publish a model from.

Policy (conservative by default):

- Shareable, including commercial use: ``Public Domain``, ``by``, ``by-sa``.
  ShareAlike (``by-sa``) propagates to the published model licence.
- Shareable only non-commercially: ``by-nc`` (opt in with ``allow_nc``).
- Never shareable as a derivative: ``by-nd``, ``by-nc-nd``, ``Unknown``, and
  any licence not listed above.
"""

from __future__ import annotations

import pandas as pd

from .data import load_metadata

LICENCE_COLUMN = "licence_Licence_Type"

SHAREABLE = frozenset({"Public Domain", "by", "by-sa"})
SHAREABLE_NC = frozenset({"by-nc"})


def licence_of(translation_ids, metadata: pd.DataFrame | None = None) -> dict[str, str]:
    """Map each translationId to its licence type ('Unknown' if unrecorded)."""
    meta = load_metadata() if metadata is None else metadata
    sub = meta[meta["translationId"].isin(list(translation_ids))]
    out = dict(zip(sub["translationId"], sub[LICENCE_COLUMN].fillna("Unknown")))
    return {t: out.get(t, "Unknown") for t in translation_ids}


def is_shareable(licence: str, allow_nc: bool = False) -> bool:
    allowed = SHAREABLE | SHAREABLE_NC if allow_nc else SHAREABLE
    return licence in allowed


def selection_licences(
    selection: pd.DataFrame, metadata: pd.DataFrame | None = None, allow_nc: bool = False
) -> pd.DataFrame:
    """Return the selection annotated with licence and a shareable flag."""
    lic = licence_of(selection["translationId"], metadata)
    out = selection[["translationId", "languageCode"]].copy()
    out["licence"] = out["translationId"].map(lic)
    out["shareable"] = out["licence"].map(lambda l: is_shareable(l, allow_nc))
    return out.reset_index(drop=True)


def check_shareable(
    selection: pd.DataFrame, metadata: pd.DataFrame | None = None, allow_nc: bool = False
) -> tuple[bool, pd.DataFrame]:
    """Return (all_shareable, offenders). Offenders are the blocking rows."""
    annotated = selection_licences(selection, metadata, allow_nc)
    offenders = annotated[~annotated["shareable"]].reset_index(drop=True)
    return offenders.empty, offenders


def model_licence_for(licences, allow_nc: bool = False,
                      base_model_licence: str | None = None) -> str:
    """The HF licence identifier a model may carry given its source licences.

    ShareAlike is the most restrictive of the permissive set, so any ``by-sa``
    source forces ``cc-by-sa-4.0``; otherwise attribution or public domain.
    Non-commercial sources (when allowed) append the NC clause.

    ``base_model_licence`` covers fine-tunes of a pretrained model: pass the
    base weights' CC licence identifier (e.g. ``cc-by-nc-4.0`` for NLLB-200)
    and its NC/SA clauses propagate to the result — a fine-tune can never be
    more permissive than its base, even on all-Public-Domain data.
    """
    licset = set(licences)
    nc = bool(licset & SHAREABLE_NC)
    sa = "by-sa" in licset
    if base_model_licence:
        nc = nc or "-nc" in base_model_licence
        sa = sa or base_model_licence.replace("-nc", "").startswith("cc-by-sa")
    if sa:
        base = "cc-by-sa-4.0"
    elif "by" in licset or nc or base_model_licence:
        base = "cc-by-4.0"
    else:  # everything is Public Domain, no base-model constraint
        base = "cc0-1.0"
    if nc and (allow_nc or base_model_licence) and base != "cc0-1.0":
        base = base.replace("cc-by", "cc-by-nc")
    return base
