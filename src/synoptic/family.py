"""Pool selection over a membership CSV.

Take every eBible language whose ISO 639-3 code is listed in a membership file
(``configs/families/<family>.csv``, or any CSV of the same shape via
``--members`` — gradient-step pools are just other membership lists), keep the
best-covered complete-NT translation per language, and force the target
editions in from the full metadata — a target may legitimately sit outside the
membership list (gradient steps exclude the target's own family by design),
and a target that cannot be forced in is an error, never a silent drop. Run:

    python -m synoptic.family --family bantu --holdouts configs/holdouts-bantu-drafting.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .data import load_metadata, repo_root
from .licensing import is_shareable, licence_of
from .selection import SelectionConfig, load_families, select_translations, write_selection

MIN_VERSES = 5000
MIN_NT_BOOKS = 27   # the complete-NT pool rule


def load_family_codes(family: str) -> dict[str, str]:
    """Read the ISO 639-3 -> branch map for a named family."""
    from .data import resource_path

    return _load_members(resource_path(f"configs/families/{family}.csv"))


def _load_members(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, comment="#")
    branches = df["branch"] if "branch" in df.columns else [""] * len(df)
    return dict(zip(df["languageCode"], branches))


def load_holdout_ids(holdouts_file: str) -> list[str]:
    raw = yaml.safe_load((repo_root() / holdouts_file).read_text(encoding="utf-8"))
    ids = list(raw["holdouts"]) + list(raw.get("verse_holdouts") or {})
    return sorted(set(ids))


def build_family_selection(
    family: str | None = None,
    holdouts_file: str = "",
    metadata: pd.DataFrame | None = None,
    shareable_only: bool = True,
    allow_nc: bool = False,
    min_nt_books: int | None = MIN_NT_BOOKS,
    members_csv: str | None = None,
) -> pd.DataFrame:
    """Build one pool selection from a membership CSV plus forced targets.

    ``holdouts_file`` is required: the target editions are only forced into
    the pool via its ``holdouts:``/``verse_holdouts:`` keys, so building a
    selection without one produces a pool with no targets.
    """
    if not holdouts_file:
        raise ValueError("holdouts_file is required — it names the target editions")
    meta = load_metadata() if metadata is None else metadata
    from .data import resource_path

    members_path = (
        repo_root() / members_csv if members_csv
        else resource_path(f"configs/families/{family}.csv")
    )
    branch_of = _load_members(members_path)
    holdout_ids = load_holdout_ids(holdouts_file)

    # Targets are forced from the FULL metadata: membership lists need not
    # contain them (gradient steps exclude the target's family by design).
    forced_meta = meta[meta["translationId"].isin(holdout_ids)]
    missing = sorted(set(holdout_ids) - set(forced_meta["translationId"]))
    if missing:
        raise ValueError(f"Holdout translation(s) {missing} not in the corpus metadata")

    in_family = meta[meta["languageCode"].isin(branch_of)]
    candidates = pd.concat([
        in_family[~in_family["translationId"].isin(holdout_ids)], forced_meta
    ]).drop_duplicates("translationId")

    if shareable_only:
        # Restrict to licences that permit sharing a derived model, so the
        # resulting run is publishable. A
        # non-shareable target is an error, not a drop.
        lic = licence_of(candidates["translationId"], meta)
        bad = [
            t for t in holdout_ids
            if not is_shareable(lic.get(t, "Unknown"), allow_nc)
        ]
        if bad:
            raise ValueError(
                f"Holdout translation(s) {bad} have non-shareable licences"
            )
        keep = candidates["translationId"].map(
            lambda t: is_shareable(lic.get(t, "Unknown"), allow_nc)
        )
        candidates = candidates[keep.to_numpy()]

    config = SelectionConfig(
        target_size=None,           # take every member language
        include=holdout_ids,        # force the exact target editions in
        min_verses=MIN_VERSES,
        one_per_language=True,      # best-covered translation per language
        min_nt_books=min_nt_books,  # complete-NT pool rule
    )
    selection = select_translations(config, candidates, load_families())
    selection["branch"] = (
        selection["languageCode"].map(branch_of).fillna("")
    )
    return selection


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a pool selection")
    ap.add_argument("--family", default=None,
                    help="named membership list: configs/families/<family>.csv")
    ap.add_argument("--members", default=None,
                    help="explicit membership CSV path (gradient-step pools)")
    ap.add_argument("--holdouts", required=True,
                    help="holdouts YAML naming the target editions (required)")
    ap.add_argument("--include-non-shareable", action="store_true",
                    help="admit licences that do not permit sharing a derived model")
    ap.add_argument("--allow-nc", action="store_true")
    ap.add_argument("--min-nt-books", type=int, default=MIN_NT_BOOKS,
                    help="complete-NT pool rule; 0 disables")
    ap.add_argument("--name", default=None, help="output suffix (selection-<name>.csv)")
    args = ap.parse_args()
    if bool(args.family) == bool(args.members):
        ap.error("exactly one of --family / --members is required")

    selection = build_family_selection(
        family=args.family, holdouts_file=args.holdouts,
        shareable_only=not args.include_non_shareable, allow_nc=args.allow_nc,
        min_nt_books=args.min_nt_books or None, members_csv=args.members,
    )
    name = args.name or (args.family or Path(args.members).stem)
    out = write_selection(selection, name)
    # re-write including the branch column for readability
    cols = [
        "translationId", "languageCode", "languageNameInEnglish", "branch",
        "script", "OTverses", "NTverses", "DCverses", "totalVerses",
    ]
    selection[[c for c in cols if c in selection.columns]].to_csv(
        Path(out), index=False
    )
    n = selection["languageCode"].nunique()
    print(f"Wrote {out}: {len(selection)} translations, {n} languages")
    print(selection["branch"].value_counts().to_string())


if __name__ == "__main__":
    main()
