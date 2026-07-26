"""Multi-source pair building.

One-to-many pairs each target verse with the fixed source translation;
multi-source concatenates n renderings of the SAME verse into one source line:

    <2tgt> <1hin> hindi text <1mar> marathi text <1san> sanskrit text

keeping one example per (vref, target) so pair counts stay directly comparable
to the one-to-many baseline. The atomic language tags are the separators.

Unlike the bible-interlingua series, the forced-first source is not an
out-of-pool composite (Greek) but a pool member — a translation id from the
selection, chosen by alignment. It is forced first
whenever its cell at that vref is usable and it is not itself the target.

Sampling (training): n ~ Uniform{k_min..k} per example — k_min=1 is the
source-dropout that keeps single-rendering inputs in-distribution; the source
is forced first when present; the remaining renderings are sampled without
replacement and their order is shuffled.

Inference (valid/test/validation): deterministic — the source plus the top-(k-1)
candidates from a branch-aware ranking (same branch first, then the rest,
ordered by total verse coverage), skipping cells that are held out at that
vref.

Leakage safety (identical rule to manytomany._present_by_vref): source
renderings are drawn only from the non-held-out usable cells (the union of
the train and valid manifests) — the forced source included — so held-out
text is never fed in, as a source or a target.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .data import VREF_COLUMN
from .preprocess import SRC_COLUMN, TGT_COLUMN, normalise, source_tag, target_tag


def present_by_vref(*manifests: pd.DataFrame) -> dict[str, list[str]]:
    """Map each vref to the translations with usable (non-held-out) text there."""
    present: dict[str, list[str]] = defaultdict(list)
    for df in manifests:
        for v, t in zip(df[VREF_COLUMN], df["translation"]):
            present[v].append(t)
    return present


def inference_source_ranking(
    selection: pd.DataFrame, policy: str = "coverage"
) -> dict[str, list[str]]:
    """Deterministic per-target candidate ordering over the selection.

    ``policy`` is the experiment's ``data.companion_ranking``:

    - ``"coverage"``: every other pool member, ordered by descending total
      verse coverage (ties by translationId for stability).
    - ``"branch-first"``: same-branch members first, then the rest, each
      group coverage-ordered. Requires a populated ``branch`` column —
      silently degrading to coverage order would let the policy vary with
      the selection CSV's schema across runs being compared.

    The forced source is handled separately by the callers (always first when
    usable), so duplicates are skipped there.
    """
    frame = selection.copy()
    if "totalVerses" not in frame.columns:
        frame["totalVerses"] = 0
    frame["totalVerses"] = pd.to_numeric(frame["totalVerses"], errors="coerce").fillna(0)
    if policy == "branch-first":
        if "branch" not in frame.columns or frame["branch"].fillna("").eq("").all():
            raise ValueError(
                "companion_ranking 'branch-first' needs a populated branch "
                "column in the selection CSV"
            )
        frame["branch"] = frame["branch"].fillna("")
    elif policy == "coverage":
        frame["branch"] = ""          # one group: pure coverage order
    else:
        raise ValueError(f"Unknown companion ranking policy {policy!r}")
    ranking: dict[str, list[str]] = {}
    for _, row in frame.iterrows():
        target = row["translationId"]
        others = frame[frame["translationId"] != target]
        same = others[others["branch"] == row["branch"]]
        rest = others[others["branch"] != row["branch"]]
        order = lambda g: g.sort_values(
            ["totalVerses", "translationId"], ascending=[False, True]
        )["translationId"].tolist()
        ranking[target] = order(same) + order(rest)
    return ranking


def _render(src_id: str, vref: str, verses: pd.DataFrame,
            language_of: dict[str, str]) -> str:
    return f"{source_tag(language_of[src_id])} {normalise(verses.at[vref, src_id])}"


def build_ms_pairs(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    verses: pd.DataFrame,
    language_of: dict[str, str],
    k: int = 4,
    k_min: int = 1,
    seed: int = 13,
    source_id: str | None = None,
    forbidden: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Build one multi-source training pair per (vref, target) in ``train``.

    ``source_id`` is the alignment-chosen source translation, forced first
    whenever its cell at the vref is usable and the target is another
    translation. It obeys the same leakage rule as every candidate: only
    cells in the train/valid manifests are usable.

    ``forbidden`` is the set of held-out (vref, translation) cells; every pick
    is asserted against it, so the source-side leakage rule is enforced at
    pair-build time rather than only holding by construction.

    Returns columns vref, translation, src, tgt — the same shape one-to-many
    produces, so everything downstream (tokeniser, datasets, validation sets) is
    unchanged.
    """
    present = present_by_vref(train, valid)
    rng = np.random.default_rng(seed)
    forbidden = forbidden or set()
    rows = []
    for v, tgt in zip(train[VREF_COLUMN], train["translation"]):
        usable = present[v]
        force = source_id if source_id in usable and source_id != tgt else None
        candidates = [t for t in usable if t != tgt and t != force]
        if not candidates and not force:
            continue
        n = int(rng.integers(k_min, k + 1))
        picks: list[str] = []
        if force:
            picks.append(force)
        n_more = min(n - len(picks), len(candidates))
        if n_more > 0:
            idx = rng.choice(len(candidates), size=n_more, replace=False)
            sampled = [candidates[i] for i in idx]
            rng.shuffle(sampled)
            picks.extend(sampled)
        for p in picks:
            assert (v, p) not in forbidden, (
                f"held-out cell ({v}, {p}) reached the source side of a "
                f"training pair for target {tgt}"
            )
        src = " ".join(
            [target_tag(language_of[tgt])]
            + [_render(s, v, verses, language_of) for s in picks]
        )
        rows.append(
            {
                VREF_COLUMN: v,
                "translation": tgt,
                SRC_COLUMN: src,
                TGT_COLUMN: normalise(verses.at[v, tgt]),
            }
        )
    return pd.DataFrame(rows, columns=[VREF_COLUMN, "translation", SRC_COLUMN, TGT_COLUMN])


def to_ms_sources(
    frame: pd.DataFrame,
    verses: pd.DataFrame,
    language_of: dict[str, str],
    present: dict[str, list[str]],
    ranking: dict[str, list[str]],
    k: int = 4,
    source_id: str | None = None,
    forbidden: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Rewrite a (vref, translation, src, tgt) frame's sources to deterministic
    multi-source form: the forced source first, then the top-ranked present
    candidates.

    Used for valid/test/validation frames and holdout generation, so inference
    sources are reproducible. Held-out cells are absent from ``present`` and
    therefore never selected — the forced source included; ``forbidden``
    (the held-out cell set) turns that rule into an assertion.

    Rows for which no rendering exists anywhere in the pool are dropped and
    counted — a tag-only input has nothing to translate from — so the scored
    verse set is "every verse at least one pool member covers", independent
    of the forced source's coverage.
    """
    out = frame.copy()
    forbidden = forbidden or set()
    srcs = []
    n_picks = []
    for v, tgt in zip(out[VREF_COLUMN], out["translation"]):
        usable = set(present.get(v, ()))
        picks: list[str] = []
        if source_id in usable and source_id != tgt:
            picks.append(source_id)
        for cand in ranking.get(tgt, ()):
            if len(picks) >= k:
                break
            if cand != tgt and cand in usable and cand not in picks:
                picks.append(cand)
        for p in picks:
            assert (v, p) not in forbidden, (
                f"held-out cell ({v}, {p}) reached the source side of an "
                f"inference pair for target {tgt}"
            )
        n_picks.append(len(picks))
        srcs.append(
            " ".join(
                [target_tag(language_of[tgt])]
                + [_render(s, v, verses, language_of) for s in picks]
            )
        )
    out[SRC_COLUMN] = srcs
    covered = pd.Series(n_picks, index=out.index) > 0
    if (dropped := int((~covered).sum())):
        print(f"  to_ms_sources: dropped {dropped} verses with no available "
              f"rendering in the pool")
    return out[covered].reset_index(drop=True)


def strip_tags(src: str) -> str:
    """Remove all leading-tag tokens from a multi-source line (for analysis)."""
    return " ".join(t for t in src.split(" ") if not (t.startswith("<") and t.endswith(">")))
