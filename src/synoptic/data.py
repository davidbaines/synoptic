"""Access to the DavidCBaines/ebible_corpus Hugging Face dataset.

`main.parquet` is a wide table: one row per vref, one column per translation.
It is 1.28 GB, so by default only the requested columns are read over the
network (HTTP range requests via the ``hf://`` filesystem) and cached locally
as small parquet subsets under ``data/cache/``. If the full file has been
downloaded to ``data/main.parquet``, it is used instead.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import pandas as pd

EBIBLE_REPO = "DavidCBaines/ebible_corpus"
MAIN_PARQUET_HF = f"hf://datasets/{EBIBLE_REPO}/main.parquet"
RANGE_MARKER = "<range>"
VREF_COLUMN = "vref"


def repo_root() -> Path:
    """The experiment repo the caller runs in.

    synoptic is installed as a dependency; everything it reads or writes at
    repo level (configs/, experiments/, data/, checkpoints/) is relative to
    $SYNOPTIC_ROOT, else the current working directory. Advisory data files
    shipped with the package resolve via :func:`resource_path` instead.
    """
    return Path(os.environ.get("SYNOPTIC_ROOT", ".")).resolve()


def resource_path(relative: str) -> Path:
    """A packaged advisory data file, unless the repo overrides it.

    ``relative`` is the repo-conventional path (e.g.
    ``configs/language_families.csv``). If the experiment repo carries that
    file, it wins; otherwise the copy shipped under ``synoptic/resources/``
    (same path minus the ``configs/`` prefix) is used.
    """
    override = repo_root() / relative
    if override.exists():
        return override
    trimmed = relative.removeprefix("configs/")
    return Path(__file__).resolve().parent / "resources" / trimmed


def data_dir() -> Path:
    d = Path(os.environ.get("SYNOPTIC_DATA_DIR", repo_root() / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


@lru_cache(maxsize=1)
def load_metadata() -> pd.DataFrame:
    """Per-translation metadata with numeric coverage columns and totalVerses."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(EBIBLE_REPO, "metadata.parquet", repo_type="dataset")
    meta = pd.read_parquet(path)
    for col in ("OTverses", "NTverses", "DCverses"):
        meta[col] = pd.to_numeric(meta[col], errors="coerce").fillna(0).astype(int)
    meta["totalVerses"] = meta[["OTverses", "NTverses", "DCverses"]].sum(axis=1)
    return meta


@lru_cache(maxsize=1)
def translation_columns() -> tuple[str, ...]:
    """Column names of main.parquet (excluding vref), read from the schema only.

    The names are cached to ``data/cache/columns.txt`` on first fetch, so a box
    whose verse data is already cached can resolve translation ids fully
    offline — the dev 3090 has intermittent DNS, and column names never change
    for a fixed dataset revision. Delete that file to force a refetch.
    """
    import pyarrow.parquet as pq

    local = data_dir() / "main.parquet"
    columns_cache = data_dir() / "cache" / "columns.txt"
    if local.exists():
        names = pq.ParquetFile(local).schema_arrow.names
    elif columns_cache.exists():
        names = columns_cache.read_text(encoding="utf-8").splitlines()
    else:
        try:
            from huggingface_hub import HfFileSystem

            fs = HfFileSystem()
            with fs.open(f"datasets/{EBIBLE_REPO}/main.parquet", "rb") as f:
                names = pq.ParquetFile(f).schema_arrow.names
            columns_cache.parent.mkdir(parents=True, exist_ok=True)
            columns_cache.write_text("\n".join(names) + "\n", encoding="utf-8")
        except Exception:
            # Offline fallback: the union of column names across already-cached
            # verse subsets. Correct for exact-id resolution (the usual case);
            # a genuinely absent id still raises a clear KeyError in
            # resolve_column. Deliberately NOT persisted as the authoritative
            # columns.txt, since it is only a partial view.
            names = _cached_column_union()
            if not names:
                raise
    return tuple(n for n in names if n != VREF_COLUMN)


def _cached_column_union() -> list[str]:
    """Union of column names across cached ``data/cache/verses-*.parquet``."""
    import pyarrow.parquet as pq

    cache_dir = data_dir() / "cache"
    names: set[str] = set()
    for path in cache_dir.glob("verses-*.parquet"):
        try:
            names.update(pq.ParquetFile(path).schema_arrow.names)
        except Exception:
            continue
    return sorted(names)


def resolve_column(translation_id: str) -> str:
    """Map a metadata translationId to its column name in main.parquet.

    Tries an exact match first, then a unique ``<languageCode>-<id>`` style
    suffix match.
    """
    cols = translation_columns()
    if translation_id in cols:
        return translation_id
    matches = [c for c in cols if c.endswith(f"-{translation_id}")]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        f"No unique main.parquet column for translation {translation_id!r}"
        f" (matches: {matches})"
    )


def load_verses(translation_ids: Sequence[str]) -> pd.DataFrame:
    """Load the requested translations as a vref-indexed DataFrame.

    Columns are named by the requested translationIds. Missing cells are
    empty strings; ``<range>`` markers are preserved for callers to filter.
    """
    resolved = {resolve_column(t): t for t in translation_ids}
    cols = sorted(resolved)

    local_full = data_dir() / "main.parquet"
    cache_dir = data_dir() / "cache"
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.sha1("|".join(cols).encode()).hexdigest()[:16]
    cached = cache_dir / f"verses-{key}.parquet"

    if cached.exists():
        df = pd.read_parquet(cached)
    elif local_full.exists():
        df = pd.read_parquet(local_full, columns=[VREF_COLUMN, *cols])
    else:
        df = pd.read_parquet(MAIN_PARQUET_HF, columns=[VREF_COLUMN, *cols])
        df.to_parquet(cached)

    df = df.rename(columns=resolved).set_index(VREF_COLUMN)
    return df[list(translation_ids)].fillna("")
