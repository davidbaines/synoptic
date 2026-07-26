"""Side-by-side generated-vs-reference sample sheets.

Passages are user-configurable in ``configs/passages.yaml``; defaults mirror
the passages Liedes published on his blog.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .data import resource_path

PASSAGES_CONFIG = "configs/passages.yaml"


def load_passages(path: Path | None = None) -> list[dict]:
    config_path = resource_path(PASSAGES_CONFIG) if path is None else path
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))["passages"]


def passage_vrefs(passage: dict, vref_index: pd.Index) -> list[str]:
    """Resolve a passage's start/end to the vrefs between them, inclusive.

    Uses positional order in the canonical vref index, so multi-chapter
    passages work without verse arithmetic.
    """
    start = vref_index.get_loc(passage["start"])
    end = vref_index.get_loc(passage["end"])
    if end < start:
        raise ValueError(f"Passage {passage['name']!r} ends before it starts")
    return list(vref_index[start : end + 1])


def make_sheet(
    language_name: str,
    hypotheses: pd.Series,
    references: pd.Series,
    passages: list[dict] | None = None,
) -> str:
    """Render one language's sample sheet as markdown.

    ``hypotheses`` and ``references`` are vref-indexed. Passages outside the
    generated books are skipped silently (e.g. NT passages for an OT holdout).
    """
    passages = load_passages() if passages is None else passages
    lines = [f"# Sample sheet: {language_name}", ""]
    for passage in passages:
        try:
            vrefs = passage_vrefs(passage, hypotheses.index)
        except KeyError:
            continue
        if not any(v in references.index for v in vrefs):
            continue
        lines += [f"## {passage['name']}", "", "| vref | generated | reference |", "|---|---|---|"]
        for v in vrefs:
            hyp = str(hypotheses.get(v, "")).replace("|", "\\|")
            ref = str(references.get(v, "")).replace("|", "\\|")
            lines.append(f"| {v} | {hyp} | {ref} |")
        lines.append("")
    return "\n".join(lines)
