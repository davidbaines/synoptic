"""Lightweight alignability score between languages, over parallel verses.

Intended tool was eflomal/fast_align, but neither builds in this environment
(no Python dev headers, no cmake). This is a compact IBM Model 1 (the model
both of those are built on): EM-train lexical translation probabilities on the
shared verses and report the corpus average log-likelihood per target token.
Higher (less negative) means the two languages' words co-occur more
predictably, i.e. they are more alignable. Symmetrised over both directions.

It is coarser than fast_align (no positional/fertility model), but adequate as
a relatedness/alignability factor for source selection.
"""

from __future__ import annotations

import math
from collections import defaultdict

from .preprocess import normalise


def _tok(s: str) -> list[str]:
    return normalise(s).split()


def _ibm1_dir(src_sents, tgt_sents, iters=4) -> float:
    """Train IBM-1 src->tgt; return mean log p(tgt token | src) over the corpus."""
    NULL = "<null>"
    t = defaultdict(lambda: defaultdict(float))
    # uniform init via observed co-occurrences
    for s, tt in zip(src_sents, tgt_sents):
        for f in tt:
            for e in [NULL] + s:
                t[e][f] = 1.0
    for e in t:
        n = len(t[e])
        for f in t[e]:
            t[e][f] = 1.0 / n
    for _ in range(iters):
        count = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)
        for s, tt in zip(src_sents, tgt_sents):
            srce = [NULL] + s
            for f in tt:
                z = sum(t[e][f] for e in srce) or 1e-12
                for e in srce:
                    c = t[e][f] / z
                    count[e][f] += c
                    total[e] += c
        for e in count:
            for f in count[e]:
                t[e][f] = count[e][f] / (total[e] or 1e-12)
    # score: mean log p(f|best e) proxy = mean log(sum_e t[e][f]/|s|)
    ll, n = 0.0, 0
    for s, tt in zip(src_sents, tgt_sents):
        srce = [NULL] + s
        for f in tt:
            p = sum(t[e].get(f, 0.0) for e in srce) / len(srce)
            ll += math.log(p or 1e-12)
            n += 1
    return ll / max(n, 1)


def alignability(a_sents, b_sents, iters=4) -> float:
    """Symmetric IBM-1 alignability score (mean of both directions)."""
    return round((_ibm1_dir(a_sents, b_sents, iters) + _ibm1_dir(b_sents, a_sents, iters)) / 2, 3)


def parallel_tokens(verses, tid_a, tid_b, books=None, limit=3000):
    """Whitespace-tokenised parallel sentences where both have real text
    (empty cells and ``<range>`` markers are skipped)."""
    from .canon import NT_BOOKS
    from .data import RANGE_MARKER
    keep = set(NT_BOOKS) if books is None else set(books)
    a, b = [], []
    for v in verses.index:
        if v.split(" ", 1)[0] not in keep:
            continue
        ta, tb = verses.at[v, tid_a], verses.at[v, tid_b]
        if ta and tb and ta != RANGE_MARKER and tb != RANGE_MARKER:
            a.append(_tok(ta)); b.append(_tok(tb))
            if len(a) >= limit:
                break
    return a, b
