"""Shared toolkit for the closed-text Bible MT series.

Multi-source fusion reads several renderings of the same verse side by side —
seeing them together, synoptically. Descended from code inspired by Sami
Liedes' Bible-MT experiments. See README.md.
"""

from importlib import metadata

try:
    __version__ = metadata.version("synoptic")
except metadata.PackageNotFoundError:  # running from a raw checkout
    __version__ = "0.0.0"

# Where remote agents install this package from. The requirements capture at
# enqueue time records synoptic as a bare name==version pair, which pip would
# resolve from PyPI (an unrelated project); train._maybe_clearml pins this
# instead.
GIT_REQUIREMENT = (
    f"synoptic @ git+https://github.com/davidbaines/synoptic@v{__version__}"
)
