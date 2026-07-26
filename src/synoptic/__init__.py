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

# Remote agents install this package by git tag: train._maybe_clearml forces
# "synoptic @ git+https://github.com/davidbaines/synoptic@v<__version__>"
# into the task requirements (keyed by package name so it replaces the
# captured PyPI-style pin).
