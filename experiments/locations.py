"""
Filesystem layout for simulation outputs.

All drivers write under the repository root: round-level panels in ``data/``,
publication-style figures in ``plots/``. Paths are absolute; callers should not
assume a particular working directory beyond what each driver adds to
``sys.path``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
RESULTS_DIR: Path = REPO_ROOT / "data"
FIGURES_DIR: Path = REPO_ROOT / "plots"
