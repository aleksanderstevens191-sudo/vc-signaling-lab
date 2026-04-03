#!/usr/bin/env python3
"""
CLI alias for the single-receiver baseline.

Prefer ``python -m experiments.baseline`` from the repository root; this file
exists so ``python experiments/run_baseline.py`` remains valid when the CWD is
the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.baseline import main  # noqa: E402

if __name__ == "__main__":
    main()
