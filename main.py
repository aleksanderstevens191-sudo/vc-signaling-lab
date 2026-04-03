#!/usr/bin/env python3
"""Repository entry point: runs the single-receiver baseline driver."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.baseline import main  # noqa: E402

if __name__ == "__main__":
    main()
