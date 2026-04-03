#!/usr/bin/env python3
"""
Heterogeneous receiver panel: several Bayesian investors observe the same signal.

Uses the built-in panel constructor (spread priors and thresholds around the
environment defaults). Funding is summarised with the same binary metrics as the
single-receiver case; the exported table uses wide per-receiver columns
(``vc_*_posterior``, …).

The single-receiver figure helpers expect scalar ``posterior`` / ``invest``; this
driver therefore persists CSV only. Multi-receiver diagnostics belong in a
separate figure script once you choose an aggregation (e.g. principal receiver
or panel averages).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from game import (  # noqa: E402
    DEFAULT_ENV,
    SimulationConfig,
    funding_evaluation_metrics,
    print_funding_evaluation_summary,
    run_simulation,
    save_results_csv,
)

from experiments.locations import RESULTS_DIR  # noqa: E402

N_ROUNDS = 5_000
RNG_SEED = 42
N_RECEIVERS = 3
RESULTS_CSV = RESULTS_DIR / "multi_receiver_panel.csv"


def main() -> None:
    spec = SimulationConfig(
        n_rounds=N_ROUNDS,
        random_seed=RNG_SEED,
        n_vcs=N_RECEIVERS,
    )
    panel = run_simulation(DEFAULT_ENV, spec)
    save_results_csv(panel, RESULTS_CSV)
    print(f"results: {RESULTS_CSV}  (n={len(panel)}  receivers={N_RECEIVERS})")
    print_funding_evaluation_summary(funding_evaluation_metrics(panel))


if __name__ == "__main__":
    main()
