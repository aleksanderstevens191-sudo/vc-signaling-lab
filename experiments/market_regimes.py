#!/usr/bin/env python3
"""
Comparative statics in the macro state :class:`~game.environment.MarketCondition`.

For each regime (hot, neutral, cold) we run an independent Monte Carlo sample of
the same length and RNG seed. Shifting the regime moves the effective prior and
investment threshold via :data:`game.environment.DEFAULT_MARKET_REGIME_ADJUSTMENTS`;
the structural signal law and payoffs are unchanged.

Output is one stacked panel with a ``market_condition`` column, suitable for
stratified estimators or regression with regime fixed effects.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd  # noqa: E402

from game import (  # noqa: E402
    DEFAULT_ENV,
    MarketCondition,
    SimulationConfig,
    funding_evaluation_metrics,
    print_funding_evaluation_summary,
    run_simulation,
    save_results_csv,
)

from experiments.locations import RESULTS_DIR  # noqa: E402

N_ROUNDS = 5_000
RNG_SEED = 42
RESULTS_CSV = RESULTS_DIR / "market_regime_sweep.csv"


def main() -> None:
    panels: list[pd.DataFrame] = []
    for regime in MarketCondition:
        spec = SimulationConfig(
            n_rounds=N_ROUNDS,
            random_seed=RNG_SEED,
            market_condition=regime,
        )
        df_m = run_simulation(DEFAULT_ENV, spec)
        panels.append(df_m)
        print_funding_evaluation_summary(funding_evaluation_metrics(df_m))
        print(f"  stratum: market_condition={regime.value!r}  rounds={len(df_m)}\n")

    stacked = pd.concat(panels, ignore_index=True)
    save_results_csv(stacked, RESULTS_CSV)
    print(f"results: {RESULTS_CSV}  (n={len(stacked)} rows)")


if __name__ == "__main__":
    main()
