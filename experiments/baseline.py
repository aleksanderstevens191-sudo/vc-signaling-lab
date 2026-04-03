#!/usr/bin/env python3
"""
Single-receiver Monte Carlo benchmark.

Draws i.i.d. rounds under :data:`game.environment.DEFAULT_ENV` with one
threshold investor. Writes a round-level panel, four figures under ``plots/``,
and console summaries (funding vs. latent type, investment rates by type).

Replication knobs (seed and length) are fixed constants below; override in
code for sensitivity analysis.
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
    plot_funding_confusion_counts,
    plot_investment_rate_by_type,
    plot_signal_distribution_by_type,
    plot_signal_vs_posterior,
    print_funding_evaluation_summary,
    run_simulation,
    save_results_csv,
)

from experiments.locations import FIGURES_DIR, RESULTS_DIR  # noqa: E402

# --- replication specification -------------------------------------------------
N_ROUNDS = 5_000
RNG_SEED = 42

RESULTS_CSV = RESULTS_DIR / "single_receiver_baseline.csv"
FIG_SIGNAL_VS_POSTERIOR = FIGURES_DIR / "baseline_signal_vs_posterior.png"
FIG_SIGNAL_DISTRIBUTION = FIGURES_DIR / "baseline_signal_distribution_by_type.png"
FIG_FUNDING_CONFUSION = FIGURES_DIR / "baseline_funding_confusion_counts.png"
FIG_INVESTMENT_RATE = FIGURES_DIR / "baseline_investment_rate_by_type.png"


def main() -> None:
    spec = SimulationConfig(n_rounds=N_ROUNDS, random_seed=RNG_SEED)
    panel = run_simulation(DEFAULT_ENV, spec)
    save_results_csv(panel, RESULTS_CSV)
    metrics = funding_evaluation_metrics(panel)

    plot_signal_vs_posterior(panel, FIG_SIGNAL_VS_POSTERIOR)
    plot_signal_distribution_by_type(panel, FIG_SIGNAL_DISTRIBUTION)
    plot_funding_confusion_counts(metrics, FIG_FUNDING_CONFUSION)
    plot_investment_rate_by_type(panel, FIG_INVESTMENT_RATE)

    p_invest = panel["invest"]
    p_post = panel["posterior"]
    overall_invest_rate = float(p_invest.mean())

    by_theta = (
        panel.groupby("founder_type", sort=True)
        .agg(
            n=("round", "count"),
            mean_invest=(p_invest.name, "mean"),
            mean_posterior=(p_post.name, "mean"),
        )
        .reset_index()
    )

    print(f"results: {RESULTS_CSV}  (n={len(panel)})")
    print(
        "figures:\n"
        f"  {FIG_SIGNAL_VS_POSTERIOR}\n"
        f"  {FIG_SIGNAL_DISTRIBUTION}\n"
        f"  {FIG_FUNDING_CONFUSION}\n"
        f"  {FIG_INVESTMENT_RATE}"
    )
    print_funding_evaluation_summary(metrics)

    print(f"\nUnconditional investment rate: {overall_invest_rate:.4f}")
    print("By latent type:")
    for _, row in by_theta.iterrows():
        theta = row["founder_type"]
        print(
            f"  {theta}:  E[invest|θ]={row['mean_invest']:.4f},  "
            f"E[π|θ]={row['mean_posterior']:.4f}  (n={int(row['n'])})"
        )


if __name__ == "__main__":
    main()
