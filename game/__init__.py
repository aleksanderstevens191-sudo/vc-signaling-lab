"""
Core model: one-sided incomplete information, Gaussian signals, Bayesian receivers,
and Monte Carlo batch machinery for reduced-form experiments.
"""

from .environment import (
    DEFAULT_ENV,
    DEFAULT_MARKET_REGIME_ADJUSTMENTS,
    FounderType,
    MarketCondition,
    MarketRegimeAdjustments,
    SignalingEnvironment,
    SignalingGameConfig,
)
from .founder import Founder, InvestmentDecision, sample_founder_type
from .protocols import BeliefReceiver, SignalStrategy
from .signaling_strategies import (
    GaussianNoiseSignalStrategy,
    SignalingRoundContext,
    StrategicGridSignalStrategy,
)
from .multi_vc import MAX_PANEL_SIZE, MIN_PANEL_SIZE, build_vc_panel
from .plotting import (
    plot_funding_confusion_counts,
    plot_investment_rate_by_type,
    plot_posterior_vs_signal,
    plot_signal_distribution_by_type,
    plot_signal_vs_posterior,
)
from .simulation import (
    FundingEvaluationMetrics,
    ReceiverOutcome,
    SignalingSimulationEngine,
    SimulationConfig,
    competitive_funding,
    flatten_receiver_columns,
    founder_funded,
    funded_series,
    funding_evaluation_metrics,
    n_interested_vcs,
    print_funding_evaluation_summary,
    receiver_outcomes,
    run_n_rounds,
    run_simulation,
    save_results_csv,
)
from .vc import VC, VentureCapitalist

__all__ = [
    "MAX_PANEL_SIZE",
    "MIN_PANEL_SIZE",
    "BeliefReceiver",
    "DEFAULT_ENV",
    "DEFAULT_MARKET_REGIME_ADJUSTMENTS",
    "Founder",
    "FounderType",
    "GaussianNoiseSignalStrategy",
    "InvestmentDecision",
    "StrategicGridSignalStrategy",
    "MarketCondition",
    "MarketRegimeAdjustments",
    "ReceiverOutcome",
    "SignalingEnvironment",
    "SignalingGameConfig",
    "SignalingRoundContext",
    "SignalingSimulationEngine",
    "SignalStrategy",
    "SimulationConfig",
    "build_vc_panel",
    "competitive_funding",
    "n_interested_vcs",
    "VC",
    "VentureCapitalist",
    "FundingEvaluationMetrics",
    "flatten_receiver_columns",
    "founder_funded",
    "funded_series",
    "funding_evaluation_metrics",
    "plot_funding_confusion_counts",
    "plot_investment_rate_by_type",
    "plot_posterior_vs_signal",
    "plot_signal_distribution_by_type",
    "plot_signal_vs_posterior",
    "print_funding_evaluation_summary",
    "receiver_outcomes",
    "run_n_rounds",
    "run_simulation",
    "sample_founder_type",
    "save_results_csv",
]
