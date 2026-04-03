"""
Pluggable rules for how the founder produces the observable signal.

The baseline is **exogenous noise** around type-conditional means (Gaussian).
Strategic signaling (e.g. costly effort chosen to maximize expected utility
given anticipated receiver behavior) can be implemented by supplying another
:class:`SignalStrategy` that reads :class:`SignalingRoundContext`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import FounderType, MarketCondition, SignalingEnvironment
from .founder import Founder, InvestmentDecision


@dataclass(frozen=True)
class SignalingRoundContext:
    """
    Public information available when the signal is chosen.

    Extend this as the model grows (e.g. reputation, past offers, rival VCs'
    known thresholds) so strategies stay a single call site.
    """

    environment: SignalingEnvironment
    founder_type: FounderType
    market_condition: MarketCondition


class GaussianNoiseSignalStrategy:
    """
    Default signal technology: sample ``s | θ`` from the environment (no
    optimization). Matches the original ``Founder.draw_signal`` behavior.
    """

    def choose_signal(
        self,
        founder: Founder,
        rng: np.random.Generator,
        *,
        context: SignalingRoundContext,
    ) -> float:
        del context  # unconditional on public context in the reduced form
        return founder.draw_signal(rng)


@dataclass(frozen=True)
class StrategicGridSignalStrategy:
    """
    Costly signaling: choose ``s`` to maximize expected payoff against fixed
    receivers (same objects as in :class:`~game.simulation.SignalingSimulationEngine`).

    Uses :meth:`Founder.best_signal_grid_search`; randomness from ``rng`` is
    unused (deterministic best response). In equilibrium terms this is the
    sender’s best reply to a fixed receiver policy; see :class:`Founder` for
    how that fits a full signaling equilibrium.
    """

    receivers: tuple[InvestmentDecision, ...]
    n_points: int = 201
    s_min: float | None = None
    s_max: float | None = None

    def choose_signal(
        self,
        founder: Founder,
        rng: np.random.Generator,
        *,
        context: SignalingRoundContext,
    ) -> float:
        del rng
        del context
        return founder.best_signal_grid_search(
            self.receivers,
            s_min=self.s_min,
            s_max=self.s_max,
            n_points=self.n_points,
        )
