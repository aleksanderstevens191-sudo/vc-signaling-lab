"""
Shared parameters for the VC–founder signaling game.

The model is a standard Bayesian game with one-sided incomplete information:
the founder observes quality; the VC only sees a noisy signal (e.g. pitch
effort, traction metrics packaged as a scalar).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Final

import numpy as np


class MarketCondition(str, Enum):
    """Macro regime: shifts optimism (prior) and funding bar (threshold) slightly."""

    HOT = "hot"
    NEUTRAL = "neutral"
    COLD = "cold"


@dataclass(frozen=True)
class MarketRegimeAdjustments:
    """
    Additive shifts to the nominal prior P(HIGH) and to the VC posterior cutoff
    by regime. Neutral deltas are typically zero; hot raises prior and/or
    lowers the bar; cold does the opposite.
    """

    prior_delta: Mapping[MarketCondition, float]
    threshold_delta: Mapping[MarketCondition, float]

    def __post_init__(self) -> None:
        for m in MarketCondition:
            if m not in self.prior_delta:
                raise ValueError(f"prior_delta must include {m!r}.")
            if m not in self.threshold_delta:
                raise ValueError(f"threshold_delta must include {m!r}.")


DEFAULT_MARKET_REGIME_ADJUSTMENTS: Final[MarketRegimeAdjustments] = MarketRegimeAdjustments(
    prior_delta=MappingProxyType(
        {
            MarketCondition.HOT: 0.05,
            MarketCondition.NEUTRAL: 0.0,
            MarketCondition.COLD: -0.05,
        }
    ),
    threshold_delta=MappingProxyType(
        {
            MarketCondition.HOT: -0.03,
            MarketCondition.NEUTRAL: 0.0,
            MarketCondition.COLD: 0.03,
        }
    ),
)


def _clamp_unit_open(x: float, eps: float = 1e-6) -> float:
    """Keep probability in (0, 1) for priors."""
    return float(min(1.0 - eps, max(eps, x)))


def _clamp_unit_closed(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


class FounderType(str, Enum):
    """Latent startup quality; only the founder observes this."""

    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True)
class SignalingEnvironment:
    """
    Immutable game specification: priors, signal technology, and payoffs.

    Signals are drawn from a type-conditional Gaussian. The VC knows the
    structural parameters (means, variance, priors) and applies Bayes' rule.

    **Market regime.** ``prior_high_nominal`` is the baseline P(HIGH) before
    applying ``market_regime``; the operative common prior is :attr:`prior_high`.
    Similarly, ``investment_threshold_base`` is the baseline cutoff before
    regime; use :meth:`effective_investment_threshold` for the VC bar.
    """

    # Baseline P(HIGH) before ``market_regime``; operative prior is ``prior_high``.
    prior_high_nominal: float

    # Type-conditional signal means (Spence-style: high types may pool or
    # separate depending on parameters and VC rule).
    signal_mean_high: float
    signal_mean_low: float

    # Common observation noise around the type-specific mean.
    signal_std: float

    # VC net return conditional on investing and the true type.
    vc_return_high: float
    vc_return_low: float

    # Founder utility: benefit from receiving investment minus quadratic
    # signaling cost c_θ(s) = (1/2) k_θ s², so marginal cost ∂c/∂s = k_θ s.
    #
    # **Imitation cost (Spence / single-crossing).** A separating signaling
    # equilibrium needs high types willing to send signals low types are not
    # willing to mimic. That requires low-quality founders to face a *higher*
    # marginal cost of raising s (for s > 0): here k_low > k_high, so the same
    # increment in signal is more expensive for the low type. Intuitively,
    # “looking like” a high type (strong traction, deep diligence, credible
    # hires) is cheaper for firms that actually have quality; bad types pay
    # more in effort, fakery, or burning resources to inflate the scalar s.
    founder_benefit_invested: float
    cost_coeff_high: float
    cost_coeff_low: float

    # Competitive round: each interested VC adds this to the reported founder valuation (0 if none invest).
    valuation_per_interested_vc: float = 0.35

    # Macro label; together with ``market_adjustments`` sets effective prior and VC cutoff.
    market_regime: MarketCondition = MarketCondition.NEUTRAL

    # Optional overrides for regime deltas; ``None`` uses ``DEFAULT_MARKET_REGIME_ADJUSTMENTS``.
    market_adjustments: MarketRegimeAdjustments | None = None

    # Baseline posterior cutoff (before regime) for VCs that track the environment.
    investment_threshold_base: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.prior_high_nominal < 1.0:
            raise ValueError("prior_high_nominal must lie strictly between 0 and 1.")
        if self.signal_std <= 0:
            raise ValueError("signal_std must be positive.")
        if self.valuation_per_interested_vc < 0:
            raise ValueError("valuation_per_interested_vc must be non-negative.")
        if not 0.0 <= self.investment_threshold_base <= 1.0:
            raise ValueError("investment_threshold_base must lie in [0, 1].")

    def _regime_map(self) -> MarketRegimeAdjustments:
        return self.market_adjustments or DEFAULT_MARKET_REGIME_ADJUSTMENTS

    @property
    def prior_high(self) -> float:
        """Effective common prior P(HIGH) after applying ``market_regime``."""
        delta = self._regime_map().prior_delta[self.market_regime]
        return _clamp_unit_open(self.prior_high_nominal + delta)

    @property
    def prior_low(self) -> float:
        return 1.0 - self.prior_high

    def effective_investment_threshold(self) -> float:
        """Posterior cutoff implied by ``investment_threshold_base`` and current regime."""
        delta = self._regime_map().threshold_delta[self.market_regime]
        return _clamp_unit_closed(self.investment_threshold_base + delta)

    def signal_mean(self, founder_type: FounderType) -> float:
        """Deterministic component of the signal for the given type."""
        if founder_type is FounderType.HIGH:
            return self.signal_mean_high
        return self.signal_mean_low

    def likelihood_signal(self, signal: float, founder_type: FounderType) -> float:
        """
        Gaussian likelihood p(signal | type) with mean signal_mean(type)
        and variance signal_std**2.
        """
        mu = self.signal_mean(founder_type)
        z = (signal - mu) / self.signal_std
        norm = 1.0 / (self.signal_std * np.sqrt(2.0 * np.pi))
        return float(norm * np.exp(-0.5 * z * z))

    def founder_signaling_cost(self, signal: float, founder_type: FounderType) -> float:
        """Quadratic cost (1/2) * k_theta * signal**2; k_low > k_high makes mimicry expensive for low types."""
        k = self.cost_coeff_high if founder_type is FounderType.HIGH else self.cost_coeff_low
        return 0.5 * k * signal * signal

    def founder_valuation(self, n_interested_vcs: int) -> float:
        """
        Simple competitive valuation: zero if no one invests; otherwise linear
        in the number of interested VCs (``n * valuation_per_interested_vc``).
        """
        if n_interested_vcs <= 0:
            return 0.0
        return float(n_interested_vcs) * self.valuation_per_interested_vc


# Sensible defaults for demos and tests (separating-ish means, costly mimicry for low type).
DEFAULT_ENV: Final[SignalingEnvironment] = SignalingEnvironment(
    prior_high_nominal=0.35,
    signal_mean_high=2.0,
    signal_mean_low=0.8,
    signal_std=0.45,
    vc_return_high=1.0,
    vc_return_low=-0.9,
    founder_benefit_invested=1.2,
    cost_coeff_high=0.15,
    cost_coeff_low=0.55,
    market_regime=MarketCondition.NEUTRAL,
    investment_threshold_base=0.5,
)


@dataclass(frozen=True)
class SignalingGameConfig:
    """
    High-level simulation settings: belief parameters, signal technology, VC
    cutoff, and optional market regime. Payoffs default from ``template`` when
    building a full ``SignalingEnvironment``.
    """

    prior_high: float
    signal_mean_high: float
    signal_mean_low: float
    signal_std: float
    investment_threshold: float = 0.5
    market: MarketCondition = MarketCondition.NEUTRAL
    market_adjustments: MarketRegimeAdjustments | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.prior_high < 1.0:
            raise ValueError("prior_high must lie strictly between 0 and 1.")
        if self.signal_std <= 0:
            raise ValueError("signal_std must be positive.")
        if not 0.0 <= self.investment_threshold <= 1.0:
            raise ValueError("investment_threshold must be between 0 and 1 inclusive.")

    def _adjustments(self) -> MarketRegimeAdjustments:
        return self.market_adjustments or DEFAULT_MARKET_REGIME_ADJUSTMENTS

    def effective_prior_high(self) -> float:
        """Prior after applying the current ``market`` shift."""
        delta = self._adjustments().prior_delta[self.market]
        return _clamp_unit_open(self.prior_high + delta)

    def effective_investment_threshold(self) -> float:
        """VC posterior cutoff after applying the current ``market`` shift."""
        delta = self._adjustments().threshold_delta[self.market]
        return _clamp_unit_closed(self.investment_threshold + delta)

    def to_signaling_environment(
        self,
        template: SignalingEnvironment | None = None,
    ) -> SignalingEnvironment:
        """
        Merge core parameters onto ``template``, storing nominal prior, regime,
        and baseline cutoff on the environment (effective values via properties
        and :meth:`SignalingEnvironment.effective_investment_threshold`).
        Defaults to :data:`DEFAULT_ENV` for payoff and cost fields.
        """
        base = template if template is not None else DEFAULT_ENV
        adjustments = (
            self.market_adjustments
            if self.market_adjustments is not None
            else base.market_adjustments
        )
        return replace(
            base,
            prior_high_nominal=self.prior_high,
            market_regime=self.market,
            market_adjustments=adjustments,
            investment_threshold_base=self.investment_threshold,
            signal_mean_high=self.signal_mean_high,
            signal_mean_low=self.signal_mean_low,
            signal_std=self.signal_std,
        )
