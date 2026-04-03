"""
Venture capitalist in a *Bayesian signaling* model.

**Structure (signaling / screening with incomplete information).** The founder’s
quality :math:`\\theta \\in \\{\\text{HIGH}, \\text{LOW}\\}` is hidden. The VC
does not observe :math:`\\theta`; they only see a real-valued *signal*
:math:`s` (e.g. pitch, metrics) drawn from a distribution that depends on
:math:`\\theta`. That is the classic signal :math:`s \\mid \\theta` in a
signaling game: beliefs are updated by Bayes’ rule from :math:`s`, not from
:math:`\\theta`.

This module implements the VC’s **Bayesian update** from prior over
:math:`\\theta` to posterior :math:`P(\\theta \\mid s)` using Gaussian
likelihoods, then a decision rule based on that posterior.
"""

from __future__ import annotations

from dataclasses import dataclass

from .environment import FounderType, MarketCondition, SignalingEnvironment


@dataclass
class VC:
    """
    Bayesian receiver in a signaling game: prior over hidden type, signal
    likelihoods, posterior via Bayes’ rule, then invest if posterior clears a
    cutoff.

    The game is *Bayesian* because the VC’s belief about :math:`\\theta` is
    represented by probabilities, updated from the common prior and the
    observed signal using Bayes’ rule (not a point estimate of :math:`\\theta`).
    """

    environment: SignalingEnvironment
    """Structural parameters: priors, signal means and noise, and payoffs."""

    investment_threshold: float | None = None
    """
    Invest iff the posterior probability of high quality is at least this value.
    ``None`` means use :meth:`SignalingEnvironment.effective_investment_threshold`
    (regime-aware). Otherwise must lie in [0, 1].
    """

    prior_high_override: float | None = None
    """
    If set, use this as P(HIGH) in Bayes' rule instead of ``environment.prior_high``.
    Likelihoods stay tied to the shared signal model in ``environment``.
    Must lie in (0, 1) when set.
    """

    def __post_init__(self) -> None:
        if self.investment_threshold is not None and not 0.0 <= self.investment_threshold <= 1.0:
            raise ValueError("investment_threshold must be between 0 and 1 inclusive.")
        if self.prior_high_override is not None:
            if not 0.0 < self.prior_high_override < 1.0:
                raise ValueError("prior_high_override must lie strictly between 0 and 1.")

    @property
    def market_condition(self) -> MarketCondition:
        """Macro regime for priors and cutoff; same as ``environment.market_regime``."""
        return self.environment.market_regime

    def resolved_investment_threshold(self) -> float:
        """Cutoff used for invest / no-invest (explicit or from environment)."""
        if self.investment_threshold is not None:
            return float(self.investment_threshold)
        return self.environment.effective_investment_threshold()

    @property
    def prior_belief_high(self) -> float:
        """
        Prior probability P(θ = HIGH) that a founder is high quality before
        any signal is observed. Uses ``prior_high_override`` when set, else
        ``environment.prior_high``.
        """
        if self.prior_high_override is not None:
            return float(self.prior_high_override)
        return self.environment.prior_high

    def _likelihood(self, signal: float, founder_type: FounderType) -> float:
        """
        Likelihood :math:`p(s \\mid \\theta)` — density of the observed signal
        given the founder’s type — as a Gaussian with type-dependent mean and
        common variance (see ``environment``).
        """
        return self.environment.likelihood_signal(signal, founder_type)

    def posterior_belief_high(self, signal: float) -> float:
        """
        **Bayesian update:** posterior :math:`P(\\theta=\\text{HIGH} \\mid s)`.

        **Setup.** Prior :math:`\\pi_H = P(\\text{HIGH})`, :math:`\\pi_L = 1-\\pi_H`.
        Likelihoods :math:`L_H(s) = p(s \\mid \\text{HIGH})`,
        :math:`L_L(s) = p(s \\mid \\text{LOW})` (normal densities here).

        **Bayes’ rule** (discrete :math:`\\theta`, continuous :math:`s`):

        .. math::

            P(\\text{HIGH} \\mid s)
            = \\frac{p(s \\mid \\text{HIGH})\\,\\pi_H}{p(s)}
            = \\frac{L_H(s)\\,\\pi_H}{L_H(s)\\,\\pi_H + L_L(s)\\,\\pi_L}.

        The denominator is the marginal :math:`p(s) = \\sum_\\theta p(s\\mid\\theta)\\pi_\\theta`
        (mixture over types). We compute it explicitly below so the update is
        visible as “(likelihood × prior) / marginal.”
        """
        env = self.environment

        # --- Ingredients of Bayes' rule (signaling: θ hidden, s observed) ---
        pi_h = self.prior_belief_high  # π_H = P(θ = HIGH) for this receiver
        pi_l = 1.0 - pi_h

        # Likelihoods p(s | θ): how probable is this signal under each type?
        likelihood_high = self._likelihood(signal, FounderType.HIGH)  # L_H(s)
        likelihood_low = self._likelihood(signal, FounderType.LOW)      # L_L(s)

        # Unnormalized joint masses: p(s, θ) = p(s | θ) P(θ)  (same π as prior)
        joint_signal_and_high = likelihood_high * pi_h  # p(s, HIGH)
        joint_signal_and_low = likelihood_low * pi_l    # p(s, LOW)

        # Marginal density of the signal: p(s) = Σ_θ p(s, θ)  (mixture over types)
        marginal_signal = joint_signal_and_high + joint_signal_and_low

        if marginal_signal <= 0.0:
            # Degenerate case; fall back to the prior (no update).
            return float(pi_h)

        # Posterior: P(HIGH | s) = p(s, HIGH) / p(s)  — explicit Bayes update
        posterior_high = joint_signal_and_high / marginal_signal
        return float(posterior_high)

    def decide_investment(self, signal: float) -> bool:
        """
        Return True if the VC invests after observing ``signal``.

        Investment occurs when the posterior belief that the founder is high
        quality is weakly above :meth:`resolved_investment_threshold`.
        """
        return self.posterior_belief_high(signal) >= self.resolved_investment_threshold()

    def expected_return_if_invest(self, signal: float) -> float:
        """
        Expected net return conditional on investing and having seen ``signal``,
        under the posterior induced by Bayes' rule.

        Useful for reporting and for rules based on expected value rather than
        a fixed posterior cutoff.
        """
        p_h = self.posterior_belief_high(signal)
        p_l = 1.0 - p_h
        env = self.environment
        return p_h * env.vc_return_high + p_l * env.vc_return_low

    # Backward-compatible names used elsewhere in the package
    posterior_high = posterior_belief_high
    should_invest = decide_investment


VentureCapitalist = VC
