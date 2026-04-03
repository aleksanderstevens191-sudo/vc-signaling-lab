"""
Multi-VC panel: several Bayesian receivers share the same founder signal.

Each VC uses the same likelihoods (shared ``SignalingEnvironment``) but may
differ in prior and investment threshold. Use :func:`build_vc_panel` to
construct 2–5 investors with a deterministic spread around the environment prior
and a baseline cutoff.
"""

from __future__ import annotations

from typing import Final

from .environment import SignalingEnvironment
from .vc import VC

MIN_PANEL_SIZE: Final[int] = 2
MAX_PANEL_SIZE: Final[int] = 5


def _clamp_open_unit(x: float, eps: float = 1e-6) -> float:
    return float(min(1.0 - eps, max(eps, x)))


def build_vc_panel(
    environment: SignalingEnvironment,
    n_vcs: int,
    *,
    base_threshold: float | None = None,
    threshold_half_spread: float = 0.07,
    prior_half_spread: float = 0.05,
) -> tuple[VC, ...]:
    """
    Build ``n_vcs`` VCs observing the same signal technology.

    Thresholds and priors are spaced linearly across
    ``[base - half_spread, base + half_spread]`` (clamped for valid priors).
    ``base_threshold`` defaults to :meth:`SignalingEnvironment.effective_investment_threshold`
    (regime-aware). ``n_vcs`` must be between :data:`MIN_PANEL_SIZE` and
    :data:`MAX_PANEL_SIZE`.
    """
    if not MIN_PANEL_SIZE <= n_vcs <= MAX_PANEL_SIZE:
        raise ValueError(
            f"n_vcs must be between {MIN_PANEL_SIZE} and {MAX_PANEL_SIZE}, got {n_vcs}."
        )
    if base_threshold is None:
        base_threshold = environment.effective_investment_threshold()
    if not 0.0 <= base_threshold <= 1.0:
        raise ValueError("base_threshold must lie in [0, 1].")
    if threshold_half_spread < 0 or prior_half_spread < 0:
        raise ValueError("half_spread parameters must be non-negative.")

    center_prior = float(environment.prior_high)
    priors = [
        _clamp_open_unit(center_prior + prior_half_spread * (2.0 * i / (n_vcs - 1) - 1.0))
        for i in range(n_vcs)
    ]
    thresholds = [
        float(min(1.0, max(0.0, base_threshold + threshold_half_spread * (2.0 * i / (n_vcs - 1) - 1.0))))
        for i in range(n_vcs)
    ]
    return tuple(
        VC(
            environment=environment,
            investment_threshold=float(thresholds[i]),
            prior_high_override=priors[i],
        )
        for i in range(n_vcs)
    )
