"""
Typing-only contracts for swapping signal rules and receiver logic.

Simulation code depends on these **protocols**, not concrete VC classes, so
you can plug in alternative receivers (different priors, robust rules) or test
doubles without changing the round loop.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .environment import SignalingEnvironment
from .founder import Founder
from .signaling_strategies import SignalingRoundContext


class SignalStrategy(Protocol):
    """How the founder maps hidden type + noise (and context) into a signal."""

    def choose_signal(
        self,
        founder: Founder,
        rng: np.random.Generator,
        *,
        context: SignalingRoundContext,
    ) -> float: ...


class BeliefReceiver(Protocol):
    """
    Minimal surface the simulation needs from each VC (or other investor).

    Implement this on custom classes, or use :class:`~game.vc.VC`, which
    satisfies the protocol.
    """

    environment: SignalingEnvironment

    def posterior_belief_high(self, signal: float) -> float: ...

    def decide_investment(self, signal: float) -> bool: ...

    def expected_return_if_invest(self, signal: float) -> float: ...
