"""Founder: hidden type, Gaussian signal, funding minus signaling cost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .environment import FounderType, SignalingEnvironment


class InvestmentDecision(Protocol):
    """
    Minimal receiver surface for strategic best response.

    The founder only needs the VC's mapping from observed signal to invest /
    pass to evaluate ``P(invest) * benefit - cost`` under a fixed rule.
    """

    def decide_investment(self, signal: float) -> bool: ...


@dataclass
class Founder:
    """
    The founder sees quality θ; outsiders only see a real-valued signal s.

    **Exogenous signaling:** :meth:`draw_signal` samples ``s | θ`` as Gaussian
    noise around the type mean (the default simulation).

    **Strategic signaling:** the founder can pick ``s`` to maximize expected
    payoff against a fixed VC rule. With a deterministic cutoff on the
    posterior, ``P(invest)`` is 0 or 1; marginal signaling cost is lower for
    high types than low types via ``environment.cost_coeff_*`` (see
    :meth:`~game.environment.SignalingEnvironment.founder_signaling_cost`).

    **Relation to signaling equilibria.** In a (perfect Bayesian) signaling
    equilibrium, the VC’s strategy (here: update beliefs from the likelihood
    and invest if posterior ≥ threshold) must be optimal given beliefs, and
    each founder type’s signal must be a best response to the VC’s rule,
    with beliefs consistent with Bayes’ rule on the equilibrium path.
    :meth:`best_signal_grid_search` computes that *best response* for a fixed
    receiver rule; it is not by itself a full equilibrium solver (e.g. it does
    not adjust the VC’s threshold or check off-path beliefs). Pooling or
    partial pooling can arise if even the best s does not separate types at
    the VC’s cutoff, given costs and priors.
    """

    founder_type: FounderType
    environment: SignalingEnvironment

    def draw_signal(self, rng: np.random.Generator) -> float:
        env = self.environment
        mu = env.signal_mean(self.founder_type)
        return float(rng.normal(mu, env.signal_std))

    def utility(self, signal: float, invested: bool) -> float:
        env = self.environment
        b = env.founder_benefit_invested if invested else 0.0
        c = env.founder_signaling_cost(signal, self.founder_type)
        return b - c

    def expected_payoff(
        self,
        signal: float,
        receivers: Sequence[InvestmentDecision],
    ) -> float:
        """
        Expected payoff under fixed receiver decision rules:

        ``P(invest) * founder_benefit_invested - signaling_cost(s | θ)``,

        where ``P(invest)`` is 0 or 1 according to whether *any* receiver
        invests (same funding rule as the simulation). Higher signals usually
        raise the posterior ``P(HIGH|s)`` and thus the chance of clearing the
        VC's threshold, but the mapping need not be monotone globally; a grid
        search does not assume monotonicity.
        """
        funded = any(r.decide_investment(signal) for r in receivers)
        return self.utility(signal, funded)

    def best_signal_grid_search(
        self,
        receivers: Sequence[InvestmentDecision],
        *,
        s_min: float | None = None,
        s_max: float | None = None,
        n_points: int = 201,
    ) -> float:
        """
        Approximately maximize :meth:`expected_payoff` over ``s`` on a uniform grid.

        Bounds default to type means ± ``4 * signal_std`` so the grid covers
        signals that are plausible under the likelihood while staying simple.

        This is the sender’s optimization step in a signaling game: given the
        VC’s fixed mapping from s to invest/pass, each θ chooses s* that
        maximizes benefit × I{invest(s)} − c_θ(s). Together with Bayes updating,
        that characterizes equilibrium play when strategies are pure on the
        grid’s resolution.
        """
        if not receivers:
            raise ValueError("At least one receiver is required.")
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")

        env = self.environment
        lo = min(env.signal_mean_high, env.signal_mean_low)
        hi = max(env.signal_mean_high, env.signal_mean_low)
        pad = 4.0 * env.signal_std
        lo_b = lo - pad if s_min is None else s_min
        hi_b = hi + pad if s_max is None else s_max
        if lo_b >= hi_b:
            raise ValueError("Grid bounds must satisfy s_min < s_max.")

        grid = np.linspace(lo_b, hi_b, n_points)
        best_s = float(grid[0])
        best_u = self.expected_payoff(best_s, receivers)
        for s in grid[1:]:
            s = float(s)
            u = self.expected_payoff(s, receivers)
            if u > best_u:
                best_u = u
                best_s = s
        return best_s


def sample_founder_type(env: SignalingEnvironment, rng: np.random.Generator) -> FounderType:
    """Draw θ from the prior: P(HIGH) = env.prior_high."""
    return FounderType.HIGH if rng.random() < env.prior_high else FounderType.LOW
