"""
Monte Carlo batch driver for the signaling environment.

Each round: draw type, realize signal (pluggable
:class:`~game.protocols.SignalStrategy`), update one or more
:class:`~game.protocols.BeliefReceiver` instances, record payoffs. The loop is
decomposed so alternative signal rules or receiver panels plug in without
changing the outer batch API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .environment import FounderType, MarketCondition, SignalingEnvironment
from .founder import Founder, sample_founder_type
from .multi_vc import build_vc_panel
from .protocols import BeliefReceiver, SignalStrategy
from .signaling_strategies import GaussianNoiseSignalStrategy, SignalingRoundContext
from .vc import VC, VentureCapitalist


def _receiver_with_environment(recv: BeliefReceiver, env: SignalingEnvironment) -> BeliefReceiver:
    """Point a dataclass receiver at ``env``; leave other implementations unchanged."""
    if is_dataclass(recv):
        return replace(recv, environment=env)
    return recv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """
    High-level knobs for batch runs: number of rounds, RNG seed, and
    :class:`~game.environment.MarketCondition` for each round.

    :func:`run_simulation` applies ``market_condition`` (or the environment's
    regime when omitted) via :func:`dataclasses.replace`; each result row stores
    ``market_condition`` for grouping.

    If ``n_vcs`` is set (2–5), :func:`run_simulation` builds a multi-investor
    panel with distinct priors and thresholds; otherwise a single VC is used.
    """

    n_rounds: int = 5_000
    random_seed: int = 42
    market_condition: MarketCondition | None = None
    n_vcs: int | None = None


# ---------------------------------------------------------------------------
# Per-receiver outcomes (one row segment per VC)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReceiverOutcome:
    """Posterior, decision, and payoff for a single investor after observing ``signal``."""

    receiver_id: str
    posterior: float
    invest: bool
    expected_vc_return: float
    vc_payoff: float


def receiver_outcomes(
    *,
    signal: float,
    founder_type: FounderType,
    environment: SignalingEnvironment,
    receivers: Sequence[BeliefReceiver],
    receiver_ids: Sequence[str],
) -> tuple[ReceiverOutcome, ...]:
    """
    Run the belief-update and investment stage for each receiver in parallel.

    All receivers observe the same ``signal``; each uses its own
    ``environment`` (typically shared) and policy. Payoffs assume the same
    return structure per type if that receiver invests.
    """
    if len(receivers) != len(receiver_ids):
        raise ValueError("receivers and receiver_ids must have the same length.")
    out: list[ReceiverOutcome] = []
    for rid, recv in zip(receiver_ids, receivers):
        posterior = recv.posterior_belief_high(signal)
        invest = recv.decide_investment(signal)
        ev_return = recv.expected_return_if_invest(signal)
        if not invest:
            vc_payoff = 0.0
        elif founder_type is FounderType.HIGH:
            vc_payoff = environment.vc_return_high
        else:
            vc_payoff = environment.vc_return_low
        out.append(
            ReceiverOutcome(
                receiver_id=rid,
                posterior=posterior,
                invest=invest,
                expected_vc_return=ev_return,
                vc_payoff=vc_payoff,
            )
        )
    return tuple(out)


def founder_funded(outcomes: Sequence[ReceiverOutcome]) -> bool:
    """Whether the founder receives capital from at least one investor."""
    return any(o.invest for o in outcomes)


def n_interested_vcs(outcomes: Sequence[ReceiverOutcome]) -> int:
    """Count of VCs who choose to invest after observing the signal."""
    return sum(1 for o in outcomes if o.invest)


def competitive_funding(outcomes: Sequence[ReceiverOutcome]) -> bool:
    """True when two or more investors want in (simple competitive round flag)."""
    return n_interested_vcs(outcomes) >= 2


def flatten_receiver_columns(
    outcomes: Sequence[ReceiverOutcome],
    *,
    single_receiver: bool,
) -> dict[str, Any]:
    """
    Build flat dict columns for a DataFrame row.

    With a single receiver, legacy names apply (``posterior``, ``invest``, …).
    With several receivers, fields are prefixed by ``receiver_id`` so columns
    stay unique.
    """
    if not outcomes:
        return {}
    if single_receiver:
        o = outcomes[0]
        return {
            "posterior": o.posterior,
            "invest": o.invest,
            "expected_vc_return": o.expected_vc_return,
            "vc_payoff": o.vc_payoff,
        }
    d: dict[str, Any] = {}
    for o in outcomes:
        rid = o.receiver_id
        d[f"{rid}_posterior"] = o.posterior
        d[f"{rid}_invest"] = o.invest
        d[f"{rid}_expected_vc_return"] = o.expected_vc_return
        d[f"{rid}_vc_payoff"] = o.vc_payoff
    return d


def round_result_column_order(receiver_ids: Sequence[str], *, single_receiver: bool) -> list[str]:
    """Stable column order for :func:`run_n_rounds` output."""
    base = [
        "round",
        "founder_type",
        "signal",
        "market_condition",
        "n_interested_vcs",
        "founder_valuation",
        "competitive_funding",
        "founder_payoff",
    ]
    if single_receiver:
        return base + ["posterior", "invest", "expected_vc_return", "vc_payoff"]
    for rid in receiver_ids:
        base.extend(
            [
                f"{rid}_posterior",
                f"{rid}_invest",
                f"{rid}_expected_vc_return",
                f"{rid}_vc_payoff",
            ]
        )
    return base


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


@dataclass
class SignalingSimulationEngine:
    """
    Stateful driver: one environment, one signal rule, one or more receivers.

    ``environment.market_regime`` should match the regime used in each round
    (``market_condition`` on the engine, or :func:`run_n_rounds` sets both via
    :func:`dataclasses.replace`).

    * **Strategic signaling:** replace ``signal_strategy`` with a class that
      optimizes over ``SignalingRoundContext`` (and optionally future args).
    * **Multiple VCs:** pass a tuple of receivers and matching ``receiver_ids``;
      the founder is funded if any receiver invests (see :func:`founder_funded`).
    """

    environment: SignalingEnvironment
    receivers: tuple[BeliefReceiver, ...]
    signal_strategy: SignalStrategy = field(default_factory=GaussianNoiseSignalStrategy)
    receiver_ids: tuple[str, ...] | None = None
    market_condition: MarketCondition | None = None
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42),
    )

    def __post_init__(self) -> None:
        if not self.receivers:
            raise ValueError("At least one receiver is required.")
        if self.receiver_ids is None:
            self.receiver_ids = tuple(f"vc_{i}" for i in range(len(self.receivers)))
        elif len(self.receiver_ids) != len(self.receivers):
            raise ValueError("receiver_ids must align with receivers.")

    def run_round(self) -> dict[str, Any]:
        """
        One independent round: θ ~ prior, signal from ``signal_strategy``,
        each receiver updates and decides, then founder / VC payoffs.
        """
        env = self.environment
        rng = self.rng
        regime = self.market_condition if self.market_condition is not None else env.market_regime

        founder_type = sample_founder_type(env, rng)
        founder = Founder(founder_type=founder_type, environment=env)
        ctx = SignalingRoundContext(
            environment=env,
            founder_type=founder_type,
            market_condition=regime,
        )
        signal = self.signal_strategy.choose_signal(founder, rng, context=ctx)

        assert self.receiver_ids is not None
        outcomes = receiver_outcomes(
            signal=signal,
            founder_type=founder_type,
            environment=env,
            receivers=self.receivers,
            receiver_ids=self.receiver_ids,
        )
        funded = founder_funded(outcomes)
        founder_payoff = founder.utility(signal, funded)
        n_inv = n_interested_vcs(outcomes)
        valuation = env.founder_valuation(n_inv)

        row: dict[str, Any] = {
            "founder_type": founder_type.value,
            "signal": signal,
            "market_condition": regime.value,
            "n_interested_vcs": n_inv,
            "founder_valuation": valuation,
            "competitive_funding": competitive_funding(outcomes),
            "founder_payoff": founder_payoff,
        }
        row.update(
            flatten_receiver_columns(
                outcomes,
                single_receiver=len(self.receivers) == 1,
            )
        )
        return row


def run_single_round(
    env: SignalingEnvironment,
    vc: VentureCapitalist,
    rng: np.random.Generator,
    market_condition: MarketCondition | None = None,
    signal_strategy: SignalStrategy | None = None,
) -> dict[str, Any]:
    """
    Simulate one independent round (legacy helper).

    Prefer :class:`SignalingSimulationEngine` or :func:`run_n_rounds` for new
    code; this keeps a thin wrapper for callers that already pass ``rng``.

    The environment's ``market_regime`` is set for this round (explicit
    ``market_condition`` or else ``env.market_regime``); the VC uses that
    environment so beliefs and thresholds stay consistent.
    """
    regime = market_condition if market_condition is not None else env.market_regime
    env_run = replace(env, market_regime=regime)
    vc_run = replace(vc, environment=env_run)
    engine = SignalingSimulationEngine(
        environment=env_run,
        receivers=(vc_run,),
        signal_strategy=signal_strategy or GaussianNoiseSignalStrategy(),
        market_condition=regime,
        rng=rng,
    )
    return engine.run_round()


def run_n_rounds(
    n: int,
    environment: SignalingEnvironment,
    *,
    receivers: Sequence[BeliefReceiver] | None = None,
    vc: VC | None = None,
    n_vcs: int | None = None,
    receiver_ids: Sequence[str] | None = None,
    signal_strategy: SignalStrategy | None = None,
    random_seed: int | None = 42,
    market_condition: MarketCondition | None = None,
    investment_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Run ``n`` i.i.d. rounds and return a tidy :class:`pandas.DataFrame`.

    Each round samples type, emits a signal via ``signal_strategy`` (default:
    Gaussian noise around type means), then each entry in ``receivers``
    updates beliefs and decides. If only ``vc`` is provided, it is wrapped as
    the sole receiver.

    Parameters
    ----------
    n
        Number of rounds (rows).
    environment
        Shared structural parameters for likelihoods and payoffs.
    receivers
        One or more objects implementing :class:`~game.protocols.BeliefReceiver`.
    vc
        Convenience: single VC used when ``receivers`` is omitted.
    n_vcs
        If set (2–5), build a default VC panel via :func:`~game.multi_vc.build_vc_panel`
        (mutually exclusive with ``receivers`` and ``vc``).
    receiver_ids
        Column prefixes when multiple receivers; auto ``vc_0``, ``vc_1``, … if omitted.
    signal_strategy
        Pluggable signal rule; default is exogenous Gaussian noise.
    random_seed
        RNG seed, or ``None`` for non-deterministic runs.
    market_condition
        Sets ``environment.market_regime`` for the run (via :func:`dataclasses.replace`).
        ``None`` uses ``environment.market_regime``. Stored on each row and passed
        in ``SignalingRoundContext`` for strategies.
    investment_threshold
        Used only when building a default ``VC`` (both ``receivers`` and ``vc`` omitted).
        ``None`` uses :meth:`SignalingEnvironment.effective_investment_threshold`.
        A float fixes the cutoff regardless of regime.

    Returns
    -------
    pandas.DataFrame
        One row per round; multi-receiver runs use prefixed columns (see
        :func:`flatten_receiver_columns`).
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    specified = sum(x is not None for x in (receivers, vc, n_vcs))
    if specified > 1:
        raise ValueError("Pass at most one of: receivers, vc, n_vcs.")

    regime = environment.market_regime if market_condition is None else market_condition
    env = replace(environment, market_regime=regime)

    if n_vcs is not None:
        recv_tuple = build_vc_panel(env, n_vcs)
    elif receivers is None:
        if vc is None:
            recv_tuple = (VC(environment=env, investment_threshold=investment_threshold),)
        else:
            recv_tuple = (_receiver_with_environment(vc, env),)
    else:
        recv_tuple = tuple(_receiver_with_environment(r, env) for r in receivers)
        if not recv_tuple:
            raise ValueError("receivers must be non-empty.")

    ids_tuple: tuple[str, ...] | None
    if receiver_ids is None:
        ids_tuple = None
    else:
        ids_tuple = tuple(receiver_ids)

    rng = np.random.default_rng(random_seed)
    engine = SignalingSimulationEngine(
        environment=env,
        receivers=recv_tuple,
        receiver_ids=ids_tuple,
        signal_strategy=signal_strategy or GaussianNoiseSignalStrategy(),
        market_condition=regime,
        rng=rng,
    )

    assert engine.receiver_ids is not None
    single = len(recv_tuple) == 1
    columns = round_result_column_order(engine.receiver_ids, single_receiver=single)

    rows: list[dict[str, Any]] = []
    for i in range(n):
        row = engine.run_round()
        row["round"] = i
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[columns]


def run_simulation(
    env: SignalingEnvironment,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    """
    Run many independent rounds using :class:`SimulationConfig`.

    This is a convenience wrapper around :func:`run_n_rounds` for scripts that
    already pass a config object.
    """
    cfg = config or SimulationConfig()
    regime = cfg.market_condition if cfg.market_condition is not None else env.market_regime
    kwargs: dict[str, Any] = {
        "random_seed": cfg.random_seed,
        "market_condition": regime,
    }
    if cfg.n_vcs is not None:
        kwargs["n_vcs"] = cfg.n_vcs
    return run_n_rounds(cfg.n_rounds, env, **kwargs)


def save_results_csv(df: pd.DataFrame, path: Path) -> None:
    """Persist simulation output for downstream analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@dataclass(frozen=True)
class FundingEvaluationMetrics:
    """
    Treat funding as a binary classifier: positive label = high-quality founder
    (should fund); prediction = at least one VC invests.

    * **FPR** — share of low-type rounds that received funding.
    * **FNR** — share of high-type rounds that received no funding.
    * **Accuracy** — fraction of rounds where (funded iff high type).
    """

    n_rounds: int
    n_high: int
    n_low: int
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    false_positive_rate: float
    false_negative_rate: float
    accuracy: float


def funded_series(df: pd.DataFrame) -> pd.Series:
    """True when at least one receiver invested (works for single- and multi-VC runs)."""
    if "n_interested_vcs" not in df.columns:
        raise ValueError("Expected column 'n_interested_vcs' in simulation results.")
    return df["n_interested_vcs"].astype(int) > 0


def funding_evaluation_metrics(df: pd.DataFrame) -> FundingEvaluationMetrics:
    """
    Confusion matrix vs. an ideal that funds all high types and no low types.

    Rows must include ``founder_type`` and ``n_interested_vcs``.
    """
    type_col = "founder_type"
    if type_col not in df.columns:
        raise ValueError("Expected column 'founder_type' in simulation results.")

    funded = funded_series(df)
    is_high = df[type_col] == FounderType.HIGH.value
    is_low = df[type_col] == FounderType.LOW.value

    n_high = int(is_high.sum())
    n_low = int(is_low.sum())
    tp = int((is_high & funded).sum())
    tn = int((is_low & ~funded).sum())
    fp = int((is_low & funded).sum())
    fn = int((is_high & ~funded).sum())

    fpr = fp / n_low if n_low else math.nan
    fnr = fn / n_high if n_high else math.nan
    acc = (tp + tn) / len(df) if len(df) else math.nan

    return FundingEvaluationMetrics(
        n_rounds=len(df),
        n_high=n_high,
        n_low=n_low,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        false_positive_rate=float(fpr),
        false_negative_rate=float(fnr),
        accuracy=float(acc),
    )


def print_funding_evaluation_summary(m: FundingEvaluationMetrics) -> None:
    """Print a compact text table of classification-style funding metrics."""
    def pct(x: float) -> str:
        return "—" if math.isnan(x) else f"{100.0 * x:.2f}%"

    rows = [
        ("Rounds (total)", str(m.n_rounds)),
        ("  High-type rounds", str(m.n_high)),
        ("  Low-type rounds", str(m.n_low)),
        ("", ""),
        ("True positives (high & funded)", str(m.true_positives)),
        ("True negatives (low & rejected)", str(m.true_negatives)),
        ("False positives (low & funded)", str(m.false_positives)),
        ("False negatives (high & rejected)", str(m.false_negatives)),
        ("", ""),
        ("False positive rate (low funded / low)", pct(m.false_positive_rate)),
        ("False negative rate (high rejected / high)", pct(m.false_negative_rate)),
        ("Overall accuracy", pct(m.accuracy)),
    ]
    label_w = max(len(label) for label, _ in rows)
    print("\nFunding vs. latent type (binary prediction of high quality)")
    print("-" * (label_w + 12))
    for label, value in rows:
        if label == "":
            print()
        else:
            print(f"{label:<{label_w}}  {value}")
    print("-" * (label_w + 12))
