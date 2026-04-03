"""
Matplotlib figures for simulation result panels.

Publication-oriented defaults: larger canvases, consistent typography, light
grids, and a grayscale palette (color only where two series must be distinguished).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .environment import FounderType
from .simulation import FundingEvaluationMetrics, funded_series

# Two-tone grayscale for high vs. low (print-safe, minimal)
_GRAY_HIGH = "0.2"
_GRAY_LOW = "0.55"
# Single fill for categorical bars
_GRAY_BAR = "0.42"

_DPI = 150
_FIGSIZE_MAIN = (9.5, 6.0)
_PAD_INCHES = 0.08

# Consistent typography and spacing (single-receiver figures)
_PUB_RC: dict[str, float | str | tuple[float, float]] = {
    "figure.figsize": _FIGSIZE_MAIN,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.titlepad": 12,
    "axes.labelpad": 7,
    "xtick.major.pad": 5,
    "ytick.major.pad": 5,
    "legend.frameon": False,
    "legend.borderpad": 0.6,
    "legend.handlelength": 1.6,
    "grid.alpha": 0.45,
    "grid.linestyle": ":",
    "grid.linewidth": 0.7,
}


def _posterior_column(df: pd.DataFrame) -> str:
    if "posterior" in df.columns:
        return "posterior"
    if "posterior_high" in df.columns:
        return "posterior_high"
    raise ValueError("Expected column 'posterior' or 'posterior_high' in panel.")


def _type_column(df: pd.DataFrame) -> str:
    if "founder_type" in df.columns:
        return "founder_type"
    if "true_type" in df.columns:
        return "true_type"
    raise ValueError("Expected column 'founder_type' or 'true_type' in panel.")


def _setup_axes(ax) -> None:
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_signal_vs_posterior(
    df: pd.DataFrame,
    path: Path,
    *,
    title: str = "Signal vs. posterior belief",
) -> None:
    """Scatter: observed signal on x, posterior P(high | signal) on y, by true type."""
    path.parent.mkdir(parents=True, exist_ok=True)
    type_col = _type_column(df)
    post_col = _posterior_column(df)

    high = df[df[type_col] == FounderType.HIGH.value]
    low = df[df[type_col] == FounderType.LOW.value]

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(layout="constrained")
        ax.scatter(
            high["signal"],
            high[post_col],
            s=16,
            alpha=0.38,
            label="High type",
            c=_GRAY_HIGH,
            edgecolors="none",
        )
        ax.scatter(
            low["signal"],
            low[post_col],
            s=16,
            alpha=0.38,
            label="Low type",
            c=_GRAY_LOW,
            edgecolors="none",
        )

        ax.set_xlabel("Signal")
        ax.set_ylabel("Posterior belief P(high quality | signal)")
        ax.set_title(title)
        ax.legend(loc="lower right")
        _setup_axes(ax)

    fig.savefig(path, dpi=_DPI, bbox_inches="tight", pad_inches=_PAD_INCHES)
    plt.close(fig)


def plot_signal_distribution_by_type(
    df: pd.DataFrame,
    path: Path,
    *,
    title: str = "Distribution of signals by founder type",
) -> None:
    """Overlapping density histograms of signal by latent type (grayscale fills)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    type_col = _type_column(df)

    high = df[df[type_col] == FounderType.HIGH.value]["signal"]
    low = df[df[type_col] == FounderType.LOW.value]["signal"]
    combined = np.concatenate([high.to_numpy(), low.to_numpy()])
    n_bins = min(50, max(25, int(np.sqrt(len(df)))))
    bins = np.linspace(float(combined.min()), float(combined.max()), n_bins + 1)

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(layout="constrained")
        ax.hist(
            low,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            label="Low type",
            color=_GRAY_LOW,
            edgecolor="none",
        )
        ax.hist(
            high,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            label="High type",
            color=_GRAY_HIGH,
            edgecolor="none",
        )

        ax.set_xlabel("Signal")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(loc="upper right")
        _setup_axes(ax)

    fig.savefig(path, dpi=_DPI, bbox_inches="tight", pad_inches=_PAD_INCHES)
    plt.close(fig)


def plot_funding_confusion_counts(
    metrics: FundingEvaluationMetrics,
    path: Path,
    *,
    title: str = "Funding outcomes vs. latent type",
) -> None:
    """Bar chart of true/false positives and negatives (funded iff at least one VC invests)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = [
        "True positives\n(high & funded)",
        "True negatives\n(low & not funded)",
        "False positives\n(low & funded)",
        "False negatives\n(high & not funded)",
    ]
    values = [
        metrics.true_positives,
        metrics.true_negatives,
        metrics.false_positives,
        metrics.false_negatives,
    ]

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(layout="constrained")
        x = np.arange(len(labels))
        ax.bar(
            x,
            values,
            color=_GRAY_BAR,
            width=0.68,
            edgecolor="0.15",
            linewidth=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Count (rounds)")
        ax.set_title(title)
        _setup_axes(ax)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.savefig(path, dpi=_DPI, bbox_inches="tight", pad_inches=_PAD_INCHES)
    plt.close(fig)


def plot_investment_rate_by_type(
    df: pd.DataFrame,
    path: Path,
    *,
    title: str = "Investment rate by founder type",
) -> None:
    """Bar chart of P(funded | type); funded means at least one interested VC."""
    path.parent.mkdir(parents=True, exist_ok=True)
    type_col = _type_column(df)
    work = df.assign(_funded=funded_series(df))
    rates = work.groupby(type_col, sort=True)["_funded"].mean()
    order = [FounderType.HIGH.value, FounderType.LOW.value]
    labels_display = ["High type", "Low type"]
    heights = [float(rates.get(t, 0.0)) for t in order]

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(layout="constrained")
        x = np.arange(len(order))
        ax.bar(
            x,
            heights,
            color=_GRAY_BAR,
            width=0.52,
            edgecolor="0.15",
            linewidth=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels_display)
        ax.set_ylabel("Share of rounds funded")
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        _setup_axes(ax)
        for i, h in enumerate(heights):
            ax.text(i, h + 0.025, f"{h:.3f}", ha="center", va="bottom", color="0.25")

    fig.savefig(path, dpi=_DPI, bbox_inches="tight", pad_inches=_PAD_INCHES)
    plt.close(fig)


# Backward-compatible names
def plot_posterior_vs_signal(
    df: pd.DataFrame,
    path: Path,
    title: str = "Bayesian posterior vs. observed signal",
) -> None:
    plot_signal_vs_posterior(df, path, title=title)
