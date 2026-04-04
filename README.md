## What is this?

This project explores how investors make decisions under uncertainty — a core problem in venture capital, private equity, and financial markets.
This project simulates how venture capitalists decide whether to invest in startups when they cannot directly observe true company quality.

Startups send signals (like traction or pitch quality), and investors update their beliefs using Bayesian inference before deciding to invest.

The goal is to study how often good startups get funded, bad startups slip through, and how information asymmetry affects investment decisions.

## Key Results

From a baseline simulation of 5,000 funding rounds:

- Accuracy: ~92%
- False Positive Rate: ~5.5% (bad startups funded)
- False Negative Rate: ~12.3% (good startups rejected)

This demonstrates strong separation between high- and low-quality founders, while preserving realistic decision errors under uncertainty.

# VC Signaling Lab
### A game-theoretic simulation of startup fundraising under asymmetric information

Computational implementation of a **sender–receiver** model with binary hidden quality, applied to venture financing: a hidden binary quality, a type-conditional Gaussian signal, posterior beliefs via Bayes’ rule, and threshold-based investment. The design prioritizes **identifiable likelihoods**, **auditable updating**, and **batch output** for figures and reduced-form statistics.

---

## Overview

Each round samples a founder type \(\theta \in \{\text{high}, \text{low}\}\) from a fixed prior. Conditional on \(\theta\), a scalar signal \(s\) is drawn from a normal distribution with type-specific mean and common variance. The VC does not observe \(\theta\); inference proceeds from \(s\) under the **correctly specified** signal law and prior. The default policy invests when the posterior probability of high quality exceeds an exogenous cutoff; the implementation also reports **expected return** conditional on \(s\) under the induced posterior.

Founder payoffs combine investment benefit with a **quadratic signaling cost** whose slope may differ by type, consistent with heterogeneous cost of mimicking high signals. The environment supports optional **market** shifts to the effective prior and threshold (`game/environment.py`), enabling comparative statics without changing the core likelihood structure.

---

## Conceptual framework

The model belongs to the family of **games with one-sided incomplete information**: the informed party observes \(\theta\); the uninformed party observes only \(s \sim F(\cdot \mid \theta)\). The relevant economic question is how much information about \(\theta\) is transmitted when \(F\) overlaps across types—equivalently, when the VC faces a **mixture** over signals and must update beliefs before acting.

The baseline treats the conditional distribution of \(s\) given \(\theta\) as **exogenous** (reduced-form signal technology). That restriction isolates the **statistical decision problem**—likelihood evaluation, marginalization, and Bayes’ rule—from equilibrium determination of signaling effort or disclosure. The modular simulation layer is built to accommodate **endogenous** signal rules later without rewriting the batch machinery.

---

## Asymmetric information and signaling

**Asymmetric information** is modeled in the standard way: \(\theta\) is the founder’s private information; the VC’s information set is \(\{s\}\) plus the common prior and structural parameters \((F, \pi)\). The VC’s action is a measurable function of the **posterior** over \(\theta\) given \(s\).

**Signaling** here denotes the statistical dependence \(s \mid \theta\): the observable is informative about \(\theta\) because its distribution shifts with the latent state. With continuous \(s\) and overlapping supports, **pooling** in signal space is generic: distinct types can generate arbitrarily similar realizations. Extensions in which founders choose costly actions that shift \(F(\cdot \mid \theta)\) recover the classic costly-signaling interpretation on top of the same updating block.

---

## Bayesian updating

Let $\pi_h = P(\theta = \text{high})$. After observing signal $s$, the VC computes the posterior probability that the founder is high quality:

$$
P(\theta = \text{high} \mid s)
=
\frac{p(s \mid \text{high}) \, \pi_h}{p(s)},
\qquad
p(s)
=
p(s \mid \text{high}) \, \pi_h
+
p(s \mid \text{low}) \, (1 - \pi_h).
$$

In code, $p(s \mid \theta)$ is modeled as a Gaussian density with mean $\mu_\theta$ and variance $\sigma^2$. The posterior is computed explicitly from the likelihoods and the prior (see `game/vc.py`).

The baseline investment rule compares $P(\theta = \text{high} \mid s)$ to a threshold in $[0,1]$. Alternative decision rules can instead be based on $\mathbb{E}[\text{net return} \mid s]$ using the same posterior.
---

## Repository layout

```
vc-game-theory-simulator/
├── main.py                 # Entry point → experiments.baseline
├── requirements.txt
├── README.md
├── data/                   # Round-level panels (gitignored)
├── plots/                  # Figures (gitignored)
├── experiments/
│   ├── locations.py        # REPO_ROOT, RESULTS_DIR, FIGURES_DIR
│   ├── baseline.py         # Single receiver: panel + figures + summaries
│   ├── market_regimes.py   # Regime sweep → stacked CSV
│   ├── multi_vc.py         # Multi-receiver panel → wide CSV
│   └── run_baseline.py     # Legacy path-friendly alias for baseline
└── game/
    ├── __init__.py
    ├── environment.py      # Priors, Gaussian signal law, payoffs, market adjustments
    ├── founder.py          # Type draw, signal draw, utility
    ├── vc.py               # Posterior, threshold rule, expected return
    ├── simulation.py       # Monte Carlo engine, exports, plots
    ├── protocols.py        # Abstractions for signal strategies and receivers
    └── signaling_strategies.py
```

---

## Environment and dependencies

**Stack:** Python 3.10+; `numpy`, `pandas`, `matplotlib` (pinned in `requirements.txt`).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the simulations

Default design: 5,000 rounds, seed 42, environment `DEFAULT_ENV` in `game/environment.py`.

**Single-receiver baseline** (panel, two figures, funding–type table):

```bash
python main.py
# or: python -m experiments.baseline
# or: python experiments/run_baseline.py
```

**Regime comparative statics** (independent samples per `MarketCondition`, one stacked CSV):

```bash
python -m experiments.market_regimes
```

**Multi-receiver panel** (three investors; CSV only—scalar-`posterior` plots are single-receiver):

```bash
python -m experiments.multi_vc
```

| Artifact | Contents |
|----------|----------|
| `data/single_receiver_baseline.csv` | Round-level panel: type, signal, posterior, invest, payoffs |
| `data/market_regime_sweep.csv` | Stacked panels; includes `market_condition` |
| `data/multi_receiver_panel.csv` | Wide columns `vc_*_posterior`, `vc_*_invest`, … |
| `plots/baseline_posterior_vs_signal.png` | \(s\) vs. \(P(\text{high} \mid s)\) |
| `plots/baseline_signal_density_by_type.png` | Empirical \(s \mid \theta\) |

---

## Reference run

Illustrative stdout from `python main.py` under the shipped defaults (fixed seed in `experiments/baseline.py`):

```
results: .../data/single_receiver_baseline.csv  (n=5000)
figures: .../plots/baseline_posterior_vs_signal.png
         .../plots/baseline_signal_density_by_type.png
...
Unconditional investment rate: 0.3464
By latent type:
  high:  E[invest|θ]=0.8774,  E[π|θ]=0.8284  (n=1770)
  low:  E[invest|θ]=0.0554,  E[π|θ]=0.0933  (n=3230)
```

Conditional investment rates and mean posteriors by **realized** \(\theta\) describe selection into funding and how beliefs co-move with the latent state.

---

## Extensions

- **Equilibrium signaling:** Endogenize signal choice (effort, bias, or disclosure) with type-dependent costs; characterize pooling/separating/hybrid outcomes relative to the reduced-form \(s \mid \theta\) used here.
- **Multiple receivers:** The engine supports several Bayesian investors on the same realized \(s\); extend to correlated errors, sequential evaluation, or bargaining.
- **Dynamics:** Enrich `MarketCondition` with state variables; allow VC learning or reputation across rounds.
- **Structural estimation:** Jointly identify \((\pi, \mu_\theta, \sigma)\) and payoff parameters from observables or experimental elicitation of priors.
- **Verification:** Unit tests for posterior algebra and deterministic round accounting; CI on pull requests.

---

## License

Specify a license before public distribution.
