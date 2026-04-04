# VC Signaling Lab

### A game-theoretic simulation of startup fundraising under asymmetric information

---

## What is this?

This project explores how investors make decisions under uncertainty, a core problem in venture capital, private equity, and financial markets.

It simulates how venture capitalists decide whether to invest in startups when they cannot directly observe true company quality.

Startups send signals (such as traction or pitch quality), and investors update their beliefs using Bayesian inference before deciding whether to invest.

The goal is to study:

* how often good startups get funded
* how often bad startups slip through
* how information asymmetry affects investment decisions

---

## Key Results

From a baseline simulation of 5,000 funding rounds:

* Accuracy: ~92%
* False Positive Rate: ~5.5% (bad startups funded)
* False Negative Rate: ~12.3% (good startups rejected)

This demonstrates strong separation between high- and low-quality founders while preserving realistic decision errors under uncertainty.

---

## Overview

The model is a **sender–receiver game** with asymmetric information:

* The founder has a hidden type:
  $$
  \theta \in {\text{high}, \text{low}}
  $$

* A signal $s$ is generated from a type-dependent normal distribution:
  $$
  s \sim \mathcal{N}(\mu_\theta, \sigma^2)
  $$

* The VC observes $s$ but not $\theta$, and must infer quality.

Investment decisions are made using posterior beliefs derived from Bayes’ rule.

---

## Bayesian Updating

Let the prior probability of a high-quality founder be:

$$
\pi_h = P(\theta = \text{high})
$$

After observing signal $s$, the VC computes:

$$
P(\theta = \text{high} \mid s)
==============================

\frac{p(s \mid \text{high}) \cdot \pi_h}{p(s)}
$$

where

$$
p(s)
====

p(s \mid \text{high}) \cdot \pi_h
+
p(s \mid \text{low}) \cdot (1 - \pi_h)
$$

The likelihood $p(s \mid \theta)$ is modeled as a Gaussian density with:

$$
s \mid \theta \sim \mathcal{N}(\mu_\theta, \sigma^2)
$$

---

## Decision Rule

The baseline investment rule is:

$$
\text{Invest if } P(\theta = \text{high} \mid s) > \tau
$$

where $\tau \in [0,1]$ is a decision threshold.

Alternative rules can use expected returns:

$$
\mathbb{E}[\text{net return} \mid s]
$$

---

## Conceptual Framework

This model belongs to the class of **games with one-sided incomplete information**:

* Founder observes $\theta$
* Investor observes only $s$
* Beliefs are updated via Bayesian inference

Because signal distributions overlap, perfect separation is impossible. This creates realistic:

* false positives
* false negatives
* probabilistic decision-making

---

## Asymmetric Information and Signaling

* $\theta$ is private information
* $s$ is informative but noisy
* Investors form posterior beliefs using $(s, \pi)$

Signaling arises from the dependence:

$$
s \mid \theta
$$

Extensions can allow founders to **strategically choose signals**, introducing costly signaling equilibria.

---

## Repository Layout

```
vc-game-theory-simulator/
├── main.py
├── requirements.txt
├── README.md
├── data/
├── plots/
├── experiments/
│   ├── baseline.py
│   ├── market_regimes.py
│   ├── multi_vc.py
└── game/
    ├── environment.py
    ├── founder.py
    ├── vc.py
    ├── simulation.py
```

---

## Environment Setup

```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Simulation

```
python main.py
```

Other modes:

```
python -m experiments.baseline
python -m experiments.market_regimes
python -m experiments.multi_vc
```

---

## Outputs

* `data/` → simulation results
* `plots/` → visualizations
* posterior vs signal plots
* signal distributions by type

---

## Reference Output

```
Unconditional investment rate: 0.3464

By type:
high → invest ≈ 0.877  
low  → invest ≈ 0.055
```

---

## Extensions

* Endogenous signaling (costly signals)
* Multiple investors
* Dynamic learning over time
* Structural estimation of parameters

---

## License

Add a license before public distribution.
