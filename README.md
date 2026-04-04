# VC Signaling Lab

### A game-theoretic simulation of startup fundraising under asymmetric information

---

## What is this?

This project explores how investors make decisions under uncertainty, a core problem in venture capital, private equity, and financial markets.

It simulates how venture capitalists decide whether to invest in startups when they cannot directly observe true company quality.

Startups send signals such as traction or pitch quality, and investors update their beliefs using Bayesian inference before deciding whether to invest.

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

---

## Overview

The model is a **sender–receiver game** with asymmetric information.

The founder has a hidden type:

```math
\theta \in \{\text{high}, \text{low}\}
```

A signal $s$ is generated from a type-dependent normal distribution:

```math
s \sim \mathcal{N}(\mu_\theta, \sigma^2)
```

The VC observes $s$ but not $\theta$, and must infer quality.

---

## Bayesian Updating

Let the prior probability of a high-quality founder be:

```math
\pi_h = P(\theta = \text{high})
```

After observing signal $s$, the VC computes:

```math
P(\theta = \text{high} \mid s)
=
\frac{p(s \mid \text{high}) \cdot \pi_h}{p(s)}
```

where

```math
p(s)
=
p(s \mid \text{high}) \cdot \pi_h
+
p(s \mid \text{low}) \cdot (1 - \pi_h)
```

The likelihood is:

```math
s \mid \theta \sim \mathcal{N}(\mu_\theta, \sigma^2)
```

---

## Decision Rule

```math
\text{Invest if } P(\theta = \text{high} \mid s) > \tau
```

where $\tau \in [0,1]$.

Alternative:

```math
\mathbb{E}[\text{net return} \mid s]
```

---

## Conceptual Framework

* Founder observes $\theta$
* Investor observes only $s$
* Beliefs updated via Bayesian inference

Because signal distributions overlap:

* false positives occur
* false negatives occur
* decisions are probabilistic

---

## Asymmetric Information and Signaling

* $\theta$ is private information
* $s$ is noisy but informative

```math
s \mid \theta
```

Extensions allow costly signaling strategies.

---

## Repository Layout

```text
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

## Run

```bash
python main.py
```

---

## Output

* data/ → simulation results
* plots/ → visualizations

---

## Example Output

```text
Unconditional investment rate: 0.3464

high -> invest ≈ 0.877  
low  -> invest ≈ 0.055  
```

---

## Extensions

* Endogenous signaling
* Multi-investor models
* Dynamic learning
* Structural estimation

---

## License

Add a license.
