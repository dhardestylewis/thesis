# Letham et al. (2019) — Constrained Bayesian Optimization with Noisy Experiments
## Full Annotation

**Authors:** Benjamin Letham, Brian Karrer, Guilherme Ottoni, Eytan Bakshy (Facebook)
**Venue:** Bayesian Analysis 14(2), 2019 (arXiv:1706.07094)

---

## High-Level Framing
- **Problem:** Optimizing continuous system parameters via randomized experiments (A/B tests) where: (1) observations are noisy; (2) there are outcome constraints (e.g., optimizing ranking quality without degrading page load time); (3) batches of experiments run in parallel.
- **Motivation:** Standard BO (EI with GP) assumes nearly noiseless observations; heuristics for noisy EI degrade significantly under high noise, making them unsuitable for A/B test environments.
- **Contribution:** Derives a proper Bayesian Expected Improvement under noisy observations and noisy constraints, approximated efficiently via quasi-Monte Carlo (qMC).

---

## Noisy Expected Improvement
Standard EI conditions on the best *observed* value $f^* = \max_i y_i$. With noise, $y_i = f(x_i) + \epsilon_i$, so $f^*$ itself is a noisy estimate of the true optimum. This leads to:
$$\text{EI}(x) = \mathbb{E}[\max(0, f(x) - f^*)]$$
where the expectation is over both the GP posterior at $x$ **and** the uncertainty in $f^*$ (the incumbent).

The paper derives:
$$\text{noisy-EI}(x) = \int \text{EI}(x \mid f^*) \, p(f^* \mid \mathcal{D}) \, df^*$$
This integral does not have a closed form under noisy observations; the paper approximates it via qMC samples from the joint GP posterior.

---

## Constrained Optimization
- Multiple outcome GPs: objective $f$ and constraints $c_1, \ldots, c_k$, all with noise.
- **Constrained EI:** $\text{cEI}(x) = \text{noisy-EI}(x) \cdot \prod_j \Pr(c_j(x) \leq 0)$
- Under independence: factorizes; in general, joint GP modeling is needed.
- The paper models constraint satisfaction as a probability under the GP posterior, not a deterministic boundary.

---

## Batch Optimization (qEI)
- Batch $q$-EI: select a set $\{x_1, \ldots, x_q\}$ jointly maximizing the expected improvement of the best point in the batch over the current best.
- qMC approximation: sample $f$ from the joint GP posterior, compute EI for each sample, average. Efficiently parallelizable.

---

## Real-World Experiments (Facebook)
1. **Ranking system optimization:** 4 continuous parameters; 3 outcome constraints. qMC-based cEI found better ranking quality within the constraint budget faster than alternatives.
2. **Compiler flag optimization:** LLVM compiler flags for server workloads. Constrained BO navigated complex interactions between flags (some improve throughput but increase latency).

---

## Connections to This Course / Thesis
- **Direct connection to thesis formulation:** The Week 10/11 project update uses exactly this framework — a protest constraint $\Pr(P=1 \mid do(A=a)) \leq \delta$ modeled as a GP with a link function. Letham et al. provide the principled derivation for noisy constraints that was used informally in the project update.
- **A/B tests ↔ zoning cases:** Each approved zoning case is an "experiment" with noisy outcomes (realized units may differ from planned units; protest filings have measurement error). The noisy EI derivation is directly applicable.
- **qMC vs. analytical EI:** The thesis implementation (Week 11 project update) uses analytical EI for tractability; switching to qMC-based cEI (as in Letham et al.) would be a natural improvement.
