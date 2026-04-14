# Eriksson et al. (2019) — Scalable Global Optimization via Local Bayesian Optimization (TuRBO)
## Full Annotation

**Authors:** David Eriksson, Michael Pearce, Jacob R. Gardner, Ryan Turner, Matthias Poloczek
**Venue:** NeurIPS 2019 (arXiv:1910.01739)

---

## High-Level Framing
- **Problem:** Global BO with GP surrogates fails on high-dimensional ($d \gtrsim 20$) and large-budget ($n \gtrsim 500$) problems due to: (1) over-exploration from inflated posterior variance in sparsely covered regions; (2) global stationarity assumption miscalibrates heterogeneous objectives.
- **Diagnosis:** A single global GP implicitly assumes the characteristic length-scale is constant across the domain. For heterogeneous objectives (e.g., RL reward with sparse regions, materials discovery), this leads to pathological acquisition values and wasted evaluations.
- **Algorithm (TuRBO):** Maintain a set of independent *trust regions*, each with its own local GP. Allocate new evaluations across trust regions via an implicit multi-armed bandit (Thompson sampling on local models).

---

## The TuRBO-1 Algorithm
1. Initialize a trust region $\mathcal{T}$ centered at the current best $x^*$, with side length $L$.
2. Fit a local GP on observations within $\mathcal{T}$ (or all observations, with the GP effectively ignoring distant ones via short length-scales).
3. Generate a batch of candidates using Thompson sampling from the local GP posterior within $\mathcal{T}$.
4. Evaluate the objective at the selected candidates; update the GP.
5. **Expand/shrink $\mathcal{T}$:** If enough consecutive improvements are made, double $L$; if enough consecutive failures, halve $L$. If $L < L_{\min}$, restart from a new random point.

**TuRBO-$m$:** Run $m$ independent trust regions simultaneously; allocate samples across regions using a bandit strategy based on local Thompson sampling, preferring regions with higher local optima.

---

## Key Theoretical Insight
- Over-exploration in global BO is an inherent consequence of the global GP's uncertainty quantification: distant, unevaluated regions always have high variance, inflating the UCB/EI there even when the global optimum is clearly localized.
- TuRBO does not fix this via a better acquisition function — it fundamentally changes the model's *scope* to local. By restricting the GP's domain, the effective RKHS complexity is lower and the surrogate calibrates faster.

---

## Empirical Results
- Benchmarks: Lunar lander (RL), robot pushing (continuous control), Rover (path planning), Branin (classic), Hartmann-6.
- TuRBO-1 and TuRBO-5 consistently outperform: standard BO (EI, UCB), CMA-ES, BOHAMIANN, SMAC, and REMBO across all benchmarks.
- Particularly dramatic improvements on RL and robotics, where heterogeneity is highest.

---

## Limitations
- No formal regret guarantees; the trust region dynamics are heuristic and problem-dependent.
- $L_{\min}$ and restart strategy introduce hyperparameters that must be tuned.
- Local independence between trust regions ignores global structure — beneficial for heterogeneous functions but wasteful if global correlations exist.

---

## Connections to This Course / Thesis
- **Spatial heterogeneity in zoning:** Austin zoning problems are plausibly heterogeneous — different neighborhoods have different NIMBY resistance dynamics (e.g., wealthy western districts vs. eastern affordable-housing corridors). TuRBO's local model assumption is more defensible than a single global GP with forced stationarity.
- **Trust region boundaries ↔ causal graph structure:** The thesis critique in Week 11 asks whether trust regions should be defined by Euclidean distance or by causal structure. TuRBO uses Euclidean distance in the action space; a causally-aware variant would use distance in the space of mechanism parameters.
- **Connection to DCBO:** Both TuRBO and Dynamic Causal BO (Aglietti 2021) address non-stationarity, but via different mechanisms. TuRBO handles *spatial* heterogeneity; DCBO handles *temporal* mechanism drift.
