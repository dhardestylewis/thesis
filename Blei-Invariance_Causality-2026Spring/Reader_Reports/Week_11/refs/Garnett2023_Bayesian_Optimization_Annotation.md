# Garnett (2023) — Bayesian Optimization
## Full Annotation (Chapter / Conceptual Synthesis)

**Author:** Roman Garnett
**Publisher:** Cambridge University Press, 2023 (https://bayesoptbook.com)

---

## High-Level Framing
- **Problem:** Finding the global maximum of an objective function $f(x)$ that is black-box, derivative-free, and highly expensive to evaluate.
- **Contribution:** Systematizes the field of Bayesian Optimization (BO). Frames it explicitly as a problem of **sequential decision making under uncertainty** (adaptive experimental design).
- **Core Loop:** 
  1. Build a probabilistic proxy (surrogate model, usually a Gaussian Process) to predict $f(x)$ and quantify the epistemic uncertainty around that prediction.
  2. Optimize a cheap "acquisition function" that leverages the surrogate to balance exploration (high uncertainty) vs. exploitation (high predicted mean).
  3. Sample the true function at the recommended point, update the posterior, and repeat.

## Modeling Beliefs (The Surrogate)
- Gaussian Processes (GPs) are the dominant workhorse because they provide closed-form Bayesian updates for the transition density $p(y|x, \mathcal{D})$.
- The prior encapsulates structural assumptions about the environment:
  - **Stationarity:** Most kernels (RBF, Matérn) assume the covariance relies only on the distance between points, not absolute position.
  - **Smoothness:** Matérn $\nu$ parameters define how differentiable the loss surface is.
- Miscalibrated priors lead to pathological failure modes (e.g., over-exploration or premature convergence).

## Adaptive Experimental Design (Acquisition)
- **Expected Improvement (EI):** Analytically closed-form over GPs. Evaluates the expected magnitude of improvement over the current best-known observation.
- **Upper Confidence Bound (UCB):** $\mu(x) + \beta \sigma(x)$. Explicitly parameterizes the fundamental explore-exploit trade-off via $\beta$.
- **Information-Theoretic Approaches (Entropy Search, Predictive Entropy Search):** Target the reduction of entropy in the posterior distribution of the optimal location $x^*$ or optimal value $f^*$.

## Connections to Causal Invariance (Week 10 Thesis Context)
- **Experimental Design vs. Ex-post Analysis:** While Synthetic Control (SC) and Invariant Risk Minimization (IRM) attempt to extract causal signals from historical, static observational sweeps, BO takes an active-learning stance: *How do we optimally design the next data-gathering intervention?*
- **The Stationary Assumption in Policy:** BO relies on the assumption that evaluating $f(x_t)$ does not fundamentally alter the underlying generative surface for $f(x_{t+1})$. 
- **Domain Generalization Constraint:** In municipal zoning policy (the thesis domain), updating density at location $X$ causes unobserved geographic contagion constraints. Deploying BO over spatial policy requires recognizing that each "sample" could trigger an unmodeled state-shift. The city's sequential approvals act exactly like a naive acquisition function subject to a NIMBY-protest safety constraint (Constrained Bayesian Optimization).
