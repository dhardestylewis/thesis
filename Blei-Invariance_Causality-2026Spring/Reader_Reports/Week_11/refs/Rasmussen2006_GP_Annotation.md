# Rasmussen & Williams (2006) — Gaussian Processes for Machine Learning
## Full Annotation (Selected Chapters)

**Authors:** Carl Edward Rasmussen, Christopher K. I. Williams
**Publisher:** MIT Press, 2006 (freely available at gaussianprocess.org/gpml)

---

## Overview
The canonical reference for GPs in machine learning. Covers GP regression, classification, model selection, and connections to other methods. For BO purposes, Chapters 2, 4, and 5 are most relevant.

---

## Chapter 2: Regression
- **GP definition:** A GP is a collection of random variables, any finite subset of which has a multivariate Gaussian distribution. Fully specified by mean $m(x) = \mathbb{E}[f(x)]$ and covariance $k(x, x') = \text{Cov}(f(x), f(x'))$.
- **Posterior (noiseless):**
  $$f_* \mid X, \mathbf{y}, X_* \sim \mathcal{N}(\bar{f}_*, \text{Var}(f_*))$$
  $$\bar{f}_* = K(X_*, X) K(X,X)^{-1} \mathbf{y}$$
  $$\text{Var}(f_*) = K(X_*, X_*) - K(X_*, X) K(X,X)^{-1} K(X, X_*)$$
- **Posterior (with noise $\sigma_n^2$):** Replace $K(X,X)$ with $K(X,X) + \sigma_n^2 I$.
- Prediction is exact given the kernel; the computational bottleneck is $O(n^3)$ for the Cholesky decomposition of $K$.

---

## Chapter 4: Covariance Functions (Kernels)
Key kernels and their properties:

| Kernel | Formula | Properties |
|--------|---------|------------|
| Squared Exponential (SE/RBF) | $\exp(-\|x-x'\|^2/2\ell^2)$ | Infinitely differentiable; over-smooth for many real functions |
| Matérn-$\nu$ | $\frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}\|r\|}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}\|r\|}{\ell}\right)$ | $\lceil\nu\rceil - 1$ times differentiable; $\nu=3/2$ once, $\nu=5/2$ twice differentiable |
| Linear | $\sigma_b^2 + \sigma_v^2 x^\top x'$ | Non-stationary; reduces GP to Bayesian linear regression |
| Periodic | $\exp\left(-\frac{2\sin^2(\pi|r|/p)}{\ell^2}\right)$ | Periodic functions with period $p$ |

- **Automatic Relevance Determination (ARD):** Replace scalar $\ell$ with vector $\boldsymbol{\ell} = (\ell_1, \ldots, \ell_d)$ in isotropic kernels. ARD length-scales effectively select informative dimensions: small $\ell_j$ = input $j$ is important; large $\ell_j$ = input $j$ is irrelevant.

---

## Chapter 5: Model Selection and Hyperparameters (Empirical Bayes)
- **Log marginal likelihood:**
  $$\log p(\mathbf{y} \mid X, \theta) = -\frac{1}{2}\mathbf{y}^\top (K_\theta + \sigma_n^2 I)^{-1} \mathbf{y} - \frac{1}{2}\log|K_\theta + \sigma_n^2 I| - \frac{n}{2}\log 2\pi$$
  - Term 1: *data fit* — how well the GP mean matches the observations.
  - Term 2: *complexity penalty* — penalizes overly complex kernels (Occam factor).
  - Term 3: normalization constant.
- **Type-II Maximum Likelihood (empirical Bayes):** Optimize $\theta = (\boldsymbol{\ell}, \sigma_f^2, \sigma_n^2)$ to maximize $\log p(\mathbf{y} \mid X, \theta)$. This is gradient-based (via automatic differentiation) and typically run with multiple restarts to avoid local optima.
- **Full Bayes vs. Type-II ML:** Full Bayes marginalizes over $\theta$: $p(\mathbf{y} \mid X) = \int p(\mathbf{y} \mid X, \theta) p(\theta) d\theta$. This is intractable in closed form; MCMC or variational inference is required. Type-II ML is a point estimate — faster but potentially miscalibrated.

---

## Chapter 8: Approximations for Large Datasets
- **Nyström approximation:** Approximate $K$ using $m \ll n$ inducing points.
- **Sparse GP (FITC):** Fully Independent Training Conditional approximation — scales to $O(nm^2)$.
- **Variational sparse GP (Titsias 2009):** Principled lower bound on the marginal likelihood; avoids the FITC miscalibration issue.

---

## Connections to This Course / Thesis
- **The reference for GP mechanics:** Chapters 2 and 4 are prerequisites for understanding every BO paper in this reading list. The BO field takes the GP posterior formulas as given.
- **ARD ↔ causal discovery:** ARD length-scale estimation is a form of automatic feature selection. In the SCM context, dimensions with large ARD length-scales correspond to variables that are causally irrelevant to the outcome (d-separated from $Y$). This provides a soft, continuous version of the causal Markov blanket.
- **Type-II ML ↔ empirical Bayes:** The Chapter 5 discussion is the formal statement of the EB procedure used throughout the thesis's BO formulation. It also provides the bridge to Blei's upcoming empirical Bayes lectures: marginal likelihood maximization is EB applied to the GP covariance structure.
