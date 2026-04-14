# Wilson et al. (2016) — Deep Kernel Learning
## Full Annotation

**Authors:** Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, Eric P. Xing
**Venue:** AISTATS 2016 (arXiv:1511.02222)

---

## High-Level Framing
- **Problem:** Standard GP kernels (RBF, Matérn) are stationary and assume smoothness properties that may not match the true function. Deep neural networks learn flexible representations but lack principled uncertainty quantification and are hard to calibrate.
- **Insight:** Instead of treating deep learning and kernel methods as alternatives, compose them: use a deep neural network as a feature extractor, then apply a standard base kernel in feature space. The result is a *deep kernel* with the flexibility of deep learning and the probabilistic rigor of GPs.
- **Algorithm (DKL):** Define kernel $k_\theta(x, x') = k_\text{base}(h_\theta(x), h_\theta(x'))$ where $h_\theta: \mathcal{X} \to \mathbb{R}^d$ is a deep neural network and $k_\text{base}$ is a spectral mixture kernel. Learn $\theta$ jointly with GP hyperparameters via marginal likelihood maximization.

---

## Technical Components

### Spectral Mixture (SM) Base Kernel
- Wilson & Adams (2013): any stationary kernel can be represented as a spectral mixture:
  $$k_\text{SM}(r) = \sum_{q} w_q \cos(2\pi \mu_q^\top r) \exp(-2\pi^2 r^\top \text{diag}(\sigma_q^2) r)$$
- The SM kernel is highly expressive and can approximate any stationary covariance function.
- Combined with DNN features: the deep kernel $k_\text{DKL}$ can represent non-stationary, structured patterns.

### Scalability via KISS-GP / SKI
- Exact GP inference costs $O(n^3)$. DKL uses *Structured Kernel Interpolation (SKI)*: approximate the kernel matrix via sparse interpolation weights on an inducing grid.
- Inference then costs $O(n)$ for training, $O(1)$ per test prediction (amortized via structured matrix operations — Kronecker and Toeplitz algebra).
- This allows DKL to scale to datasets with $n = 2 \times 10^6$ observations.

### Joint Learning
- Both the DNN parameters $\theta$ and GP hyperparameters (SM weights, means, variances, noise) are optimized simultaneously via gradient descent on the log marginal likelihood.
- This is an **empirical Bayes** procedure: the marginal likelihood integrates out the GP function values and optimizes a hierarchical prior over the kernel structure.

---

## Empirical Results
- Benchmarks: UCI datasets (energy, protein, flight delay — up to 2M examples), MNIST, CIFAR.
- DKL consistently outperforms: standard GPs with flexible kernels, standalone DNNs, and deep GPs (Damianou & Lawrence 2013) on most tasks.
- Particularly strong on structured data where DNNs learn useful representations (images, tabular data with feature interactions).

---

## Limitations
- The marginal likelihood objective may overfit the kernel structure to the training data, especially with many SM components.
- The inducing grid approximation (SKI) trades exactness for scalability; accuracy degrades for highly non-stationary functions.
- The deep feature space is not interpretable in the same way as original input dimensions — making it harder to reason about which features the kernel "uses."

---

## Connections to This Course / Thesis
- **Empirical Bayes connection:** DKL is the most direct example of using empirical Bayes (marginal likelihood) to learn a complex prior structure. The Week 11 question — "is GP marginal likelihood the same EB as Efron's shrinkage?" — is partly addressed here: both are Type-II ML, but DKL operates on the kernel (covariance structure) while Efron's EB operates on the prior mean.
- **Deep features for zoning:** The SCM for zoning has both tabular features (density, distance) and geographic structure (spatial adjacency). A deep kernel over a GNN (graph neural network) feature extractor could represent protest risk as a function of the neighborhood's position in the spatial graph — preserving geometric structure that Matérn kernels cannot capture.
- **Scalability:** With $n = 7,074$ zoning cases, DKL is computationally feasible and would replace the ARD Matérn surrogate with a more expressive (but harder to interpret) alternative.
