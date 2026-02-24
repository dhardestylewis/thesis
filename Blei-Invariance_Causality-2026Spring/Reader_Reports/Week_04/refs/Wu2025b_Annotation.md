# Wu et al. (2025) — Bayesian Invariance Modeling of Multi-Environment Data
## Full Annotation (line-by-line reading)

**Authors:** Luhuan Wu, Mingzhang Yin, Yixin Wang, John P. Cunningham, David M. Blei
**arXiv:** 2506.22675 (July 2025)

---

## Lines 1–87: Title, Authors, Abstract
- **Problem:** Identify invariant features — those with a stable predictive relationship to outcome across multiple environments.
- **Contribution:** Bayesian Invariant Prediction (BIP): a probabilistic model encoding invariant set as a latent variable z ∈ {0,1}^p, recovered via posterior inference.
- **Key claims:** (1) Under Peters et al. (2016) assumptions, BIP posterior targets true invariant features. (2) Posterior is consistent. (3) Greater environment heterogeneity → faster contraction. (4) VI-BIP scales to high dimensions.

## Lines 68–227: §1 Introduction
- **Invariant prediction** (Peters et al. 2016): given data from E environments, find features that govern the outcome the same way across all environments.
- Previous methods: hypothesis testing (ICP) or regularized optimization (EILLS).
- BIP key idea: **two assumptions**: (1) features follow different distributions across environments; (2) outcome depends on a subset of x in the same way across environments — this subset is the invariant set.
- BIP does NOT require linearity (unlike original ICP).
- **Invariant set = latent variable z**, posterior p(z|D).
- **Equation 1** — the BIP generative model:
  ```
  p(z, D) = p(z) ∏_e ∏_i pe(xz_ei) g(yei | xz_ei) pe(x−z_ei | yei, xz_ei)
  ```
  where g is invariant (same across environments) and pe varies by environment.
- **4 contributions:** (1) BIP model, (2) posterior consistency theory, (3) VI-BIP scalability, (4) empirical outperformance.

## Lines 229–261: §1.1 Related Works
- Invariance ↔ Independent Causal Mechanism (Peters et al. 2017, Schölkopf et al. 2021): p(y|x) and p(x) are independent under causal structure x → y.
- Historical roots: autonomy/modularity (Frisch et al. 1948, Hoover 2008), stable parent-child in causal graphs (Pearl 2009).
- Extensions: Heinze-Deml et al. (2018) nonlinear ICP; Rothenhäusler et al. (2019) causal Dantzig; Fan et al. (2023) EILLS; Gu et al. (2025) NP-hardness of exact invariance.

## Lines 262–528: §2 Bayesian Invariant Prediction
### §2.1 (Lines 264–297): Multi-environment data and invariance assumption
- **Assumption 1 (Invariance):** ∃ z* ∈ {0,1}^p such that pe(y | xz*) is invariant ∀ e ∈ E.
- **Eq. 2:** pe(x,y) = pe(xz*) p*(y|xz*) pe(x−z* | xz*, y)
- Goal: infer z* from observed multi-environment data.

### §2.2 (Lines 299–528): The BIP model
- **Definition 1 (Pooled conditional):** g(y|xz) := weighted average of local conditionals pooled across environments.
- **Proposition 1 (Identifiability):** g(y|xz) = pe(y|xz) ∀e ⟺ z = z*.
- **Generative process:** Draw z ~ p(z); for each environment e and observation i: (1) draw xz from pe(xz), (2) draw y from g(y|xz), (3) draw x−z from pe(x−z|xz,y).
- **Proposition 2 (Posterior expression — Eq. 3):**
  ```
  p(z|D) ∝ p(z) ∏_e ∏_i g(yei|xz_ei) / pe(yei|xz_ei)
  ```
  Posterior is a product of likelihood ratios between pooled and local conditionals. At z=z*, ratio=1 (maximum). At z≠z*, ratio<1.

### §2.3 (Lines 530–663): Exact Posterior Inference
- **Algorithm 1:** Iterate over all z in support of p(z). For each z, estimate local {pe(y|xz)} and pooled g(y|xz) via MLE (Eqs. 4–5), compute likelihood ratio, normalize.
- Complexity: O(2^p · c(D,p)) with full prior; O(p^pmax · c(D,pmax)) with size-restricted prior.

### §2.4 (Lines 665–745): Variational Inference (VI-BIP)
- Mean-field Bernoulli variational family: qφ(z) = ∏_j Bernoulli(z(j) | sigmoid(φ(j)))
- Maximize **ELBO** (Eq. 7) via stochastic optimization.
- Uses **U2G gradient estimator** (Yin et al. 2020) for discrete latent variables.
- Complexity: O(T·M·c(D,p)) for T iterations, M gradient samples.

## Lines 747–982: §3 Theory
### §3.1 (Lines 770–938): Main Results
- **Assumptions 2–5:** Uniqueness of z*, positive prior mass, estimation consistency, finite variance of log-likelihood ratio.
- **Theorem 1 (Posterior Consistency):** As n,E → ∞: (a) posterior mode → z*; (b) p(z*|D) → 1.
- **Theorem 2 (Contraction Rate):** TV(p(z|D), δ_{z*}) = O(R · e^{−κnEμ_min}) where:
  - **R** depends on prior (informative prior → smaller R → faster contraction)
  - **μ_min** = minimum KL between local and pooled conditionals among non-invariant z's — captures environment **heterogeneity** (more heterogeneous → larger μ_min → faster contraction)

### §3.2 (Lines 941–982): Finite-Sample Extensions & Assumption Violations
- Fixed E, n→∞: consistency holds if z* identifiable within observed environments.
- Fixed n, E→∞: need to control estimation bias (Assumption 8).
- **Violations:** Invariance violated → posterior targets most approximately invariant z. Non-unique z* → posterior distributes mass among all solutions. Prior misspecified → posterior targets best within support. Model misspecified → targets best within model class.

## Lines 984–1248: §4 Synthetic Study
- **§4.1:** p=3, confirm theory — posterior at z* converges to 1 with increasing n,E; faster under stronger interventions.
- **§4.2:** p=2, uncertainty quantification — multi-modal posterior with limited environments; adding environments resolves ambiguity.
- **§4.3:** p=10 and p=450 comparisons. Low-dim: BIP/VI-BIP competitive with exact inference. High-dim: VI-BIP outperforms all methods (existing methods fail at pre-screening step).

## Lines 1251–1391: §5 Gene Perturbation Study
- 6,170 genes in yeast (Kemmeren et al. 2014). 2 environments: 262 observational + 1,479 interventional samples.
- VI-BIP: highest precision, moderate recall. ICP-s: second most accurate but most conservative.
- VI-BIP predictions closely match top 10 findings of Meinshausen et al. (2016).

## Lines 1450–1489: §6 Discussion & Future Work
- **Limitations:** (1) BIP relies on fixed estimates of per-environment distributions — estimation bias can obscure posterior, especially with small n. (2) Only linear Gaussian models tested. (3) Mean-field VI may limit posterior expressiveness.
- **Future:** Integrate estimation uncertainty into probabilistic framework; richer model classes with amortized computation; better VI and discrete optimization methods.

## Lines 1619–3890: Supplementary Material
- **§A:** Full proofs of Theorems 1–7, Lemmas 1–3.
- **§B:** VI-BIP details: U2G gradient estimator (Algorithm 3), optimization hyperparameters (M=10–20 gradient samples, cyclical LR scheduler, SGD), implementation tricks (analytical KL gradients, penalty for infeasible z).
- **§C:** Synthetic data details: full generative process, 3 examples of uncertainty quantification with multiple invariant solutions.
- **§D:** Gene data details: evaluation protocol, hyperparameter grids for all methods, VI-BIP initialization (σ0=0.02, pmax=200, 10,000 iterations).
