# Fan et al. (2024/2026) — Causal Invariance Learning via Efficient Nonconvex Optimization
## Full Annotation (line-by-line reading of main body, 12,806 lines total)

**Authors:** Zhenyu Wang, Yifan Hu, Peter Bühlmann, Zijian Guo
**arXiv:** 2412.11850 (January 2026 revision)

---

## Lines 1–100: Title, Abstract
- **Problem:** Identify direct causes of an outcome from multi-environment data via invariance principle.
- **Contribution:** Negative Weighted DRO (NegDRO) — continuous minimax formulation avoiding exhaustive subset enumeration.
- **Three claims:** (i) Sufficient & nearly necessary identification conditions. (ii) Benign nonconvex landscape where all stationary points approximate the causal model. (iii) Gradient-based algorithm with convergence rates in both sample size and iterations.
- **Running example:** Advertising channels (search ads X₁, social media X₂) → sales Y.

## Lines 100–235: §1 Introduction
- **Gap Q1 (Identification):** Prior conditions (Peters 2016, Fan 2024-EILLS, Arjovsky IRM) either abstract or overly restrictive (requiring |E|≥p environments or exhaustive heterogeneity).
- **Gap Q2 (Computation):** ICP, EILLS require exhaustive enumeration over 2^p subsets → exponential cost.
- **NegDRO answer:** Continuous optimization, polynomial-time, works with as few as **2 environments**.

## Lines 236–472: §1.2–1.4 Contributions, Related Work, Preliminaries
- **4 contributions:** NegDRO formulation; identification conditions; benign landscape + efficient algorithm; weaker conditions for limited interventions.
- **Related work:** Multi-source learning (Group DRO), IRM, MM-REx, global optimality of nonconvex optimization.
- **Definition 1 (SEMs):** Linear structural equation models with Zⱼ = fⱼ(Pa(Zⱼ), εⱼ).

## Lines 493–670: §2 Multi-environment Causal Invariance Learning
### §2.1 Risk-Invariance Principle
- **Linear causal model (Eq. 1):** Y^e = (β*)ᵀ X^e_{S*} + ε^e_Y, invariant across e ∈ E.
- **Risk-invariance (Eq. 2):** E[ε^e_Y]² = σ²_Y for all e (weaker than requiring identical noise distributions).
- **Invariant prediction model set (Eq. 3):** B_inv = {b: E[Y^e − bᵀX^e]² = E[Y^f − bᵀX^f]² ∀ e,f}.
- **Contrast with conditional-mean invariance (Fan et al. 2024 EILLS):** Requires E[ε^e_Y|X^e_{S*}]=0, which excludes hidden confounding. Risk-invariance allows it.

### §2.2 Existing Methods: Exhaustive Enumeration
- **Example 2:** 2 environments, p=2 features. ICP enumerates ∅, {1}, {2}, {1,2}. Only S={1} gives environment-independent risk → β*=(1,0).
- **EILLS (Fan et al. 2024):** Regularized least squares with indicator penalty; nonconvex, still requires subset enumeration.

### §2.3 NegDRO Formulation (Eq. 9)
- **Constrained problem (Eq. 7):** min_b E[Y¹−bᵀX¹]² s.t. risk invariance.
- **Penalty relaxation (Eq. 8):** Parameter γ balances max risk vs risk discrepancy.
- **Minimax form (Eq. 9):**
  ```
  b^γ_Neg = arg min_b max_{w∈U(γ)} Σ_e w_e E[Y^e − bᵀX^e]²
  where U(γ) = {w: Σw_e=1, min w_e ≥ −γ}
  ```
- **Key insight:** γ=0 → Group DRO (simplex). γ>0 → **negative weights allowed** → enforces risk invariance but introduces nonconvexity (difference-of-convex).

## Lines 1025–1600: §3 Additive Intervention & Causal Identification
### §3.1 Additive Intervention Regime (Eq. 10–12)
- SEM: (Y,X) = B(Y,X) + (ε_Y, ε_X). Noise decomposes as η (systematic) + (0, δ^e) (environment-specific).
- η_Y never intervened (ensures risk invariance). E[η(δ^e)ᵀ] = 0. Hidden confounding allowed via E[η_Y η_X] ≠ 0.

### §3.2 Identification Conditions
- **Condition 1a (Statistical):** ∃ disjoint E₁, E₂ ⊆ E, weights w, w': Σ_{e∈E₁} w_e E[X^eX^{eᵀ}] ≻ Σ_{f∈E₂} w'_f E[X^fX^{fᵀ}].
- **Theorem 1:** Under additive intervention + Condition 1a → B_inv = {β*} (unique invariant model).
- **Theorem 2:** Near-necessity — if each environment intervenes on only one coordinate, uniqueness ⟹ Condition 1a.
- **Condition 1b (Optimization):** ∃ w₀ ∈ Δ_{|E|} s.t. λ = λ_min(A(w₀)) > 0, where A(w) = Σ_e (w_e − 1/|E|) E[X^eX^{eᵀ}].
- **Proposition 1:** Conditions 1a ⟺ 1b.
- **λ** captures **degree of environmental heterogeneity** — larger λ → easier identification.

### §3.3 Finite γ (Proposition 2)
- ‖b^γ_Neg − β*‖₂ ≲ 1/(λ(1+γ|E|)). Larger γ, more environments, more heterogeneity → closer to β*.

## Lines 1611–2100: §4 Computationally Efficient Algorithm
### §4.1 Benign Landscape (Theorem 4)
- For any b ∈ ℝᵖ:
  ```
  ‖b − β*‖₂ ≲ (1/λ)(1/(1+γ|E|)) + ‖∇Φ_μ(b)‖₂ + √(μ/λ)
  ```
- **All stationary points are near β*.** First result of its kind in causal invariance learning.

### §4.2 Algorithm 1: Gradient Descent-Maximization
- **Inner step:** Closed-form weight update (strongly concave in w due to ‖w‖² penalty μ).
- **Outer step:** Gradient descent on b with Danskin's theorem.
- **Theorem 5 (Convergence):** With optimal μ≍T^{−1/2}:
  ```
  ‖b̂^γ − β*‖₂ ≲ 1/(1+γ|E|) + T^{−1/4} + n^{−1/4}
  ```
- For ε-accuracy: γ=Ω(ε⁻¹), n=Ω(ε⁻⁴), T=Ω(ε⁻⁴). **Polynomial in p** (vs exponential for ICP/EILLS).

## Lines 2239–2565: §5 Limited Additive Interventions
- **Condition 4:** Only outcome's children D need intervention. Strictly weaker than Condition 1b.
- **Theorem 6:** Under limited interventions + no hidden confounding, β* achieves smallest risk among B_inv → constrained problem still recovers β*.
- **Comparison:** CausalDantzig fails (singular matrix). DRIG fails (no reference environment). NegDRO succeeds in all three regimes (limited/weak/strong).

## Lines 2565–2760: §6 Numerical Results & §7 Conclusion
- **Setup:** p∈{5,10,40,100}, 4 environments, n up to 20,000.
- **Key findings:**
  - NegDRO converges to β* as γ, n, T increase (consistent with theory).
  - EILLS competitive at p≤20 but hits 30-min time limit at p≥25.
  - ICP hits time limit at p≥15.
  - NegDRO runs in polynomial time at all dimensions tested.
- **Conclusion:** NegDRO transforms combinatorial subset search into gradient-based continuous optimization. Identification conditions reveal geometric structure enabling benign landscape.

## Lines 3177–12806: Appendix (Proofs)
- **Appendix A:** Proofs of Theorems 1–6, Propositions 1–2.
- **Appendix B:** Unpenalized (nonsmooth) NegDRO analysis using generalized stationary points (Clarke subdifferential).
- **Appendix C:** Limited intervention extensions (finite-sample analysis, hidden confounding), comparison with CausalDantzig/DRIG.
- **Appendix D–E:** Full technical proofs and supporting lemmas.
- **Appendix F:** SEM illustrations, additional simulation details including hidden confounders.

---

## Critical Notes (Week 06 additions)

### Cross-paper connections
- **Same invariance condition, different operationalization:** Peters et al. (2016), NegDRO, and BIP (Wu 2025) all enforce $P_e(Y|X_{S^*}) = \text{const}$ but operationalize it differently. Peters: sequential hypothesis test, set-valued output. NegDRO: continuous minimax, point estimate. BIP: Bayesian posterior over selectors. None of the papers explicitly positions itself against the others' operationalization choices.
- **NegDRO as adversarial Bühlmann:** Bühlmann (2020) frames invariance as robustness to distributional shift (worst-case over a shift class). NegDRO's $U(\gamma)$ with negative weights is an adversarial extension — it can reverse-stress specific environments below zero weight, going beyond standard DRO. This makes NegDRO strictly more aggressive than Bühlmann's robustness framing.
- **λ (NegDRO) vs. μ_min (BIP):** Both are "heterogeneity" measures but they are not the same. λ is a spectral condition on second moments of X across environments (Condition 1b). μ_min is a KL divergence between pooled and local conditionals of Y|X. A dataset could satisfy one but not the other. Which condition is easier to certify empirically is not discussed in either paper.
- **BIP can handle what NegDRO cannot:** NegDRO requires a point identification (unique $\beta^*$); with few environments and many features, this may not hold. BIP's posterior honestly represents ambiguity in that case via multi-modality. Conversely, NegDRO operates in polynomial time; BIP's exact inference is $O(2^p)$.

### Benchmarking critique
- **NegDRO benchmarks are synthetic:** Simulations at p ∈ {5, 10, 40, 100}, 4 environments, n up to 20,000. This is more representative of applied settings than BIP's gene study, but still no real-data benchmark.
- **Comparison set is limited:** NegDRO is compared to EILLS and ICP, but not to BIP, IRM (Arjovsky 2019), or REx (Krueger 2021) — the closest related methods. The paper's claim of outperformance is relative only to methods that hit time limits at $p \geq 20$.
- **Runtime advantage is the main empirical contribution** — at p=100, competitors time out. The causal accuracy at feasible dimensions is comparable to EILLS.

### Open questions / disagreements
- **Negative weights' interpretation:** $w_e < 0$ means an environment is de-weighted below zero — the optimization is actively penalized for performing well on that environment. The geometric interpretation is clear (it enforces risk parity) but the statistical interpretation is unusual. What does a negative environment weight mean as a prior over interventions?
- **Condition 1b vs. Condition 4 (limited interventions):** The paper relaxes to Condition 4 (children of Y only need intervention), which is strictly weaker. But Condition 4 still requires knowing the causal graph structure of Y's children — a strong assumption in practice. This is not flagged as a limitation.
- **NegDRO as pre-screening filter for BIP:** Wu et al. propose a size-restricted prior with parameter $p_{\max}$ to cap BIP's support. NegDRO's point estimate $\hat\beta^\gamma$ could serve as a pre-screening filter — setting $z_j=0$ for features with $|\hat\beta^\gamma_j| < \varepsilon$ reduces the effective model count from $2^p$ to $2^{p'}$, analogous to the lasso pre-screening that Peters et al. suggest for ICP. This is defensible because it restricts the support rather than contaminating the prior with data-driven information. The naive version of this idea (using NegDRO to initialize BIP's prior) has two problems: (1) λ (NegDRO's heterogeneity measure, a spectral condition on $\Sigma^e_X$) and $\mu_{\min}$ (BIP's heterogeneity measure, a KL condition on $Y|X$ conditionals) are not the same quantity — the most informative environment pair under λ may not be the most informative under $\mu_{\min}$; (2) using the same data to form the prior and the likelihood is empirical Bayes, which overstates posterior confidence.
