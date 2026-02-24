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
