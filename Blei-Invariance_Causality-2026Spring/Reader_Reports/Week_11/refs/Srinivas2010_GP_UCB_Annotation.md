# Srinivas et al. (2010) — Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design
## Full Annotation

**Authors:** Niranjan Srinivas, Andreas Krause, Sham M. Kakade, Matthias Seeger
**Venue:** ICML 2010 (extended journal version, arXiv:0912.3995)

---

## High-Level Framing
- **Problem:** Optimizing an unknown, noisy, expensive-to-evaluate function $f$ over a compact domain — framed as a multi-armed bandit problem with structured payoffs.
- **Contribution:** First sublinear regret bounds for GP optimization. Introduces GP-UCB and connects BO to experimental design via the maximum information gain $\gamma_T$.
- **Key result:** Cumulative regret $R_T \leq \sqrt{C_1 T \beta_T \gamma_T}$ with high probability, i.e.\ $O^*(\sqrt{T\gamma_T})$ suppressing log factors in $\beta_T$, where $C_1 = 8/\log(1+\sigma^{-2})$.

---

## The GP-UCB Algorithm
- At each round $t$, select:
  $$x_t = \arg\max_{x \in \mathcal{X}} \mu_{t-1}(x) + \beta_t^{1/2} \sigma_{t-1}(x)$$
- $\mu_{t-1}$, $\sigma_{t-1}$: GP posterior mean and standard deviation after $t-1$ observations.
- $\beta_t$: exploration parameter — theoretically set as $\beta_t = 2\log(|\mathcal{X}| t^2 \pi^2 / 6\delta)$ for finite $\mathcal{X}$, adapted for continuous domains.
- Intuition: UCB selects points that are either estimated to be good (exploitation) or highly uncertain (exploration). The $\beta_t$ parameter controls the trade-off.

---

## Maximum Information Gain $\gamma_T$
- $\gamma_T = \max_{A \subseteq \mathcal{X}, |A|=T} I(\mathbf{y}_A; f)$ — maximum mutual information between $T$ observations and the function.
- **Key bounds on $\gamma_T$** (Table 1 in paper):
  - **Linear kernel:** $\gamma_T = O(d \log T)$
  - **RBF/SE kernel:** $\gamma_T = O((\log T)^{d+1})$ — nearly dimension-free for large $T$
  - **Matérn-$\nu$ kernel:** $\gamma_T = O(T^{d(d+1)/(2\nu+d(d+1))} (\log T))$ — slower if $\nu$ is small (rough functions). Note: $d/(2\nu+d)$ is the exponent for the *expected* information gain $E[I(y_T;f_T)]$, a different quantity.
- The proof uses submodularity of the information gain function, which allows greedy selection to approximate the optimum.

---

## Regret Bounds
- **Theorem 1 (Bayesian regret):** Under GP prior, $\Pr(R_T \leq \sqrt{C_1 T \beta_T \gamma_T} \text{ for all } T \geq 1) \geq 1 - \delta$
- **Theorem 2 (Frequentist/RKHS regret):** For $f$ with bounded RKHS norm $\|f\|_k \leq B$, same form of bound holds with adjusted $\beta_T$.
- The two regimes (Bayesian and frequentist/RKHS) require different proofs but yield the same functional form.

---

## Connection to Experimental Design
- GP-UCB is shown to approximately minimize the integrated posterior variance — connecting it to classical optimal experimental design (D-optimal, A-optimal criteria).
- This bridges the bandit (regret-minimization) and design (uncertainty-reduction) objectives: they are not as different as previously thought.
- The submodularity argument is key: it allows greedy information gain to achieve at least $(1-1/e)$ of the optimum, controlling the regret.

---

## Connections to This Course / Thesis
- **BO ↔ active causal discovery:** The Matérn smoothness parameter $\nu$ directly encodes assumptions about how the causal mechanism varies spatially. In the thesis domain (Austin zoning), choosing $\nu$ amounts to a prior on how smoothly protest hazard varies across neighborhoods.
- **Regret under mechanism shift:** The bound $O(\sqrt{T \gamma_T \beta_T})$ assumes the kernel is correctly specified throughout. If the causal mechanism changes at $T^*$, the effective $\gamma_T$ resets — implying the regret accumulated pre-change-point is not recoverable.
- **Experimental design angle:** Each zoning approval is an experiment. The SC framework (Shi et al. 2022) estimates what happened; GP-UCB would prescribe what to do next. The paper's experimental design interpretation is the formal link between the two.
