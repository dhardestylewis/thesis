# Russo & Van Roy (2014) — Learning to Optimize via Information-Directed Sampling
## Full Annotation

**Authors:** Daniel Russo, Benjamin Van Roy
**Venue:** NeurIPS 2014 (extended arXiv:1403.5556)

---

## High-Level Framing
- **Problem:** Multi-armed bandit / online optimization where a decision-maker must balance exploration and exploitation. Standard approaches (UCB, Thompson sampling) fail to properly account for *which type of information* an action reveals.
- **Key idea:** An action should be selected not only if it has high expected reward (exploitation) or high uncertainty (exploration), but if it provides the most *useful* information relative to the regret it incurs.
- **Algorithm (IDS):** At each round, select the action $a_t$ that minimizes:
  $$\Psi_t(a) = \frac{[\Delta_t(a)]^2}{I_t(a)}$$
  where $\Delta_t(a) = \mathbb{E}[f(a^*) - f(a) \mid \mathcal{H}_t]$ is the expected single-period regret and $I_t(a) = I(a^*; y_t \mid \mathcal{H}_t, a_t = a)$ is the mutual information between the optimal action and the next observation.

---

## Information-Directed Sampling Objective
- IDS minimizes the *ratio* of squared expected regret to information gain, not their difference.
- This ratio, called the **information ratio** $\Psi_t(a)$, captures the efficiency of learning: high information gain is only valuable if it reduces future regret proportionally.
- **Degenerate cases:**
  - If $I_t(a) = 0$ (action reveals nothing): $\Psi_t(a) = \infty$ unless $\Delta_t(a) = 0$ (pure exploitation).
  - If $\Delta_t(a) = 0$ (clearly optimal action): select it regardless of information gain.

---

## Regret Bounds
- **Theorem:** For IDS, $\mathbb{E}[R_T] \leq \sqrt{H(\pi^*) \cdot T \cdot \Gamma^*}$ where $H(\pi^*)$ is the entropy of the optimal action distribution and $\Gamma^*$ is the maximum information ratio.
- This bound applies across a very general class of models — Bernoulli, Gaussian, linear, combinatorial bandits — without requiring separate proofs for each.
- IDS achieves order-optimal bounds (up to poly-log) in all the standard settings where UCB/Thompson are known to be good.
- **Surprise:** IDS empirically outperforms UCB and Thompson sampling even on Bernoulli bandits where those algorithms are asymptotically optimal.

---

## Information Ratio Examples
- **Bernoulli bandit:** IDS with binary actions has $\Gamma^* = 1/2$, matching Thompson sampling's optimal rate.
- **Linear bandits:** IDS recovers the $\sqrt{dT}$ bound via the structure of the mutual information.
- **Cascading bandits (semi-bandit feedback):** IDS significantly outperforms UCB here because observing which items are clicked reveals information about *multiple* arms simultaneously — something UCB ignores.

---

## Practical Considerations
- The optimization over $a$ of $\Psi_t(a)$ may require solving a single-period problem that can be expensive. The paper addresses this via randomized policies (mixtures over actions) and closed-form solutions for specific models.
- IDS can be implemented as a randomized policy by mixing between exploration and exploitation actions to achieve the optimal information ratio.

---

## Connections to This Course / Thesis
- **Information-theoretic BO:** IDS is the direct antecedent to information-theoretic BO acquisition functions (Entropy Search, Predictive Entropy Search — Hennig & Schuler 2012, Hernández-Lobato et al. 2014). These acquisition functions minimize $H(x^* \mid \mathcal{D}_t)$ rather than the information ratio, but the spirit is the same.
- **Connecting to empirical Bayes:** The information ratio $I_t(a)$ requires a prior over $a^*$. In the empirical Bayes approach to BO, this prior is estimated from marginal likelihood — providing the bridge between IDS and the EB-GP approach.
- **Zoning as IDS:** In the thesis context, IDS would prescribe approving the zoning case that best resolves uncertainty about the spatial protest mechanism relative to the housing yield sacrifice. This is qualitatively different from GP-UCB, which approves where yield + uncertainty is highest without regard for what is learned.
