# Sui et al. (2015) — Safe Exploration for Optimization with Gaussian Processes (SafeOpt)
## Full Annotation

**Authors:** Yanan Sui, Alkis Gotovos, Joel W. Burdick, Andreas Krause
**Venue:** ICML 2015, JMLR W&CP vol. 37
**Source:** proceedings.mlr.press/v37/sui15.pdf

---

## High-Level Framing
- **Problem:** Optimize an unknown function $f$ from noisy GP-modeled observations, subject to the hard constraint that *every* evaluation must satisfy $f(x_t) \geq h$ (a safety threshold). Standard BO (including GP-UCB) violates this by design: it explores below the threshold to gather information.
- **Setting:** Single objective $f$, single safety threshold $h \in \mathbb{R}$. **Not a multi-objective or Pareto method.**
- **Key idea:** Maintain a *safe set* $S_t \subseteq D$ — the set of decisions currently known with high probability to satisfy the constraint. Only sample within $S_t$.

---

## Algorithm
At each round $t$, maintain three sets using GP confidence intervals $[l_t(x), u_t(x)]$:
1. **Safe set** $S_t = \{x : l_t(x) \geq h\}$ — decisions provably safe with high probability.
2. **Expanders** $G_t \subseteq S_t$ — safe decisions whose evaluation could expand $S_t$ (i.e., reveal new safe regions).
3. **Maximizers** $M_t \subseteq S_t$ — safe decisions that could contain the optimum (high upper confidence bound).

Select $x_t \in G_t \cup M_t$: either expand the safe frontier or exploit a promising region. Never sample outside $S_t$.

---

## Main Theorem (Theorem 1)
Assume $f$ is $L$-Lipschitz, $\|f\|_k^2 \leq B$, noise is sub-Gaussian. Let $\bar{R}_0(S_0)$ be the $\epsilon$-reachable set from seed $S_0$, and $t^*$ be the smallest integer satisfying:
$$\frac{t^*}{\beta_{t^*} \gamma_{t^*}} \geq \frac{C_1 (|\bar{R}_0(S_0)| + 1)}{\epsilon^2}, \quad C_1 = \frac{8}{\log(1+\sigma^{-2})}$$

Then with probability $\geq 1-\delta$, jointly:
- **Safety:** $\forall t \geq 1,\ f(x_t) \geq h$
- **Near-optimality:** $\forall t \geq t^*,\ f(\hat{x}_t) \geq f^*_\epsilon - \epsilon$

where $f^*_\epsilon$ is the optimum within the $\epsilon$-reachable set.

**Critical nuance:** SafeOpt only guarantees convergence to the optimum *reachable under safety constraints* — $f^*_\epsilon$, not the global $f^*$. If the global optimum is not reachable from $S_0$ without crossing unsafe regions, SafeOpt cannot find it.

---

## What SafeOpt Is NOT
- **Not Pareto/multi-objective:** It has a single objective $f$ and a single binary safety constraint $f(x) \geq h$. There is no second optimization objective.
- **Not a constrained optimizer in the Letham sense:** Letham et al. (2019) treat constraints as soft (probabilistic); SafeOpt treats the safety constraint as hard (must hold at every evaluation with high probability).
- **Comparison to outcomes:** When comparing SafeOpt's *policy outcomes* to another policy's outcomes on two metrics (e.g., yield and protest rate), if SafeOpt dominates on both dimensions, one can correctly say the outcomes are Pareto-superior — but this is a statement about outcome comparison, not about SafeOpt's internal optimization structure.

---

## Connections to This Course / Thesis
- **Zoning application:** The protest petition constraint maps naturally to SafeOpt's safety threshold $h$: require $\Pr(P=1 \mid do(A=a)) \leq \delta$, i.e., the latent safety margin $g(a) \leq 0$. SafeOpt would only approve zoning cases that are currently known to satisfy this constraint with high probability.
- **Reachability constraint:** SafeOpt's $\epsilon$-reachable set formalizes the key limitation: if all high-yield development corridors are "protest-unsafe," SafeOpt will never explore them, even if they might yield the global optimum. This is precisely the NIMBY resistance problem in Austin — the best development sites may be unreachable under safety constraints.
- **Seed set $S_0$:** SafeOpt requires an initial safe set. For the zoning problem, $S_0$ would be the set of parcel types with historically zero protest petitions — likely only low-density single-family rezonings, a strongly conservative starting point.
- **Comparison to Week 10 formulation:** Week 10 used *Constrained EI* (Letham 2019), which allows unsafe evaluations with some probability. SafeOpt provides strictly stronger safety guarantees but at the cost of convergence to a more limited reachable optimum.
