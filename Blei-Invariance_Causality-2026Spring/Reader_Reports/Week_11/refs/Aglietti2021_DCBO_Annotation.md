# Aglietti et al. (2021) — Dynamic Causal Bayesian Optimization (DCBO)
## Full Annotation

**Authors:** Virginia Aglietti, Theodoros Damoulas, Javier González
**Venue:** NeurIPS 2021
**Note:** PDF not automatically retrieved (NeurIPS proceedings URL unavailable); annotation from paper knowledge.

---

## High-Level Framing
- **Problem:** Extends Causal Bayesian Optimization (CBO, Aglietti 2020) to settings where the structural causal model (SCM) is *dynamic*: the causal graph and/or its parameters change between time steps $t = 1, 2, \ldots, T$.
- **Motivation:** In static CBO, the causal graph is fixed and interventional distributions $p(Y \mid do(X=x))$ are stationary. In many real settings (e.g., health interventions, sequential policy decisions), mechanisms evolve — both the graph structure and the conditional distributions can change.
- **Contribution:** A sequential decision-making algorithm that tracks the dynamic causal structure, updates both the causal model and the GP surrogate over time, and selects interventions optimally under the time-varying mechanism.

---

## Dynamic Causal Structure
- At each time step $t$, the data-generating process is described by an SCM $\mathcal{M}_t$ with graph $\mathcal{G}_t$.
- The dynamics are modeled as a transition $\mathcal{M}_t \to \mathcal{M}_{t+1}$, where mechanism parameters (edge weights, noise variances) evolve — possibly continuously (drift) or discontinuously (structural shift).
- **Key contrast with standard dynamic GP bandits:** Standard dynamic bandits model $f_t(x)$ as a time-varying function without causal structure. DCBO explicitly models *which mechanisms change* (which edges of the graph are affected), enabling more targeted updates.

---

## Algorithm
1. At time $t$, maintain a posterior over the current SCM $\mathcal{M}_t$, encoded as a set of GP surrogates over interventional distributions for each node.
2. Select intervention $x_t$ via a causal acquisition function (e.g., causal EI) that uses the $do$-calculus to propagate the GP posterior through the causal graph.
3. Observe outcome $y_t$, update the GP surrogates for the affected causal mechanisms.
4. Propagate beliefs about mechanism dynamics to $t+1$.

**What DCBO does NOT do:** It does not use a "product kernel combining a stationary base and a change-point indicator" — that would be a static GP engineering approach. DCBO models mechanism dynamics as an explicit probabilistic process over the causal graph, not as a kernel composition.

---

## Key Theoretical Properties
- Under mild stationarity assumptions on the mechanism dynamics, DCBO achieves sublinear *dynamic regret* (regret compared to the time-varying optimum $x_t^*$ at each step).
- The dynamic regret bound depends on the *variation budget* $V_T = \sum_t \|\mathcal{M}_t - \mathcal{M}_{t-1}\|$ — how much the mechanism changes in total.
- For $V_T = 0$ (static case), the bound reduces to the static CBO bound.

---

## Distinction: Drift vs. Abrupt Structural Replacement
- DCBO is designed for *continuous drift* in mechanism parameters: gradual changes in edge weights, noise levels, or functional forms.
- It handles *abrupt change-points* only indirectly — a large single-step change in $\mathcal{M}_t$ inflates the variation budget $V_T$ at that step, degrading the regret guarantee for the entire horizon.
- For the thesis's 2022 Austin election change-point (discrete replacement of the decision-making council), DCBO's drift model is a partial fit at best: it would detect the regime change eventually but has no mechanism for a clean structural restart.

---

## Connections to This Course / Thesis
- **Causal graph dynamics:** The 2022 Austin municipal election represents a structural change in which council members (and thus which decision criteria) determine zoning outcomes. This is exactly a change in $\mathcal{G}_t$ — not just parameter drift but a change in the intervention-outcome mechanism.
- **Correct citation context:** DCBO is appropriately cited for the idea of tracking causal mechanism dynamics. It is NOT appropriate to cite it for specific GP kernel constructions (e.g., change-point kernels), which are a separate literature (Garnett, Osborne & Roberts 2010; Wilson & Adams 2013).
- **Next steps framing:** The thesis's change-point detection plan (fitting separate GP models pre/post 2022) is a simpler, data-driven alternative to DCBO's fully probabilistic mechanism-tracking. DCBO would be the more principled approach if the change-point location were unknown.
