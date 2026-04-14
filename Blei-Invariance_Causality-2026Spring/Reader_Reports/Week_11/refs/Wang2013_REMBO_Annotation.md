# Wang et al. (2013/2016) — Bayesian Optimization in a Billion Dimensions via Random Embeddings (REMBO)
## Full Annotation

**Authors:** Ziyu Wang, Frank Hutter, Masrour Zoghi, David Matheson, Nando de Freitas
**Venue:** JAIR 2016 (extended; arXiv:1301.1942, originally IJCAI 2013)

---

## High-Level Framing
- **Problem:** Standard BO with GP surrogates scales cubically in $n$ and degrades in $d$. In practice, many optimization problems have $D \gg 100$ nominal dimensions but low *effective* dimensionality.
- **Key insight:** If the objective $f$ depends on only $d_e \ll D$ dimensions (the *effective* subspace), a random linear projection $A \in \mathbb{R}^{d_e \times D}$ will, with high probability, map the original optimal $x^*$ to a point $y^*$ in the low-dimensional space such that $f(Ay) \approx f(x^*)$.
- **Algorithm (REMBO):** Run standard BO in a random $d_e$-dimensional subspace, map query points back to $\mathbb{R}^D$ via $x = Ay$, evaluate the true function, update GP in the low-dimensional space.

---

## The Random Embedding
- Draw $A \sim \mathcal{N}(0,1)^{d_e \times D}$ (each entry i.i.d. Gaussian).
- By the Johnson-Lindenstrauss lemma, random projections preserve pairwise distances with high probability.
- **Theorem (Recovery guarantee):** If the effective dimensionality is $d_e$, then for any $x^* \in \mathcal{X}$, with probability 1 over the draw of $A$, there exists $y^* \in \mathbb{R}^{d_e}$ such that $Ay^* = x^*$ (restricted to the effective subspace). The random projection "hits" the effective subspace with probability 1.
- **Key assumption:** The active (effective) subspace is *axis-aligned* or at least low-dimensional. If the function interacts across all $D$ dimensions, the embedding fails.

---

## Practical Considerations
- **Choice of $d_e$:** In practice, set $d_e = 2$ or small. The algorithm is robust to mild overestimates of $d_e$.
- **Constraint handling:** The inverse map $x = Ay$ may land outside the box $[-1,1]^D$. The paper clips $x$ to the boundary, arguing the optimal $x^*$ is likely reachable without clipping.
- **Multiple restarts:** Running REMBO with multiple independent random matrices $A_1, \ldots, A_k$ and aggregating observations reduces the probability of missing $x^*$.
- **Categorical variables:** The embedding idea extends naturally by relaxing categorical variables to continuous ones and rounding.

---

## Empirical Results
- Demonstrated on: (1) synthetic functions up to $10^9$ dimensions; (2) optimization of 47 hyperparameters of the CPLEX MILP solver, achieving state-of-the-art performance.
- REMBO significantly outperforms naive high-dimensional GP-BO and is competitive with SMAC (random forest surrogate) on the CPLEX benchmark.

---

## Limitations
- The effective-dimensionality assumption is strong and may not hold in spatially structured domains where *all* dimensions contribute (e.g., per-neighborhood policy variables with geographic spillovers).
- The random projection is fixed at the start; if the effective subspace is misaligned with the draw of $A$, the algorithm recovers slowly.
- No regret guarantees are provided; the analysis is asymptotic and informal in the original paper.

---

## Connections to This Course / Thesis
- **Effective dimensionality in zoning:** The ARD kernel analysis (Week 11 project update) suggested effective $d_e \approx 3$ for the Austin zoning problem (density, distance, district). REMBO's assumption appears empirically justified.
- **Alternative to TuRBO:** REMBO reduces dimensionality globally; TuRBO reduces it locally via trust regions. They target the same scaling problem from different angles.
- **Causal structure and subspace:** The "active subspace" in causal terms corresponds to the set of variables that are parents of the outcome in the SCM. Causal graph knowledge could replace the random projection with a principled low-dimensional parameterization.
