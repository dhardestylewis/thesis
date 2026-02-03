# Reference Atlas: Grounding & Mapping

Central repository for bibliographic tracking, technical concept mapping, and project applications.

## 1. Bibliographic Index
Mapping of foundational texts to core project claims.

| Source | Key Concepts | Verified Pages |
| :--- | :--- | :--- |
| **Arjovsky (2019)** | IRM, Spurious correlations, OOD | p. 1-6 |
| **Peters (2016)** | Invariant Prediction, Direct Causes | p. 1-4, 48 |
| **Schölkopf (2021)** | Representation Learning, Modularity | p. 1-29 |
| **Wu & Blei (2025b)** | Bayesian Invariance, BIP | p. 1-2 |
| **Pearl (2016)** | Interventions, Counterfactuals | (Processing Ch 3--4) |

---

## 2. Concept Library
Detailed technical distillation of seminar themes.

### Invariant Risk Minimization (IRM)
Find representation $\Phi$ such that the same linear classifier $w$ is optimal across all training environments.
*   **Equation:** $\min_{\Phi} \sum_{e \in \mathcal{E}_{train}} R^e(w \circ \Phi) + \lambda \cdot \| \nabla_{w|w=1.0} R^e(w \cdot \Phi) \|^2$

### Causal Representation Learning
Inverting the generative process to recover latent causal variables from high-dimensional observations.

### Invariant Prediction (ICP)
Exploiting the invariance of $P(Y|PA(Y))$ across environments to discover the causal set $PA(Y)$.

---

## 3. Project Application Map
Direct connections between seminar theory and the "NIMBYism" project.

*   **IRM Strategy:** Partitioning data into time blocks (e.g., 2007-2018) to learn features that predict resistance consistently across market regimes.
*   **Robustness Argument:** Framing the project as a "Policy Tool" that generalizes to future distributional shifts (Bühlmann 2020).
*   **NLP Application:** Mapping protest letter text to latent causal sentiments (Traffic, Change Aversion) using Representation Learning.

---

## 4. Canonical Examples & DAGs
Representative causal stories used in the literature.

### Simpson's Paradox
*   **Structure:** $X \rightarrow Y \leftarrow Z \rightarrow X$ (Confounding)
*   **Lesson:** Subgroup reversal of overall trends.

### Backdoor Adjustment
*   **Structure:** $X \leftarrow Z \rightarrow Y$ and $X \rightarrow Y$.
*   **Lesson:** Isulating $P(Y \mid do(X))$ by blocking non-causal paths.
