# Index of References: Grounding & Annotations

This index maps the Project's key technical terms and methodologies to original source material in the Reference library.

## Primary Methodological Sources

### [Arjovsky et al. (2019) Invariant Risk Minimization](file:///c:/Users/dhl/data/thesis/thesis/Blei-Invariance_Causality-2026Spring/References/Arjovsky_2019_IRM.pdf)
- **Problem Statement:** Data marred by selection biases and confounding factors (p. 1).
- **Spurious vs. Stable:** Spurious correlations defined as non-stable properties related to context (p. 1-2).
- **IRM Principle:** Find representation $\Phi$ such that the optimal classifier is invariant across environments (Sec. 3, p. 2).
- **Generalization:** Focus on Out-of-Distribution (OOD) performance (p. 2).

### [Peters et al. (2016) Invariant Prediction](file:///c:/Users/dhl/data/thesis/thesis/Blei-Invariance_Causality-2026Spring/References/Peters_2016_Invariant_Prediction.pdf)
- **Main Idea:** Conditional distribution $P(Y|PA(Y))$ is invariant across interventions on everything except $Y$ (p. 2).
- **Direct Causes:** Grounding for "Causal Drivers/Parents" (p. 2, 48).
- **Environments:** Defined as different experimental conditions $e \in \mathcal{E}$ (p. 4).
- **Autonomy:** Theoretical link between invariance and modularity (p. 2).

### [Schölkopf et al. (2021) Causal Representation Learning](file:///c:/Users/dhl/data/thesis/thesis/Blei-Invariance_Causality-2026Spring/References/Schoelkopf_2021_Causal_Rep_Learning.pdf)
- **Representation $\Phi$:** Goal of learning causal variables from pixels/high-dimensional data (p. 15, 957).
- **Modularity:** ICM principle for independent causal mechanisms (p. 674).
- **Generalization:** Distinction between standard and OOD generalization (p. 29).

| Term | Primary Citation | Section/Page | Notes |
| :--- | :--- | :--- | :--- |
| **Invariance** | Peters et al. (2016) | Assumption 1, p. 4 | $P(Y\|X, e) = P(Y\|X)$. |
| **Spurious Association** | Arjovsky et al. (2019) | Intro, p. 1 | Non-invariant "cheats." |
| **Environment** | Arjovsky et al. (2019) | Intro, p. 1 | Basis for partitioning data. |
| **SCM** | Peters et al. (2018) | Sec 3.1, p. 84 | Structural Causal Model. |
| **OOD Generalization** | Krueger et al. (2021) | Intro, p. 1 | Performance under shift. |
| **Stable Prediction** | Peters (2016) | p. 1, 16 | Constant predictive accuracy. |
| **Predictive Stability** | Peters (2016) | p. 1, 2 | The "property" of invariance. |
| **Robustness** | Bühlmann (2020) | p. 1 | Performance across environments. |
| **Predictive Performance** | Arjovsky (2019) | p. 2 | Secondary metric (AUC). |
| **Separate Predictor** | Peters (2016) | p. 1 | Model selection logic. |
| Distribution Shift | [Krueger et al., 2021, p. 1] | Generalization challenge across environments. |
| Invariant Causal Mechanism | [Peters et al., 2018, p. 107] | Stable underlying process $P(Y|X_{pa(Y)})$. |
| Mechanism Change | [Peters et al., 2018, p. 107] | Structural change in the causal process over time. |
| Gradient Norm Penalty | [Arjovsky et al., 2019, p. 6] | Regularization for representation invariance. |
| Fixed Dummy Classifier | [Arjovsky et al., 2019, p. 6] | Baseline for evaluating feature invariance. |
| Feature Selection Vector | [Wu et al., 2025b, p. 2] | Latent indicator $z$ for the invariant set. |
| Stable Predictive Relationship | [Wu et al., 2025b, p. 1] | Target of the BIP posterior inference. |
| Future Horizons | Project Term | OOD evaluation windows (2019--2025). |
| **Stable Properties** | Arjovsky (2019) | p. 1 | Invariant mechanisms. |
| **Identification** | Peters (2016) | p. 1 | Discovery of causal drivers. |
| **ERM** | Arjovsky (2019) | Sec. 2.1 | Standard ML baseline. |
| **Reliability** | Schölkopf (2021) | p. 29 | Trustworthy performance under shift. |

## Annotation Notes
- **"Resistance"** is mapped to the **Target Variable $Y$** in IRM/ICP literature.
- **"Causal Drivers"** is the project's stylistic term for the **Direct Causes** $PA(Y)$ identified via invariant prediction.
