# Technical Glossary & Notation

Master list of technical terms, mathematical symbols, and their grounding in the seminar literature.

## 1. Notation Table

| Symbol | Definition | Context |
| :--- | :--- | :--- |
| $G$ | A Directed Acyclic Graph (DAG) representing causal structure. | SCMs |
| $U$ | Exogenous (Ex) variables; background factors common to the system. | Structural equations |
| $V$ | Endogenous (En) variables; variables determined within the system. | Structural equations |
| $P(Y \mid do(X=x))$ | The interventional distribution of $Y$ given an intervention on $X$. | Intervention (Ch 3) |
| $Y_x(u)$ | The potential outcome of $Y$ had $X$ been $x$, for a specific unit $u$. | Counterfactuals (Ch 4) |
| $e \in \mathcal{E}$ | An environment or experimental condition. | Arjovsky/Peters |
| $\Phi(X)$ | A data representation or feature extractor. | IRM / Rep Learning |

---

## 2. Terminology

**Counterfactual**
A "what-if" statement about a specific unit $u$ and an alternative scenario that did not occur. Represented as $Y_x(u)$.
*   *Grounding:* [Pearl et al., 2016, Ch 4]

**Distribution Shift**
A change in the joint distribution $P(X, Y)$ across environments, often caused by the shift of non-invariant (spurious) features.
*   *Grounding:* [Krueger et al., 2021, p. 1]

**Environment**
A specific experimental setting, dataset partition, or context where the data-generating process may vary.
*   *Grounding:* [Arjovsky et al., 2019, p. 1]

**Invariance**
The property of a relationship $P(Y|X)$ being constant across multiple **Environments**. Causal relationships are fundamentally invariant.
*   *Grounding:* [Peters et al., 2016, p. 2 / Peters et al., 2018, p. 10]

**Invariant Risk Minimization (IRM)**
A learning paradigm that finds a representation $\Phi$ such that the same linear classifier $w$ is optimal across all training environments.
*   *Grounding:* [Arjovsky et al., 2019, p. 2]

**Modularity**
The "Independent Causal Mechanism" (ICM) principle; changing one causal mechanism does not change others.
*   *Grounding:* [Peters et al., 2018, p. 107]

**Out-of-Distribution (OOD) Generalization**
The ability of a model to perform well on **Test Environments** that were not seen during training.
*   *Grounding:* [Arjovsky et al., 2019, p. 2]

**Partitioning**
The procedural act of dividing the dataset into discrete **Environments** based on metadata (e.g., time, location). 
*   *Grounding:* [Procedural Term / Arjovsky et al., 2019, p. 1]

**Spurious Correlation**
An association between $X$ and $Y$ that arises from shared confounders or selection bias rather than direct causation. These shift across environments.
*   *Grounding:* [Arjovsky et al., 2019, p. 1]

**Structural Causal Model (SCM)**
A set of structural equations $V_i = f_i(PA_i, U_i)$ and a distribution $P(U)$ over exogenous variables.
*   *Grounding:* [Peters et al., 2018, p. 84]
