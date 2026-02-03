# Reference Atlas: Grounding & Mapping

Central repository for bibliographic tracking, technical concept mapping, and project applications.

## 1. Bibliographic Index: Grounding & Annotations
*Mapping foundational texts to core project claims.*

### Primary Methodological Sources

#### [Arjovsky et al. (2019) Invariant Risk Minimization](file:///c:/Users/dhl/data/thesis/thesis/Blei-Invariance_Causality-2026Spring/References/Arjovsky_2019_IRM.pdf)
- **Problem Statement:** Data marred by selection biases and confounding factors (p. 1).
- **Spurious vs. Stable:** Spurious correlations defined as non-stable properties related to context (p. 1-2).
- **IRM Principle:** Find representation $\Phi$ such that the optimal classifier is invariant across environments (Sec. 3, p. 2).
- **Generalization:** Focus on Out-of-Distribution (OOD) performance (p. 2).

#### [Pearl et al. (2016) Causal Inference in Statistics: A Primer](file:///c:/Users/dhl/data/thesis/thesis/Blei-Invariance_Causality-2026Spring/References/Pearl_2016_Primer.pdf)
- **Intervention ($do(x)$):** "Surgery" on the graph; changes the world/system mechanics.
- **Conditioning ($X=x$):** Filtering data; changes perception/probability.
- **RCT:** Gold standard for causal discovery; eliminates confounding by randomization.
- **Backdoor/Front-Door:** Graphical criteria for identification from observational data (Ch 3.3-3.4).
- **Counterfactuals ($Y_x(u)$):** Computed via Abduction-Action-Prediction (Ch 4).

#### [Peters et al. (2016) Invariant Prediction](file:///c:/Users/dhl/data/thesis/thesis/Blei-Invariance_Causality-2026Spring/References/Peters_2016_Invariant_Prediction.pdf)
- **Main Idea:** Conditional distribution $P(Y|PA(Y))$ is invariant across interventions on everything except $Y$ (p. 2).
- **Direct Causes:** Grounding for "Causal Drivers/Parents" (p. 2, 48).
- **Environments:** Defined as different experimental conditions $e \in \mathcal{E}$ (p. 4).
- **Autonomy:** Theoretical link between invariance and modularity (p. 2).

#### [Schölkopf et al. (2021) Causal Representation Learning](file:///c:/Users/dhl/data/thesis/thesis/Blei-Invariance_Causality-2026Spring/References/Schoelkopf_2021_Causal_Rep_Learning.pdf)
- **Representation $\Phi$:** Goal of learning causal variables from pixels/high-dimensional data (p. 15, 957).
- **Modularity:** ICM principle for independent causal mechanisms (p. 674).
- **Generalization:** Distinction between standard and OOD generalization (p. 29).

### Terminology Source Map

| Term | Primary Citation | Section/Page | Notes |
| :--- | :--- | :--- | :--- |
| **Invariance** | Peters et al. (2016) | Assumption 1, p. 4 | $P(Y\|X, e) = P(Y\|X)$. |
| **Spurious Association** | Arjovsky et al. (2019) | Intro, p. 1 | Non-invariant "cheats." |
| **Environment** | Arjovsky et al. (2019) | Intro, p. 1 | Basis for partitioning data. |
| **SCM** | Peters et al. (2018) | Sec 3.1, p. 84 | Structural Causal Model. |
| **OOD Generalization** | Krueger et al. (2021) | Intro, p. 1 | Performance under shift. |
| **Stable Prediction** | Peters (2016) | p. 1, 16 | Constant predictive accuracy. |
| **Robustness** | Bühlmann (2020) | p. 1 | Performance across environments. |
| **Separate Predictor** | Peters (2016) | p. 1 | Model selection logic. |
| **Distribution Shift** | [Krueger et al., 2021, p. 1] | Generalization challenge across environments. |
| **Invariant Causal Mechanism** | [Peters et al., 2018, p. 107] | Stable underlying process $P(Y|X_{pa(Y)})$. |
| **Gradient Norm Penalty** | [Arjovsky et al., 2019, p. 6] | Regularization for invariance. |
| **Fixed Dummy Classifier** | [Arjovsky et al., 2019, p. 6] | Baseline for evaluating feature invariance. |
| **Future Horizons** | Project Term | OOD evaluation windows (2019--2025). |

---

## 2. Concept Library (Encyclopedia)
*Detailed technical distillation of seminar themes.*

### Intervention vs. Conditioning
**Origin:** Pearl (2016) Ch 3.1
**Concept:**
*   **Intervention ($do(X=x)$):** Fixing a variable's value and changing the system mechanics. "Surgery" on the graph (removing incoming edges to $X$). Changes the *world*.
*   **Conditioning ($X=x$):** Narrowing focus to a subset of cases where $X=x$. Changes *perception* (probability/belief), not the world.
**Relevance:** Distinguishing policy changes (rezoning) from mere observation of existing zones.

### Randomized Controlled Experiment (RCT)
**Origin:** Pearl (2016) Ch 3.1
**Concept:** The "gold standard" where all influencing factors are static or random except the treatment.
**VS Observational Study:** Where we merely record data; difficult to untangle causal from correlative due to confounding.

### Invariant Risk Minimization (IRM)
**Origin:** Arjovsky et al. (2019)
**Concept:** A learning paradigm that estimates a data representation $\Phi(X)$ such that the optimal linear classifier $w$ on top of this representation is optimal across all **training environments** $\mathcal{E}_{train}$.
**Key Equation:** $\min_{\Phi, w} \sum_{e \in \mathcal{E}_{train}} R^e(w \circ \Phi) + \lambda \cdot \| \nabla_{w|w=1.0} R^e(w \cdot \Phi) \|^2$
**Relevance:** Finding features of housing developments that predict opposition consistently, regardless of neighborhood wealth or year.

### Causal Representation Learning
**Origin:** Schölkopf et al. (2021)
**Concept:** Learning high-level causal variables $S_1, ..., S_n$ from low-level high-dimensional observations $X$ (e.g., images, text). Goal: invert generation to recover latent causal structure.
**Relevance:** Mapping unstructured protest text to causal concepts ("traffic concern", "density fear") rather than raw tokens.

### Invariant Prediction
**Origin:** Peters et al. (2016)
**Concept:** Finding subsets of features $S$ such that $Y \perp E | X_S$. The causal mechanism $P(Y|PA(Y))$ is invariant.
**Relevance:** If "homeowner age" is a causal driver, its effect on opposition should be the same in 2008 and 2024.

### The Adjustment Formula
**Origin:** Pearl (2016) Eq 3.5
**Concept:** Computing the effect of an intervention $P(Y|do(x))$ using only observational quantities by adjusting for parents $Z$ of $X$.
**Equation:** $P(Y=y|do(X=x)) = \sum_z P(Y=y|X=x, Z=z)P(Z=z)$
**Derivation:** Relies on the invariance of $P(Y|x,z)$ and $P(z)$ under graph surgery.

### Average Causal Effect (ACE)
**Origin:** Pearl (2016) Eq 3.1
**Concept:** The difference in outcomes between two interventions.
**Equation:** $ACE = P(Y=1|do(X=1)) - P(Y=1|do(X=0))$
**Relevance:** The metric we likely want to estimate when comparing policy interventions (e.g., Upzoning vs No-Upzoning).
**Equation:** $ACE = P(Y=1|do(X=1)) - P(Y=1|do(X=0))$
**Relevance:** The metric we likely want to estimate when comparing policy interventions (e.g., Upzoning vs No-Upzoning).

### The Backdoor Criterion
**Origin:** Pearl (2016) Def 3.3.1
**Definition:** A set $Z$ satisfies the Backdoor Criterion relative to $(X, Y)$ if:
1.  No node in $Z$ is a descendant of $X$.
2.  $Z$ blocks every path between $X$ and $Y$ that contains an arrow into $X$ (backdoor paths).
**Implication:** If $Z$ satisfies this, we can use the Adjustment Formula.
**Logic:** Block spurious paths; leave directed paths unperturbed; create no new spurious paths.

**Logic:** Block spurious paths; leave directed paths unperturbed; create no new spurious paths.

### The Front-Door Criterion
**Origin:** Pearl (2016) Def 3.4.1
**Definition:** A set $Z$ allows identifying causal effects even if the path $X \leftarrow U \rightarrow Y$ (backdoor) is open / unobserved, provided:
1.  $Z$ intercepts all directed paths from $X$ to $Y$ (Mechanism).
2.  No backdoor path exists from $X$ to $Z$ (Unconfoundedness of Treatment-Mediator).
3.  All backdoor paths from $Z$ to $Y$ are blocked by $X$ (Unconfoundedness of Mediator-Outcome).
**Relevance:** Useful when we cannot calculate the Backdoor adjustment due to hidden confounders.

**Relevance:** Useful when we cannot calculate the Backdoor adjustment due to hidden confounders.

### The Counterfactual Algorithm
**Origin:** Pearl (2016) Thm 4.2.1
**Process (Abduction-Action-Prediction):**
1.  **Abduction:** Use evidence $E=e$ to update probability $P(U)$ of exogenous variables.
2.  **Action:** Replace equations for $X$ with $X=x$ (Model $M_x$).
3.  **Prediction:** Compute target $Y$ in $M_x$ using updated $U$.
**Relevance:** Estimating individual-level effects ("Was it the drug that cured *this specific patient*?") versus population averages.

### Empirical Bayes
**Origin:** Efron / Robbins
**Concept:** Using the data itself to estimate the prior distribution.
**Relevance:** Modeling the "population" of rezoning cases to set priors for new cases based on aggregate history.

---

## 3. Project Application Map
*Direct connections between seminar theory and "NIMBYism" project mechanics.*

### 1. Core Framework: Invariant Risk Minimization
- **Arjovsky et al. (2019):** Provides mathematical justification for minimizing loss across environments.
- **Project Strategy:** Define loss $\mathcal{L}_{IRM}$ over time blocks (2007-2018) to learn stable features.

### 2. Validation: Causal Inference using Invariant Prediction (ICP)
- **Peters et al. (2016):** Establishes "Invariance" as a proxy for "Causality".
- **Project Validation:** Check if selected features have stable coefficients across spatial environments (East vs. West Austin).

### 3. Importance: Robustness & Stability
- **Bühlmann (2020):** Connects invariance to "Robustness".
- **Project Justification:** Framing the project as creating a "Robust Policy Tool" that withstands market cycle shifts.

### 4. Method: Representation Learning
- **Schölkopf et al. (2021):** Extracting causal variables from high-dimensional inputs.
- **Protest Letters:** Treating letters as observations $X$ generated by latent causal sentiments $S$.

### 5. Method: Empirical Bayes
- **Efron (2012):** Large-Scale Inference.
- **Population Modeling:** Treating 10,000+ rezoning cases as parameters to estimate population priors.

---

## 4. Canonical Examples & DAGs
*Representative causal stories.*

### Simpson's Paradox
*   **Structure:** $X \rightarrow Y \leftarrow Z \rightarrow X$ (Confounding)
*   **Lesson:** Subgroup reversal of overall trends.

### Backdoor Adjustment
*   **Structure:** $X \leftarrow Z \rightarrow Y$ and $X \rightarrow Y$.
*   **Lesson:** Isolating $P(Y \mid do(X))$ by blocking non-causal paths.

### Ice Cream & Crime (Hot Weather Confounder)
*   **Origin:** Pearl (2016) Ch 3.1
*   **Structure:** $X$ (Ice Cream) $\leftarrow Z$ (Temperature) $\rightarrow Y$ (Crime).
*   **Lesson:** Correlation is not causation. Intervening on $X$ (banning ice cream) does not affect $Y$ because it does not change $Z$.

### Drug Study (Simpson's Paradox)
*   **Origin:** Pearl (2016) Fig 3.3
*   **Structure:** $X$ (Drug) $\leftarrow Z$ (Gender) $\rightarrow Y$ (Recovery) and $X \rightarrow Y$.
*   **Analysis:** $Z$ is a confounder. To estimate causal effect of Drug ($X$), we must adjust for Gender ($Z$) using the Adjustment Formula.
