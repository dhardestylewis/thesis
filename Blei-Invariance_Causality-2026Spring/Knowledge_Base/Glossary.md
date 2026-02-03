# Glossary of Terms

## Causal Mechanism & Structure

**Structural Causal Model (SCM)**
A set of assignments describing the causal mechanism generating data, typically $X_j := f_j(PA_j, N_j)$, where $PA_j$ are parents and $N_j$ is noise. The causal mechanism is assumed invariant, while parent distributions may shift.
*   *Grounding:* [Peters et al., 2018, Sec 3.1, p. 84 / Schölkopf et al., 2021, Sec II-C, p. 5]

**Causal Parents ($PA(Y)$)**
The direct causes of a target variable $Y$ in an SCM. Interventions on other variables leave the mechanism $P(Y|PA(Y))$ unchanged.
*   *Grounding:* [Peters et al., 2016, p. 2 / Peters et al., 2018, p. 5115]

**Target Variable ($Y$)**
The outcome variable of interest in a prediction task. In this project, $Y$ denotes **Resistance**.
*   *Grounding:* [Peters et al., 2018, p. 2555 / Peters et al., 2016, p. 1]

**Resistance**
The target variable ($Y$) in this project, representing the degree of neighborhood opposition to housing developments (NIMBYism). It is the outcome predicted by the model, where the objective is to isolate causal drivers that remain stable across shifting environments.
*   *Grounding:* [Project Specific / Grounded in Target Variable $Y$ (Peters et al., 2018)]

**Market Regimes**
Stylistic term in this project for the distinct economic and social periods (e.g., 2018 bull market vs 2023 rate hikes) that constitute different **Environments**.
*   *Grounding:* [Project Specific / Contextual Environments]

**Intervention**
An external action that modifies the mechanism generating a subset of variables in an SCM. In Invariant Prediction, we assume interventions occur on variables *other than* the target $Y$.
*   *Grounding:* [Peters et al., 2016, p. 1 / Peters et al., 2018, p. 3175]

**Modularity**
The principle that the causal mechanism $P(Y | PA(Y))$ is local and independent of the mechanisms generating other variables. It implies that changing one mechanism (a **Mechanism Change**) does not affect others. This is the theoretical foundation for **Invariance**.
*   *Grounding:* [Peters et al., 2016, p. 2 / Schölkopf et al., 2021, p. 674]

**Autonomy**
A synonym for **Modularity**, emphasizing that a causal mechanism can be changed without affecting others.
*   *Grounding:* [Peters et al., 2016, p. 2 / Aldrich, 1989]

**Independent Causal Mechanisms (ICM)**
The assumption that the world is composed of autonomous modules (mechanisms) that do not inform or influence each other. Learning these is the goal of **Causal Representation Learning**.
*   *Grounding:* [Schölkopf et al., 2021, p. 674 / Peters et al., 2018, Sec 2.1]

## Data & Environments

**Environment ($e \in \mathcal{E}$)**
A distinct setting or context (e.g., location, time, intervention) where data is generated. Environments provide the signal for both **Training** (to identify what is invariant) and **Testing** (to verify generalization across **Distribution Shifts**).
*   *Grounding:* [Arjovsky et al., 2019, p. 1 / Peters et al., 2016, p. 4]

**Expanding Window Cross-Validation**
The methodological strategy of training on a growing history ($\mathcal{E}_{tr}$) and testing on all future horizons ($\mathcal{E}_{te}$) to simulate real-world **Out-of-Distribution** shifts.
*   *Grounding:* [Project Specific / Standard Time-Series Methodology]

**Training Environments ($\mathcal{E}_{tr} \subset \mathcal{E}$)**
The subset of environments observed during training. The objective is to find relationships invariant across these.
*   *Grounding:* [Arjovsky et al., 2019, p. 2 / Peters et al., 2018 style]

**Test Environments ($\mathcal{E}_{te}$)**
Unseen environments used to validate **Out-of-Distribution (OOD)** generalization. In this project, 2019--2025 data.
*   *Grounding:* [Arjovsky et al., 2019, p. 2]

**Environment-specific Confounder**
A variable $Z$ that causes both predictors $X$ and target $Y$ but whose influence or distribution varies across environments. Standard models overfit to $Z$, creating **Spurious Associations**.
*   *Grounding:* [Arjovsky et al., 2019, p. 1 "confounding factors" / Peters et al., 2016, p. 2]

**Selection Bias**
A bias where the data collection process itself depends on the variables $X$ or $Y$, often varying by environment (e.g. which neighborhoods are sampled).
*   *Grounding:* [Arjovsky et al., 2019, p. 1 / Peters et al., 2018, p. 104]

## Statistical Properties

**Invariance**
The property of a conditional distribution (e.g. $P(Y|X)$) remaining constant across different environments.
*   *Formal:* $P(Y|X, e) = P(Y|X)$ for all $e \in \mathcal{E}$.
*   *Grounding:* [Peters et al., 2016, p. 2 / Schölkopf et al., 2021, p. 694]

**Robustness**
The property of a model maintaining its **Predictive Performance** across all environments $\mathcal{E}_{all}$. In this project, it is demonstrated when **AUC** is invariant under **Distribution Shift**.
*   *Grounding:* [Arjovsky et al., 2019, p. 1-2 / Bühlmann, 2020, p. 1]

**Spurious Association**
An association between $X$ and $Y$ that arises from shared confounders or selection bias rather than direct causation. These associations shift when environments change.
*   *Grounding:* [Arjovsky et al., 2019, p. 1 / Peters et al., 2018, p. 504 (referred to as "spurious correlation")]

**Out-of-Distribution (OOD) Generalization**
The ability of a model to maintain performance under **Distribution Shift**—specifically when the test data $P(X, Y | e_{te})$ follows a different distribution than the training data $\mathcal{E}_{tr}$.
*   *Grounding:* [Arjovsky et al., 2019, p. 2 / Krueger et al., 2021 / Schölkopf et al., 2021, p. 29]

**Distribution Shift**
The change in the joint distribution $P(X, Y)$ across different environments. Causal models are designed to be robust to shifts in the marginal distribution $P(X)$ while maintaining an invariant conditional $P(Y|X)$.
*   *Grounding:* [Schölkopf et al., 2021, p. 29 / Krueger et al., 2021, p. 1]

## Modeling & Learning

**Invariant Risk Minimization (IRM)**
A learning paradigm that seeks a representation $\Phi(X)$ such that the optimal classifier $w$ is simultaneously optimal across all training environments.
*   *Grounding:* [Arjovsky et al., 2019, Sec. 3, p. 2]

**Representation ($\Phi(X)$)**
A mapping from input space to a feature space. IRM seeks a representation that discards spurious (non-invariant) features.
*   *Grounding:* [Arjovsky et al., 2019, p. 2 / Schölkopf et al., 2021, p. 15]

**Stable Predictor**
A model $f \circ \Phi$ that relies only on invariant/causal features to make predictions.
*   *Grounding:* [Arjovsky et al., 2019, p. 1 / Schölkopf et al., 2021, p. 101]

**Stable Prediction**
A prediction that performs consistently across different environments because it relies on invariant (causal) features. Grounded in the requirement that causal models show "invariance in their predictive accuracy" across regimes.
*   *Grounding:* [Arjovsky et al., 2019, p. 1 / Peters et al., 2016, p. 1]

**Predictive Stability**
The primary research objective: the **invariance** of model performance across environment shifts. This is measured by the **Consistency of AUC** (absence of a performance gap). Failure to maintain this stability is referred to in the literature as a model being "potentially very wrong" under shift.
*   *Grounding:* [Arjovsky et al., 2019, p. 1 / Peters et al., 2016, p. 1, 2]

**Predictive Performance**
The quality of a model's predictions as quantified by a specific metric. In this project, **AUC** (Area Under the ROC Curve) is the chosen metric for performance. Stability is demonstrated when this performance metric remains invariant.
*   *Grounding:* [Standard ML Metric]

**Invariant Causal Drivers**
Stylistic term used in this project for the set of invariant features $X_S$ that directly cause the target variable $Y$.
*   *Grounding:* [Project Specific / Grounded in Causal Features]

**Causal Features**
Formal terminology for the variables $X_S$ involved in the invariant causal mechanism $P(Y|X_S)$.
*   *Grounding:* [Arjovsky et al., 2019, p. 1 / Schölkopf et al., 2021, p. 1250]

**Stable Structural Properties**
The underlying causal mechanisms $P(Y | PA(Y))$ that remain invariant across environments. Finding these is the core objective of Invariant Prediction.
*   *Grounding:* [Arjovsky et al., 2019, p. 1 "stable properties" / Peters et al., 2016, p. 2 "modularity"]

**Causal Identification**
The process of determining whether a specific variable or mechanism is a true direct cause of the target variable, distinguishing it from spurious associations.
*   *Grounding:* [Peters et al., 2016, Title / Sec 2]

**Invariant Causal Mechanism**
The conditional distribution $P(Y | PA(Y))$ that remains constant across environments. In Invariant Prediction, we assume that as long as we do not intervene on $Y$, this mechanism is fixed.
*   *Grounding:* [Peters et al., 2016, p. 1-2 / Peters et al., 2018, p. 34]

**Mechanism Change**
A violation of the invariance assumption where the generating process $P(Y | PA(Y))$ itself is modified, either through a direct intervention on the target variable $Y$ or a fundamental structural shift in the system.
*   *Grounding:* [Peters et al., 2016, p. 2 / Arjovsky et al., 2019, p. 2]

**Separate Predictor**
The analytical workflow of **identifying** and discriminating between invariant causal drivers and spurious associations.
*   *Grounding:* [Project Specific / Derived from Peters 2016 Identification goal]

**Empirical Risk Minimization (ERM)**
The standard machine learning baseline which minimizes the average loss over the training data. ERM assumes that training and test distributions are identical ($P_{tr} = P_{te}$) and often fails under **Distribution Shift** by absorbing **Spurious Associations**.
*   *Grounding:* [Arjovsky et al., 2019, Sec. 2.1 / Vapnik, 1992]

**Risk ($R^e(f)$)**
The expected value of the **Loss Function** for a predictor $f$ on environment $e$. In IRM, we seek to minimize risk simultaneously across all training environments.
*   *Grounding:* [Arjovsky et al., 2019, p. 4]

**Empirical Risk**
The measurable estimate of **Risk** calculated as the average loss over a finite dataset $\mathcal{D}_e$.
*   *Grounding:* [Arjovsky et al., 2019, Sec. 2.1]

**Loss Function ($\ell$)**
A function measuring the discrepancy between a prediction and the true label. In this project, the logistic loss (for binary outcome **Resistance**) serves as the basis for the **Empirical Risk**.
*   *Grounding:* [Arjovsky et al., 2019, p. 4]

**Gradient Penalty**
The core regularization term in **IRM**: $\|\nabla_{w|w=1.0} R^e(w \cdot \Phi)\|^2$. It acts as a selection pressure by **penalizing** representations where the optimal classifier would vary across environments, thus "filtering" for invariance.
*   *Grounding:* [Arjovsky et al., 2019, Eq. 5 / Wu et al. 2025b p. 3]

**Reliability**
The property of a model generating stable and trustworthy predictions across varying environments. In causal learning, reliability is achieved when the model relies on the **Invariant Causal Mechanism**, making it robust to context shifts.
*   *Grounding:* [Schölkopf et al., 2021, p. 29]

**Invariance Filter**
The technical process where the invariance criterion (e.g., $P(Y|X, e) = P(Y|X)$) acts as a selection pressure. It "filters out" features with non-stable relationships by assigning them zero weight or excluding them from the causal set, ensuring only **Stable Structural Properties** remain.
*   *Grounding:* [Methodological Metaphor / Arjovsky et al. 2019 "discarded" / Peters 2016 "model selection"]
