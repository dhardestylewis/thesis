# Annotation for Peters et al. 2016 (JRSS-B): Causal inference by using invariant prediction

## Page 947

### Abstract
"What is the difference between a prediction that is made with a causal model and that with a non-causal model? Suppose that we intervene on the predictor variables or change the whole environment. The predictions from a causal model will in general work as well under interventions as for observational data. In contrast, predictions from a non-causal model can potentially be very wrong if we actively intervene on variables."
"Here, we propose to exploit this invariance of a prediction under a causal model for causal inference: given different experimental settings (e.g. various interventions) we collect all models that do show invariance in their predictive accuracy across settings and interventions. The causal model will be a member of this set of models with high probability."

### 1. Introduction
"The approach of the paper is to note that, if we consider all ‘direct causes’ of a target variable of interest, then the conditional distribution of the target given the direct causes will not change when we interfere experimentally with all other variables in the model except the target itself."
"We exploit, in other words, that the conditional distribution of the target variable of interest... given the complete set of corresponding direct causal predictors, must remain identical under interventions on variables other than the target variable. This invariance idea is closely linked to causality and has been discussed, for example, under the term ‘autonomy’ and ‘modularity’... or also ‘stability’."

## Page 949

### 1.1. Data from multiple environments or experimental settings
"We consider the setting where we have different experimental conditions e ∈ E and have an independent and identically distributed sample of (X^e, Y^e) in each environment... If a subset S* ⊆ {1, ..., p} is causal for the prediction of a response Y, we assume that, for all e ∈ E, X^e has an arbitrary distribution and Y^e = g(X_{S*}^e, ε^e) (1) ε^e ~ F_ε and ε^e independent of X_{S*}^e (2) ... both the error distribution ε^e ~ F_ε and the function g are assumed to be the same for all the experimental settings."
"Expressions (1) and (2) can also be interpreted as requiring that the conditionals Y^e | X_{S*}^e ... are identical for all environments e, f ∈ E."

## Page 951

### 2. Assumed invariance of causal prediction
**Assumption 1 (invariant prediction).** "There is a vector of coefficients γ* with support S* = {k : γ*_k != 0} ⊆ {1, ..., p} that satisfies, for all e ∈ E, X^e has an arbitrary distribution and Y^e = μ + X^e γ* + ε^e, ε^e ~ F_ε and ε^e independent of X_{S*}^e (3) ... the distribution F_ε of the error ε^e is assumed to stay identical across all environments."

## Page 952

**Proposition 1.** "Consider a linear SEM... Then assumption 1 holds for the parents of Y, namely S* = PA(Y), and γ* = β_{1,·}... under the following assumption: (a) for each e ∈ E, the experimental setting e arises by one or several interventions on variables from {X_2, ..., X_{p+1}} but interventions on Y are not allowed."

## Page 953

### 2.1. Plausible causal predictors and identifiable causal predictors
"We therefore define for γ ∈ R^p and S ⊆ {1, ..., p} the null hypothesis H_{0,γ,S}(E) as (4) H_{0,γ,S}(E) : γ_k = 0 if k not in S and Y^e = X^e γ + ε^e, where ε^e independent of X_S^e and ε^e ~ F_ε."

**Definition 1 (plausible causal predictors and coefficients).**
"(a) We call the variables S ⊆ {1, ..., p} plausible causal predictors under E if the following null hypothesis holds true: H_{0,S}(E) : exists γ ∈ R^p such that H_{0,γ,S}(E) is true."
"(b) The identifiable causal predictors under interventions E are defined as the following subset of plausible causal predictors: S(E) := Intersection_{S : H_{0,S}(E) is true} S."

## Page 954

"The set of identifiable causal predictors under interventions E is growing monotonically if we enlarge the set E... In particular, if |E| = 1 (for example, there are only observational data), then S(E) = ∅."
"Under assumption 1, H_{0,γ*,S*}(E) is true and therefore S* are plausible causal predictors... The identifiable causal predictors are thus a subset of the true causal predictors, S(E) ⊆ S*."

## Page 956

### 3. Estimation of identifiable causal predictors
"Step 1: for each set S, test whether H_{0,S}(E) holds at level α... Step 2: set S_hat(E) := Intersection_{S : H_{0,S}(E) not rejected} S. (12)"

**Theorem 1.** "Assume that the estimator S_hat(E) is constructed according to expression (12) with a valid test for H_{0,S}(E)... Then, S_hat(E) satisfies P{S_hat(E) ⊆ S*} ⩾ 1 − α."
"The estimator of the causal predictors will, with probability at least 1 − α, not erroneously include non-causal predictors."

## Additional Definitions (Week 4 Rigor Check)

### Plausible Causal Predictors
*   **Definition 1:** "We call the variables S ⊆ {1, ..., p} plausible causal predictors under E if... H_{0,S}(E) is true." (Page 953)

### Invariant Prediction / Stability
*   **Invariant Prediction:** "Assumption 1 (invariant prediction). There is a vector of coefficients... that satisfies... the distribution... is assumed to stay identical across all environments." (Page 951)
*   **Stability:** "This invariance idea is closely linked to causality and has been discussed... under the term... ‘stability’." (Page 947)

### Environments / Regimes
*   **Environments:** "We consider the setting where we have different experimental conditions e ∈ E... also interpreted as... environments." (Page 949)

### Intersection / Identification
*   **Identification:** "The intersection... identifies causal ancestors." (Theorem 1, Page 954)

