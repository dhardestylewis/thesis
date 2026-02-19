# Annotation for Bühlmann 2020: Invariance, Causality and Robustness

## Page 404

### Abstract
"We discuss recent work for causal inference and predictive robustness in a unifying way. The key idea relies on a notion of probabilistic invariance or stability: it opens up new insights for formulating causality as a certain risk minimization problem with a corresponding notion of robustness. The invariance itself can be estimated from general heterogeneous or perturbation data which frequently occur with nowadays data collection."
"The novel methodology is potentially useful in many applications, offering more robustness and better “causal-oriented” interpretation than machine learning or estimation in standard regression or classification frameworks."

### Introduction
"Understanding the causal relationships in a system or application of interest is perhaps the most desirable goal in terms of understanding and interpretability."
"One might think that for pure prediction tasks, without any ambition of interpretability, knowing the causes or the causal structure is not important. We will explain here how these problems are related and as a consequence: (i) one can obtain “better” predictions when incorporating causal aspects and (ii) one can infer causal structure from a certain predictive perspective."

"Often though, the data at hand does not come from a (fully) randomized study: the question is now whether one can still infer causal effects and under what kind of assumptions this is possible."
"Since causal inference is very ambitious, these techniques should be thought as “geared towards causality” but not necessarily able to infer the underlying true causal effects."

## Page 405

### 1.1 A Framework Based on Invariance Properties
"We will focus here on a particular framework with corresponding methods which are “geared towards” causal solutions: with stronger assumptions (but less strong than for some competitor methods) they infer causal effects while under more relaxed and perhaps more realistic assumptions, they are still providing solutions for a “diluted form of causality” which are often more meaningful than what is provided by regression or classification techniques."
"The construction of methods relies on exploiting invariance from heterogeneous data. The heterogeneity can be unspecific perturbations and in this sense, the current work adds to the still yet quite small literature on statistics for perturbation data."

### 2. PREDICTING POTENTIAL OUTCOMES, HETEROGENEITY AND WORST-CASE RISK OPTIMIZATION
"Causality deals with a quantitative answer (a prediction) to a “What if I do question” or a “What if I perturb question”."
"The problem is to predict the flowering time of the plant when making single gene interventions... The data is from the observational state of the system only without any interventions. Therefore, this is a problem of predicting a potential outcome which has never been observed in the data."

## Page 406

### 2.2 The Heterogeneous Setting with Different Environments
"We consider data from different observed (known) environments, and we sometimes refer to them also as experimental settings or subpopulations or perturbations... (2.1)"
"Heterogeneity can also occur outside the observed data. Thus, we consider a space of unobserved environments F (2.2) which is typically much larger than the space of observed environments E."

### 2.3 A Prediction Problem and Worst-Case Risk Optimization
"We consider the following prediction problem. Predict Y e given Xe such that the prediction “works well” or is “robust” for all e ∈ F based on data from much fewer environments e ∈ E."
"The terminology “works well” or is “robust” is understood here in the sense of performing well in worst-case scenarios."

"In a linear model setting, this prediction task exhibits a relation to the following worst-case L2 -risk optimization: (2.3) argmin max E ||Y_e - X_e b||^2"
"This problem has an interesting connection to causality."
"The causal parameter or the causal solution is optimizing a certain worst-case risk. This opens the door to think about causality in terms of optimizing a certain (worst-case) risk."

### 3. INVARIANCE OF CONDITIONAL DISTRIBUTIONS
"A key assumption for inferring causality from heterogeneous data as in (2.1) is an invariance assumption. It reads as follows: (A(E)): There exists a subset S* ⊆ {1, ..., p} of the covariate indices (including the empty set) such that L(Y_e | X_S*) is the same for all e ∈ E. That is, when conditioning on the covariates from S* ... the conditional distribution is invariant across all environments from E."

## Page 407

"(A(F)): Analogous but now for the much larger set of environments F."
"In a linear model setting... the resulting regression parameter and error term distribution are the same for all environments e ∈ E."

### 3.1 Invariance and Causality
"To address this at least in part, we consider structural equation models (SEMs): Y ← fY(X_pa(Y), ε_Y) (3.1)"
"The environments or perturbations e change the distributions of Y and X in model (3.1)... (B(E)) The structural equation in (3.1) remains the same, that is, for all e ∈ E... ε_Y^e independent of X_pa(Y)^e, ε_Y^e has the same distribution as ε_Y."
"We note that the distributions of X^e are allowed to change."

**Proposition 3.1.** "Assume a partial structural equation model as in (3.1). Consider the set of environments F such that (B(F)) holds. Then, the set of causal variables S_causal = pa(Y) satisfies the invariance assumption with respect to F, that is (AS_causal(F)) holds."
"The proof is trivial... Proposition 3.1 says that causal variables lead to invariance: this has been known since a long time, dating back to Haavelmo (1943)."

## Page 408

### 3.2 Invariant Causal Prediction
"Roughly speaking, Haavelmo (1943) already realized that causal variables implies Invariance. The reverse relation (3.3) causal structures implied by Invariance has not been considered until recently (Peters, Bühlmann and Meinshausen, 2016)."
"This might be due to the fact that with nowadays large-scale data, it is much easier to infer invariance from data and thus, the implication from invariance to causal structures becomes much more interesting and useful."

"The starting point is to perform a statistical test whether a subset of covariates S satisfies the invariance assumption for the observed environments in E."
"To address the identifiability issue, we intersect all subsets of covariates S which lead to invariance, that is, S_hat(E) = intersection of S such that H_0,S(E) not rejected by test at significance level α. (3.4)"
"The procedure in (3.4) is called Invariant Causal Prediction (ICP)."

**Theorem 3.1 (Peters, Bühlmann and Meinshausen, 2016).** "Assume a structural equation model for the response Y as in (3.1) and that the environments or perturbations in E satisfy the assumption (B(E)). Furthermore, assume that the tests used in (3.4) are valid, controlling the type I error. Then, for α ∈ (0, 1) we have that P(S_hat(E) ⊆ pa(Y)) ≥ 1 − α."

"The interesting fact is that one does not need to care about identifiability: it is addressed automatically in the sense that if a variable is in S_hat(E), it must be identifiable as causal variable for Y, at least with controllable probability 1 − α."

## Page 410

### 4. ANCHOR REGRESSION: RELAXING CONDITIONS
"The main concern with ICP in (3.4) and the underlying invariance principle is the violation of the assumption in (B(E))... such a violation can happen under various scenarios... Perhaps the most prominent violation is in terms of hidden confounding variables H."

### 4.1 The Anchor Regression Model
"We will allow now that the environments can act directly also on H and Y, relaxing a main assumption in IV regression models... We consider now the IV regression model... (4.1) ... where (all the components of) A, ε are jointly independent. ... The main assumption is that A is a source node and thus, the contribution of A enters as an additional linear term MA. Because of this, we use the terminology “anchor”: it is the anchor which is not influenced by other variables in the system and thus, it remains as the “static pole”."

## Page 411

### 4.2 Causal Regularization and the Anchor Regression Estimator
"Similar to the idea in (4.2), we define the anchor regression estimator by using a regularization term, referred to as causal regularization, which encourages orthogonality or uncorrelatedness of the residuals with the anchor variables A."
"(4.3) β_hat(γ) = argmin ||(I - Pi_A)(Y - Xb)||^2 / n + γ ||Pi_A(Y - Xb)||^2 / n."
"For γ = 1, β_hat(1) equals the ordinary least squares estimator, for γ → ∞ we obtain the two-stage least squares procedure from IV regression and for γ → 0 we adjust for the anchor variables in A."

## Page 412

### 4.3 Shift Perturbations and Robustness of the Anchor Regression Estimator
"The anchor regression estimator solves a worst case risk optimization problem over a class of shift perturbations."
"Theorem 4.1... establishes an exact duality between the causal regularized risk (which is the population version of the objective function for the estimator in (4.3)) and worst-case risk over the class of shift perturbations."
"A useful interpretation of the theorem is as follows. The worst-case risk over shift perturbations can be considered as the one corresponding to future unseen data: this risk for future unseen data can be represented as a regularized risk for the data which we observe in the training sample."
"β(γ) is the minimizer of the worst-case risk: β(γ) = argmin_b sup_{v in C_γ} E[ (Y^v - (X^v)^T b)^2 ]."

## Page 413

### 4.3.2 Diluted form of causality.
"If the assumptions for instrumental variables regression are fulfilled... then the anchor regression estimator with γ → ∞ equals the unique two stage least squares estimator and consistently infers β... If the IV assumptions do not hold... the parameter β(γ) with γ → ∞ or γ being large is often a more meaningful quantity than the standard regression parameter... For large values of γ, the corresponding β(γ) is minimizing a worst-case risk over a class of large shift perturbations."
"In fact, for γ → ∞, we can define supp(β(γ → ∞)) to be the set of variables which are called “diluted causal” for the response Y (the variables which are relevant for Y in a stable way across many strong shift perturbations)."

## Page 414

### 4.5 Distributional Robustness
"Anchor regression and causality can be viewed from the angle of distributional robustness... Distributional robustness refers to optimizing a worst-case risk over a class of distributions."
"For causality and anchor regression, the class of distributions P is given by a causal or “anchor-type” model consisting of perturbation distributions... Thus, with anchor regression, the class of distributions is not pre-defined via a metric d(·, ·) and a radius ρ but rather through the observed heterogeneities in span(M) and a strength of perturbations or “radius” γ."

## Additional Definitions (Week 4 Rigor Check)

### Anchor Regression
*   **Anchor Regression:** "We define the anchor regression estimator by using a regularization term... which encourages orthogonality... of the residuals with the anchor variables A." (Page 411)

### Diluted Causality
*   **Diluted Causality:** "The variables which are relevant for Y in a stable way across many strong shift perturbations... called 'diluted causal'." (Page 413)

### Robustness / Worst-Case Risk
*   **Robustness:** "The anchor regression estimator solves a worst case risk optimization problem over a class of shift perturbations." (Page 412)
*   **Stability:** "The key idea relies on a notion of probabilistic invariance or stability." (Page 404)

