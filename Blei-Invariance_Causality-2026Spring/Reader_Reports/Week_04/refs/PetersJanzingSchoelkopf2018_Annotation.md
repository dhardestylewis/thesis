# Annotation for Peters, Janzing, and Schoelkopf (2018): Elements of Causal Inference

## Chapter 1: Statistical and Causal Models

### 1.3 Causal Modeling and Learning
"Causal modeling starts from another, arguably more fundamental, structure. A causal structure entails a probability model, but it contains additional information not contained in the latter... Causal reasoning... allows us to analyze the effect of interventions or distribution changes." (Page 5)

"Reichenbach’s common cause principle establishes a link between statistical properties and causal structures. A statistical dependence between two observables X and Y indicates that they are caused by a variable Z... Furthermore, this variable Z screens X and Y from each other in the sense that given Z, they become independent." (Page 7)

## Chapter 2: Assumptions for Causal Inference

### 2.1 The Principle of Independent Mechanisms
"Principle 2.1 (Independent mechanisms) The causal generative process of a system’s variables is composed of autonomous modules that do not inform or influence each other. In the probabilistic case, this means that the conditional distribution of each variable given its causes (i.e., its mechanism) does not inform or influence the other conditional distributions." (Page 19)

"If we change the altitude A, then we assume that the physical mechanism p(t|a) responsible for producing an average temperature ... is still in place... This would hold true independent of the distribution from which we have sampled the cities, and thus independent of p(a)." (Page 17)

"If A -> T is the correct causal structure, then ... p(a) and p(t|a) are autonomous, modular, or invariant mechanisms or objects in the world." (Page 17)

## Chapter 7: Learning Multivariate Causal Models

### 7.1.6 Observational and Experimental Data
"The key assumption is the existence of an unknown set PA_Y ... (one may think of the direct causes of Y) such that the conditional Y given PA_Y is invariant over all environments, that is, for all e, f in E we have P(Y_e | PA_Y_e) = P(Y_f | PA_f)." (Page 142)

"This assumption is satisfied if the distributions are induced by an underlying SCM and the different environments correspond to different intervention distributions, for which Y has not been intervened on." (Page 142)

### 7.2.5 Observational and Experimental Data -> Different Environments
"We have defined the set S as the collection of all sets S ... that satisfy invariant prediction... In practice, we can test the hypothesis of invariant prediction at level alpha and collect all sets S that pass the test as an estimate S_hat for the set S." (Page 153)

"Because the true set of parents PA_Y is a member of S_hat with high probability (1 - alpha), we obtain the coverage statement (intersection of S in S_hat) is a subset of PA_Y with high probability (1 - alpha). The left-hand side ... is the output of a method called 'invariant causal prediction' [Peters et al., 2016]." (Page 154)

"Code Snippet 7.11 ... The method of invariant causal prediction outputs only the causal parents of Y, that is, X1 and X2." (Page 154)

## Note on Week 4 Relevance
This book provides the foundational theory for the "invariance" concept discussed in Bühlmann 2020 and implemented in Peters et al. 2016. It links the philosophical "Principle of Independent Mechanisms" to the practical algorithm of "Invariant Causal Prediction".

## Additional Definitions (Week 4 Rigor Check)

### Mechanism / Autonomous / Invariant
*   **Principle of Independent Mechanisms:** "The causal generative process of a system’s variables is composed of autonomous modules that do not inform or influence each other. In the probabilistic case, this means that the conditional distribution of each variable given its causes (i.e., its mechanism) does not inform or influence the other conditional distributions." (Page 19)
*   **Invariant Mechanisms:** "If A -> T is the correct causal structure, then ... p(a) and p(t|a) are autonomous, modular, or invariant mechanisms." (Page 17)

### Shift / Covariate Shift
*   **Covariate Shift:** "The influence of Y consists only in shifting the mean of X. Under this assumption... covariate shift." (Page 112)

