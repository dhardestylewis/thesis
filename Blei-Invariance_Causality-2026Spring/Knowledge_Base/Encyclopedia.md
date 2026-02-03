# Encyclopedia of Concepts

## Invariant Risk Minimization (IRM)
**Origin:** Arjovsky et al. (2019)
**Concept:** A learning paradigm that estimates a data representation $\Phi(X)$ such that the optimal linear classifier $w$ on top of this representation is optimal across all **training environments** $\mathcal{E}_{train}$.
**Key Equation:** $\min_{\Phi, w} \sum_{e \in \mathcal{E}_{train}} R^e(w \circ \Phi) + \lambda \cdot \| \nabla_{w|w=1.0} R^e(w \cdot \Phi) \|^2$
**Relevance:** We can use IRM to find features of housing developments that predict opposition consistently, regardless of the neighborhood's wealth or the year.

## Causal Representation Learning
**Origin:** Schölkopf et al. (2021)
**Concept:** Learning high-level causal variables $S_1, ..., S_n$ from low-level high-dimensional observations $X$ (e.g., images, text). The goal is to invert the generative process to recover the latent causal structure.
**Relevance:** If our input data involves unstructured text (protest letters) or images, we need to map these to causal concepts (ie. "traffic concern", "density fear") rather than just correlating raw tokens.

## Invariant Prediction
**Origin:** Peters et al. (2016)
**Concept:** Assumes that the causal mechanism $P(Y|PA(Y))$ is the same in all environments. By finding subsets of features $S$ such that $Y \perp E | X_S$, one can reconstruct the set of causal predictors.
**Relevance:** In the NIMBYism context, this means that if "homeowner age" is a causal driver, its effect on opposition should be the same in 2008 and 2024.

## Empirical Bayes
**Origin:** Efron / Robbins
**Concept:** Using the data itself to estimate the prior distribution in a Bayesian framework.
**Relevance:** Modeling the "population" of rezoning cases where each case is a parameter, and we learn the structure of the prior from the aggregate history.
