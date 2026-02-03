**Project Proposal Update: Invariant Predictors of NIMBYism**

*   **1. What problem am I solving?** Investigating causal drivers of neighborhood opposition (NIMBYism) that are invariant across market regimes. The goal is to separate stable, **invariant predictors** from **spurious associations** driven by **environment-specific confounders**.
*   **2. Why is this problem important?** Current models overfit to specific eras, failing when market regimes shift (Arjovsky et al., 2019). Identifying invariant causes enables robust policy design.
*   **3. What is my strategy for solving it?** Using **Invariant Risk Minimization (IRM)** on **Training Environments** ($\mathcal{E}_{tr}$) to learn a representation $\Phi(X)$ that yields stable predictions.
*   **4. Why is this solution good? Where does it fall short?** **It is good because** it moves beyond correlation to causal structure (Peters et al., 2016). **It falls short because** it assumes the causal mechanism itself is stable across eras.
*   **5. How can I demonstrate that the solution works?** Testing for OOD Generalization in **Test Environments** ($\mathcal{E}_{te}$) using an expanding window:
| Window | Training ($\mathcal{E}_{tr}$) | Test ($\mathcal{E}_{te}$) |
| :--- | :--- | :--- |
| 1 | 2018 | 2019--2025 |
| 2 | 2018--2019 | 2020--2025 |
| ... | ... | ... |
| N | 2018--2024 | 2025 |

