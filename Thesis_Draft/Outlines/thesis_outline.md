# Thesis Outline: Predicting Zoning Opposition

**Author:** Daniel Hardesty Lewis  
**Program:** Master's Thesis, Urban Planning, Columbia University GSAPP  

## Chapter 1: Introduction (3-5 pages)
* **1.1 Background and Context:** The rise of zoning opposition and NIMBYism in growing cities. The importance of understanding protest risks for urban planning.
* **1.2 Austin, Texas as a Case Study:** The unique context of Austin's housing market, rapid growth, and history of zoning protests.
* **1.3 Research Objectives and Questions:** Can machine learning models, specifically logistic regression and conditional diffusion (DDPM), accurately forecast per-parcel zoning protest petition risks?
* **1.4 Thesis Structure:** Brief overview of the chapters to follow.

## Chapter 2: Literature Review (5-7 pages)
* **2.1 The Dynamics of Zoning Opposition:** Reviewing established literature on NIMBYism, homevoter hypotheses, and the drivers of neighborhood resistance.
* **2.2 Quantitative Methods in Urban Planning:** Exploring how predictive modeling has been applied to urban phenomena.
* **2.3 Generative Models and Causal Inference:** The emergence of diffusion models (DDPM) and Invariant Causal Prediction (ICP) in structured tabular forecasting.

## Chapter 3: Data and Methodology (8-10 pages)
* **3.1 Data Sources:** Detailed breakdown of the five primary datasets (Protest Petitions 2007-2025, TCAD EARS 2018-2025, Land Database, Land Use Inventory, ACS 5-Year Estimates). *(Drafted in `data_section.tex`)*
* **3.2 Panel Construction and ID Crosswalk:** Methodology for building the balanced property-year panel (282,772 parcels $\times$ 6 years). *(Drafted in `data_section.tex`)*
* **3.3 Temporal Leakage Audit:** Ensuring predictors do not encode information about future outcomes (SAFE, CAUTION, and EXCLUDE categories). *(Drafted in `data_section.tex`)*
* **3.4 Invariant Causal Prediction Setup:** Treating calendar years as distinct environments ($\mathcal{E}$) to find stable predictors across economic shifts. *(Drafted in `data_section.tex`)*

## Chapter 4: Modeling Approach (5-8 pages)
* **4.1 Baseline Models:** Setting up the Logistic Regression and XGBoost benchmarks.
* **4.2 Conditional Diffusion (DDPM):** Methodology for training the generative diffusion model for tabular protest risk forecasting.
* **4.3 Evaluation Metrics:** Defining success (RMSE, MAE, CRPS, Scenario Std) and the out-of-sample validation strategy (2018-2025 cases).

## Chapter 5: Results and Discussion (10-12 pages)
* **5.1 Baseline Model Performance:** Feature importance and predictive accuracy of the traditional models.
* **5.2 Generative Model Performance:** Evaluating the DDPM's accuracy and its ability to generate realistic counterfactuals.
* **5.3 Comparative Analysis:** Contrasting baseline vs. generative approaches (referencing backtest and generative timelapse visualizations).
* **5.4 Policy Implications:** How planners and policymakers can utilize these predictive tools for proactive engagement and rezoning strategies.

## Chapter 6: Conclusion (2-4 pages)
* **6.1 Summary of Findings:** High-level takeaways from the data pipeline and modeling results.
* **6.2 Limitations:** Addressing data gaps (e.g., ID crosswalk mismatch rates), assumption constraints, and model bounding.
* **6.3 Future Work:** Potential for scaling to other jurisdictions or integrating with NLP-parsed qualitative petition data.

---
**Total Estimated Length:** 33-46 pages (excluding Abstract, References, and Appendices)
